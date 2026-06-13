"""Monkey patch vLLM V1 scheduler/worker/engine for TriAttention runtime integration.

This keeps vLLM class identities unchanged (native Scheduler/Worker) while
injecting the minimum TriAttention hooks needed for current runtime behavior.
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any, Callable, cast

from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput

from .ascend_defaults import apply_ascend_fast_recency_defaults
from .ascend_graph_mode_patch import make_patched_ascend_forward_context
from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
from .kv_allocation_sync import (
    prepare_request_effective_num_computed,
    resolve_request_effective_num_computed,
)
from .input_patch_backend import install_runtime_input_patch
from .logging_control import runtime_logging_enabled
from .planner import CompressionPlanner
from .request_key_compat import iter_scheduled_token_items
from .scheduler import TriAttentionScheduler
from .signals import CompressionSignal
from .thresholds import is_ascend_environment_available
from .worker import (
    TriAttentionWorker,
    _debug_early_install_proxy_enabled,
    _should_early_install_proxy,
    should_install_triattention_runner_proxy,
)

_PATCHED = False
_PATCHED_SCHEDULER_ACTIVE = False
_PATCHED_WORKER_ACTIVE = False
_ORIG_SCHED_INIT: Callable[..., Any] | None = None
_ORIG_SCHED_SCHEDULE: Callable[..., Any] | None = None
_ORIG_SCHED_UPDATE_FROM_OUTPUT: Callable[..., Any] | None = None
_ORIG_WORKER_INIT_DEVICE: Callable[..., Any] | None = None
_ORIG_WORKER_EXECUTE_MODEL: Callable[..., Any] | None = None
_ORIG_ASCEND_WORKER_METHODS: dict[type, dict[str, Callable[..., Any]]] = {}
_ORIG_ASCEND_SET_FORWARD_CONTEXT: Callable[..., Any] | None = None
_ORIG_ASCEND_MODEL_RUNNER_SET_FORWARD_CONTEXT: Callable[..., Any] | None = None
_ORIG_KVCACHE_ALLOCATE_SLOTS: Callable[..., Any] | None = None
_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE: Callable[..., Any] | None = None
_DEFER_PREFILL_BOUNDARY_CACHE: bool | None = None
_ASYNC_BOUNDARY_ENABLED_CACHE: bool | None = None


def _maybe_rewrite_v2_output_req_map(scheduler_output: Any, model_runner_output: Any) -> None:
    if os.environ.get("TRIATTN_DEBUG_V2_REWRITE_OUTPUT_REQ_MAP", "0") != "1":
        return
    req_map = getattr(model_runner_output, "req_id_to_index", None)
    if not isinstance(req_map, dict):
        return
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict) or not scheduled:
        return
    scheduled_req_ids = sorted(scheduled.keys(), key=lambda k: scheduled[k])
    if len(scheduled_req_ids) > len(req_map):
        return
    output_req_ids = list(req_map.keys())
    if not output_req_ids:
        return
    if any(req_id in req_map for req_id in scheduled_req_ids):
        return
    rewritten = {
        req_id: idx for idx, req_id in enumerate(scheduled_req_ids)
    }
    setattr(model_runner_output, "req_id_to_index", rewritten)
    try:
        setattr(model_runner_output, "req_ids", scheduled_req_ids)
    except Exception:
        pass
    if runtime_logging_enabled():
        logger.info(
            "TriAttention debug rewrote V2 output req map: scheduled_req_ids=%s original_output_req_ids_head=%s",
            scheduled_req_ids,
            output_req_ids[: min(4, len(output_req_ids))],
        )


def _refresh_scheduler_stats_kv_usage(outputs: Any, kv_usage: float) -> None:
    """Best-effort refresh for scheduler_stats.kv_cache_usage in returned outputs.

    In V1, TriAttention reclaim is applied after the base scheduler emits stats.
    Refreshing this field keeps the per-step exported usage aligned with the
    post-reclaim block-pool state without changing core scheduling behavior.
    """
    if not isinstance(outputs, dict):
        return
    usage = float(kv_usage)
    for engine_output in outputs.values():
        scheduler_stats = getattr(engine_output, "scheduler_stats", None)
        if scheduler_stats is not None:
            scheduler_stats.kv_cache_usage = usage


def _patched_scheduler_init(self, *args, **kwargs):
    assert _ORIG_SCHED_INIT is not None
    _ORIG_SCHED_INIT(self, *args, **kwargs)
    cfg = TriAttentionRuntimeConfig.from_env()
    if is_ascend_environment_available():
        apply_ascend_fast_recency_defaults(cfg)
    # Always attach config/state once patched to keep behavior deterministic.
    self.triattention_config = cfg
    self._planner = CompressionPlanner(cfg)
    self._effective_len_tracker = EffectiveCacheLenTracker()
    self._prefill_lens = {}
    self._prefill_compression_counts = {}
    self._length_threshold_cache = {}
    self._last_signal_log_steps = {}
    self._long_context_guard_logged = set()
    self._triattention_step = 0
    if cfg.logging_enabled:
        logger.info(
            "TriAttention monkeypatched Scheduler initialized: budget=%d divide_length=%d "
            "min_reclaim_blocks_on_ascend=%d protect_prefill=%s "
            "disable_compression=%s kv_usage_trigger_enabled=%s",
            cfg.kv_budget,
            cfg.divide_length,
            int(getattr(cfg, "min_reclaim_blocks_on_ascend", 0) or 0),
            cfg.protect_prefill,
            cfg.disable_compression,
            cfg.enable_kv_usage_trigger,
        )


def _compute_max_chunk_for_compression(self, cfg: TriAttentionRuntimeConfig) -> int | None:
    """Compute max tokens per scheduling step to allow compression cycling.

    When physical KV cache is smaller than budget + default chunk size,
    chunked prefill cannot fit the next chunk after compression. Cap the
    per-step token budget so that budget + chunk <= physical KV capacity.
    Returns None if no cap is needed.
    """
    block_pool = getattr(getattr(self, "kv_cache_manager", None), "block_pool", None)
    if block_pool is None:
        return None
    total_blocks = getattr(block_pool, "num_gpu_blocks", 0)
    if total_blocks <= 0:
        return None
    block_size = int(getattr(self, "block_size", 16) or 16)
    physical_kv = total_blocks * block_size
    headroom = physical_kv - cfg.kv_budget
    if headroom <= 0:
        return None
    # Leave a small margin (one block) for allocation bookkeeping.
    headroom = max(1, headroom - block_size)
    return headroom


def _patched_scheduler_schedule(self):
    assert _ORIG_SCHED_SCHEDULE is not None
    TriAttentionScheduler._sync_effective_kv_offsets_before_schedule(self)

    cfg = getattr(self, "triattention_config", None)
    orig_max_scheduled = None
    if cfg and not cfg.disable_compression:
        max_chunk = _compute_max_chunk_for_compression(self, cfg)
        if max_chunk is not None:
            current_max = getattr(self, "max_num_scheduled_tokens", None)
            if current_max is not None and max_chunk < current_max:
                orig_max_scheduled = current_max
                self.max_num_scheduled_tokens = max_chunk

    scheduler_output = _ORIG_SCHED_SCHEDULE(self)

    if orig_max_scheduled is not None:
        self.max_num_scheduled_tokens = orig_max_scheduled

    if cfg is None:
        return scheduler_output

    self._triattention_step += 1
    TriAttentionScheduler._sync_prefill_lens(self, scheduler_output)

    if (
        cfg.disable_compression
        and not cfg.enable_kv_usage_trigger
        and not TriAttentionScheduler._has_active_effective_len_overrides(self)
    ):
        triattention_signals = {}
    else:
        triattention_signals = TriAttentionScheduler._build_signals(self, scheduler_output)

    setattr(scheduler_output, "triattention_step", self._triattention_step)
    setattr(scheduler_output, "triattention_signals", triattention_signals)
    return scheduler_output


def _patched_scheduler_update_from_output(self, scheduler_output, model_runner_output):
    assert _ORIG_SCHED_UPDATE_FROM_OUTPUT is not None
    _maybe_rewrite_v2_output_req_map(scheduler_output, model_runner_output)
    try:
        outputs = _ORIG_SCHED_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)
    except KeyError:
        raise

    cfg = getattr(self, "triattention_config", None)
    if cfg is None:
        return outputs

    # Prefer events from model_runner_output (V0 / sync path), fall back to
    # scheduler_output (V1 async path where execute_model returns None).
    compression_events = getattr(
        model_runner_output,
        "triattention_compression_events",
        None,
    )
    source = "model_runner_output" if compression_events else None
    if not compression_events:
        compression_events = getattr(
            scheduler_output,
            "triattention_compression_events",
            None,
        )
        if compression_events:
            source = "scheduler_output"
    if compression_events:
        applied = [e for e in compression_events if e.get("status") == "applied"]
        if cfg.logging_enabled:
            log_fn = logger.info if applied and cfg.log_decisions else logger.debug
            log_fn(
                "TriAttention update_from_output: received %d events (%d applied) via %s",
                len(compression_events), len(applied), source,
            )
        TriAttentionScheduler._apply_compression_events(self, compression_events)
        _refresh_scheduler_stats_kv_usage(outputs, self.kv_cache_manager.usage)

    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._last_signal_log_steps.pop(req_id, None)
        self._long_context_guard_logged.discard(req_id)
        self._effective_len_tracker.remove_request(req_id)
    return outputs


def _install_triattention_runner_proxy_state(self) -> None:
    if not _PATCHED_WORKER_ACTIVE:
        return
    if getattr(self, "_triattention_runner_proxy_installed", False):
        return
    # Reuse TriAttentionWorker lazy-injection fields on native Worker instance.
    self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
    self._triattention_runner_proxy_installed = False
    if _debug_early_install_proxy_enabled() or _should_early_install_proxy(
        self,
        self._triattention_runtime_config,
    ):
        self._triattention_installing_during_init = True
        try:
            TriAttentionWorker._ensure_triattention_runner_proxy(self)
        finally:
            self._triattention_installing_during_init = False
        if bool(getattr(self._triattention_runtime_config, "log_decisions", False)):
            logger.debug("TriAttention: eagerly installed runner proxy during worker init_device")


def _patched_worker_init_device(self):
    assert _ORIG_WORKER_INIT_DEVICE is not None
    _ORIG_WORKER_INIT_DEVICE(self)
    _install_triattention_runner_proxy_state(self)


def _patched_worker_execute_model(self, scheduler_output):
    assert _ORIG_WORKER_EXECUTE_MODEL is not None
    if _PATCHED_WORKER_ACTIVE and should_install_triattention_runner_proxy(
        self,
        scheduler_output,
    ):
        TriAttentionWorker._ensure_triattention_runner_proxy(self)
    return _ORIG_WORKER_EXECUTE_MODEL(self, scheduler_output)


def _resolve_vllm_gpu_worker_class() -> type | None:
    try:
        import vllm.v1.worker.gpu_worker as worker_mod

        return worker_mod.Worker
    except Exception:
        if runtime_logging_enabled():
            logger.debug("TriAttention could not import vLLM GPU Worker", exc_info=True)
        return None


def _patch_vllm_gpu_worker_class_for_triattention(worker_cls: type | None) -> bool:
    global _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    if worker_cls is None:
        return False
    if _ORIG_WORKER_INIT_DEVICE is not None:
        return False
    init_device = getattr(worker_cls, "init_device", None)
    execute_model = getattr(worker_cls, "execute_model", None)
    if not callable(init_device) or not callable(execute_model):
        if runtime_logging_enabled():
            logger.warning(
                "Could not install TriAttention runtime worker patches for vLLM: "
                "%s.%s is missing init_device or execute_model",
                getattr(worker_cls, "__module__", "<unknown>"),
                getattr(worker_cls, "__name__", "<unknown>"),
            )
        return False
    _ORIG_WORKER_INIT_DEVICE = init_device
    _ORIG_WORKER_EXECUTE_MODEL = execute_model
    worker_cls.init_device = _patched_worker_init_device
    worker_cls.execute_model = _patched_worker_execute_model
    worker_cls._ensure_triattention_runner_proxy = (
        TriAttentionWorker._ensure_triattention_runner_proxy
    )
    if runtime_logging_enabled():
        logger.info(
            "Installed TriAttention runtime worker patches for vLLM: %s.%s",
            getattr(worker_cls, "__module__", "<unknown>"),
            getattr(worker_cls, "__name__", "<unknown>"),
        )
    return True


def _resolve_original_ascend_worker_method(
    worker: Any,
    method_name: str,
) -> Callable[..., Any]:
    for cls in type(worker).__mro__:
        methods = _ORIG_ASCEND_WORKER_METHODS.get(cls)
        if methods is not None and method_name in methods:
            return methods[method_name]
    raise RuntimeError(f"missing_original_ascend_worker_method:{method_name}")


def _patched_ascend_worker_init_device(self):
    _resolve_original_ascend_worker_method(self, "init_device")(self)
    _install_triattention_runner_proxy_state(self)


def _patched_ascend_worker_execute_model(self, scheduler_output):
    if _PATCHED_WORKER_ACTIVE and should_install_triattention_runner_proxy(
        self,
        scheduler_output,
    ):
        TriAttentionWorker._ensure_triattention_runner_proxy(self)
    return _resolve_original_ascend_worker_method(self, "execute_model")(
        self,
        scheduler_output,
    )


def _patch_worker_class_for_triattention(
    worker_cls: type,
    *,
    patch_init: bool,
    patch_execute: bool,
) -> bool:
    if worker_cls in _ORIG_ASCEND_WORKER_METHODS:
        return False
    methods: dict[str, Callable[..., Any]] = {}
    if patch_init:
        init_device = getattr(worker_cls, "init_device", None)
        if callable(init_device):
            methods["init_device"] = init_device
            worker_cls.init_device = _patched_ascend_worker_init_device
    if patch_execute:
        execute_model = getattr(worker_cls, "execute_model", None)
        if callable(execute_model):
            methods["execute_model"] = execute_model
            worker_cls.execute_model = _patched_ascend_worker_execute_model
    if not methods:
        return False
    worker_cls._ensure_triattention_runner_proxy = (
        TriAttentionWorker._ensure_triattention_runner_proxy
    )
    _ORIG_ASCEND_WORKER_METHODS[worker_cls] = methods
    return True


def _install_optional_ascend_forward_context_patch() -> None:
    global _ORIG_ASCEND_SET_FORWARD_CONTEXT, _ORIG_ASCEND_MODEL_RUNNER_SET_FORWARD_CONTEXT
    patched: list[str] = []
    try:
        import vllm_ascend.ascend_forward_context as ascend_context_mod

        original = getattr(ascend_context_mod, "set_ascend_forward_context", None)
        if callable(original) and not getattr(original, "_triattention_patched", False):
            _ORIG_ASCEND_SET_FORWARD_CONTEXT = original
            ascend_context_mod.set_ascend_forward_context = (
                make_patched_ascend_forward_context(original)
            )
            patched.append("vllm_ascend.ascend_forward_context.set_ascend_forward_context")
    except Exception:
        if runtime_logging_enabled():
            log_fn = logger.warning if is_ascend_environment_available() else logger.debug
            log_fn(
                "Could not install TriAttention Ascend forward-context patch",
                exc_info=True,
            )

    try:
        import vllm_ascend.worker.model_runner_v1 as ascend_model_runner_mod

        original = getattr(ascend_model_runner_mod, "set_ascend_forward_context", None)
        if callable(original) and not getattr(original, "_triattention_patched", False):
            _ORIG_ASCEND_MODEL_RUNNER_SET_FORWARD_CONTEXT = original
            ascend_model_runner_mod.set_ascend_forward_context = (
                make_patched_ascend_forward_context(original)
            )
            patched.append("vllm_ascend.worker.model_runner_v1.set_ascend_forward_context")
    except Exception:
        if runtime_logging_enabled():
            log_fn = logger.warning if is_ascend_environment_available() else logger.debug
            log_fn(
                "Could not install TriAttention Ascend model-runner context patch",
                exc_info=True,
            )

    if patched and runtime_logging_enabled():
        logger.info(
            "Installed TriAttention Ascend graph-mode guard patches: %s",
            ", ".join(patched),
        )


def _install_optional_ascend_worker_patches() -> None:
    patched: list[str] = []
    try:
        import vllm_ascend.worker.worker as ascend_worker_mod

        worker_cls = ascend_worker_mod.NPUWorker
        was_patched = worker_cls in _ORIG_ASCEND_WORKER_METHODS
        if _patch_worker_class_for_triattention(
            worker_cls,
            patch_init=True,
            patch_execute=True,
        ):
            patched.append("vllm_ascend.worker.worker.NPUWorker")
        elif not was_patched and runtime_logging_enabled():
            logger.warning(
                "Could not install TriAttention runtime worker patches for Ascend: "
                "vllm_ascend.worker.worker.NPUWorker is missing init_device or execute_model"
            )
    except Exception:
        if runtime_logging_enabled():
            log_fn = (
                logger.warning
                if is_ascend_environment_available()
                else logger.debug
            )
            log_fn(
                "Could not import vllm_ascend.worker.worker.NPUWorker for "
                "TriAttention runtime worker patches",
                exc_info=True,
            )
        return

    optional_workers = (
        ("vllm_ascend._310p.worker_310p", "NPUWorker310"),
        ("vllm_ascend.xlite.xlite_worker", "XliteWorker"),
    )
    for module_name, class_name in optional_workers:
        try:
            module = __import__(module_name, fromlist=[class_name])
            worker_cls = getattr(module, class_name)
            if _patch_worker_class_for_triattention(
                worker_cls,
                patch_init=True,
                patch_execute=False,
            ):
                patched.append(f"{module_name}.{class_name}")
        except Exception:
            continue

    if patched and runtime_logging_enabled():
        logger.info(
            "Installed TriAttention runtime worker patches for Ascend: %s",
            ", ".join(patched),
        )


def _install_worker_patches(worker_cls: type | None) -> None:
    _patch_vllm_gpu_worker_class_for_triattention(worker_cls)
    _install_optional_ascend_worker_patches()
    _install_optional_ascend_forward_context_patch()
    cfg = TriAttentionRuntimeConfig.from_env()
    if bool(getattr(cfg, "preinstall_input_patch", True)):
        install_runtime_input_patch()


def _patched_kv_cache_allocate_slots(
    self,
    request,
    num_new_tokens,
    *args,
    **kwargs,
):
    """Keep vLLM allocation math aligned with TriAttention effective KV length.

    Once a request has been physically compacted, its live KV layout no longer
    matches vLLM's original contiguous-prefix block-hash chain. Continuing to
    commit prefix-cache hashes for later full blocks is therefore invalid and
    can trip BlockPool invariants on the next cache update. We keep slot
    allocation but skip vLLM's cache-commit step for compressed requests.
    """
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    # Ensure effective marker is refreshed — _sync_effective_kv_offsets only
    # covers RUNNING requests, but preempted WAITING requests also need it.
    prepare_request_effective_num_computed(request)
    effective_num_computed = resolve_request_effective_num_computed(request)
    # === BUG-RECL 断点 11（LE-A-001）：allocate_slots effective 路径决策 ===
    if os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1":
        _logical = getattr(request, "num_computed_tokens", None)
        _offset = getattr(request, "_triattention_effective_kv_offset", None)
        _will_use = effective_num_computed is not None and isinstance(_logical, int) and effective_num_computed < _logical
        print(f"BUG-RECL [LE-A-001] allocate_slots req={getattr(request, 'request_id', '?')} effective_num_computed={effective_num_computed} logical_num_computed={_logical} offset={_offset} will_use_effective={_will_use}", flush=True)
    # === BUG-RECL 断点 11 结束 ===
    if effective_num_computed is None:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    logical_num_computed = getattr(request, "num_computed_tokens", None)
    if not isinstance(logical_num_computed, int):
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    if effective_num_computed >= logical_num_computed:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    kwargs = dict(kwargs)
    kwargs["delay_cache_blocks"] = True
    setattr(request, "num_computed_tokens", int(effective_num_computed))
    try:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    finally:
        setattr(request, "num_computed_tokens", logical_num_computed)


def _scheduler_output_has_compression_boundary(scheduler_output: Any) -> bool:
    if not _async_compression_boundary_enabled():
        return False
    signals = getattr(scheduler_output, "triattention_signals", None)
    if not isinstance(signals, dict) or not signals:
        return False
    for sig in signals.values():
        if not bool(getattr(sig, "should_compress", False)):
            continue
        scheduled_tokens = max(1, int(getattr(sig, "scheduled_tokens", 1) or 1))
        if scheduled_tokens > 1 and _should_defer_prefill_boundary():
            continue
        return True
    return False


def _async_compression_boundary_enabled() -> bool:
    global _ASYNC_BOUNDARY_ENABLED_CACHE
    if _ASYNC_BOUNDARY_ENABLED_CACHE is not None:
        return _ASYNC_BOUNDARY_ENABLED_CACHE
    cfg = TriAttentionRuntimeConfig.from_env()
    _ASYNC_BOUNDARY_ENABLED_CACHE = bool(
        getattr(cfg, "enable_async_compression_boundary", False)
    )
    return _ASYNC_BOUNDARY_ENABLED_CACHE


def _should_defer_prefill_boundary() -> bool:
    global _DEFER_PREFILL_BOUNDARY_CACHE
    if _DEFER_PREFILL_BOUNDARY_CACHE is not None:
        return _DEFER_PREFILL_BOUNDARY_CACHE
    cfg = TriAttentionRuntimeConfig.from_env()
    _DEFER_PREFILL_BOUNDARY_CACHE = bool(
        getattr(cfg, "defer_prefill_compression", False)
    ) or (
        bool(getattr(cfg, "defer_prefill_compression_on_ascend", False))
        and is_ascend_environment_available()
    )
    return _DEFER_PREFILL_BOUNDARY_CACHE


def _batch_queue_has_pending_compression_boundary(batch_queue: Any) -> bool:
    if batch_queue is None:
        return False
    try:
        items = list(batch_queue)
    except Exception:
        return False
    for item in items:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        scheduler_output = item[1]
        if bool(getattr(scheduler_output, "_triattention_force_boundary_sync", False)):
            return True
    return False


def _patched_engine_core_step_with_batch_queue(self):
    """vLLM async step with a per-compression boundary queue barrier.

    Normal async decode remains unchanged. The only special case is a batch that
    is predicted to hit a TriAttention compression boundary in this step:

    1. do not let the batch queue run ahead of that batch, and
    2. drain the queued boundary batch before scheduling newer work.

    This keeps the async speedup for ordinary decode while avoiding the exact
    stale-state window that appears when a compression batch is still waiting in
    the queue.
    """
    batch_queue = self.batch_queue
    assert batch_queue is not None

    model_executed = False
    deferred_scheduler_output = None

    boundary_pending = _batch_queue_has_pending_compression_boundary(batch_queue)
    if self.scheduler.has_requests() and not boundary_pending:
        scheduler_output = self.scheduler.schedule()
        boundary_current = _scheduler_output_has_compression_boundary(scheduler_output)
        if boundary_current:
            setattr(scheduler_output, "_triattention_force_boundary_sync", True)
            if runtime_logging_enabled():
                logger.debug(
                    "TriAttention async boundary: delaying queue lookahead for compression batch"
                )
        exec_future = self.model_executor.execute_model(
            scheduler_output, non_block=True
        )
        # vLLM version compatibility: `is_ec_producer` exists in some versions and is missing in others. Default to False.
        if not getattr(self, "is_ec_producer", False):
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if getattr(self, "is_pooling_model", False) or not model_executed:
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            if not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
            else:
                deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if (
                model_executed
                and len(batch_queue) < self.batch_queue_size
                and not batch_queue[-1][0].done()
                and not boundary_current
            ):
                return None, True

    elif not batch_queue:
        return None, False

    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    if deferred_scheduler_output:
        if getattr(self, "use_spec_decode", False):
            draft_token_ids = self.model_executor.take_draft_token_ids()
            assert draft_token_ids is not None
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_scheduler_output
        )
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed


def install_vllm_integration_monkeypatches(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
) -> None:
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    global _ORIG_KVCACHE_ALLOCATE_SLOTS, _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE
    global _PATCHED_SCHEDULER_ACTIVE, _PATCHED_WORKER_ACTIVE
    if _PATCHED:
        if patch_worker and not _PATCHED_WORKER_ACTIVE:
            _install_worker_patches(_resolve_vllm_gpu_worker_class())
        _PATCHED_SCHEDULER_ACTIVE = _PATCHED_SCHEDULER_ACTIVE or bool(patch_scheduler)
        _PATCHED_WORKER_ACTIVE = _PATCHED_WORKER_ACTIVE or bool(patch_worker)
        return

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
    import vllm.v1.engine.core as engine_core_mod

    EngineCore = engine_core_mod.EngineCore
    Scheduler = sched_mod.Scheduler
    KVCacheManager = kv_cache_manager_mod.KVCacheManager
    Worker = _resolve_vllm_gpu_worker_class()

    if patch_scheduler:
        _ORIG_SCHED_INIT = Scheduler.__init__
        _ORIG_SCHED_SCHEDULE = Scheduler.schedule
        _ORIG_SCHED_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
        Scheduler.__init__ = _patched_scheduler_init
        Scheduler.schedule = _patched_scheduler_schedule
        Scheduler.update_from_output = _patched_scheduler_update_from_output
        # Attach helper methods used by the patched wrappers.
        Scheduler._resolve_prefill_len = TriAttentionScheduler._resolve_prefill_len
        Scheduler._compute_length_threshold = TriAttentionScheduler._compute_length_threshold
        Scheduler._ensure_runtime_fields = TriAttentionScheduler._ensure_runtime_fields
        Scheduler._sync_prefill_lens = TriAttentionScheduler._sync_prefill_lens
        Scheduler._signal_log_interval_steps = (
            TriAttentionScheduler._signal_log_interval_steps
        )
        Scheduler._should_log_signal_trigger = (
            TriAttentionScheduler._should_log_signal_trigger
        )
        Scheduler._log_long_context_guard_skip = (
            TriAttentionScheduler._log_long_context_guard_skip
        )
        Scheduler._has_active_effective_len_overrides = (
            TriAttentionScheduler._has_active_effective_len_overrides
        )
        Scheduler._build_signals = TriAttentionScheduler._build_signals
        Scheduler._sync_effective_kv_offsets_before_schedule = (
            TriAttentionScheduler._sync_effective_kv_offsets_before_schedule
        )
        Scheduler._apply_compression_events = TriAttentionScheduler._apply_compression_events
        _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
        KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots
        _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE = EngineCore.step_with_batch_queue
        EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue

    if patch_worker:
        _install_worker_patches(Worker)

    # Relax the KV cache memory check: TriAttention compresses KV cache
    # during generation, so the physical blocks needed are less than what
    # max_model_len implies.  Turn the hard ValueError into a warning.
    try:
        import vllm.v1.core.kv_cache_utils as _kv_utils

        _legacy_check = getattr(_kv_utils, "_check_enough_kv_cache_memory", None)
        _public_check = getattr(_kv_utils, "check_enough_kv_cache_memory", None)

        if _legacy_check is not None:

            def _relaxed_legacy_check(available_memory, get_needed_memory,
                                      max_model_len,
                                      estimate_max_model_len):
                if available_memory <= 0:
                    _legacy_check(available_memory, get_needed_memory,
                                  max_model_len, estimate_max_model_len)
                    return
                needed = get_needed_memory()
                if needed > available_memory:
                    est = estimate_max_model_len(available_memory)
                    logger.warning(
                        "[TriAttention] KV cache check relaxed: max_model_len=%d "
                        "needs %.2f GiB but only %.2f GiB available (est max %d). "
                        "Compression will keep actual usage within limits.",
                        max_model_len, needed / (1 << 30),
                        available_memory / (1 << 30), est,
                    )

            _kv_utils._check_enough_kv_cache_memory = _relaxed_legacy_check
            if runtime_logging_enabled():
                logger.info(
                    "Relaxed legacy KV cache memory check for TriAttention compression"
                )

        if _public_check is not None:

            def _relaxed_public_check(vllm_config, kv_cache_spec,
                                      available_memory):
                if available_memory <= 0:
                    _public_check(vllm_config, kv_cache_spec, available_memory)
                    return

                needed = _kv_utils.max_memory_usage_bytes(
                    vllm_config, kv_cache_spec.values())
                if needed <= available_memory:
                    return

                est = _kv_utils.estimate_max_model_len(vllm_config, kv_cache_spec,
                                                       available_memory)
                logger.warning(
                    "[TriAttention] KV cache check relaxed: max_model_len=%d "
                    "needs %.2f GiB but only %.2f GiB available (est max %d). "
                    "Compression will keep actual usage within limits.",
                    vllm_config.model_config.max_model_len,
                    needed / (1 << 30),
                    available_memory / (1 << 30),
                    est,
                )

            _kv_utils.check_enough_kv_cache_memory = _relaxed_public_check
            if runtime_logging_enabled():
                logger.info(
                    "Relaxed public KV cache memory check for TriAttention compression"
                )

        if _legacy_check is None and _public_check is None:
            logger.warning("Could not find a KV cache memory check symbol to relax")
    except Exception:
        logger.warning("Could not relax KV cache memory check", exc_info=True)

    _PATCHED_SCHEDULER_ACTIVE = bool(patch_scheduler)
    _PATCHED_WORKER_ACTIVE = bool(patch_worker)
    _PATCHED = True
    if runtime_logging_enabled():
        logger.info(
            "Installed TriAttention runtime monkeypatch integration: patch_scheduler=%s patch_worker=%s",
            patch_scheduler,
            patch_worker,
        )
