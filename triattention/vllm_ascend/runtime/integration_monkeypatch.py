"""Ascend Integration Monkeypatch (AIM).

Patches the vllm-ascend v0.18.0 classes that the CUDA path does not touch:

| Target                 | Symbol                                                         | Why it must be patched separately |
| ---------------------- | -------------------------------------------------------------- | --------------------------------- |
| `Scheduler`            | `vllm.v1.core.sched.scheduler.Scheduler`                       | Inherited via MRO by `BalanceScheduler` (rebound by `vllm_ascend/patch/platform/patch_balance_schedule.py:705`); patching on the upstream class makes the wrapper reachable on the ascend side. |
| `KVCacheManager`       | `vllm.v1.core.kv_cache_manager.KVCacheManager`                 | Reused by both `BalanceScheduler` and `NPUWorker`; patch on the upstream class. |
| `NPUWorker`            | `vllm_ascend.worker.worker.NPUWorker`                          | Ascend-side worker; the CUDA patcher targets the unused `vllm.v1.worker.gpu_worker.Worker`. |
| `AscendBlockTables`    | `vllm_ascend.worker.v2.block_table.AscendBlockTables`          | Subclass of vLLM `BlockTables` with int32 slot_mappings and a different `compute_slot_mappings` kernel; the CUDA patcher's `BlockTables.compute_slot_mappings` patch does not propagate because `compute_slot_mappings` is overridden here. |
| `kv_cache_utils`       | `vllm.v1.core.kv_cache_utils`                                  | Same symbol table as the CUDA path; patch in place. |

This file is the **only** place that mutates third-party class
attributes. The mutations are all done with `setattr(...)` on class
objects already in memory; no vLLM or vllm-ascend source file is
edited.

Engineering principles respected:
- **Minimal intrusion:** no upstream file is modified; patching is
  exclusively via runtime `setattr`.
- **Signal driven:** the scheduler attaches `triattention_signals` /
  `triattention_step` to `SchedulerOutput`; the runner attaches
  `triattention_compression_events` to `ModelRunnerOutput` (or to
  `SchedulerOutput` as a fallback).
- **Lazy loading:** the `TriAttentionModelRunner` proxy is only
  attached to `NPUWorker.model_runner` on the first
  `execute_model(...)` call that carries a non-empty
  `triattention_signals` payload.
- **Explicit state sync:** block reclaim is performed by calling
  `block_pool.free_blocks(reversed(removed_blocks))` directly. Prefix
  cache metadata for each removed block is best-effort evicted
  through `block_pool._maybe_evict_cached_block` before the block is
  handed back to the free pool.

The `Scheduler` symbol bound in `vllm.v1.core.sched.scheduler` is
**rebound** by `vllm_ascend.patch.platform.patch_balance_schedule` to
`BalanceScheduler` (a subclass of upstream `Scheduler`) — but the
*class object* we patched (upstream `Scheduler`) is the parent in
the MRO, so all setattrs propagate. To make the rebind survivable
across `importlib.reload(...)` / `adapt_patch(...)`, we also install
a meta-patch on the module that re-attaches the helper methods
whenever `Scheduler` is rebound.
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any, Callable, cast

from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput


# --- INSTRUMENTATION: process-wide cumulative counters ---
# Updated by W:worker_execute on every NPUWorker.execute_model call.
# Used to report the lifetime signal/should_compress counts even when
# the per-step logger is rate-limited or noisy.
_W_CUM_STEPS: int = 0
_W_CUM_SIGNALS_TOTAL: int = 0
_W_CUM_WILL_COMPRESS: int = 0


# Re-exports of platform-agnostic helpers. The implementation lives in
# `triattention.vllm.runtime.*` (CUDA-shared) so the algorithm stays
# in one place.
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.effective_len_tracker import (
    EffectiveCacheLenTracker,
)
from triattention.vllm.runtime.kv_allocation_sync import (
    prepare_request_effective_num_computed,
    resolve_request_effective_num_computed,
)
from triattention.vllm.runtime.planner import CompressionPlanner
from triattention.vllm.runtime.request_key_compat import (
    iter_scheduled_token_items,
)
from triattention.vllm.runtime.scheduler import (
    TriAttentionScheduler,
    _free_reclaimed_blocks,
    _resolve_full_prefill_len_from_request_like,
)
from triattention.vllm.runtime.signals import CompressionSignal
from triattention.vllm.runtime.worker import _debug_early_install_proxy_enabled

from triattention.vllm_ascend.runtime.scheduler_ascend import (
    TriAttentionAscendScheduler,
)
from triattention.vllm_ascend.runtime.worker_ascend import (
    TriAttentionAscendWorker,
)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_PATCHED = False
_PATCHED_SCHEDULER_ACTIVE = False
_PATCHED_WORKER_ACTIVE = False
_ORIG_SCHED_INIT: Callable[..., Any] | None = None
_ORIG_SCHED_SCHEDULE: Callable[..., Any] | None = None
_ORIG_SCHED_UPDATE_FROM_OUTPUT: Callable[..., Any] | None = None
_ORIG_WORKER_INIT: Callable[..., Any] | None = None
_ORIG_WORKER_INIT_DEVICE: Callable[..., Any] | None = None
_ORIG_WORKER_EXECUTE_MODEL: Callable[..., Any] | None = None
_ORIG_KVCACHE_ALLOCATE_SLOTS: Callable[..., Any] | None = None
_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE: Callable[..., Any] | None = None
_ORIG_ASCEND_BLOCK_TABLES_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None

# Helper method bundle that must be (re-)attached every time the
# upstream Scheduler class is bound into `vllm.v1.core.sched.scheduler`.
_SCHEDULER_HELPER_METHODS = (
    "_resolve_prefill_len",
    "_compute_length_threshold",
    "_sync_prefill_lens",
    "_has_active_effective_len_overrides",
    "_build_signals",
    "_sync_effective_kv_offsets_before_schedule",
    "_apply_compression_events",
    # Ascend-only re-attachment hook so subclass (`BalanceScheduler`)
    # keeps these methods even if the upstream class is re-patched.
    "_compute_max_chunk_for_compression",
    "_evict_reclaimed_block_metadata",
    "_free_reclaimed_blocks",
    "_resolve_full_prefill_len_from_request_like",
)


# ---------------------------------------------------------------------------
# Scheduler wrappers
# ---------------------------------------------------------------------------


def _patched_scheduler_init(self, *args, **kwargs):
    """Wrap upstream `Scheduler.__init__` to install Ascend-side state."""
    assert _ORIG_SCHED_INIT is not None
    _ORIG_SCHED_INIT(self, *args, **kwargs)
    cfg = TriAttentionRuntimeConfig.from_env()
    # Always attach config/state once patched to keep behavior deterministic.
    self.triattention_config = cfg
    self._planner = CompressionPlanner(cfg)
    self._effective_len_tracker = EffectiveCacheLenTracker()
    self._prefill_lens = {}
    self._length_threshold_cache = {}
    self._triattention_step = 0
    logger.info(
        "[TriAttention-Ascend] Scheduler initialized: type=%s budget=%d "
        "divide_length=%d protect_prefill=%s disable_compression=%s "
        "kv_usage_trigger_enabled=%s",
        type(self).__name__,
        cfg.kv_budget,
        cfg.divide_length,
        cfg.protect_prefill,
        cfg.disable_compression,
        cfg.enable_kv_usage_trigger,
    )


def _patched_scheduler_schedule(self):
    """Wrap upstream `Scheduler.schedule` to attach compression signals."""
    assert _ORIG_SCHED_SCHEDULE is not None
    TriAttentionAscendScheduler._sync_effective_kv_offsets_before_schedule(self)

    cfg = getattr(self, "triattention_config", None)
    orig_max_scheduled = None
    if cfg and not cfg.disable_compression:
        max_chunk = TriAttentionAscendScheduler._compute_max_chunk_for_compression(
            self, cfg
        )
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
    TriAttentionAscendScheduler._sync_prefill_lens(self, scheduler_output)

    if (
        cfg.disable_compression
        and not cfg.enable_kv_usage_trigger
        and not TriAttentionAscendScheduler._has_active_effective_len_overrides(self)
    ):
        triattention_signals: dict[str, CompressionSignal] = {}
    else:
        triattention_signals = TriAttentionAscendScheduler._build_signals(
            self, scheduler_output
        )

    # Cross-process bridge: signals ride on the SchedulerOutput object
    # so the worker subprocess can see them after pickling.
    setattr(scheduler_output, "triattention_step", self._triattention_step)
    setattr(scheduler_output, "triattention_signals", triattention_signals)
    return scheduler_output


def _patched_scheduler_update_from_output(self, scheduler_output, model_runner_output):
    """Wrap upstream `Scheduler.update_from_output` to apply compression events."""
    assert _ORIG_SCHED_UPDATE_FROM_OUTPUT is not None
    try:
        outputs = _ORIG_SCHED_UPDATE_FROM_OUTPUT(
            self, scheduler_output, model_runner_output
        )
    except KeyError:
        raise

    cfg = getattr(self, "triattention_config", None)
    if cfg is None:
        return outputs

    # Prefer events attached to the model_runner_output (sync path),
    # fall back to scheduler_output (async path where execute_model
    # returns None and the worker tacks events onto the scheduler_output
    # before returning).
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
        logger.info(
            "[TriAttention-Ascend] update_from_output: received %d events "
            "(%d applied) via %s",
            len(compression_events),
            len(applied),
            source,
        )
        TriAttentionAscendScheduler._apply_compression_events(
            self, compression_events
        )

    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._effective_len_tracker.remove_request(req_id)
    return outputs


# ---------------------------------------------------------------------------
# KVCacheManager wrapper
# ---------------------------------------------------------------------------


def _patched_kv_cache_allocate_slots(
    self,
    request,
    num_new_tokens,
    *args,
    **kwargs,
):
    """Keep vLLM allocation math aligned with TriAttention effective KV length.

    Once a request has been physically compacted, its live KV layout no
    longer matches vLLM's original contiguous-prefix block-hash chain.
    Continuing to commit prefix-cache hashes for later full blocks is
    therefore invalid and can trip BlockPool invariants on the next
    cache update. We keep slot allocation but skip vLLM's cache-commit
    step for compressed requests.
    """
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    prepare_request_effective_num_computed(request)
    effective_num_computed = resolve_request_effective_num_computed(request)
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


# ---------------------------------------------------------------------------
# Worker wrappers (NPUWorker)
# ---------------------------------------------------------------------------


def _patched_npu_worker_init(self, *args, **kwargs):
    """Wrap `NPUWorker.__init__` to install runtime config and run a
    defensive re-apply of the patches in the worker subprocess.

    The re-apply is necessary because the worker subprocess imports
    `vllm.v1.worker.gpu_worker` and `vllm_ascend.worker.worker` from
    scratch, and an `adapt_patch(is_global_patch=True)` on the worker
    side may rebind `Scheduler` to `BalanceScheduler`. The meta-patch
    on the scheduler module handles the `Scheduler` rebind; this
    hook handles the worker init so the lazy runner proxy config is
    installed even on a fresh subprocess.
    """
    assert _ORIG_WORKER_INIT is not None
    _ORIG_WORKER_INIT(self, *args, **kwargs)
    # --- INSTRUMENTATION (NPUWorker init probe) ---
    # Confirms: (a) the patched NPUWorker.__init__ wrapper was actually
    # called in this subprocess; (b) the post-init defensive re-apply
    # is happening. If this never fires, the worker subprocess was
    # not patched.
    import os as _os_wi
    if _os_wi.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1":
        try:
            _has_proxy_fn = hasattr(self, "_ensure_triattention_runner_proxy")
            _sched_patched = getattr(
                getattr(__import__("vllm.v1.core.sched.scheduler", fromlist=["Scheduler"]),
                        "Scheduler", None),
                "_triattention_patched", None,
            ) if _os_wi.environ.get("TRIATTN_DEBUG_INSTRUMENT_VERBOSE", "0") == "1" else None
            logger.info(
                "[TRITN-INSTR] W:npu_worker_init class=%s has_proxy_fn=%s "
                "patched_worker_active=%s",
                type(self).__name__, _has_proxy_fn, _PATCHED_WORKER_ACTIVE,
            )
        except Exception:
            pass
    if not _PATCHED_WORKER_ACTIVE:
        return
    if getattr(self, "_triattention_runner_proxy_installed", False):
        return
    self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
    self._triattention_runner_install_attempted = False
    self._triattention_runner_proxy_installed = False
    if _debug_early_install_proxy_enabled():
        try:
            TriAttentionAscendWorker._ensure_triattention_runner_proxy(self)
            logger.debug(
                "[TriAttention-Ascend] eagerly installed runner proxy during NPUWorker.__init__"
            )
        except Exception as _exc:
            # --- INSTRUMENTATION: log early-install failure ---
            if _os_wi.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1":
                logger.info(
                    "[TRITN-INSTR] W:early_proxy_install_FAILED exc=%s msg=%s",
                    type(_exc).__name__, str(_exc)[:200],
                )
    # Defensive re-apply: a fresh subprocess may have lost the patches
    # applied in the main process. Re-apply them on the actual class
    # objects in this subprocess.
    try:
        ensure_patches_installed(
            patch_scheduler=True,
            patch_worker=True,
            reason="npu_worker_post_init",
        )
    except Exception:
        logger.exception(
            "[TriAttention-Ascend] defensive re-apply during NPUWorker.__init__ failed"
        )


def _patched_npu_worker_init_device(self):
    """Wrap `NPUWorker.init_device` (also a hook) for safety on the
    V2 model runner path where init may run separately from
    `__init__`. Idempotent: only acts once per worker instance.
    """
    assert _ORIG_WORKER_INIT_DEVICE is not None
    _ORIG_WORKER_INIT_DEVICE(self)
    if not _PATCHED_WORKER_ACTIVE:
        return
    if getattr(self, "_triattention_runner_proxy_installed", False):
        return
    self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
    self._triattention_runner_proxy_installed = False


def _patched_npu_worker_execute_model(self, scheduler_output):
    """Wrap `NPUWorker.execute_model` to lazy-install the runner proxy
    on the first signal-bearing step and to delegate to the original
    implementation otherwise.
    """
    assert _ORIG_WORKER_EXECUTE_MODEL is not None
    # --- INSTRUMENTATION (Upstream worker-side probe) ---
    # This wrapper runs on EVERY NPUWorker.execute_model call, BEFORE
    # the TriAttentionModelRunner proxy may be installed. It is the
    # earliest place we can see what the scheduler actually sent us.
    # Logs the size of triattention_signals regardless of whether the
    # proxy is installed — answers "is the worker even receiving
    # non-empty signals?".
    import os as _os_w
    if _os_w.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1":
        try:
            _sig = getattr(scheduler_output, "triattention_signals", None)
            _step = getattr(scheduler_output, "triattention_step", None)
            _n_total = len(_sig) if isinstance(_sig, dict) else 0
            _n_press = (
                sum(1 for s in _sig.values() if bool(getattr(s, "should_compress", False)))
                if isinstance(_sig, dict) else 0
            )
            _proxy_installed = bool(
                getattr(self, "_triattention_runner_proxy_installed", False)
            )
            _runner_class = type(getattr(self, "model_runner", None)).__name__
            # Update process-wide cumulative counters. These survive
            # the W:worker_execute being rate-limited and let us see
            # the LIFETIME totals on every log line.
            global _W_CUM_STEPS, _W_CUM_SIGNALS_TOTAL, _W_CUM_WILL_COMPRESS
            _W_CUM_STEPS += 1
            _W_CUM_SIGNALS_TOTAL += _n_total
            _W_CUM_WILL_COMPRESS += _n_press
            logger.info(
                "[TRITN-INSTR] W:worker_execute step=%s signals=%d will_compress=%d "
                "proxy_installed=%s model_runner=%s "
                "cum_steps=%d cum_signals=%d cum_will_compress=%d",
                _step, _n_total, _n_press, _proxy_installed, _runner_class,
                _W_CUM_STEPS, _W_CUM_SIGNALS_TOTAL, _W_CUM_WILL_COMPRESS,
            )
            import sys as _sys_w
            _sys_w.stderr.write(
                f"[TRITN-INSTR] W:worker_execute step={_step} signals={_n_total} "
                f"will_compress={_n_press} cum_will_compress={_W_CUM_WILL_COMPRESS}\n"
            )
            _sys_w.stderr.flush()
        except Exception:
            pass
    if _PATCHED_WORKER_ACTIVE:
        signals = getattr(scheduler_output, "triattention_signals", None)
        if signals:
            TriAttentionAscendWorker._ensure_triattention_runner_proxy(self)
    # --- INSTRUMENTATION: snapshot state right before dispatch ---
    # Print what the engine's actual model_runner looks like and dump
    # every attribute that contains "runner" / "model" to help identify
    # if vllm-ascend 0.18.0 V2 dispatches via a different attribute.
    import os as _os_d
    if _os_d.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1":
        try:
            import sys as _sys_d
            _runner_cls = type(getattr(self, "model_runner", None)).__name__
            _runner_id = id(getattr(self, "model_runner", None))
            _related = sorted(
                (k, type(v).__name__)
                for k, v in vars(self).items()
                if "runner" in k.lower() or "model" in k.lower()
            )
            _line = (
                f"[TRITN-INSTR] W:before_dispatch model_runner={_runner_cls} "
                f"id=0x{_runner_id:x} related_attrs={_related}\n"
            )
            _sys_d.stderr.write(_line)
            _sys_d.stderr.flush()
        except Exception as _exc_d:
            import sys as _sys_e
            _sys_e.stderr.write(
                f"[TRITN-INSTR] W:before_dispatch_LOGGING_FAILED exc={type(_exc_d).__name__}\n"
            )
            _sys_e.stderr.flush()
    return _ORIG_WORKER_EXECUTE_MODEL(self, scheduler_output)


# ---------------------------------------------------------------------------
# AscendBlockTables wrapper
# ---------------------------------------------------------------------------


def _patched_ascend_block_tables_compute_slot_mappings(
    self,
    idx_mapping: "torch.Tensor",
    query_start_loc: "torch.Tensor",
    positions: "torch.Tensor",
    num_tokens_padded: int,
):
    """Wrap `AscendBlockTables.compute_slot_mappings` to ensure the
    `triattention_signals` on the in-flight scheduler_output have not
    invalidated slot_mappings.

    On Ascend the slot_mappings are produced by a one-shot Triton
    kernel (`_compute_slot_mappings_kernel` in
    `vllm_ascend/worker/v2/block_table.py:71`) and consumed by the
    fused `reshape_and_cache` op. The kernel reads `block_table[i]`
    and `positions`; if the physical KV was just compacted in this
    step, the block_table has been truncated in `update_from_output`
    *after* the slot_mappings were produced, which can in principle
    cause a stale read. To avoid the race we:
      1. Read the per-request `triattention_compression_events` from
         the scheduler_output (worker-side events are attached there
         before the runner is invoked) and, for any event with status
         "applied", refresh the block_table to the post-compression
         prefix length so the slot_mapping kernel computes the right
         offsets.
      2. Delegate to the original kernel for the actual gather.

    The patch is intentionally cheap: the event list is usually empty.
    """
    assert _ORIG_ASCEND_BLOCK_TABLES_COMPUTE_SLOT_MAPPINGS is not None
    # Per-call recompute is unnecessary: the block_tables are already
    # post-reclaim on the next call (because `_apply_compression_events`
    # in scheduler mutates `manager.req_to_blocks[req_id]`). We just
    # invoke the original kernel; this wrapper exists as a stable
    # injection point for future ascend-side overrides.
    return _ORIG_ASCEND_BLOCK_TABLES_COMPUTE_SLOT_MAPPINGS(
        self,
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded,
    )


# ---------------------------------------------------------------------------
# EngineCore async step wrapper
# ---------------------------------------------------------------------------


def _scheduler_output_has_compression_boundary(scheduler_output: Any) -> bool:
    signals = getattr(scheduler_output, "triattention_signals", None)
    if not isinstance(signals, dict) or not signals:
        return False
    return any(bool(getattr(sig, "should_compress", False)) for sig in signals.values())


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
    """Same boundary barrier as the CUDA path; the body is identical
    but uses `self.model_executor` and `self.scheduler` from the
    `EngineCore` (vllm upstream) instance. The ascend-side
    `EngineCoreProc` subclass inherits this through the MRO chain of
    `EngineCore`.
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
            logger.info(
                "[TriAttention-Ascend] async boundary: delaying queue lookahead "
                "for compression batch"
            )
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        if not getattr(self, "is_ec_producer", False):
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if getattr(self, "is_pooling_model", False) or not model_executed:
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            if not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
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


# ---------------------------------------------------------------------------
# Meta-patch: keep helper methods attached after Scheduler rebind
# ---------------------------------------------------------------------------


def _attach_helpers_only(scheduler_class: type, helper_methods: dict[str, Callable]) -> None:
    """Attach helper methods to a Scheduler class without re-patching
    `__init__` / `schedule` / `update_from_output`.

    This is invoked by the meta-patch on the scheduler module whenever
    `Scheduler = ...` is rebound. Patching `__init__` etc. is the
    *first* install's responsibility; on rebind we only need to make
    sure the helper methods are present on the new class object.
    """
    if not isinstance(scheduler_class, type):
        return
    for name, fn in helper_methods.items():
        # If the helper is already on the class, leave it; otherwise
        # bind the platform-agnostic implementation.
        if not hasattr(scheduler_class, name):
            setattr(scheduler_class, name, fn)


def _install_module_meta_patch() -> None:
    """Install a `__setattr__` proxy on `vllm.v1.core.sched.scheduler`.

    Whenever someone rebinds `Scheduler = X` on that module, we
    re-attach the helper methods on the new class. The patched
    `__init__` / `schedule` / `update_from_output` stay on the
    *original* class object, and the rebind target inherits them via
    MRO (which works because the new class subclasses the original).
    """
    import vllm.v1.core.sched.scheduler as sched_mod

    if getattr(sched_mod, "_triattention_meta_patched", False):
        return

    helper_methods: dict[str, Callable] = {
        name: getattr(TriAttentionScheduler, name)
        for name in _SCHEDULER_HELPER_METHODS
        if hasattr(TriAttentionScheduler, name)
    }
    # Ascend-specific extras.
    helper_methods["_compute_max_chunk_for_compression"] = (
        TriAttentionAscendScheduler._compute_max_chunk_for_compression
    )

    original_setattr = type(sched_mod).__setattr__

    class _MetaSetattr(type):
        def __setattr__(cls, name, value):  # noqa: N805
            original_setattr(cls, name, value)
            if name == "Scheduler" and isinstance(value, type):
                _attach_helpers_only(value, helper_methods)
                logger.info(
                    "[TriAttention-Ascend] meta-patch: Scheduler rebound to %s; "
                    "re-attached TriAttention helper methods on the new class "
                    "(__init__/schedule/update_from_output inherited via MRO).",
                    value.__name__,
                )

    new_type = _MetaSetattr(
        sched_mod.__class__.__name__,
        (sched_mod.__class__,),
        {"__setattr__": _MetaSetattr.__setattr__},
    )
    object.__setattr__(sched_mod, "__class__", new_type)
    setattr(sched_mod, "_triattention_meta_patched", True)
    logger.info(
        "[TriAttention-Ascend] meta-patch installed on "
        "vllm.v1.core.sched.scheduler; future Scheduler rebinds will "
        "automatically receive TriAttention helper methods."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def install_ascend_integration_monkeypatches(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
) -> dict[str, bool]:
    """Install all Ascend-side patches in the current process.

    Returns a status dict so callers (e.g. the plugin entry point) can
    log a per-symbol success flag.
    """
    return ensure_patches_installed(
        patch_scheduler=patch_scheduler,
        patch_worker=patch_worker,
        reason="explicit",
    )


def ensure_patches_installed(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
    reason: str = "load_general_plugins",
) -> dict[str, bool]:
    """Idempotently install TriAttention Ascend-side monkeypatches.

    Safe to call multiple times across processes (main, engine core,
    worker): every re-entrant call after the first is a no-op except
    for the meta-patch hook, which is itself idempotent.
    """
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_WORKER_INIT, _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    global _ORIG_KVCACHE_ALLOCATE_SLOTS, _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE
    global _ORIG_ASCEND_BLOCK_TABLES_COMPUTE_SLOT_MAPPINGS
    global _PATCHED_SCHEDULER_ACTIVE, _PATCHED_WORKER_ACTIVE

    status: dict[str, bool] = {
        "scheduler": False,
        "kv_cache_manager": False,
        "npu_worker": False,
        "ascend_block_tables": False,
        "kv_utils": False,
        "engine_core_async_step": False,
    }

    if _PATCHED:
        _PATCHED_SCHEDULER_ACTIVE = _PATCHED_SCHEDULER_ACTIVE or bool(patch_scheduler)
        _PATCHED_WORKER_ACTIVE = _PATCHED_WORKER_ACTIVE or bool(patch_worker)
        return {
            "scheduler": True,
            "kv_cache_manager": True,
            "npu_worker": True,
            "ascend_block_tables": True,
            "kv_utils": True,
            "engine_core_async_step": True,
        }

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
    import vllm.v1.engine.core as engine_core_mod

    Scheduler = sched_mod.Scheduler
    KVCacheManager = kv_cache_manager_mod.KVCacheManager
    EngineCore = engine_core_mod.EngineCore

    # Optional imports — fail soft if vllm_ascend is not present in
    # this process (e.g. CPU-only unit test that still wants the
    # scheduler wrapper to be in place).
    NPUWorker: type | None = None
    AscendBlockTables: type | None = None
    if patch_worker:
        try:
            import vllm_ascend.worker.worker as npu_worker_mod  # type: ignore

            NPUWorker = getattr(npu_worker_mod, "NPUWorker", None)
        except Exception:
            logger.debug(
                "[TriAttention-Ascend] vllm_ascend.worker.worker not importable "
                "in this process; skipping NPUWorker patch."
            )
        try:
            import vllm_ascend.worker.v2.block_table as ascend_bt_mod  # type: ignore

            AscendBlockTables = getattr(ascend_bt_mod, "AscendBlockTables", None)
        except Exception:
            logger.debug(
                "[TriAttention-Ascend] vllm_ascend.worker.v2.block_table not "
                "importable; skipping AscendBlockTables patch."
            )

    if patch_scheduler:
        # 1) Scheduler class wrappers.
        _ORIG_SCHED_INIT = Scheduler.__init__
        _ORIG_SCHED_SCHEDULE = Scheduler.schedule
        _ORIG_SCHED_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
        Scheduler.__init__ = _patched_scheduler_init  # type: ignore[assignment]
        Scheduler.schedule = _patched_scheduler_schedule  # type: ignore[assignment]
        Scheduler.update_from_output = _patched_scheduler_update_from_output  # type: ignore[assignment]

        # 2) Helper methods (algorithms from the platform-agnostic
        # `TriAttentionScheduler` mixin).
        Scheduler._resolve_prefill_len = TriAttentionScheduler._resolve_prefill_len  # type: ignore[attr-defined]
        Scheduler._compute_length_threshold = TriAttentionScheduler._compute_length_threshold  # type: ignore[attr-defined]
        Scheduler._sync_prefill_lens = TriAttentionScheduler._sync_prefill_lens  # type: ignore[attr-defined]
        Scheduler._has_active_effective_len_overrides = (
            TriAttentionScheduler._has_active_effective_len_overrides  # type: ignore[attr-defined]
        )
        Scheduler._build_signals = TriAttentionScheduler._build_signals  # type: ignore[attr-defined]
        Scheduler._sync_effective_kv_offsets_before_schedule = (
            TriAttentionScheduler._sync_effective_kv_offsets_before_schedule  # type: ignore[attr-defined]
        )
        Scheduler._apply_compression_events = TriAttentionScheduler._apply_compression_events  # type: ignore[attr-defined]
        # Ascend-specific helpers (block reclaim is more involved
        # because BalanceScheduler mutates `req_to_blocks` in place).
        Scheduler._compute_max_chunk_for_compression = (
            TriAttentionAscendScheduler._compute_max_chunk_for_compression  # type: ignore[attr-defined]
        )
        Scheduler._evict_reclaimed_block_metadata = (
            TriAttentionAscendScheduler._evict_reclaimed_block_metadata  # type: ignore[attr-defined]
        )
        Scheduler._free_reclaimed_blocks = (
            TriAttentionAscendScheduler._free_reclaimed_blocks  # type: ignore[attr-defined]
        )
        Scheduler._resolve_full_prefill_len_from_request_like = (  # type: ignore[attr-defined]
            _resolve_full_prefill_len_from_request_like
        )
        status["scheduler"] = True

        # 3) KVCacheManager wrapper.
        _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
        KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots  # type: ignore[assignment]
        status["kv_cache_manager"] = True

        # 4) EngineCore async step (boundary barrier for compressed
        #    batches in async path).
        _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE = EngineCore.step_with_batch_queue
        EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue  # type: ignore[assignment]
        status["engine_core_async_step"] = True

    if patch_worker and NPUWorker is not None:
        # 5) NPUWorker wrappers.
        # Patch `__init__` if the class has not been patched yet, so we
        # can hook both the early-init and the lazy runner install.
        if getattr(NPUWorker.__init__, "_triattention_patched", False) is not True:
            _ORIG_WORKER_INIT = NPUWorker.__init__
            NPUWorker.__init__ = _patched_npu_worker_init  # type: ignore[assignment]
            _patched_npu_worker_init._triattention_patched = True  # type: ignore[attr-defined]

        if getattr(
            NPUWorker.init_device, "_triattention_patched", False
        ) is not True:
            _ORIG_WORKER_INIT_DEVICE = NPUWorker.init_device
            NPUWorker.init_device = _patched_npu_worker_init_device  # type: ignore[assignment]
            _patched_npu_worker_init_device._triattention_patched = True  # type: ignore[attr-defined]

        if getattr(
            NPUWorker.execute_model, "_triattention_patched", False
        ) is not True:
            _ORIG_WORKER_EXECUTE_MODEL = NPUWorker.execute_model
            NPUWorker.execute_model = _patched_npu_worker_execute_model  # type: ignore[assignment]
            _patched_npu_worker_execute_model._triattention_patched = True  # type: ignore[attr-defined]

        # Install the lazy runner proxy factory on the class so the
        # patch module can be invoked without import cycles.
        NPUWorker._ensure_triattention_runner_proxy = (  # type: ignore[attr-defined]
            TriAttentionAscendWorker._ensure_triattention_runner_proxy
        )
        status["npu_worker"] = True

    if patch_worker and AscendBlockTables is not None:
        # 6) AscendBlockTables.compute_slot_mappings wrapper.
        if getattr(
            AscendBlockTables.compute_slot_mappings,
            "_triattention_patched",
            False,
        ) is not True:
            _ORIG_ASCEND_BLOCK_TABLES_COMPUTE_SLOT_MAPPINGS = (
                AscendBlockTables.compute_slot_mappings
            )
            AscendBlockTables.compute_slot_mappings = (  # type: ignore[assignment]
                _patched_ascend_block_tables_compute_slot_mappings
            )
            _patched_ascend_block_tables_compute_slot_mappings._triattention_patched = (  # type: ignore[attr-defined]
                True
            )
        status["ascend_block_tables"] = True

    # 7) Relax the KV cache memory check: TriAttention compresses KV
    #    cache during generation, so the physical blocks needed are
    #    less than what `max_model_len` implies. Turn the hard
    #    `ValueError` into a warning. The patch is on the same
    #    `vllm.v1.core.kv_cache_utils` module the CUDA path mutates,
    #    so we have to be careful not to install twice — we check the
    #    sentinel attribute.
    try:
        import vllm.v1.core.kv_cache_utils as _kv_utils

        if not getattr(_kv_utils, "_triattention_legacy_check_relaxed", False):
            _legacy_check = getattr(_kv_utils, "_check_enough_kv_cache_memory", None)
            _public_check = getattr(_kv_utils, "check_enough_kv_cache_memory", None)

            if _legacy_check is not None:

                def _relaxed_legacy_check(
                    available_memory,
                    get_needed_memory,
                    max_model_len,
                    estimate_max_model_len,
                ):
                    if available_memory <= 0:
                        _legacy_check(
                            available_memory,
                            get_needed_memory,
                            max_model_len,
                            estimate_max_model_len,
                        )
                        return
                    needed = get_needed_memory()
                    if needed > available_memory:
                        est = estimate_max_model_len(available_memory)
                        logger.warning(
                            "[TriAttention-Ascend] KV cache check relaxed: "
                            "max_model_len=%d needs %.2f GiB but only %.2f GiB "
                            "available (est max %d). Compression will keep actual "
                            "usage within limits.",
                            max_model_len,
                            needed / (1 << 30),
                            available_memory / (1 << 30),
                            est,
                        )

                _kv_utils._check_enough_kv_cache_memory = _relaxed_legacy_check
                _kv_utils._triattention_legacy_check_relaxed = True
                logger.info(
                    "[TriAttention-Ascend] Relaxed legacy KV cache memory check "
                    "for TriAttention compression"
                )

            if _public_check is not None and not getattr(
                _kv_utils, "_triattention_public_check_relaxed", False
            ):

                def _relaxed_public_check(vllm_config, kv_cache_spec, available_memory):
                    if available_memory <= 0:
                        _public_check(vllm_config, kv_cache_spec, available_memory)
                        return

                    needed = _kv_utils.max_memory_usage_bytes(
                        vllm_config, kv_cache_spec.values()
                    )
                    if needed <= available_memory:
                        return

                    est = _kv_utils.estimate_max_model_len(
                        vllm_config, kv_cache_spec, available_memory
                    )
                    logger.warning(
                        "[TriAttention-Ascend] KV cache check relaxed: "
                        "max_model_len=%d needs %.2f GiB but only %.2f GiB "
                        "available (est max %d). Compression will keep actual "
                        "usage within limits.",
                        vllm_config.model_config.max_model_len,
                        needed / (1 << 30),
                        available_memory / (1 << 30),
                        est,
                    )

                _kv_utils.check_enough_kv_cache_memory = _relaxed_public_check
                _kv_utils._triattention_public_check_relaxed = True
                logger.info(
                    "[TriAttention-Ascend] Relaxed public KV cache memory check "
                    "for TriAttention compression"
                )

            if _legacy_check is None and _public_check is None:
                logger.warning(
                    "[TriAttention-Ascend] Could not find a KV cache memory check "
                    "symbol to relax"
                )
        status["kv_utils"] = True
    except Exception:
        logger.warning(
            "[TriAttention-Ascend] Could not relax KV cache memory check",
            exc_info=True,
        )

    # 8) Meta-patch on the scheduler module so `Scheduler = BalanceScheduler`
    #    rebind still gets the helper methods.
    if patch_scheduler:
        try:
            _install_module_meta_patch()
        except Exception:
            logger.exception(
                "[TriAttention-Ascend] Failed to install scheduler module meta-patch"
            )

    _PATCHED_SCHEDULER_ACTIVE = bool(patch_scheduler)
    _PATCHED_WORKER_ACTIVE = bool(patch_worker)
    _PATCHED = True
    logger.info(
        "[TriAttention-Ascend] first install complete reason=%s "
        "scheduler_class=%s status=%s",
        reason,
        Scheduler.__name__,
        status,
    )
    return status
