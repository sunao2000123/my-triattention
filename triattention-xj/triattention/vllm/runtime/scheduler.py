"""TriAttention v2 scheduler integration."""

from __future__ import annotations

import os
from typing import Any

# === BUG-RECL 总开关辅助函数（断点排查专用） ===
# 设 TRIATTN_BUG_RECL_DEBUG=1 启用；默认 0 关闭
def _bug_recl_enabled() -> bool:
    return os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
# === BUG-RECL 总开关辅助函数结束 ===

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

from .ascend_defaults import apply_ascend_fast_recency_defaults
from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
from .fast_recency_guard import should_guard_fast_recency_long_context
from .kv_allocation_sync import (
    clear_request_allocation_sync_state,
    prepare_request_effective_num_computed,
    update_request_effective_kv_offset,
)
from .planner import CompressionPlanner
from .prefill_phase import is_prefill_phase_for_limit
from .request_key_compat import iter_scheduled_token_items
from .signals import CompressionSignal
from .thresholds import (
    compression_length_threshold,
    is_ascend_environment_available,
)
from .version import RUNTIME_BUILD_ID

def _evict_reclaimed_block_metadata(block_pool: Any, block: Any) -> None:
    """Best-effort clear of prefix-cache metadata before reusing a block."""
    if block_pool is None or block is None:
        return
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return

    maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
    if callable(maybe_evict):
        maybe_evict(block)


def _free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
    """Free reclaimed tail blocks after clearing any stale prefix-cache identity."""
    if not removed_blocks:
        return False
    block_pool = getattr(manager, "block_pool", None)
    for block in removed_blocks:
        _evict_reclaimed_block_metadata(block_pool, block)
    if block_pool is None:
        return False
    block_pool.free_blocks(reversed(removed_blocks))
    return True


def _resolve_full_prefill_len_from_request_like(request_like: Any) -> int:
    candidates: list[int] = []

    prompt_token_ids = getattr(request_like, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            candidates.append(len(prompt_token_ids))
        except Exception:
            pass

    for attr_name in ("prompt_token_ids_len", "num_prompt_tokens"):
        raw_value = getattr(request_like, attr_name, None)
        if raw_value is None:
            continue
        try:
            candidates.append(int(raw_value))
        except (TypeError, ValueError):
            continue

    prefill_token_ids = getattr(request_like, "prefill_token_ids", None)
    if prefill_token_ids is not None:
        try:
            candidates.append(len(prefill_token_ids))
        except Exception:
            pass

    return max(candidates, default=0)


def _is_ascend_scheduler_instance(scheduler: Any) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    device_config = getattr(vllm_config, "device_config", None)
    for attr_name in ("device", "device_type"):
        raw = getattr(device_config, attr_name, None)
        if raw is None:
            continue
        value = str(raw).lower()
        if "npu" in value or "ascend" in value:
            return True
    platform = getattr(vllm_config, "platform", None)
    if platform is not None:
        platform_repr = repr(platform).lower()
        if "ascend" in platform_repr or "npu" in platform_repr:
            return True
    return "vllm_ascend" in repr(type(scheduler)) or is_ascend_environment_available()


def _should_defer_prefill_compression_for_scheduler(scheduler: Any) -> bool:
    cfg = getattr(scheduler, "triattention_config", None)
    if cfg is None:
        return False
    if bool(getattr(cfg, "defer_prefill_compression", False)):
        return True
    return bool(getattr(cfg, "defer_prefill_compression_on_ascend", False)) and (
        _is_ascend_scheduler_instance(scheduler)
    )


class TriAttentionScheduler(Scheduler):
    """Scheduler subclass that emits per-request compression signals."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.triattention_config = TriAttentionRuntimeConfig.from_env()
        if _is_ascend_scheduler_instance(self):
            apply_ascend_fast_recency_defaults(self.triattention_config)
        self._planner = CompressionPlanner(self.triattention_config)
        self._effective_len_tracker = EffectiveCacheLenTracker()
        self._prefill_lens: dict[str, int] = {}
        self._prefill_compression_counts: dict[str, int] = {}
        self._length_threshold_cache: dict[str, int] = {}
        self._last_signal_log_steps: dict[str, int] = {}
        self._long_context_guard_logged: set[str] = set()
        self._triattention_step = 0

        if self.triattention_config.logging_enabled:
            logger.info(
                "TriAttentionScheduler initialized: budget=%d divide_length=%d "
                "min_reclaim_blocks_on_ascend=%d protect_prefill=%s "
                "kv_usage_trigger_enabled=%s block_reclaim_enabled=%s "
                "defer_prefill_on_ascend=%s score_max_layers=%d "
                "score_max_layers_on_ascend=%d "
                "prefill_min_reclaim_blocks_on_ascend=%d "
                "prefill_max_compressions_on_ascend=%d "
                "fast_recency_only=%s fast_recency_accuracy_guard=%s "
                "fast_recency_long_context_guard=%s "
                "fast_recency_long_context_guard_tokens=%d "
                "auto_fast_recency_on_ascend=%s "
                "early_install_proxy_on_ascend=%s "
                "zero_copy_recency=%s zero_copy_recency_only_on_ascend=%s "
                "build=%s",
                self.triattention_config.kv_budget,
                self.triattention_config.divide_length,
                self.triattention_config.min_reclaim_blocks_on_ascend,
                self.triattention_config.protect_prefill,
                self.triattention_config.enable_kv_usage_trigger,
                self.triattention_config.enable_experimental_block_reclaim,
                self.triattention_config.defer_prefill_compression_on_ascend,
                self.triattention_config.score_max_layers,
                self.triattention_config.score_max_layers_on_ascend,
                self.triattention_config.prefill_min_reclaim_blocks_on_ascend,
                self.triattention_config.prefill_max_compressions_on_ascend,
                self.triattention_config.fast_recency_only,
                self.triattention_config.fast_recency_accuracy_guard,
                self.triattention_config.fast_recency_long_context_guard,
                self.triattention_config.fast_recency_long_context_guard_tokens,
                self.triattention_config.auto_fast_recency_on_ascend,
                self.triattention_config.early_install_proxy_on_ascend,
                self.triattention_config.enable_zero_copy_recency,
                self.triattention_config.zero_copy_recency_only_on_ascend,
                RUNTIME_BUILD_ID,
            )

    def _resolve_prefill_len(self, req_id: str) -> int:
        if req_id in self._prefill_lens:
            return self._prefill_lens[req_id]
        request = self.requests.get(req_id)
        if request is None:
            return 0
        return request.num_prompt_tokens

    def _compute_length_threshold(
        self,
        prefill_len: int,
        *,
        is_prefill_step: bool = False,
    ) -> int:
        return compression_length_threshold(
            self.triattention_config,
            prefill_len=prefill_len,
            block_size=int(getattr(self, "block_size", 1) or 1),
            is_ascend=_is_ascend_scheduler_instance(self),
            is_prefill_step=is_prefill_step,
        )

    def _ensure_runtime_fields(self) -> None:
        """Lazily initialize fields when methods run on monkeypatched schedulers."""
        if getattr(self, "triattention_config", None) is None:
            self.triattention_config = TriAttentionRuntimeConfig.from_env()
            if _is_ascend_scheduler_instance(self):
                apply_ascend_fast_recency_defaults(self.triattention_config)
        if getattr(self, "_planner", None) is None:
            self._planner = CompressionPlanner(self.triattention_config)
        if getattr(self, "_effective_len_tracker", None) is None:
            self._effective_len_tracker = EffectiveCacheLenTracker()
        if getattr(self, "_prefill_lens", None) is None:
            self._prefill_lens = {}
        if getattr(self, "_prefill_compression_counts", None) is None:
            self._prefill_compression_counts = {}
        if getattr(self, "_length_threshold_cache", None) is None:
            self._length_threshold_cache = {}
        if getattr(self, "_last_signal_log_steps", None) is None:
            self._last_signal_log_steps = {}
        if getattr(self, "_long_context_guard_logged", None) is None:
            self._long_context_guard_logged = set()
        if getattr(self, "_triattention_step", None) is None:
            self._triattention_step = 0

    def _sync_prefill_lens(self, scheduler_output: SchedulerOutput) -> None:
        self._ensure_runtime_fields()
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            is_first_seen = req_id not in self._prefill_lens
            if is_first_seen:
                # Chunked-prefill may surface the same request multiple times in
                # scheduled_new_reqs. Only the first appearance should reset the
                # effective-length tracker; later repeats are continuation of the
                # same lifecycle, not a new request.
                self._effective_len_tracker.reset_request(
                    req_id,
                    new_req.num_computed_tokens,
                )
            prefill_len = _resolve_full_prefill_len_from_request_like(new_req)
            self._prefill_lens[req_id] = prefill_len
            self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)

        for req_id in scheduler_output.finished_req_ids:
            req = self.requests.get(req_id)
            if req is not None:
                clear_request_allocation_sync_state(req)
            self._prefill_lens.pop(req_id, None)
            self._prefill_compression_counts.pop(req_id, None)
            self._length_threshold_cache.pop(req_id, None)
            self._last_signal_log_steps.pop(req_id, None)
            self._long_context_guard_logged.discard(req_id)
            self._effective_len_tracker.remove_request(req_id)

        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            resumed_req_ids: list[str] = []
        else:
            resumed_req_ids = getattr(cached_reqs, "resumed_req_ids", None)
            if resumed_req_ids is None:
                resumed_req_ids = getattr(cached_reqs, "req_ids", []) or []
        for req_id in resumed_req_ids:
            if req_id not in self._prefill_lens:
                prefill_len = self._resolve_prefill_len(req_id)
                self._prefill_lens[req_id] = prefill_len
                self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)

    def _signal_log_interval_steps(self) -> int:
        cfg = self.triattention_config
        block_size = max(1, int(getattr(self, "block_size", 1) or 1))
        if _is_ascend_scheduler_instance(self):
            min_blocks = int(getattr(cfg, "min_reclaim_blocks_on_ascend", 0) or 0)
        else:
            min_blocks = int(getattr(cfg, "min_reclaim_blocks", 0) or 0)
        reclaim_interval = max(1, min_blocks) * block_size
        return max(1, int(getattr(cfg, "divide_length", 1) or 1), reclaim_interval)

    def _should_log_signal_trigger(self, req_id: str) -> bool:
        self._ensure_runtime_fields()
        last_step = self._last_signal_log_steps.get(req_id)
        if last_step is not None:
            elapsed = self._triattention_step - last_step
            if elapsed < self._signal_log_interval_steps():
                return False
        self._last_signal_log_steps[req_id] = self._triattention_step
        return True

    def _log_long_context_guard_skip(
        self,
        *,
        req_id: str,
        effective_tokens: int,
        prefill_len: int,
        scheduled_tokens: int,
    ) -> None:
        self._ensure_runtime_fields()
        if not self.triattention_config.logging_enabled:
            return
        if req_id in self._long_context_guard_logged:
            return
        self._long_context_guard_logged.add(req_id)
        log_fn = (
            logger.info
            if bool(self.triattention_config.log_decisions)
            else logger.debug
        )
        log_fn(
            "TriAttention compression skipped req=%s "
            "reason=fast_recency_long_context_guard effective_tokens=%d "
            "prefill_len=%d scheduled_tokens=%d guard_tokens=%d "
            "fast_recency_only=%s",
            req_id,
            effective_tokens,
            prefill_len,
            scheduled_tokens,
            int(
                getattr(
                    self.triattention_config,
                    "fast_recency_long_context_guard_tokens",
                    0,
                )
                or 0
            ),
            bool(getattr(self.triattention_config, "fast_recency_only", False)),
        )

    def _has_active_effective_len_overrides(self) -> bool:
        self._ensure_runtime_fields()
        checker = getattr(self._effective_len_tracker, "has_any_effective_len_overrides", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    def _build_signals(self, scheduler_output: SchedulerOutput) -> dict[str, CompressionSignal]:
        self._ensure_runtime_fields()
        kv_usage_enabled = bool(self.triattention_config.enable_kv_usage_trigger)
        kv_usage = self.kv_cache_manager.usage if kv_usage_enabled else None
        compression_disabled = bool(self.triattention_config.disable_compression)
        signals: dict[str, CompressionSignal] = {}
        scheduled_items = list(iter_scheduled_token_items(scheduler_output))
        if (
            self.triattention_config.log_decisions
            and not scheduled_items
            and self._triattention_step % 500 == 0
        ):
            raw = getattr(scheduler_output, "num_scheduled_tokens", "MISSING")
            logger.info(
                "TriAttention _build_signals: no scheduled items step=%d "
                "num_scheduled_tokens_type=%s len=%s",
                self._triattention_step,
                type(raw).__name__,
                len(raw) if isinstance(raw, dict) else "N/A",
            )
        for _raw_key, req_id, scheduled_tokens in scheduled_items:
            request = self.requests.get(req_id)
            if request is None:
                continue
            scheduled_tokens_i = int(scheduled_tokens)
            prefill_len = self._prefill_lens.get(req_id)
            if prefill_len is None:
                prefill_len = self._resolve_prefill_len(req_id)
                self._prefill_lens[req_id] = prefill_len
                self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)
            effective_tokens = max(
                int(getattr(request, "num_computed_tokens", 0) or 0),
                prefill_len,
            )
            if should_guard_fast_recency_long_context(
                config=self.triattention_config,
                effective_tokens=effective_tokens,
                prefill_len=prefill_len,
            ):
                TriAttentionScheduler._log_long_context_guard_skip(
                    self,
                    req_id=req_id,
                    effective_tokens=effective_tokens,
                    prefill_len=prefill_len,
                    scheduled_tokens=scheduled_tokens_i,
                )
                continue
            if (
                _should_defer_prefill_compression_for_scheduler(self)
                and scheduled_tokens_i > 1
            ):
                continue
            is_prefill_step = scheduled_tokens_i > 1
            is_prefill_step_for_limit = is_prefill_phase_for_limit(
                scheduler_output=scheduler_output,
                req_id=req_id,
                scheduled_tokens=scheduled_tokens_i,
                prefill_len=prefill_len,
                num_computed_tokens=int(getattr(request, "num_computed_tokens", 0)),
            )
            if (
                _is_ascend_scheduler_instance(self)
                and is_prefill_step_for_limit
                and self._prefill_compression_counts.get(req_id, 0)
                >= int(
                    getattr(
                        self.triattention_config,
                        "prefill_max_compressions_on_ascend",
                        1,
                    )
                    or 0
                )
            ):
                continue
            has_override = self._effective_len_tracker.has_effective_len_override(req_id)
            if has_override:
                effective_base_len = self._effective_len_tracker.observe_num_computed(
                    req_id=req_id,
                    num_computed_tokens=request.num_computed_tokens,
                )
            else:
                # Common pre-compression path: effective cache length is exactly
                # num_computed_tokens, so avoid tracker writes in the decode hot path.
                effective_base_len = request.num_computed_tokens
            estimated_cache_len = effective_base_len + scheduled_tokens_i
            if not has_override:
                if compression_disabled and not kv_usage_enabled:
                    continue
                if not kv_usage_enabled and not compression_disabled:
                    if is_prefill_step:
                        threshold = self._compute_length_threshold(
                            prefill_len,
                            is_prefill_step=True,
                        )
                    else:
                        threshold = self._length_threshold_cache.get(req_id)
                    if threshold is None:
                        threshold = self._compute_length_threshold(
                            prefill_len,
                            is_prefill_step=is_prefill_step,
                        )
                        self._length_threshold_cache[req_id] = threshold
                        if self.triattention_config.log_decisions:
                            logger.info(
                                "TriAttention threshold computed req=%s threshold=%d "
                                "prefill_len=%d is_prefill_step=%s budget=%d divide_length=%d",
                                req_id, threshold, prefill_len,
                                is_prefill_step,
                                self.triattention_config.kv_budget,
                                self.triattention_config.divide_length,
                            )
                    if estimated_cache_len < threshold:
                        continue

            signal = self._planner.build_signal(
                req_id=req_id,
                estimated_cache_len=estimated_cache_len,
                prefill_len=prefill_len,
                step=self._triattention_step,
                kv_usage=kv_usage,
                scheduled_tokens=scheduled_tokens_i,
                length_threshold=self._compute_length_threshold(
                    prefill_len,
                    is_prefill_step=is_prefill_step,
                ),
            )
            # Keep scheduler->runner side-channel sparse to reduce per-step IPC
            # metadata overhead in the common no-compression decode path.
            #
            # Runner only needs full signal payload for:
            # 1) compression trigger execution in this step; or
            # 2) requests that have already been compressed and still need
            #    effective-length updates for runtime input overrides.
            if signal.should_compress or has_override:
                if signal.should_compress and self.triattention_config.logging_enabled:
                    log_fn = (
                        logger.info
                        if bool(self.triattention_config.log_decisions)
                        else logger.debug
                    )
                    if TriAttentionScheduler._should_log_signal_trigger(self, req_id):
                        log_fn(
                            "TriAttention signal triggered req=%s step=%d "
                            "estimated_cache_len=%d reason=%s",
                            req_id, self._triattention_step,
                            estimated_cache_len, signal.reason,
                        )
                signals[req_id] = signal
        return signals

    def _sync_effective_kv_offsets_before_schedule(self) -> None:
        self._ensure_runtime_fields()
        running = getattr(self, "running", None)
        if not isinstance(running, list):
            return
        for request in running:
            prepare_request_effective_num_computed(request)

    def _compute_max_chunk_for_compression(self) -> int | None:
        """Max tokens per step to allow compression cycling within physical KV."""
        block_pool = getattr(getattr(self, "kv_cache_manager", None), "block_pool", None)
        if block_pool is None:
            return None
        total_blocks = getattr(block_pool, "num_gpu_blocks", 0)
        if total_blocks <= 0:
            return None
        block_size = int(getattr(self, "block_size", 16) or 16)
        physical_kv = total_blocks * block_size
        headroom = physical_kv - self.triattention_config.kv_budget
        if headroom <= 0:
            return None
        # Leave a small margin (one block) for allocation bookkeeping.
        headroom = max(1, headroom - block_size)
        return headroom

    def schedule(self) -> SchedulerOutput:
        self._sync_effective_kv_offsets_before_schedule()

        orig_max_scheduled = None
        if not self.triattention_config.disable_compression:
            max_chunk = self._compute_max_chunk_for_compression()
            if max_chunk is not None:
                current_max = getattr(self, "max_num_scheduled_tokens", None)
                if current_max is not None and max_chunk < current_max:
                    orig_max_scheduled = current_max
                    self.max_num_scheduled_tokens = max_chunk

        scheduler_output = super().schedule()

        if orig_max_scheduled is not None:
            self.max_num_scheduled_tokens = orig_max_scheduled

        self._triattention_step += 1
        self._sync_prefill_lens(scheduler_output)
        if (
            self.triattention_config.disable_compression
            and not self.triattention_config.enable_kv_usage_trigger
            and not self._has_active_effective_len_overrides()
        ):
            # FullKV / no-compression path: avoid per-step planner work entirely.
            triattention_signals = {}
        else:
            triattention_signals = self._build_signals(scheduler_output)

        # Attach v2 side-channel metadata to scheduler output.
        setattr(scheduler_output, "triattention_step", self._triattention_step)
        setattr(scheduler_output, "triattention_signals", triattention_signals)

        if self.triattention_config.log_decisions and triattention_signals:
            hits = [
                req_id
                for req_id, signal in triattention_signals.items()
                if signal.should_compress
            ]
            if hits:
                logger.debug(
                    "TriAttention schedule step=%d trigger_reqs=%s",
                    self._triattention_step,
                    hits,
                )

        return scheduler_output

    def _apply_compression_events(self, compression_events: list[dict[str, Any]]) -> None:
        self._ensure_runtime_fields()
        coordinator = getattr(self.kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        block_size = int(getattr(self, "block_size", 1))
        if block_size <= 0:
            block_size = 1
        if self.triattention_config.log_decisions:
            logger.debug(
                "TriAttention _apply_compression_events: kv_cache_manager=%s "
                "coordinator=%s managers=%s block_size=%d reclaim_enabled=%s",
                type(self.kv_cache_manager).__name__,
                type(coordinator).__name__ if coordinator else None,
                type(managers).__name__ if managers else None,
                block_size,
                getattr(self, "triattention_config", None)
                and self.triattention_config.enable_experimental_block_reclaim,
            )
        # === BUG-RECL 断点 8（LE-S-002）：_apply_compression_events 入口 ===
        if _bug_recl_enabled():
            n_events = len(compression_events) if isinstance(compression_events, list) else 0
            n_applied = sum(1 for e in compression_events if isinstance(e, dict) and e.get("status") == "applied") if isinstance(compression_events, list) else 0
            print(f"BUG-RECL [LE-S-002] enter _apply_compression_events n_events={n_events} n_applied={n_applied} reclaim_enabled={self.triattention_config.enable_experimental_block_reclaim}", flush=True)
        # === BUG-RECL 断点 8 结束 ===

        def _num_required_blocks(token_len: int) -> int:
            if token_len <= 0:
                return 0
            return (token_len + block_size - 1) // block_size

        for event in compression_events:
            if event.get("status") != "applied":
                continue
            req_id = event.get("req_id")
            if req_id is None:
                continue
            event_step = int(event.get("step", -1))
            cache_len_after = event.get("cache_len_after")
            if not isinstance(cache_len_after, int):
                continue
            req = self.requests.get(req_id)
            if req is None:
                continue
            prefill_len = self._prefill_lens.get(req_id)
            if prefill_len is None:
                try:
                    prefill_len = int(event.get("prefill_len", 0) or 0)
                except Exception:
                    prefill_len = 0
            scheduled_tokens = int(event.get("scheduled_tokens", 1) or 1)
            num_computed_tokens = int(getattr(req, "num_computed_tokens", 0) or 0)
            if (
                scheduled_tokens > 1
                or (prefill_len > 0 and num_computed_tokens < prefill_len)
            ):
                self._prefill_compression_counts[req_id] = (
                    self._prefill_compression_counts.get(req_id, 0) + 1
                )
            self._effective_len_tracker.apply_compression(
                req_id=req_id,
                cache_len_after=cache_len_after,
                num_computed_tokens=req.num_computed_tokens,
            )
            self._last_signal_log_steps.pop(req_id, None)

            _evt_scheduled = int(event.get("scheduled_tokens", 1))
            if not self.triattention_config.enable_experimental_block_reclaim:
                continue
            details = event.get("details")
            retained_cache_len = (
                details.get("retained_cache_len")
                if isinstance(details, dict)
                else None
            )
            if not isinstance(retained_cache_len, int):
                retained_cache_len = cache_len_after
            required_blocks = _num_required_blocks(retained_cache_len)
            expected_shrink_gids: set[int] = set()
            reclaim_applied_any = False
            req_groups_seen = 0
            if isinstance(managers, (list, tuple)):
                for gid, manager in enumerate(managers):
                    req_blocks = manager.req_to_blocks.get(req_id)
                    if req_blocks and required_blocks < len(req_blocks):
                        expected_shrink_gids.add(gid)
                    if req_blocks:
                        req_groups_seen += 1

            block_reclaim = event.get("block_reclaim")
            reclaim_mode = (
                block_reclaim.get("mode")
                if isinstance(block_reclaim, dict)
                else "truncate_tail"
            )
            if reclaim_mode not in {"truncate_tail", "remap_tail"}:
                reclaim_mode = "truncate_tail"
            groups = (
                block_reclaim.get("groups")
                if isinstance(block_reclaim, dict)
                else None
            )
            if self.triattention_config.log_decisions:
                logger.debug(
                    "TriAttention block reclaim: req=%s required_blocks=%d "
                    "expected_shrink_gids=%s block_reclaim=%s groups=%s",
                    req_id, required_blocks, expected_shrink_gids,
                    type(block_reclaim).__name__ if block_reclaim else None,
                    bool(groups),
                )
            if not isinstance(groups, list):
                # In V1 batch-queue mode, consecutive compression steps can
                # race: the worker already truncated blocks in an earlier step
                # whose events the scheduler hasn't consumed yet.  When that
                # happens the later event legitimately has block_reclaim=None.
                # Synthesize the reclaim by truncating to required_blocks.
                #
                # Safety: during chunked prefill (scheduled_tokens > 1),
                # _update_states may have appended new blocks after the hook
                # ran.  Without block_ids_before we cannot distinguish old
                # blocks from new ones — skip synthesis to avoid freeing
                # blocks the worker is still using.
                if _evt_scheduled > 1:
                    # === BUG-RECL 断点 10（LE-S-004）：synthesized prefill 跳过 ===
                    if _bug_recl_enabled():
                        _exp_sg = sorted(expected_shrink_gids) if 'expected_shrink_gids' in dir() else []
                        print(f"BUG-RECL [LE-S-004] synth-skip req={req_id} _evt_scheduled={_evt_scheduled} is_prefill=True expected_shrink_gids={_exp_sg}", flush=True)
                    # === BUG-RECL 断点 10 结束 ===
                    if self.triattention_config.log_decisions:
                        logger.debug(
                            "TriAttention block reclaim: skipping synthesized "
                            "reclaim during prefill (no groups, "
                            "scheduled_tokens=%d) req=%s",
                            _evt_scheduled, req_id,
                        )
                elif expected_shrink_gids and isinstance(managers, (list, tuple)):
                    for gid in sorted(expected_shrink_gids):
                        manager = managers[gid]
                        req_blocks = manager.req_to_blocks.get(req_id)
                        if not req_blocks or required_blocks >= len(req_blocks):
                            continue
                        kept_blocks = req_blocks[:required_blocks]
                        removed_blocks = req_blocks[required_blocks:]
                        manager.req_to_blocks[req_id] = kept_blocks
                        if req_id in manager.num_cached_block:
                            manager.num_cached_block[req_id] = min(
                                manager.num_cached_block[req_id],
                                len(kept_blocks),
                            )
                        if _free_reclaimed_blocks(manager, removed_blocks):
                            reclaim_applied_any = True
                # === BUG-RECL 断点 7a（LE-S-001a）：synthesized 路径 reclaim_applied_any 终态 ===
                if _bug_recl_enabled():
                    print(f"BUG-RECL [LE-S-001a] synth-path req={req_id} reclaim_applied_any={reclaim_applied_any}", flush=True)
                # === BUG-RECL 断点 7a 结束 ===
                if reclaim_applied_any:
                    update_request_effective_kv_offset(
                        request=req,
                        cache_len_after=cache_len_after,
                    )
                continue
            if not isinstance(managers, (list, tuple)):
                continue

            seen_gids: set[int] = set()
            for group in groups:
                if not isinstance(group, dict):
                    continue
                gid = group.get("gid")
                block_ids_after = group.get("block_ids_after")
                if not isinstance(gid, int) or gid < 0 or gid >= len(managers):
                    continue
                if not isinstance(block_ids_after, list):
                    continue
                if not all(isinstance(block_id, int) for block_id in block_ids_after):
                    continue

                manager = managers[gid]
                req_blocks = manager.req_to_blocks.get(req_id)
                if not req_blocks:
                    continue

                seen_gids.add(gid)
                curr_ids = [block.block_id for block in req_blocks]
                kept_len = len(block_ids_after)
                if kept_len > len(curr_ids):
                    raise RuntimeError(
                        "TriAttention block reclaim invalid length: "
                        f"req={req_id} gid={gid} kept_len={kept_len} "
                        f"curr_len={len(curr_ids)}"
                    )
                if len(set(block_ids_after)) != kept_len:
                    raise RuntimeError(
                        "TriAttention block reclaim contains duplicate block ids: "
                        f"req={req_id} gid={gid} block_ids_after={block_ids_after}"
                    )
                # Use block_ids_before to distinguish old blocks (present
                # when the hook ran) from new blocks appended by
                # _update_states after the hook.  Only free old tail blocks
                # that the hook removed; preserve new blocks the worker needs.
                block_ids_before = group.get("block_ids_before")
                if isinstance(block_ids_before, list):
                    original_count = len(block_ids_before)
                else:
                    original_count = len(req_blocks)
                new_blocks_this_step = list(req_blocks[original_count:])
                old_blocks = list(req_blocks[:original_count])

                if reclaim_mode == "remap_tail":
                    old_by_id = {block.block_id: block for block in old_blocks}
                    missing_ids = [
                        block_id
                        for block_id in block_ids_after
                        if block_id not in old_by_id
                    ]
                    if missing_ids:
                        raise RuntimeError(
                            "TriAttention block remap references unknown blocks: "
                            f"req={req_id} gid={gid} missing={missing_ids} "
                            f"curr_ids={curr_ids}"
                        )
                    kept_old_blocks = [old_by_id[block_id] for block_id in block_ids_after]
                    removed_ids_raw = group.get("block_ids_removed")
                    if (
                        isinstance(removed_ids_raw, list)
                        and all(isinstance(block_id, int) for block_id in removed_ids_raw)
                    ):
                        removed_ids = [
                            block_id for block_id in removed_ids_raw
                            if block_id in old_by_id
                        ]
                    else:
                        kept_set = set(block_ids_after)
                        removed_ids = [
                            block.block_id
                            for block in old_blocks
                            if block.block_id not in kept_set
                        ]
                    kept_set = set(block_ids_after)
                    removed_old_blocks = [
                        old_by_id[block_id]
                        for block_id in removed_ids
                        if block_id in old_by_id and block_id not in kept_set
                    ]
                else:
                    expected_prefix = curr_ids[:kept_len]
                    if expected_prefix != block_ids_after:
                        raise RuntimeError(
                            "TriAttention block reclaim prefix mismatch: "
                            f"req={req_id} gid={gid} "
                            f"expected_prefix={expected_prefix} "
                            f"actual_after={block_ids_after}"
                        )
                    if (
                        getattr(
                            self.triattention_config,
                            "require_physical_reclaim",
                            False,
                        )
                        and gid in expected_shrink_gids
                        and kept_len != required_blocks
                    ):
                        raise RuntimeError(
                            "TriAttention block reclaim insufficient shrink: "
                            f"req={req_id} gid={gid} kept_len={kept_len} "
                            f"required_blocks={required_blocks}"
                        )
                    kept_old_blocks = list(req_blocks[:kept_len])
                    removed_old_blocks = list(req_blocks[kept_len:original_count])

                # Reassemble: kept old blocks + new blocks from this step.
                # remap_tail keeps old tail blocks in compact logical order;
                # truncate_tail keeps the compacted old prefix.
                reassembled = kept_old_blocks + new_blocks_this_step
                manager.req_to_blocks[req_id] = reassembled
                if req_id in manager.num_cached_block:
                    manager.num_cached_block[req_id] = min(
                        manager.num_cached_block[req_id], len(reassembled)
                    )
                if removed_old_blocks:
                    if self.triattention_config.log_decisions:
                        logger.debug(
                            "TriAttention scheduler FREE_BLOCKS: req=%s gid=%d "
                            "freed=%d kept=%d new=%d",
                            req_id, gid, len(removed_old_blocks),
                            len(kept_old_blocks), len(new_blocks_this_step),
                        )
                    # === BUG-RECL 断点 9（LE-S-003）：truncate_tail 物理回收入口 ===
                    if _bug_recl_enabled():
                        print(f"BUG-RECL [LE-S-003] pre-free-blocks req={req_id} gid={gid} mode=truncate_tail removed_n={len(removed_old_blocks)}", flush=True)
                    # === BUG-RECL 断点 9 结束 ===
                    if _free_reclaimed_blocks(manager, removed_old_blocks):
                        reclaim_applied_any = True

            # Synthesize reclaim for groups that were expected but not
            # covered by the explicit block_reclaim payload (V1 batch-queue
            # race — worker already truncated in an earlier step).
            # Same safety as above: skip during chunked prefill without
            # block_ids_before to avoid freeing new blocks.
            missing_gids = expected_shrink_gids - seen_gids
            if reclaim_mode == "remap_tail":
                missing_gids = set()
            if missing_gids and _evt_scheduled <= 1:
                for gid in sorted(missing_gids):
                    manager = managers[gid]
                    req_blocks = manager.req_to_blocks.get(req_id)
                    if not req_blocks or required_blocks >= len(req_blocks):
                        continue
                    kept_blocks = req_blocks[:required_blocks]
                    removed_blocks = req_blocks[required_blocks:]
                    manager.req_to_blocks[req_id] = kept_blocks
                    if req_id in manager.num_cached_block:
                        manager.num_cached_block[req_id] = min(
                            manager.num_cached_block[req_id],
                            len(kept_blocks),
                        )
                    if _free_reclaimed_blocks(manager, removed_blocks):
                        reclaim_applied_any = True
            elif missing_gids and _evt_scheduled > 1:
                if self.triattention_config.log_decisions:
                    logger.debug(
                        "TriAttention block reclaim: skipping synthesized "
                        "reclaim for missing gids %s during prefill "
                        "(scheduled_tokens=%d) req=%s",
                        sorted(missing_gids), _evt_scheduled, req_id,
                    )

            # === BUG-RECL 断点 7b（LE-S-001b）：主路径 reclaim_applied_any 终态 ===
            if _bug_recl_enabled():
                _req = self.requests.get(req_id)
                _offset = getattr(_req, "_triattention_effective_kv_offset", None) if _req is not None else None
                print(f"BUG-RECL [LE-S-001b] main-path req={req_id} reclaim_applied_any={reclaim_applied_any} view4_offset_after={_offset}", flush=True)
            # === BUG-RECL 断点 7b 结束 ===
            if reclaim_applied_any:
                update_request_effective_kv_offset(
                    request=req,
                    cache_len_after=cache_len_after,
                )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)

        compression_events = getattr(
            model_runner_output,
            "triattention_compression_events",
            None,
        )
        if compression_events:
            if self.triattention_config.log_decisions:
                logger.debug(
                    "TriAttention compression events step=%d events=%s",
                    self._triattention_step,
                    compression_events,
                )
            self._apply_compression_events(compression_events)
            usage = float(self.kv_cache_manager.usage)
            for engine_output in outputs.values():
                scheduler_stats = getattr(engine_output, "scheduler_stats", None)
                if scheduler_stats is not None:
                    scheduler_stats.kv_cache_usage = usage

        for req_id in scheduler_output.finished_req_ids:
            self._prefill_lens.pop(req_id, None)
            self._prefill_compression_counts.pop(req_id, None)
            self._long_context_guard_logged.discard(req_id)
            self._effective_len_tracker.remove_request(req_id)
        return outputs
