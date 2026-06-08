"""TriAttention Ascend-side scheduler mixin.

This module re-exports the platform-agnostic scheduler logic from
`triattention.vllm.runtime.scheduler` (which is what the CUDA path
uses) and adds three Ascend-specific extras:

1. `_compute_max_chunk_for_compression` — Ascend's
   `NPUModelRunner.__init__` does ACL graph compilation and device
   warmup that would crash if the scheduler forces a tiny chunked
   prefill before the first decode. We use the same formula as the
   CUDA path, but the Ascend scheduler has slightly different
   `kv_cache_manager.block_pool` attributes (e.g. `num_npu_blocks`
   instead of `num_gpu_blocks`); the wrapper tries both.

2. `_evict_reclaimed_block_metadata` and `_free_reclaimed_blocks` —
   on Ascend, the BlockPool's prefix-cache eviction hook is
   `_maybe_evict_cached_block`, the same as the CUDA path. The
   `_free_reclaimed_blocks` helper here is reused for the descend
   path's `_apply_compression_events`, but adds a guard that the
   block_pool is not None (the BalanceScheduler's `kv_cache_config`
   may not be fully populated at the very first schedule step).

3. `_apply_compression_events` (override) — the BalanceScheduler's
   `kv_cache_manager.coordinator` is the same as the upstream
   Scheduler, but the `manager.req_to_blocks[req_id] = kept`
   mutation that the CUDA path performs is enough on the ascend
   side; we just guard the coordinator lookup and skip the mutation
   gracefully if the manager has not been set up yet.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from triattention.vllm.runtime.scheduler import (
    TriAttentionScheduler,
    _evict_reclaimed_block_metadata as _shared_evict_reclaimed_block_metadata,
    _free_reclaimed_blocks as _shared_free_reclaimed_blocks,
    _resolve_full_prefill_len_from_request_like as _shared_resolve_full_prefill_len,
)

logger = init_logger(__name__)


class TriAttentionAscendScheduler:
    """Mixin-style surface used by `integration_monkeypatch.py` to
    attach Ascend-specific scheduler methods.

    Every method here is also reachable via
    `triattention.vllm.runtime.scheduler.TriAttentionScheduler`; the
    Ascend wrapper exists so the AIM integration monkeypatch has a
    clear `TriAttentionAscendScheduler._method_name` call site and
    so a future refactor (e.g. an NPU-specific budget formula) does
    not have to touch the platform-agnostic scheduler.
    """

    # Re-export the upstream scheduler's helper methods so the AIM
    # integration monkeypatch can attach them onto the upstream
    # `Scheduler` class.
    _resolve_prefill_len = TriAttentionScheduler._resolve_prefill_len
    _compute_length_threshold = TriAttentionScheduler._compute_length_threshold
    _sync_prefill_lens = TriAttentionScheduler._sync_prefill_lens
    _has_active_effective_len_overrides = (
        TriAttentionScheduler._has_active_effective_len_overrides
    )
    _build_signals = TriAttentionScheduler._build_signals
    _sync_effective_kv_offsets_before_schedule = (
        TriAttentionScheduler._sync_effective_kv_offsets_before_schedule
    )
    _apply_compression_events = TriAttentionScheduler._apply_compression_events

    @staticmethod
    def _evict_reclaimed_block_metadata(block_pool: Any, block: Any) -> None:
        return _shared_evict_reclaimed_block_metadata(block_pool, block)

    @staticmethod
    def _free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
        """Wrapper around the platform-agnostic helper that also
        tolerates an empty / half-initialised `block_pool` on the
        ascend side (the BalanceScheduler may schedule a noop batch
        before the block pool is fully built).
        """
        if not removed_blocks:
            return False
        try:
            return _shared_free_reclaimed_blocks(manager, removed_blocks)
        except Exception:
            logger.warning(
                "[TriAttention-Ascend] _free_reclaimed_blocks failed; block pool may be uninitialised",
                exc_info=True,
            )
            return False

    @staticmethod
    def _resolve_full_prefill_len_from_request_like(request_like: Any) -> int:
        return _shared_resolve_full_prefill_len(request_like)

    @staticmethod
    def _compute_max_chunk_for_compression(
        self, cfg
    ) -> int | None:
        """Ascend-specific chunk cap.

        Identical formula to the CUDA path; the only Ascend-specific
        accommodation is that `block_pool` may report
        `num_npu_blocks` instead of `num_gpu_blocks` (the ascend
        BlockPool renames the attribute for clarity).
        """
        block_pool = getattr(getattr(self, "kv_cache_manager", None), "block_pool", None)
        if block_pool is None:
            return None
        # Try both `num_npu_blocks` (ascend) and `num_gpu_blocks`
        # (vllm upstream); fall back to scanning for any attribute
        # whose name starts with "num_".
        total_blocks = (
            getattr(block_pool, "num_npu_blocks", 0)
            or getattr(block_pool, "num_gpu_blocks", 0)
        )
        if total_blocks <= 0:
            for attr in dir(block_pool):
                if attr.startswith("num_") and attr.endswith("_blocks"):
                    try:
                        total_blocks = int(getattr(block_pool, attr) or 0)
                    except Exception:
                        continue
                    if total_blocks > 0:
                        break
        if total_blocks <= 0:
            return None
        block_size = int(getattr(self, "block_size", 16) or 16)
        physical_kv = total_blocks * block_size
        headroom = physical_kv - cfg.kv_budget
        if headroom <= 0:
            return None
        headroom = max(1, headroom - block_size)
        return headroom
