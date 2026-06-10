"""TriAttention Ascend-side GPU sequence-length patch.

**The actual hot path on vLLM-Ascend 0.18.0 is V1, not V2.** Earlier
revisions of this module patched
`vllm_ascend.worker.v2.block_table.AscendBlockTables.compute_slot_mappings`
(Triton kernel) which is NOT the path actually invoked by
`NPUModelRunner._prepare_inputs` on vllm-ascend 0.18.0 — that path calls
`self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)`,
which is the **numpy CPU path** in
`vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping`.  The
V2 Triton-kernel class is only instantiated by the formal V2 model state
path (not exercised by `NPUInputBatch`).  Patching the wrong class
silently no-op'd the entire fix.

**Why a patch is mandatory on Ascend.** Both V1 and V2 paths index
`block_table[row, position // block_size]` **unconditionally** to look
up the block id for a token.  Neither path consults
`num_blocks_per_row[row]`.  After a TriAttention compaction:

  - The scheduler has already truncated `manager.req_to_blocks[req_id]`
    and called `block_pool.free_blocks(...)` to return tail blocks to
    the free list.
  - The worker_reclaim_sync has decremented
    `num_blocks_per_row[req_index]` to match the post-reclaim count.
  - But the **block-id slots in the row at indices `>=
    num_blocks_per_row[req_index]` still hold the freed block ids**,
    which by the time the next `compute_slot_mapping` runs may have been
    recycled to another request.
  - When the next scheduler step schedules this request with
    `positions` (the absolute decode position) growing past the
    post-reclaim cap, `position // block_size` lands in the freed tail
    range, and the kernel reads *another request's KV* at that token.

This module wraps the V1 `BlockTable.compute_slot_mapping` and the
V1 `MultiGroupBlockTable.compute_slot_mapping` to detect those
out-of-bounds token positions and rewrite the resulting `slot_mapping`
slots to `PAD_SLOT_ID`, so the attention kernel masks them instead of
attending over foreign KV data.

Engineering principles respected:

- **Minimal intrusion:** the patch is installed once at process start
  via `setattr(BlockTable, "compute_slot_mapping", wrapper)` and
  `setattr(MultiGroupBlockTable, "compute_slot_mapping", wrapper)`; no
  vllm-ascend source file is touched.
- **Signal driven:** the per-req cap tensor is rebuilt from
  `block_table.num_blocks_per_row` arrays the runner / scheduler
  already maintain.  No new IPC is needed.
- **Lazy loading:** the patch is registered the first time
  `install_seq_len_override_patch()` is called, which happens lazily on
  the first execute_model that bears a non-empty
  `triattention_signals` payload.
- **Explicit state sync:** the cap tensor is derived directly from
  `num_blocks_per_row` (the same source of truth as the scheduler's
  `manager.req_to_blocks` truncation), so the kernel sees exactly the
  blocks the scheduler allocated.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process state for the override window.
# ---------------------------------------------------------------------------


_ASCEND_PATCH_INSTALLED: bool = False
_ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING: Any | None = None
_ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING: Any | None = None


def _debug_disable_seq_override() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_SEQ_OVERRIDE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mask_ascend_slot_mapping_to_num_blocks_per_row(
    *,
    table: Any,
    req_indices: np.ndarray,
    positions: np.ndarray,
    out_slot_mapping: np.ndarray,
) -> int:
    """Rewrite the already-computed `slot_mapping` for out-of-bounds tokens.

    Both V1 and V2 paths index `block_table[row, position // block_size]`
    unconditionally, ignoring `num_blocks_per_row[row]`.  After a
    TriAttention compaction, any token whose `block_index = position //
    block_size` is `>= num_blocks_per_row[req]` reads a recycled block
    id and the attention kernel computes over foreign KV.  Here we
    walk the same index that the original kernel walked and, for any
    token whose `block_index >= num_blocks_per_row[req]`, set the
    corresponding slot in `out_slot_mapping` to `PAD_SLOT_ID`.

    Returns the number of tokens that were masked.
    """
    try:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore
    except Exception:
        PAD_SLOT_ID = -1

    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if num_blocks_per_row is None or not isinstance(num_blocks_per_row, np.ndarray):
        return 0

    block_size = int(getattr(table, "block_size", 0) or 0)
    if block_size <= 0:
        return 0

    blocks_per_phys_block = int(getattr(table, "blocks_per_phys_block", 1) or 1)
    if blocks_per_phys_block <= 0:
        blocks_per_phys_block = 1

    dcp_world_size = int(getattr(table, "dcp_world_size", 1) or 1)
    pcp_world_size = int(getattr(table, "pcp_world_size", 1) or 1)
    cp_world = dcp_world_size * pcp_world_size
    cp_kv_cache_interleave_size = int(
        getattr(table, "cp_kv_cache_interleave_size", 1) or 1
    )
    if cp_kv_cache_interleave_size <= 0:
        cp_kv_cache_interleave_size = 1
    use_hybrid_blocks = bool(getattr(table, "use_hybrid_blocks", False))
    if use_hybrid_blocks:
        # hybrid mode expands the per-row block count by
        # blocks_per_phys_block; the V1 path already accounts for that
        # in `block_table_indices` (line 181 of the V1 BlockTable).
        # We mirror it here so the per-row cap matches the indexing.
        pass

    # When context parallelism is on, BlockTable.compute_slot_mapping
    # uses a "virtual block" of size block_size * cp_world_size for
    # block-table-index calculation.  Mirror that here.
    if cp_world > 1:
        virtual_block_size = block_size * cp_world
        block_indices = positions // virtual_block_size
    else:
        block_indices = positions // block_size

    # The V1 path ALSO does `block_table_indices = req_indices *
    # max_num_blocks_per_req * blocks_per_phys_block + logical_block_idx`.
    # We don't need to reproduce that; we only need the per-token
    # `num_blocks_per_row[req]` to compare against.
    if use_hybrid_blocks:
        logical_cap = num_blocks_per_row[req_indices] * blocks_per_phys_block
    else:
        logical_cap = num_blocks_per_row[req_indices]

    # out-of-bounds: block_index >= logical_cap
    overflow = block_indices >= logical_cap
    # Don't touch padding cells the caller may have filled beyond the
    # valid token count.
    n = min(int(out_slot_mapping.shape[0]), int(positions.shape[0]))
    if overflow.shape[0] != n:
        overflow = overflow[:n]
    n_overflow = int(overflow.sum())
    if n_overflow <= 0:
        return 0
    out_slot_mapping[:n][overflow] = int(PAD_SLOT_ID)
    return n_overflow


def _patched_v1_blocktable_compute_slot_mapping(
    self: Any,
    req_indices: np.ndarray,
    positions: np.ndarray,
) -> None:
    """V1 numpy path wrapper.  Calls the original, then masks the
    out-of-bounds slots based on `num_blocks_per_row`.
    """
    assert _ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING is not None
    if _debug_disable_seq_override():
        return _ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING(self, req_indices, positions)

    _ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING(self, req_indices, positions)

    # The V1 BlockTable writes to `self.slot_mapping.np[:req_indices.shape[0]]`.
    out = getattr(getattr(self, "slot_mapping", None), "np", None)
    if out is None:
        return
    n_masked = _mask_ascend_slot_mapping_to_num_blocks_per_row(
        table=self,
        req_indices=req_indices,
        positions=positions,
        out_slot_mapping=out,
    )
    if n_masked > 0:
        logger.debug(
            "[TriAttention-Ascend] v1 BlockTable.compute_slot_mapping: "
            "masked %d/%d tokens beyond num_blocks_per_row",
            n_masked,
            int(req_indices.shape[0]),
        )


def _patched_v1_multigroupblocktable_compute_slot_mapping(
    self: Any,
    req_indices: np.ndarray,
    positions: np.ndarray,
) -> None:
    """V1 MultiGroupBlockTable wrapper.  Forwards to the original, then
    masks each per-group table.
    """
    assert _ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING is not None
    if _debug_disable_seq_override():
        return _ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING(
            self, req_indices, positions
        )

    _ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING(self, req_indices, positions)

    inner = getattr(self, "block_tables", None)
    if not isinstance(inner, list):
        return
    for table in inner:
        out = getattr(getattr(table, "slot_mapping", None), "np", None)
        if out is None:
            continue
        n_masked = _mask_ascend_slot_mapping_to_num_blocks_per_row(
            table=table,
            req_indices=req_indices,
            positions=positions,
            out_slot_mapping=out,
        )
        if n_masked > 0:
            logger.debug(
                "[TriAttention-Ascend] v1 MultiGroupBlockTable.compute_slot_mapping: "
                "masked %d/%d tokens beyond num_blocks_per_row on one group",
                n_masked,
                int(req_indices.shape[0]),
            )


def _install_ascend_v1_patch() -> bool:
    """Idempotently install the V1 slot_mappings mask on the
    `vllm_ascend.worker.block_table` classes.
    """
    global _ASCEND_PATCH_INSTALLED
    global _ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING
    global _ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING
    if _ASCEND_PATCH_INSTALLED:
        return True
    try:
        import vllm_ascend.worker.block_table as ascend_bt_mod  # type: ignore
    except Exception as exc:
        logger.debug(
            "[TriAttention-Ascend] vllm_ascend.worker.block_table not importable; "
            "v1 slot_mappings patch will be skipped (%s)",
            type(exc).__name__,
        )
        return False

    BlockTable = getattr(ascend_bt_mod, "BlockTable", None)
    MultiGroupBlockTable = getattr(ascend_bt_mod, "MultiGroupBlockTable", None)
    if BlockTable is None or MultiGroupBlockTable is None:
        return False

    if not getattr(BlockTable.compute_slot_mapping, "_triattention_patched", False):
        _ORIGINAL_BLOCKTABLE_COMPUTE_SLOT_MAPPING = BlockTable.compute_slot_mapping
        BlockTable.compute_slot_mapping = _patched_v1_blocktable_compute_slot_mapping  # type: ignore[assignment]
        _patched_v1_blocktable_compute_slot_mapping._triattention_patched = True  # type: ignore[attr-defined]
    if not getattr(
        MultiGroupBlockTable.compute_slot_mapping,
        "_triattention_patched",
        False,
    ):
        _ORIGINAL_MULTIGROUPBLOCKTABLE_COMPUTE_SLOT_MAPPING = (
            MultiGroupBlockTable.compute_slot_mapping
        )
        MultiGroupBlockTable.compute_slot_mapping = (  # type: ignore[assignment]
            _patched_v1_multigroupblocktable_compute_slot_mapping
        )
        _patched_v1_multigroupblocktable_compute_slot_mapping._triattention_patched = (  # type: ignore[attr-defined]
            True
        )
    _ASCEND_PATCH_INSTALLED = True
    logger.info(
        "[TriAttention-Ascend] installed v1 slot_mappings mask patch on "
        "vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping and "
        "MultiGroupBlockTable.compute_slot_mapping (post-reclaim "
        "num_blocks_per_row enforced via PAD_SLOT_ID)."
    )
    return True


def install_seq_len_override_patch() -> bool:
    """Ascend-side entry point. Idempotent.

    Replaces the previous no-op stub.  Returns True once the
    `vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping` and
    `MultiGroupBlockTable.compute_slot_mapping` masks are in place.
    The runner treats `False` as "use the default (un-patched) input
    prep path", which on Ascend is wrong but the call still completes
    without raising.
    """
    return _install_ascend_v1_patch()


# ---------------------------------------------------------------------------
# Second patch: wrap NPUModelRunner._update_states so that
# num_computed_tokens_cpu[req_idx] is overwritten with the post-reclaim
# length BEFORE the GPU-side seq_lens / positions are derived.  Without
# this, the attention kernel sees a seq_lens that grows unbounded with
# the logical decode position (i.e. the KV_BUDGET never actually shrinks
# the per-step attention work, and TPOT stays at the regressed 50-100ms
# level).
# ---------------------------------------------------------------------------


_ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES: Any | None = None
_NPU_UPDATE_STATES_PATCH_OK: bool | None = None


def _patched_npu_model_runner_update_states(self: Any, scheduler_output: Any) -> None:
    """Wrap NPUModelRunner._update_states.

    After the vLLM base class has written `num_computed_tokens_cpu[req_idx]`
    with the logical (pre-reclaim) value, we clamp it to the post-reclaim
    `current_cache_len` recorded by TriAttention's `RequestStateStore`.
    The TriAttention runner publishes the store on
    `self._triattention_state_store` (and the pending compression events
    on `self._triattention_pending_compression_events`) just before
    invoking this code path, so we can read both here.

    Effect: the next `_prepare_inputs` derives positions / seq_lens /
    slot_mapping from the **post-reclaim** length, so the Ascend
    attention kernel only does softmax over the surviving KV
    (matching the CUDA path's `prepare_pos_seq_lens` seq_lens-override
    semantics).
    """
    assert _ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES is not None
    _ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES(self, scheduler_output)

    state_store = getattr(self, "_triattention_state_store", None)
    pending_events = getattr(self, "_triattention_pending_compression_events", None)
    if state_store is None or not isinstance(pending_events, list) or not pending_events:
        return
    input_batch = getattr(self, "input_batch", None)
    if input_batch is None:
        return
    nct = getattr(input_batch, "num_computed_tokens_cpu", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(nct, np.ndarray) or not isinstance(req_id_to_index, dict):
        return
    clamped = 0
    for event in pending_events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if not isinstance(req_id, str):
            continue
        req_idx = req_id_to_index.get(req_id)
        if not isinstance(req_idx, int):
            continue
        tri_state = state_store.get(req_id)
        if tri_state is None:
            continue
        target = int(getattr(tri_state, "current_cache_len", 0) or 0)
        if target <= 0:
            continue
        original = int(nct[req_idx])
        if target < original:
            nct[req_idx] = target
            clamped += 1
    if clamped:
        logger.info(
            "[TriAttention-Ascend] clamped num_computed_tokens_cpu for %d "
            "compressed requests in _update_states (post-reclaim current_cache_len)",
            clamped,
        )


def _install_npu_update_states_patch() -> bool:
    global _ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES, _NPU_UPDATE_STATES_PATCH_OK
    if _NPU_UPDATE_STATES_PATCH_OK is True:
        return True
    try:
        import vllm_ascend.worker.model_runner_v1 as mr_mod  # type: ignore
    except Exception as exc:
        logger.debug(
            "[TriAttention-Ascend] vllm_ascend.worker.model_runner_v1 not "
            "importable; _update_states patch will be skipped (%s)",
            type(exc).__name__,
        )
        _NPU_UPDATE_STATES_PATCH_OK = False
        return False
    NPUModelRunner = getattr(mr_mod, "NPUModelRunner", None)
    if NPUModelRunner is None:
        _NPU_UPDATE_STATES_PATCH_OK = False
        return False
    if getattr(NPUModelRunner._update_states, "_triattention_patched", False):
        _ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES = NPUModelRunner._update_states
        _NPU_UPDATE_STATES_PATCH_OK = True
        return True
    _ORIGINAL_NPU_MODEL_RUNNER_UPDATE_STATES = NPUModelRunner._update_states
    NPUModelRunner._update_states = (  # type: ignore[assignment]
        _patched_npu_model_runner_update_states
    )
    _patched_npu_model_runner_update_states._triattention_patched = True  # type: ignore[attr-defined]
    _NPU_UPDATE_STATES_PATCH_OK = True
    logger.info(
        "[TriAttention-Ascend] installed _update_states clamp patch on "
        "NPUModelRunner (post-reclaim num_computed_tokens_cpu enforced so "
        "positions / seq_lens / slot_mapping all reflect the surviving KV "
        "length)."
    )
    return True


def install_seq_len_override_patch() -> bool:  # noqa: F811
    """Ascend-side entry point. Idempotent.

    Replaces the previous no-op stub.  Returns True once the
    `vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping` and
    `MultiGroupBlockTable.compute_slot_mapping` masks are in place, and
    the NPUModelRunner._update_states post-reclaim clamp is in place.
    The runner treats `False` as "use the default (un-patched) input prep
    path", which on Ascend is wrong but the call still completes
    without raising.
    """
    ok_v1 = _install_ascend_v1_patch()
    ok_update = _install_npu_update_states_patch()
    return bool(ok_v1 and ok_update)
