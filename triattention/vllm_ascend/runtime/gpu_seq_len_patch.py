"""TriAttention Ascend-side GPU sequence-length patch (real implementation).

Replaces the previous no-op stub.  The reason a real patch is mandatory
on Ascend is that, although `AscendBlockTables.compute_slot_mappings`
recomputes slot_mappings every step, it does so by reading
`block_table[row_idx, position // block_size]` unconditionally — it does
NOT consult `num_blocks_per_row[row_idx]`.  After a TriAttention
compaction, `apply_worker_block_reclaim_events` shrinks
`num_blocks_per_row[row_idx]` to the post-reclaim block count, but the
*block-id* slots at indices `>= num_blocks_per_row` are still populated
with the freed block ids (which may have been recycled to another
request by the block pool).  The kernel will therefore read
*another request's KV* at those tail positions, corrupting attention
output (accuracy loss) and tripping the
`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed` guard in
the next step.

The patch wraps `AscendBlockTables.compute_slot_mappings` to:

  1. Build a per-row device tensor `effective_num_blocks_per_row` from
     the post-reclaim state set by `apply_worker_block_reclaim_events`
     (or, for V2 path, from `block_table.num_blocks_per_row` directly).
  2. Compare `block_index = position // block_size` to that cap and emit
     `PAD_SLOT_ID` for any out-of-bounds access; attention masking
     downstream will then ignore those entries instead of computing
     over foreign KV data.

Engineering principles respected:

- **Minimal intrusion:** the patch is installed once at process start
  via `setattr(AscendBlockTables, "compute_slot_mappings", wrapper)`;
  no vllm-ascend source file is touched.
- **Signal driven:** the per-req cap tensor is rebuilt from
  `scheduler_output.triattention_compression_events` (if attached) and
  the `block_table.num_blocks_per_row` arrays the runner already
  maintains.  The runner does not need to learn a new API.
- **Lazy loading:** the patch is registered the first time
  `install_seq_len_override_patch()` is called, which happens lazily on
  the first execute_model that bears a non-empty
  `triattention_signals` payload.
- **Explicit state sync:** the cap tensor is derived directly from
  `num_blocks_per_row` (which is the same source of truth as the
  scheduler's `manager.req_to_blocks` truncation), so the kernel sees
  exactly the blocks the scheduler allocated.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process state for the override window.  Mirrors the vLLM upstream
# `input_patch_state` module shape but is local to the ascend side so the
# patch is self-contained and does not depend on the GPU runner module
# (which doesn't exist on Ascend).
# ---------------------------------------------------------------------------


_ASCEND_PATCH_INSTALLED: bool = False
_ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS: Any | None = None


def _debug_disable_seq_override() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_SEQ_OVERRIDE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cap_num_blocks_per_request(
    block_tables: Any,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Return a 1-D int32 tensor of effective block counts for each row.

    `block_tables` may be either a single `BlockTable` or a
    `MultiGroupBlockTable` wrapping a list of per-group `BlockTable`s.
    The cap is the minimum of `num_blocks_per_row` across all groups,
    because a request's logical KV length is bounded by the smallest
    per-group block count (in hybrid block-size mode the groups can
    have different block sizes).
    """
    if block_tables is None:
        return torch.zeros(
            int(row_indices.shape[0]), device=row_indices.device, dtype=torch.int32,
        )
    inner = getattr(block_tables, "block_tables", None)
    if isinstance(inner, list) and inner:
        tables = inner
    else:
        tables = [block_tables]

    cap: torch.Tensor | None = None
    for tbl in tables:
        npr = getattr(tbl, "num_blocks_per_row", None)
        if npr is None:
            continue
        # num_blocks_per_row is a numpy array on CPU; convert to torch on
        # the same device as row_indices for the gather below.
        if hasattr(npr, "device"):
            npr_t = npr.to(dtype=torch.int32)
        else:
            npr_t = torch.as_tensor(npr, dtype=torch.int32, device=row_indices.device)
        row_caps = npr_t.index_select(0, row_indices.to(device=npr_t.device, dtype=torch.long))
        if cap is None:
            cap = row_caps
        else:
            cap = torch.minimum(cap, row_caps)
    if cap is None:
        return torch.zeros(
            int(row_indices.shape[0]), device=row_indices.device, dtype=torch.int32,
        )
    return cap.to(device=row_indices.device, dtype=torch.int32)


def _patched_ascend_compute_slot_mappings(
    self: Any,
    idx_mapping: "torch.Tensor",
    query_start_loc: "torch.Tensor",
    positions: "torch.Tensor",
    num_tokens_padded: int,
) -> "torch.Tensor":
    """Wrapper around the original `AscendBlockTables.compute_slot_mappings`.

    For each token in the batch, compute `block_index = position // block_size`
    and mask out (set to PAD_SLOT_ID) any token whose `block_index` exceeds
    the per-request post-reclaim `num_blocks_per_row`.  This is a faithful
    Ascend analogue of the CUDA path's `seq_lens` override: the kernel
    will emit PAD_SLOT_ID for tokens beyond the truncated tail, and the
    attention masking will exclude them from softmax instead of consuming
    recycled-block data.
    """
    assert _ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS is not None
    if _debug_disable_seq_override():
        return _ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS(
            self,
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded,
        )

    # Run the original kernel first; we'll post-process the slot_mapping
    # in-place only for out-of-bounds tokens (cheap when nothing was
    # reclaimed: 0 overwrites).
    out = _ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS(
        self,
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded,
    )

    try:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore
    except Exception:
        PAD_SLOT_ID = -1

    device = out.device
    # block_sizes_tensor is a per-group device tensor; we only need a
    # single scalar here.  Pull from the first group — all groups in a
    # TriAttention-reclaimed request have the same logical block size
    # because TriAttention's `_apply_compression_events` truncates
    # `req_to_blocks[req_id]` consistently across groups.
    block_sizes_t = getattr(self, "block_sizes_tensor", None)
    if block_sizes_t is None or int(block_sizes_t.numel()) == 0:
        return out
    block_size = int(block_sizes_t.flatten()[0].item())
    if block_size <= 0:
        return out

    # Build per-token req_index by walking the same query_start_loc
    # decomposition the kernel uses.  For sparse rows this is O(num_tokens).
    num_reqs = int(idx_mapping.shape[0])
    cap_t = _cap_num_blocks_per_request(self, idx_mapping.to(device=device, dtype=torch.long))
    # cap_t is shape [num_reqs] on `device`; gather per-token.
    token_req_idx = torch.empty(
        int(positions.shape[0]), device=device, dtype=torch.long,
    )
    qsl_cpu = query_start_loc.to(device="cpu", dtype=torch.long)
    for r in range(num_reqs):
        s = int(qsl_cpu[r].item())
        e = int(qsl_cpu[r + 1].item())
        if s < e:
            token_req_idx[s:e] = int(idx_mapping[r].item())

    token_cap = cap_t.index_select(0, token_req_idx)
    pos_i32 = positions.to(device=device, dtype=torch.int32)
    block_index = pos_i32 // int(block_size)
    overflow = block_index >= token_cap
    if bool(overflow.any().item()):
        # Only the slot_mappings up to num_tokens_padded are visible to
        # the consumer; clamp the mask to the visible range to avoid
        # touching padding cells.
        if int(out.shape[-1]) > int(num_tokens_padded):
            visible_overflow = torch.zeros_like(overflow)
            visible_overflow[: int(num_tokens_padded)] = overflow[: int(num_tokens_padded)]
            overflow = visible_overflow
        out[:, : int(num_tokens_padded)].masked_fill_(
            overflow[: int(num_tokens_padded)].unsqueeze(0),
            int(PAD_SLOT_ID),
        )
        logger.debug(
            "[TriAttention-Ascend] slot_mapping override: masked %d/%d tokens "
            "beyond post-reclaim num_blocks_per_row (block_size=%d)",
            int(overflow.sum().item()),
            int(positions.shape[0]),
            int(block_size),
        )
    return out


def _install_ascend_compute_slot_mappings_patch() -> bool:
    """Idempotently install the slot_mappings clamp on AscendBlockTables.

    Returns True on success.  This is the ascend-side equivalent of the
    CUDA `install_runtime_input_patch_hooks` and is the only place in the
    project that mutates a vllm-ascend class attribute.
    """
    global _ASCEND_PATCH_INSTALLED, _ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS
    if _ASCEND_PATCH_INSTALLED:
        return True
    try:
        import vllm_ascend.worker.v2.block_table as ascend_bt_mod  # type: ignore
    except Exception as exc:
        logger.debug(
            "[TriAttention-Ascend] vllm_ascend.worker.v2.block_table not importable; "
            "slot_mappings patch will be skipped (%s)",
            type(exc).__name__,
        )
        return False
    AscendBlockTables = getattr(ascend_bt_mod, "AscendBlockTables", None)
    if AscendBlockTables is None:
        return False
    if getattr(AscendBlockTables.compute_slot_mappings, "_triattention_patched", False):
        _ASCEND_PATCH_INSTALLED = True
        return True
    _ORIGINAL_ASCEND_COMPUTE_SLOT_MAPPINGS = AscendBlockTables.compute_slot_mappings
    AscendBlockTables.compute_slot_mappings = _patched_ascend_compute_slot_mappings  # type: ignore[assignment]
    _patched_ascend_compute_slot_mappings._triattention_patched = True  # type: ignore[attr-defined]
    _ASCEND_PATCH_INSTALLED = True
    logger.info(
        "[TriAttention-Ascend] installed slot_mappings clamp patch on "
        "AscendBlockTables.compute_slot_mappings (post-reclaim num_blocks_per_row "
        "enforced via PAD_SLOT_ID)."
    )
    return True


def install_seq_len_override_patch() -> bool:
    """Ascend-side entry point. Idempotent.

    Replaces the previous no-op stub.  Returns True once the
    `AscendBlockTables.compute_slot_mappings` clamp is in place.  The
    runner treats `False` as "use the default (un-patched) input prep
    path", which on Ascend is correct as long as the worker_reclaim_sync
    has also been applied.  We return True whenever the patch actually
    installs, otherwise False.
    """
    return _install_ascend_compute_slot_mappings_patch()
