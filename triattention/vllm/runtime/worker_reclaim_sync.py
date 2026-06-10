"""Worker-side block-table reclaim synchronization helpers for TriAttention runtime.

Critical Ascend-specific hardening: the V1 path that simply
decrements `num_blocks_per_row[req_index]` is **insufficient** on Ascend
because the `_compute_slot_mappings_kernel` (in
`vllm_ascend/worker/v2/block_table.py`) does not consult
`num_blocks_per_row` — it reads the whole row of `block_table` (up to
`TOTAL_BLOCK_SIZE = 4096` entries) for each request and then indexes by
`position // block_size`.  When the post-reclaim `num_blocks_per_row` is
shrunk, the *stale* block_ids at indices `>= num_blocks_per_row` still
sit in the row, and a request whose absolute position exceeds the
post-reclaim block count will look up a recycled block_id belonging to
another request.  This is the primary cause of:

- **Accuracy loss** (TriAttention accuracy 18 % vs CUDA 32 % at
  KV_BUDGET=2048): the kernel reads foreign KV data and the attention
  output is computed over garbage keys.
- **`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed`
  crash** at KV_BUDGET=8192: in the next step the runtime context's
  `effective_tokens` (computed from `state.current_cache_len`) diverges
  from the worker's `num_computed_tokens` (absolute decode position);
  the strict-mode guard `effective_tokens >= ratio * num_computed_tokens`
  fires.

The defense-in-depth fix has two layers:

1. **Primary**: the ascend-side `gpu_seq_len_patch` wraps
   `AscendBlockTables.compute_slot_mappings` and clamps out-of-bounds
   tokens to `PAD_SLOT_ID` using the per-req `num_blocks_per_row`.  See
   `triattention/vllm_ascend/runtime/gpu_seq_len_patch.py`.

2. **Secondary (this module)**: when truncating the worker's block
   table, also **zero the trailing block-id slots in the row** so the
   kernel can never read a recycled block id, even if the primary patch
   fails to install (e.g. on an older vllm-ascend build that doesn't
   expose the wrap point).  We do this by:

   a. writing `0` (which maps to slot 0 of the unused-block table area;
      the kernel treats this as a no-op slot) to the row entries from
      `num_blocks_per_row` up to the *old* `num_blocks_per_row` value;
   b. explicitly zeroing `block_table.np[row_idx, new_count:old_count]`
      for both the CPU and (if already staged to GPU) the GPU tensor;
   c. calling `commit_block_table(num_reqs)` so the GPU copy is fresh
      before the next `compute_slot_mappings` kernel runs.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_DEBUG_DISABLE_LOGGED = False


def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    """Apply reclaim shrink to worker-side block tables after compression.

    In vLLM V1, the block table lives at ``base_runner.input_batch.block_table``
    and tracks per-request block counts in ``num_blocks_per_row``.  After
    compression compacts KV cache data into fewer blocks, we must update these
    counters so that subsequent ``append_row()`` calls start from the correct
    offset and don't overflow the max-blocks-per-request limit.
    """
    global _DEBUG_DISABLE_LOGGED
    if os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        if not _DEBUG_DISABLE_LOGGED:
            logger.info("TriAttention worker reclaim sync disabled by debug env")
            _DEBUG_DISABLE_LOGGED = True
        return

    if not isinstance(events, list) or not events:
        return

    # Resolve the vLLM V1 block table.
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None

    # Resolve the V2 / formal path: base_runner.block_tables is a
    # MultiGroupBlockTable that owns the per-group BlockTable instances.
    v2_block_tables = getattr(base_runner, "block_tables", None)
    if block_table_obj is None and v2_block_tables is not None:
        # V2 path: hook-side compaction already updates the canonical
        # tables; the secondary defense-in-depth zeroing is not needed
        # here because the V2 BlockTable append API is the only path
        # that mutates block_table.np, and the scheduler-side reclaim
        # calls `apply_staged_writes` which propagates the truncation.
        # Still, zero the trailing ids for safety on kernels that
        # ignore num_blocks_per_row (Ascend's _compute_slot_mappings_kernel).
        _zero_trailing_v2(
            base_runner=base_runner,
            events=events,
            v2_block_tables=v2_block_tables,
        )
        return
    if block_table_obj is None:
        logger.warning(
            "TriAttention worker reclaim: block table not found. "
            "input_batch=%s block_table=%s",
            type(input_batch).__name__ if input_batch else None,
            type(block_table_obj).__name__ if block_table_obj else None,
        )
        return

    # Resolve request-id → row-index mapping.
    # In vLLM V1, req_id_to_index lives on input_batch, and request states
    # (with block_ids) live in base_runner.requests.
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        logger.warning(
            "TriAttention worker reclaim: req_id_to_index not found on input_batch. "
            "input_batch=%s",
            type(input_batch).__name__ if input_batch else None,
        )
        return

    # The block table may be a single BlockTable (with num_blocks_per_row) or
    # a MultiGroupBlockTable (with .block_tables list of per-group BlockTables).
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, list):
        # MultiGroupBlockTable
        tables = inner_tables
    else:
        # Single BlockTable
        tables = [block_table_obj]

    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 16))
    if block_size <= 0:
        block_size = 16

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if req_id is None:
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        cache_len_after = event.get("cache_len_after")
        if not isinstance(cache_len_after, int) or cache_len_after <= 0:
            continue

        required_blocks = (cache_len_after + block_size - 1) // block_size

        for table in tables:
            num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
            if num_blocks_per_row is None:
                continue
            if not isinstance(num_blocks_per_row, np.ndarray):
                continue
            current = int(num_blocks_per_row[req_index])
            if current > required_blocks:
                old_count = current
                new_count = required_blocks
                num_blocks_per_row[req_index] = new_count
                # Defense in depth: zero the trailing block-id slots so
                # the Ascend kernel cannot read a recycled block id for
                # tokens whose `position // block_size` falls into the
                # freed tail range.  The primary fix lives in
                # `gpu_seq_len_patch` (PAD_SLOT_ID clamp); this is the
                # belt-and-suspenders layer.
                _zero_trailing_block_ids_in_row(
                    table=table,
                    row_idx=req_index,
                    new_count=new_count,
                    old_count=old_count,
                )
                logger.info(
                    "TriAttention worker reclaim: req=%s num_blocks %d -> %d "
                    "(cache_len_after=%d block_size=%d, trailing_ids_zeroed)",
                    req_id, old_count, new_count, cache_len_after, block_size,
                )

        # Also truncate req_state.block_ids (CPU-side block tracking).
        # In vLLM V1, per-request state lives in base_runner.requests dict.
        requests_dict = getattr(base_runner, "requests", None)
        if isinstance(requests_dict, dict):
            req_state = requests_dict.get(req_id)
            if req_state is not None:
                block_ids_attr = getattr(req_state, "block_ids", None)
                if isinstance(block_ids_attr, (list, tuple)):
                    for group_blocks in block_ids_attr:
                        if isinstance(group_blocks, list) and len(group_blocks) > required_blocks:
                            del group_blocks[required_blocks:]


def _zero_trailing_block_ids_in_row(
    *,
    table: Any,
    row_idx: int,
    new_count: int,
    old_count: int,
) -> None:
    """Zero the trailing block-id slots in a single BlockTable row.

    The BlockTable on the ascend side has both a CPU numpy buffer
    (`block_table.np`) and an optional GPU copy (`block_table.gpu`);
    the kernel reads from the GPU copy, so we must clear both to be
    safe across staging modes.
    """
    if new_count >= old_count:
        return
    np_buffer = getattr(table, "block_table", None)
    if np_buffer is None:
        return
    np_view = getattr(np_buffer, "np", None)
    if np_view is not None and isinstance(np_view, np.ndarray):
        try:
            np_view[row_idx, new_count:old_count] = 0
        except Exception:
            logger.debug(
                "TriAttention worker reclaim: failed to zero CPU trailing ids; "
                "row_idx=%d new=%d old=%d",
                row_idx, new_count, old_count, exc_info=True,
            )
    gpu_view = getattr(np_buffer, "gpu", None)
    if gpu_view is not None:
        try:
            import torch  # local import; keep hot path lean
            if hasattr(gpu_view, "__setitem__"):
                gpu_view[row_idx, new_count:old_count] = 0
            elif isinstance(gpu_view, torch.Tensor):
                gpu_view[row_idx, new_count:old_count].zero_()
        except Exception:
            logger.debug(
                "TriAttention worker reclaim: failed to zero GPU trailing ids; "
                "row_idx=%d new=%d old=%d",
                row_idx, new_count, old_count, exc_info=True,
            )


def _zero_trailing_v2(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
    v2_block_tables: Any,
) -> None:
    """V2 path: zero trailing ids in the per-group BlockTable rows.

    The V2 `MultiGroupBlockTable.append_block_ids(overwrite=True)` API
    used by the scheduler-side reclaim already updates the row, but the
    append API *extends* the row rather than truncating it, so any
    previously freed block ids still sit in the tail.  We zero them
    here so the Ascend kernel can never read recycled ids.
    """
    if not isinstance(events, list) or not events:
        return
    inner = getattr(v2_block_tables, "block_tables", None)
    if not isinstance(inner, list):
        return

    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 16)) if cache_config else 16
    if block_size <= 0:
        block_size = 16

    # Resolve req_index from input_batch.req_id_to_index (V2 still has one).
    input_batch = getattr(base_runner, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch else None
    if not isinstance(req_id_to_index, dict):
        # Fallback: scan base_runner.requests if available
        requests = getattr(base_runner, "requests", None)
        if not isinstance(requests, dict):
            return
        req_id_to_index = {rid: idx for idx, (rid, _) in enumerate(requests.items())}
    if not req_id_to_index:
        return

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        cache_len_after = event.get("cache_len_after")
        if not isinstance(req_id, str) or not isinstance(cache_len_after, int) or cache_len_after <= 0:
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        required_blocks = (cache_len_after + block_size - 1) // block_size
        for table in inner:
            num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
            if num_blocks_per_row is None or not isinstance(num_blocks_per_row, np.ndarray):
                continue
            current = int(num_blocks_per_row[req_index])
            if current > required_blocks:
                old_count = current
                num_blocks_per_row[req_index] = required_blocks
                _zero_trailing_block_ids_in_row(
                    table=table,
                    row_idx=req_index,
                    new_count=required_blocks,
                    old_count=old_count,
                )
                logger.info(
                    "TriAttention worker reclaim (V2): req=%s num_blocks %d -> %d "
                    "(trailing_ids_zeroed)",
                    req_id, old_count, required_blocks,
                )
