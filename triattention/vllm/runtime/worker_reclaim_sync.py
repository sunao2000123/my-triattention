"""Worker-side block-table reclaim synchronization helpers for TriAttention runtime.

CRITICAL CORRECTION (post-mortem of an earlier regression):

An earlier version of this module additionally **zeroed the trailing
block-id slots in the row** when truncating `num_blocks_per_row`.  This
turned out to be **catastrophic** on the vLLM-Ascend 0.18.0 V1 path:

- vLLM-Ascend 0.18.0's hot path is the V1 numpy-based
  `vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping` (NOT
  the V2 Triton kernel in `vllm_ascend/worker/v2/block_table.py` —
  `AscendBlockTables` is only used by the formal V2 model state path
  that is not exercised by `NPUInputBatch`).
- The V1 path indexes `block_table.np.ravel()[block_table_indices]`
  unconditionally — there is **no `num_blocks_per_row` mask** anywhere
  in the call.  After a TriAttention compaction, a request whose
  absolute decode position exceeds the post-reclaim block count reads
  recycled block ids and the attention kernel computes over foreign
  KV data.
- Zeroing the row tail changed "read a recycled block_id" into "read
  block_id = 0" — **block 0 is the system-reserved block whose slot 0
  is the FIRST request's prefix-cache slot**, so the affected token
  was now reading ANOTHER request's actual prefix KV.  This turned the
  18 % accuracy loss into a 6 % accuracy loss (worse) because the
  foreign KV it picked up was *consistent enough* to bias the logits
  away from the correct answer.

The correct fix is therefore NOT to zero the row tail, but to wrap
`compute_slot_mapping` (and `seq_lens` assembly) on the V1 path so the
out-of-bounds tokens are mapped to `PAD_SLOT_ID` instead of an
arbitrary block id.  That is implemented in
`triattention/vllm_ascend/runtime/gpu_seq_len_patch.py` (v2 of the
patch targets the actual hot path: the
`vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping` numpy
implementation).  This module only does the SAFE subset:

  1. Truncate `num_blocks_per_row[req_index]` to the post-reclaim block
     count (so `append_row()` doesn't overflow the cap on the next
     prefill chunk).
  2. Truncate `req_state.block_ids` (the CPU-side per-request block-id
     list) to the same count, so `_apply_compression_events` on the
     next round sees the truncated list as the truth.

We deliberately do **not** mutate `block_table.np` here.
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
        # tables via the scheduler-side `_apply_compression_events`
        # which mutates `req_to_blocks[req_id]` and calls
        # `block_pool.free_blocks`.  No further worker-side truncation
        # is needed here.  We deliberately do NOT zero the row tail —
        # doing so on the vLLM-Ascend 0.18.0 path made things worse
        # because `block_id = 0` is a real (system-reserved) block
        # whose slot 0 belongs to a different request's prefix KV.
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
                num_blocks_per_row[req_index] = required_blocks
                logger.info(
                    "TriAttention worker reclaim: req=%s num_blocks %d -> %d "
                    "(cache_len_after=%d block_size=%d)",
                    req_id, current, required_blocks, cache_len_after, block_size,
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
