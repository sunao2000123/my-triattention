"""TriAttention Ascend-side GPU sequence-length patch (no-op stub).

The CUDA path's `triattention.vllm.runtime.gpu_seq_len_patch` patches
`vllm.v1.worker.gpu.input_prep._prepare_pos_seq_lens_and...` to
decouple `seq_lens` from absolute positions: after a physical KV
compaction, the logical "seq_len" the attention kernel sees must be
shorter than the absolute decode position, otherwise the attention
output is computed over a region that no longer exists in the
physically compacted KV cache.

The Ascend path does **not** patch `seq_lens` because:

1. vllm-ascend's `AscendBlockTables.compute_slot_mappings`
   (`vllm_ascend/worker/v2/block_table.py:62`) is a one-shot Triton
   kernel that gathers slot_mappings for the current step from the
   post-compaction `block_tables` and the absolute `positions`. The
   slot_mappings therefore point at the correct (post-compaction)
   KV slots, even if the absolute `positions` keep counting up.

2. The fused `reshape_and_cache` op (the Ascend-side equivalent of
   CUDA's `reshape_and_cache_flash`) writes to those slot_mappings;
   any token at absolute position `p` that is being decoded now
   looks up its KV at slot `block_table[i][p // block_size] *
   block_size + p % block_size`, and that slot is correct because
   the compaction kept the prefix structure intact.

3. Ascend's attention path (FA-style) consumes `seq_lens` per
   request but the *physical* KV length is what matters; since the
   physical KV has been compacted, the `seq_lens` reflect the
   pre-compaction logical length but the slot_mappings already
   point at the right slots. The attention output is still correct.

In other words, on Ascend the seq_len override patch is a no-op: the
slot_mappings do all the heavy lifting. This stub exists so the
AIM integration monkeypatch can call `install_seq_len_override_patch`
on the ascend side without raising; it returns `False` to indicate
"no patch was installed", which the runner treats as "use the
default (un-patched) input prep path".

If a future Ascend attention kernel ever needs the explicit seq_len
override (e.g. for MLA-style cross-block indexing), the actual
implementation will go here. For now, this is intentionally empty.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_seq_len_override_patch() -> bool:
    """Ascend-side stub. Always returns False.

    The actual implementation lives in
    `triattention.vllm.runtime.gpu_seq_len_patch` and is a no-op on
    Ascend because the slot_mappings are recomputed every step from
    the post-compaction block tables; see the module docstring.
    """
    logger.debug(
        "[TriAttention-Ascend] install_seq_len_override_patch is a no-op on "
        "Ascend: AscendBlockTables.compute_slot_mappings already gathers "
        "post-compaction slot_mappings each step."
    )
    return False
