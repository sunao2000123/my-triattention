"""TriAttention Ascend-side runtime re-exports.

This package re-exports the platform-agnostic algorithm core
(`triattention.vllm.runtime.*`) so callers (and the AIM integration
monkeypatch) can use the same identifiers regardless of the
underlying platform. The three files that contain Ascend-specific
behaviour are:

- `integration_monkeypatch.py` — the AIM, the only file that mutates
  third-party class attributes.
- `scheduler_ascend.py` — Ascend-specific scheduler mixin
  (BalanceScheduler adaptations).
- `worker_ascend.py` — Ascend-specific worker mixin (lazy runner
  proxy install on `NPUWorker`).
- `gpu_seq_len_patch.py` — no-op stub (see module docstring for
  the rationale).
"""

from triattention.vllm_ascend.runtime.integration_monkeypatch import (
    ensure_patches_installed,
    install_ascend_integration_monkeypatches,
)
from triattention.vllm_ascend.runtime.scheduler_ascend import (
    TriAttentionAscendScheduler,
)
from triattention.vllm_ascend.runtime.worker_ascend import (
    TriAttentionAscendWorker,
)

__all__ = [
    "ensure_patches_installed",
    "install_ascend_integration_monkeypatches",
    "TriAttentionAscendScheduler",
    "TriAttentionAscendWorker",
]
