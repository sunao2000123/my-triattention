"""TriAttention Ascend integration package.

This package is the dedicated entry point for vllm-ascend v0.18.0
deployments. The `triattention_ascend` entry point (registered in
setup.py under `vllm.general_plugins`) calls
`triattention.vllm_ascend.plugin:register_triattention_backend`.

The Ascend-side code reuses the platform-agnostic algorithm core in
`triattention.vllm.runtime.*` and only adds:

- `plugin.py` — entry point that loads the AIM.
- `runtime/integration_monkeypatch.py` — the AIM that mutates
  `Scheduler`, `KVCacheManager`, `NPUWorker`, `AscendBlockTables`,
  and `kv_cache_utils` via runtime `setattr`.
- `runtime/scheduler_ascend.py` — Ascend-specific scheduler
  helpers (chunk cap formula, block reclaim defensive guards).
- `runtime/worker_ascend.py` — lazy runner proxy install bound to
  `NPUWorker`.
- `runtime/gpu_seq_len_patch.py` — no-op stub (see module
  docstring for the rationale).
"""

from triattention.vllm_ascend.plugin import register_triattention_backend

__all__ = ["register_triattention_backend"]
