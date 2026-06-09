"""TriAttention Ascend-side worker mixin.

Re-exports the lazy runner proxy install used by the CUDA path and
pins it to the Ascend-side base classes:

- `base_runner` is an `NPUModelRunner` (not a `GPUModelRunner`).
  The lazy install does not need to know this — `TriAttentionModelRunner`
  only ever reads `base_runner.execute_model(...)` and
  `base_runner.triattention_apply_compression(...)`. The model
  loader is opaque to the proxy.

- The `_ensure_triattention_runner_proxy` method is bound onto
  `NPUWorker` as a class-level function (not a bound method) by
  the AIM integration monkeypatch, so it must be a staticmethod
  here.
"""

from __future__ import annotations

import os
from pathlib import Path

from vllm.logger import init_logger

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.hook_impl import install_runner_compression_hook
from triattention.vllm.runtime.runner import TriAttentionModelRunner

logger = init_logger(__name__)


def _debug_early_install_proxy_enabled() -> bool:
    return os.environ.get("TRIATTN_DEBUG_EARLY_INSTALL_PROXY", "0") == "1"


def _maybe_backfill_model_path(worker, config: TriAttentionRuntimeConfig) -> None:
    if config.model_path is not None:
        return
    model_config = getattr(worker, "model_config", None)
    model_path = getattr(model_config, "model", None)
    if isinstance(model_path, str) and model_path.strip():
        config.model_path = Path(model_path.strip())


class TriAttentionAscendWorker:
    """Mixin-style surface used by the AIM integration monkeypatch.

    Methods here are bound as classmethods / staticmethods onto
    `NPUWorker` so the patched `NPUWorker.execute_model` can call
    `self._ensure_triattention_runner_proxy()` exactly as the CUDA
    path does. The actual proxy is the same `TriAttentionModelRunner`
    the CUDA path uses, so the algorithm runs from one source of
    truth.
    """

    @staticmethod
    def _ensure_triattention_runner_proxy(worker) -> None:
        """Lazy-install `TriAttentionModelRunner` on the given NPUWorker.

        Idempotent: once installed, subsequent calls are a no-op.

        Why lazy: vllm-ascend's `NPUModelRunner.__init__` performs ACL
        graph compilation, device warmup and weight dispatch. Wrapping
        the runner proxy too early would intercept all of those and
        break the warmup. The proxy is only needed once a real
        TriAttention signal arrives on the scheduler_output, so we
        wait for that.
        """
        # --- INSTRUMENTATION: every entry to this function ---
        import os as _os_p
        _p_instr = _os_p.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1"
        if _p_instr:
            try:
                _already = bool(
                    getattr(worker, "_triattention_runner_proxy_installed", False)
                )
                _runner_class = type(
                    getattr(worker, "model_runner", None)
                ).__name__
                _has_stats = bool(
                    getattr(
                        getattr(worker, "_triattention_runtime_config", None),
                        "sparse_stats_path",
                        None,
                    )
                )
                logger.info(
                    "[TRITN-INSTR] W:ensure_proxy_entry already_installed=%s "
                    "current_runner=%s has_stats_path=%s",
                    _already, _runner_class, _has_stats,
                )
            except Exception:
                pass
        if getattr(worker, "_triattention_runner_proxy_installed", False):
            return
        if isinstance(getattr(worker, "model_runner", None), TriAttentionModelRunner):
            worker._triattention_runner_proxy_installed = True
            return
        config = getattr(worker, "_triattention_runtime_config", None)
        if config is None:
            config = TriAttentionRuntimeConfig.from_env()
            worker._triattention_runtime_config = config
        _maybe_backfill_model_path(worker, config)
        base_runner = worker.model_runner
        # --- INSTRUMENTATION: log the hook install attempt and any
        # exception that it raises. install_runner_compression_hook
        # raises RuntimeError if the stats file is missing or invalid.
        if _p_instr:
            try:
                logger.info(
                    "[TRITN-INSTR] W:before_install_hook runner_class=%s "
                    "stats_path=%s",
                    type(base_runner).__name__,
                    str(getattr(config, "sparse_stats_path", None)),
                )
            except Exception:
                pass
        try:
            install_runner_compression_hook(base_runner=base_runner, config=config)
        except Exception as _exc:
            if _p_instr:
                logger.info(
                    "[TRITN-INSTR] W:install_hook_FAILED exc=%s msg=%s "
                    "(this is why proxy never installs, check stats file)",
                    type(_exc).__name__, str(_exc)[:300],
                )
            raise
        worker.model_runner = TriAttentionModelRunner(
            base_runner=base_runner,
            config=config,
        )
        worker._triattention_runner_proxy_installed = True
        logger.info(
            "[TriAttention-Ascend] lazily installed runner proxy: budget=%d "
            "divide_length=%d stats_path=%s model_path=%s protect_prefill=%s "
            "window_size=%s",
            config.kv_budget,
            config.divide_length,
            str(config.sparse_stats_path) if config.sparse_stats_path is not None else None,
            str(config.model_path) if config.model_path is not None else None,
            config.protect_prefill,
            config.window_size,
        )
