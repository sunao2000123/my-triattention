"""vLLM plugin entrypoint for TriAttention runtime (V2) integration.

Default behavior:
- Install runtime scheduler/worker monkeypatches for TriAttention V2 path.
- Bridge legacy `TRIATTENTION_*` env vars into `TRIATTN_RUNTIME_*` when needed.

Legacy V1 custom backend registration is retired.

On Ascend platforms this plugin is a no-op: the dedicated
`triattention.vllm_ascend.plugin:register_triattention_backend` entry
point (registered under the same `vllm.general_plugins` group as
`triattention_ascend`) takes over and patches the Ascend-side classes
(`NPUWorker`, `BalanceScheduler`, `AscendBlockTables`). The two entry
points never both patch the same class; the CUDA patcher only acts on
upstream `vllm.v1.worker.gpu_worker.Worker` and vLLM upstream
`Scheduler`, while the Ascend patcher acts on
`vllm_ascend.worker.worker.NPUWorker` and the same upstream `Scheduler`
class (inherited via MRO by `BalanceScheduler`).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _set_if_absent(target: str, source: str) -> None:
    if os.environ.get(target):
        return
    value = os.environ.get(source)
    if value is not None and value != "":
        os.environ[target] = value


def _bridge_legacy_env_to_runtime() -> None:
    # Core runtime controls.
    _set_if_absent("TRIATTN_RUNTIME_KV_BUDGET", "TRIATTENTION_KV_BUDGET")
    _set_if_absent("TRIATTN_RUNTIME_DIVIDE_LENGTH", "TRIATTENTION_DIVIDE_LENGTH")
    _set_if_absent("TRIATTN_RUNTIME_WINDOW_SIZE", "TRIATTENTION_WINDOW_SIZE")
    _set_if_absent("TRIATTN_RUNTIME_LOG_DECISIONS", "TRIATTENTION_LOG_DECISIONS")
    _set_if_absent("TRIATTN_RUNTIME_SPARSE_STATS_PATH", "TRIATTENTION_STATS_PATH")

    # Keep default runtime behavior strict enough for real compression runs.
    os.environ.setdefault("TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM", "true")

    pruning_mode = os.environ.get("TRIATTN_RUNTIME_PRUNING_MODE")
    if not pruning_mode:
        pruning_mode = os.environ.get("TRIATTENTION_PRUNING_MODE")
        if pruning_mode:
            mode = pruning_mode.strip().lower()
            if mode == "per_layer_head":
                mode = "per_layer_per_head"
            os.environ["TRIATTN_RUNTIME_PRUNING_MODE"] = mode


def _is_running_on_ascend() -> bool:
    """Detect vllm-ascend environment.

    The check is intentionally side-effect free: we never `import
    vllm_ascend` here (it would force the Ascend platform's heavy
    import chain in subprocesses that do not need it).
    """
    # 1. explicit env flag
    if os.environ.get("VLLM_ASCEND", "0") == "1":
        return True
    if os.environ.get("VLLM_USE_V1", "0") == "0" and os.environ.get(
        "VLLM_ASCEND_PLATFORM"
    ):
        return True
    # 2. scan sys.modules
    for name in list(sys.modules.keys()):
        if name == "vllm_ascend" or name.startswith("vllm_ascend."):
            return True
    return False


def register_triattention_backend():
    """Install TriAttention runtime integration when plugin is loaded by vLLM."""
    # Allow baseline mode: skip all integration when explicitly disabled.
    if not _truthy(os.environ.get("ENABLE_TRIATTENTION"), default=True):
        return

    # Lazy import: keep this top-level import free so cold-start cost is
    # only paid when the plugin is actually invoked.
    import sys

    quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
    interface_mode = os.environ.get("TRIATTENTION_INTERFACE", "runtime").strip().lower()

    if interface_mode in {"legacy", "legacy_custom", "v1", "custom"}:
        if not quiet:
            logger.info(
                "[TriAttention] Legacy V1 backend plugin registration is retired; "
                "use runtime interface (TRIATTENTION_INTERFACE=runtime)."
            )
        return

    # On Ascend we step aside: the `triattention_ascend` entry point owns
    # the patching of `NPUWorker` / `BalanceScheduler` / `AscendBlockTables`.
    # The CUDA patcher is still import-safe but its patches would land on
    # the wrong (unloaded) `vllm.v1.worker.gpu_worker.Worker` class.
    if _is_running_on_ascend():
        if not quiet:
            logger.info(
                "[TriAttention] Detected vllm-ascend platform; deferring to "
                "triattention_ascend entry point (NPUWorker/AscendBlockTables)."
            )
        return

    _bridge_legacy_env_to_runtime()

    patch_scheduler = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_SCHEDULER"),
        default=True,
    )
    patch_worker = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_WORKER"),
        default=True,
    )

    try:
        from triattention.vllm.runtime.integration_monkeypatch import (
            install_vllm_integration_monkeypatches,
        )

        install_vllm_integration_monkeypatches(
            patch_scheduler=patch_scheduler,
            patch_worker=patch_worker,
        )
        if not quiet:
            logger.info(
                "[TriAttention] Runtime (V2) plugin activated: "
                "patch_scheduler=%s patch_worker=%s",
                patch_scheduler, patch_worker,
            )
    except Exception as exc:  # pragma: no cover - safety guard
        if not quiet:
            logger.error("[TriAttention] Runtime plugin activation failed: %s: %s", type(exc).__name__, exc)
        raise
