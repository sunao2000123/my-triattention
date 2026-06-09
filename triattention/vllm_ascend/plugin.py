"""vLLM-Ascend plugin entrypoint for TriAttention runtime (V2) integration.

This is the dedicated Ascend-side entry point. It is registered under the
`vllm.general_plugins` group with the name `triattention_ascend` (see
setup.py).

Why a separate entry point:
- The CUDA path (`triattention.vllm.plugin:register_triattention_backend`)
  only patches `vllm.v1.worker.gpu_worker.Worker` and vLLM upstream
  `Scheduler`. On Ascend, the actual worker class is
  `vllm_ascend.worker.worker.NPUWorker` and the actual scheduler is a
  `BalanceScheduler` subclass of upstream `Scheduler` that is rebound
  onto the module by `vllm_ascend.patch.platform.patch_balance_schedule`
  (see `vllm_ascend/patch/platform/patch_balance_schedule.py:705`).
- A second entry point lets us run the Ascend-specific monkeypatcher
  (AIM = Ascend Integration Monkeypatch) without forking the CUDA one.
  Both entry points are safe to load: the CUDA plugin detects Ascend
  and bails out early; the Ascend plugin only activates on Ascend.

Compliance with the four core engineering principles:
- Minimal intrusion: never mutates vllm / vllm-ascend source files; uses
  `vllm.plugins` as the single discovery entry point and only `setattr`s
  on already-imported class objects at runtime.
- Signal driven: cross-process cross-object state rides on
  `setattr(scheduler_output, "triattention_*")` / `setattr(output, "triattention_*")`.
- Lazy loading: `TriAttentionModelRunner` proxy is only attached to
  `NPUWorker.model_runner` the first time a `triattention_signals` payload
  is observed on the wire.
- Explicit state sync: physical KV block reclaim is performed by calling
  `block_pool.free_blocks(removed_blocks)` directly; prefix-cache
  metadata is best-effort evicted on the same blocks before release.
"""

from __future__ import annotations

import logging
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _is_running_on_ascend() -> bool:
    """Detect vllm-ascend environment without importing vllm_ascend eagerly.

    The platform code path is the most reliable signal: if the current
    platform resolves to an NPU/Ascend platform, the import will succeed
    and `is_ascend` will be True. We also fall back to scanning the
    loaded modules for the vllm_ascend package.
    """
    try:
        from vllm.platforms import current_platform  # type: ignore

        platform_name = type(current_platform).__name__
        if "Ascend" in platform_name or "NPU" in platform_name:
            return True
    except Exception:
        pass
    # Fallback: vllm_ascend is importable only in an Ascend environment.
    try:
        import vllm_ascend  # noqa: F401

        return True
    except Exception:
        return False


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

    # --- NPU SAFETY OVERRIDES ---
    # On vllm-ascend v0.18.0:
    #   1. The global per-head selector path inside
    #      ``selector_hf._select_keep_indices_for_group_per_head`` calls
    #      ``compute_scores_triton`` whose Triton kernel uses
    #      ``tl.static_range(num_offsets)``. The triton-ascend backend
    #      (CANN) does not fully support ``num_offsets > 1`` in
    #      ``tl.static_range`` and emits ``TypeError`` at JIT compile
    #      time. The selector_hf path now has a transparent
    #      Triton->PyTorch fallback, but the cleanest safety net on
    #      Ascend is to default the pruning mode to ``per_layer``
    #      (single-pass, no static_range) unless the user explicitly
    #      opts into per_head semantics.
    #
    #   2. The kv_cache layout on NPU is 4D
    #      ``[num_blocks, block_size, H, D]`` whereas CUDA is 5D with a
    #      K/V split axis. The selector helper has been patched to
    #      accept both layouts, but a fresh deploy with default
    #      ``per_head`` + ``hf_aligned_global_per_head`` is the most
    #      common source of the historic
    #      ``TRIATTN_FATAL_TRITON_SCORING_REQUIRED:...:global_per_head:TypeError``
    #      failure. We default to ``per_layer`` so first-time deploys
    #      do not silently trip this.
    #
    # Users may opt back into the legacy per_head behavior by
    # explicitly exporting either of:
    #   TRIATTN_ASCEND_FORCE_PER_LAYER=0
    #   TRIATTN_RUNTIME_PRUNING_MODE=per_head
    _force_per_layer = _truthy(
        os.environ.get("TRIATTN_ASCEND_FORCE_PER_LAYER"),
        default=True,
    )
    _explicit_user_pruning = (
        os.environ.get("TRIATTN_RUNTIME_PRUNING_MODE")
        or os.environ.get("TRIATTENTION_PRUNING_MODE")
    )
    if (
        _force_per_layer
        and not _explicit_user_pruning
    ):
        os.environ["TRIATTN_RUNTIME_PRUNING_MODE"] = "per_layer"
        os.environ["TRIATTN_RUNTIME_ALLOW_PER_LAYER_MODE"] = "true"
        os.environ.setdefault(
            "TRIATTN_ASCEND_DEFAULT_PRUNING_MODE_OVERRIDE",
            "per_layer",
        )


def register_triattention_backend():
    """TriAttention Ascend-side plugin entry point.

    Loaded by `vllm.plugins.load_general_plugins()` because we register
    `triattention_ascend = triattention.vllm_ascend.plugin:register_triattention_backend`
    in setup.py under the `vllm.general_plugins` group.
    """
    if not _truthy(os.environ.get("ENABLE_TRIATTENTION"), default=True):
        logger.info(
            "[TriAttention-Ascend] ENABLE_TRIATTENTION is false; skipping registration."
        )
        return

    if not _is_running_on_ascend():
        logger.info(
            "[TriAttention-Ascend] Ascend platform not detected; "
            "letting the CUDA plugin (triattention) handle it. "
            "If you intended Ascend, ensure `vllm_ascend` is installed "
            "and that `vllm.platforms.current_platform` resolves to an NPU/Ascend class."
        )
        return

    quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"

    _bridge_legacy_env_to_runtime()

    patch_scheduler = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_SCHEDULER"),
        default=True,
    )
    patch_worker = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_WORKER"),
        default=True,
    )

    logger.info(
        "[TriAttention-Ascend] plugin entry point invoked: "
        "reason=load_general_plugins ascend_detected=True "
        "patch_scheduler=%s patch_worker=%s",
        patch_scheduler,
        patch_worker,
    )

    try:
        from triattention.vllm_ascend.runtime.integration_monkeypatch import (
            ensure_patches_installed,
        )

        status = ensure_patches_installed(
            patch_scheduler=patch_scheduler,
            patch_worker=patch_worker,
            reason="load_general_plugins",
        )
        if not quiet:
            logger.info(
                "[TriAttention-Ascend] Runtime (V2) plugin activated: "
                "patch_scheduler=%s patch_worker=%s status=%s",
                patch_scheduler,
                patch_worker,
                status,
            )
    except Exception as exc:
        logger.error(
            "[TriAttention-Ascend] plugin activation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        raise
