"""Base-runner compression hook implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

# === BUG-RECL 总开关辅助函数（断点排查专用） ===
# 设 TRIATTN_BUG_RECL_DEBUG=1 启用；默认 0 关闭
def _bug_recl_enabled() -> bool:
    return os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
# === BUG-RECL 总开关辅助函数结束 ===

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import (
    build_keep_token_indices,
    compact_request_kv_in_place,  # compatibility for tests monkeypatching hook_impl symbol
    compact_request_kv_in_place_per_head,  # compatibility for tests monkeypatching
    gather_request_k_dense,  # compatibility for tests importing hook_impl symbol
)
from .hook_runtime_context import build_hook_runtime_context
from .hook_group_pipeline import (
    finalize_hook_placement_result,
    run_group_compaction_pipeline,
    try_build_recency_tail_block_remap,
)
from .hook_preflight import resolve_hook_compaction_inputs, resolve_hook_request_context
from .fast_recency_guard import (
    should_guard_fast_recency_long_context,
    uses_pure_fast_recency,
)
from .kv_group_resolver import resolve_group_tensors as _resolve_group_tensors
from .selector_hf import build_triattention_selector as _build_triattention_selector_impl
from .signals import CompressionSignal
from .thresholds import is_ascend_runtime

try:
    from vllm.logger import logger as _runtime_logger
except Exception:  # pragma: no cover - fallback for lightweight tests
    _runtime_logger = logging.getLogger(__name__)

# Selector implementation moved to triattention_runtime/selector_hf.py (D-017).

def make_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
) -> Callable[..., dict[str, Any]]:
    """Create a hook function bound to a concrete base runner."""
    # Route through extracted selector module. `_build_triattention_selector` symbol is
    # kept below for unit tests that monkeypatch the hook_impl-local name.
    try:
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_triattention_selector(config, base_runner=base_runner)
    except TypeError:
        # Backward-compatible path for unit tests that monkeypatch the selector
        # builder with the legacy single-arg signature.
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_triattention_selector(config)
    group_tensors_cache: dict[int, list[tuple[int, Any]]] | None = None
    compressed_once: set[str] = set()
    log_execution_path = bool(
        getattr(config, "logging_enabled", True)
        and getattr(config, "log_execution_path", True)
    )
    log_execution_path_core_only = bool(
        log_execution_path
        and getattr(config, "log_execution_path_core_only", False)
    )
    log_core_trace = bool(
        log_execution_path
        and getattr(config, "log_core_trace", False)
    )
    log_selector_debug = bool(
        log_execution_path
        and getattr(config, "log_selector_debug", False)
    )

    if log_execution_path:
        _runtime_logger.info(
            "TRIATTN_EXEC_PATH hook_installed selector_status=%s "
            "compaction=%s block_reclaim=%s sparse_stats=%s",
            selector_status,
            bool(getattr(config, "enable_experimental_kv_compaction", False)),
            bool(getattr(config, "enable_experimental_block_reclaim", False)),
            str(getattr(config, "sparse_stats_path", None)),
        )

    def _get_group_tensors() -> dict[int, list[tuple[int, Any]]]:
        nonlocal group_tensors_cache
        if group_tensors_cache is None:
            group_tensors_cache = _resolve_group_tensors(base_runner)
        return group_tensors_cache

    def _hook(req_id: str, signal: CompressionSignal, scheduler_output: Any) -> dict[str, Any]:
        setattr(base_runner, "_triattention_active_req_id", req_id)
        if log_execution_path:
            _runtime_logger.info(
                "TRIATTN_EXEC_PATH worker_hook_enter req=%s step=%d "
                "signal_reason=%s scheduled_tokens=%d estimated_cache_len=%d "
                "prefill_len=%d selector_status=%s",
                req_id,
                int(getattr(signal, "step", 0)),
                getattr(signal, "reason", None),
                int(getattr(signal, "scheduled_tokens", 1)),
                int(getattr(signal, "estimated_cache_len", 0)),
                int(getattr(signal, "prefill_len", 0)),
                selector_status,
            )
        # === BUG-RECL 断点 13（LE-H-001）：hook 入口观测 ===
        if _bug_recl_enabled():
            _under_budget = effective_tokens <= budget_total
            print(f"BUG-RECL [LE-H-001] hook-enter req={req_id} step={int(getattr(signal, 'step', 0))} selector_status={selector_status} select_keep_indices_present={select_keep_indices is not None} effective_tokens={int(effective_tokens)} budget_total={int(budget_total)} under_budget={_under_budget}", flush=True)
        # === BUG-RECL 断点 13 结束 ===
        strict_triton_required = bool(
            config.enable_experimental_kv_compaction and config.require_triton_scoring
        )
        req_ctx = resolve_hook_request_context(
            base_runner=base_runner,
            req_id=req_id,
            scheduler_output=scheduler_output,
        )
        if isinstance(req_ctx, dict):
            return req_ctx
        req_state = req_ctx.req_state
        req_runtime_state = req_ctx.req_runtime_state
        recent_unabsorbed_tokens: int | None = None
        cache_config_hint = getattr(base_runner, "cache_config", None)
        block_size_hint = int(getattr(cache_config_hint, "block_size", 0))
        if block_size_hint <= 0:
            block_size_hint = 1
        original_block_ids_by_group = getattr(req_state, "block_ids", None)
        runtime_ctx = build_hook_runtime_context(
            base_runner=base_runner,
            config=config,
            req_id=req_id,
            req_state=req_state,
            req_runtime_state=req_runtime_state,
            signal=signal,
            scheduler_output=scheduler_output,
            compressed_once=compressed_once,
            original_block_ids_by_group=original_block_ids_by_group,
            block_size_hint=block_size_hint,
        )
        num_computed_tokens = runtime_ctx.num_computed_tokens
        effective_tokens = runtime_ctx.effective_tokens
        budget_total = runtime_ctx.budget_total
        recent_unabsorbed_tokens = runtime_ctx.recent_unabsorbed_tokens
        should_defer_recompress = runtime_ctx.should_defer_recompress
        retained_token_padding = (
            0
            if bool(getattr(signal, "_post_forward", False))
            else int(runtime_ctx.scheduled_tokens)
        )
        if log_execution_path and not log_execution_path_core_only:
            _runtime_logger.info(
                "TRIATTN_EXEC_PATH worker_hook_runtime_context req=%s step=%d "
                "effective_tokens=%d budget_total=%d recent_unabsorbed=%s "
                "strict_triton_required=%s",
                req_id,
                int(getattr(signal, "step", 0)),
                int(effective_tokens),
                int(budget_total),
                recent_unabsorbed_tokens,
                strict_triton_required,
            )
        if should_defer_recompress:
            return {
                "applied": False,
                "reason": runtime_ctx.defer_reason or "defer_recompress",
                "cache_len_after": effective_tokens,
            }
        if effective_tokens <= budget_total:
            return {
                "applied": False,
                "reason": "under_budget",
                "cache_len_after": effective_tokens,
            }
        signal_prefill_len = int(getattr(signal, "prefill_len", 0) or 0)
        if should_guard_fast_recency_long_context(
            config=config,
            effective_tokens=effective_tokens,
            prefill_len=signal_prefill_len,
        ):
            return {
                "applied": False,
                "reason": "fast_recency_long_context_guard",
                "cache_len_after": effective_tokens,
                "effective_tokens": int(effective_tokens),
                "prefill_len": signal_prefill_len,
                "guard_tokens": int(
                    getattr(config, "fast_recency_long_context_guard_tokens", 0) or 0
                ),
            }

        if not config.enable_experimental_kv_compaction:
            keep_indices_plan = build_keep_token_indices(
                total_tokens=effective_tokens,
                kv_budget=config.kv_budget,
                prefill_len=signal.prefill_len,
                protect_prefill=signal.protect_prefill,
                include_prefill_in_budget=config.include_prefill_in_budget,
            )
            if keep_indices_plan is None:
                return {"applied": False, "reason": "prefill_exceeds_budget"}
            return {
                "applied": False,
                "reason": "plan_only",
                "cache_len_after": len(keep_indices_plan),
            }
        if strict_triton_required and select_keep_indices is None:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:selector_unavailable:{selector_status}"
            )

        compaction_inputs = resolve_hook_compaction_inputs(
            base_runner=base_runner,
            original_block_ids_by_group=original_block_ids_by_group,
        )
        if isinstance(compaction_inputs, dict):
            return compaction_inputs
        block_size = compaction_inputs.block_size
        mutable_block_ids_by_group = compaction_inputs.mutable_block_ids_by_group

        zero_copy_outcome = try_build_recency_tail_block_remap(
            config=config,
            mutable_block_ids_by_group=mutable_block_ids_by_group,
            effective_tokens=effective_tokens,
            budget_total=budget_total,
            block_size=block_size,
        )
        if zero_copy_outcome is not None:
            if log_execution_path:
                _runtime_logger.info(
                    "TRIATTN_EXEC_PATH zero_copy_tail_enter req=%s step=%d "
                    "effective_tokens=%d budget_total=%d block_size=%d",
                    req_id,
                    int(getattr(signal, "step", 0)),
                    int(effective_tokens),
                    int(budget_total),
                    int(block_size),
                )
            compressed_once.add(req_id)
            return finalize_hook_placement_result(
                req_state=req_state,
                original_block_ids_by_group=original_block_ids_by_group,
                config=config,
                selector_status=str(selector_status),
                outcome=zero_copy_outcome,
                effective_tokens=effective_tokens,
                budget_total=budget_total,
                recent_unabsorbed_tokens=recent_unabsorbed_tokens,
            )
        if (
            uses_pure_fast_recency(config)
            and bool(getattr(config, "enable_zero_copy_recency", True))
            and bool(getattr(config, "zero_copy_recency_only_on_ascend", True))
            and is_ascend_runtime(base_runner)
        ):
            return {
                "applied": False,
                "reason": "zero_copy_recency_not_ready",
                "cache_len_after": effective_tokens,
            }

        group_tensors = _get_group_tensors()
        if log_execution_path:
            layer_count = sum(len(v) for v in group_tensors.values())
            _runtime_logger.info(
                "TRIATTN_EXEC_PATH group_pipeline_enter req=%s step=%d "
                "groups=%d layers=%d block_size=%d effective_tokens=%d "
                "budget_total=%d selector_status=%s",
                req_id,
                int(getattr(signal, "step", 0)),
                len(group_tensors),
                layer_count,
                int(block_size),
                int(effective_tokens),
                int(budget_total),
                selector_status,
            )
            setattr(
                base_runner,
                "_triattention_active_signal_step",
                int(getattr(signal, "step", 0)),
            )
            if log_core_trace:
                _runtime_logger.info(
                    "TRIATTN_CORE_TRACE enter run_group_compaction_pipeline req=%s "
                    "step=%d groups=%d layers=%d effective_tokens=%d budget_total=%d",
                    req_id,
                    int(getattr(signal, "step", 0)),
                    len(group_tensors),
                    layer_count,
                    int(effective_tokens),
                    int(budget_total),
                )
        pipeline_out = run_group_compaction_pipeline(
            req_id=req_id,
            signal=signal,
            config=config,
            strict_triton_required=strict_triton_required,
            num_computed_tokens=num_computed_tokens,
            effective_tokens=effective_tokens,
            budget_total=budget_total,
            block_size=block_size,
            retained_token_padding=retained_token_padding,
            mutable_block_ids_by_group=mutable_block_ids_by_group,
            group_tensors=group_tensors,
            select_keep_indices=select_keep_indices,
            select_keep_indices_for_group=select_keep_indices_for_group,
            shared_compact_fn=compact_request_kv_in_place,
            per_head_compact_fn=compact_request_kv_in_place_per_head,
            gather_dense_fn=gather_request_k_dense,
        )
        if log_core_trace:
            if isinstance(pipeline_out, dict):
                _runtime_logger.info(
                    "TRIATTN_CORE_TRACE exit run_group_compaction_pipeline "
                    "req=%s step=%d result_type=dict applied=%s reason=%s",
                    req_id,
                    int(getattr(signal, "step", 0)),
                    pipeline_out.get("applied"),
                    pipeline_out.get("reason"),
                )
            else:
                _runtime_logger.info(
                    "TRIATTN_CORE_TRACE exit run_group_compaction_pipeline "
                    "req=%s step=%d result_type=GroupPipelineOutcome "
                    "applied=True selection_mode=%s cache_len_after=%d",
                    req_id,
                    int(getattr(signal, "step", 0)),
                    pipeline_out.selection_mode,
                    int(pipeline_out.cache_len_after),
                )
        if isinstance(pipeline_out, dict):
            if log_execution_path:
                _runtime_logger.info(
                    "TRIATTN_EXEC_PATH group_pipeline_result req=%s step=%d "
                    "applied=False reason=%s",
                    req_id,
                    int(getattr(signal, "step", 0)),
                    pipeline_out.get("reason"),
                )
            return pipeline_out

        compressed_once.add(req_id)
        if log_execution_path:
            selector_debug = pipeline_out.selector_debug if log_selector_debug else None
            _runtime_logger.info(
                "TRIATTN_EXEC_PATH group_pipeline_result req=%s step=%d "
                "applied=True selection_mode=%s cache_len_after=%d selector_debug=%s",
                req_id,
                int(getattr(signal, "step", 0)),
                pipeline_out.selection_mode,
                int(pipeline_out.cache_len_after),
                selector_debug,
            )
        return finalize_hook_placement_result(
            req_state=req_state,
            original_block_ids_by_group=original_block_ids_by_group,
            config=config,
            selector_status=str(selector_status),
            outcome=pipeline_out,
            effective_tokens=effective_tokens,
            budget_total=budget_total,
            recent_unabsorbed_tokens=recent_unabsorbed_tokens,
            retained_cache_len=int(pipeline_out.cache_len_after)
            + max(0, int(retained_token_padding)),
        )

    return _hook


# Keep hook_impl-local symbol for backward compatibility with existing tests
# and monkeypatch call sites, while routing runtime behavior to the extracted
# selector implementation module.
_build_triattention_selector = _build_triattention_selector_impl


def install_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
) -> None:
    """Install default hook on the underlying base runner if missing."""
    if hasattr(base_runner, "triattention_apply_compression"):
        return
    setattr(
        base_runner,
        "triattention_apply_compression",
        make_runner_compression_hook(base_runner=base_runner, config=config),
    )
