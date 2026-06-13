"""TriAttention v2 model runner proxy."""

from __future__ import annotations

import os
import time
from typing import Any

from vllm.logger import logger

from .config import TriAttentionRuntimeConfig
from .executor import CompressionExecutor, RunnerHookCompressionExecutor
from .fast_recency_guard import should_guard_fast_recency_long_context
from .input_patch_backend import install_runtime_input_patch
from .prefill_phase import is_prefill_phase_for_limit
from .request_key_compat import get_scheduled_token_items
from .runner_compression_actions import execute_runner_compression_actions
from .runner_output_bridge import (
    attach_execute_model_compression_events,
    attach_sample_tokens_compression_events,
    execute_base_model_with_effective_overrides,
)
from .runner_state_updates import (
    _resolve_full_prefill_len_from_request_like,
    cleanup_finished_requests,
    consume_runner_signals,
    mark_preemptions,
    mark_resumed,
    register_new_requests,
)
from .perf_profile import TriAttentionPerfProfile
from .phase_profile import (
    make_timed_wrapper,
    phase_elapsed_ms,
    phase_now,
    phase_profile_enabled,
    record_phase,
    register_model_probes,
)
from .signals import CompressionSignal
from .state import RequestStateStore
from .thresholds import (
    compression_length_threshold,
    is_ascend_environment_available,
    is_ascend_runtime,
)
from .worker_reclaim_sync import apply_worker_block_reclaim_events


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_positive_int(value: Any) -> int | None:
    number = _safe_int(value)
    if number is None or number <= 0:
        return None
    return number


def _ceil_div_positive(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        return 0
    return (value + divisor - 1) // divisor


def _shape0(value: Any) -> int | None:
    try:
        return int(value.shape[0])
    except Exception:
        return None


def _numel(value: Any) -> int | None:
    try:
        return int(value.numel())
    except Exception:
        try:
            return int(value.size)
        except Exception:
            return None


def _first_tensor_like(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for item in value:
            if hasattr(item, "shape") or hasattr(item, "numel"):
                return item
    return value


def _resolve_tensor_parallel_rank(base_runner: Any) -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        pass
    for attr_name in ("tp_rank", "tensor_parallel_rank"):
        raw = getattr(base_runner, attr_name, None)
        if raw is not None:
            try:
                return int(raw)
            except Exception:
                pass
    for parallel_config in (
        getattr(base_runner, "parallel_config", None),
        getattr(getattr(base_runner, "vllm_config", None), "parallel_config", None),
    ):
        raw = getattr(parallel_config, "tensor_parallel_rank", None)
        if raw is not None:
            try:
                return int(raw)
            except Exception:
                pass
    return 0


def _module_forward_details(
    *,
    layer_idx: int,
    kind: str,
) -> Any:
    def _details(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        first_arg = args[0] if args else None
        hidden_states = _first_tensor_like(kwargs.get("hidden_states", first_arg))
        result_tensor = _first_tensor_like(result)
        return {
            "layer": layer_idx,
            "kind": kind,
            "input_rows": _shape0(hidden_states),
            "input_numel": _numel(hidden_states),
            "result_rows": _shape0(result_tensor),
            "result_numel": _numel(result_tensor),
        }

    return _details


def _as_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return [value[idx] for idx in range(len(value))]
    except Exception:
        return None


def _applied_compression_events_by_req_id(events: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if req_id is not None:
            out[req_id] = event
    return out


def _event_retained_cache_len(event: dict[str, Any] | None) -> int | None:
    if event is None:
        return None
    details = event.get("details")
    retained_cache_len = (
        details.get("retained_cache_len")
        if isinstance(details, dict)
        else None
    )
    retained_cache_len_i = _safe_positive_int(retained_cache_len)
    if retained_cache_len_i is not None:
        return retained_cache_len_i
    return _safe_positive_int(event.get("cache_len_after"))


def _block_table_inner_tables(block_table_obj: Any) -> list[Any]:
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, (list, tuple)) and inner_tables:
        return list(inner_tables)
    return [block_table_obj]


def _table_row_block_count(table: Any, req_index: int) -> int | None:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if num_blocks_per_row is None:
        return None
    try:
        return int(num_blocks_per_row[req_index])
    except Exception:
        return None


def _table_block_size(table: Any, fallback: int | None) -> int | None:
    for attr_name in ("block_size", "logical_block_size", "physical_block_size"):
        value = _safe_positive_int(getattr(table, attr_name, None))
        if value is not None:
            return value
    return fallback


def _table_max_blocks(table: Any, block_table_obj: Any) -> int | None:
    for owner in (table, block_table_obj):
        value = _safe_positive_int(getattr(owner, "max_num_blocks_per_req", None))
        if value is not None:
            return value
    return None


def _resolve_model_layers(model: Any) -> list[Any]:
    roots = [
        model,
        getattr(model, "model", None),
        getattr(model, "decoder", None),
        getattr(model, "transformer", None),
    ]
    for root in roots:
        if root is None:
            continue
        for attr_name in ("layers", "h", "blocks"):
            layers = _as_sequence(getattr(root, attr_name, None))
            if layers:
                return layers
    return []


def _parse_layer_probe_indices(raw: str, layer_count: int) -> set[int]:
    indices: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx = int(item)
        except Exception:
            continue
        if idx < 0:
            idx = layer_count + idx
        if 0 <= idx < layer_count:
            indices.add(idx)
    return indices


def _select_layer_probe_indices(layer_count: int) -> list[int]:
    if layer_count <= 0:
        return []
    explicit = os.environ.get("TRIATTN_RUNTIME_MODEL_PHASE_PROBE_LAYERS")
    if explicit:
        return sorted(_parse_layer_probe_indices(explicit, layer_count))
    limit = max(0, _env_int("TRIATTN_RUNTIME_MODEL_PHASE_PROBE_LAYER_LIMIT", 4))
    if limit <= 0:
        return []
    if layer_count <= limit:
        return list(range(layer_count))
    if limit == 1:
        return [0]
    selected = {
        int(round(i * (layer_count - 1) / (limit - 1)))
        for i in range(limit)
    }
    return sorted(idx for idx in selected if 0 <= idx < layer_count)


class TriAttentionModelRunner:
    """Proxy wrapper around vLLM model runner.

    Phase 1 behavior:
    - consume scheduler-side compression signals;
    - maintain request lifecycle-safe state;
    - keep vLLM forward path untouched.
    """

    def __init__(self, base_runner: Any, config: TriAttentionRuntimeConfig | None = None):
        self._base_runner = base_runner
        self.config = config or TriAttentionRuntimeConfig.from_env()
        self.state_store = RequestStateStore()
        # Expose request-level compression state to the installed hook so it can
        # apply recent-window semantics without relying on logical token order.
        setattr(base_runner, "_triattention_state_store", self.state_store)
        self.executor: CompressionExecutor = RunnerHookCompressionExecutor(base_runner)
        self._last_step = 0
        self._logger = logger
        self._perf = TriAttentionPerfProfile.from_env(self._logger)
        self._log_worker_events = (
            bool(self.config.logging_enabled)
            and (
                bool(self.config.log_all_worker_events)
                or _resolve_tensor_parallel_rank(base_runner) == 0
            )
        )
        self._pending_compression_events: list[dict[str, Any]] = []
        self._strict_no_downgrade = bool(self.config.enable_experimental_kv_compaction)
        self._log_execution_path = bool(
            getattr(self.config, "logging_enabled", True)
            and getattr(self.config, "log_execution_path", True)
        )
        self._log_execution_path_core_only = bool(
            self._log_execution_path
            and getattr(self.config, "log_execution_path_core_only", False)
        )
        self._logged_execution_path_trigger_guards: set[tuple[str, str]] = set()
        self._runtime_input_patch_installed = False
        if bool(getattr(self.config, "preinstall_input_patch", True)):
            self._runtime_input_patch_installed = bool(install_runtime_input_patch())
        self._install_base_runner_phase_probes()
        self._install_model_submodule_phase_probes()
        self._allowed_strict_skip_reasons = {
            "under_budget",
            "prefill_incomplete",
            "prefill_compression_limit",
            "defer_recompress",
            "prefill_exceeds_budget",
            "req_state_not_found",
            "batch_queue_dedup",
            "fast_recency_long_context_guard",
            "initial_decode_grace",
            "zero_copy_recency_not_ready",
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_runner, name)

    def _install_base_runner_phase_probes(self) -> None:
        """Attach fallback timing probes directly to the concrete runner instance."""

        if not phase_profile_enabled():
            return
        method_phases = {
            "_prepare_inputs": "base_runner_prepare_inputs",
            "prepare_inputs": "base_runner_prepare_inputs",
            "_build_attention_metadata": "base_runner_build_attention_metadata",
            "_model_forward": "base_runner_model_forward",
            "_preprocess": "base_runner_preprocess",
            "_determine_batch_execution_and_padding": "base_runner_determine_batch",
            "_sync_batch_across_dp": "base_runner_sync_batch_across_dp",
            "_sample": "base_runner_sample",
            "_bookkeeping_sync": "base_runner_bookkeeping_sync",
            "_update_states": "base_runner_update_states",
            "_update_states_after_model_execute": "base_runner_update_states_after_execute",
            "postprocess": "base_runner_postprocess",
        }
        installed: list[str] = []
        for method_name, phase_name in method_phases.items():
            original = getattr(self._base_runner, method_name, None)
            if not callable(original) or bool(
                getattr(original, "_triattention_phase_timed", False)
            ):
                continue
            try:
                setattr(
                    self._base_runner,
                    method_name,
                    make_timed_wrapper(phase_name, original),
                )
                installed.append(method_name)
            except Exception:
                continue
        model = getattr(self._base_runner, "model", None)
        compute_logits = getattr(model, "compute_logits", None)
        if callable(compute_logits) and not bool(
            getattr(compute_logits, "_triattention_phase_timed", False)
        ):
            try:
                setattr(
                    model,
                    "compute_logits",
                    make_timed_wrapper("base_model_compute_logits", compute_logits),
                )
                installed.append("model.compute_logits")
            except Exception:
                pass
        if (
            installed
            and bool(getattr(self._perf, "enabled", False))
            and self.config.logging_enabled
        ):
            self._logger.info(
                "TriAttention installed base runner phase probes: %s",
                ",".join(installed),
            )

    def _install_model_submodule_phase_probes(self) -> None:
        """Attach sampled model/layer probes for Ascend forward bottleneck analysis."""

        default_enabled = phase_profile_enabled()
        if not _env_bool("TRIATTN_RUNTIME_MODEL_PHASE_PROBES", default_enabled):
            return
        model = getattr(self._base_runner, "model", None)
        layers = _resolve_model_layers(model)
        selected_layers = _select_layer_probe_indices(len(layers))
        if not selected_layers:
            return

        installed: list[str] = []

        def _wrap_forward(target: Any, label: str, phase: str, layer_idx: int, kind: str) -> None:
            original = getattr(target, "forward", None)
            if not callable(original) or bool(
                getattr(original, "_triattention_phase_timed", False)
            ):
                return
            try:
                setattr(
                    target,
                    "forward",
                    make_timed_wrapper(
                        phase,
                        original,
                        _module_forward_details(layer_idx=layer_idx, kind=kind),
                    ),
                )
                installed.append(label)
            except Exception:
                return

        for layer_idx in selected_layers:
            layer = layers[layer_idx]
            _wrap_forward(
                layer,
                f"layer[{layer_idx}].forward",
                "model_layer_forward",
                layer_idx,
                "layer",
            )
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                self_attn = getattr(layer, "attention", None)
            if self_attn is not None:
                _wrap_forward(
                    self_attn,
                    f"layer[{layer_idx}].self_attn.forward",
                    "model_self_attn_forward",
                    layer_idx,
                    "self_attn",
                )
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                mlp = getattr(layer, "feed_forward", None)
            if mlp is not None:
                _wrap_forward(
                    mlp,
                    f"layer[{layer_idx}].mlp.forward",
                    "model_mlp_forward",
                    layer_idx,
                    "mlp",
                )

        if (
            installed
            and bool(getattr(self._perf, "enabled", False))
            and self.config.logging_enabled
        ):
            register_model_probes(installed)
            self._logger.info(
                "TriAttention installed model submodule phase probes: %s",
                ",".join(installed),
            )

    def _register_new_requests(self, scheduler_output: Any) -> None:
        register_new_requests(
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            protect_prefill=bool(self.config.protect_prefill),
        )

    def _cleanup_finished_requests(self, scheduler_output: Any) -> None:
        cleanup_finished_requests(state_store=self.state_store, scheduler_output=scheduler_output)

    def _mark_preemptions(self, scheduler_output: Any) -> None:
        mark_preemptions(state_store=self.state_store, scheduler_output=scheduler_output)

    def _mark_resumed(self, scheduler_output: Any) -> None:
        mark_resumed(state_store=self.state_store, scheduler_output=scheduler_output)

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        step, signals = consume_runner_signals(
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            last_step=self._last_step,
            logger=self._logger,
            log_decisions=bool(self.config.log_decisions),
        )
        self._last_step = step
        return signals

    def _get_actual_kv_from_block_table(self, req_id: str) -> int | None:
        """Return actual KV token count from Worker's block table.

        Uses ``num_blocks_per_row * block_size`` which reflects the true
        physical state, free of async Scheduler lag.  Returns *None* when
        the information is unavailable (e.g. request not yet in input_batch).
        """
        input_batch = getattr(self._base_runner, "input_batch", None)
        if input_batch is None:
            return None
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if not isinstance(req_id_to_index, dict):
            return None
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            return None
        block_table_obj = getattr(input_batch, "block_table", None)
        if block_table_obj is None:
            return None
        inner_tables = getattr(block_table_obj, "block_tables", None)
        first_table = (
            inner_tables[0]
            if isinstance(inner_tables, list) and inner_tables
            else block_table_obj
        )
        num_blocks_per_row = getattr(first_table, "num_blocks_per_row", None)
        if num_blocks_per_row is None:
            return None
        cache_config = getattr(self._base_runner, "cache_config", None)
        blk_size = int(getattr(cache_config, "block_size", 0))
        if blk_size <= 0:
            return None
        return int(num_blocks_per_row[req_index]) * blk_size

    def _compression_threshold(
        self,
        prefill_len: int,
        *,
        is_prefill_step: bool = False,
    ) -> int:
        cache_config = getattr(self._base_runner, "cache_config", None)
        block_size = int(getattr(cache_config, "block_size", 1) or 1)
        return compression_length_threshold(
            self.config,
            prefill_len=prefill_len,
            block_size=block_size,
            is_ascend=is_ascend_runtime(self._base_runner),
            is_prefill_step=is_prefill_step,
        )

    def _should_defer_chunked_prefill_compression(self) -> bool:
        if bool(getattr(self.config, "defer_prefill_compression", False)):
            return True
        if not bool(getattr(self.config, "defer_prefill_compression_on_ascend", False)):
            return False
        return is_ascend_runtime(self._base_runner) or is_ascend_environment_available()

    def _resolve_prefill_len_for_existing_request(self, req_id: str) -> int:
        """Best-effort prompt length lookup for requests that predate proxy state."""
        candidates: list[int] = []
        requests = getattr(self._base_runner, "requests", None)
        if isinstance(requests, dict):
            req_state = requests.get(req_id)
            if req_state is not None:
                candidates.append(_resolve_full_prefill_len_from_request_like(req_state))

        input_batch = getattr(self._base_runner, "input_batch", None)
        req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch else None
        req_index = req_id_to_index.get(req_id) if isinstance(req_id_to_index, dict) else None
        if isinstance(req_index, int):
            for attr_name in ("num_prompt_tokens", "prompt_lens", "prompt_lengths"):
                prompt_lens = getattr(input_batch, attr_name, None)
                if prompt_lens is None:
                    continue
                try:
                    value = prompt_lens[req_index]
                    if hasattr(value, "item"):
                        value = value.item()
                    candidates.append(int(value))
                except Exception:
                    continue

        return max(candidates, default=0)

    def _ensure_state_for_existing_request(self, req_id: str) -> Any:
        state = self.state_store.get(req_id) if hasattr(self.state_store, "get") else None
        if state is not None:
            return state
        prefill_len = self._resolve_prefill_len_for_existing_request(req_id)
        state = self.state_store.ensure(
            req_id=req_id,
            prefill_len=prefill_len,
            protect_prefill=bool(self.config.protect_prefill),
        )
        if self.config.log_decisions:
            self._logger.debug(
                "TriAttention backfilled runtime state for cached request: "
                "req=%s prefill_len=%d",
                req_id,
                prefill_len,
            )
        return state

    def _log_execution_path_trigger_guard(
        self,
        *,
        req_id: str,
        reason: str,
        **fields: Any,
    ) -> None:
        if not self._log_execution_path:
            return
        if self._log_execution_path_core_only:
            return
        key = (req_id, reason)
        if key in self._logged_execution_path_trigger_guards:
            return
        self._logged_execution_path_trigger_guards.add(key)
        parts = [
            "TRIATTN_EXEC_PATH runner_trigger_guard",
            "req=%s",
            "step=%d",
            "reason=%s",
            "core_entered=%s",
        ]
        values: list[Any] = [req_id, self._last_step, reason, False]
        for key in sorted(fields):
            parts.append(f"{key}=%s")
            values.append(fields[key])
        self._logger.info(" ".join(parts), *values)

    def _supplement_worker_self_triggers(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> dict[str, CompressionSignal]:
        """Generate self-trigger signals for requests the Scheduler missed.

        The Scheduler's estimate can lag behind the Worker's actual KV length
        by several chunks due to async scheduling.  This check uses the real
        Worker-side state so compression triggers at the right time.
        """
        if not self.config.enable_experimental_kv_compaction:
            return signals
        scheduled_items = get_scheduled_token_items(scheduler_output)
        defer_chunked_prefill = self._should_defer_chunked_prefill_compression()
        is_ascend = is_ascend_runtime(self._base_runner) or is_ascend_environment_available()
        for _raw_key, req_id, scheduled_tokens in scheduled_items:
            scheduled_tokens_i = max(1, int(scheduled_tokens))
            # If Scheduler already sent a trigger, check if we should
            # override it with a more accurate block-table-based estimate.
            existing = signals.get(req_id)
            state = self._ensure_state_for_existing_request(req_id)
            prefill_len = state.prefill_len
            existing_estimate = int(
                getattr(existing, "estimated_cache_len", 0) or 0
            ) if existing is not None else 0
            is_prefill_step_for_threshold = (
                scheduled_tokens_i > 1
                or (prefill_len > 0 and 0 < existing_estimate < prefill_len)
            )
            req_state = None
            requests = getattr(self._base_runner, "requests", None)
            if isinstance(requests, dict):
                req_state = requests.get(req_id)
            num_computed_tokens = (
                int(getattr(req_state, "num_computed_tokens", 0))
                if req_state is not None
                else None
            )
            is_prefill_step_for_limit = is_prefill_phase_for_limit(
                scheduler_output=scheduler_output,
                req_id=req_id,
                scheduled_tokens=scheduled_tokens_i,
                prefill_len=prefill_len,
                num_computed_tokens=num_computed_tokens,
            )
            if defer_chunked_prefill and is_prefill_step_for_threshold:
                if existing is not None and existing.should_compress:
                    signals.pop(req_id, None)
                    if hasattr(self.state_store, "mark_compression_skipped"):
                        self.state_store.mark_compression_skipped(
                            req_id=req_id,
                            reason="defer_prefill",
                            step=self._last_step,
                        )
                    if self.config.log_decisions:
                        self._logger.debug(
                            "TriAttention dropped chunked-prefill trigger: "
                            "req=%s scheduled=%d",
                            req_id,
                            scheduled_tokens_i,
                        )
                self._log_execution_path_trigger_guard(
                    req_id=req_id,
                    reason="defer_prefill",
                    prefill_len=prefill_len,
                    scheduled=scheduled_tokens_i,
                    scheduler_had_signal=bool(existing),
                )
                continue
            max_prefill_compressions = int(
                getattr(self.config, "prefill_max_compressions_on_ascend", 1)
                or 0
            )
            if (
                is_prefill_step_for_limit
                and is_ascend
                and int(getattr(state, "compression_count", 0) or 0)
                >= max_prefill_compressions
            ):
                if existing is not None and existing.should_compress:
                    signals.pop(req_id, None)
                    if hasattr(self.state_store, "mark_compression_skipped"):
                        self.state_store.mark_compression_skipped(
                            req_id=req_id,
                            reason="prefill_compression_limit",
                            step=self._last_step,
                        )
                    if self.config.log_decisions:
                        self._logger.debug(
                            "TriAttention dropped prefill trigger after limit: "
                            "req=%s scheduled=%d compression_count=%d limit=%d",
                            req_id,
                            scheduled_tokens_i,
                            int(getattr(state, "compression_count", 0) or 0),
                            max_prefill_compressions,
                        )
                self._log_execution_path_trigger_guard(
                    req_id=req_id,
                    reason="prefill_compression_limit",
                    compression_count=int(getattr(state, "compression_count", 0) or 0),
                    limit=max_prefill_compressions,
                    prefill_len=prefill_len,
                    scheduled=scheduled_tokens_i,
                    scheduler_had_signal=bool(existing),
                )
                continue
            # Compute actual KV length on the Worker side.
            # kv_from_blocks: block table already includes current step's
            # allocated blocks (update_states runs before execute_model),
            # so scheduled_tokens must NOT be added again.
            kv_from_blocks = False
            if state.compression_count > 0 and state.current_cache_len > 0:
                actual_kv = state.current_cache_len
            else:
                # First-time compression: use block table for ground truth.
                # num_computed_tokens from base_runner.requests has async lag
                # (Scheduler runs 2-3 chunks ahead), so we derive actual KV
                # from the physical block count on the Worker side.
                actual_kv_bt = self._get_actual_kv_from_block_table(req_id)
                if actual_kv_bt is not None:
                    actual_kv = actual_kv_bt
                    kv_from_blocks = True
                else:
                    # Fallback: base_runner.requests (may be stale).
                    if req_state is not None:
                        actual_kv = int(getattr(req_state, "num_computed_tokens", 0))
                    else:
                        continue
            threshold = self._compression_threshold(
                prefill_len,
                is_prefill_step=is_prefill_step_for_threshold,
            )
            if kv_from_blocks:
                # Block table capacity already covers scheduled tokens.
                effective_kv = actual_kv
            else:
                effective_kv = actual_kv + scheduled_tokens_i
            if should_guard_fast_recency_long_context(
                config=self.config,
                effective_tokens=effective_kv,
                prefill_len=prefill_len,
            ):
                guard_tokens = int(
                    getattr(
                        self.config,
                        "fast_recency_long_context_guard_tokens",
                        0,
                    )
                    or 0
                )
                if existing is not None and existing.should_compress:
                    signals.pop(req_id, None)
                    if hasattr(self.state_store, "mark_compression_skipped"):
                        self.state_store.mark_compression_skipped(
                            req_id=req_id,
                            reason="fast_recency_long_context_guard",
                            step=self._last_step,
                        )
                    if self.config.log_decisions:
                        self._logger.debug(
                            "TriAttention dropped long-context fast-recency "
                            "trigger: req=%s effective_kv=%d prefill_len=%d "
                            "guard_tokens=%d scheduled=%d from_blocks=%s",
                            req_id,
                            effective_kv,
                            prefill_len,
                            guard_tokens,
                            scheduled_tokens_i,
                            kv_from_blocks,
                        )
                self._log_execution_path_trigger_guard(
                    req_id=req_id,
                    reason="fast_recency_long_context_guard",
                    actual_kv=actual_kv,
                    effective_kv=effective_kv,
                    from_blocks=kv_from_blocks,
                    guard_tokens=guard_tokens,
                    hint="set_sparse_stats_path_or_disable_long_context_guard",
                    prefill_len=prefill_len,
                    scheduled=scheduled_tokens_i,
                    scheduler_had_signal=bool(existing),
                    threshold=threshold,
                )
                continue
            if effective_kv < threshold:
                if existing is not None and existing.should_compress:
                    signals.pop(req_id, None)
                    if hasattr(self.state_store, "mark_compression_skipped"):
                        self.state_store.mark_compression_skipped(
                            req_id=req_id,
                            reason="below_worker_threshold",
                            step=self._last_step,
                        )
                    if self.config.log_decisions:
                        self._logger.debug(
                            "TriAttention dropped stale scheduler trigger below "
                            "worker threshold: req=%s actual_kv=%d effective_kv=%d "
                            "threshold=%d scheduled=%d from_blocks=%s",
                            req_id, actual_kv, effective_kv, threshold,
                            scheduled_tokens, kv_from_blocks,
                        )
                    self._log_execution_path_trigger_guard(
                        req_id=req_id,
                        reason="below_worker_threshold",
                        actual_kv=actual_kv,
                        effective_kv=effective_kv,
                        from_blocks=kv_from_blocks,
                        prefill_len=prefill_len,
                        scheduled=scheduled_tokens_i,
                        scheduler_had_signal=True,
                        threshold=threshold,
                    )
                continue
            if (
                existing is not None
                and existing.should_compress
                and int(getattr(existing, "scheduled_tokens", 1)) <= 1
            ):
                continue
            if self.config.log_decisions:
                self._logger.info(
                    "TriAttention worker self-trigger: req=%s actual_kv=%d "
                    "effective_kv=%d scheduled=%d threshold=%d "
                    "from_blocks=%s scheduler_had_signal=%s",
                    req_id, actual_kv, effective_kv, scheduled_tokens,
                    threshold, kv_from_blocks, bool(existing),
                )
            signals[req_id] = CompressionSignal(
                req_id=req_id,
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=effective_kv,
                step=self._last_step,
                kv_usage=None,
                protect_prefill=self.config.protect_prefill,
                prefill_len=prefill_len,
                scheduled_tokens=scheduled_tokens_i,
            )
        return signals

    def _execute_compression_actions(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> None:
        self._pending_compression_events = execute_runner_compression_actions(
            executor=self.executor,
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            signals=signals,
            strict_no_downgrade=self._strict_no_downgrade,
            allowed_strict_skip_reasons=self._allowed_strict_skip_reasons,
            logger=self._logger,
            log_decisions=bool(self.config.log_decisions),
            log_worker_events=bool(self._log_worker_events),
            logging_enabled=bool(self.config.logging_enabled),
            log_execution_path=bool(self._log_execution_path),
            log_execution_path_core_only=bool(self._log_execution_path_core_only),
            log_selector_debug=bool(
                self._log_execution_path
                and getattr(self.config, "log_selector_debug", False)
            ),
        )
        # === BUG-RECL 断点 12（LE-R-001）：pending 事件统计 ===
        if os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1":
            n_total = len(self._pending_compression_events)
            n_applied = sum(1 for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied")
            n_applied_with_reclaim = sum(1 for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied" and isinstance(e.get("block_reclaim"), dict))
            n_applied_no_reclaim = n_applied - n_applied_with_reclaim
            reqs = [e.get("req_id") for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied"][:5]
            print(f"BUG-RECL [LE-R-001] pending n_total={n_total} n_applied={n_applied} n_applied_with_reclaim={n_applied_with_reclaim} n_applied_no_reclaim={n_applied_no_reclaim} sample_reqs={reqs}", flush=True)
        # === BUG-RECL 断点 12 结束 ===

    def _apply_worker_block_reclaim_events(self) -> None:
        """Apply reclaim shrink to worker-side block tables before prepare_inputs()."""
        apply_worker_block_reclaim_events(
            base_runner=self._base_runner,
            events=self._pending_compression_events,
        )

    def _patch_scheduler_output_for_compressed_reqs(self, scheduler_output: Any) -> None:
        """Trim stale new_block_ids for compressed requests (V1 batch-queue).

        After compression, the worker's block table has been shrunk via
        worker_reclaim_sync.  But the scheduler may still send excess
        new_block_ids based on its stale view.  This trims those to match the
        just-retained cache length and then applies the worker row capacity cap.
        """
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return
        req_ids = getattr(cached_reqs, "req_ids", None)
        new_block_ids_list = getattr(cached_reqs, "new_block_ids", None)
        if not isinstance(req_ids, list) or not isinstance(new_block_ids_list, list):
            return
        if len(req_ids) != len(new_block_ids_list):
            return

        # Get block table info from worker.
        input_batch = getattr(self._base_runner, "input_batch", None)
        block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
        if block_table_obj is None:
            return

        tables = _block_table_inner_tables(block_table_obj)
        if not tables:
            return

        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if not isinstance(req_id_to_index, dict):
            return

        fallback_block_size = _safe_positive_int(
            getattr(getattr(self._base_runner, "cache_config", None), "block_size", None)
        )
        events_by_req_id = _applied_compression_events_by_req_id(
            self._pending_compression_events
        )

        def _group_limits_for_event(req_index: int, retained_cache_len: int | None) -> list[int | None]:
            limits: list[int | None] = []
            for table in tables:
                current = _table_row_block_count(table, req_index)
                if current is None:
                    limits.append(None)
                    continue
                limit: int | None = None
                block_size = _table_block_size(table, fallback_block_size)
                if retained_cache_len is not None and block_size is not None:
                    required = _ceil_div_positive(retained_cache_len, block_size)
                    limit = max(0, required - current)
                max_blocks = _table_max_blocks(table, block_table_obj)
                if max_blocks is not None:
                    capacity_limit = max(0, max_blocks - current)
                    limit = capacity_limit if limit is None else min(limit, capacity_limit)
                limits.append(limit)
            return limits

        def _trim_group(group: Any, limit: int | None) -> tuple[Any, int | None, int | None, bool]:
            if not isinstance(group, (list, tuple)):
                return group, None, None, False
            before = len(group)
            if limit is None:
                return group, before, before, False
            after = max(0, min(before, int(limit)))
            if after == before:
                return group, before, after, False
            if isinstance(group, tuple):
                return tuple(group[:after]), before, after, True
            return list(group[:after]), before, after, True

        def _trim_new_block_ids(
            new_block_ids: Any,
            group_limits: list[int | None],
        ) -> tuple[Any, int, int, bool]:
            if not isinstance(new_block_ids, (list, tuple)):
                return new_block_ids, 0, 0, False
            if (
                len(tables) == 1
                and not (
                    len(new_block_ids) == 1
                    and isinstance(new_block_ids[0], (list, tuple))
                )
                and all(not isinstance(item, (list, tuple)) for item in new_block_ids)
            ):
                trimmed, before, after, changed = _trim_group(
                    new_block_ids,
                    group_limits[0] if group_limits else None,
                )
                return trimmed, int(before or 0), int(after or 0), changed

            trimmed_groups: list[Any] = []
            before_max = 0
            after_max = 0
            changed_any = False
            for gid, group in enumerate(new_block_ids):
                limit = group_limits[gid] if gid < len(group_limits) else None
                trimmed, before, after, changed = _trim_group(group, limit)
                trimmed_groups.append(trimmed)
                if before is not None:
                    before_max = max(before_max, int(before))
                if after is not None:
                    after_max = max(after_max, int(after))
                changed_any = changed_any or changed
            if not changed_any:
                return new_block_ids, before_max, after_max, False
            if isinstance(new_block_ids, tuple):
                return tuple(trimmed_groups), before_max, after_max, True
            return trimmed_groups, before_max, after_max, True

        for i, req_id in enumerate(req_ids):
            rt_state = self.state_store.get(req_id)
            if rt_state is None or rt_state.compression_count <= 0:
                continue

            new_block_ids = new_block_ids_list[i]
            if new_block_ids is None:
                continue

            req_index = req_id_to_index.get(req_id)
            if not isinstance(req_index, int):
                continue

            event = events_by_req_id.get(req_id)
            retained_cache_len = _event_retained_cache_len(event)
            group_limits = _group_limits_for_event(req_index, retained_cache_len)
            if not any(limit is not None for limit in group_limits):
                continue

            trimmed, before_max, after_max, changed = _trim_new_block_ids(
                new_block_ids,
                group_limits,
            )
            if changed:
                new_block_ids_list[i] = trimmed
                if self.config.log_decisions:
                    self._logger.debug(
                        "TriAttention patched new_block_ids: req=%s "
                        "retained_cache_len=%s group_limits=%s "
                        "new_trimmed=%d->%d",
                        req_id, retained_cache_len, group_limits,
                        before_max, after_max,
                    )

    def _needs_effective_input_overrides(self, scheduler_output: Any) -> bool:
        # Tighten scope to "current scheduled batch includes a compressed
        # request". Compression application updates request-local state before
        # this check, so we do not need to keep a separate step-local event path
        # here.
        scheduled_items = get_scheduled_token_items(scheduler_output)
        scheduled_req_ids: list[str] = [req_id for _raw_key, req_id, _scheduled_tokens in scheduled_items]
        if not scheduled_req_ids:
            return False
        checker = getattr(self.state_store, "has_compressed_request_in", None)
        if callable(checker):
            try:
                return bool(checker(scheduled_req_ids))
            except Exception:
                return False
        # Backward-compatible fallback if state_store is substituted in tests.
        checker_any = getattr(self.state_store, "has_active_compressed_requests", None)
        if callable(checker_any):
            try:
                return bool(checker_any())
            except Exception:
                return False
        return False

    def _ensure_runtime_input_patch_if_needed(self, need_effective_overrides: bool) -> None:
        if not need_effective_overrides:
            return
        # Unit tests may instantiate TriAttentionModelRunner with a lightweight fake
        # base runner that does not expose vLLM GPU input-prep internals.
        #
        # vLLM 0.15 runtime paths may populate `input_batch` while leaving
        # `req_states` unset, so treat either surface as sufficient evidence
        # that the native GPU input-prep hooks are available.
        if (
            getattr(self._base_runner, "req_states", None) is None
            and getattr(self._base_runner, "input_batch", None) is None
            and os.environ.get("TRIATTN_DEBUG_ENABLE_V1_OVERRIDE_PATH", "0") != "1"
        ):
            return
        if self._runtime_input_patch_installed:
            return
        patch_ok = install_runtime_input_patch()
        if not patch_ok:
            raise RuntimeError(
                "TriAttention runtime requires gpu seq_len/slot_mapping patch when "
                "effective-length overrides are active, but patch installation failed"
            )
        self._runtime_input_patch_installed = True

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> Any:
        perf_enabled = bool(getattr(self._perf, "enabled", False))
        e2e_enabled = bool(getattr(self._perf, "e2e_enabled", False))
        timed_enabled = perf_enabled or e2e_enabled
        step_phases: dict[str, float] | None = {} if e2e_enabled else None

        def _timed(name: str, fn: Any) -> Any:
            if not e2e_enabled:
                return fn()
            t_phase = time.perf_counter()
            try:
                return fn()
            finally:
                assert step_phases is not None
                step_phases[name] = (time.perf_counter() - t_phase) * 1000.0

        t_total = time.perf_counter() if timed_enabled else 0.0
        t0 = time.perf_counter() if timed_enabled else 0.0
        _timed(
            "register_new_requests",
            lambda: self._register_new_requests(scheduler_output),
        )
        _timed(
            "cleanup_finished_requests",
            lambda: self._cleanup_finished_requests(scheduler_output),
        )
        _timed("mark_preemptions", lambda: self._mark_preemptions(scheduler_output))
        _timed("mark_resumed", lambda: self._mark_resumed(scheduler_output))
        signals = _timed("consume_signals", lambda: self._consume_signals(scheduler_output))
        signals = _timed(
            "supplement_worker_self_triggers",
            lambda: self._supplement_worker_self_triggers(scheduler_output, signals),
        )
        if self._log_execution_path and not self._log_execution_path_core_only:
            triggered_req_ids = [
                req_id
                for req_id, signal in signals.items()
                if bool(getattr(signal, "should_compress", False))
            ]
            if triggered_req_ids:
                self._logger.info(
                    "TRIATTN_EXEC_PATH runner_execute_model_compression_boundary "
                    "step=%d triggered=%d reqs=%s",
                    self._last_step,
                    len(triggered_req_ids),
                    ",".join(str(req_id) for req_id in triggered_req_ids[:8]),
                )
        t_state_ms = (time.perf_counter() - t0) * 1000.0 if timed_enabled else 0.0
        t0 = time.perf_counter() if timed_enabled else 0.0
        _timed(
            "execute_compression_actions",
            lambda: self._execute_compression_actions(scheduler_output, signals),
        )
        t_compress_ms = (time.perf_counter() - t0) * 1000.0 if timed_enabled else 0.0
        _timed(
            "record_compression_events",
            lambda: self._perf.record_compression_events(self._pending_compression_events),
        )
        t0 = time.perf_counter() if timed_enabled else 0.0
        _timed("apply_worker_block_reclaim_events", self._apply_worker_block_reclaim_events)
        _timed(
            "patch_scheduler_output_for_compressed_reqs",
            lambda: self._patch_scheduler_output_for_compressed_reqs(scheduler_output),
        )
        t_reclaim_ms = (time.perf_counter() - t0) * 1000.0 if timed_enabled else 0.0
        need_effective_overrides = _timed(
            "needs_effective_input_overrides",
            lambda: self._needs_effective_input_overrides(scheduler_output),
        )
        _timed(
            "ensure_runtime_input_patch",
            lambda: self._ensure_runtime_input_patch_if_needed(need_effective_overrides),
        )
        bridge_perf: dict[str, float] | None = {} if timed_enabled else None
        output = execute_base_model_with_effective_overrides(
            base_runner=self._base_runner,
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            use_effective_overrides=need_effective_overrides,
            config=self.config,
            perf_out=bridge_perf,
        )
        if e2e_enabled and step_phases is not None and bridge_perf is not None:
            step_phases["override_prep"] = float(
                bridge_perf.get("override_prep_ms", 0.0)
            )
            step_phases["base_execute_model"] = float(
                bridge_perf.get("base_exec_ms", 0.0)
            )
        _timed("record_model_output", lambda: self._perf.record_model_output(output))
        t_total_exec_ms = (time.perf_counter() - t_total) * 1000.0 if timed_enabled else 0.0
        has_trigger = any(bool(sig.should_compress) for sig in signals.values()) if signals else False
        self._perf.record_step(
            has_trigger=has_trigger,
            uses_overrides=bool(need_effective_overrides),
            t_state_ms=t_state_ms,
            t_compress_ms=t_compress_ms,
            t_reclaim_ms=t_reclaim_ms,
            t_override_prep_ms=float((bridge_perf or {}).get("override_prep_ms", 0.0)),
            t_base_exec_ms=float((bridge_perf or {}).get("base_exec_ms", 0.0)),
            t_total_exec_ms=t_total_exec_ms,
        )
        pending_before_attach = len(self._pending_compression_events)
        t_attach_execute = time.perf_counter() if e2e_enabled else 0.0
        output, self._pending_compression_events = attach_execute_model_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
            scheduler_output=scheduler_output,
        )
        if e2e_enabled and step_phases is not None:
            step_phases["attach_execute_model_events"] = (
                time.perf_counter() - t_attach_execute
            ) * 1000.0
            step_phases["execute_model_total"] = (time.perf_counter() - t_total) * 1000.0
            num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
            num_reqs = len(num_scheduled) if isinstance(num_scheduled, dict) else None
            total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
            if total_tokens is None and isinstance(num_scheduled, dict):
                try:
                    total_tokens = sum(int(v) for v in num_scheduled.values())
                except Exception:
                    total_tokens = None
            try:
                total_tokens_i = int(total_tokens)
            except Exception:
                total_tokens_i = None
            self._perf.record_e2e_step(
                step_phases,
                num_reqs=num_reqs,
                total_tokens=total_tokens_i,
                has_trigger=has_trigger,
                uses_overrides=bool(need_effective_overrides),
                pending_events=pending_before_attach,
            )
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        # In vLLM V1 async path, execute_model returns None and the actual
        # ModelRunnerOutput (with sampled_token_ids) is produced here.
        profile_enabled = phase_profile_enabled()
        e2e_enabled = bool(getattr(self._perf, "e2e_enabled", False))
        sample_phases: dict[str, float] | None = {} if e2e_enabled else None
        t0 = phase_now() if profile_enabled else 0.0
        t_sample = time.perf_counter() if e2e_enabled else 0.0
        try:
            output = self._base_runner.sample_tokens(grammar_output)
        finally:
            if e2e_enabled and sample_phases is not None:
                sample_phases["base_sample_tokens"] = (
                    time.perf_counter() - t_sample
                ) * 1000.0
            if profile_enabled:
                record_phase(
                    "base_runner_sample_tokens",
                    phase_elapsed_ms(t0),
                    {
                        "pending_events": len(self._pending_compression_events),
                    },
                )
        t_attach = time.perf_counter() if e2e_enabled else 0.0
        output, self._pending_compression_events = attach_sample_tokens_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
        )
        if e2e_enabled and sample_phases is not None:
            sample_phases["attach_sample_tokens_events"] = (
                time.perf_counter() - t_attach
            ) * 1000.0
            self._perf.record_e2e_sample(
                sample_phases,
                pending_events=len(self._pending_compression_events),
            )
        return output

    def snapshot_states(self) -> dict[str, Any]:
        """Return debug snapshot for observability tests."""
        return self.state_store.snapshot()
