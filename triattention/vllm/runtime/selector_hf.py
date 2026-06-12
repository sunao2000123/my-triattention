"""HF-aligned TriAttention selector implementation for TriAttention runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import gather_request_k_dense_range

# --- INSTRUMENTATION (Level H helpers) ---
# Module-level state for "first call only" throttling. The selector is called
# once per (req, layer) per compression, so even with verbose dump the per-step
# cost is bounded; we still cap the *detailed* log emission to the first few
# (req_id, layer) tuples the selector sees in this process, to avoid log flood
# during long generations.  The user can override the cap with
# TRIATTN_DEBUG_INSTRUMENT_VERBOSE_MAX.
_INSTR_VERBOSE_MAX = int(os.environ.get("TRIATTN_DEBUG_INSTRUMENT_VERBOSE_MAX", "8"))
_INSTR_VERBOSE_DUMPED: set[tuple[str, int]] = set()


def _instr_verbose_allowed(req_id: str | None, layer_idx: int) -> bool:
    """Return True at most ``_INSTR_VERBOSE_MAX`` times per (req_id, layer).

    Gated by ``TRIATTN_DEBUG_INSTRUMENT=1``. The first ``_INSTR_VERBOSE_MAX``
    distinct (req_id, layer) pairs emit a Level H detailed dump; subsequent
    pairs emit only a one-line summary.  This is the "first call only" gate
    requested by the user: each (request, layer) gets one full dump, then we
    stop.  ``req_id`` may be None (closure-built selector) in which case we
    fall back to a global counter.
    """
    if os.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") != "1":
        return False
    if req_id is None:
        # No req_id context: dump the first _INSTR_VERBOSE_MAX times total.
        key = ("__noreq__", layer_idx)
    else:
        key = (str(req_id), int(layer_idx))
    if key in _INSTR_VERBOSE_DUMPED:
        return False
    if len(_INSTR_VERBOSE_DUMPED) >= _INSTR_VERBOSE_MAX:
        return False
    _INSTR_VERBOSE_DUMPED.add(key)
    return True


def _format_index_sample(
    indices: torch.Tensor,
    *,
    head_count: int,
    sample_heads: tuple[int, ...],
    max_indices: int,
) -> str:
    """Compact dump of a [H, K] keep-index tensor.

    For each sampled head, prints the first ``max_indices`` kept token
    positions plus a few summary stats (min/max/mean/std).  Keeps log
    line short even when K is large.
    """
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return f"empty(shape={tuple(indices.shape) if isinstance(indices, torch.Tensor) else None})"
    parts: list[str] = []
    for h in sample_heads:
        if h < 0 or h >= head_count:
            continue
        row = indices[h]
        if row.numel() == 0:
            parts.append(f"h{h}=[]")
            continue
        row_cpu = row.detach().to(device="cpu", dtype=torch.long)
        n = int(row_cpu.numel())
        k = min(max_indices, n)
        head_vals = row_cpu[:k].tolist()
        stats = (
            f"min={int(row_cpu.min().item())} "
            f"max={int(row_cpu.max().item())} "
            f"mean={float(row_cpu.float().mean().item()):.1f} "
            f"std={float(row_cpu.float().std(unbiased=False).item()):.1f}"
        )
        more = f"...(+{n - k})" if n > k else ""
        parts.append(f"h{h}=[{','.join(str(int(x)) for x in head_vals)}{more}] {stats}")
    return " | ".join(parts)


def _format_score_quantiles(scores: torch.Tensor) -> str:
    """Per-head score quantiles for the dynamic (non-pinned) region.

    Returns a short string: ``p50=X p90=X p99=X max=X`` averaged over heads.
    """
    if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
        return "no_scores"
    flat = scores.detach().to(dtype=torch.float32, device="cpu").flatten()
    if flat.numel() == 0:
        return "no_scores"
    # Use torch.quantile (sync, but tiny — only fires when verbose-gated).
    try:
        qs = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float32)
        q = torch.quantile(flat, qs)
        return (
            f"p50={float(q[0]):.3f} p90={float(q[1]):.3f} "
            f"p99={float(q[2]):.3f} max={float(q[3]):.3f} "
            f"min={float(flat.min().item()):.3f} mean={float(flat.mean().item()):.3f}"
        )
    except Exception:
        return f"min={float(flat.min().item()):.3f} max={float(flat.max().item()):.3f}"


def build_triattention_selector(
    config: TriAttentionRuntimeConfig,
    base_runner: Any | None = None,
) -> tuple[
    Callable[..., dict[str, Any] | None] | None,
    Callable[..., dict[str, Any] | None] | None,
    str,
]:
    """Build TriAttention selector callable.

    The returned selector emits either:
    - {"mode": "shared", "indices": Tensor|list[int]}
    - {"mode": "per_head", "indices": Tensor|list[list[int]]}
    """
    requested_pruning_mode = config.pruning_mode
    if requested_pruning_mode == "per_layer" and not bool(
        getattr(config, "allow_per_layer_mode", False)
    ):
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:per_layer_mode_disabled:"
            "set allow_per_layer_mode=True for explicit opt-in"
        )

    strict_triton_required = bool(
        config.enable_experimental_kv_compaction and config.require_triton_scoring
    )
    if config.sparse_stats_path is None:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_set"
            )
        return None, None, "stats_path_not_set"

    stats_path = Path(config.sparse_stats_path).expanduser()
    if not stats_path.exists():
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_found"
            )
        return None, None, "stats_path_not_found"

    try:
        from triattention.vllm.core.config import TriAttentionConfig
        from triattention.vllm.core.compressor import TriAttentionCompressor
        from triattention.vllm.core.scoring import compute_scores_triton
        from triattention.vllm.core.utils import normalize_scores
    except Exception as exc:  # pragma: no cover - import safety
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:import_failed:{type(exc).__name__}"
        ) from exc

    if requested_pruning_mode not in {"per_layer", "per_head", "per_layer_per_head"}:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:unsupported_pruning_mode:{requested_pruning_mode}"
            )
        return None, None, f"unsupported_pruning_mode:{requested_pruning_mode}"
    # Keep per-head score tensor and decide aggregation in selector;
    # this matches HF path better than forcing mean aggregation inside scoring.
    pruning_mode = "per_head"
    per_head_semantics = config.per_head_selection_semantics

    def _resolve_effective_model_path() -> Path | None:
        if getattr(config, "model_path", None) is not None:
            return Path(config.model_path)
        if base_runner is None:
            return None
        candidates: list[Any] = []
        candidates.append(getattr(getattr(base_runner, "model_config", None), "model", None))
        candidates.append(
            getattr(
                getattr(getattr(base_runner, "vllm_config", None), "model_config", None),
                "model",
                None,
            )
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return Path(candidate)
            if isinstance(candidate, Path):
                return candidate
        return None

    effective_model_path = _resolve_effective_model_path()

    # Resolve the actual compute device from the base_runner.
    # The default in TriAttentionConfig is torch.device("cuda")
    # which on vllm-ascend 0.18.0 (CPU-only torch + torch_npu
    # plugin) raises NotImplementedError for aten::empty.
    # We try several attributes in order of reliability.
    effective_device: torch.device = torch.device("cpu")
    if base_runner is not None:
        _device_candidates = (
            "device",                # vllm v1 base worker
            "drafter_device",        # spec decode
        )
        for _dattr in _device_candidates:
            _dv = getattr(base_runner, _dattr, None)
            if isinstance(_dv, torch.device):
                effective_device = _dv
                break
            if isinstance(_dv, str) and _dv.strip():
                try:
                    effective_device = torch.device(_dv)
                    break
                except Exception:
                    pass
        # Fall back: ascend-specific attribute (NPUModelRunner stores
        # it as torch.device('npu:0') on self.device).
        if effective_device.type == "cpu" and base_runner is not None:
            _dv = getattr(base_runner, "device", None)
            if isinstance(_dv, torch.device) and _dv.type != "cpu":
                effective_device = _dv
            elif isinstance(_dv, str) and _dv.strip():
                try:
                    _tmp = torch.device(_dv)
                    if _tmp.type != "cpu":
                        effective_device = _tmp
                except Exception:
                    pass

    tri_cfg = TriAttentionConfig(
        stats_path=stats_path,
        model_path=effective_model_path,
        kv_budget=config.kv_budget,
        divide_length=config.divide_length,
        pruning_mode=pruning_mode,
        score_aggregation=config.sparse_score_aggregation,
        sparse_normalize_scores=config.sparse_normalize_scores,
        window_size=min(config.window_size, max(config.kv_budget - 1, 0)),
        include_prefill_in_budget=config.include_prefill_in_budget,
        protect_prefill=config.protect_prefill,
        disable_mlr=config.disable_mlr,
        disable_trig=config.disable_trig,
        disable_top_n_high_freq=config.disable_top_n_high_freq,
        use_triton_scoring=True,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        device=effective_device,
    )
    compressor = TriAttentionCompressor(tri_cfg)
    available_layers_sorted: tuple[int, ...] | None = None
    available_layers_set: set[int] | None = None

    def _resolve_runtime_heads(kv_cache: torch.Tensor) -> int:
        """Resolve the per-layer ``num_kv_heads`` from a kv cache tensor.

        CUDA 5D layout:
          ``[2, num_blocks, block_size, H, D]`` -> ``H`` is ``shape[3]``
          ``[num_blocks, 2, block_size, H, D]`` -> ``H`` is ``shape[3]``

        Ascend (vllm-ascend v0.18.0) 4D layout:
          ``[num_blocks, block_size, H, D]`` -> ``H`` is ``shape[2]``

        The previous code hard-assumed ``shape[3]`` which is the
        head_dim on the Ascend 4D layout, causing ``runtime_heads``
        to be 128/256/etc instead of e.g. 8, which downstream
        triggers ``repeat_interleave(group_size=0)`` and a
        ``TypeError`` that gets re-wrapped into
        ``TRIATTN_FATAL_TRITON_SCORING_REQUIRED``.
        """
        if kv_cache.ndim == 4:
            # Ascend: [num_blocks, block_size, H, D]
            return int(kv_cache.shape[2])
        if kv_cache.ndim == 5:
            # CUDA: K/V split on dim0 or dim1; heads always on dim3
            return int(kv_cache.shape[3])
        raise RuntimeError(
            f"unsupported_kv_cache_layout_for_head_resolve:ndim={kv_cache.ndim}"
        )

    def _resolve_effective_recent_count(total_tokens: int) -> int:
        if total_tokens <= 0 or config.window_size <= 0:
            return 0
        # The runtime selector must preserve the same trailing protection window
        # regardless of request lifecycle details. Tying this to transient
        # "recent_unabsorbed" bookkeeping lets live serve requests under-protect
        # the tail (often collapsing to zero) even though fresh/offline
        # selection correctly preserves `window_size` tokens. That divergence
        # changes the keep set and cascades into output corruption.
        return min(config.window_size, total_tokens)

    def _resolve_layer_idx_for_stats(layer_idx: int) -> int:
        nonlocal available_layers_sorted
        nonlocal available_layers_set
        compressor._lazy_init()
        if available_layers_sorted is None or available_layers_set is None:
            available_layers_sorted = tuple(sorted(compressor.head_stats.keys()))
            available_layers_set = set(available_layers_sorted)
        if not available_layers_sorted:
            raise RuntimeError("empty_head_stats")
        if layer_idx in available_layers_set:
            return layer_idx
        return available_layers_sorted[layer_idx % len(available_layers_sorted)]

    reduced_head_stats_cache: dict[tuple[int, int], tuple[dict[str, torch.Tensor], torch.Tensor]] = {}

    def _build_reduced_layer_stats(
        *,
        resolved_layer_idx: int,
        target_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        cache_key = (resolved_layer_idx, target_heads)
        cached = reduced_head_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        layer_stats = compressor.head_stats[resolved_layer_idx]
        layer_freq_scale_sq = compressor.freq_scale_sq[resolved_layer_idx]
        source_heads = int(layer_freq_scale_sq.shape[0])
        if source_heads == target_heads:
            reduced = (layer_stats, layer_freq_scale_sq)
            reduced_head_stats_cache[cache_key] = reduced
            return reduced
        if target_heads <= 0 or source_heads % target_heads != 0:
            raise RuntimeError(
                f"incompatible_head_mapping:source={source_heads},target={target_heads}"
            )
        group_size = source_heads // target_heads

        reduced_stats: dict[str, torch.Tensor] = {}
        q_abs_mean = layer_stats.get("q_abs_mean")
        if isinstance(q_abs_mean, torch.Tensor):
            reduced_stats["q_abs_mean"] = (
                q_abs_mean.reshape(target_heads, group_size, q_abs_mean.shape[1])
                .mean(dim=1)
                .contiguous()
            )

        q_mean_complex = layer_stats.get("q_mean_complex")
        if isinstance(q_mean_complex, torch.Tensor):
            reduced_stats["q_mean_complex"] = (
                q_mean_complex.reshape(
                    target_heads,
                    group_size,
                    q_mean_complex.shape[1],
                    q_mean_complex.shape[2],
                )
                .mean(dim=1)
                .contiguous()
            )

        reduced_freq_scale_sq = (
            layer_freq_scale_sq.reshape(
                target_heads,
                group_size,
                layer_freq_scale_sq.shape[1],
            )
            .mean(dim=1)
            .contiguous()
        )
        reduced = (reduced_stats, reduced_freq_scale_sq)
        reduced_head_stats_cache[cache_key] = reduced
        return reduced

    def _compute_layer_scores(
        keys_dense: torch.Tensor,
        *,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = int(keys_dense.shape[1])
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )

        scores = _compute_layer_scores_raw(
            keys_dense=keys_dense,
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            round_start=round_start,
        )

        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    def _resolve_layer_score_inputs(
        *,
        layer_idx: int,
        runtime_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, int]:
        resolved_layer_idx = _resolve_layer_idx_for_stats(layer_idx)
        layer_head_stats = compressor.head_stats[resolved_layer_idx]
        layer_freq_scale_sq = compressor.freq_scale_sq[resolved_layer_idx]
        stats_heads = int(layer_freq_scale_sq.shape[0])
        use_hf_group_max = (
            stats_heads != runtime_heads
            and (
                (
                    requested_pruning_mode == "per_head"
                    and per_head_semantics == "hf_aligned_global_per_head"
                )
                or requested_pruning_mode == "per_layer_per_head"
            )
        )
        score_head_stats = layer_head_stats
        score_freq_scale_sq = layer_freq_scale_sq
        group_size = 1
        if use_hf_group_max:
            if runtime_heads <= 0 or stats_heads % runtime_heads != 0:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:incompatible_head_mapping:source={stats_heads},target={runtime_heads}"
                )
            group_size = stats_heads // runtime_heads
        elif stats_heads != runtime_heads:
            score_head_stats, score_freq_scale_sq = _build_reduced_layer_stats(
                resolved_layer_idx=resolved_layer_idx,
                target_heads=runtime_heads,
            )
        return score_head_stats, score_freq_scale_sq, use_hf_group_max, group_size

    def _reduce_grouped_head_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        group_size: int,
        aggregate_mode: str,
    ) -> torch.Tensor:
        grouped = scores.view(
            scores.shape[0],
            runtime_heads,
            group_size,
            scores.shape[-1],
        )
        if aggregate_mode == "mean":
            return grouped.mean(dim=2)
        return grouped.max(dim=2).values

    def _layer_group_aggregation_mode() -> str:
        if requested_pruning_mode == "per_layer_per_head":
            return config.layer_perhead_aggregation
        return "max"

    def _compute_layer_scores_raw(
        *,
        keys_dense: torch.Tensor,
        score_head_stats: dict[str, torch.Tensor],
        score_freq_scale_sq: torch.Tensor,
        use_hf_group_max: bool,
        group_size: int,
        round_start: int,
    ) -> torch.Tensor:
        score_inputs = (
            keys_dense.repeat_interleave(group_size, dim=1).contiguous()
            if use_hf_group_max and group_size > 1
            else keys_dense
        )
        try:
            return compute_scores_triton(
                key_states=score_inputs,
                cache_positions=None,
                head_stats=score_head_stats,
                omega=compressor.omega,
                offsets=compressor.offsets,
                freq_scale_sq=score_freq_scale_sq,
                config=tri_cfg,
                round_start=round_start,
                trig_cache=getattr(compressor, "trig_cache", None),
            )
        except Exception as exc:
            # --- TRANSPARENT TRITON->PYTORCH FALLBACK (NPU) ---
            # On vllm-ascend v0.18.0 the triton-ascend backend does NOT
            # fully support ``tl.static_range(num_offsets > 1)`` inside
            # ``triattention_scoring_kernel`` and may raise ``TypeError``
            # at JIT compile time. The PyTorch implementation in
            # ``compute_scores_pytorch`` is mathematically equivalent
            # (same R-KV formula, same freq/RoPE math), so when we are
            # running on a non-CUDA compute device we transparently
            # fall back. The downstream ``require_triton_scoring`` flag
            # in ``runner_compression_actions`` will not trigger a
            # ``TRIATTN_FATAL_TRITON_SCORING_REQUIRED`` because we are
            # intentionally downgrading only this single call and the
            # algorithm still produces a valid keep-set.
            _is_npu = (
                effective_device.type == "npu"
                or os.environ.get("TRIATTN_ASCEND_TRANSPARENT_TRITON_FALLBACK", "1")
                == "1"
                and effective_device.type != "cuda"
            )
            if _is_npu:
                try:
                    from triattention.vllm.core.scoring import compute_scores_pytorch
                    _scores = compute_scores_pytorch(
                        key_states=score_inputs,
                        cache_positions=None,
                        head_stats=score_head_stats,
                        omega=compressor.omega,
                        offsets=compressor.offsets,
                        freq_scale_sq=score_freq_scale_sq,
                        config=tri_cfg,
                        round_start=round_start,
                    )
                    logger.warning(
                        "[TriAttention] transparent Triton->PyTorch fallback on "
                        "device=%s: %s: %s",
                        effective_device,
                        type(exc).__name__,
                        str(exc)[:160],
                    )
                    return _scores
                except Exception as exc2:
                    raise RuntimeError(
                        f"{TRITON_SCORING_REQUIRED_MARKER}:score_failed_after_fallback:"
                        f"{type(exc).__name__}->{type(exc2).__name__}:{str(exc2)[:120]}"
                    ) from exc2
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:score_failed:{type(exc).__name__}"
                f":{str(exc)[:160]}"
            ) from exc

    def _finalize_layer_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        use_hf_group_max: bool,
        group_size: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:

        if config.sparse_normalize_scores:
            scores = normalize_scores(scores)
        mutate_scores = (
            config.window_size > 0
            or (protect_prefill and prefill_len > 0)
        )
        if mutate_scores:
            scores = scores.clone()
        if config.window_size > 0:
            total_tokens = int(scores.shape[-1])
            recent_count = _resolve_effective_recent_count(total_tokens)
            if recent_count > 0:
                scores[..., total_tokens - recent_count :] = float("inf")
        if protect_prefill and prefill_len > 0:
            scores[..., :prefill_len] = float("inf")
        if use_hf_group_max:
            scores = _reduce_grouped_head_scores(
                scores=scores,
                runtime_heads=runtime_heads,
                group_size=group_size,
                aggregate_mode=_layer_group_aggregation_mode(),
            )
        return scores

    def _compute_layer_scores_paged(
        *,
        kv_cache: torch.Tensor,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = _resolve_runtime_heads(kv_cache)
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
        chunks: list[torch.Tensor] = []
        start = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            keys_chunk = gather_request_k_dense_range(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                start_token=start,
                num_tokens=curr_tokens,
            )
            chunk_scores = _compute_layer_scores_raw(
                keys_dense=keys_chunk,
                score_head_stats=score_head_stats,
                score_freq_scale_sq=score_freq_scale_sq,
                use_hf_group_max=use_hf_group_max,
                group_size=group_size,
                round_start=round_start,
            )
            chunks.append(chunk_scores)
            start += curr_tokens
        scores = torch.cat(chunks, dim=-1)
        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    # Process-local cache for `_build_token_guard_mask` outputs. The
    # selector hot path builds the same bool mask once per chunk per
    # layer; on NPU each `torch.arange` + `torch.zeros_like` + bool OR
    # pays an aclnn launch, and the cached mask is read-only downstream.
    # Keyed on the only parameters that change per build call.
    _guard_mask_cache: dict[tuple, torch.Tensor] = {}
    _GUARD_MASK_CACHE_MAX = 8

    def _build_token_guard_mask(
        *,
        start_token: int,
        num_tokens: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        if config.window_size <= 0 and not (protect_prefill and prefill_len > 0):
            return None
        # Cache key includes start_token because the chunked loop calls
        # this with non-zero start_token for chunks 1..N-1.
        cache_key = (
            int(start_token),
            int(num_tokens),
            int(total_tokens),
            int(prefill_len),
            int(config.window_size),
            bool(protect_prefill),
            (
                device.index
                if isinstance(device, torch.device)
                else -1
            ),
        )
        cached = _guard_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        token_positions = torch.arange(
            start_token,
            start_token + num_tokens,
            device=device,
            dtype=torch.long,
        )
        guard_mask = torch.zeros_like(token_positions, dtype=torch.bool)
        if config.window_size > 0:
            recent_count = _resolve_effective_recent_count(total_tokens)
            window_start = max(0, total_tokens - recent_count)
            guard_mask |= token_positions >= window_start
        if protect_prefill and prefill_len > 0:
            guard_mask |= token_positions < prefill_len
        if len(_guard_mask_cache) >= _GUARD_MASK_CACHE_MAX:
            _guard_mask_cache.pop(next(iter(_guard_mask_cache)))
        _guard_mask_cache[cache_key] = guard_mask
        return guard_mask

    def _apply_token_guards(
        *,
        scores: torch.Tensor,
        start_token: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        guard_mask = _build_token_guard_mask(
            start_token=start_token,
            num_tokens=int(scores.shape[-1]),
            total_tokens=total_tokens,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
            device=scores.device,
        )
        if guard_mask is None:
            return scores
        # Avoid host sync on guard_mask.any().item() in hot path.
        # masked_fill is a no-op when guard_mask has no true elements.
        return scores.masked_fill(guard_mask.view(1, 1, -1), float("inf"))

    _NPU_CHUNK_FLOOR = 16384
    _is_npu_device: bool | None = None

    def _detect_npu_device() -> bool:
        nonlocal _is_npu_device
        if _is_npu_device is not None:
            return _is_npu_device
        try:
            _dev = effective_device
            _is_npu_device = (
                (isinstance(_dev, torch.device) and _dev.type == "npu")
                or (isinstance(_dev, str) and _dev.strip() == "npu")
            )
        except Exception:
            _is_npu_device = False
        return _is_npu_device

    def _score_chunk_tokens(block_size: int, total_tokens: int) -> int:
        upper = max(block_size, int(config.score_chunk_max_tokens))
        # Small/medium effective lengths do not need chunking; avoiding chunk splits
        # reduces Python loop overhead and kernel launches in the hot scoring path.
        if total_tokens <= upper:
            return max(block_size, total_tokens)
        # Ascend NPU optimization: each chunk below ~16k pays a fixed
        # kernel-launch + host-sync cost that dominates wall time on NPU
        # because each torch.topk / masked_fill / scatter_ launches an
        # aclnn op and the launch path is ~5x slower than CUDA. Bump the
        # chunk ceiling to 16k on NPU when the full sequence is at
        # least 16k, so a 20k cache splits into 2 chunks instead of 5
        # (saves ~7 kernel launches per layer x 36 layers = ~10s of
        # launch overhead avoided on a typical 36-layer 20k-token
        # compression).
        if _detect_npu_device() and total_tokens >= _NPU_CHUNK_FLOOR:
            return _NPU_CHUNK_FLOOR
        return upper

    def _select_keep_indices_paged_streaming(
        *,
        kv_cache: torch.Tensor,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any]:
        runtime_heads = _resolve_runtime_heads(kv_cache)
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
        k = min(budget_total, total_tokens)
        if k <= 0:
            return {"mode": "shared", "indices": []}

        norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        raw_chunk_scores_cache: list[torch.Tensor] | None = None
        if config.sparse_normalize_scores:
            eps = 1e-8
            sum_vec: torch.Tensor | None = None
            sumsq_vec: torch.Tensor | None = None
            count = 0
            raw_chunk_scores_cache = []
            start = 0
            while start < total_tokens:
                curr_tokens = min(chunk_tokens, total_tokens - start)
                keys_chunk = gather_request_k_dense_range(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    block_size=block_size,
                    start_token=start,
                    num_tokens=curr_tokens,
                )
                raw_scores = _compute_layer_scores_raw(
                    keys_dense=keys_chunk,
                    score_head_stats=score_head_stats,
                    score_freq_scale_sq=score_freq_scale_sq,
                    use_hf_group_max=use_hf_group_max,
                    group_size=group_size,
                    round_start=round_start,
                )[0]
                raw_chunk_scores_cache.append(raw_scores)
                raw_fp32 = raw_scores.to(dtype=torch.float32)
                chunk_sum = raw_fp32.sum(dim=-1)
                chunk_sumsq = (raw_fp32 * raw_fp32).sum(dim=-1)
                if sum_vec is None:
                    sum_vec = chunk_sum
                    sumsq_vec = chunk_sumsq
                else:
                    sum_vec = sum_vec + chunk_sum
                    sumsq_vec = sumsq_vec + chunk_sumsq
                count += curr_tokens
                start += curr_tokens
            if sum_vec is None or sumsq_vec is None or count <= 0:
                return None
            mean = sum_vec / float(count)
            if count > 1:
                var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
            else:
                var = torch.zeros_like(mean)
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var)
            std_safe = torch.where(std < eps, torch.ones_like(std), std)
            norm_stats = (mean, std_safe)

        # normalize_scores is z-score along token axis (affine monotonic per head/layer),
        # but for paths that aggregate across heads (e.g. max), normalization must be
        # preserved for HF alignment semantics. We use a two-pass chunked statistics
        # accumulation above instead of materializing full sequence scores.
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head:
            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
        else:
            best_scores = None
            best_indices = None

        start = 0
        chunk_idx = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            if raw_chunk_scores_cache is not None and chunk_idx < len(raw_chunk_scores_cache):
                chunk_scores = raw_chunk_scores_cache[chunk_idx].unsqueeze(0)
            else:
                keys_chunk = gather_request_k_dense_range(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    block_size=block_size,
                    start_token=start,
                    num_tokens=curr_tokens,
                )
                chunk_scores = _compute_layer_scores_raw(
                    keys_dense=keys_chunk,
                    score_head_stats=score_head_stats,
                    score_freq_scale_sq=score_freq_scale_sq,
                    use_hf_group_max=use_hf_group_max,
                    group_size=group_size,
                    round_start=round_start,
                )
            if norm_stats is not None:
                mean, std_safe = norm_stats
                chunk_scores = (
                    chunk_scores - mean.view(1, -1, 1)
                ) / std_safe.view(1, -1, 1)
            if use_hf_group_max:
                chunk_scores = _reduce_grouped_head_scores(
                    scores=chunk_scores,
                    runtime_heads=runtime_heads,
                    group_size=group_size,
                    aggregate_mode=_layer_group_aggregation_mode(),
                )
            chunk_scores = _apply_token_guards(
                scores=chunk_scores,
                start_token=start,
                total_tokens=total_tokens,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
            )

            if wants_per_head and chunk_scores.ndim == 3:
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            else:
                if chunk_scores.ndim == 3:
                    chunk_scores = chunk_scores.max(dim=1).values
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            start += curr_tokens
            chunk_idx += 1

        if best_indices is None:
            return {"mode": "shared", "indices": []}
        if wants_per_head and best_indices.ndim == 2:
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            # --- INSTRUMENTATION (Level H return) ---
            # Dump the actual per-head topk indices for this layer so we can
            # see *which* token positions were chosen. Gated by
            # TRIATTN_DEBUG_INSTRUMENT=1 and a "first N (req,layer) pairs"
            # cap so this fires for the first few selections only.
            if _instr_verbose_allowed(req_id=None, layer_idx=layer_idx):
                try:
                    import logging as _lg
                    runtime_h = int(kv_cache.shape[2]) if kv_cache.ndim == 4 else int(kv_cache.shape[3])
                    sample_heads = tuple(
                        sorted({0, runtime_h // 2, runtime_h - 1})
                    )
                    prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                    window_count = _resolve_effective_recent_count(total_tokens)
                    tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                    dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                    keep_dyn_actual = max(0, int(keep_per_head.shape[-1]) - prefill_pinned - tail_pinned)
                    indices_dump = _format_index_sample(
                        keep_per_head,
                        head_count=runtime_h,
                        sample_heads=sample_heads,
                        max_indices=12,
                    )
                    # Score quantiles over the *best* scores tensor (topk scores,
                    # already aggregated across chunks). This tells us the
                    # dynamic-region score distribution the selector saw.
                    score_q = _format_score_quantiles(best_scores if best_scores is not None else keep_per_head)
                    _lg.getLogger(__name__).info(
                        "[TRITN-INSTR] H:selector_topk_inside layer=%d mode=paged_per_head "
                        "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                        "dynamic_total=%d keep_per_head=%d keep_dyn_actual=%d "
                        "score_q=[%s] sample_keep=[%s]",
                        layer_idx, total_tokens, k,
                        prefill_pinned, tail_pinned, dynamic_total,
                        int(keep_per_head.shape[-1]), keep_dyn_actual,
                        score_q, indices_dump,
                    )
                except Exception:
                    pass
            return {"mode": "per_head", "indices": keep_per_head}
        keep = torch.sort(best_indices, dim=-1).values.contiguous()
        if _instr_verbose_allowed(req_id=None, layer_idx=layer_idx):
            try:
                import logging as _lg
                prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                window_count = _resolve_effective_recent_count(total_tokens)
                tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                indices_dump = _format_index_sample(
                    keep.unsqueeze(0) if keep.ndim == 1 else keep,
                    head_count=1,
                    sample_heads=(0,),
                    max_indices=16,
                )
                score_q = _format_score_quantiles(best_scores)
                _lg.getLogger(__name__).info(
                    "[TRITN-INSTR] H:selector_topk_inside layer=%d mode=paged_shared "
                    "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                    "dynamic_total=%d keep_count=%d score_q=[%s] sample_keep=[%s]",
                    layer_idx, total_tokens, k,
                    prefill_pinned, tail_pinned, dynamic_total,
                    int(keep.numel()), score_q, indices_dump,
                )
            except Exception:
                pass
        return {"mode": "shared", "indices": keep}

    def _select_keep_indices(
        *,
        keys_dense: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        block_ids: list[int] | torch.Tensor | None = None,
        block_size: int | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
        req_id: str | None = None,
    ) -> dict[str, Any] | None:
        # --- INSTRUMENTATION (Level E entry) ---
        # Per-layer selector entry. Logs the key parameters used to
        # make the keep/drop decision: how many tokens, what budget,
        # what layer. The return is logged at the end of the method.
        import os as _os_sel
        _sel_instr = _os_sel.environ.get("TRIATTN_DEBUG_INSTRUMENT", "0") == "1"
        if _sel_instr:
            try:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "[TRITN-INSTR] E:select_keep_enter layer=%d total_tokens=%d "
                    "prefill_len=%d protect_prefill=%s round_start=%d budget=%d "
                    "input_path=%s",
                    layer_idx, total_tokens, prefill_len, protect_prefill,
                    round_start, budget_total,
                    "keys_dense" if keys_dense is not None else
                    ("paged" if kv_cache is not None else "NONE"),
                )
            except Exception:
                pass
        if total_tokens <= budget_total:
            return {"mode": "shared", "indices": list(range(total_tokens))}
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None

        if keys_dense is not None:
            scores = _compute_layer_scores(
                keys_dense=keys_dense,
                layer_idx=layer_idx,
                round_start=round_start,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
            )
        elif kv_cache is not None and block_ids is not None and block_size is not None:
            paged_result = _select_keep_indices_paged_streaming(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                total_tokens=total_tokens,
                layer_idx=layer_idx,
                round_start=round_start,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                budget_total=budget_total,
            )
            return paged_result
        else:
            raise RuntimeError("missing scoring inputs for selector")

        k = min(int(budget_total), int(scores.shape[-1]))
        if k <= 0:
            return {"mode": "shared", "indices": []}
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head and scores.ndim == 3:
            topk = torch.topk(
                scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices[0]
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            if _sel_instr:
                try:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[TRITN-INSTR] E:select_keep_return layer=%d mode=per_head "
                        "keep_count=%d (per head) total_tokens=%d scores_shape=%s",
                        layer_idx, int(keep_per_head.shape[-1]), total_tokens,
                        tuple(scores.shape),
                    )
                except Exception:
                    pass
            # --- INSTRUMENTATION (Level H detailed dump, dense per-head) ---
            if _instr_verbose_allowed(req_id=req_id, layer_idx=layer_idx):
                try:
                    import logging as _lg
                    runtime_h = int(scores.shape[1]) if scores.ndim >= 2 else 1
                    sample_heads = tuple(sorted({0, runtime_h // 2, runtime_h - 1}))
                    prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                    window_count = _resolve_effective_recent_count(total_tokens)
                    tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                    dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                    keep_dyn_actual = max(0, int(keep_per_head.shape[-1]) - prefill_pinned - tail_pinned)
                    indices_dump = _format_index_sample(
                        keep_per_head,
                        head_count=runtime_h,
                        sample_heads=sample_heads,
                        max_indices=12,
                    )
                    score_q = _format_score_quantiles(scores)
                    _lg.getLogger(__name__).info(
                        "[TRITN-INSTR] H:selector_topk_inside layer=%d req=%s mode=dense_per_head "
                        "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                        "dynamic_total=%d keep_per_head=%d keep_dyn_actual=%d "
                        "score_q=[%s] sample_keep=[%s]",
                        layer_idx, req_id, total_tokens, k,
                        prefill_pinned, tail_pinned, dynamic_total,
                        int(keep_per_head.shape[-1]), keep_dyn_actual,
                        score_q, indices_dump,
                    )
                except Exception:
                    pass
            return {"mode": "per_head", "indices": keep_per_head}

        scores_agg = scores
        if scores_agg.ndim == 3:
            scores_agg = scores_agg.max(dim=1).values
        selected = torch.topk(
            scores_agg,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices[0]
        keep = torch.sort(selected).values.contiguous()
        if _sel_instr:
            try:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "[TRITN-INSTR] E:select_keep_return layer=%d mode=shared keep_count=%d "
                    "total_tokens=%d",
                    layer_idx, int(keep.numel()), total_tokens,
                )
            except Exception:
                pass
        # --- INSTRUMENTATION (Level H detailed dump, dense shared) ---
        if _instr_verbose_allowed(req_id=req_id, layer_idx=layer_idx):
            try:
                import logging as _lg
                prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                window_count = _resolve_effective_recent_count(total_tokens)
                tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                indices_dump = _format_index_sample(
                    keep.unsqueeze(0) if keep.ndim == 1 else keep,
                    head_count=1,
                    sample_heads=(0,),
                    max_indices=16,
                )
                score_q = _format_score_quantiles(scores_agg)
                _lg.getLogger(__name__).info(
                    "[TRITN-INSTR] H:selector_topk_inside layer=%d req=%s mode=dense_shared "
                    "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                    "dynamic_total=%d keep_count=%d score_q=[%s] sample_keep=[%s]",
                    layer_idx, req_id, total_tokens, k,
                    prefill_pinned, tail_pinned, dynamic_total,
                    int(keep.numel()), score_q, indices_dump,
                )
            except Exception:
                pass
        return {"mode": "shared", "indices": keep}

    def _select_keep_indices_for_group_per_head(
        *,
        layer_inputs: list[tuple[int, torch.Tensor]] | None = None,
        layer_input_iter: Callable[[], Iterable[tuple[int, torch.Tensor]]] | None = None,
        layer_kv_iter: Callable[
            [],
            Iterable[tuple[int, torch.Tensor, list[int] | torch.Tensor, int]],
        ]
        | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        round_start: int,
        budget_total: int,
        req_id: str | None = None,
    ) -> dict[str, Any] | None:
        if requested_pruning_mode != "per_head":
            return None
        if per_head_semantics != "hf_aligned_global_per_head":
            return None
        if total_tokens <= budget_total:
            head_count = 0
            if layer_inputs:
                head_count = int(layer_inputs[0][1].shape[1])
            elif layer_input_iter is not None:
                first_item = next(iter(layer_input_iter()), None)
                if first_item is not None:
                    head_count = int(first_item[1].shape[1])
            elif layer_kv_iter is not None:
                first_item = next(iter(layer_kv_iter()), None)
                if first_item is not None:
                    # 4D Ascend vs 5D CUDA — use the layout-aware helper
                    head_count = _resolve_runtime_heads(first_item[1])
            if head_count <= 0:
                return {"mode": "per_head", "indices": []}
            all_indices = torch.arange(
                total_tokens,
                dtype=torch.long,
                device=(
                    layer_inputs[0][1].device
                    if layer_inputs
                    else (
                        first_item[1].device
                        if first_item is not None
                        else torch.device("cpu")
                    )
                ),
            )
            return {
                "mode": "per_head",
                "indices": all_indices.unsqueeze(0).expand(head_count, -1).contiguous(),
            }
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None
        if layer_kv_iter is not None:
            iter_inputs = layer_kv_iter()
            iter_mode = "paged"
        elif layer_input_iter is not None:
            iter_inputs = layer_input_iter()
            iter_mode = "dense_iter"
        else:
            iter_inputs = layer_inputs or []
            iter_mode = "dense_list"
        if not iter_inputs:
            return None

        if iter_mode == "paged":
            group_agg_mode = os.environ.get(
                "TRIATTN_RUNTIME_DEBUG_GROUP_PERHEAD_AGG_MODE",
                "mean",
            ).strip().lower()
            if group_agg_mode not in {"mean", "max"}:
                group_agg_mode = "mean"
            layer_entries = list(iter_inputs)
            if not layer_entries:
                return None
            k = min(budget_total, total_tokens)
            if k <= 0:
                return {"mode": "per_head", "indices": []}
            prepared_layers: list[dict[str, Any]] = []
            for layer_idx, kv_cache, block_ids, layer_block_size in layer_entries:
                runtime_heads = _resolve_runtime_heads(kv_cache)
                (
                    score_head_stats,
                    score_freq_scale_sq,
                    use_hf_group_max,
                    group_size,
                ) = _resolve_layer_score_inputs(
                    layer_idx=layer_idx,
                    runtime_heads=runtime_heads,
                )
                prepared_layers.append(
                    {
                        "layer_idx": layer_idx,
                        "kv_cache": kv_cache,
                        "block_ids": block_ids,
                        "block_size": layer_block_size,
                        "runtime_heads": runtime_heads,
                        "score_head_stats": score_head_stats,
                        "score_freq_scale_sq": score_freq_scale_sq,
                        "use_hf_group_max": use_hf_group_max,
                        "group_size": group_size,
                    }
                )

            prepared_layer_indices = [int(entry["layer_idx"]) for entry in prepared_layers]

            min_block_size = min(entry["block_size"] for entry in prepared_layers)
            chunk_tokens = _score_chunk_tokens(min_block_size, total_tokens)
            norm_stats: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(prepared_layers)
            raw_scores_cache_by_layer: list[list[torch.Tensor] | None] = [None] * len(prepared_layers)
            if config.sparse_normalize_scores:
                eps = 1e-8
                for layer_pos, entry in enumerate(prepared_layers):
                    sum_vec: torch.Tensor | None = None
                    sumsq_vec: torch.Tensor | None = None
                    count = 0
                    layer_raw_scores: list[torch.Tensor] = []
                    start = 0
                    while start < total_tokens:
                        curr_tokens = min(chunk_tokens, total_tokens - start)
                        keys_chunk = gather_request_k_dense_range(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            start_token=start,
                            num_tokens=curr_tokens,
                        )
                        raw_scores = _compute_layer_scores_raw(
                            keys_dense=keys_chunk,
                            score_head_stats=entry["score_head_stats"],
                            score_freq_scale_sq=entry["score_freq_scale_sq"],
                            use_hf_group_max=entry["use_hf_group_max"],
                            group_size=entry["group_size"],
                            round_start=round_start,
                        )[0]
                        layer_raw_scores.append(raw_scores)
                        raw_fp32 = raw_scores.to(dtype=torch.float32)
                        chunk_sum = raw_fp32.sum(dim=-1)
                        chunk_sumsq = (raw_fp32 * raw_fp32).sum(dim=-1)
                        if sum_vec is None:
                            sum_vec = chunk_sum
                            sumsq_vec = chunk_sumsq
                        else:
                            sum_vec = sum_vec + chunk_sum
                            sumsq_vec = sumsq_vec + chunk_sumsq
                        count += curr_tokens
                        start += curr_tokens

                    if (
                        sum_vec is None
                        or sumsq_vec is None
                        or count <= 0
                    ):
                        return None
                    mean = sum_vec / float(count)
                    if count > 1:
                        var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
                    else:
                        var = torch.zeros_like(mean)
                    var = torch.clamp(var, min=0.0)
                    std = torch.sqrt(var)
                    std_safe = torch.where(std < eps, torch.ones_like(std), std)
                    norm_stats[layer_pos] = (
                        mean,
                        std_safe,
                    )
                    raw_scores_cache_by_layer[layer_pos] = layer_raw_scores

            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
            start = 0
            chunk_idx = 0
            while start < total_tokens:
                curr_tokens = min(chunk_tokens, total_tokens - start)
                chunk_guard_mask = _build_token_guard_mask(
                    start_token=start,
                    num_tokens=curr_tokens,
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    device=prepared_layers[0]["kv_cache"].device,
                )
                chunk_agg: torch.Tensor | None = None
                layer_count = 0
                for layer_pos, entry in enumerate(prepared_layers):
                    layer_raw_cache = raw_scores_cache_by_layer[layer_pos]
                    if layer_raw_cache is not None and chunk_idx < len(layer_raw_cache):
                        chunk_scores = layer_raw_cache[chunk_idx].unsqueeze(0)
                    else:
                        keys_chunk = gather_request_k_dense_range(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            start_token=start,
                            num_tokens=curr_tokens,
                        )
                        chunk_scores = _compute_layer_scores_raw(
                            keys_dense=keys_chunk,
                            score_head_stats=entry["score_head_stats"],
                            score_freq_scale_sq=entry["score_freq_scale_sq"],
                            use_hf_group_max=entry["use_hf_group_max"],
                            group_size=entry["group_size"],
                            round_start=round_start,
                        )
                    if config.sparse_normalize_scores:
                        mean, std_safe = norm_stats[layer_pos] or (None, None)
                        if mean is None or std_safe is None:
                            return None
                        chunk_scores = (chunk_scores - mean.view(1, -1, 1)) / std_safe.view(1, -1, 1)
                    if chunk_guard_mask is not None:
                        chunk_scores = chunk_scores.masked_fill(
                            chunk_guard_mask.view(1, 1, -1),
                            float("inf"),
                        )
                    if entry["use_hf_group_max"]:
                        chunk_scores = _reduce_grouped_head_scores(
                            scores=chunk_scores,
                            runtime_heads=entry["runtime_heads"],
                            group_size=entry["group_size"],
                            aggregate_mode="max",
                        )
                    if chunk_scores.ndim != 3:
                        raise RuntimeError(
                            f"unexpected_score_rank_for_per_head:{chunk_scores.ndim}"
                        )
                    layer_scores = chunk_scores[0]
                    if chunk_agg is None:
                        chunk_agg = layer_scores.clone()
                    else:
                        if group_agg_mode == "max":
                            chunk_agg = torch.maximum(chunk_agg, layer_scores)
                        else:
                            chunk_agg.add_(layer_scores)
                    layer_count += 1

                if chunk_agg is None or layer_count <= 0:
                    return None
                chunk_final = (
                    chunk_agg
                    if group_agg_mode == "max"
                    else chunk_agg.div(float(layer_count))
                )
                cand_k = min(k, int(chunk_final.shape[-1]))
                cand = torch.topk(
                    chunk_final,
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
                start += curr_tokens
                chunk_idx += 1

            if best_indices is None:
                return None
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            # --- INSTRUMENTATION (Level H detailed dump, group per-head paged) ---
            _emit_group_dump = _instr_verbose_allowed(
                req_id=req_id, layer_idx=-1
            )
            if _emit_group_dump:
                try:
                    import logging as _lg
                    runtime_h = int(keep_per_head.shape[0]) if keep_per_head.ndim >= 1 else 1
                    sample_heads = tuple(sorted({0, runtime_h // 2, runtime_h - 1}))
                    prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                    window_count = _resolve_effective_recent_count(total_tokens)
                    tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                    dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                    keep_dyn_actual = max(0, int(keep_per_head.shape[-1]) - prefill_pinned - tail_pinned)
                    indices_dump = _format_index_sample(
                        keep_per_head,
                        head_count=runtime_h,
                        sample_heads=sample_heads,
                        max_indices=12,
                    )
                    score_q = _format_score_quantiles(best_scores if best_scores is not None else keep_per_head)
                    _lg.getLogger(__name__).info(
                        "[TRITN-INSTR] H:selector_topk_inside layer=GROUP req=%s mode=group_paged_per_head "
                        "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                        "dynamic_total=%d keep_per_head=%d keep_dyn_actual=%d "
                        "agg_mode=%s group_layers=%s score_q=[%s] sample_keep=[%s]",
                        req_id, total_tokens, k,
                        prefill_pinned, tail_pinned, dynamic_total,
                        int(keep_per_head.shape[-1]), keep_dyn_actual,
                        group_agg_mode, list(prepared_layer_indices)[:8],
                        score_q, indices_dump,
                    )
                except Exception:
                    pass
            return {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": group_agg_mode,
                "debug_group_layer_indices": prepared_layer_indices,
                "debug_recent_count": _resolve_effective_recent_count(total_tokens),
            }
        else:
            aggregated_scores: torch.Tensor | None = None
            layer_count = 0
            dense_layer_indices: list[int] = []
            for layer_idx, keys_dense in iter_inputs:
                dense_layer_indices.append(int(layer_idx))
                scores = _compute_layer_scores(
                    keys_dense=keys_dense,
                    layer_idx=layer_idx,
                    round_start=round_start,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                )
                if scores.ndim != 3:
                    raise RuntimeError(
                        f"unexpected_score_rank_for_per_head:{scores.ndim}"
                    )
                layer_scores = scores[0]
                if aggregated_scores is None:
                    aggregated_scores = layer_scores.clone()
                else:
                    aggregated_scores.add_(layer_scores)
                layer_count += 1
            if aggregated_scores is None or layer_count <= 0:
                return None
            aggregated_scores.div_(layer_count)
            k = min(budget_total, aggregated_scores.shape[-1])
            if k <= 0:
                return {"mode": "per_head", "indices": []}

            topk = torch.topk(
                aggregated_scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            # --- INSTRUMENTATION (Level H detailed dump, group per-head dense) ---
            if _instr_verbose_allowed(req_id=req_id, layer_idx=-1):
                try:
                    import logging as _lg
                    runtime_h = int(keep_per_head.shape[0]) if keep_per_head.ndim >= 1 else 1
                    sample_heads = tuple(sorted({0, runtime_h // 2, runtime_h - 1}))
                    prefill_pinned = max(0, int(prefill_len)) if protect_prefill else 0
                    window_count = _resolve_effective_recent_count(total_tokens)
                    tail_pinned = min(int(window_count), max(0, int(total_tokens) - prefill_pinned))
                    dynamic_total = max(0, int(total_tokens) - prefill_pinned - tail_pinned)
                    keep_dyn_actual = max(0, int(keep_per_head.shape[-1]) - prefill_pinned - tail_pinned)
                    indices_dump = _format_index_sample(
                        keep_per_head,
                        head_count=runtime_h,
                        sample_heads=sample_heads,
                        max_indices=12,
                    )
                    score_q = _format_score_quantiles(aggregated_scores)
                    _lg.getLogger(__name__).info(
                        "[TRITN-INSTR] H:selector_topk_inside layer=GROUP req=%s mode=group_dense_per_head "
                        "total_tokens=%d budget=%d prefill_pinned=%d tail_pinned=%d "
                        "dynamic_total=%d keep_per_head=%d keep_dyn_actual=%d "
                        "agg_mode=mean group_layers=%s score_q=[%s] sample_keep=[%s]",
                        req_id, total_tokens, k,
                        prefill_pinned, tail_pinned, dynamic_total,
                        int(keep_per_head.shape[-1]), keep_dyn_actual,
                        list(dense_layer_indices)[:8],
                        score_q, indices_dump,
                    )
                except Exception:
                    pass
            return {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": "mean",
                "debug_group_layer_indices": dense_layer_indices,
                "debug_recent_count": _resolve_effective_recent_count(total_tokens),
            }

    setattr(_select_keep_indices, "_supports_paged", True)
    setattr(_select_keep_indices_for_group_per_head, "_supports_paged_group", True)
    return _select_keep_indices, _select_keep_indices_for_group_per_head, "enabled"
