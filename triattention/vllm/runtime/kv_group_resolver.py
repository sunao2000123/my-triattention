"""Resolve vLLM KV cache tensors grouped by kv-cache group/layer for TriAttention runtime."""

from __future__ import annotations

import re
from typing import Any

import torch

from .kv_compaction import register_kv_layout_axis_hint


def infer_layer_idx(layer_name: str, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", layer_name)
    if matches:
        return int(matches[-1])
    return fallback_idx


def _unwrap_kv_cache_entry(entry: Any) -> torch.Tensor | None:
    """Unwrap the per-layer ``kv_cache`` attribute to a single K-cache tensor.

    vLLM upstream ``bind_kv_cache`` writes ``layer.kv_cache = [kv_cache]`` —
    a list with one element. On vllm-ascend v0.18.0 the inner value is a
    tuple ``(k_cache, v_cache)`` (see
    ``vllm_ascend/worker/model_runner_v1.py:3017-3023``), so the
    attribute looks like ``[(k_tensor, v_tensor)]``.

    Some legacy backends store ``layer.kv_cache = [K, V]`` (a flat list of
    two tensors); others store ``layer.kv_cache = tensor`` (bare).

    This helper returns the K-cache tensor in all four layouts, or
    ``None`` if the entry cannot be interpreted.
    """
    if isinstance(entry, torch.Tensor):
        return entry
    if isinstance(entry, tuple):
        # (K, V) — Ascend pattern, or (K, V, aux1, aux2) — Ascend DSA
        for item in entry:
            if isinstance(item, torch.Tensor):
                return item
        return None
    if isinstance(entry, list):
        if not entry:
            return None
        first = entry[0]
        if isinstance(first, torch.Tensor):
            # CUDA upstream pattern: [tensor]
            return first
        if isinstance(first, tuple):
            # Ascend pattern: [(K, V)]
            for item in first:
                if isinstance(item, torch.Tensor):
                    return item
            return None
        return None
    return None


def _unwrap_kv_cache_pair(entry: Any) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Unwrap the per-layer ``kv_cache`` attribute to ``(K, V)`` pair.

    Returns ``(K_tensor, V_tensor)`` when both are present, or just
    ``(K_tensor, None)`` when only a single tensor is exposed (CUDA upstream
    layout, where K/V are fused in one 5D tensor).

    Returns ``None`` if the entry cannot be interpreted.
    """
    if isinstance(entry, torch.Tensor):
        return entry, None
    if isinstance(entry, tuple):
        # (K, V) — Ascend pattern; first two tensor slots are K and V.
        tensors = [item for item in entry if isinstance(item, torch.Tensor)]
        if not tensors:
            return None
        k = tensors[0]
        v = tensors[1] if len(tensors) > 1 else None
        return k, v
    if isinstance(entry, list):
        if not entry:
            return None
        first = entry[0]
        if isinstance(first, torch.Tensor):
            # CUDA upstream pattern: [tensor] — fused K/V
            return first, None
        if isinstance(first, tuple):
            # Ascend pattern: [(K, V, ...)]
            tensors = [item for item in first if isinstance(item, torch.Tensor)]
            if not tensors:
                return None
            k = tensors[0]
            v = tensors[1] if len(tensors) > 1 else None
            return k, v
        return None
    return None


def _infer_kv_axis_from_group_backend(base_runner: Any, gid: int) -> int | None:
    attn_groups = getattr(base_runner, "attn_groups", None)
    if not isinstance(attn_groups, (list, tuple)):
        return None
    if gid < 0 or gid >= len(attn_groups):
        return None
    group = attn_groups[gid]
    backend = getattr(group, "backend", None)
    if backend is None:
        return None

    backend_cls = backend if isinstance(backend, type) else backend.__class__
    get_kv_cache_shape = getattr(backend_cls, "get_kv_cache_shape", None)
    if callable(get_kv_cache_shape):
        try:
            # Probe with num_blocks=3 to avoid (2, 2, ...) ambiguity.
            shape = tuple(
                int(x)
                for x in get_kv_cache_shape(
                    3,   # num_blocks
                    16,  # block_size (vLLM backends require multiple of 16)
                    1,   # num_kv_heads
                    1,   # head_size
                )
            )
            if len(shape) >= 2:
                dim0_is_kv = shape[0] == 2
                dim1_is_kv = shape[1] == 2
                if dim0_is_kv ^ dim1_is_kv:
                    return 0 if dim0_is_kv else 1
        except Exception:
            pass

    # Conservative fallback for fake backends in tests or unknown vLLM variants.
    module_name = str(getattr(backend_cls, "__module__", ""))
    cls_name = str(getattr(backend_cls, "__name__", ""))
    ident = f"{module_name}.{cls_name}".lower()
    if "flash_attn" in ident:
        return 0
    if "triton_attn" in ident:
        return 1
    return None


def resolve_group_tensors(
    base_runner: Any,
) -> dict[int, list[tuple[int, torch.Tensor, torch.Tensor | None]]]:
    """Resolve kv cache tensors for each kv cache group.

    Returns:
        gid -> list of ``(layer_idx, k_cache_tensor, v_cache_tensor_or_None)``

        The V tensor is ``None`` for upstream CUDA layouts where K and V
        are fused in one 5D tensor (``[2, num_blocks, block_size, H, D]``).
        For vllm-ascend v0.18.0 the V tensor is a separate 4D tensor
        (``[num_blocks, block_size, H, D]``); the compactor applies the
        same permutation to both K and V in lock-step so the attention
        backend reads consistent data after reclaim.
    """
    group_tensors: dict[int, list[tuple[int, torch.Tensor, torch.Tensor | None]]] = {}

    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    compilation_config = getattr(base_runner, "compilation_config", None)
    static_forward_context = (
        getattr(compilation_config, "static_forward_context", None)
        if compilation_config is not None
        else None
    )

    if kv_cache_config is None or not isinstance(static_forward_context, dict):
        fallback = getattr(base_runner, "kv_caches", None)
        if isinstance(fallback, list):
            tensors: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
            for idx, entry in enumerate(fallback):
                pair = _unwrap_kv_cache_pair(entry)
                if pair is not None:
                    k_tensor, v_tensor = pair
                    tensors.append((idx, k_tensor, v_tensor))
            if tensors:
                group_tensors[0] = tensors
        return group_tensors

    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return group_tensors

    for gid, group in enumerate(kv_cache_groups):
        layer_names = getattr(group, "layer_names", None)
        if not isinstance(layer_names, (list, tuple)):
            continue
        tensors: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
        seen_ptrs: set[int] = set()
        for local_idx, layer_name in enumerate(layer_names):
            layer = static_forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache_attr = getattr(layer, "kv_cache", None)
            pair = _unwrap_kv_cache_pair(kv_cache_attr)
            if pair is None:
                continue
            k_tensor, v_tensor = pair
            ptr = k_tensor.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors.append(
                (
                    infer_layer_idx(
                        layer_name=layer_name,
                        layer_obj=layer,
                        fallback_idx=local_idx,
                    ),
                    k_tensor,
                    v_tensor,
                )
            )
        if tensors:
            kv_axis_hint = _infer_kv_axis_from_group_backend(base_runner=base_runner, gid=gid)
            if kv_axis_hint is not None:
                for _layer_idx, k_tensor, _v_tensor in tensors:
                    try:
                        register_kv_layout_axis_hint(k_tensor, kv_axis_hint)
                    except ValueError:
                        # Best effort registration only; compaction path will fail-fast if
                        # an ambiguous layout cannot be safely disambiguated.
                        pass
            group_tensors[gid] = tensors
    return group_tensors
