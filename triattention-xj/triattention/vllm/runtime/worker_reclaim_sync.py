"""Worker-side block-table reclaim synchronization helpers for TriAttention runtime."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from vllm.logger import logger

from .logging_control import runtime_logging_enabled

_DEBUG_DISABLE_LOGGED = False

# === BUG-RECL 总开关辅助函数（断点排查专用） ===
# 设 TRIATTN_BUG_RECL_DEBUG=1 启用；默认 0 关闭
def _bug_recl_enabled() -> bool:
    return os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
# === BUG-RECL 总开关辅助函数结束 ===


def _event_reclaim_groups(event: dict[str, Any]) -> tuple[str, dict[int, dict[str, Any]]]:
    block_reclaim = event.get("block_reclaim")
    if not isinstance(block_reclaim, dict):
        return "truncate_tail", {}
    mode = block_reclaim.get("mode")
    if mode not in {"truncate_tail", "remap_tail"}:
        mode = "truncate_tail"
    groups = block_reclaim.get("groups")
    if not isinstance(groups, list):
        return str(mode), {}
    by_gid: dict[int, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = group.get("gid")
        if isinstance(gid, int):
            by_gid[gid] = group
    return str(mode), by_gid


def _block_ids_after(group: dict[str, Any] | None) -> list[int] | None:
    if not isinstance(group, dict):
        return None
    block_ids_after = group.get("block_ids_after")
    if not isinstance(block_ids_after, list):
        return None
    if not all(isinstance(block_id, int) for block_id in block_ids_after):
        return None
    if len(set(block_ids_after)) != len(block_ids_after):
        return None
    return list(block_ids_after)


def _clear_table_row_tail(table: Any, req_index: int, used_blocks: int) -> bool:
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if not isinstance(block_table_np, np.ndarray):
        return False
    if block_table_np.ndim != 2:
        return False
    if req_index < 0 or req_index >= int(block_table_np.shape[0]):
        return False
    start = max(0, min(int(used_blocks), int(block_table_np.shape[1])))
    if start >= int(block_table_np.shape[1]):
        return True
    block_table_np[req_index, start:] = 0
    return True


def _row_block_count(table: Any, req_index: int, fallback: int) -> int:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if isinstance(num_blocks_per_row, np.ndarray):
        if 0 <= req_index < int(num_blocks_per_row.shape[0]):
            return int(num_blocks_per_row[req_index])
    return int(fallback)


def _rewrite_table_row(table: Any, req_index: int, block_ids: list[int]) -> bool:
    add_row = getattr(table, "add_row", None)
    if callable(add_row):
        add_row(block_ids, req_index)
        _clear_table_row_tail(
            table,
            req_index,
            _row_block_count(table, req_index, len(block_ids)),
        )
        return True

    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if not isinstance(num_blocks_per_row, np.ndarray):
        return False
    if not isinstance(block_table_np, np.ndarray):
        return False
    if len(block_ids) > block_table_np.shape[1]:
        return False
    block_table_np[req_index, :] = 0
    block_table_np[req_index, :len(block_ids)] = block_ids
    num_blocks_per_row[req_index] = len(block_ids)
    return True


def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    """Apply reclaim shrink to worker-side block tables after compression.

    In vLLM V1, the block table lives at ``base_runner.input_batch.block_table``
    and tracks per-request block counts in ``num_blocks_per_row``.  After
    compression compacts KV cache data into fewer blocks, we must update these
    counters so that subsequent ``append_row()`` calls start from the correct
    offset and don't overflow the max-blocks-per-request limit.
    """
    global _DEBUG_DISABLE_LOGGED
    if os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        if not _DEBUG_DISABLE_LOGGED and runtime_logging_enabled():
            logger.info("TriAttention worker reclaim sync disabled by debug env")
            _DEBUG_DISABLE_LOGGED = True
        return

    if not isinstance(events, list) or not events:
        # === BUG-RECL 断点 1（LE-W-001）：函数入口与短路检查 ===
        if _bug_recl_enabled():
            n_events_total = len(events) if isinstance(events, list) else 0
            n_applied = sum(1 for e in events if isinstance(e, dict) and e.get("status") == "applied") if isinstance(events, list) else 0
            env_disable = os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0") == "1"
            print(f"BUG-RECL [LE-W-001] enter apply_worker_block_reclaim_events n_events={n_events_total} n_applied={n_applied} env_disable={env_disable}", flush=True)
        # === BUG-RECL 断点 1 结束 ===
        return

    # === BUG-RECL 断点 1（LE-W-001）：函数入口与短路检查（events 非空分支） ===
    if _bug_recl_enabled():
        n_events_total = len(events) if isinstance(events, list) else 0
        n_applied = sum(1 for e in events if isinstance(e, dict) and e.get("status") == "applied") if isinstance(events, list) else 0
        env_disable = os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0") == "1"
        print(f"BUG-RECL [LE-W-001] enter apply_worker_block_reclaim_events n_events={n_events_total} n_applied={n_applied} env_disable={env_disable}", flush=True)
    # === BUG-RECL 断点 1 结束 ===

    # Resolve the vLLM V1 block table.
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        if getattr(base_runner, "block_tables", None) is not None:
            # Formal V2 runner manages block tables directly on base_runner
            # rather than on input_batch. In that path, hook-side compaction
            # already updates the canonical tables, so there is nothing for the
            # old V1 reclaim-sync helper to do here.
            return
        logger.warning(
            "TriAttention worker reclaim: block table not found. "
            "input_batch=%s block_table=%s",
            type(input_batch).__name__ if input_batch else None,
            type(block_table_obj).__name__ if block_table_obj else None,
        )
        return

    # Resolve request-id → row-index mapping.
    # In vLLM V1, req_id_to_index lives on input_batch, and request states
    # (with block_ids) live in base_runner.requests.
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        logger.warning(
            "TriAttention worker reclaim: req_id_to_index not found on input_batch. "
            "input_batch=%s",
            type(input_batch).__name__ if input_batch else None,
        )
        return

    # The block table may be a single BlockTable (with num_blocks_per_row) or
    # a MultiGroupBlockTable (with .block_tables list of per-group BlockTables).
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, list):
        # MultiGroupBlockTable
        tables = inner_tables
    else:
        # Single BlockTable
        tables = [block_table_obj]

    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 16))
    if block_size <= 0:
        block_size = 16

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if req_id is None:
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        cache_len_after = event.get("cache_len_after")
        if not isinstance(cache_len_after, int) or cache_len_after <= 0:
            continue

        details = event.get("details")
        retained_cache_len = (
            details.get("retained_cache_len")
            if isinstance(details, dict)
            else None
        )
        if not isinstance(retained_cache_len, int) or retained_cache_len <= 0:
            retained_cache_len = cache_len_after
        required_blocks = (retained_cache_len + block_size - 1) // block_size
        reclaim_mode, groups_by_gid = _event_reclaim_groups(event)

        # === BUG-RECL 断点 2（LE-W-002）：per-event 解析 ===
        if _bug_recl_enabled():
            print(f"BUG-RECL [LE-W-002] event req={req_id} req_index={req_index} cache_len_after={cache_len_after} retained_cache_len={retained_cache_len} required_blocks={required_blocks} block_size={block_size}", flush=True)
        # === BUG-RECL 断点 2 结束 ===

        for gid, table in enumerate(tables):
            num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
            if num_blocks_per_row is None:
                continue
            if not isinstance(num_blocks_per_row, np.ndarray):
                continue
            current = int(num_blocks_per_row[req_index])
            if reclaim_mode == "remap_tail":
                block_ids_after = _block_ids_after(groups_by_gid.get(gid))
                if block_ids_after is not None:
                    # === BUG-RECL 断点 4（LE-W-004）：remap_tail 校验与 rewrite 结果 ===
                    if _bug_recl_enabled():
                        _ba = block_ids_after
                        print(f"BUG-RECL [LE-W-004] remap_tail req={req_id} gid={gid} block_ids_after_valid={_ba is not None} n_block_ids_after={len(_ba) if _ba else 0} block_ids_after={(_ba or [])[:8]}", flush=True)
                    # === BUG-RECL 断点 4 结束 ===
                    if _rewrite_table_row(table, req_index, block_ids_after):
                        if runtime_logging_enabled():
                            logger.debug(
                                "TriAttention worker remap: req=%s gid=%d "
                                "num_blocks %d -> %d",
                                req_id, gid, current, len(block_ids_after),
                            )
                    else:
                        logger.warning(
                            "TriAttention worker remap failed: req=%s gid=%d "
                            "table=%s",
                            req_id, gid, type(table).__name__,
                        )
                    continue
            # === BUG-RECL 断点 3（LE-W-003）：truncate_tail vs remap_tail 分支点 ===
            if _bug_recl_enabled():
                print(f"BUG-RECL [LE-W-003] before-branch req={req_id} gid={gid} reclaim_mode={reclaim_mode} current_num_blocks={current} required_blocks={required_blocks} will_shrink={current > required_blocks}", flush=True)
            # === BUG-RECL 断点 3 结束 ===
            if current > required_blocks:
                num_blocks_per_row[req_index] = required_blocks
                if runtime_logging_enabled():
                    logger.debug(
                        "TriAttention worker reclaim: req=%s num_blocks %d -> %d "
                        "(cache_len_after=%d block_size=%d)",
                        req_id, current, required_blocks, cache_len_after, block_size,
                    )
            _clear_table_row_tail(
                table,
                req_index,
                _row_block_count(table, req_index, min(current, required_blocks)),
            )

        # Also truncate req_state.block_ids (CPU-side block tracking).
        # In vLLM V1, per-request state lives in base_runner.requests dict.
        requests_dict = getattr(base_runner, "requests", None)
        if isinstance(requests_dict, dict):
            req_state = requests_dict.get(req_id)
            if req_state is not None:
                block_ids_attr = getattr(req_state, "block_ids", None)
                if isinstance(block_ids_attr, (list, tuple)):
                    if reclaim_mode == "remap_tail":
                        rewritten_groups: list[Any] = []
                        changed = False
                        for gid, group_blocks in enumerate(block_ids_attr):
                            block_ids_after = _block_ids_after(groups_by_gid.get(gid))
                            if block_ids_after is None:
                                rewritten_groups.append(group_blocks)
                                continue
                            if isinstance(group_blocks, tuple):
                                rewritten_groups.append(tuple(block_ids_after))
                            else:
                                rewritten_groups.append(list(block_ids_after))
                            changed = True
                        if changed:
                            if isinstance(block_ids_attr, tuple):
                                setattr(req_state, "block_ids", tuple(rewritten_groups))
                            else:
                                setattr(req_state, "block_ids", rewritten_groups)
                            # === BUG-RECL 断点 5b（LE-W-005b）：视图 3 remap_tail 写入结果 ===
                            if _bug_recl_enabled():
                                try:
                                    _rb = getattr(req_state, "block_ids", None)
                                    _len = [len(g) for g in _rb] if isinstance(_rb, (list, tuple)) else None
                                    print(f"BUG-RECL [LE-W-005b] view3-after-remap req={req_id} container={type(_rb).__name__} per_group_len={_len}", flush=True)
                                except Exception as _e:
                                    print(f"BUG-RECL [LE-W-005b] view3-read-error req={req_id} err={_e}", flush=True)
                            # === BUG-RECL 断点 5b 结束 ===
                    else:
                        for group_blocks in block_ids_attr:
                            if (
                                isinstance(group_blocks, list)
                                and len(group_blocks) > required_blocks
                            ):
                                del group_blocks[required_blocks:]
                        # === BUG-RECL 断点 5a（LE-W-005a）：视图 3 truncate_tail 写入结果 ===
                        if _bug_recl_enabled():
                            try:
                                _len = [len(g) for g in block_ids_attr] if isinstance(block_ids_attr, (list, tuple)) else None
                                _container_type = type(block_ids_attr).__name__
                                print(f"BUG-RECL [LE-W-005a] view3-after-truncate req={req_id} container={_container_type} per_group_len={_len} required_blocks={required_blocks}", flush=True)
                            except Exception as _e:
                                print(f"BUG-RECL [LE-W-005a] view3-read-error req={req_id} err={_e}", flush=True)
                        # === BUG-RECL 断点 5a 结束 ===

    # === BUG-RECL 断点 6（LE-W-006）：视图 1+2+3 同步校验（函数末尾） ===
    if _bug_recl_enabled():
        try:
            _sync_report = []
            for _ev in (events if isinstance(events, list) else []):
                if not isinstance(_ev, dict) or _ev.get("status") != "applied":
                    continue
                _rid = _ev.get("req_id")
                if not isinstance(_rid, str):
                    continue
                _idx = req_id_to_index.get(_rid)
                if not isinstance(_idx, int):
                    continue
                _view1_per_gid = [int(t.num_blocks_per_row[_idx]) for t in tables if isinstance(getattr(t, "num_blocks_per_row", None), np.ndarray)]
                _req_state = getattr(base_runner, "requests", {}).get(_rid)
                _view3_lens = [len(g) for g in getattr(_req_state, "block_ids", [])] if _req_state is not None else None
                _view3_container = type(getattr(_req_state, "block_ids", None)).__name__ if _req_state is not None else "no_req_state"
                _sync_report.append({
                    "req": _rid, "view1_per_gid": _view1_per_gid,
                    "view3_container": _view3_container, "view3_lens": _view3_lens,
                })
            print(f"BUG-RECL [LE-W-006] final-sync {_sync_report}", flush=True)
        except Exception as _e:
            print(f"BUG-RECL [LE-W-006] final-sync-error err={_e}", flush=True)
    # === BUG-RECL 断点 6 结束 ===
