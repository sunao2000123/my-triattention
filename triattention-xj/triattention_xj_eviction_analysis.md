# TriAttention vLLM-Ascend 0.18.0 — 逻辑驱逐（Block Table 视图）失效问题静态分析

> 任务范围：基于 `triattention-xj/triattention`（含 `triattention/vllm/runtime/**`）与 `other_code/vllm-ascend-releases-v0.18.0`（含 `vllm_ascend/**`、依赖链上 `vllm/v1/core/**`、`vllm/v1/worker/**`）的 0.18.0 原生源码，**聚焦「逻辑驱逐」（即 TriAttention 通过 `worker_reclaim_sync.py` 等模块对 Block Table / `req_state.block_ids` / `num_blocks_per_row` / `_triattention_effective_kv_offset` 的视图层修改）失效**这一异常现象的根因候选点，并设计纯日志（`print`/logger）断点排查方案。
>
> **关注点重构**：本版本不再以 `block_pool.free_blocks` 物理释放为分析重心（这只能归还 ref_cnt==0 的 block，且 vLLM 内部已经按 ref_cnt 正确处理），而是**把 TriAttention 当作一个「逻辑驱逐器」**：它的「驱逐」语义是「让 vLLM 上游（Scheduler 的 `kv_cache_manager.allocate_slots`、NPU 的 `compute_slot_mapping`、NPU attention 的 `seq_len`）从此把该请求当作更短」，而不是「把 NPU 上的物理 block 还给 free_block_queue」。
>
> 重要约束：**仅做静态代码分析、逻辑推演、差异比对，禁止任何运行/调试/测试/验证**；只面向昇腾 NPU 适配分支；最小侵入原则，未定位前不修改任何业务代码。

---

## 模块一：当前代码 TriAttention「逻辑驱逐」完整逻辑与原理

下文以一次「典型场景：32k 输入 Token、保留 2k Top-K KV Token、`block_size=16`、默认 `kv_budget=2048`」的运行时调度为线索，全链路梳理「逻辑驱逐」代码执行流；所有文件名均位于 `triattention-xj`（即仓库 `/Users/sunao2000/triattention-xj/`）或 `triattention-xj/other_code/vllm-ascend-releases-v0.18.0/` 之内。

### 1.1 关键术语澄清（与物理回收的区分）

| 维度 | 物理回收（vLLM 自带） | 逻辑驱逐（TriAttention 引入） |
| --- | --- | --- |
| 释放对象 | `block_pool.blocks[block_id]`，依赖 `ref_cnt == 0` | **Block Table 视图**（`input_batch.block_table.num_blocks_per_row` + `block_table.np[row, :]`） |
| 调用方 | vLLM `KVCacheManager.free(request)` 链 | `triattention/vllm/runtime/worker_reclaim_sync.py:apply_worker_block_reclaim_events` |
| 触发位置 | `vllm/v1/core/kv_cache_manager.py:390 free()` | `triattention/vllm/runtime/runner.py:1171 _apply_worker_block_reclaim_events()` |
| 可见效果 | `kv_cache_manager.usage` 降低、`free_block_queue` 变长 | `num_blocks_per_row[req_idx]` 减小、`block_table.np[row, kept_n:]` 清零、`req_state.block_ids` 截断 |
| 关键依赖 | 没有任何 prefix-cache 共享 | `_triattention_effective_kv_offset` 必须被正确写入才能让上游 `allocate_slots` 知道「逻辑 token 比实际短」 |

> 关键事实 1：**逻辑驱逐是物理回收的前置必要条件**——若 `num_blocks_per_row` 没被截短，下次 `update_states` 中的 `block_table.append_row` 会把新分配的 block 续接在错误起点，物理水位（`kv_cache_usage`）上不去。
>
> 关键事实 2：**逻辑驱逐并不必然带来物理回收**——`block_table.np` 是「该请求占用了哪些 block」的逻辑映射；底层的 `block_pool.blocks[block_id]` 是否归还，取决于 `ref_cnt`。TriAttention 只动前者不动后者，物理回收仍由 vLLM 自己在 `coordinator.free(request_id)`（`vllm/v1/core/block_pool.py:409-423 free_blocks`）里完成。

### 1.2 整体分层与补丁面

TriAttention 对 vLLM 0.18.0 V1 路径采用 **运行时 Monkeypatch + 透明代理（Proxy）** 的方式接入，不修改任何 vLLM/Ascend 上游类身份。关键接入面（与本任务强相关的「逻辑驱逐」链路用 ★ 标出）：

- ★ **Plugin 入口**：`triattention/vllm/plugin.py:60-110 register_triattention_backend()`，由 vLLM 通过 `vllm.platform_plugins` 自动发现并加载；`triattention/vllm/runtime/integration_monkeypatch.py:687-828 install_vllm_integration_monkeypatches()` 完成下列替换：
  - ★ `vllm.v1.core.sched.scheduler.Scheduler.__init__ / schedule / update_from_output` → `_patched_scheduler_init / _patched_scheduler_schedule / _patched_scheduler_update_from_output`
  - ★ `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots` → `_patched_kv_cache_allocate_slots`（带 `delay_cache_blocks` 兼容）
  - `vllm.v1.engine.core.EngineCore.step_with_batch_queue` → `_patched_engine_core_step_with_batch_queue`（按压缩边界节流异步流水线）
  - `vllm_ascend.worker.worker.NPUWorker.init_device / execute_model` → `_patched_ascend_worker_init_device / _patched_ascend_worker_execute_model`（用于提前挂载 Proxy）
  - `vllm_ascend.ascend_forward_context.set_ascend_forward_context` 与 `vllm_ascend.worker.model_runner_v1.set_ascend_forward_context`（用于 NPU Graph 模式守卫）
  - `vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory / check_enough_kv_cache_memory` 替换为「带警告的放宽校验」（因为压缩后实际占用小于 `max_model_len` 推算值）
- **Worker 入口**：`triattention/vllm/runtime/worker.py:200-291 TriAttentionWorker`（`__getattr__` 透传）只在第一次满足 `should_install_triattention_runner_proxy()` 条件时，才把 `self.model_runner` 替换为 `TriAttentionModelRunner(base_runner, config)`；NPU Worker 的 `init_device` 阶段会通过 `early_install_proxy_on_ascend=True` 提前挂载。
- **Runner 入口**：`triattention/vllm/runtime/runner.py:290-1290 TriAttentionModelRunner`，对原生 `base_runner.execute_model` 包裹「生命周期注册 → 信号消费 → 压缩执行 → **Block Table 逻辑驱逐（worker_reclaim_sync）** → 输入 Patch → 调用 `base_runner.execute_model` → 事件回传」流水；并通过 `__getattr__` 完全透传未重写的方法/属性给 `base_runner`（这是 vLLM-Ascend `NPUModelRunner` 大量子组件仍能正常工作的关键）。
- **Hook 注册**：`triattention/vllm/runtime/hook_impl.py:41-369 make_runner_compression_hook()` 在挂载 Proxy 时调用 `install_runner_compression_hook()` 把 `base_runner.triattention_apply_compression` 设为实际的 `_hook` 函数（后续 Worker 内 Selector 调用、Runner 调度都通过此 hook 触发）。

### 1.3 逻辑驱逐的「五个视图」与一次完整生命周期

TriAttention 的「逻辑驱逐」涉及 5 个相互独立的「视图」对象，必须全部同步才能让上游相信「该请求逻辑上变短了」：

| # | 视图对象 | 物理意义 | 写入点 | 读取点 |
| - | - | - | - | - |
| 1 | `input_batch.block_table.num_blocks_per_row[req_idx]` | 该请求在 NPU 端 Block Table 的行长度（逻辑 block 数） | `worker_reclaim_sync.py:220-221`（truncate） / `worker_reclaim_sync.py:73-82 _rewrite_table_row`（remap） | `vllm_ascend/worker/block_table.py:97-101 append_row` 的 `start` 索引、`compute_slot_mapping` 的行容量 |
| 2 | `input_batch.block_table.block_table.np[req_idx, :]` | 该请求占用的物理 block_id 列表 | `worker_reclaim_sync.py:49-62 _clear_table_row_tail` / `worker_reclaim_sync.py:73-96 _rewrite_table_row` | `vllm_ascend/worker/block_table.py:120-186 compute_slot_mapping` |
| 3 | `base_runner.requests[req_id].block_ids` | CPU 侧 CachedRequestState 中该请求的 block 列表 | `worker_reclaim_sync.py:236-266` | vLLM `_update_states` 末尾 `block_ids.extend(new_ids)`（`vllm/v1/worker/gpu_model_runner.py:1248-1250`） |
| 4 | `request._triattention_effective_kv_offset` | `num_computed_tokens - cache_len_after` 的逻辑偏移 | `triattention/vllm/runtime/scheduler.py:710-714 / 877-881` 调 `kv_allocation_sync.py:35-58 update_request_effective_kv_offset` | `kv_allocation_sync.py:61-81 prepare_request_effective_num_computed` 在 `kv_cache_manager.allocate_slots` 入口被读取 |
| 5 | `req_state.block_ids`（仅在 hook 入口/出口被改） | Hook 自身的「下一次驱逐起点」基线 | `triattention/vllm/runtime/hook_group_pipeline.py:452-453` 与 `triattention/vllm/runtime/runner.py:232-256` | `triattention/vllm/runtime/hook_impl.py:127-141 resolve_hook_compaction_inputs` |

下面对每个视图在一次压缩事件中的「写入-读取-失效」生命周期做静态推演。

### 1.4 视图 1 + 2：`input_batch.block_table.num_blocks_per_row` 与 `block_table.np` 的截断/重映射（**worker_reclaim_sync.py 主体**）

`triattention/vllm/runtime/worker_reclaim_sync.py:apply_worker_block_reclaim_events` 是 **NPU 端 Block Table 逻辑驱逐的唯一执行点**。该函数在 `triattention/vllm/runtime/runner.py:1171 _apply_worker_block_reclaim_events` 处被调用，时序上位于 `_execute_compression_actions`（已生成 `_pending_compression_events`）之后、`execute_base_model_with_effective_overrides`（真正 NPU forward）之前。

#### 1.4.1 入口与 disabled 短路（`worker_reclaim_sync.py:99-125`）

```python
def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    global _DEBUG_DISABLE_LOGGED
    if os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0") ... in {"1", ...}:
        ...
        return                                # 短路 1：env 显式关闭
    if not isinstance(events, list) or not events:
        return                                # 短路 2：没有 applied 事件
    # 后续逻辑…
```

- 短路 1：`TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC=1` 强制关闭整个逻辑驱逐函数（已用于 debugging/对照实验，但**默认开启**，不会触发）。
- 短路 2：`_pending_compression_events` 列表为空时直接 return。注意 `execute_runner_compression_actions`（`runner_compression_actions.py:38-313`）**即使所有 trigger 都被 skip 也会产生 `status="skipped"` 事件**，所以这里判断的是「applied 事件数 == 0」的情形（参见 `apply_worker_block_reclaim_events` 内部的 `if not isinstance(event, dict) or event.get("status") != "applied": continue`）。

#### 1.4.2 Block Table 解析（`worker_reclaim_sync.py:127-170`）

```python
input_batch = getattr(base_runner, "input_batch", None)
block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
if block_table_obj is None:
    if getattr(base_runner, "block_tables", None) is not None:
        # Formal V2 runner manages block tables directly on base_runner
        # rather than on input_batch. ...
        return                                # 短路 3：V2 runner 不走此路径
    logger.warning("TriAttention worker reclaim: block table not found. ...")
    return                                    # 短路 4：V1 input_batch 路径也找不到

req_id_to_index = getattr(input_batch, "req_id_to_index", None)
if not isinstance(req_id_to_index, dict):
    logger.warning("TriAttention worker reclaim: req_id_to_index not found ...")
    return                                    # 短路 5：req_id→row 映射缺失

# 区分单表 / MultiGroup
inner_tables = getattr(block_table_obj, "block_tables", None)
if isinstance(inner_tables, list):
    tables = inner_tables                      # MultiGroupBlockTable（G >= 2，如 MLA）
else:
    tables = [block_table_obj]                # 单 BlockTable

cache_config = getattr(base_runner, "cache_config", None)
block_size = int(getattr(cache_config, "block_size", 16))
if block_size <= 0:
    block_size = 16
```

- 短路 3：V2 runner 把 block_tables 直接挂在 `base_runner.block_tables` 上，且 hook 端已经更新了规范表，因此**不再走 V1 的 `input_batch.block_table` 同步**。这是 V2 与 V1 在 Block Table 视图管理上的关键差异。
- 短路 4：理论上 NPU 平台下 `input_batch.block_table` 一定存在（`vllm_ascend/worker/npu_input_batch.py:94 self.block_table = MultiGroupBlockTable(...)`），但若 NPUModelRunner 尚未初始化到这一步，会被 `logger.warning` 提示。
- 短路 5：若 `input_batch.req_id_to_index` 不是 dict（极少见），整个函数直接 return，**所有视图 1+2+3 都不动**——但不影响视图 4+5（由 Scheduler 端在 `_apply_compression_events` 里写）。

#### 1.4.3 事件预处理（`worker_reclaim_sync.py:172-194`）

```python
for event in events:
    if not isinstance(event, dict) or event.get("status") != "applied":
        continue
    req_id = event.get("req_id")
    if req_id is None:
        continue
    req_index = req_id_to_index.get(req_id)
    if not isinstance(req_index, int):
        continue                              # 跳过：req 尚未加入 input_batch
    cache_len_after = event.get("cache_len_after")
    if not isinstance(cache_len_after, int) or cache_len_after <= 0:
        continue                              # 跳过：cache_len 非法

    details = event.get("details")
    retained_cache_len = (details.get("retained_cache_len")
                          if isinstance(details, dict) else None)
    if not isinstance(retained_cache_len, int) or retained_cache_len <= 0:
        retained_cache_len = cache_len_after  # 退化：details 缺字段
    required_blocks = (retained_cache_len + block_size - 1) // block_size
    reclaim_mode, groups_by_gid = _event_reclaim_groups(event)
```

- `retained_cache_len` 优先取自 `details["retained_cache_len"]`（来自 `finalize_hook_placement_result` 的 `placement_plan.retained_cache_len`，`hook_group_pipeline.py:530-534`），若缺失退化为 `cache_len_after`。
- `required_blocks = ceil(retained_cache_len / block_size)` 是本函数决定要保留的「逻辑 block 数」。

#### 1.4.4 视图 1+2 写入主路径：truncate_tail（`worker_reclaim_sync.py:196-232`）

```python
for gid, table in enumerate(tables):
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if num_blocks_per_row is None:
        continue
    if not isinstance(num_blocks_per_row, np.ndarray):
        continue
    current = int(num_blocks_per_row[req_index])
    if reclaim_mode == "remap_tail":
        ...                                    # 见 1.4.5
    if current > required_blocks:
        num_blocks_per_row[req_index] = required_blocks    # 视图 1 写入
        if runtime_logging_enabled():
            logger.debug("TriAttention worker reclaim: req=%s num_blocks %d -> %d ...", ...)
    _clear_table_row_tail(                     # 视图 2 写入
        table,
        req_index,
        _row_block_count(table, req_index, min(current, required_blocks)),
    )
```

- **视图 1 写入点**：`num_blocks_per_row[req_index] = required_blocks`。**这正是「逻辑驱逐」的本质**——下次 `_update_states` 中的 `block_table.append_row(new_block_ids, req_index)` 走 `start = self.num_blocks_per_row[row_idx]`（`vllm_ascend/worker/block_table.py:97-98`），起点是 `required_blocks` 而不是 `current`，**新分配的 block 物理写入但逻辑上续接在压缩后的尾段**。
- **视图 2 写入点**：`_clear_table_row_tail`（`worker_reclaim_sync.py:49-62`）把 `block_table_np[req_index, used_blocks:] = 0`。`used_blocks = min(current, required_blocks)`：
  - 当 `current > required_blocks`：`used_blocks = required_blocks`，清零 `[:, required_blocks:]`；
  - 当 `current <= required_blocks`：`used_blocks = current`，清零 `[:, current:]`（**实际是 no-op**）。
- **关键不对称**：truncate_tail 模式下，**视图 1 被设小、视图 2 的 `[required_blocks:current]` 段被清零，但 `block_pool.blocks[block_id].ref_cnt` 完全没动**——这意味着「该请求不再使用这些 block_id」但「这些 block 仍在 block_pool 中被本请求的 ref 占用」。物理归还需要靠 vLLM 自身的 `coordinator.free(request_id)`，但 `coordinator.free` 仅在「请求 finished」时触发；**逻辑驱逐已经把视图 1+2 切短，物理 ref 还得等下次 allocate_slots 走补丁路径或请求结束**。

#### 1.4.5 视图 1+2 写入旁路：remap_tail（`worker_reclaim_sync.py:203-219`）

```python
if reclaim_mode == "remap_tail":
    block_ids_after = _block_ids_after(groups_by_gid.get(gid))
    if block_ids_after is not None:
        if _rewrite_table_row(table, req_index, block_ids_after):
            logger.debug("TriAttention worker remap: req=%s gid=%d num_blocks %d -> %d", ...)
        else:
            logger.warning("TriAttention worker remap failed: req=%s gid=%d table=%s", ...)
        continue                               # 跳过 truncate_tail
```

- `_block_ids_after`（`worker_reclaim_sync.py:36-46`）严格校验 `block_ids_after`：必须为 `list[int]`、无重复、有序（隐式）。校验失败返回 `None`，**整个 gid 走 `continue`，视图 1+2 都不动**。
- `_rewrite_table_row`（`worker_reclaim_sync.py:73-96`）：
  - 优先调 `table.add_row(block_ids, req_index)`（vllm 0.18.0 `BlockTable.add_row`，`vllm_ascend/worker/block_table.py:103-105`）—— `add_row` 内部 `self.num_blocks_per_row[row_idx] = 0; self.append_row(block_ids, row_idx)`，**视图 1 被显式置 0**；
  - 再调 `_clear_table_row_tail(...)` 清掉可能残留的尾段。
  - 兜底分支：直接 `block_table_np[req_index, :] = 0; block_table_np[req_index, :len(block_ids)] = block_ids; num_blocks_per_row[req_index] = len(block_ids)`。
- **关键不对称（remap_tail 特有）**：`remap_tail` 不减少 `block_ids` 总数（用「物理保留前 N 块 + 重映射到不同物理 block」），因此：
  - 视图 1 被改成 `len(block_ids_after)`；
  - 视图 2 被重写为 `block_ids_after` 的物理 block_id 列表；
  - **但前段被「逻辑上让出」的 block_id 仍在 `block_pool` 中被本请求 ref**（因为 `req_state.block_ids` 后续会更新，但 `req_to_blocks[req_id]` 在 Scheduler 端才会更新，**与 Worker 端 rewrite 之间存在竞态**）。

#### 1.4.6 视图 3 写入：`base_runner.requests[req_id].block_ids`（`worker_reclaim_sync.py:234-266`）

```python
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
            else:
                for group_blocks in block_ids_attr:
                    if (
                        isinstance(group_blocks, list)
                        and len(group_blocks) > required_blocks
                    ):
                        del group_blocks[required_blocks:]   # 视图 3 写入
```

- **truncate_tail 模式**：`del group_blocks[required_blocks:]`——**直接修改 list 对象本身**。这是 Python 引用语义下的关键：vLLM 的 `gpu_model_runner.py:1248-1250` 中 `for block_ids, new_ids in zip(req_state.block_ids, new_block_ids): block_ids.extend(new_ids)` 遍历的就是这个 `group_blocks` 列表。
- **remap_tail 模式**：逐 gid 替换 `group_blocks` 为 `block_ids_after`（**整体重写**）。注意 `setattr(req_state, "block_ids", ...)` 会整体替换 `block_ids_attr`（list/tuple 容器），但 list 内的 `group_blocks` 是 list 对象时是 mutate 替换；为 tuple 时是 new tuple 替换。
- **视图 1+2 与视图 3 的同步关系**：
  - `worker_reclaim_sync.py:220-232` truncate_tail 主路径只写视图 1+2，**视图 3 由 `worker_reclaim_sync.py:234-266` 在同一事件循环里再写**。但顺序是先 1+2 后 3，且**两次写入之间没有 `commit_block_table` 或 `copy_to_gpu`**——NPU 端的 `block_table.gpu`（`vllm_ascend/worker/block_table.py:77`）在 `_update_states` 末尾的 `commit_block_table(num_reqs)` 才会被同步到 NPU。
  - remap_tail 模式下，**视图 3 与视图 1+2 的写入是分开的**：视图 1+2 在 `_rewrite_table_row` 中写入；视图 3 在 `block_ids_attr` 重写时写入。两者都在 `_update_states` 之前，顺序无关。

#### 1.4.7 视图 1+2 与下次 `_update_states` 的时序交互

`vllm_ascend/worker/model_runner_v1.py:1108 execute_model` 入口第一行是 `self._update_states(scheduler_output)`（`vllm_ascend/worker/model_runner_v1.py:1136`），它会执行：

```python
# vllm/v1/worker/gpu_model_runner.py:1247-1278（vllm 0.18.0 通用逻辑，NPU 端同源）
if not resumed_from_preemption:
    if new_block_ids is not None:
        # Append the new blocks to the existing block IDs.
        for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
            block_ids.extend(new_ids)
# ...
self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
if new_block_ids is not None:
    self.input_batch.block_table.append_row(new_block_ids, req_index)
```

- **`req_state.block_ids.extend(new_ids)`**（视图 3 的延续）：在 NPU 端 `block_table.append_row` 之前先扩 CPU 列表。**这就是为何 `_patch_scheduler_output_for_compressed_reqs`（`runner.py:915-1057`）要在 attach_execute_model_compression_events 之前裁剪 `new_block_ids`**——否则 `block_ids.extend(new_ids)` 会把 32k 完整长度的 new_block_id 续接回 `req_state.block_ids`，把刚刚被 `del group_blocks[required_blocks:]` 截掉的逻辑尾巴又长回来。
- **`block_table.append_row(new_block_ids, req_index)`**（视图 1+2 的续写）：`vllm_ascend/worker/block_table.py:86-101`：
  ```python
  num_blocks = len(block_ids)
  start = self.num_blocks_per_row[row_idx]                # ← 这里读视图 1
  self.num_blocks_per_row[row_idx] += num_blocks
  self.block_table.np[row_idx, start : start + num_blocks] = block_ids
  ```
  - 视图 1 的 `start` 已经是 TriAttention 截短后的 `required_blocks`（`worker_reclaim_sync.py:221`），新 block 物理写入但起点对齐；
  - 视图 2 的 `block_table_np[req_index, :]` 在 `_clear_table_row_tail` 中已经把 `[required_blocks:current]` 段清零，新 block 续接不会覆盖有效数据。

#### 1.4.8 视图 1+2 的「零写退化」风险

下列任一情形下，`apply_worker_block_reclaim_events` 对视图 1+2 **完全不写入**：

| 触发条件 | 文件:行 | 现象 |
| - | - | - |
| `TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC=1` | `worker_reclaim_sync.py:113-122` | 函数整体 return |
| `events` 列表为空或非 list | `worker_reclaim_sync.py:124-125` | 提前 return |
| `input_batch.block_table is None` 且 `base_runner.block_tables is not None` | `worker_reclaim_sync.py:130-136` | V2 runner 短路，依赖 hook 端同步 |
| `input_batch.block_table is None`（两个分支均不命中） | `worker_reclaim_sync.py:137-143` | `logger.warning` 后 return |
| `input_batch.req_id_to_index` 非 dict | `worker_reclaim_sync.py:148-155` | 提前 return |
| `event.status != "applied"` | `worker_reclaim_sync.py:173-174` | 该 event 跳过 |
| `req_id_to_index.get(req_id)` 不是 int | `worker_reclaim_sync.py:178-180` | 该 event 跳过（**典型：跨 step 时序错位**） |
| `cache_len_after` 非正 int | `worker_reclaim_sync.py:181-183` | 该 event 跳过 |
| `num_blocks_per_row is None` 或非 ndarray | `worker_reclaim_sync.py:198-201` | 该 gid 跳过 |
| `reclaim_mode == "remap_tail"` 且 `_block_ids_after` 返回 None | `worker_reclaim_sync.py:204-205` 与 `worker_reclaim_sync.py:36-46` | **该 gid 既不走 remap 也不走 truncate**（直接 `continue`） |

> 这是「**逻辑驱逐**」**最易踩到的静默失败点**——上游所有触发条件都成功，但 `apply_worker_block_reclaim_events` 内任意一条 continue / return 都会让视图 1+2 在该 (req_id, gid) 上不被写入。

### 1.5 视图 4：`request._triattention_effective_kv_offset`（kv_allocation_sync.py 主体）

「逻辑驱逐」要让上游 `kv_cache_manager.allocate_slots` 知道「该请求的 num_computed_tokens 在物理上没增长」，靠的是 `request._triattention_effective_kv_offset` 这一个属性：

- 写入点（`triattention/vllm/runtime/kv_allocation_sync.py:35-58 update_request_effective_kv_offset`）：
  ```python
  def update_request_effective_kv_offset(*, request, cache_len_after):
      logical = int(request.num_computed_tokens)
      effective = int(cache_len_after)
      if effective > logical:
          effective = logical
      offset = logical - effective
      if offset <= 0:
          delattr / setattr None
          return 0
      setattr(request, "_triattention_effective_kv_offset", offset)   # 视图 4 写入
      return offset
  ```
- 读取点（`kv_allocation_sync.py:61-81 prepare_request_effective_num_computed`）：
  ```python
  def prepare_request_effective_num_computed(request):
      logical = int(request.num_computed_tokens)
      offset = int(request._triattention_effective_kv_offset)
      if offset is None or offset <= 0:
          delattr / None
          return None
      if logical == 0 or logical < offset:
          clear_request_allocation_sync_state(request)
          return None
      effective = logical - offset
      setattr(request, "_triattention_effective_num_computed_tokens", effective)  # 视图 4' 写入
      return effective
  ```
- **视图 4 写入的触发位置**：`triattention/vllm/runtime/scheduler.py:710-714` 与 `877-881`，**两处都仅在 `reclaim_applied_any` 为 True 时调 `update_request_effective_kv_offset`**。换言之：
  - 视图 4 的写入条件 = `reclaim_applied_any == True`
  - `reclaim_applied_any == True` ⇐ `_free_reclaimed_blocks(manager, removed_old_blocks)` 成功调 `block_pool.free_blocks` 且 `removed_old_blocks` 非空（**触发物理回收**）
  - **这是一个奇怪的耦合：视图 4（纯逻辑驱逐的入口）居然依赖物理回收的成败**。

> 关键事实 3：**这是「逻辑驱逐失效」的核心耦合点**。Scheduler 端 `_apply_compression_events` 中 `_free_reclaimed_blocks` 的失败（最常见：`removed_old_blocks` 为空、synthesized reclaim 在 prefill 阶段被跳过、`block_pool.free_blocks` 因 ref_cnt 未归零而 no-op）会直接让视图 4 不被写入，**下次 `allocate_slots` 走原始路径**，上游按 `request.num_computed_tokens = 32768` 继续分配 block——这是「filter 生效但 Block 池水位不降」的最大可能原因。

#### 视图 4 在 `allocate_slots` 路径上的消费

`_patched_kv_cache_allocate_slots`（`triattention/vllm/runtime/integration_monkeypatch.py:493-534`）：

```python
def _patched_kv_cache_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
    prepare_request_effective_num_computed(request)         # 视图 4' 准备
    effective_num_computed = resolve_request_effective_num_computed(request)
    if effective_num_computed is None:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)           # 退化为原始路径
    logical_num_computed = request.num_computed_tokens
    if not isinstance(logical_num_computed, int):
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)
    if effective_num_computed >= logical_num_computed:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)           # 退化：偏移没意义
    kwargs = dict(kwargs)
    kwargs["delay_cache_blocks"] = True
    setattr(request, "num_computed_tokens", int(effective_num_computed))   # 临时改写
    try:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)
    finally:
        setattr(request, "num_computed_tokens", logical_num_computed)     # 还原
```

- 关键：`effective_num_computed = logical - offset`，`offset = logical - cache_len_after`，所以 `effective_num_computed = cache_len_after`。在 32k+2k 场景下 `effective_num_computed = 2048`（或者 `2048 + scheduled_tokens`），远小于 `logical = 32768`。
- **`delay_cache_blocks=True`**：让 vLLM 上游 `kv_cache_manager.allocate_slots` 末尾的 `self.coordinator.cache_blocks(request, num_tokens_to_cache)` 不执行（`vllm/v1/core/kv_cache_manager.py:386`）—— 避免把压缩后的 block 写进 prefix-cache。
- **若视图 4 未被写入**：`prepare_request_effective_num_computed` 返回 `None`，整个 `if effective_num_computed is None` 分支直接 `return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)` 走原始路径——**逻辑驱逐在此处对 `allocate_slots` 失效**。

### 1.6 视图 5：Hook 内的 `req_state.block_ids`（hook 内部态）

Hook 端还有一份「下次压缩的起点基线」要维护：

- **写入点 A**：`triattention/vllm/runtime/hook_group_pipeline.py:452-453`，`execute_group_compaction` 末尾：
  ```python
  if config.enable_experimental_block_reclaim and group_cache_len_after is not None:
      mutable_block_ids_by_group[gid] = list(group_outcome.kept_block_ids)
  ```
  这里直接修改了 hook 内部的「mutable block_ids 列表」，但**不写回 `base_runner.requests[req_id].block_ids`**（那是视图 3，由 worker_reclaim_sync.py 写）。
- **写入点 B**：`triattention/vllm/runtime/hook_group_pipeline.py:494-543 finalize_hook_placement_result`：
  ```python
  if config.enable_experimental_block_reclaim and outcome.block_reclaim_groups:
      reassigned_block_ids = []
      for idx, group_block_ids in enumerate(outcome.mutable_block_ids_by_group):
          if group_block_ids is None:
              reassigned_block_ids.append(original_block_ids_by_group[idx])
          else:
              reassigned_block_ids.append(group_block_ids)
      req_state.block_ids = (
          tuple(reassigned_block_ids)
          if isinstance(original_block_ids_by_group, tuple)
          else reassigned_block_ids
      )
      block_reclaim_payload = ReclaimEvent(mode=outcome.reclaim_mode, groups=outcome.block_reclaim_groups)
  ```
  - **`req_state` 是哪里来的？** `triattention/vllm/runtime/hook_impl.py:122-141` 通过 `resolve_hook_request_context`（`triattention/vllm/runtime/hook_preflight.py`）从 `base_runner.requests[req_id]` 取，**与视图 3 是同一对象**。
  - **隐患**：这里 `req_state.block_ids = reassigned_block_ids` **整体重写** 视图 3，与 worker_reclaim_sync.py 后面再做的 `del group_blocks[required_blocks:]`（视图 3 二次写入）是**同一对象的两次修改**。如果重写后 `reassigned_block_ids` 容器结构变化（例如原本是 `tuple[list[...], list[...]]`、被改成 `list[list[...]]`），worker_reclaim_sync.py 的 `del group_blocks[required_blocks:]` 会按 `for group_blocks in block_ids_attr` 遍历新容器，仍然能工作（list of list），但**若新容器是 list of tuple 则 `del` 会失败**（tuple 不可变）——不过 truncate_tail 路径下不会被走到，因为 finalize 出来的 `reclaim_mode` 由 hook 控制。

> 关键事实 4：视图 5 与视图 3 实质上是**同一对象的两个写入者**——hook 在前 `setattr` 整体替换，worker_reclaim_sync 在后 `del` 局部截断。**两者的写入时序不重叠**（hook 在 `_hook` 返回前完成，worker_reclaim_sync 在 `apply_worker_block_reclaim_events` 中），但都是「逻辑驱逐」不可缺的一环。

### 1.7 逻辑驱逐的输入 Patch（让 NPU attention 看到「变短」）

仅修 Block Table 视图还不够——NPU 内部的 attention kernel 在每个 step 还要知道：

- 「`seq_len` 是多少」（不能让 NPU 按 32k 算 attention）
- 「`slot_mapping` 怎么映射」（压缩后的前段不能映射到原始 block 的 slot）

这块由 `triattention/vllm/runtime/input_patch_vllm_v1_backend.py` 与 `triattention/vllm/runtime/effective_overrides.py` 协同完成：

- `effective_overrides.py:build_effective_sparse_overrides`（`effective_overrides.py:230-384`）：
  - 入口对每个 `scheduled_item`：
    - 首选「Stable 路径」：`effective_before_step = cache_len_after + (current_nct - nct_at_compression)`，取 `delta = cache_len_after - nct_at_compression`；
    - 退化为「Fallback 路径」：`abs_progress = req_state.num_computed_tokens`（带 chunked-prefill 滞后修正），`effective_before_step = current_cache_len - scheduled_tokens`；
  - 输出 `(seq_bases, pos_deltas, single_seq_base, single_pos_delta)`：
    - `seq_bases[req_idx] = effective_before_step`：把 `seq_len` 改写成有效长度；
    - `pos_deltas[req_idx] = effective_before_step - abs_progress`（负数）：`slot_mapping` 起始位置左移。
  - **`delta == 0` 直接 continue**：意味着 `effective_before_step == abs_progress`，即「没有压缩」或「scheduler 反馈了一个本 step 还没压缩的 view」，**不写 override**。
- `input_patch_vllm_v1_backend.py:_build_effective_slot_positions`（`input_patch_vllm_v1_backend.py:336-400+`）：基于 `pos_deltas` 改写 `positions_np`，让 NPU 端 `compute_slot_mapping` 输出的 `slot_mapping` 落在压缩后的物理 block 上。
- `runner_output_bridge.py:execute_base_model_with_effective_overrides`（`runner_output_bridge.py:142-233`）：用 `active_effective_input_overrides(overrides)` 上下文管理器把 `seq_bases`/`pos_deltas` 注入到 `input_patch_state._patch_state.ACTIVE_*` 模块全局，NPU 准备 inputs 时按需读取。

> 关键事实 5：**输入 Patch 是「逻辑驱逐」在 NPU forward 阶段的对偶物**——Block Table 视图改短（视图 1+2+3）让 NPU 知道「该用多少 block」；输入 Patch（seq_lens / slot_mapping）让 NPU 知道「在那些 block 上怎么索引」。**两者必须协同**，否则会出现 Block Table 截到 `required_blocks=128` 但 NPU attention 仍按 32k 计算 attention 的「假装截短」状态，模型输出会乱。

### 1.8 关联联动模块

- **KV Cache 管理器与 Block 分配机制**：
  - 上游 vLLM：`vllm/v1/core/kv_cache_manager.py:218-388 allocate_slots()`、`vllm/v1/core/block_pool.py:409-423 free_blocks()`，后者是 `ref_cnt` 计数 + 自由队列的经典实现。**这是物理回收的入口**，但 **与 TriAttention 逻辑驱逐正交**——TriAttention 通过 `_patched_kv_cache_allocate_slots` 间接让 `allocate_slots` 知道「该请求逻辑短了」，但**不直接调 `free_blocks`**。
  - **逻辑驱逐对 allocate_slots 的影响**：`effective_num_computed = cache_len_after`（远小于 `num_computed_tokens`）→ `allocate_slots` 内部按 `effective_num_computed` 算「需补多少 block」→ **新分配的 block 数大幅减少**（这是「并发水位」下降的真正驱动）。
- **输入 Override Patch（`seq_len / slot_mapping / positions`）**：详见 1.7。
- **Block Table 形状管理（核心）**：
  - vLLM V1 通用：`vllm/v1/worker/block_table.py:9-251 BlockTable / MultiGroupBlockTable`，`append_row`/`add_row` 在 `num_blocks_per_row[row_idx]` 之后追加。
  - vLLM-Ascend 0.18.0 NPU：`vllm_ascend/worker/block_table.py:9-321 BlockTable / MultiGroupBlockTable`，**API 与上游 vLLM 一致**，但底层 CpuGpuBuffer 的 `num_blocks_per_row` 是 `np.int32` 数组，**`append_row` 用 `start = self.num_blocks_per_row[row_idx]` 然后 `self.block_table.np[row_idx, start:start+num_blocks] = block_ids`**（这与 vLLM V1 完全一致）。
  - **`vllm_ascend/worker/npu_input_batch.py:32-... NPUInputBatch(InputBatch)` 继承上游 `InputBatch`，仅在构造时换用 `vllm_ascend.worker.block_table.MultiGroupBlockTable`**——**这是 TriAttention worker_reclaim_sync.py 直接操作的对象**。
- **缓存复用 / prefix cache 清理**：
  - `_evict_reclaimed_block_metadata`（`triattention/vllm/runtime/scheduler.py:35-46`）→ `block_pool._maybe_evict_cached_block()`（`vllm/v1/core/block_pool.py:352-390`）属于物理回收范畴（清除 `cached_block_hash_to_block` 中的引用），**但仅在视图 4 写入路径被触发**（见 1.5）。
  - `enable_prefix_caching=False`（README 强制要求）是 TriAttention 正常工作的前提；若用户漏配，`cached_block_hash_to_block` 仍可能保留旧 hash，使 `block_pool.get_usage()` 持续在高位——但这影响的是「物理水位」而非「逻辑驱逐」。
- **内存释放的对外可观测**：
  - `kv_cache_manager.usage` 反映 `block_pool.get_usage()`，受 ref_cnt 归零后 free_block_queue 长度影响。**这是物理回收的对外信号，不是逻辑驱逐的**。
  - TriAttention 自身：`worker_reclaim_sync.py:222-227` 对视图 1 的变化打 debug log；`worker_reclaim_sync.py:208-212` 对视图 1+2 的 remap_tail 变化打 debug log；`runner_compression_actions.py:178-258` 对每个 `applied` 事件打 log；`integration_monkeypatch.py:101-102 _refresh_scheduler_stats_kv_usage` 把最新 `usage` 写回 outputs。

---

## 模块二：代码可疑风险点梳理

按「潜在问题 → 文件/位置 → 静态推演结论 → 致「逻辑驱逐失效、并发无提升」的因果链」逐条列示。所有结论均基于 v0.18.0 源码静态推导，未做任何运行验证。**与 worker_reclaim_sync.py 强相关的部分用 ★ 标出**。

### 2.1 ★ [P0] `_patch_scheduler_output_for_compressed_reqs` 在没有 `events_by_req_id[req_id]` 时整体跳过裁剪

- **位置**：`triattention/vllm/runtime/runner.py:915-1057 _patch_scheduler_output_for_compressed_reqs()`
- **问题**：当 `state_store.compression_count > 0` 但 `events_by_req_id` 查不到（因为本次 step 实际是「defer 之后的下一 step」，`event` 是历史最近一次而不是本次）时，`retained_cache_len = None`（`runner.py:1039 _event_retained_cache_len`），`group_limits = _group_limits_for_event(req_index, None)`：
  - `_table_max_blocks(table, block_table_obj)` 取自 `max_num_blocks_per_req`，`max(0, max_num_blocks_per_req - current)` 通常很大；
  - `_ceil_div_positive(retained_cache_len, block_size)` 由于 `retained_cache_len is None` 直接返回 `None`；
  - 最终 `limit = max(0, required - current) if retained_cache_len is not None else capacity_limit` —— 但因为外层 `if not any(limit is not None for limit in group_limits): continue`（`runner.py:1041-1042`），**直接跳过不裁剪**。
  - 此时 `update_states` 中的 `block_table.append_row(new_block_ids, req_index)` 会把本 step 的 new_block_id 全部 append 到已 reclaim 的位置 `start = self.num_blocks_per_row[req_index]` 之后，**追加到物理未 reclaim 的新 block**。
- **因果推导**：当 `compression_count > 0` 但本次 step 无 compression event 时，Worker 视图 1+2 下的 `num_blocks_per_row` 已是 `required_blocks`，但 vLLM 上游调度的 `new_block_ids` 仍按原始 `num_computed_tokens` 分配追加；**`req_state.block_ids.extend(new_ids)` 与 `block_table.append_row` 双双重写**——`req_state.block_ids` 实际长度恢复到 `original + new` 而非 `required_blocks + new`，**视图 3 整个被冲掉**。这是一个**与 worker_reclaim_sync.py 1.4.7 段耦合的失效放大器**：即使 worker_reclaim_sync.py 1.4.4 写过视图 3，下一次 `_update_states` 会立刻把它覆盖回原始长度。

### 2.2 ★ [P0] `apply_worker_block_reclaim_events` 在 remap_tail 模式下 `_block_ids_after` 校验失败时 `continue`，视图 1+2+3 整体静默不写

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:204-205` 与 `36-46 _block_ids_after`
- **问题**：
  ```python
  if reclaim_mode == "remap_tail":
      block_ids_after = _block_ids_after(groups_by_gid.get(gid))
      if block_ids_after is not None:
          if _rewrite_table_row(table, req_index, block_ids_after): ...
          else:
              logger.warning("TriAttention worker remap failed: ...")
          continue                                   # 关键：直接 continue
      # 若 block_ids_after is None，落到下面 truncate_tail 路径
      if current > required_blocks:
          num_blocks_per_row[req_index] = required_blocks
      ...
  ```
  - `_block_ids_after`（`worker_reclaim_sync.py:36-46`）严格校验 `block_ids_after`：必须为 `list[int]`、**所有元素必须是 int**、**无重复**、且**有序**。
  - **校验失败时**（如 `block_ids_after` 包含 None、有重复、不是 list）：`_block_ids_after` 返回 `None`，整个 gid 走 `if current > required_blocks` 分支（truncate_tail）—— 这一段**仍然会写视图 1**，但**视图 2 不会被重写为 remap_tail 的目标 block_ids**，而是被 `_clear_table_row_tail` 清零（视图 2 退化为「保留前 required_blocks 块」）。
  - **关键陷阱**：remap_tail 的语义是「保留最近的 N 块」（zero-copy recency）；若 `_block_ids_after` 校验失败，**退化为 truncate_tail 等价行为（保留前 N 块）**，但**压缩后 N 块不是「最近 N 块」而是「最早 N 块」**——这是「filter 生效但内容错乱」的隐性 bug（虽然不直接致「并发无提升」，但会让 attention 拿到的 KV 是 prompt 前段而非「最近」内容）。
- **因果推导**：典型触发场景是 hook 端构造 `block_ids_after` 时引入了 Python 端的 `numpy.int64` 或 `torch.Tensor` 元素（不是 `int`），或 multi-group 下 group 拼接时产生了重复 block_id。`hook_group_pipeline.py:131-132` 的 `if len(kept_tail_block_ids) != budget_blocks: return None` 兜底会返回 None，但 `keep_block_ids = kept_tail_block_ids + trailing_block_ids`（`hook_group_pipeline.py:138`）后 `block_ids_after` 是 Python list，元素类型不严格——这是一个值得用日志验证的点。

### 2.3 ★ [P0] `num_blocks_per_row[req_index] = required_blocks` 在 multi-group 场景下与 `req_state.block_ids` 的 group 结构不一一对应

- **位置**：
  - `triattention/vllm/runtime/worker_reclaim_sync.py:196-232`（对 `tables = inner_tables` 逐个写）
  - `triattention/vllm/runtime/worker_reclaim_sync.py:260-266`（对 `block_ids_attr` 逐 group 写 `del group_blocks[required_blocks:]`）
- **问题**：
  - **关键假设**：`tables[gid]` 与 `block_ids_attr[gid]` 一一对应（gid 索引相同）。
  - 在 vLLM V1 `MultiGroupBlockTable`（`vllm_ascend/worker/block_table.py:231-321`）与 NPU 0.18.0 主流模型（无 MLA）下，这个假设成立。
  - 但在 MLA / multi-group 场景下，**`tables[gid]` 数量可能与 `block_ids_attr` 数量不匹配**（比如 hook 端把 K/V group 合并输出，但 `input_batch.block_table` 仍然拆成 2 个 group）。**`worker_reclaim_sync.py:260-266` 用 `for group_blocks in block_ids_attr:` 遍历**——若 `block_ids_attr` 长度与 `tables` 不同，会出现「某些 gid 写视图 1 但未写视图 3」或反之。
  - **`required_blocks` 是单一值**（`worker_reclaim_sync.py:193`）：`(retained_cache_len + block_size - 1) // block_size`，**对所有 gid 用同一阈值**。但 multi-group 下不同 group 的 `block_size` 可能不同（MLA 主+辅），**单一阈值会过度截断或欠截断**——例如主 group `block_size=16`、辅 group `block_size=128`，用 2048/16=128 block 截辅 group 会把它的 128*128=16384 token 都视为 1 block，**视图 2 被错误清零**。
- **因果推导**：MLA 模型上 multi-group 实际生效时，**视图 1+2 写入与视图 3 写入可能出现非对称**——`num_blocks_per_row[gid=0]` 被设到 128、`num_blocks_per_row[gid=1]` 也被设到 128（同一阈值），但 `block_ids_attr[1]`（K group）可能只该保留 8 块、`block_ids_attr[1]`（V group）保留 16 块……这会让 `_update_states` 的 `block_ids.extend(new_ids)` 在错误起点续接，**视图 3 与视图 1+2 永久失同步**。

### 2.4 ★ [P1] `_patched_kv_cache_allocate_slots` 中 `effective_num_computed` 取决于视图 4，而视图 4 写入仅在 `reclaim_applied_any=True` 路径

- **位置**：
  - `triattention/vllm/runtime/integration_monkeypatch.py:493-534 _patched_kv_cache_allocate_slots`
  - `triattention/vllm/runtime/kv_allocation_sync.py:35-58 update_request_effective_kv_offset()`
  - `triattention/vllm/runtime/scheduler.py:877-881`（仅当 `reclaim_applied_any` 才调 `update_request_effective_kv_offset`）
- **问题**：
  - `_apply_compression_events`（`scheduler.py:570-881`）中，只有 `_free_reclaimed_blocks(manager, removed_old_blocks)` 真正成功（即 `removed_old_blocks` 非空、且 `block_pool.free_blocks` 推进 free_block_queue）后，`reclaim_applied_any` 才为 True。
  - 若 prefill 阶段被 2.6 的 synthesized-skip 路径拦住，或 `removed_old_blocks` 因 `expected_shrink_gids` 为空而 0 个元素（如：已经被早些 step 的 zero-copy remap 接管），则 `reclaim_applied_any=False`，**视图 4 不被写入**。
  - 下次 `prepare_request_effective_num_computed` 返回 `None`，`_patched_kv_cache_allocate_slots` 走原始 `allocate_slots` 路径，**vLLM 继续按 `request.num_computed_tokens` 重新分配 block 池** → 物理 Block 数从头再涨。
- **因果推导**：这是「压缩逻辑跑过、worker_reclaim_sync 写过视图 1+2+3、但物理 Block 池水位不降反升」的典型死循环。**关键耦合：视图 4 的写入完全依赖物理回收的成败**——但 `apply_worker_block_reclaim_events` 写视图 1+2+3 是纯逻辑驱逐，**不保证 view 4 会被写入**。

### 2.5 [P0] chunked prefill 阶段 synthesized reclaim 被静默跳过

- **位置**：`triattention/vllm/runtime/scheduler.py:673-714` 与 849-875
- **可疑代码段**（`scheduler.py:673-693`）：
  ```python
  if not isinstance(groups, list):
      # ... V1 batch-queue race note ...
      if _evt_scheduled > 1:
          # 跳过 synthesized reclaim during prefill
          logger.debug("skipping synthesized reclaim during prefill ...")
      elif expected_shrink_gids and isinstance(managers, (list, tuple)):
          # ... 真正执行 _free_reclaimed_blocks ...
  ```
- **问题**：当 `event["block_reclaim"]` 为 `None` 或 groups 不是 list（V1 批量队列竞态：Worker 已经在更早 step 把 block 截了，但 Scheduler 这边 `event` 是后到的「第二次压缩事件」，自然没有 groups 字段），会进入「synthesized reclaim」分支：
  - **如果 `scheduled_tokens > 1`（即 chunked prefill）**：直接 `continue`，**不释放任何 Block**（`scheduler.py:685-692`）；
  - 如果 `scheduled_tokens == 1`（decode）：按 `expected_shrink_gids` 合成 free。
- **因果推导**：32k 输入典型场景 prefill 必然走 chunked（`max_num_batched_tokens=1024`）。如果第一次压缩事件恰好落在 prefill 阶段，**被 synthesize 跳过的所有后续 prefill step 都不释放 Block**（这是物理回收层面的），**视图 4 在 prefill 阶段不会被写入**——这与 2.4 联动放大：worker_reclaim_sync.py 写过视图 1+2+3，但视图 4 缺席导致 `_patched_kv_cache_allocate_slots` 走原始路径，**物理 Block 持续按 32k 分配**。

### 2.6 ★ [P1] 视图 1+2 与视图 3 的写入顺序错位导致 `_update_states` 读到 stale 状态

- **位置**：
  - `triattention/vllm/runtime/runner.py:1161-1171`（执行顺序：`_execute_compression_actions` → `_apply_worker_block_reclaim_events`）
  - `triattention/vllm/runtime/runner.py:1186-1247`（`execute_base_model_with_effective_overrides`）
  - `vllm_ascend/worker/model_runner_v1.py:1136`（`_update_states` 入口）
- **问题**：
  - `_apply_worker_block_reclaim_events` 在 `execute_base_model_with_effective_overrides` 之前完成，**视图 1+2+3 已经写入**。
  - **但 `execute_base_model_with_effective_overrides` 内部还会调 `base_runner.execute_model`**，后者进入 `_update_states`：
    ```python
    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
        block_ids.extend(new_ids)
    ```
    **视图 3 被 `_update_states` 改写**——`req_state.block_ids` 已经是 worker_reclaim_sync.py 截短后的列表，`new_ids` 来自 scheduler_output.scheduled_cached_reqs.new_block_ids（已被 `_patch_scheduler_output_for_compressed_reqs` 裁剪过）。
  - **关键**：`_patch_scheduler_output_for_compressed_reqs` 裁剪的阈值基于「上一次事件」的 `retained_cache_len`，**与当前 step 真实的 num_computed_tokens 不严格对齐**（因为 effective_len_tracker 的偏移）。若 `_patch_scheduler_output_for_compressed_reqs` 误判（详见 2.1），new_block_ids 过长，`block_ids.extend(new_ids)` 会把视图 3 推回到 32k 长度。
- **因果推导**：视图 3 在「worker_reclaim_sync 写入 → _update_states 续写」两个时点之间存在**双重写入竞争**。当 `_patch_scheduler_output_for_compressed_reqs` 与 worker_reclaim_sync 的截断值不一致时，**最终视图 3 长度 = max(worker_reclaim_sync_required, _update_states_需要的) = _update_states_需要的**，**视图 1 维持 worker_reclaim_sync 截短值**——视图 1 与视图 3 失同步，`compute_slot_mapping` 会读到 `block_table.np[req_idx, :num_blocks_per_row[req_idx]]` 与 `req_state.block_ids` 不一致的物理 block id 列表，**NPU attention 收到错误的 slot_mapping**。

### 2.7 [P1] `_patched_engine_core_step_with_batch_queue` 把后续 batch 推回原 batch（`schedule` 跳过 `self.scheduler.schedule()`），可能饿死 prefill trigger

- **位置**：`triattention/vllm/runtime/integration_monkeypatch.py:594-684`
- **可疑代码段**（`integration_monkeypatch.py:613-625`）：
  ```python
  boundary_pending = _batch_queue_has_pending_compression_boundary(batch_queue)
  if self.scheduler.has_requests() and not boundary_pending:
      scheduler_output = self.scheduler.schedule()
      ...
  ```
- **问题**：
  - 当 `boundary_pending=True`（说明 batch_queue 中已有 trigger step），**直接跳过 `self.scheduler.schedule()`**，走 `elif not batch_queue: return None, False` 的兜底分支。
  - 在 batch_queue 持续有 trigger step 的高并发下，新请求永远不被 schedule，触发信号堆积在 batch_queue，**物理 Block 永远没机会被释放**（视图 4 永远没机会被写入）。
- **因果推导**：32k 典型 prefill 场景下，所有 prefill 阶段都被 `defer_prefill` 静默丢弃，到 decode 阶段才有第一次 trigger 落地——**这意味着 32k 场景下 prefill 阶段累积的 Block 永远无法被 TriAttention 主动释放**，完全靠 decode 阶段回收。

### 2.8 [P1] `_apply_worker_block_reclaim_events` 中 `_clear_table_row_tail` 在 `use_hybrid_blocks=True` 时索引语义未变但语义错位

- **位置**：
  - `vllm_ascend/worker/block_table.py:69-72` 与 198-213（`_convert_physical_to_logical_blocks`）
  - `triattention/vllm/runtime/worker_reclaim_sync.py:49-62 _clear_table_row_tail`
- **问题**：当 `use_hybrid_blocks=True`（Ascend 上 `kernel_sizes != [0]` 时），**逻辑 block size < 物理 block size**，`num_blocks_per_row` 计数是「逻辑块」数；`worker_reclaim_sync.py:57-62`：
  ```python
  start = max(0, min(int(used_blocks), int(block_table_np.shape[1])))
  block_table_np[req_index, start:] = 0
  ```
  `used_blocks` 来自 `num_blocks_per_row[req_index]`（逻辑块数），但 `block_table_np` 也是按逻辑块展平（`vllm_ascend/worker/block_table.py:77 self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, ...)`）—— **索引上是匹配的，理论上 OK**。
- **真正可疑**：remap_tail 路径下 `_rewrite_table_row` 直接 `add_row(block_ids, req_index)`，`add_row` 内部 `num_blocks_per_row = 0` 然后 `append_row(block_ids)`（`vllm_ascend/worker/block_table.py:103-105`）。当 `block_ids` 来自 `block_ids_after`（按物理 block id 给的），`append_row` 会调 `_convert_physical_to_logical_blocks`（`vllm_ascend/worker/block_table.py:94-95`），把 1 个物理块拆成 `blocks_per_phys_block` 个逻辑块——**这意味着 num_blocks_per_row 翻倍**。**`_apply_compression_events` 中的 `expected_shrink_gids`/`required_blocks` 仍按物理块算（与 `retained_cache_len` 一致），但 worker_reclaim_sync.py 的视图 1 已按逻辑块算——两侧对不上**。
- **因果推导**：典型 Ascend 部署（`kernel_sizes=[0]` 或 `[16]`）一般走 `use_hybrid_blocks=False`，但若用户自定义了 `kernel_block_sizes`（`vllm_ascend/worker/npu_input_batch.py:43-50`），这个不匹配会让 `num_blocks_per_row` 在 remap_tail 路径下被错误地撑大，下一次 `append_row` 又把物理 block id 写回错误 slot。

### 2.9 [P1] `enable_zero_copy_recency=True` + `zero_copy_recency_only_on_ascend=True` 在典型 32k+2k 场景会优先走 zero-copy remap 路径

- **位置**：
  - `triattention/vllm/runtime/hook_group_pipeline.py:77-176 try_build_recency_tail_block_remap`
  - `triattention/vllm/runtime/hook_impl.py:222-261`
- **问题**：
  - `try_build_recency_tail_block_remap` 返回的 `GroupPipelineOutcome(selection_mode="zero_copy_tail", reclaim_mode="remap_tail", ...)` 是**纯 block table remap**，没有 KV tensor 物理移动；
  - 在 32k → 2k 场景：原始 block 数 ≈ 32k/16=2048，预算 block 数 = 2k/16=128，`removed_block_ids` 长度 ≈ 1920；
  - `worker_reclaim_sync.py:203-219` remap_tail 路径仅 `_rewrite_table_row` 重写 `block_table.np` 与 `num_blocks_per_row`，**视图 2 被完全重写为新物理 block id 列表**；
  - 但 `req_state.block_ids` 也在 `worker_reclaim_sync.py:241-259` 被整体替换——若 `_block_ids_after` 校验失败（详见 2.2），**视图 2 与视图 3 写入的 block_ids 不一致**（视图 2 是 remap 的目标 block_ids，视图 3 是 truncate 等价的 `[required_blocks:] = 0`）。
- **因果推导**：zero-copy remap_tail 路径**完全依赖 `_block_ids_after` 校验通过**。校验失败时退化为「截前 N 块」，**这是与 zero-copy 语义相反的逻辑**——remap_tail 应该是「保留最近 N 块」，但 truncate_tail 等价是「保留最早 N 块」。**这会让 attention 拿到的 KV 内容错位**（虽然不直接致「并发无提升」）。

### 2.10 [P1] `try_build_recency_tail_block_remap` 在 multi-group 下要求 `cache_len_after` 一致，否则 return None

- **位置**：`triattention/vllm/runtime/hook_group_pipeline.py:143-146`
  ```python
  if cache_len_after is None:
      cache_len_after = int(group_cache_len_after)
  elif cache_len_after != int(group_cache_len_after):
      return None
  ```
- **问题**：multi-group 场景下（如 MLA 主+辅 group），各 group 的物理布局不同，保留的 tail 长度可能不一致；一旦不一致，**整个 remap_tail 路径直接放弃**，回退到 truncate_tail。
- **因果推导**：NPU MLA 模型（`vllm_ascend/attention/mla_v1.py`）可能有这种 multi-group 布局；这会触发回退到 truncate_tail 路径，从而走 2.2/2.3 描述的脆弱分支。

### 2.11 [P1] `_block_ids_after` 严格校验导致合法路径被静默放弃

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:36-46 _block_ids_after`
  ```python
  def _block_ids_after(group):
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
  ```
- **问题**：
  - 校验项 1：`block_ids_after` 必须是 Python `list`（不是 `tuple`、不是 `np.ndarray`）。
  - 校验项 2：所有元素必须是 Python `int`（**不是 `numpy.int64` 或 `torch.Tensor.item()`**）。
  - 校验项 3：去重后长度等于原长度（**严格无重复**）。
- **因果推导**：
  - hook 端 `finalize_hook_placement_result`（`hook_group_pipeline.py:494-543`）把 `outcome.mutable_block_ids_by_group`（list of list of int）写入 `req_state.block_ids`，但 `block_reclaim_groups` 中的 `block_ids_after` 来自 `ReclaimGroup` 的构造（`hook_group_pipeline.py:147-156`），其中 `kept_block_ids` 是 `list(normalized_block_ids[start_block:before_required])`（Python list of int）。**这一段是纯 Python int**，理论上能通过校验。
  - 但 `try_build_recency_tail_block_remap` 的 `kept_tail_block_ids = list(normalized_block_ids[start_block:before_required])`（`hook_group_pipeline.py:130`）也是 Python list——也是 OK。
  - **真正风险点**：multi-group 场景下，`seen_gids` 集合（`scheduler.py:718-736`）遍历的 `group["block_ids_after"]` 来自 `event["block_reclaim"]["groups"]`，**该字段在 hook 端 `finalize_hook_placement_result` 中组装**——若 hook 在 `_core_trace` 等错误处理路径中返回 `pipeline_out` 为 dict 而非 `GroupPipelineOutcome`，**block_reclaim 字段被置为 None**——`_event_reclaim_groups`（`worker_reclaim_sync.py:16-33`）走 `if not isinstance(block_reclaim, dict): return "truncate_tail", {}` 兜底——`groups_by_gid` 为空 dict，**remap_tail 路径下所有 gid 都不命中，进入 truncate_tail 兜底**。这是「remap_tail 退化为 truncate_tail」的最常见路径。

### 2.12 [P2] 视图 5（Hook 内 `req_state.block_ids`）与视图 3 的写入者协调

- **位置**：
  - 视图 5 写入：`triattention/vllm/runtime/hook_group_pipeline.py:452-453`（`mutable_block_ids_by_group[gid] = list(group_outcome.kept_block_ids)`）与 494-543（`finalize_hook_placement_result` 中的 `req_state.block_ids = ...`）
  - 视图 3 写入：`triattention/vllm/runtime/worker_reclaim_sync.py:236-266`
- **问题**：
  - 视图 5 的「hook 写」+ 视图 3 的「worker_reclaim_sync 写」**两次都修改 `req_state.block_ids`**：
    - 视图 5：`req_state.block_ids = tuple(reassigned_block_ids) if isinstance(original, tuple) else reassigned_block_ids`——**整体重写**；
    - 视图 3（truncate_tail）：`for group_blocks in block_ids_attr: del group_blocks[required_blocks:]`——**原地截断**。
  - 若视图 5 整体重写为 `list[list[int], list[int]]`（list of list），视图 3 的 `for group_blocks in block_ids_attr` 仍然能 iterate（list 可迭代），`del group_blocks[required_blocks:]` 仍能 mutate 内层 list。
  - 若视图 5 重写为 `tuple[list[int], list[int]]`（tuple of list），视图 3 的 `for group_blocks in block_ids_attr` 仍能 iterate，但**`del group_blocks[required_blocks:]` 报 AttributeError**（tuple 不可变）。这会被 `worker_reclaim_sync.py:266` 的「外层无 try/except」直接抛出——但**这个异常会被 `apply_worker_block_reclaim_events` 的调用方吞掉**（`triattention/vllm/runtime/runner.py:1171` 直接调用，无 try/except），会让 `_apply_worker_block_reclaim_events` 异常抛出，**后续 `execute_base_model_with_effective_overrides` 不会跑**。
- **因果推导**：Hook 端 `finalize_hook_placement_result` 中 `req_state.block_ids` 的容器类型由 `original_block_ids_by_group` 的类型决定（`if isinstance(original, tuple)`）——若 `original_block_ids_by_group` 是 tuple，**视图 5 写完是 tuple of list**；视图 3 后续的 `del` 会抛 AttributeError。**这是 worker_reclaim_sync.py 与 hook 端的耦合陷阱**。

### 2.13 [P2] `block_pool._maybe_evict_cached_block` 在「`enable_prefix_caching=True`」下不会清空 prefix cache

- **位置**：`vllm/v1/core/block_pool.py:352-390`，`triattention/vllm/runtime/scheduler.py:43-46 _evict_reclaimed_block_metadata`
- **问题**：当 `enable_prefix_caching=True`（用户未按 README 显式关掉）时，被 reclaim 的 block 仍带 `block_hash`，会留在 `cached_block_hash_to_block` 中。`_maybe_evict_cached_block` 仅当 `cached_block_hash_to_block.pop(block_hash, block.block_id) is not None` 时返回 True 并 reset；`free_blocks` 仅在 `ref_cnt == 0` 时 push 到 free_block_queue。当 prefix cache 命中率较高时，**被 reclaim 的 block 因 hash 表中的引用让 `free_block_queue` 不增长，`kv_cache_usage` 不降**。
- **因果推导**：这是物理回收层面的副作用，但**对逻辑驱逐的影响有限**——视图 1+2+3 仍由 worker_reclaim_sync.py 写到位。仅当用户漏配 `--enable-prefix-caching false` 时，**并发水位仍可能上不去**（物理水位不下）。

### 2.14 [P2] `select_keep_indices is None` 时 hook 抛 `TRITON_SCORING_REQUIRED_MARKER`，事件被 `mark_compression_skipped` 标记

- **位置**：`triattention/vllm/runtime/hook_impl.py:208-211`、`triattention/vllm/runtime/executor.py:101-120`
- **问题**：当 `select_keep_indices is None`（即 selector 不可用、例如 Triton 编译失败且 PyTorch fallback 也失败）时，hook 抛 `RuntimeError(TRITON_SCORING_REQUIRED_MARKER:selector_unavailable:...)`。`runner_compression_actions.py:101-120` 捕到后：strict 模式直接 re-raise；非 strict 模式 `state_store.mark_compression_skipped(reason="executor_exception:...")`。此时**没有任何 `applied` 事件**，**`apply_worker_block_reclaim_events` 收到空 events 列表直接 return**（`worker_reclaim_sync.py:124-125`），**视图 1+2+3 都不写**。
- **因果推导**：仅当 Triton 编译失败且 PyTorch fallback 同时不可用时才会触发；典型场景是 NPU 上 Triton kernel 不可用、`scoring_backend` 配置为 `triton` 而非 `auto` 时。但 vLLM-Ascend 的 Triton 兼容性问题（参见 vllm-ascend docs）确实存在，需要排查 `select_keep_indices is not None` 的实际后端。

### 2.15 ★ [P1] 视图 4 写入的入口位置 `scheduler.py:710-714` 紧邻 `reclaim_applied_any` 检查，逻辑与 2.4 高度耦合

- **位置**：`triattention/vllm/runtime/scheduler.py:707-714`（synthesized 路径内）与 877-881（主路径内）
- **问题**：
  ```python
  # 707-714 synthesized 路径：
  if reclaim_applied_any:
      update_request_effective_kv_offset(
          request=req,
          cache_len_after=cache_len_after,
      )
  continue
  # 877-881 主路径：
  if reclaim_applied_any:
      update_request_effective_kv_offset(
          request=req,
          cache_len_after=cache_len_after,
      )
  ```
- **两个写入点的 `reclaim_applied_any` 状态可能不一致**：
  - synthesized 路径：`reclaim_applied_any` 由 685-708 行 `for gid in sorted(expected_shrink_gids)` 循环中 `_free_reclaimed_blocks(manager, removed_blocks)` 决定；
  - 主路径：`reclaim_applied_any` 由 760-866 行 `for group in groups` 与 853-867 行 `for gid in sorted(missing_gids)` 两个循环共同决定。
- **因果推导**：synthesized 路径在 prefill 阶段被 `if _evt_scheduled > 1: ...`（scheduler.py:685）跳过，**`reclaim_applied_any` 永远为 False**；主路径在 32k 场景下走 truncate_tail，`reclaim_applied_any` 取决于 `removed_old_blocks` 是否有非空元素。**两侧的失败模式叠加**会让视图 4 在大多数 step 都不被写入。

### 2.16 [P3] `effective_overrides.py:build_effective_sparse_overrides` 在 `state.current_cache_len_semantics != "effective_pre_step"` 时退回 fallback

- **位置**：`triattention/vllm/runtime/effective_overrides.py:49-96`
- **问题**：
  - `state_store.mark_compressed`（`state.py:101-127`）会写 `state.current_cache_len_semantics = "estimated_with_scheduled"`（非 `"effective_pre_step"`）；
  - `_state_marks_effective_pre_step_base`（`effective_overrides.py:49-70`）仅在 `semantics == "effective_pre_step"` 且 `state_step == scheduler_step` 时返回 True；
  - 退回 fallback：`_effective_base_before_step` 返回 `max(0, current_cache_len - max(0, scheduled_tokens))`——**这个差值未必等于 `cache_len_after`**。
- **因果推导**：当 `state.current_cache_len_semantics` 标错或 step 错位时，**输入 Patch 的 `seq_bases[req_idx] = effective_before_step` 不准确**——NPU 收到的 `seq_len` 与 `cache_len_after` 不严格相等，**attention 看到的 KV 序列长度略微偏长**（多算几个被驱逐但还没被 prefix cache 清理的 token），但不会直接致「并发无提升」。

### 2.17 [P3] `_patched_kv_cache_allocate_slots` 中 `delay_cache_blocks=True` 反复触发

- **位置**：`triattention/vllm/runtime/integration_monkeypatch.py:526-534`
- **问题**：`delay_cache_blocks=True` 让 vLLM 的 `allocate_slots` 末尾的 `cache_blocks(request, num_computed_tokens)` 跳过（推迟），这导致 prefix cache 复用被关闭（即使 `--enable-prefix-caching true`），`cached_block_hash_to_block` 不再更新；后续请求的 prefix-cache 命中永远找不到 TriAttention 已压缩的 block。
- **因果推导**：仅当 user 错误开启 prefix cache 时副作用明显；正常用法（`--enable-prefix-caching false`）下不影响。

### 2.18 综合因果链

把上述 P0/P1 串起来，可以形成如下「**逻辑驱逐生效、并发水位不升**」的最有可能路径：

1. 32k 典型 prefill 走 chunked（`max_num_batched_tokens=1024`），至少 32 个 step；
2. 第 1 次压缩 trigger 在 `num_computed_tokens ≈ 2304` 时触发（`threshold = 2048 + 256`），`signal.scheduled_tokens=1`（decode），进入 `run_group_compaction_pipeline`；
3. `try_build_recency_tail_block_remap` 走 remap_tail 路径（`enable_zero_copy_recency=True`，2k 预算是 128 整 block）；
4. `finalize_hook_placement_result` 写视图 5（`req_state.block_ids` 整体重写）；
5. **`apply_worker_block_reclaim_events` 走 remap_tail 路径**（视图 1+2+3 都被写）；
6. `attach_execute_model_compression_events` 把事件挂到 `scheduler_output.triattention_compression_events`；
7. `update_from_output` 调 `_apply_compression_events`，**`reclaim_applied_any` 取决于 `_free_reclaimed_blocks` 是否真的推进 free_block_queue**：
   - **若 `removed_old_blocks` 为空**（典型：zero-copy remap_tail 的 `block_ids_removed` 在 `keep_block_ids = kept_tail_block_ids + trailing_block_ids`（`hook_group_pipeline.py:138`）拼接下，**`removed_block_ids` 是 list(normalized_block_ids[:start_block])**（`hook_group_pipeline.py:139`）—— 是有内容的，但 `block_ids_after` 中的 `trailing_block_ids` 在 `reassigned_block_ids` 中也保留了，**调度端的 reassembled 仍包含所有物理 block**——`_free_reclaimed_blocks` 仅对前段 block 调 `block_pool.free_blocks`，**这些 block 的 `ref_cnt` 已经是 1（前段只有本请求用）**，所以**能正常归还**，`reclaim_applied_any=True`）。
   - **但视图 4 的写入**：`reclaim_applied_any=True` → `update_request_effective_kv_offset` 写入 `_triattention_effective_kv_offset = logical - effective = 32768 - 2048 = 30720`。**这一次写入是正确的**。
8. 下次 `kv_cache_manager.allocate_slots(request, num_new_tokens=1)`：
   - `_patched_kv_cache_allocate_slots` 看到 `effective_num_computed = 2048`，`num_new_tokens = 1` → **只需分配 1 个新 block**；
   - `block_pool.free_blocks`（物理回收路径）在分配前自动调用——`block_pool.get_usage()` 应当下降。
9. **但 `kv_cache_usage` 仍然高**——为什么？

**真正的瓶颈点**：在 chunked prefill 阶段（`scheduled_tokens > 1`），**视图 4 没机会被写入**（2.15），所以 prefill 阶段的 32k 个 block 持续按 32k 分配。**只有 decode 阶段（`scheduled_tokens == 1`）才能让视图 4 写入并让 `allocate_slots` 走 effective 路径**——但到那时，prefill 累积的 32k block 已经分配完毕，需要靠**后续 N 个 decode step 逐步把 effective_num_computed 累积回收**才能让 `block_pool.get_usage()` 真正下降。

**最可能根因候选组合**：2.1（`_patch_scheduler_output_for_compressed_reqs` 缺 events 跳过裁剪） + 2.2（remap_tail 校验失败时视图 1+2+3 静默不写） + 2.4（视图 4 写入依赖物理回收成败） + 2.6（视图 1+2+3 与 `_update_states` 续写时序错位）联动。

---

## 模块三：Print 断点日志排查实施方案

下文为「**纯日志（`print`/`logger.info`）** 断点方案，**不修改任何业务逻辑、不改变任何代码执行流程**」。所有断点位置选定原则：

- 覆盖 P0/P1 关键路径上的关键变量与状态；
- 围绕 worker_reclaim_sync.py 的视图 1+2+3 写入路径密集布点；
- 用统一的 `TRIATTN_LOGICAL_EVICT_DEBUG` 前缀便于 grep 过滤；
- 在用户最关心的「Block Table 视图是否真被截短」与「视图 4 是否真被写入」上加观测点；
- 单条日志尽量短（< 200 字符），避免日志爆炸；
- 在每个函数中只插 1~3 行 `print`，便于回退。

> **强制约束**：本文档不实施任何代码修改；下列每条断点仅给出「文件 / 函数 / 行号 / 打印内容 / 排查目的 / 判定标准」五项，**严禁把打印代码直接复制到生产**。需要执行时由后续任务单独按 1~N 条插入。

### 3.1 断点编号约定

格式：`LE-<模块>-<序号>`（Logical Eviction），示例 `LE-W-001` = Worker reclaim_sync 侧 001 号断点。

### 3.2 worker_reclaim_sync.py 侧断点（LE-W-*，本任务最关注）

#### LE-W-001 `apply_worker_block_reclaim_events` 入口与短路

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:124-125`（`if not isinstance(events, list) or not events: return` 之前）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-001] n_events=... n_applied=... disabled_by_env=... block_table_resolved=... req_id_to_index_resolved=... n_tables=... block_size=...
  ```
- **排查目的**：观察函数是否被调用、是否被 env 短路、能否解析到 NPU BlockTable 与 req_id 映射。
- **判定**：
  - `n_events==0` 或 `n_applied==0` → 上游 2.14/2.11 排查；
  - `disabled_by_env=True` → 立即停止排查，env 配置错误；
  - `block_table_resolved=False` → 走 V2 runner 短路（`worker_reclaim_sync.py:130-136`），hook 端应该已经写过 BlockTable，**改为排查 hook 端写入**；
  - `req_id_to_index_resolved=False` → 走 1.4.2 短路 5，整个函数无效。

#### LE-W-002 per-event 处理决策

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:172-194`（`for event in events:` 循环内）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-002] req=... status=... req_index=... cache_len_after=... retained_cache_len=... required_blocks=... reclaim_mode=... n_groups=...
  ```
- **排查目的**：观察每个 applied 事件在 worker_reclaim_sync 入口的字段值。
- **判定**：
  - `req_index is None` → req 尚未加入 input_batch（`worker_reclaim_sync.py:178-180`），本 step 不会动视图 1+2+3；
  - `cache_len_after <= 0` → 事件异常，跳过；
  - `retained_cache_len < cache_len_after` → 来自 `details` 的字段缺失或异常（详见 2.1）。

#### LE-W-003 per-gid 视图 1+2 写入决策（truncate_tail 主路径）

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:220-227`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-003] req=... gid=... reclaim_mode=... current_num_blocks=... required_blocks=... view1_will_shrink=... view2_will_clear_from=... n_tables=...
  ```
- **排查目的**：观察视图 1（num_blocks_per_row）的截断值、视图 2（block_table_np）的清零起点、是否真的写入。
- **判定**：
  - `view1_will_shrink=False` 且 `reclaim_mode=truncate_tail` → 视图 1 不动，可能 `required_blocks >= current`（极端：32k → 0 的极端压缩、`required_blocks > current`、或 `retained_cache_len` 异常大）；
  - `view1_will_shrink=True` 但后续 B-S-008 显示 `allocate_slots` 没走 effective 路径 → 走 2.4 排查。

#### LE-W-004 per-gid 视图 1+2 写入决策（remap_tail 路径）

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:203-219`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-004] req=... gid=... reclaim_mode=remap_tail block_ids_after_n=... block_ids_after_valid=... rewrite_ok=... view1_will_set_to=...
  ```
- **排查目的**：观察 remap_tail 路径下 `_block_ids_after` 校验结果与 `_rewrite_table_row` 成功与否。
- **判定**：
  - `block_ids_after_valid=False` → 走 2.2 排查（`block_ids_after` 含 None / 重复 / 非 int）；
  - `block_ids_after_valid=True` 但 `rewrite_ok=False` → 走 2.3 排查（hybrid blocks 索引错位）；
  - `block_ids_after_valid=True` 且 `rewrite_ok=True` → 视图 1+2 正常写入。

#### LE-W-005 视图 3（req_state.block_ids）写入决策

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py:236-266`（`requests_dict = ...; if isinstance(requests_dict, dict):` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-005] req=... req_state_present=... block_ids_container_type=... block_ids_group_count=... reclaimed=truncate_tail/remap_tail/none container_will_change=.../no
  ```
- **排查目的**：观察视图 3 是否真的被 mutate 或替换。
- **判定**：
  - `req_state_present=False` → `base_runner.requests` 中没有该 req（可能被 condense 清掉）；
  - `block_ids_container_type=tuple` 且 `reclaimed=truncate_tail` → **AttributeError 风险**（`del` 不能在 tuple 上跑，详见 2.12）；
  - `container_will_change=no` 但 `reclaimed != none` → 视图 3 与视图 1+2 失同步（`worker_reclaim_sync.py:260-266` 的 `del` 实际没改 list 因为 `len(group_blocks) <= required_blocks`）。

#### LE-W-006 视图 1+2 与视图 3 同步校验

- **位置**：`triattention/vllm/runtime/worker_reclaim_sync.py` 末尾（`for event in events:` 循环结束前）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-W-006] req=... view1_per_gid=[...] view3_group_lens=[...] sync=ok/diverged max_diff=...
  ```
- **排查目的**：检查视图 1（`num_blocks_per_row`）与视图 3（`req_state.block_ids` 各 group 长度）是否一致。
- **判定**：
  - `diverged=True` 且 `max_diff > 1` → 走 2.3 排查（multi-group 下 `required_blocks` 单一值不适用）；
  - `diverged=True` 且 `max_diff == 0` 但 `view3_group_lens != [required_blocks for _ in range(G)]` → 视图 3 之前的写入被覆盖（2.6 排查）。

### 3.3 Runner 侧断点（LE-R-*）

#### LE-R-001 `_execute_compression_actions` 结束后的 pending 事件

- **位置**：`triattention/vllm/runtime/runner.py:1161-1164`（`_execute_compression_actions` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-R-001] step=... n_events=... n_applied=... n_skipped=... event_reqs_with_block_reclaim=[...] event_reclaim_modes=... event_block_reclaim_none_count=...
  ```
- **排查目的**：观察 `execute_runner_compression_actions` 实际产生的事件结构（重点：哪些事件带 `block_reclaim` 字段、哪些没有）。
- **判定**：
  - `n_applied=0` 或 `event_reqs_with_block_reclaim` 全空 → 走 2.11/2.14 排查；
  - `event_block_reclaim_none_count > 0` 比例高 → 走 2.11 排查（hook 端 `block_reclaim` 字段未生成）。

#### LE-R-002 `_apply_worker_block_reclaim_events` 结束

- **位置**：`triattention/vllm/runtime/runner.py:1171-1175`（`_apply_worker_block_reclaim_events` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-R-002] step=... events_processed=... block_table_writes=... remap_writes=... truncate_writes=... skipped_writes=...
  ```
- **排查目的**：在 `_apply_worker_block_reclaim_events` 内层插入计数（需要修改该函数，建议改函数签名为「返回统计 dict」—— 但本任务不允许改函数，所以在外层改用 `_pending_compression_events` 数量与 `req_state.block_ids` 长度做间接观测）。
- **判定**：
  - `block_table_writes=0` 且 `events_processed > 0` → 走 LE-W-001~LE-W-006 排查；
  - `remap_writes=0` 且 `_pending_compression_events` 中有 `reclaim_mode=remap_tail` → 走 2.2 排查。

#### LE-R-003 `_patch_scheduler_output_for_compressed_reqs` 触发与裁剪结果

- **位置**：`triattention/vllm/runtime/runner.py:1048-1057`（`if changed: new_block_ids_list[i] = trimmed` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-R-003] req=... group=... before_max=... after_max=... retained_cache_len=... group_limits=... events_by_req_id_hit=True/False
  ```
- **排查目的**：观察 `new_block_ids` 裁剪是否真的发生（与 2.1 联动）。
- **判定**：
  - `events_by_req_id_hit=False` → 走 2.1 排查（本次 step 无 compression event 但 state.compression_count>0）；
  - `before_max == after_max` → 裁剪无效，`group_limits` 全是 None（retained_cache_len 缺失）。

#### LE-R-004 `_needs_effective_input_overrides` 决策

- **位置**：`triattention/vllm/runtime/runner.py:1059-1081`（`_needs_effective_input_overrides` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-R-004] step=... need_overrides=... has_compressed_request_in_scheduled=... scheduled_reqs=[...]
  ```
- **排查目的**：观察本次 step 是否需要激活输入 Patch（与视图 4 的「视图 1+2+3 写入后立即被 NPU forward 看到」配合）。
- **判定**：
  - `need_overrides=False` 但 `state.compression_count > 0` → `state_store.has_compressed_request_in` 误判或 scheduler_output.scheduled_cached_reqs 缺 req_id；
  - `need_overrides=True` 但 `effective_overrides` 返回空 → 走 2.16 排查（`current_cache_len_semantics` 标错）。

### 3.4 Scheduler 侧断点（LE-S-*）

#### LE-S-001 视图 4 写入与 `reclaim_applied_any` 联动

- **位置**：`triattention/vllm/runtime/scheduler.py:707-714`（synthesized 路径内）与 877-881（主路径内），分别打印
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-S-001] req=... path=synthesized/main reclaim_applied_any=... update_offset_called=True/False offset_value=... logical=... effective=...
  ```
- **排查目的**：直接观测视图 4 写入条件（与 2.4/2.15 联动）。
- **判定**：
  - `path=synthesized` 且 `reclaim_applied_any=False` 且 `_evt_scheduled > 1` → 2.5/2.15 触发；
  - `path=main` 且 `reclaim_applied_any=False` 但 `removed_old_blocks` 非空 → `_free_reclaimed_blocks` 失败（ref_cnt 不归零，详见 2.13）。

#### LE-S-002 `_apply_compression_events` 入口

- **位置**：`triattention/vllm/runtime/scheduler.py:571-587`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-S-002] n_events=... n_applied=... source=... event_reqs=[...] event_reclaim_modes=[...] event_block_reclaim_none=...
  ```
- **排查目的**：与 LE-R-001 配合，验证事件从 model_runner_output / scheduler_output 回到 `_apply_compression_events`。
- **判定**：与 LE-R-001 对照，源应该一致。

#### LE-S-003 主路径物理回收调用

- **位置**：`triattention/vllm/runtime/scheduler.py:760-866`（`for group in groups:` 与 `for gid in sorted(missing_gids):` 循环内，分别打）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-S-003] req=... gid=... path=main/missing_gids reclaim_mode=... removed_n=... free_blocks_called=... reclaim_applied_any_so_far=...
  ```
- **排查目的**：观察主路径下 `_free_reclaimed_blocks` 的实际调用次数。
- **判定**：
  - `free_blocks_called=0` 且 `path=main` → 主路径根本没走 `removed_old_blocks` 截断，**视图 4 不会写入**（2.4 触发）；
  - `free_blocks_called>0` 但 `reclaim_applied_any_so_far` 没累加 → `_free_reclaimed_blocks` 内部异常。

#### LE-S-004 synthesized 路径下 prefill 跳过

- **位置**：`triattention/vllm/runtime/scheduler.py:685-693`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-S-004] req=... groups=is_list=... _evt_scheduled=... expected_shrink_gids=... skipped_prefill=... reclaimed_in_synthesize=...
  ```
- **排查目的**：定位 2.5 的 prefill-skip。
- **判定**：
  - `skipped_prefill=True` 且 `_evt_scheduled > 1` → 直接命中 2.5；
  - `reclaimed_in_synthesize > 0` → prefill 阶段也走了 synthesize 路径。

### 3.5 Hook / Selector 侧断点（LE-H-*）

#### LE-H-001 Hook entry 与 selector 选择

- **位置**：`triattention/vllm/runtime/hook_impl.py:97-111`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-H-001] req=... step=... selector_status=... select_keep_indices_present=... effective_tokens=... budget_total=... under_budget=... prefill_exceeds=...
  ```
- **排查目的**：观察 hook 是否被调用，以及 `selector_status` 实际值。
- **判定**：
  - `selector_status == "none"` 持续 → 走 2.14 排查；
  - `under_budget=True` → 跳到 `applied=False reason="under_budget"`，**视图 1+2+3 不写**。

#### LE-H-002 zero-copy recency 路径命中

- **位置**：`triattention/vllm/runtime/hook_impl.py:229-250`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-H-002] req=... step=... path=zero_copy_tail cache_len_after=... reclaim_groups=... reclaim_mode=...
  ```
- **排查目的**：观察 zero-copy remap 路径是否真的命中（与 2.9 联动）。
- **判定**：
  - `path=zero_copy_tail` 持续出现 → 32k 典型场景下都走 remap_tail；
  - `reclaim_groups=0` 但 `path=zero_copy_tail` → 内部 bug（`try_build_recency_tail_block_remap` 返回了 `block_reclaim_groups=[]` 的 outcome）。

#### LE-H-003 `run_group_compaction_pipeline` 出口与 `finalize_hook_placement_result` 写入

- **位置**：`triattention/vllm/runtime/hook_group_pipeline.py:475-491`（创建 `GroupPipelineOutcome` 之后）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-H-003] req=... step=... groups=... selection_mode=... cache_len_after=... reclaim_groups=... reclaim_mode=... block_reclaim_payload_present=...
  ```
- **排查目的**：观察 hook 端写入的 `block_reclaim` 字段是否齐全（与 LE-W-001~LE-W-004 对照）。
- **判定**：
  - `reclaim_groups=0` 且 `reclaim_mode=truncate_tail` → 走 2.11 排查（hook 端没生成有效 groups）；
  - `block_reclaim_payload_present=False` → 走 2.11 排查（`outcome.block_reclaim_groups` 为空）。

#### LE-H-004 `try_build_recency_tail_block_remap` 失败原因

- **位置**：`triattention/vllm/runtime/hook_group_pipeline.py:117-160` 函数中各 `return None` 分支前
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-H-004] req=... step=... reason=... gid=... budget_blocks=... before_required=... group_capacity_tokens=...
  ```
- **排查目的**：定位 zero-copy remap 路径为何回退（multi-group 不一致 / 预算非整 block / 早 return 等）。
- **判定**：
  - 连续多 gid 的 `cache_len_after` 不一致 → 走 2.10 排查；
  - `budget_total % block_size != 0` 持续 → 检查用户配置 `kv_budget`；
  - `len(normalized_block_ids) < before_required` → 走 hook 端 `mutable_block_ids_by_group` 初始化问题排查。

### 3.6 物理回收层（vLLM 自身）断点（LE-P-*）

#### LE-P-001 `block_pool.free_blocks` 入口的 ref_cnt 变化

- **位置**：`vllm/v1/core/block_pool.py:409-417`（在 `block_pool.py` 上层加一层包装，但本期不做；改用 Worker/Side LE-S-003 间接观测）
- **替代观测点**：`triattention/vllm/runtime/scheduler.py:48-58 _free_reclaimed_blocks` 中在 `block_pool.free_blocks` 前后打印 `block_pool.get_free_block_count()` 差值
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-P-001] req=... removed_n=... free_block_count_before=... free_block_count_after=... delta=...
  ```
- **排查目的**：直接观测 Block 池自由队列是否真的变长（最权威的物理释放指标，**与视图 4 写入强相关**）。
- **判定**：
  - `delta == 0` 且 `removed_n > 0` → 走 2.13 排查（ref_cnt 不为 0 阻止入队）；
  - `delta < removed_n` → 部分 block 仍被 ref；
  - `delta == removed_n` → 物理回收成功，**视图 4 应当被写入**（与 LE-S-001 对照）。

#### LE-P-002 `block_pool._maybe_evict_cached_block` 命中

- **位置**：在 `_evict_reclaimed_block_metadata`（`triattention/vllm/runtime/scheduler.py:35-46`）调用前后
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-P-002] req=... block_id=... block_hash_present=... evict_returned=... prefix_cache_cleared=...
  ```
- **排查目的**：观察 prefix-cache 清理是否真的发生（与 2.13 联动）。
- **判定**：`prefix_cache_cleared=False` 持续 → 即使 ref_cnt 归 0，prefix cache 仍引用，free_block_queue 不增长。

### 3.7 输入 Patch 侧断点（LE-I-*）

#### LE-I-001 `build_effective_sparse_overrides` 输出

- **位置**：`triattention/vllm/runtime/effective_overrides.py:230-384`（`build_effective_sparse_overrides` 函数返回前）
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-I-001] has_active_compressed=... n_sparse_overrides=... n_pos_deltas=... single_seq_base=... single_pos_delta=... sample_seq_bases={req_idx: base, ...} sample_pos_deltas={req_idx: delta, ...}
  ```
- **排查目的**：直接观测输入 Patch 生成的 `seq_bases` 与 `pos_deltas`（与视图 1+2 配合）。
- **判定**：
  - `n_sparse_overrides=0` 但 `has_active_compressed=True` → 走 2.16 排查（`current_cache_len_semantics` 标错）；
  - `n_pos_deltas=0` → 所有 `delta == 0`，**NPU 仍按 32k 算 seq_len**——但这不会让 Block 池水位下不去。

#### LE-I-002 `_validate_v1_block_table_bounds` 异常

- **位置**：`triattention/vllm/runtime/input_patch_vllm_v1_backend.py:257-333`
- **打印内容**：
  ```text
  [TRIATTN_LOGICAL_EVICT_DEBUG LE-I-002] rows=... seq_len_per_row={...} capacity_per_row={...} seq_len_oob_count=...
  ```
- **排查目的**：观察 NPU 端 Block Table 容量校验是否触发。
- **判定**：
  - `seq_len_oob_count > 0` → 视图 1 与 `seq_len` 失同步（视图 1 截到 `required_blocks=128`，但 `seq_len` 仍按 32k）——**这是视图 1+2 与输入 Patch 失同步的直接信号**。

### 3.8 排查路径汇总（自上而下决策树）

按下列顺序对照日志输出即可定位「**逻辑驱逐失效**」的根因（无需修改任何代码即可执行）：

```
[1] 看 LE-W-001 / LE-R-001 / LE-R-002：worker_reclaim_sync 是否真的被调用？处理了多少 applied 事件？
    - 没被调用 / 短路 → 走 1.4.1/1.4.2 短路 3/4/5
    - 短路 4（block_table not found）→ 检查 NPU input_batch 是否真的被初始化
    - 调用了但 n_applied==0 → 走 LE-R-001 上游排查

[2] 看 LE-W-002：每个 applied 事件的 required_blocks 与 reclaim_mode 是什么？
    - required_blocks 异常大（>= 2048）→ 走 2.1（retained_cache_len 取错）
    - reclaim_mode=truncate_tail 但 current<=required → 视图 1 不会写
    - reclaim_mode=remap_tail 但 block_ids_after 校验失败 → 走 2.2/2.11

[3] 看 LE-W-003 / LE-W-004：视图 1+2 真的被写了吗？
    - view1_will_shrink=False → 视图 1 没动，走 2.6 / LE-W-006 排查
    - view1_will_shrink=True 但 view2_will_clear_from=0 → 视图 2 没清零
    - remap_tail: rewrite_ok=False → 走 2.3 (hybrid blocks) / 2.8 (multi-group)

[4] 看 LE-W-005：视图 3 (req_state.block_ids) 写入了吗？
    - req_state_present=False → base_runner.requests 缺该 req
    - block_ids_container_type=tuple 且 reclaimed=truncate_tail → AttributeError 风险（2.12）
    - container_will_change=no → 视图 3 没动

[5] 看 LE-W-006：视图 1 与视图 3 是否同步？
    - diverged=True → 走 2.3（multi-group required_blocks 单一值）
    - diverged=False 但视图 3 长度仍为原值 → 走 2.6（_update_states 续写覆盖）

[6] 看 LE-R-003：_patch_scheduler_output_for_compressed_reqs 真的裁剪了吗？
    - events_by_req_id_hit=False → 2.1 触发
    - before_max==after_max → 裁剪无效

[7] 看 LE-S-001：视图 4 (_triattention_effective_kv_offset) 写入了吗？
    - reclaim_applied_any=False → 走 2.4 / 2.15
    - update_offset_called=True 但 offset_value=0 → 视图 4 写了但值无效

[8] 看 LE-P-001：物理回收 _free_reclaimed_blocks 真的归还 block 吗？
    - delta==0 且 removed_n>0 → 走 2.13（ref_cnt 不归零）
    - delta==removed_n → 物理回收成功，**视图 4 应当被写入**（与 LE-S-001 对照）

[9] 看 LE-S-002 / LE-S-003：_apply_compression_events 真的处理事件了吗？
    - n_applied=0 但 LE-R-001 显示 n_applied>0 → 事件从 model_runner_output → scheduler_output 路径有 bug
    - free_blocks_called=0 → 走 2.5 (prefill-skip)

[10] 看 LE-S-004：prefill 阶段 synthesized 是否被跳过？
    - skipped_prefill=True → 2.5 触发
    - reclaimed_in_synthesize>0 → prefill 阶段也走了 synthesize

[11] 看 LE-H-001~LE-H-004：hook 端是否正常生成 block_reclaim？
    - selector_status=none 持续 → 2.14
    - reclaim_groups=0 → 2.11
    - path=zero_copy_tail 但 reclaim_mode != remap_tail → 内部不一致

[12] 看 LE-I-001 / LE-I-002：输入 Patch 是否与 Block Table 同步？
    - LE-I-002 seq_len_oob_count > 0 → 视图 1 与 seq_len 失同步
    - LE-I-001 n_sparse_overrides=0 → 走 2.16

最终判定（按现象定位到 P0）：
- 现象：「filter 生效但并发无提升」+「Block Table 视图 1+2+3 看起来正常」+「kv_cache_usage 持续高」
- 最可能根因：2.4（视图 4 写入依赖物理回收成败）+ 2.1（_patch_scheduler_output 跳过裁剪）
- 次可能根因：2.6（视图 1+2+3 与 _update_states 续写时序错位）
- 物理回收的「真并发」由 2.5 决定，但 32k 场景下 prefill 阶段必踩 prefill-skip
```

### 3.9 配套环境变量与采样建议

- 启用所有现有 `TRIATTN_RUNTIME_LOG_*` 开关（`README.md:271-294`）作为外围观测。
- 在并发水位异常场景下，把 `TRIATTN_RUNTIME_KV_BUDGET=2048` 与 `TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND=16` 同时保持默认；**禁止在排查期间修改**，避免多变量叠加。
- 日志采样间隔：在 `LE-W-003`、`LE-W-004` 这种高频打印点上，加 `if step % 50 == 0` 节流（建议作为第 2 步插入，再后续逐个补齐）。
- 所有 `LE-*` 断点统一加 `step=` 字段，便于在异步时间线错乱时排序。
- **建议优先插入顺序**（由最易踩到到最深层）：
  1. **LE-W-001**（确认函数是否被调用）
  2. **LE-W-003**（确认视图 1 是否被写）
  3. **LE-W-004**（确认 remap_tail 校验）
  4. **LE-W-005**（确认视图 3 是否被写）
  5. **LE-S-001**（确认视图 4 是否被写）
  6. **LE-P-001**（物理回收是否成功）
  7. **LE-W-006**（视图 1 与视图 3 同步校验）
  8. **LE-R-003**（_patch_scheduler_output 是否裁剪）
  9. **LE-I-001**（输入 Patch 是否生成）
  10. **LE-H-001~LE-H-004**（hook 端）
- **严禁**同时跑 `TRIATTN_DEBUG_VALIDATE_COMPACTION_CONTENT=1`（`triattention/vllm/runtime/kv_compaction.py:24`）做高负载跑测，会引入同步开销改变时序；仅在单请求 debug 时开启。

---

## 四、附：worker_reclaim_sync.py 核心调用栈速查图

```
[EngineCore.step_with_batch_queue]   ← integration_monkeypatch.py:594 _patched_engine_core_step_with_batch_queue
    └─ self.scheduler.schedule()      ← integration_monkeypatch.py:158 _patched_scheduler_schedule
        └─ setattr(scheduler_output, "triattention_signals", ...)

[NPUWorker.execute_model]            ← integration_monkeypatch.py:342 _patched_ascend_worker_execute_model
    └─ TriAttentionModelRunner.execute_model
        ├─ _consume_signals
        ├─ _supplement_worker_self_triggers
        ├─ _execute_compression_actions
        │    └─ executor.execute()  → base_runner.triattention_apply_compression
        │         └─ run_group_compaction_pipeline
        │              ├─ try_build_recency_tail_block_remap
        │              └─ execute_group_compaction → compact_request_kv_in_place[_per_head]
        │         └─ finalize_hook_placement_result      ← 视图 5 写入（req_state.block_ids 整体重写）
        ├─ ★ _apply_worker_block_reclaim_events
        │    └─ apply_worker_block_reclaim_events         ← worker_reclaim_sync.py:99
        │         ├─ 输入短路检查                            ← worker_reclaim_sync.py:113-155
        │         ├─ 解析 tables / req_id_to_index
        │         ├─ for event in events:                  ← 视图 1+2 写入点
        │         │    ├─ ★ truncate_tail: num_blocks_per_row[req_idx] = required_blocks
        │         │    │                  _clear_table_row_tail(...)
        │         │    └─ ★ remap_tail:   _rewrite_table_row(table, req_idx, block_ids_after)
        │         └─ for event in events:                  ← 视图 3 写入点
        │              ├─ remap_tail: 重写 req_state.block_ids
        │              └─ truncate_tail: del group_blocks[required_blocks:]
        ├─ _patch_scheduler_output_for_compressed_reqs   ← 视图 3 续写裁剪（防 _update_states 覆盖）
        ├─ execute_base_model_with_effective_overrides
        │    └─ base_runner.execute_model
        │         └─ vllm_ascend.worker.model_runner_v1.execute_model
        │              ├─ _update_states                 ← 视图 3 续写：block_ids.extend(new_ids)
        │              │                            + 视图 1 续写：block_table.append_row(new_ids)
        │              └─ set_ascend_forward_context → NPU attention
        └─ attach_execute_model_compression_events       ← 事件回传 scheduler_output / model_output

[EngineCore.update_from_output]      ← integration_monkeypatch.py:197 _patched_scheduler_update_from_output
    ├─ 原 Scheduler.update_from_output
    └─ ★ _apply_compression_events
         ├─ ★ update_request_effective_kv_offset         ← 视图 4 写入（_triattention_effective_kv_offset）
         ├─ _free_reclaimed_blocks                       ← 物理回收（block_pool.free_blocks）
         └─ (下一次) KVCacheManager.allocate_slots
              ← ★ integration_monkeypatch.py:493 _patched_kv_cache_allocate_slots
                  读取视图 4 → effective_num_computed
                  若有效 → delay_cache_blocks=True + 临时改 num_computed_tokens
                  若无效 → 走原始 allocate_slots 路径（按 32k 分配）
```

---

> **收尾声明**：本分析仅做静态推演与日志排查指引；**未对任何业务代码做修改、未做运行验证、未做 GPU/NPU 平台取舍**。后续若要根因修复，应基于上述模块三的「断点决策树」完成实证定位后，再做最小侵入代码改造；改造方案将另起任务输出。

---

## 模块四：实际 print 断点插入方案（含代码片段 + 意图讲解）

> 本模块是给「新手」的可执行清单——我会在每个文件、每个位置把**完整的 print 代码片段**贴出来，附**插入位置（行号）**、**为什么在这一行**、**你期望看到的输出**、**根据输出如何判断根因**。所有 print 都加了**总开关** `TRIATTN_BUG_RECL_DEBUG`（默认关闭），不污染正常输出。

### 4.0 总体设计原则

1. **统一前缀 `BUG-RECL`**：所有调试 print 都以 `BUG-RECL ` 开头，便于 `grep "BUG-RECL"` 一把抓出。
2. **总开关 `TRIATTN_BUG_RECL_DEBUG`**：默认 `0`（关闭），设 `1` 启用所有 BUG-RECL 断点；不依赖现有 `log_decisions` 等开关，避免被现有 logging 体系过滤。
3. **全部用 `print(..., flush=True)`**：因为 vLLM-Ascend 默认 logger 有时候被 `TRIATTENTION_QUIET=1` 或 NPU 异步流压制，**`print + flush=True` 一定能在 stdout/stderr 看到**，对调试最稳。
4. **最小侵入**：每处只插 1~3 行 print，不修改任何业务逻辑。
5. **就近插入**：在「关键变量计算后立即打印」，避免后置观测被其他逻辑覆盖。
6. **持久化辅助函数**：在 `triattention/vllm/runtime/worker_reclaim_sync.py` 顶部一次性加一个 `_bug_recl_enabled()` 辅助函数（1 行），其他文件用 `os.environ.get(...)` 内联判断即可——这样每个断点都是「独立可插拔」的，**不需要新建公共文件**。

### 4.1 步骤 0：先插入「总开关辅助函数」（1 个文件、1 行 import + 3 行函数）

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 12 行（`from .logging_control import runtime_logging_enabled` 这一行**之前**——我设计为插在 import 段开头）

**为什么是这一行**：
- 整个 `worker_reclaim_sync.py` 都没有用 `os`（除了已有的 `import os`），我们用 `os.environ.get` 内联就够；
- 但其他文件（如 `scheduler.py`、`runner.py`）要判断开关时需要统一约定，所以**先在 worker_reclaim_sync.py 里**写一个非常简单的辅助函数（哪怕只是一个 `return os.environ.get("TRIATTN_BUG_RECL_DEBUG") == "1"`），**后续其他文件直接复制 `def _bug_recl_enabled(): return ...` 这一行**即可。

**插入代码**（**注意：保留原 import 不动，只是把新代码插在它**之前**或模块顶部空闲区**）：

```python
# === BUG-RECL 总开关辅助函数（断点排查专用） ===
# 用法：在任何需要打印的代码前调用
#   if _bug_recl_enabled():
#       print("BUG-RECL ...", flush=True)
# 设 TRIATTN_BUG_RECL_DEBUG=1 启用；默认 0 关闭
import os as _os  # 已有 import os，重复无害（用 _os 避免污染命名）
def _bug_recl_enabled() -> bool:
    return _os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
# === BUG-RECL 总开关辅助函数结束 ===
```

> **等等——我注意到这个文件顶部已经 `import os` 了**（第 5 行）。所以你只需要插：
>
> ```python
> def _bug_recl_enabled() -> bool:
>     return os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
> ```
>
> 插在 `import numpy as np`（第 8 行）**之后**、`_DEBUG_DISABLE_LOGGED = False`（第 13 行）**之前**——这段是模块顶部空闲区，**不会触碰任何现有代码**。

**为什么这样设计**：
- 不需要新建公共文件（保持最小侵入）；
- 每个文件独立判断开关，断点之间互不依赖；
- `def _bug_recl_enabled()` 的 `_` 前缀表明是「私有辅助」，**PEP8 不报警**；
- 函数体只有一行，运行时开销可忽略（一次 `os.environ.get`）。

---

### 4.2 步骤 1：worker_reclaim_sync.py 内的 6 个断点（**LE-W-001 ~ LE-W-006**）

> 这是**最重要的断点群**——`worker_reclaim_sync.py` 是「逻辑驱逐」的实际执行点，所有 5 个视图（视图 1+2+3+5）的写入都在这里发生。

#### 断点 1（LE-W-001）：函数入口与所有短路条件

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 124 行（`if not isinstance(events, list) or not events: return` 这一行**之前**）

**为什么是这一行**：
- 函数体起点 = 第一次能观测到函数是否被调用；
- 第 113-122 行（env 短路）和第 124-125 行（空 events 短路）都还没 return，**插在这里能同时观测到 env 短路标志和 events 列表状态**。

**插入代码**：

```python
    # === BUG-RECL 断点 1（LE-W-001）：函数入口与短路检查 ===
    if _bug_recl_enabled():
        # 统计 applied 事件数；events 可能是 list 或 None
        n_events_total = len(events) if isinstance(events, list) else 0
        n_applied = sum(1 for e in events if isinstance(e, dict) and e.get("status") == "applied") if isinstance(events, list) else 0
        env_disable = os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0") == "1"
        print(f"BUG-RECL [LE-W-001] enter apply_worker_block_reclaim_events n_events={n_events_total} n_applied={n_applied} env_disable={env_disable}", flush=True)
    # === BUG-RECL 断点 1 结束 ===
```

**你期望看到的输出**（一次压缩事件后）：

```
BUG-RECL [LE-W-001] enter apply_worker_block_reclaim_events n_events=3 n_applied=1 env_disable=False
```

**判定**：
- `n_events=0` 或 `n_applied=0` → 走模块三决策树第 [1] 步：上游 2.14/2.11 排查；
- `env_disable=True` → 立即停止：env 配错了，把 `TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC` 去掉；
- `n_applied >= 1` 但下游断点（断点 3、4）一个都没触发 → 进了 `for event in events` 循环后又走了 `continue`，走断点 2 排查。

#### 断点 2（LE-W-002）：per-event 解析

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 193 行（`required_blocks = (retained_cache_len + block_size - 1) // block_size` 这一行**之后**）——这里 `required_blocks` 刚刚算好，是观测它的最佳位置。

**为什么是这一行**：
- 第 194 行（`reclaim_mode, groups_by_gid = _event_reclaim_groups(event)`）马上要被执行，**插在它前面**可以同时观测 `required_blocks` 和 `reclaim_mode` 即将计算前的状态；
- 也方便后续对照 `reclaim_mode` 的实际值。

**插入代码**：

```python
        # === BUG-RECL 断点 2（LE-W-002）：per-event 解析 ===
        if _bug_recl_enabled():
            print(f"BUG-RECL [LE-W-002] event req={req_id} req_index={req_index} cache_len_after={cache_len_after} retained_cache_len={retained_cache_len} required_blocks={required_blocks} block_size={block_size}", flush=True)
        # === BUG-RECL 断点 2 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-W-002] event req=cmpl-xxx req_index=7 cache_len_after=2048 retained_cache_len=2048 required_blocks=128 block_size=16
```

**判定**：
- `req_index=None` → 该 req 还没加入 input_batch（vLLM 异步延迟），本 event 在本 step 不处理；
- `cache_len_after <= 0` → 事件异常（hook 端 bug），跳过；
- `retained_cache_len < cache_len_after` → 来自 `details["retained_cache_len"]` 异常，**走 2.1 排查**（`_patch_scheduler_output_for_compressed_reqs` 用错的 retained_cache_len 裁剪）；
- `retained_cache_len=cache_len_after` → 正常，details 缺字段，**降级路径**。

#### 断点 3（LE-W-003）：truncate_tail 视图 1+2 写入

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 220 行（`if reclaim_mode == "remap_tail":` 这一行**之前**）——在 `reclaim_mode` 判断之前，能同时看到即将走哪条路径。

**为什么是这一行**：
- 第 220 行的 `if reclaim_mode == "remap_tail":` 是分支点；
- 第 220-232 行 truncate_tail 主体里有 `num_blocks_per_row[req_index] = required_blocks`（视图 1 写入）与 `_clear_table_row_tail(...)`（视图 2 写入），**插在 220 之前能完整观测即将进入哪条路径**。

**插入代码**：

```python
        # === BUG-RECL 断点 3（LE-W-003）：truncate_tail vs remap_tail 分支点 ===
        if _bug_recl_enabled():
            print(f"BUG-RECL [LE-W-003] before-branch req={req_id} gid={gid} reclaim_mode={reclaim_mode} current_num_blocks={current} required_blocks={required_blocks} will_shrink={current > required_blocks}", flush=True)
        # === BUG-RECL 断点 3 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-W-003] before-branch req=cmpl-xxx gid=0 reclaim_mode=truncate_tail current_num_blocks=2048 required_blocks=128 will_shrink=True
```

**判定**：
- `reclaim_mode=truncate_tail` 且 `will_shrink=False` → 视图 1 不会动，**走 2.6 排查**；
- `reclaim_mode=remap_tail` → 走断点 4；
- `reclaim_mode=truncate_tail` 且 `will_shrink=True` 但**断点 5 显示视图 3 没动** → 走 2.6 时序问题。

#### 断点 4（LE-W-004）：remap_tail 校验与 rewrite

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 205 行（`if block_ids_after is not None:` 这一行**之前**）——在 `_block_ids_after` 校验完成、`_rewrite_table_row` 即将被调之前。

**为什么是这一行**：
- 第 204 行 `block_ids_after = _block_ids_after(groups_by_gid.get(gid))` 刚跑完，**`block_ids_after` 可能是 None（校验失败）或 list（校验成功）**——这是观测校验结果的最佳位置；
- 第 206 行 `if _rewrite_table_row(...)` 紧跟其后，**观测 remap 写入成败**。

**插入代码**：

```python
        # === BUG-RECL 断点 4（LE-W-004）：remap_tail 校验与 rewrite 结果 ===
        if _bug_recl_enabled():
            # 注意：block_ids_after 在本行 204 之后才赋值
            # 所以这里要重新算一下，方便观测校验结果
            _ba = block_ids_after  # 已经是 None 或 list
            print(f"BUG-RECL [LE-W-004] remap_tail req={req_id} gid={gid} block_ids_after_valid={_ba is not None} n_block_ids_after={len(_ba) if _ba else 0} block_ids_after={(_ba or [])[:8]}", flush=True)
        # === BUG-RECL 断点 4 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-W-004] remap_tail req=cmpl-xxx gid=0 block_ids_after_valid=True n_block_ids_after=128 block_ids_after=[3, 17, 42, 89, ...]
```

**判定**：
- `block_ids_after_valid=False` → `_block_ids_after` 校验失败，**走 2.2 排查**（block_ids_after 含 None / 重复 / 非 int）；
- `block_ids_after_valid=True` 但 `n_block_ids_after=0` → 异常（zero-copy remap 不该出 0）；
- `block_ids_after_valid=True` 且 n>0 → remap_tail 校验通过，**视图 1+2 准备被重写**。

#### 断点 5（LE-W-005）：视图 3（req_state.block_ids）写入

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 266 行（`for group_blocks in block_ids_attr: if isinstance(group_blocks, list) and len(group_blocks) > required_blocks: del group_blocks[required_blocks:]` 这个 `for` 循环**之后**）——视图 3 truncate_tail 写入完成。

**为什么是这一行**：
- 第 261-266 行的 for 循环是 truncate_tail 模式下视图 3 的实际 mutate ；
- remap_tail 模式下视图 3 在 256-259 行被整体替换；
- **插在 266 之后能观测到 truncate_tail 完成后的视图 3 状态**；
- remap_tail 模式下第 266 行不在分支中，**只会跑 truncate_tail 分支**——所以需要分开打。

**插入代码**（**truncate_tail 部分**）：

```python
            # === BUG-RECL 断点 5a（LE-W-005a）：视图 3 truncate_tail 写入结果 ===
            if _bug_recl_enabled():
                try:
                    _len = [len(g) for g in block_ids_attr] if isinstance(block_ids_attr, (list, tuple)) else None
                    _container_type = type(block_ids_attr).__name__
                    print(f"BUG-RECL [LE-W-005a] view3-after-truncate req={req_id} container={_container_type} per_group_len={_len} required_blocks={required_blocks}", flush=True)
                except Exception as _e:
                    print(f"BUG-RECL [LE-W-005a] view3-read-error req={req_id} err={_e}", flush=True)
            # === BUG-RECL 断点 5a 结束 ===
```

**插入位置**（**remap_tail 部分**）：

**位置**：第 259 行（`setattr(req_state, "block_ids", rewritten_groups)` 这一行**之后**）

**插入代码**：

```python
                # === BUG-RECL 断点 5b（LE-W-005b）：视图 3 remap_tail 写入结果 ===
                if _bug_recl_enabled():
                    try:
                        _rb = getattr(req_state, "block_ids", None)
                        _len = [len(g) for g in _rb] if isinstance(_rb, (list, tuple)) else None
                        print(f"BUG-RECL [LE-W-005b] view3-after-remap req={req_id} container={type(_rb).__name__} per_group_len={_len}", flush=True)
                    except Exception as _e:
                        print(f"BUG-RECL [LE-W-005b] view3-read-error req={req_id} err={_e}", flush=True)
                # === BUG-RECL 断点 5b 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-W-005a] view3-after-truncate req=cmpl-xxx container=list per_group_len=[128] required_blocks=128
```

**判定**：
- `container=tuple` 且走 truncate_tail 模式 → **AttributeError 风险**（2.12 排查），断点会进 `_e` 分支；
- `per_group_len != [required_blocks]` → 视图 3 没真的被截断（`del` 没生效），**走 2.6 排查**；
- `per_group_len == [required_blocks]` → 视图 3 正常写入。

#### 断点 6（LE-W-006）：函数末尾的视图 1+2+3 同步校验

**文件**：`triattention/vllm/runtime/worker_reclaim_sync.py`

**位置**：第 267 行（文件末尾 return 之前）——把当前 event 循环内累计的状态打出来。

**为什么是这一行**：
- 第 267 行是 `for event in events` 循环的末尾，**所有视图写入都已完成**；
- 这里能一次性把视图 1（`num_blocks_per_row`）、视图 2（`block_table.np`）、视图 3（`req_state.block_ids`）的值对照。

**插入代码**：

```python
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
```

**你期望看到的输出**：

```
BUG-RECL [LE-W-006] final-sync [{'req': 'cmpl-xxx', 'view1_per_gid': [128], 'view3_container': 'list', 'view3_lens': [128]}]
```

**判定**：
- `view1_per_gid != view3_lens` → 视图 1 与视图 3 失同步，**走 2.3 排查**；
- `view3_container=tuple` 但 `reclaim_mode=truncate_tail` → 走 2.12 排查（AttributeError 风险）；
- `view3_lens=None` 或 `view3_container=no_req_state` → 走 2.6 排查（base_runner.requests 里找不到该 req）。

---

### 4.3 步骤 2：scheduler.py 内的 4 个断点（**LE-S-001 ~ LE-S-004**）

> 这 4 个断点观测**视图 4 写入**与**物理回收**的联动——是「逻辑驱逐失效」的核心耦合点。

#### 断点 7（LE-S-001）：视图 4 写入与 `reclaim_applied_any` 联动（**两处**）

**文件**：`triattention/vllm/runtime/scheduler.py`

**位置 A**：第 707 行（`if _free_reclaimed_blocks(manager, removed_blocks): reclaim_applied_any = True` 这一行**之后**）——synthesized 路径。

**位置 B**：第 875 行（`logger.debug("TriAttention block reclaim: skipping synthesized reclaim for missing gids %s ..."` 这一行**之后**）——主路径。

**为什么是这两处**：
- A 处是 synthesized 路径，reclaim_applied_any 仅在 `_free_reclaimed_blocks` 返回 True 时变 True；
- B 处是主路径末尾，reclaim_applied_any 反映了主路径的真实状态；
- **两处的 `reclaim_applied_any` 决定了 `update_request_effective_kv_offset` 是否被调用**（视图 4 写入）。

**插入代码**（**A 处**）：

```python
                        # === BUG-RECL 断点 7a（LE-S-001a）：synthesized 路径 reclaim_applied_any ===
                        if _bug_recl_enabled():
                            print(f"BUG-RECL [LE-S-001a] synth-path req={req_id} gid={gid} reclaim_applied_any={reclaim_applied_any}", flush=True)
                        # === BUG-RECL 断点 7a 结束 ===
```

**插入代码**（**B 处**）：

```python
                # === BUG-RECL 断点 7b（LE-S-001b）：主路径 reclaim_applied_any 终态 ===
                if _bug_recl_enabled():
                    _req = self.requests.get(req_id)
                    _offset = getattr(_req, "_triattention_effective_kv_offset", None) if _req is not None else None
                    print(f"BUG-RECL [LE-S-001b] main-path req={req_id} reclaim_applied_any={reclaim_applied_any} view4_offset_after={_offset}", flush=True)
                # === BUG-RECL 断点 7b 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-S-001a] synth-path req=cmpl-xxx gid=0 reclaim_applied_any=True
BUG-RECL [LE-S-001b] main-path req=cmpl-xxx reclaim_applied_any=True view4_offset_after=30720
```

**判定**：
- `reclaim_applied_any=False` 持续 → `_free_reclaimed_blocks` 没成功推进 free_block_queue，**走 2.4 / 2.15 排查**；
- `reclaim_applied_any=True` 但 `view4_offset_after=None` → `_apply_compression_events` 的视图 4 写入路径没跑到，**走 2.4 排查**；
- `view4_offset_after=0` 但 `cache_len_after=2048` → 写入的 offset 不正确，**走 2.4 排查**。

**注意**：scheduler.py 顶部要加 `import os` 或内联 `os.environ.get`，**由于这个文件**没有 `import os`，所以辅助函数需要单独插入。在 `scheduler.py` 顶部第 8 行（`from typing import Any` 之后）加：

```python
def _bug_recl_enabled() -> bool:
    import os as _os
    return _os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
```

> **为什么每个文件都自己定义一个本地辅助函数**？——保持每个文件独立可插拔，不需要新增公共文件，也避免循环 import。代价是 4 行重复代码，可接受。

#### 断点 8（LE-S-002）：`_apply_compression_events` 入口

**文件**：`triattention/vllm/runtime/scheduler.py`

**位置**：第 577 行（`if self.triattention_config.log_decisions:` 这一行**之后**）——已有的 log 块之后插入。

**为什么是这一行**：
- 第 570 行函数定义，第 572-587 行是已有的初始化与 log，**插在已有 log 之后**是观测事件刚进入函数的状态；
- 紧跟着第 594 行 `for event in compression_events:` 是主循环。

**插入代码**：

```python
        # === BUG-RECL 断点 8（LE-S-002）：_apply_compression_events 入口 ===
        if _bug_recl_enabled():
            n_events = len(compression_events) if isinstance(compression_events, list) else 0
            n_applied = sum(1 for e in compression_events if isinstance(e, dict) and e.get("status") == "applied") if isinstance(compression_events, list) else 0
            print(f"BUG-RECL [LE-S-002] enter _apply_compression_events n_events={n_events} n_applied={n_applied} reclaim_enabled={self.triattention_config.enable_experimental_block_reclaim}", flush=True)
        # === BUG-RECL 断点 8 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-S-002] enter _apply_compression_events n_events=3 n_applied=1 reclaim_enabled=True
```

**判定**：
- `n_applied=0` 但 worker_reclaim_sync 已经写过 → 事件从 model_runner_output / scheduler_output 路径有 bug，**走 2.6 排查**；
- `reclaim_enabled=False` → 配置被改，**立即停止排查**，先恢复默认 `enable_experimental_block_reclaim=True`。

#### 断点 9（LE-S-003）：主路径 `_free_reclaimed_blocks` 实际调用

**文件**：`triattention/vllm/runtime/scheduler.py`

**位置**：第 841 行（`if _free_reclaimed_blocks(manager, removed_old_blocks): reclaim_applied_any = True` 这一行**之前**）——truncate_tail 物理回收点。

**为什么是这一行**：
- 这是主路径下**真正调 block_pool.free_blocks** 的入口；
- 第 866 行（missing_gids 循环的 `_free_reclaimed_blocks`）类似，但本次只观测主路径。

**插入代码**：

```python
                # === BUG-RECL 断点 9（LE-S-003）：truncate_tail 物理回收入口 ===
                if _bug_recl_enabled():
                    print(f"BUG-RECL [LE-S-003] pre-free-blocks req={req_id} gid={gid} mode=truncate_tail removed_n={len(removed_old_blocks) if 'removed_old_blocks' in dir() and removed_old_blocks else 0}", flush=True)
                # === BUG-RECL 断点 9 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-S-003] pre-free-blocks req=cmpl-xxx gid=0 mode=truncate_tail removed_n=1920
```

**判定**：
- 完全没看到这条 print → 主路径根本不走 `removed_old_blocks` 截断，**走 2.4 / 2.15 排查**；
- `removed_n=0` 但 `expected_shrink_gids` 非空 → 异常，**走 2.3 排查**。

#### 断点 10（LE-S-004）：synthesized 路径 prefill 跳过

**文件**：`triattention/vllm/runtime/scheduler.py`

**位置**：第 685 行（`if _evt_scheduled > 1:` 这一行**之后**）——prefill 跳过判断点。

**为什么是这一行**：
- 第 685-692 行是 prefill 跳过逻辑，**插在 685 之后**能确认是否真的走了 prefill 跳过；
- 第 693 行起是 `_free_reclaimed_blocks` 真实调用，**插在 685 之后能区分两种情况**。

**插入代码**：

```python
                # === BUG-RECL 断点 10（LE-S-004）：synthesized prefill 跳过 ===
                if _bug_recl_enabled():
                    print(f"BUG-RECL [LE-S-004] synth-skip req={req_id} _evt_scheduled={_evt_scheduled} is_prefill={_evt_scheduled > 1} expected_shrink_gids={list(expected_shrink_gids) if 'expected_shrink_gids' in dir() else 'N/A'}", flush=True)
                # === BUG-RECL 断点 10 结束 ===
```

**你期望看到的输出**：

```
BUG-RECL [LE-S-004] synth-skip req=cmpl-xxx _evt_scheduled=1024 is_prefill=True expected_shrink_gids=[0]
```

**判定**：
- `is_prefill=True` 持续 → **2.5 直接命中**（prefill 阶段被静默跳过）；
- `is_prefill=False` 但视图 4 还是没被写入 → 走 2.4 排查。

---

### 4.4 步骤 3：integration_monkeypatch.py 内的 1 个断点（**LE-A-001**）

> `_patched_kv_cache_allocate_slots` 是「视图 4 → 物理水位」的**最终消费点**——必须观测它是否真的进了 effective 路径。

#### 断点 11（LE-A-001）：`_patched_kv_cache_allocate_slots` 走 effective 路径

**文件**：`triattention/vllm/runtime/integration_monkeypatch.py`

**位置**：第 512 行（`prepare_request_effective_num_computed(request)` 这一行**之后**）——`effective_num_computed` 刚刚算好。

**为什么是这一行**：
- 第 511 行准备，第 512 行 `effective_num_computed = resolve_request_effective_num_computed(request)` 立即拿到值；
- 第 513-516 行的 `if effective_num_computed is None: return _ORIG_KVCACHE_ALLOCATE_SLOTS(...)` 是「退化为原始路径」的出口；
- **插在 512 之后能观测到 effective 是否真有效**。

**插入代码**：

```python
    prepare_request_effective_num_computed(request)
    effective_num_computed = resolve_request_effective_num_computed(request)
    # === BUG-RECL 断点 11（LE-A-001）：allocate_slots effective 路径决策 ===
    if os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1":
        _logical = getattr(request, "num_computed_tokens", None)
        _offset = getattr(request, "_triattention_effective_kv_offset", None)
        print(f"BUG-RECL [LE-A-001] allocate_slots req={getattr(request, 'request_id', '?')} effective_num_computed={effective_num_computed} logical_num_computed={_logical} offset={_offset} will_use_effective={effective_num_computed is not None and isinstance(_logical, int) and effective_num_computed < _logical}", flush=True)
    # === BUG-RECL 断点 11 结束 ===
```

> 注意：这个文件**已经有 `import os` 了**（第 9 行），所以可以直接用 `os.environ.get`，不需要再 import。

**你期望看到的输出**（一次压缩事件后）：

```
BUG-RECL [LE-A-001] allocate_slots req=cmpl-xxx effective_num_computed=2048 logical_num_computed=32768 offset=30720 will_use_effective=True
```

**判定**：
- `effective_num_computed=None` 持续 → 视图 4 没被写入，**走断点 7 排查**（`LE-S-001a/b`）；
- `will_use_effective=False` 但 `offset != None` → 写入的值异常，**走 2.4 排查**；
- `will_use_effective=True` 但 `kv_cache_usage` 还是不下 → 走 2.13（ref_cnt 不归零）。

---

### 4.5 步骤 4：runner.py 内的 1 个断点（**LE-R-001**）

#### 断点 12（LE-R-001）：`_execute_compression_actions` 后的 pending 事件

**文件**：`triattention/vllm/runtime/runner.py`

**位置**：第 1164 行（`self._pending_compression_events = execute_runner_compression_actions(...)` 这一行**之后**）。

**为什么是这一行**：
- `execute_runner_compression_actions` 跑完，`self._pending_compression_events` 列表已经填好；
- 这是 worker_reclaim_sync 即将消费的 events，**观测这里能确认 hook 端是否真产生 applied 事件 + block_reclaim 字段**。

**插入代码**：

```python
        self._pending_compression_events = execute_runner_compression_actions(...)
        # === BUG-RECL 断点 12（LE-R-001）：pending 事件统计 ===
        if os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1":
            n_total = len(self._pending_compression_events)
            n_applied = sum(1 for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied")
            n_applied_with_reclaim = sum(1 for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied" and isinstance(e.get("block_reclaim"), dict))
            n_applied_no_reclaim = n_applied - n_applied_with_reclaim
            reqs = [e.get("req_id") for e in self._pending_compression_events if isinstance(e, dict) and e.get("status") == "applied"][:5]
            print(f"BUG-RECL [LE-R-001] pending n_total={n_total} n_applied={n_applied} n_applied_with_reclaim={n_applied_with_reclaim} n_applied_no_reclaim={n_applied_no_reclaim} sample_reqs={reqs}", flush=True)
        # === BUG-RECL 断点 12 结束 ===
```

> runner.py 顶部有 `import os`（第 5 行），可以直接用。

**你期望看到的输出**：

```
BUG-RECL [LE-R-001] pending n_total=3 n_applied=1 n_applied_with_reclaim=1 n_applied_no_reclaim=0 sample_reqs=['cmpl-xxx']
```

**判定**：
- `n_applied=0` 持续 → 走 2.11/2.14 排查（hook 端没产生 applied 事件）；
- `n_applied_no_reclaim > 0` → 走 2.11 排查（hook 端 applied 事件不带 block_reclaim 字段）。

---

### 4.6 步骤 5：hook_impl.py 内的 1 个断点（**LE-H-001**）

#### 断点 13（LE-H-001）：hook 入口与 selector 选择

**文件**：`triattention/vllm/runtime/hook_impl.py`

**位置**：第 111 行（`log_fn = ...` 这一行**之后**）——已有的 log 后插入。

**为什么是这一行**：
- 第 97 行函数定义，第 99-111 行是已有的 log 块，**插在已有 log 后**能完整观测 hook 入口状态；
- 第 115 行 `req_ctx = resolve_hook_request_context(...)` 紧跟其后。

**插入代码**：

```python
        if log_execution_path:
            _runtime_logger.info(...)
        # === BUG-RECL 断点 13（LE-H-001）：hook 入口观测 ===
        if os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1":
            _under_budget = effective_tokens <= budget_total
            print(f"BUG-RECL [LE-H-001] hook-enter req={req_id} step={getattr(signal, 'step', 0)} selector_status={selector_status} select_keep_indices_present={select_keep_indices is not None} effective_tokens={effective_tokens} budget_total={budget_total} under_budget={_under_budget}", flush=True)
        # === BUG-RECL 断点 13 结束 ===
```

> hook_impl.py 顶部**没有** `import os`（只有 `import logging`）。所以你需要在 hook_impl.py 顶部第 6 行（`from typing import Any, Callable` **之后**、空行**之前**）加：
>
> ```python
> import os
> ```
>
> 然后上面的 `os.environ.get(...)` 就能直接用。
>
> **为什么这么改**：hook_impl.py 是整个 hook 链的入口，**观测这里 = 观测「为什么 hook 被调用 / 为什么 hook 跳过 / 为什么 selector 失败」**。这是「filter 是否真生效」最直接的位置。

**你期望看到的输出**：

```
BUG-RECL [LE-H-001] hook-enter req=cmpl-xxx step=145 selector_status=triton select_keep_indices_present=True effective_tokens=2304 budget_total=2304 under_budget=False
```

**判定**：
- `selector_status=none` 持续 → 走 2.14 排查（Triton 编译失败 + PyTorch fallback 也不可用）；
- `under_budget=True` 持续 → 跳到 `applied=False reason="under_budget"`，**视图 1+2+3 都不会写**（`hook_impl.py:170-175` 行为），本 step 不产生任何压缩事件；
- `selector_status=triton` 但 `select_keep_indices_present=False` → 异常（hook 端内部不一致）。

---

### 4.7 「按优先级插入」的实操建议（你只需要插这几个就够定位 80% 的问题）

> 我设计的所有 13 个断点中，**5 个就够覆盖「逻辑驱逐失效」的主线**——如果你时间紧张，先插这 5 个：

| 优先级 | 断点 | 文件 | 行号 | 一句话功能 |
| - | - | - | - | - |
| ★★★★★ | **LE-W-001** | `worker_reclaim_sync.py` | 124 | 函数是否被调用 + 是否被 env 短路 |
| ★★★★★ | **LE-W-003** | `worker_reclaim_sync.py` | 220 | 视图 1 是否被写（truncate_tail 主路径） |
| ★★★★★ | **LE-S-001b** | `scheduler.py` | 875 | 视图 4 写入条件 `reclaim_applied_any` 终态 |
| ★★★★★ | **LE-A-001** | `integration_monkeypatch.py` | 512 | allocate_slots 是否走 effective 路径 |
| ★★★★★ | **LE-R-001** | `runner.py` | 1164 | pending 事件是否带 block_reclaim |

**插入方法**：
1. 打开 `worker_reclaim_sync.py`，先在第 5 行 `import os` **之后**插 1 行：
   ```python
   def _bug_recl_enabled() -> bool:
       return os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"
   ```
2. 然后按 4.2 节插入断点 1、3 的代码；
3. 打开 `scheduler.py`，在第 5 行 `import os`（或 `from typing import Any`）**之后**插同样的 4 行辅助函数；
4. 然后按 4.3 节插入断点 7b、8 的代码；
5. 打开 `integration_monkeypatch.py`（已经有 `import os`），按 4.4 节插入断点 11；
6. 打开 `runner.py`（已经有 `import os`），按 4.5 节插入断点 12；
7. **运行测试前**：`export TRIATTN_BUG_RECL_DEBUG=1`；
8. **运行后**：`grep "BUG-RECL" your_log_file.log`，对照 4.2-4.6 节末尾的「判定」段定位。

---

### 4.8 「总开关」的使用

```bash
# 关闭（默认）—— 所有 BUG-RECL print 都不输出
unset TRIATTN_BUG_RECL_DEBUG

# 开启 —— 所有 BUG-RECL print 都会输出到 stdout
export TRIATTN_BUG_RECL_DEBUG=1

# 关闭 vLLM 自身的 logger 噪音（可选，让 BUG-RECL 更突出）
export TRIATTENTION_QUIET=1
```

> **设计这个开关的理由**：
> 1. 现有 `TRIATTN_RUNTIME_LOG_DECISIONS` 等开关被 `TRIATTN_RUNTIME_LOGGING=0` 一键压制，**不适合作为本任务的调试开关**；
> 2. 用 `print + flush=True` 而不是 logger，能在 NPU 异步流、`TRIATTENTION_QUIET=1`、日志被截断等极端情况下**稳定输出**；
> 3. 用 `BUG-RECL` 前缀便于 `grep` 一键抓取，**不会和 vLLM 自身的 logger 输出混在一起**；
> 4. 总开关用 `os.environ.get("TRIATTN_BUG_RECL_DEBUG", "0") == "1"` 是最朴素的判断，**没有依赖任何模块**（即使 `triattention.vllm.runtime` 还没 import 也能用）。

---

### 4.9 「如果跑出来的日志什么都不对」的兜底

| 现象 | 兜底建议 |
| - | - |
| 完全看不到任何 BUG-RECL 输出 | 1. 检查 `TRIATTN_BUG_RECL_DEBUG=1` 是否设上；2. 检查 print 是否被 stdout 重定向到别处（`nohup` 等）；3. 在 `worker_reclaim_sync.py` 顶部插一个 `print("BUG-RECL worker_reclaim_sync.py 加载成功", flush=True)` 看文件是否被 import |
| BUG-RECL 输出有但 BUG-RECL [LE-W-001] 没有 | 说明 `apply_worker_block_reclaim_events` 根本没被调到——走模块三决策树第 [1] 步的"V2 runner 短路"分支（`worker_reclaim_sync.py:130-136`） |
| BUG-RECL [LE-W-001] 有但 [LE-W-003] 没有 | 说明函数被调了但 `for event in events:` 循环里 `continue` 掉了——走断点 2 排查（`req_id_to_index.get(req_id)` 返回 None，或 `cache_len_after <= 0`） |
| BUG-RECL [LE-W-003] 有但 [LE-S-001b] 仍说 `reclaim_applied_any=False` | 说明视图 1+2 被写了但 `_free_reclaimed_blocks` 没推进 free_block_queue——走 2.13 排查（`enable_prefix_caching=True`） |
| 所有断点都"正常"但 `kv_cache_usage` 仍不下 | 说明前序都通了，问题在 vLLM 上游的 `block_pool.get_usage()` 计算本身，或用户的 `--gpu-memory-utilization` 配错——这不是 TriAttention bug |

---

> **收尾声明**：本分析仅做静态推演与日志排查指引；**未对任何业务代码做修改、未做运行验证、未做 GPU/NPU 平台取舍**。模块四给出了 13 个 print 断点的**完整可复制代码片段**与**插入位置**，你只需要按 4.7 节「按优先级插入」即可在不开新文件、不改任何业务逻辑的前提下完成排查。后续若要根因修复，应基于上述模块三的「断点决策树」与模块四的「断点输出」完成实证定位后，再做最小侵入代码改造；改造方案将另起任务输出。

