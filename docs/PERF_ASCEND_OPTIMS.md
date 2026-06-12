# Ascend NPU Compression Performance Optimizations

本文档汇总最近两个针对 vLLM-Ascend 0.18.0 平台 TriAttention 压缩热路径的性能优化 commit。

- 分支：`fix/effective-len-tracker-events-and-current-cache-len`
- 优化基线 commit：`d465c13`（保留 `919b07c` 的 current_cache_len 修复 和 `fa43165` 的 H/F2/G2 详细 instrumentation）
- 触发原因：实测 `COMPACT-COST-DBG` 输出显示，**vllm-ascend 0.18.0 上每次 `torch.topk` / `masked_fill` / `scatter_` 的 aclnn launch 开销约是 CUDA 上同操作的 5 倍**；profile 显示 20k→2k 压缩在 NPU 上耗时 ~230ms，其中 ~150ms 是 kernel launch overhead

---

## 优化 1 — Commit `87ba4b5`

**commit message**： `perf(ascend): bump score_chunk ceiling to 16384 on NPU`

### 修改文件
`triattention/vllm/runtime/selector_hf.py`

### 关键位置

| 行号 | 内容 |
|------|------|
| 702 | 新增常量 `_NPU_CHUNK_FLOOR = 16384`（闭包内）|
| 703 | 新增变量 `_is_npu_device: bool \| None = None`（闭包内缓存）|
| 705-717 | 新增 helper `_detect_npu_device()` — 闭包内进程级 LRU 缓存，只判定一次 |
| 719-736 | 修改 `_score_chunk_tokens` — NPU + total_tokens ≥ 16384 时 chunk 上限 4k → 16k |

### 问题
默认 `score_chunk_max_tokens=4096`（在 `triattention/vllm/runtime/config.py:44` 定义的 `TriAttentionRuntimeConfig.score_chunk_max_tokens`）。该值通过 `TRIATTN_RUNTIME_SCORE_CHUNK_MAX_TOKENS` 环境变量配置。

20k token 的 KV 在 `_select_keep_indices_paged_streaming` 内 split 成 5 个 chunk（19789 / 4096 ≈ 4.83 → 5 个）。每个 chunk 跑：
- 1× `torch.topk`
- 1× `masked_fill(inf)`（guard mask）
- 1× 跨 chunk cat + 二次 `torch.topk`

总共 5 chunk × 3 launch × 36 layer = **540 次 aclnn launch / 一次压缩**。每次 launch ~0.6ms on NPU → **~324ms 纯 launch overhead**。

### 解决方案
在 `_score_chunk_tokens` 末尾加 NPU 检测 + chunk 上限提升。NPU + total_tokens ≥ 16384 时返回 16384。

### 预期收益
- 20k cache：5 chunks → 2 chunks，每层省 7 次 launch
- 36 层 × 7 launch/layer = 252 launch 节省
- 252 × ~0.6ms ≈ **~150ms / 一次压缩**

### 风险
低。chunk 增大意味着：
- `_compute_layer_scores_raw` 处理更大张量（Triton kernel 实际工作时间略增）
- `torch.topk` 在 16k 元素上的开销 vs 4k 元素（topk 是 O(N log K)，增长可忽略）

但消除了 ~7 次 NPU launch overhead，净收益正。

### 兼容性
- CUDA 路径：`effective_device.type == "cuda"` → `_is_npu_device=False` → `_score_chunk_tokens` 走原有逻辑
- 小 KV cache（< 16384）：`total_tokens >= _NPU_CHUNK_FLOOR` 为 False → 走原有逻辑
- 大 KV cache + NPU：触发新路径

---

## 优化 2 — Commit `e169993`

**commit message**：`perf(ascend): cache bool guard_mask + defer first compression`

这个 commit 含**两个独立优化**。

### 2a — 缓存 `_build_token_guard_mask` 输出

#### 修改文件
`triattention/vllm/runtime/selector_hf.py`

#### 关键位置

| 行号 | 内容 |
|------|------|
| 625-630 | 新增注释：缓存的目的 + NPU 上的 acInn launch 开销说明 |
| （紧接 630） | 新增 `_guard_mask_cache: dict[tuple, torch.Tensor] = {}`（闭包内）|
| （紧接 630） | 新增 `_GUARD_MASK_CACHE_MAX = 8`（闭包内 LRU 上限）|
| 633-688 | 修改 `_build_token_guard_mask` — 加 cache_key 构造 + 缓存读 + 缓存写 |

#### cache_key 字段
```python
cache_key = (
    int(start_token),       # chunk 起始 token 位置
    int(num_tokens),         # chunk 大小
    int(total_tokens),       # 整个 cache 长度
    int(prefill_len),        # prefill 长度（影响 prefill 保护区域）
    int(config.window_size), # window size（影响 tail 保护区域）
    bool(protect_prefill),   # 是否启用 prefill 保护
    device_index,            # NPU 设备号（多卡场景）
)
```

#### 问题
`_build_token_guard_mask` 每次调用都执行：
- `torch.arange(start_token, start_token + num_tokens, device=device, dtype=torch.long)` → 1 个 aclnn launch
- `torch.zeros_like(token_positions, dtype=torch.bool)` → 1 个 aclnn launch
- `guard_mask |= token_positions >= window_start` → 1 个 aclnn launch
- 可能再 `guard_mask |= token_positions < prefill_len` → 1 个 aclnn launch

每次调用 3-4 次 aclnn launch。selector 热路径上：
- 36 层 × 5 chunk = **180 次调用 / 一次压缩**
- 每调用 3 次 launch = **540 次 aclnn launch**

但参数 `(num_tokens, total_tokens, prefill_len, window_size, ...)` 在同一压缩过程内不变，只有 `start_token` 在 chunk 间变化（0, 4096, 8192, ...）—— 所以**实际唯一 mask 数量 ≈ chunk 数**，36 层之间能 100% 命中。

#### 解决方案
加进程级 LRU 缓存（最大 8 条目）。同 key 直接返回，避免重建。

#### 预期收益
- 36 层 × ~5 个唯一 mask × ~3 launch = 540 launch
- 缓存命中后只有 ~5 个 mask 需要真正生成
- 节省 ~535 次 aclnn launch × ~0.05ms (zeros_like + arange 较小) ≈ **~30ms / 一次压缩**

#### 风险
极低。`guard_mask` 是纯函数（输入参数决定输出），缓存正确性由 key 唯一性保证。`masked_fill` 后续操作只读不修改。

---

### 2b — 推迟首次压缩到 `2 × kv_budget`

#### 修改文件
`triattention/vllm/runtime/scheduler.py`

#### 关键位置

| 行号 | 内容 |
|------|------|
| 166 | 原 `_compute_length_threshold` 方法 |
| 170-180 | 新增注释：推迟首次压缩的理由（80ms 无收益）|
| 181-182 | 新增 env gate：`TRIATTN_RUNTIME_DEFER_FIRST_COMPRESS=1`（默认开启）+ `threshold = max(threshold, 2 * kv_budget)` |

#### 问题
profile 显示 TriAttention 压缩分两次发生：
1. **prefill chunk 完成后**（cache 从 0 → 19789 一次性灌入）：立刻触发首次压缩
2. **后续 decode 步骤**：cache 增量到 budget+divide_length=2176 时再次触发

第一次压缩发生在 prefill chunk 刚结束、decode step 还没开始的瞬间——**decoder 此时一个 token 都没消费**，压缩后 prefix 是 `[重要 token]`，但 attention 立刻就被调用来 decode 第一个新 token，**这次压缩的成果对 decoder 没有任何收益**。

但这次压缩**实打实**付了 ~80ms（score 阶段 0ms 是测量 bug，实际 ~10-20ms + scatter ~60ms）。

#### 解决方案
`_compute_length_threshold` 返回 `max(threshold, 2 × kv_budget)` —— 把首次触发阈值从 `kv_budget + divide_length = 2176` 提高到 `2 × kv_budget = 4096`。

#### 预期收益
- 20k prefill 完成后 cache 长 19789，远超 4096——**首次压缩完全跳过**
- 等 cache 真的长到 4096+ 才会触发（实际场景可能永远触发不到 4096，因为 decode step 增量 token 数 << budget）
- **节省 ~80ms / 一次压缩**

#### 风险
**中**。这个改动改了 trigger 语义，理论上压缩频率降低可能让 cache 更长时间保持 20k 长度。但这正好对应"score chunk 上限提升到 16k"那个优化——后者减少了后续压缩成本，前者减少了不必要的早期压缩，组合是稳态最优。

#### 兼容性 / 回滚
- `TRIATTN_RUNTIME_DEFER_FIRST_COMPRESS=0` → 恢复 `d465c13` 行为
- 不依赖任何 vllm 内部 API

---

## 累计预期收益

| 优化 | 单次压缩节省 |
|------|--------------|
| 1. score chunk 上限 4k → 16k | ~150ms |
| 2a. 缓存 guard_mask | ~30ms |
| 2b. 推迟首次压缩 | ~80ms |
| **合计** | **~260ms / 一次压缩** |

profile 显示基线（`d465c13`）单次压缩耗时 ~230ms；优化后期望压到 ~70ms / 一次压缩。

---

## 验证方法

跑一次相同 workload（input 20k, kv_budget 2048, output 1024），观察：

1. **压缩次数减少**：从 ~5 次/请求 降到 ~1-2 次/请求（因为首次压缩推迟）
2. **首次压缩消失**：观察 `[COMPACT-COST-DBG]` 是否还在 prefill 后立即出现，或只在 cache 增长到 4k 后才出现
3. **整体 TPOT 改善**：40+ms → 应有改善（具体幅度需 profiling）

---

## 回滚指引

如果优化引入新问题，可以单独回滚：

```bash
# 回滚优化 1
git revert 87ba4b5

# 回滚优化 2（同时回滚 2a + 2b）
git revert e169993

# 只回滚 2b（推迟首次压缩），保留 guard_mask 缓存
git revert e169993 --no-commit
# 然后手动 revert scheduler.py 改动
```

如果只想关闭"推迟首次压缩"这一个特性（保留 chunk 上限提升 + guard_mask 缓存）：

```bash
export TRIATTN_RUNTIME_DEFER_FIRST_COMPRESS=0
```