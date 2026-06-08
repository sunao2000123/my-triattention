# RUN.md — TriAttention-Ascend v0.18.0 部署与验证教程

> 配 `ideal_WORKFLOW.md`（讲代码怎么走）和 `TRIATTENTION_ASCEND_ANALYSIS.md`（讲问题为什么、怎么修）一起看。本文档**只讲操作**——零基础、step-by-step、复制粘贴就能跑。

---

## 目录

1. [0. 部署前置清单](#0-部署前置清单)
2. [1. 安装 vllm 0.18.0](#1-安装-vllm-0180)
3. [2. 安装 vllm-ascend 0.18.0](#2-安装-vllm-ascend-0180)
4. [3. 安装 TriAttention（含 ascend 适配）](#3-安装-triattention含-ascend-适配)
5. [4. 准备 stats 文件](#4-准备-stats-文件)
6. [5. 启动服务](#5-启动服务)
7. [6. 验证关键日志](#6-验证关键日志)
8. [7. 验证算法真的运行](#7-验证算法真的运行)
9. [8. 关闭 TriAttention（baseline 对比）](#8-关闭-triattentionbaseline-对比)
10. [9. 故障排查速查表](#9-故障排查速查表)

---

## 0. 部署前置清单

| 依赖 | 版本 | 验证命令 | 期望 |
| ---- | ---- | -------- | ---- |
| OS | 任意 Linux（推荐 openEuler 22.03 / Ubuntu 22.04） | `cat /etc/os-release` | — |
| Python | ≥ 3.10 | `python --version` | `Python 3.10.x` 或更高 |
| NPU 驱动 | 与 CANN 套件匹配 | `npu-smi info` | 看到 NPU 设备 |
| CANN | 与 vllm-ascend 0.18.0 配套（推荐 8.0+） | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` | — |
| torch | 与 vllm-ascend 0.18.0 配套 | `python -c "import torch; print(torch.__version__)"` | — |
| torch_npu | 与 vllm-ascend 0.18.0 配套 | `python -c "import torch_npu; print(torch_npu.__version__)"` | — |
| vllm | 0.18.0 | `python -c "import vllm; print(vllm.__version__)"` | `0.18.0` |
| vllm-ascend | 0.18.0 | `python -c "import vllm_ascend; print(vllm_ascend.__file__)"` | 非空 |
| triton-ascend | ≥ 2.0（CANN 兼容层） | `python -c "import triton; print(triton.__version__)"` | — |
| 磁盘 | ≥ 50 GB | `df -h` | 装 vllm-ascend + 模型权重 + stats 缓存 |

---

## 1. 安装 vllm 0.18.0

```bash
# 1.1 准备 vllm 0.18.0 源码
#    （如果是从 vllm-ascend-releases-v0.18.0 来的，配套的 vllm 在
#     vllm-ascend 的 requirements.txt 里已经 pin 了 0.18.0）

cd /path/to/vllm-releases-v0.18.0
pip install -e . --no-build-isolation

# 1.2 验证
python -c "import vllm; print(vllm.__version__)"
# 期望：0.18.0
```

---

## 2. 安装 vllm-ascend 0.18.0

```bash
cd /path/to/vllm-ascend-releases-v0.18.0
pip install -e . --no-build-isolation

# 验证
python -c "
import vllm_ascend
print('vllm_ascend:', vllm_ascend.__file__)
from vllm.platforms import current_platform
print('platform:', type(current_platform).__name__)
"
# 期望：
#   vllm_ascend: /.../vllm_ascend/__init__.py
#   platform: Ascend... （名字含 Ascend 或 NPU）
```

**关键**：`platform` 必须含 `Ascend` 或 `NPU`。如果不是，检查：

- `vllm_ascend` 是否真的装到了当前 Python 环境（`which python`、`pip list | grep vllm`）
- `ASCEND_HOME_PATH` env 是否设了
- `vllm_ascend.platform.AscendPlatform` 是否被 `vllm.platforms.register_platform(...)` 注册（查看 `vllm_ascend/__init__.py` 的 platform 注册顺序）

---

## 3. 安装 TriAttention（含 ascend 适配）

```bash
cd /Users/sunao2000/my_tri
pip install -e . --no-deps --force-reinstall

# 验证 entry point 注册（最关键的一步！）
python -c "
import importlib.metadata as m
eps = m.entry_points().get('vllm.general_plugins', [])
print('Registered vllm.general_plugins entry points:')
for e in eps:
    print(f'  {e.name:25s} -> {e.value}')
"
```

**期望输出（必须两行都有）**：

```
Registered vllm.general_plugins entry points:
  triattention               -> triattention.vllm.plugin:register_triattention_backend
  triattention_ascend        -> triattention.vllm_ascend.plugin:register_triattention_backend
```

如果只看到 `triattention` 一行：

```bash
# 修复：检查 setup.py 的 entry_points，重新装
grep -A 10 "entry_points" /Users/sunao2000/my_tri/setup.py
# 期望看到两行 entry point 注册
pip install -e /Users/sunao2000/my_tri --force-reinstall --no-deps
```

---

## 4. 准备 stats 文件

TriAttention 需要预计算的 Q/K 频率统计（每个模型一份）：

```bash
# 仓库自带两份：
ls /Users/sunao2000/my_tri/triattention/vllm/stats/
# 期望：
#   gpt_oss_120b_stats.pt
#   qwen3_32b_int4_stats.pt

# 复制到一个永久位置
mkdir -p ~/tri_stats
cp /Users/sunao2000/my_tri/triattention/vllm/stats/qwen3_32b_int4_stats.pt ~/tri_stats/
# 如果你的模型不是 Qwen3-32B-INT4，需要先 calibration（见 docs/calibration.md）
```

**stats 文件不匹配的症状**：压缩结果错误（attention 输出对不上），但**不报错**。模型与 stats 必须严格匹配。

---

## 5. 启动服务

```bash
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=~/tri_stats/qwen3_32b_int4_stats.pt
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# 启动
vllm serve Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code \
  --enable-prefix-caching false \
  --max-num-batched-tokens 1024
```

**关键 flag 解释**：

- `--enable-prefix-caching false`：prefix caching 与 TriAttention 物理压缩不兼容（一旦 KV 被压缩，block hash 链就断了，prefix cache 命中会读到错误数据）
- `--max-num-batched-tokens 1024`：限制 prefill chunk。更大的 chunk 会在第一次 prefill 时超过 budget 导致 OOM
- `--enforce-eager`：跳过 CUDA graph / ACL graph capture（让首步立即跑，省掉 warmup 干扰；生产环境可去掉）
- `--trust-remote-code`：Qwen3 等模型需要

---

## 6. 验证关键日志

启动后立刻看前 200 行日志，**必须按顺序看到**以下 6 条：

```bash
vllm serve ... 2>&1 | tee /tmp/vllm_serve.log
```

期望日志序列：

```
# (1) 主进程：plugin discovery
[TriAttention-Ascend] plugin entry point invoked: reason=load_general_plugins ascend_detected=True patch_scheduler=True patch_worker=True

# (2) 主进程：AIM first install
[TriAttention-Ascend] first install complete reason=load_general_plugins scheduler_class=Scheduler status={'scheduler': True, 'kv_cache_manager': True, 'npu_worker': True, 'ascend_block_tables': True, 'kv_utils': True, 'engine_core_async_step': True}

# (3) 主进程：meta-patch 装上
[TriAttention-Ascend] meta-patch installed on vllm.v1.core.sched.scheduler; future Scheduler rebinds will automatically receive TriAttention helper methods.

# (4) 主进程：plugin activation
[TriAttention-Ascend] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True status=...

# (5) 主进程：adapt_patch 触发 BalanceScheduler rebind（vllm-ascend 自有）
[TriAttention-Ascend] meta-patch: Scheduler rebound to BalanceScheduler; re-attached TriAttention helper methods on the new class (__init__/schedule/update_from_output inherited via MRO)

# (6) engine core 子进程：BalanceScheduler 实例化
[TriAttention-Ascend] Scheduler initialized: type=BalanceScheduler budget=2048 divide_length=128 ...

# (7) worker 子进程：NPUWorker 实例化
[TriAttention-Ascend] NPUWorker initialized: type=NPUWorker budget=2048 ...
```

**任何一个 log 没出现 = 对应链路断了**。参见 [§9 故障排查速查表](#9-故障排查速查表)。

---

## 7. 验证算法真的运行

```bash
# 7.1 发一个超过 KV budget (2048 tokens) 的 prompt
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Solve this AIME problem step by step: 1+2+...+n=1000, find n. Show all reasoning."}],
    "max_tokens": 4096
  }'
```

**期望在请求进行中**（看 `/tmp/vllm_serve.log`）：

```
# 第一个 signal 触发
[TriAttention-Ascend] signal triggered req=... step=N estimated_cache_len=M reason=length_threshold

# 同一个 step：worker 端 lazy install
[TriAttention-Ascend] lazily installed runner proxy: budget=2048 ...

# 下一个 step：scheduler 端 update_from_output
[TriAttention-Ascend] update_from_output: received N events (M applied) via model_runner_output
```

### 7.2 进一步验证：算法核心真的被调到

在 `triattention/vllm/core/scoring.py:compute_scores_triton` 第一行加临时 log：

```python
def compute_scores_triton(...):
    logger.info("[TriAttention] compute_scores_triton called: num_tokens=%d budget=%d", ...)  # 临时加
    ...
```

重启服务、再发请求，**必须**看到该日志。

### 7.3 进一步验证：物理 KV 真的压缩了

```bash
npu-smi info
# 期望：发送超长 prompt 过程中，KV 占用峰值比不开 TriAttention 时低
```

或者在请求前后打印 `block_pool.get_num_free_blocks()`：

```python
# 临时加在 _apply_compression_events 末尾
free_after = self.kv_cache_manager.block_pool.get_num_free_blocks()
logger.info("[TriAttention-Ascend] post-compression free blocks: %d", free_after)
```

---

## 8. 关闭 TriAttention（baseline 对比）

```bash
# 8.1 杀掉服务
pkill -f "vllm serve"

# 8.2 关闭 TriAttention 重新启动
export ENABLE_TRIATTENTION=0
vllm serve Qwen/Qwen3-8B ... # 同样的参数
```

**期望日志里完全没有 `[TriAttention-Ascend]` 字样**——证明总开关有效。

---

## 9. 故障排查速查表

| 现象 | 根因 | 修复 |
| ---- | ---- | ---- |
| 启动日志里完全没有 `[TriAttention-Ascend]` | entry point 没注册 | 回到 §3 重新装 |
| 启动日志里有 `plugin entry point invoked` 但 `first install complete status=...False` | AIM 异常 | 向上翻日志找 `ERROR` 行（通常在 `first install complete` 之前） |
| `first install complete` 里有 `npu_worker: False` | `vllm_ascend.worker.worker` import 失败 | 检查 `python -c "import vllm_ascend.worker.worker"` |
| `first install complete` 里有 `ascend_block_tables: False` | vllm-ascend 装的是 V1 block tables，不是 V2 | 检查 `vllm_ascend/worker/v2/block_table.py` 是否存在；如果不存在说明装的 vllm-ascend 版本不对 |
| `Scheduler initialized: type=Scheduler`（不是 `BalanceScheduler`） | vllm-ascend 的 `adapt_patch` 没跑 | 检查 `VLLM_ASCEND_BALANCE_SCHEDULING=1` 是否设了 |
| `signal triggered` 没出现 | 1) request 长度没超过 budget；2) `_build_signals` 跑过但所有 req 都没满足 threshold | 调小 `TRIATTN_RUNTIME_KV_BUDGET=256` 试一下 |
| `lazily installed runner proxy` 没出现 | `signals` 在 worker 端为空 | 检查 `_consume_signals` 是否真的拿到 signal；可能是 pickling 跨进程丢字段（vLLM 0.18.0 不会丢，除非用户用了一些非常规的 comm layer） |
| `lazily installed runner proxy` 出现但 `update_from_output: received N events` 是 0 applied | 算法核心跑了但物理压缩没成功 | 检查 stats 文件路径；检查 `num_layers` / `num_kv_heads` 是否匹配当前模型 |
| 内存没下降 | `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0` 或物理压缩没成功 | 检查上一个；如果都 OK，把 `enable_experimental_block_reclaim=1` 试一下 |
| 报 `RuntimeError: Could not find a KV cache memory check symbol to relax` | vllm 0.18.0 的 `kv_cache_utils` 内部签名变了 | 打开 `TRIATTENTION_QUIET=0` 查具体错；该 warning 不影响功能 |

### 9.1 完整 re-apply 流程（worker 子进程没装上的情况下）

如果怀疑 AIM 没在 worker 子进程里被调到：

```bash
# 1. 杀掉服务
pkill -f "vllm serve"

# 2. 清掉 Python 缓存
find /Users/sunao2000/my_tri/triattention -name "__pycache__" -type d | xargs rm -rf

# 3. 重新装
pip install -e /Users/sunao2000/my_tri --force-reinstall --no-deps

# 4. 重启服务
vllm serve ...
```

### 9.2 启用 debug 日志

```bash
export TRIATTENTION_QUIET=0
export TRIATTN_DEBUG_EARLY_INSTALL_PROXY=1   # 提前装 runner proxy（牺牲一些性能换可调试性）
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve ...
```

---

## 附录 A：完整的环境变量清单

| 变量 | 默认 | 用途 |
| ---- | ---- | ---- |
| `ENABLE_TRIATTENTION` | `true` | 总开关 |
| `TRIATTN_RUNTIME_KV_BUDGET` | `2048` | 保留的 KV 长度上限 |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | `128` | 每 N 个新 token 检查一次 |
| `TRIATTN_RUNTIME_WINDOW_SIZE` | `128` | 永远保留的最近 N 个 token |
| `TRIATTN_RUNTIME_PRUNING_MODE` | `per_head` | `per_head` 或 `per_layer_per_head` |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | (无) | stats 文件路径 |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | `false` | 是否保护 prompt 头 |
| `TRIATTN_RUNTIME_INCLUDE_PREFILL_IN_BUDGET` | `false` | prompt 是否计入 budget |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION` | `true` | 启用 in-place KV 压缩 |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM` | `true` | 启用物理 block 回收 |
| `TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING` | `true` | 必须用 triton kernel 评分 |
| `TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM` | `true` | 必须真的回收物理 block |
| `TRIATTN_RUNTIME_PATCH_SCHEDULER` | `true` | 是否 patch scheduler |
| `TRIATTN_RUNTIME_PATCH_WORKER` | `true` | 是否 patch worker |
| `TRIATTN_RUNTIME_LOG_DECISIONS` | `false` | 打印每个 decision 详细 log |
| `TRIATTENTION_QUIET` | `0` | 静默一些 INFO |
| `TRIATTN_DEBUG_EARLY_INSTALL_PROXY` | `0` | 提前装 runner proxy |
| `TRIATTN_DEBUG_V2_REWRITE_OUTPUT_REQ_MAP` | `0` | 调试 V2 output req map |
| `TRIATTENTION_INTERFACE` | `runtime` | `runtime` (新) / `legacy` (旧，已废弃) |
| `TRIATTENTION_KV_BUDGET` | (无) | 旧名字，自动 bridge 到 `TRIATTN_RUNTIME_KV_BUDGET` |
| `TRIATTENTION_STATS_PATH` | (无) | 旧名字，自动 bridge |

## 附录 B：与原 `TRIATTENTION_*` 环境变量的兼容

`triattention.vllm_ascend.plugin._bridge_legacy_env_to_runtime()` 在 entry point 入口自动把以下旧名字映射到 `TRIATTN_RUNTIME_*`：

- `TRIATTENTION_KV_BUDGET` → `TRIATTN_RUNTIME_KV_BUDGET`
- `TRIATTENTION_DIVIDE_LENGTH` → `TRIATTN_RUNTIME_DIVIDE_LENGTH`
- `TRIATTENTION_WINDOW_SIZE` → `TRIATTN_RUNTIME_WINDOW_SIZE`
- `TRIATTENTION_LOG_DECISIONS` → `TRIATTN_RUNTIME_LOG_DECISIONS`
- `TRIATTENTION_STATS_PATH` → `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
- `TRIATTENTION_PRUNING_MODE` → `TRIATTN_RUNTIME_PRUNING_MODE`（`per_layer_head` → `per_layer_per_head`）

所以旧文档里的环境变量**继续可用**。
