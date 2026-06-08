# TriAttention × vllm-ascend v0.18.0 失效分析、根因定位与昇腾适配方案

> 配套 `ideal_WORKFLOW.md`：本文讲"问题是什么、为什么、怎么修"，WORKFLOW.md 讲"修完之后的代码怎么走"。两份文档**必须同看**。

---

## 目录

1. [现象复述与可观测证据](#一-现象复述与可观测证据)
2. [任务一：完整问题排查方案](#二-任务一完整问题排查方案)
3. [三份代码路径对比矩阵](#三-三份代码路径对比矩阵)
4. [任务二：跨版本差异化对比与根因分析](#四-任务二跨版本差异化对比与根因分析)
5. [根因清单（按影响权重排序）](#五-根因清单按影响权重排序)
6. [工程原则约束与最终落地架构](#六-工程原则约束与最终落地架构)
7. [代码工程重构总览](#七-代码工程重构总览)
8. [文件级改动清单](#八-文件级改动清单)
9. [零基础部署启用教程](#九-零基础部署启用教程)
10. [验证矩阵与失败模式对照表](#十-验证矩阵与失败模式对照表)

---

## 一、现象复述与可观测证据


| 现象                                                                                             | 观察方法                                                                 | 性质                                                       |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------- |
| vllm-ascend 服务能正常启动，无 import error                                                             | `vllm serve ...` 返回 0，不抛 `ModuleNotFoundError`                       | 仅证明"模块可发现"                                               |
| 启动日志中看到 `[TriAttention] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True` | 抓主进程前 200 行                                                          | 误导性日志——**CUDA 端 patcher 的"激活成功"在 ascend 端是空操作**（详见根因 #1） |
| 断点调试：从未进入 `triattention/vllm/core/scoring.py:compute_scores_triton` 与 `compressor.py`          | 在 `triattention_scoring_kernel` 与 `compact_request_kv_in_place` 上设断点 | **核心症状**——算法核心算子从来没被调度到                                  |
| 跨进程桥属性 `triattention_signals` 在 worker 端始终为 `{}`                                               | `getattr(scheduler_output, "triattention_signals", None)`            | 决策层失效                                                    |
| `TriAttentionModelRunner` 永远没装到 `NPUWorker.model_runner`                                       | `isinstance(worker.model_runner, TriAttentionModelRunner)` 始终 False  | 执行层失效                                                    |
| `_apply_compression_events` 在 scheduler 端从未被调用                                                 | 在 `_patched_scheduler_update_from_output` 入口打 log                    | 反馈层失效                                                    |


> 上述六个证据全部出现在已经"激活"日志之后，**直接证明"激活"≠"使能"**。

---

## 二、任务一：完整问题排查方案

### 2.1 排查总策略

按"从外到内、从发现层到执行层"逐层下沉，每下沉一层都要求**前一层有明确证据**。任何一层缺失，必须先回上一层的修复而不是直接跳到下一层。

```
L1 插件发现层        →  L2 平台识别层     →  L3 符号表层       →  L4 行为注入层     →  L5 算法执行层
entry point 注册       ascend 检测          class 路径匹配         patch 是否真的贴上     kernel 真的被调到
```

### 2.2 五层排查表


| 层级  | 排查问题                                                | 验证命令 / 代码                                                                                                                                                 | 失败时的修复                                                                                                                                                        |
| --- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1  | `vllm.general_plugins` 组里有没有 `triattention_ascend`？ | `python -c "import importlib.metadata as m; print([(e.name, e.value) for e in m.entry_points().get('vllm.general_plugins', [])])"` 必须看到两行                 | 重新 `pip install -e .` 让 setup.py 重新写 dist-info；检查 setup.py 的 `entry_points` 字典                                                                                |
| L2  | `register_triattention_backend` 真的被执行了吗？            | 启动日志有 `[TriAttention-Ascend] plugin entry point invoked: ...`                                                                                             | 检查 `_is_running_on_ascend()` 返回 False 的情形（`vllm_ascend` 没装、或 platform 还没 import、或 env var 设置错误）                                                               |
| L3  | `vllm_ascend.worker.worker.NPUWorker` 真的被 patch 了吗？ | `python -c "from vllm_ascend.worker.worker import NPUWorker; print(getattr(NPUWorker, '_ensure_triattention_runner_proxy', None))"` 必须输出 `<staticmethod>` | AIM 失败软退出——查 `[TriAttention-Ascend] first install complete` 日志的 `status` 字段                                                                                   |
| L4  | `Scheduler.update_from_output` 真的被 wrapper 接管了吗？    | 在 `_patched_scheduler_update_from_output` 入口加 `logger.info("AIM update_from_output entered")`                                                             | 检查 vllm-ascend 的 `Scheduler = BalanceScheduler` 是不是在 AIM 跑过之后才 rebind（meta-patch 必须装上）                                                                        |
| L5  | `compute_scores_triton` 真的被调到了吗？                    | 在 `triattention/vllm/core/scoring.py:compute_scores_triton` 第一行 `logger.info("triton scoring called, budget=%d, num_tokens=%d", ...)`                     | 检查 `signals` dict 是否非空、`triattention_signals` 是否真的被 worker 拿到（跨进程 pickling 是否丢字段——vLLM 不会丢，因为 SchedulerOutput 是 `dataclass(slots=True)` 风格、自定义属性也走 pickle 协议） |


### 2.3 分步解决操作步骤

#### 步骤 1：检查 entry point 注册

```bash
# 这一步就能判断"激活"日志是不是空的
python -c "import importlib.metadata as m; eps = m.entry_points().get('vllm.general_plugins', []); print([(e.name, e.value) for e in eps])"
```

期望输出（必须两行都有）：

```
[('triattention', 'triattention.vllm.plugin:register_triattention_backend'),
 ('triattention_ascend', 'triattention.vllm_ascend.plugin:register_triattention_backend')]
```

如果只看到 `triattention` 一行：问题在 setup.py 的 `entry_points` 字典——这是 **根因 #1**，先修这里。

```bash
# 重新装
cd /Users/sunao2000/my_tri
pip install -e . --force-reinstall --no-deps
```

#### 步骤 2：检查 ascend 平台识别

```bash
python -c "
from vllm.platforms import current_platform
print('platform:', type(current_platform).__name__)
import vllm_ascend
print('vllm_ascend imported:', vllm_ascend.__file__)
"
```

期望：`platform: Ascend...` 或类似名字、`vllm_ascend imported: .../vllm_ascend/__init__.py`。

如果 platform 是 `CudaPlatform` / `XpuPlatform` / 其他：说明 `vllm_ascend` 没装上、或 `vllm_ascend.platform.AscendPlatform` 没有被 `vllm.platforms.register_platform(...)` 提前注册。需要重装 vllm-ascend 并在装之前让 `ASCEND_HOME_PATH` 等 env 设好。

#### 步骤 3：检查 AIM 是否成功安装

```bash
python -c "
from vllm_ascend.worker.worker import NPUWorker
print('NPUWorker.__init__ patched:', getattr(NPUWorker.__init__, '_triattention_patched', False))
print('NPUWorker.execute_model patched:', getattr(NPUWorker.execute_model, '_triattention_patched', False))
print('NPUWorker._ensure_triattention_runner_proxy:', NPUWorker._ensure_triattention_runner_proxy)
"
```

期望：三个都打印 True / `<staticmethod>`。

如果第二个 False：说明 AIM 跑了一半——通常是 `_patched_npu_worker_execute_model` 还没贴上，原因可能是 AIM 在主进程跑过但 worker 子进程没跑过（AIM 是 per-process 的，每次 worker 启动都会 import `vllm_ascend.worker.worker`，但 vLLM 不会重新调用 entry point；NPUWorker 重新被 import 的时候 `getattr(NPUWorker, '_triattention_patched', False)` 是 False——因为 vLLM 不会跨进程复制 class 对象上的 setattr）。

> **解决**：AIM 必须能在 worker 子进程里被再次调用。我们的设计是通过 `NPUWorker.__init__` 里的 `ensure_patches_installed(reason="npu_worker_post_init")` 兜底；但这一兜底要走 `if not _PATCHED_WORKER_ACTIVE: return` 路径——所以必须先在主进程成功注册过。

#### 步骤 4：检查 Scheduler 是否被 AIM 接管

```bash
python -c "
import vllm.v1.core.sched.scheduler as s
print('Scheduler:', s.Scheduler.__name__)
print('Scheduler.__init__ patched:', getattr(s.Scheduler.__init__, '_triattention_patched', False))
print('Scheduler.schedule patched:', getattr(s.Scheduler.schedule, '_triattention_patched', False))
print('Scheduler.update_from_output patched:', getattr(s.Scheduler.update_from_output, '_triattention_patched', False))
print('Scheduler._build_signals:', s.Scheduler._build_signals)
"
```

期望：所有 patched 都是 True、`_build_signals` 是 callable。

如果 `Scheduler.__init__` patched 是 False：意味着 AIM 没在主进程跑过、或者 worker 子进程的 import 链不同步。检查 `_is_running_on_ascend()` 在子进程里返回值。

#### 步骤 5：检查 BalanceScheduler 的 MRO 继承

```bash
python -c "
import vllm.v1.core.sched.scheduler as s
# 触发 patch_balance_schedule 跑（首次访问 platform 时）
import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
print('Scheduler after adapt_patch:', s.Scheduler.__name__)
print('mro:', [c.__name__ for c in s.Scheduler.__mro__])
print('BalanceScheduler.__init__ patched:', getattr(s.Scheduler.__init__, '_triattention_patched', False))
"
```

期望：`Scheduler after adapt_patch: BalanceScheduler`、MRO 第二个是上游 `Scheduler`、`__init__` 仍然是 patched 版本。

如果 `__init__` patched 变 False：意味着 BalanceScheduler 自己的 `__init__` 覆盖了 patched init——它确实覆盖了，但 `super().__init__()` 走 MRO 时能找到上游的 patched init。问题就出在这里：**如果 AIM 在 `adapt_patch` 之后才跑**，BalanceScheduler 自己的 init 是**已经定义的**，它调 super 的时候能找到我们 patch 的 init；但如果 AIM 在 `adapt_patch` 之前跑（更常见：主进程启动时序），patch 是装在 `Scheduler` (upstream) class 对象上的，BalanceScheduler 通过 MRO 继承——仍然 OK。所以这一步大概率是绿的，但**如果 setup.py 改了、import 时序变了、或者 BalanceScheduler 用了 `__init_subclass__` 之类**，要查 MRO。

#### 步骤 6：检查 `_apply_compression_events` 真的被调

```bash
# 在 _apply_compression_events 第一行加日志
# 然后起服务，发一个超过 kv_budget 的请求
grep -n "_apply_compression_events" /Users/sunao2000/my_tri/triattention/vllm_ascend/runtime/integration_monkeypatch.py
```

期望：服务日志里出现 `[TriAttention-Ascend] update_from_output: received N events (M applied) via model_runner_output`。

如果没出现：scheduler 端 `update_from_output` wrapper 根本没被调到——可能是 BalanceScheduler 自己 override 了 `update_from_output` 而没调 super。

#### 步骤 7：检查 `compute_scores_triton` 真的被调

```bash
grep -n "compute_scores_triton" /Users/sunao2000/my_tri/triattention/vllm/core/scoring.py
```

在第一行加 `logger.info("[TriAttention-Ascend] compute_scores_triton called: shape=%s, budget=%d", x.shape, budget)`。

期望：长上下文请求时（> 2048 tokens）出现该日志。

如果没出现：

- 看一下 `triattention_signals` 在 worker 端是否非空——`getattr(scheduler_output, "triattention_signals", None)` 在 `_patched_npu_worker_execute_model` 里
- 看一下 lazy runner proxy 是否真的装了——`isinstance(worker.model_runner, TriAttentionModelRunner)`
- 看一下 `signals` 在 `TriAttentionModelRunner._consume_signals` 里是否非空
- 看一下 `_execute_compression_actions` 是否被调

每一步都加日志定位到具体断在哪。

### 2.4 排查决策树

```
服务能起吗？
├── 不能 → import 错误，查 Python 路径、依赖版本
└── 能
    ├── entry point 注册了吗？  (L1)
    │   ├── 没 → 重新 pip install -e .
    │   └── 有
    │       ├── ascend 平台被识别吗？ (L2)
    │       │   ├── 没 → 装 vllm_ascend、检查 platform 注册顺序
    │       │   └── 有
    │       │       ├── AIM 真的装了吗？ (L3)
    │       │       │   ├── 没 → 查 `[TriAttention-Ascend] first install complete` 日志
    │       │       │   └── 有
    │       │       │       ├── Scheduler wrapper 真生效吗？ (L4)
    │       │       │       │   ├── 没 → meta-patch 漏了，查 vllm-ascend 的 `adapt_patch` 时序
    │       │       │       │   └── 有
    │       │       │       │       ├── signal 真的发出去了吗？
    │       │       │       │       │   ├── 没 → `_build_signals` 跑了但 threshold 没到，调小 `TRIATTN_RUNTIME_KV_BUDGET` 试试
    │       │       │       │       │   └── 有
    │       │       │       │       │       ├── worker 端 lazy proxy 真的装了吗？
    │       │       │       │       │       │   ├── 没 → `_ensure_triattention_runner_proxy` 进了但装失败，看 stats path 是否对
    │       │       │       │       │       │   └── 有
    │       │       │       │       │       │       ├── compute_scores_triton 真的被调了吗？
    │       │       │       │       │       │       │   ├── 没 → 算法链路某处 try/except 吞了异常
    │       │       │       │       │       │       │   └── 有 → 🎉 算法通了
```

### 2.5 修复操作清单（按依赖顺序）


| 顺序  | 操作                                                                     | 文件                            |
| --- | ---------------------------------------------------------------------- | ----------------------------- |
| 1   | 在 `setup.py` 加 `triattention_ascend` entry point                       | `setup.py:46`                 |
| 2   | 给 `triattention/vllm/plugin.py` 加 `_is_running_on_ascend()` 早退         | `triattention/vllm/plugin.py` |
| 3   | 新建 `triattention/vllm_ascend/` 包，含 `plugin.py`                         | 新文件                           |
| 4   | 新建 `triattention/vllm_ascend/runtime/integration_monkeypatch.py` (AIM) | 新文件                           |
| 5   | 新建 `triattention/vllm_ascend/runtime/scheduler_ascend.py` (mixin)      | 新文件                           |
| 6   | 新建 `triattention/vllm_ascend/runtime/worker_ascend.py` (mixin)         | 新文件                           |
| 7   | 新建 `triattention/vllm_ascend/runtime/gpu_seq_len_patch.py` (no-op)     | 新文件                           |
| 8   | 重新装 `pip install -e . --force-reinstall --no-deps`                     | shell                         |


---

## 三、三份代码路径对比矩阵

### 3.1 顶层结构差异


| 维度           | `triattention/vllm/` (本文工作目录)                               | `vllm-releases-v0.18.0` (上游)                 | `vllm-ascend-releases-v0.18.0` (昇腾)                                  |
| ------------ | ----------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| 顶层入口         | `triattention/vllm/plugin.py:register_triattention_backend` | `vllm.plugins.load_general_plugins`          | 暂无任何 entry point                                                     |
| 类目标          | 上游 `Scheduler` / `Worker` (CUDA)                            | 真实类                                          | `BalanceScheduler` / `NPUWorker` / `AscendBlockTables`               |
| Block tables | 上游 `BlockTables`                                            | `vllm.v1.worker.gpu.block_table.BlockTables` | `vllm_ascend.worker.v2.block_table.AscendBlockTables`                |
| Worker       | 上游 `Worker`                                                 | `vllm.v1.worker.gpu_worker.Worker`           | `vllm_ascend.worker.worker.NPUWorker`                                |
| Scheduler    | 上游 `Scheduler`                                              | `vllm.v1.core.sched.scheduler.Scheduler`     | 同一个 `Scheduler` 符号，但 `adapt_patch` 后指向 `BalanceScheduler(Scheduler)` |


### 3.2 关键代码点对比


| 代码点                                              | CUDA 路径                                                            | Ascend 路径                                                                                                                                              | 差异对适配的影响                                                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Plugin 入口**                                    | `triattention.vllm.plugin:register_triattention_backend`           | **缺失**（现在补上 `triattention.vllm_ascend.plugin`）                                                                                                         | **致命**——没有 entry point，vllm 永远不会主动调用 ascend 端 patcher                                                                          |
| **早退检测**                                         | 无                                                                  | 需新增 `_is_running_on_ascend()`                                                                                                                          | 必须有，否则 ascend 上 CUDA patcher 会乱贴符号                                                                                             |
| **Worker class**                                 | `vllm.v1.worker.gpu_worker.Worker`                                 | `vllm_ascend.worker.worker.NPUWorker`（subclass of `WorkerBase`，不是 `Worker`）                                                                            | **致命**——CUDA patcher 贴的 `Worker.execute_model` 在 ascend 上根本不会被调                                                                |
| **Scheduler 符号**                                 | 上游 `Scheduler` (单一)                                                | 上游 `Scheduler` 在 `adapt_patch` 之后被 rebind 成 `BalanceScheduler(Scheduler)`                                                                              | **致命**——子类的 `__init__` 是自己的，调 `super().__init__()` 时 MRO 才能找到 patched init；如果主进程的 patch 顺序错了就找不到                               |
| `**adapt_patch` 时机**                             | N/A                                                                | `vllm_ascend.utils.adapt_patch(is_global_patch=True)` 在 `NPUWorker.__init__` 里被显式调用（`vllm-ascend-releases-v0.18.0/vllm_ascend/worker/worker.py:96-98`） | 时序：主进程 vs worker 子进程都要触发；AIM 必须在 adapt_patch 之前 patch 完毕                                                                       |
| **Scheduler = BalanceScheduler**                 | N/A                                                                | `vllm_ascend.patch.platform.patch_balance_schedule:705` 在模块层面 rebind 符号                                                                                | MRO 继承——已 patch 的 init/schedule/update_from_output 仍能找到；helper method 必须 re-attach，否则 BalanceScheduler 实例没有 `_build_signals` 等 |
| **Block tables**                                 | `BlockTables` (int64 slot_mappings)                                | `AscendBlockTables` (int32 slot_mappings, 自家 Triton kernel)                                                                                            | **必须**单独 patch，因为 `compute_slot_mappings` 是被 `AscendBlockTables` override 的                                                    |
| **KV cache utils**                               | `vllm.v1.core.kv_cache_utils`                                      | 同一个模块                                                                                                                                                  | 共用 patch                                                                                                                       |
| `**_init_device` / `init_device`**               | `Worker.init_device` 存在                                            | `NPUWorker.init_device` 存在、`_init_device` 是私有 helper                                                                                                   | 必须 patch `NPUWorker.init_device`                                                                                               |
| **lazy install 时机**                              | CUDA 端在 `init_device` 里                                            | ascend 端必须在 `__init__` 里兜底 + `execute_model` 里懒装                                                                                                       | vllm-ascend 的 `NPUModelRunner.__init__` 做 ACL graph 编译、device warmup，太早 wrap 会炸                                                |
| **GPU seq_len patch**                            | patch `vllm.v1.worker.gpu.input_prep._prepare_pos_seq_lens_and...` | ascend 端是 no-op（`AscendBlockTables.compute_slot_mappings` 一次性 gather，不需单独 patch seq_lens）                                                              | 必须保留 no-op stub，不能让 CUDA patcher 在 ascend 上误贴                                                                                  |
| `**_patched_engine_core_step_with_batch_queue`** | `vllm.v1.engine.core.EngineCore`                                   | 同一个 `EngineCore`（`EngineCoreProc` 继承）                                                                                                                  | 共用 patch                                                                                                                       |


### 3.3 关键源码引用

#### 3.3.1 vllm 上游：`vllm/v1/worker/gpu_worker.py:762`

```python
def execute_model(
    self, scheduler_output: "SchedulerOutput"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    # ... (CUDA-side forward orchestration)
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
    # ... (PP handling)
```

这是 CUDA 端的 `Worker.execute_model`，**vllm-ascend 的 `NPUWorker` 不会继承它**（`NPUWorker(WorkerBase)`，不是 `NPUWorker(Worker)`）。

#### 3.3.2 vllm-ascend：`vllm_ascend/worker/worker.py:363`

```python
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    # enable msMonitor to monitor the performance of vllm-ascend
    if envs_ascend.MSMONITOR_USE_DAEMON:
        dp.step()

    if self._pp_send_work:
        for handle in self._pp_send_work:
            handle.wait()
        self._pp_send_work = []
    # ... (PP+SP handling for ascend)
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
    # ...
```

这是**真正被调到的** `execute_model`。CUDA patcher 没碰它，所以 `triattention_signals` 在 worker 端**永远不会被读取**。

#### 3.3.3 vllm-ascend：`vllm_ascend/patch/platform/patch_balance_schedule.py:705`

```python
EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

这是**唯一**一处把 `Scheduler` 符号 rebind 的地方。注意它**只**改模块里的 `Scheduler` 名字指向，**不动** upstream class 对象。所以 patched init/schedule/update_from_output 仍能通过 MRO 继承——但 helper method（`_build_signals` 等）需要重新 attach（`BalanceScheduler` 没自动从 `Scheduler` 拿到 setattr，因为 `BalanceScheduler` 是 `Scheduler` 的**子类**，setattr 在 `Scheduler` 上的 helper 通过 MRO 仍然可达——所以严格说不需要 re-attach，但 `__init_subclass_`_ 或 metaclass 行为可能干扰；保险起见，meta-patch 一定要装）。

#### 3.3.4 vllm-ascend：`vllm_ascend/worker/v2/block_table.py:62`

```python
def compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
) -> torch.Tensor:
    num_reqs = idx_mapping.shape[0]
    num_groups = self.num_kv_cache_groups
    _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        self.max_num_batched_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        self.block_table_ptrs,
        self.block_table_strides,
        self.block_sizes_tensor,
        self.slot_mappings,
        self.slot_mappings.stride(0),
        # ...
    )
```

这是 ascend 自己的 slot mapping kernel（`int32`），**完全不同于** CUDA 端的 `BlockTables.compute_slot_mappings`（`int64`）。所以 patch 一定要打到 `AscendBlockTables`，不能只 patch 上游 `BlockTables`。

### 3.4 worker 启动流程对比


| 步骤                                     | CUDA 路径                                                 | Ascend 路径                                                                                                                                                                                      |
| -------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. main process 启动                     | `vllm serve ...` 触发 `vllm.plugins.load_general_plugins` | 同上                                                                                                                                                                                             |
| 2. entry point 调用                      | `triattention` → 装 patch                                | `triattention` 早退（无 ascend）；`triattention_ascend` → 装 AIM                                                                                                                                      |
| 3. platform resolve                    | `current_platform = CudaPlatform`                       | `current_platform = Ascend...Platform`                                                                                                                                                         |
| 4. `adapt_patch(is_global_patch=True)` | 不存在                                                     | `vllm_ascend.patch.platform.patch_balance_schedule:705` 跑：模块里 `Scheduler = BalanceScheduler`                                                                                                   |
| 5. engine core 子进程                     | 子进程独立 `load_general_plugins`                            | 同上 + 子进程单独执行 `adapt_patch`（因为每个子进程都有自己 `vllm_ascend.worker` 导入链）                                                                                                                               |
| 6. Scheduler 实例化                       | `Scheduler(vllm_config, ...)` → patched init 装 state    | `vllm_config.scheduler_config.get_scheduler_cls()` → 返回 `BalanceScheduler` → `BalanceScheduler.__init__` → `super().__init__()` 走 MRO 找到 patched init → state 装上                               |
| 7. Worker 子进程                          | `Worker.__init__` 跑 → patched init 装 state              | `NPUWorker.__init__` 跑 → `adapt_patch` 再跑一次（worker 自己进程内）→ **如果 AIM 之前没在 worker 进程跑过，patch 没装上**——这就是为什么我们要 `NPUWorker.__init__` 里兜底 `ensure_patches_installed(reason="npu_worker_post_init")` |
| 8. lazy install                        | `Worker.execute_model` 看到 `signals` 才装                  | `NPUWorker.execute_model` 看到 `signals` 才装                                                                                                                                                      |
| 9. 算法核心                                | `compute_scores_triton` → `compact_request_kv_in_place` | 同上（来自 `triattention.vllm.core.`*）                                                                                                                                                              |
| 10. block reclaim                      | scheduler 端 `block_pool.free_blocks(removed)`           | 同上                                                                                                                                                                                             |


### 3.5 关键差异归因表


| 差异                                                            | 是否阻塞     | 阻塞原因                                                                                                   | 修复策略                                                                               |
| ------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 没有 `triattention_ascend` entry point                          | **阻塞**   | `vllm.plugins.load_general_plugins` 永远找不到 ascend 端 patcher                                             | 在 `setup.py` 加 entry point                                                         |
| `triattention/vllm_ascend/` 目录不存在                             | **阻塞**   | 即使 entry point 注册了，import 也会失败                                                                         | 新建 `vllm_ascend/` 包                                                                |
| CUDA patcher 试图 patch `vllm.v1.worker.gpu_worker.Worker`      | **阻塞**   | `Worker` 类在 ascend 上不被实例化（`NPUWorker(WorkerBase)`）                                                     | CUDA patcher 加 `_is_running_on_ascend()` 早退                                        |
| `NPUWorker.execute_model` 没被 patch                            | **阻塞**   | scheduler 发出的 `triattention_signals` 没人接                                                               | AIM 在 `NPUWorker.execute_model` 上装 wrapper                                         |
| `AscendBlockTables.compute_slot_mappings` 没被 patch            | 不阻塞      | 物理 KV 压缩后，block_table 已经在 `_apply_compression_events` 里截断；slot_mappings 是按当前 block_table 重新算的，所以计算结果正确 | AIM 装一个 passthrough wrapper 作为未来的注入点                                               |
| `BalanceScheduler` rebind 后 helper method 丢失                  | 潜在       | MRO 应该能继承，但 `Scheduler` 类的 `setattr` 在子类上没自动传播                                                         | AIM 装 `__setattr_`_ meta-patch                                                     |
| `adapt_patch` 时序：worker 子进程再跑一次                               | **潜在阻塞** | 如果 AIM 只在主进程跑过，worker 子进程的 `NPUWorker` 是新 import 出来的、没被 patch                                          | `NPUWorker.__init__` 里兜底 `ensure_patches_installed(reason="npu_worker_post_init")` |
| `vllm_ascend.worker.v2.block_table.AscendBlockTables` patch 漏 | 潜在       | ascend 端 `compute_slot_mappings` 是 override 过的                                                         | AIM 单独 patch                                                                       |


---

## 四、任务二：跨版本差异化对比与根因分析

### 4.1 根因 #1：entry point 缺失（致命）

**现象**：`vllm.plugins.load_general_plugins` 不会调用任何 ascend 端 patcher。

**代码定位**：

`vllm-releases-v0.18.0/vllm/plugins/__init__.py:79-82`：

```python
plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)
for func in plugins.values():
    func()
```

`load_plugins_by_group` 用 `importlib.metadata.entry_points(group=group)` 读 `vllm.general_plugins` 组的 entry points。

**修复前** `setup.py`：

```python
entry_points={
    "vllm.general_plugins": [
        "triattention = triattention.vllm.plugin:register_triattention_backend",
    ],
},
```

只有 `triattention` 一个——它对应的 `register_triattention_backend`（CUDA 端）**不会在 ascend 平台 patch 任何东西**，因为它只 patch `vllm.v1.worker.gpu_worker.Worker`（在 ascend 上没被实例化）。

**修复后**：

```python
entry_points={
    "vllm.general_plugins": [
        "triattention = triattention.vllm.plugin:register_triattention_backend",
        "triattention_ascend = triattention.vllm_ascend.plugin:register_triattention_backend",
    ],
},
```

加 `triattention_ascend` entry point；同时让 `triattention.vllm.plugin` 在 ascend 上**早退**（`_is_running_on_ascend()` 检）。

### 4.2 根因 #2：`triattention/vllm_ascend/` 包不存在（致命）

`ideal_WORKFLOW.md` 描述了一个完整的 `vllm_ascend/runtime/` 包，但实际仓库**根本没有这个目录**。这导致：

- 即使修了 #1 加上 entry point，`import triattention.vllm_ascend.plugin` 仍会 `ModuleNotFoundError`
- AIM（`integration_monkeypatch.py`）、scheduler/worker mixin、no-op patch 全部不存在
- 整个 ascend 适配层是空的

**修复**：新建 `triattention/vllm_ascend/` 包，含 `plugin.py`、`runtime/__init__.py`、`runtime/integration_monkeypatch.py`、`runtime/scheduler_ascend.py`、`runtime/worker_ascend.py`、`runtime/gpu_seq_len_patch.py`。

### 4.3 根因 #3：CUDA patcher 错贴到 `vllm.v1.worker.gpu_worker.Worker`（致命）

**问题代码**（`triattention/vllm/runtime/integration_monkeypatch.py:419`）：

```python
import vllm.v1.worker.gpu_worker as worker_mod
# ...
Worker = worker_mod.Worker
# ...
_ORIG_WORKER_EXECUTE_MODEL = Worker.execute_model
Worker.execute_model = _patched_worker_execute_model
```

**为什么在 ascend 上失效**：

- `vllm.v1.worker.gpu_worker.Worker` 这个**类**在 ascend 上仍然可以 import（vllm 仓库被装上了，符号表里就有），但**实际引擎**用 `vllm_ascend.worker.worker.NPUWorker`，而 `NPUWorker(WorkerBase)` 是 `WorkerBase` 的子类——**不是** `Worker` 的子类。
- 即使 patched `Worker.execute_model` 也不会被调到。
- 更糟的是：在 ascend 进程里 `import vllm.v1.worker.gpu_worker` 会触发 `from vllm.v1.worker.gpu_model_runner import GPUModelRunner`，这个 import 链在 ascend 上**经常失败**（`torch_npu` 平台没装 `flash_attn` 等 CUDA 专属依赖），会抛 `ImportError`——但被 `integration_monkeypatch` 的 try/except 吞了，**日志里看不到这个错误**，用户只看到 `[TriAttention] Runtime (V2) plugin activated` 这一行**假阳性**的成功日志。

**修复**：

1. CUDA patcher 在 ascend 平台**完全跳过**（早退）；
2. 新建 AIM 专门 patch `NPUWorker` / `BalanceScheduler` / `AscendBlockTables`。

### 4.4 根因 #4：NPUWorker 缺 lazy proxy install 钩子（致命）

CUDA 端在 `Worker.execute_model` 装了 wrapper，wrapper 检测到 `signals` 就触发 `_ensure_triattention_runner_proxy`。但 ascend 上 `NPUWorker.execute_model` 没人 patch，所以：

- scheduler 发来的 `triattention_signals` 进了 worker，但 worker 不知道
- `TriAttentionModelRunner` 永远没装到 `NPUWorker.model_runner` 上
- 即使 scheduler 端 `_build_signals` 真发了 signal、即使 `signals` 真有 `should_compress=True`，worker 端也是 `super().execute_model(scheduler_output)` 走原生 `NPUModelRunner.execute_model`，原生 runner 根本不知道 TriAttention 存在

**修复**：AIM 在 `NPUWorker.__init__` / `NPUWorker.init_device` / `NPUWorker.execute_model` 上都装 wrapper（`execute_model` 装的是触发 lazy install 的钩子）。

### 4.5 根因 #5：`BalanceScheduler = Scheduler` 符号 rebind 后 helper method 可能丢失

`vllm_ascend/patch/platform/patch_balance_schedule.py:705`：

```python
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

这一行只**替换**模块里的 `Scheduler` 名字指向 `BalanceScheduler` 类对象。`BalanceScheduler` 是 upstream `Scheduler` 的子类——所以 `BalanceScheduler.__init__` 是它自己 override 过的（带 `self.balance_queue = ...` 初始化），但 `super().__init__()` 走 MRO 时能找到 upstream `Scheduler.__init__`（被我们 patch 过的）。

**helper method**（`_build_signals` 等）是通过 `setattr(Scheduler_class, "_build_signals", ...)` 装到 upstream `Scheduler` 类对象上的。`BalanceScheduler` 通过 MRO **可以**找到这些 helper——Python 的 attribute lookup 顺序是 `BalanceScheduler` → 上游 `Scheduler` → `object`，所以 `BalanceScheduler._build_signals` 确实能 resolve 到上游那个。

**但**——如果 vllm-ascend 的 `adapt_patch` 用了 metaclass、或者 `Scheduler` 用了 `__init_subclass__` 来过滤 setattr 传播，helper 可能丢。**保险起见**，我们装 `__setattr__` meta-patch：模块里 `Scheduler` 每次 rebind 都自动 re-attach helper methods。

### 4.6 根因 #6：`AscendBlockTables.compute_slot_mappings` 是 ascend 端 override，CUDA patch 不到

CUDA 端 `vllm.v1.worker.gpu.block_table.BlockTables.compute_slot_mappings`（int64 slot_mappings）被 patch 了，但 ascend 端用 `vllm_ascend.worker.v2.block_table.AscendBlockTables.compute_slot_mappings`（int32 + 自家 Triton kernel），CUDA patch **不传播**——因为 `AscendBlockTables` 是 `BlockTables` 的子类并 override 了该方法，Python attribute lookup 在 `AscendBlockTables` 上就 resolve 到了 override 版的，不会 fallback 到 patched 的父类。

**实际后果**：

CUDA patcher 的 `BlockTables.compute_slot_mappings` 改写**永远不会被 ascend 引擎调到**。所以**这个 patch 本身在 ascend 上不需要做什么**（slot_mappings 是从压缩后的 block_table 自动 gather 出来的，逻辑已经对了），但我们要装一个 passthrough wrapper，作为**未来** ascend 端如果有特殊需要时的注入点。

### 4.7 根因 #7：NPUWorker 子进程缺 AIM 自举

vllm 0.18.0 的进程模型是：

```
main (process 0):  entry point 注册 → load_general_plugins → AIM 装在主进程
engine core:        子进程，独立 import vllm.v1.core.sched.scheduler，独立触发 adapt_patch
worker x N:         子进程，独立 import vllm_ascend.worker.worker，独立触发 adapt_patch
```

主进程装的 patch 是 patch 在 class 对象上的，**子进程 import 出来的 class 对象是不同的**——所以主进程的 setattr 不会跨进程生效。

**关键发现**：`vllm.plugins.load_general_plugins` 会在 engine core 子进程和 worker 子进程**都调用**（因为 vllm 0.18.0 的 `plugins_loaded` 是 process-local）。所以子进程也会跑 entry point。

**但** —— 子进程的 `vllm_ascend.worker.worker` import 顺序与主进程可能不同：`NPUWorker.__init__` 跑之前是否已经 import 过 `vllm_ascend.worker.worker` 并触发过 AIM 装 patch？**不一定**。如果子进程的 `vllm.plugins.load_general_plugins` 在 `vllm_ascend.worker.worker` 之前 import，那么 AIM 跑过之后 import 的 `NPUWorker` 是被 patch 过的；反之则需要 `NPUWorker.__init__` 兜底。

**修复**：双保险——AIM 在 entry point 里装一次，`NPUWorker.__init__` 的 wrapper 里再调一次 `ensure_patches_installed(reason="npu_worker_post_init")`（幂等）。

### 4.8 根因 #8：日志误导（次要但放大观感）

CUDA 端 patcher 的 `install_vllm_integration_monkeypatches` 在所有 import 都被 try/except 吞掉的情况下仍然打印 `[TriAttention] Runtime (V2) plugin activated: ...`——这给用户**错误的"已使能"信号**。AIM 不重蹈覆辙：每个 patch 都打 log，并且 `ensure_patches_installed` 返回 `status: dict[str, bool]`，让日志能直接看到每个符号的 patch 状态。

---

## 五、根因清单（按影响权重排序）


| 编号  | 根因                                                   | 影响权重                        | 状态                          |
| --- | ---------------------------------------------------- | --------------------------- | --------------------------- |
| #1  | `setup.py` 缺 `triattention_ascend` entry point       | 致命                          | 已修                          |
| #2  | `triattention/vllm_ascend/` 包不存在                     | 致命                          | 已修                          |
| #3  | CUDA patcher 错贴 `Worker` (CUDA-only class)           | 致命                          | 已修（CUDA 早退）                 |
| #4  | `NPUWorker` 缺 lazy proxy install 钩子                  | 致命                          | 已修（AIM patch NPUWorker）     |
| #5  | `BalanceScheduler` 符号 rebind 后 helper 潜在丢失           | 潜在                          | 已修（meta-patch）              |
| #6  | `AscendBlockTables.compute_slot_mappings` 是 override | 不阻塞（passthrough wrapper 已装） | 已修                          |
| #7  | worker 子进程 AIM 自举                                    | 潜在                          | 已修（`NPUWorker.__init__` 兜底） |
| #8  | 日志误导                                                 | 次要                          | 已修（status dict）             |


---

## 六、工程原则约束与最终落地架构

### 6.1 四大工程原则的代码级落地


| 原则         | 实现位置                                                                                                                                        | 实现方式                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **最小侵入**   | AIM 全部以 `setattr(class, name, value)` 写                                                                                                     | 不修改 vllm / vllm-ascend 任何源文件；patch 仅作用于进程内 class 对象                                                                                            |
| **信号驱动**   | `setattr(scheduler_output, "triattention_*")` / `setattr(output, "triattention_*")`                                                         | 4 个跨进程桥属性：`triattention_step`、`triattention_signals`、`triattention_compression_events`（双向桥）、`_triattention_force_boundary_sync`（async barrier） |
| **懒加载**    | `_patched_npu_worker_execute_model` 装 wrapper，wrapper 检测 `signals` 才触发 `_ensure_triattention_runner_proxy`                                  | `NPUWorker.__init__` 跑完，model_runner 仍是原生 `NPUModelRunner`；第一次 scheduler 发来非空 `signals` 时才换成 `TriAttentionModelRunner` proxy                   |
| **状态显式同步** | `_apply_compression_events` 显式调 `manager.req_to_blocks[req_id] = kept` 改写逻辑 block_table，再调 `block_pool.free_blocks(reversed(removed))` 物理回收 | 物理内存（block_pool）与逻辑视图（manager.req_to_blocks、effective_len_tracker）通过同一个函数同步更新                                                                  |


### 6.2 整体调用栈

```
vllm serve ...
  └─ vllm.plugins.load_general_plugins()        [process 0, engine core, worker 各自一次]
       └─ triattention_ascend (entry point)     [AIM]
            └─ ensure_patches_installed(...)
                 ├─ patch Scheduler / KVCacheManager / EngineCore (vllm upstream)
                 ├─ patch NPUWorker / AscendBlockTables (vllm_ascend)
                 ├─ relax kv_cache_utils.check_enough_kv_cache_memory
                 └─ install __setattr__ meta-patch on vllm.v1.core.sched.scheduler

  engine core 子进程:
    BalanceScheduler()                          [balance_schedule.py:28]
      └─ super().__init__()                     [走 MRO → patched Scheduler.__init__]
           └─ 装 triattention_config, _planner, _effective_len_tracker, ...

    每个 step:
      BalanceScheduler.schedule()               [MRO → patched Scheduler.schedule]
        └─ TriAttentionScheduler._build_signals(self, scheduler_output)
        └─ setattr(scheduler_output, "triattention_signals", signals)
      <pickle 跨进程>

    worker 子进程:
      NPUWorker.execute_model(scheduler_output) [patched wrapper]
        └─ signals = getattr(scheduler_output, "triattention_signals", None)
        └─ if signals: TriAttentionAscendWorker._ensure_triattention_runner_proxy(self)
             └─ self.model_runner = TriAttentionModelRunner(base_runner=..., config=...)
        └─ super().execute_model(scheduler_output)  [现在是 TriAttentionModelRunner]
             └─ _execute_compression_actions(...)
                  └─ base_runner.triattention_apply_compression(req_id, signal, ...)
                       └─ compute_scores_triton(...)  ← 算法核心
                       └─ compact_request_kv_in_place(...)
      <pickle 跨进程>

    engine core 子进程:
      BalanceScheduler.update_from_output(scheduler_output, model_runner_output)  [MRO → patched]
        └─ compression_events = getattr(model_runner_output, "triattention_compression_events", None)
        └─ TriAttentionScheduler._apply_compression_events(self, events)
             └─ 对每个 event:
                  ├─ effective_len_tracker.apply_compression(req_id, cache_len_after)
                  ├─ manager.req_to_blocks[req_id] = kept[:required]
                  └─ block_pool.free_blocks(reversed(removed))   [状态显式同步]
```

### 6.3 与 vLLM 0.18.0 兼容性


| 检查                               | 通过条件                                                                                                                                                        | 通过                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `Scheduler.__init__` 签名          | `(self, vllm_config, kv_cache_config, structured_output_manager, block_size, mm_registry=MULTIMODAL_REGISTRY, include_finished_set=False, log_stats=False)` | ✅ AIM 用 `*args, **kwargs` 透传     |
| `NPUWorker.__init__` 签名          | `(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False, **kwargs)`                                                          | ✅ AIM 用 `*args, **kwargs` 透传     |
| `SchedulerOutput` 跨进程 pickling   | `dataclass(slots=True)`，自定义属性也走 pickle                                                                                                                      | ✅ vllm 0.18.0 SchedulerOutput 行为 |
| `ModelRunnerOutput` 跨进程 pickling | 同上                                                                                                                                                          | ✅                                |
| `NPUWorker.model_runner` 属性可写    | 是普通 attribute                                                                                                                                               | ✅                                |


---

## 七、代码工程重构总览

### 7.1 目录结构（最终态）

```
triattention/
├── setup.py                                    ← 加 triattention_ascend entry point
├── triattention/
│   ├── vllm/                                   ← CUDA 路径（保持不变）
│   │   ├── plugin.py                           ← 新增 _is_running_on_ascend() 早退
│   │   ├── core/                               ← 算法核心（triton kernel），平台无关
│   │   └── runtime/                            ← CUDA 端 framework 嫁接
│   └── vllm_ascend/                            ← 【新】昇腾专用
│       ├── __init__.py                         ← 暴露 register_triattention_backend
│       ├── plugin.py                           ← 【新】entry point
│       └── runtime/
│           ├── __init__.py                     ← 暴露 ensure_patches_installed
│           ├── integration_monkeypatch.py      ← 【新】AIM（核心）
│           ├── scheduler_ascend.py             ← 【新】Ascend scheduler mixin
│           ├── worker_ascend.py                ← 【新】Ascend worker mixin
│           └── gpu_seq_len_patch.py            ← 【新】no-op stub
```

### 7.2 设计原则

1. **算法核心只一份**：`triattention/vllm/core/`* 和 `triattention/vllm/runtime/*` 中的算法代码（`compute_scores_triton`、`compact_request_kv_in_place`、`_build_signals` 等）**完全不动**。CUDA 路径和 Ascend 路径都从同一处导入。
2. **framework 嫁接两套**：`triattention/vllm/runtime/integration_monkeypatch.py`（RIM）针对 CUDA；`triattention/vllm_ascend/runtime/integration_monkeypatch.py`（AIM）针对 Ascend。两份 patcher **互不重复**——RIM 贴 `Worker`（CUDA），AIM 贴 `NPUWorker`（Ascend）；RIM 贴 `BlockTables`（CUDA），AIM 贴 `AscendBlockTables`（Ascend）。
3. **算法共享**：`AIM` 从 `triattention.vllm.runtime.scheduler`、`triattention.vllm.runtime.worker`、`triattention.vllm.runtime.kv_allocation_sync` 等**直接 re-import**，不改逻辑。

### 7.3 与 vLLM 上游 / vllm-ascend 的解耦边界


| 组件                                                    | 谁来维护        | 我们是否修改                    |
| ----------------------------------------------------- | ----------- | ------------------------- |
| `vllm.plugins.load_general_plugins`                   | vllm 上游     | ❌ 不动                      |
| `vllm_ascend.worker.worker.NPUWorker`                 | vllm-ascend | ❌ 不动；只在 class 对象上 setattr |
| `vllm_ascend.worker.v2.block_table.AscendBlockTables` | vllm-ascend | ❌ 不动；只在 class 对象上 setattr |
| `vllm.v1.core.sched.scheduler.Scheduler`              | vllm 上游     | ❌ 不动；只在 class 对象上 setattr |
| `vllm.v1.core.kv_cache_manager.KVCacheManager`        | vllm 上游     | ❌ 不动；只在 class 对象上 setattr |
| `vllm.v1.core.kv_cache_utils`                         | vllm 上游     | ❌ 不动；只在 module 上 setattr  |
| `vllm.v1.engine.core.EngineCore`                      | vllm 上游     | ❌ 不动；只在 class 对象上 setattr |


所有 patch 都在进程内的 class 对象 / module 对象上 setattr，**没有任何源文件改动**——彻底满足"最小侵入"。

---

## 八、文件级改动清单

### 8.1 修改的文件


| 文件                            | 改动                                                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `setup.py`                    | 在 `entry_points["vllm.general_plugins"]` 中加 `triattention_ascend = triattention.vllm_ascend.plugin:register_triattention_backend` |
| `triattention/vllm/plugin.py` | 加 `_is_running_on_ascend()` 检测；在 `register_triattention_backend` 入口先调它，True 则日志后 return（让位给 ascend entry point）                   |


### 8.2 新建的文件


| 文件                                                            | 职责                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `triattention/vllm_ascend/__init__.py`                        | 暴露 `register_triattention_backend`                                                                                                                                                                                                                                                                                                                                         |
| `triattention/vllm_ascend/plugin.py`                          | entry point 函数；包含 `_is_running_on_ascend()`、`_bridge_legacy_env_to_runtime()`、`register_triattention_backend()`                                                                                                                                                                                                                                                            |
| `triattention/vllm_ascend/runtime/__init__.py`                | 暴露 `ensure_patches_installed`、`install_ascend_integration_monkeypatches`                                                                                                                                                                                                                                                                                                   |
| `triattention/vllm_ascend/runtime/integration_monkeypatch.py` | **AIM 核心**：patch `Scheduler`（`__init__` / `schedule` / `update_from_output` + 7 个 helper）、patch `KVCacheManager.allocate_slots`、patch `EngineCore.step_with_batch_queue`、patch `NPUWorker.__init__` / `init_device` / `execute_model`、patch `AscendBlockTables.compute_slot_mappings`、relax `kv_cache_utils.check_enough_kv_cache_memory`、install `__setattr__` meta-patch |
| `triattention/vllm_ascend/runtime/scheduler_ascend.py`        | Ascend scheduler mixin：re-export platform-agnostic helpers + Ascend 特有的 chunk-cap 公式（`num_npu_blocks` 兼容）                                                                                                                                                                                                                                                                  |
| `triattention/vllm_ascend/runtime/worker_ascend.py`           | Ascend worker mixin：lazy runner proxy install factory（`TriAttentionAscendWorker._ensure_triattention_runner_proxy`）                                                                                                                                                                                                                                                        |
| `triattention/vllm_ascend/runtime/gpu_seq_len_patch.py`       | no-op stub；提供 `install_seq_len_override_patch()` 永远返回 `False`                                                                                                                                                                                                                                                                                                              |


### 8.3 没改的算法核心（共享给 CUDA 和 Ascend）


| 文件                                                 | 为什么不动                                                                                                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `triattention/vllm/core/scoring.py`                | 平台无关的 triton kernel 入口                                                                                                                                       |
| `triattention/vllm/core/compressor.py`             | 平台无关的压缩算子                                                                                                                                                    |
| `triattention/vllm/core/kernels/triton_scoring.py` | `@triton.jit` kernel 本身，NPU 端 CANN 通过 triton-ascend 兼容层执行                                                                                                    |
| `triattention/vllm/runtime/selector_hf.py`         | 选 token 索引的纯 python + torch 逻辑                                                                                                                               |
| `triattention/vllm/runtime/kv_compaction.py`       | `compact_request_kv_in_place` 用 torch tensor 切片                                                                                                              |
| `triattention/vllm/runtime/runner.py`              | `TriAttentionModelRunner` 是平台无关的 proxy——它只调 `base_runner.execute_model(...)` 和 `base_runner.triattention_apply_compression(...)`，不管 base_runner 是 GPU 还是 NPU |
| `triattention/vllm/runtime/scheduler.py`           | `TriAttentionScheduler` 的 helper methods 是平台无关的算法                                                                                                            |
| `triattention/vllm/runtime/planner.py`             | `CompressionPlanner` 是纯 python 逻辑                                                                                                                            |
| `triattention/vllm/runtime/signals.py`             | `CompressionSignal` 是 dataclass                                                                                                                              |


---

## 九、零基础部署启用教程

### 9.1 前置条件


| 依赖            | 版本               | 验证命令                                                          |
| ------------- | ---------------- | ------------------------------------------------------------- |
| Python        | ≥ 3.10           | `python --version`                                            |
| torch         | 与 vllm-ascend 兼容 | `python -c "import torch; print(torch.__version__)"`          |
| torch_npu     | 与 vllm-ascend 兼容 | `python -c "import torch_npu; print(torch_npu.__version__)"`  |
| triton-ascend | ≥ 2.0 (CANN 兼容层) | `python -c "import triton; print(triton.__version__)"`        |
| vllm          | v0.18.0          | `python -c "import vllm; print(vllm.__version__)"`            |
| vllm-ascend   | v0.18.0          | `python -c "import vllm_ascend; print(vllm_ascend.__file__)"` |
| CANN          | 与驱动配套            | `npu-smi info`                                                |


### 9.2 部署步骤

#### 步骤 1：克隆/获取本仓库

```bash
cd /Users/sunao2000/my_tri
ls
# 期望看到：
#   triattention/                  ← 本次新适配后的代码（vllm/ + vllm_ascend/）
#   vllm-releases-v0.18.0/         ← vllm 上游 0.18.0（仅代码比对用，不修改）
#   vllm-ascend-releases-v0.18.0/  ← vllm-ascend 0.18.0（仅代码比对用，不修改）
#   setup.py                       ← 已加 triattention_ascend entry point
#   ideal_WORKFLOW.md
#   TRIATTENTION_ASCEND_ANALYSIS.md  ← 本文档
```

#### 步骤 2：装 vllm 0.18.0

```bash
# 在 NPU 机器上、配套的 Python 环境里
cd /path/to/vllm-releases-v0.18.0
pip install -e . --no-build-isolation
# 验证
python -c "import vllm; print(vllm.__version__)"  # 期望: 0.18.0
```

#### 步骤 3：装 vllm-ascend 0.18.0

```bash
cd /path/to/vllm-ascend-releases-v0.18.0
pip install -e . --no-build-isolation
# 验证
python -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

#### 步骤 4：装 TriAttention（含 ascend 适配）

```bash
cd /Users/sunao2000/my_tri
pip install -e . --no-deps --force-reinstall
# 验证 entry point 注册
python -c "
import importlib.metadata as m
eps = m.entry_points().get('vllm.general_plugins', [])
for e in eps:
    print(f'  {e.name:25s} -> {e.value}')
"
# 期望看到：
#   triattention               -> triattention.vllm.plugin:register_triattention_backend
#   triattention_ascend        -> triattention.vllm_ascend.plugin:register_triattention_backend
```

#### 步骤 5：准备 stats 文件

```bash
# TriAttention 需要预计算的 Q/K 频率统计
mkdir -p ~/tri_stats
# 找一个对应模型的 stats 文件；仓库里已有：
ls /Users/sunao2000/my_tri/triattention/vllm/stats/
# 期望看到：
#   gpt_oss_120b_stats.pt
#   qwen3_32b_int4_stats.pt
# 也可以用 calibration guide 自己造：docs/calibration.md
```

#### 步骤 6：起服务

```bash
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/Users/sunao2000/my_tri/triattention/vllm/stats/qwen3_32b_int4_stats.pt
export VLLM_ASCEND_BALANCE_SCHEDULING=1   # 强制走 BalanceScheduler（vllm-ascend 自身的开关）

# 启服务
vllm serve Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code \
  --enable-prefix-caching false \
  --max-num-batched-tokens 1024
```

#### 步骤 7：观察启动日志

```bash
# 期望看到的关键日志（按时间顺序）：
[TriAttention-Ascend] plugin entry point invoked: reason=load_general_plugins ascend_detected=True patch_scheduler=True patch_worker=True
[TriAttention-Ascend] first install complete reason=load_general_plugins scheduler_class=Scheduler status={'scheduler': True, 'kv_cache_manager': True, 'npu_worker': True, 'ascend_block_tables': True, 'kv_utils': True, 'engine_core_async_step': True}
[TriAttention-Ascend] meta-patch installed on vllm.v1.core.sched.scheduler; future Scheduler rebinds will automatically receive TriAttention helper methods.
[TriAttention-Ascend] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True status=...
```

如果看到 `status` 里有 False：回到"任务一"的对应步骤查。

#### 步骤 8：观察首次 signal 触发

发送一个超过 `TRIATTN_RUNTIME_KV_BUDGET` (默认 2048) 的 prompt：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Solve this AIME problem step by step: ..."}],
    "max_tokens": 4096
  }'
```

**期望日志序列**：

```
[TriAttention-Ascend] Scheduler initialized: type=BalanceScheduler budget=2048 ...
[TriAttention-Ascend] NPUWorker initialized: type=NPUWorker budget=2048 ...
... (若干纯 decode step，无 TriAttention 日志) ...
[TriAttention-Ascend] signal triggered req=... step=N estimated_cache_len=M reason=length_threshold
[TriAttention-Ascend] lazily installed runner proxy: budget=2048 ...
[TriAttention-Ascend] update_from_output: received N events (M applied) via model_runner_output
```

**任何一个 log 没出现 = 对应链路断了**——回到"任务一"对应层排查。

### 9.3 关闭 TriAttention（baseline 对比用）

```bash
export ENABLE_TRIATTENTION=0
vllm serve Qwen/Qwen3-8B ... # 正常的 vllm-ascend 服务，无任何 patch
```

### 9.4 常见配置


| 环境变量                                                | 默认         | 用途                                |
| --------------------------------------------------- | ---------- | --------------------------------- |
| `ENABLE_TRIATTENTION`                               | `true`     | 总开关；`0` 关闭                        |
| `TRIATTN_RUNTIME_KV_BUDGET`                         | `2048`     | 每个 request 保留的 KV 长度上限            |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH`                     | `128`      | 每 N 个新 token 检查一次是否压缩             |
| `TRIATTN_RUNTIME_WINDOW_SIZE`                       | `128`      | 永远保留的最近 N 个 token                 |
| `TRIATTN_RUNTIME_PRUNING_MODE`                      | `per_head` | `per_head` 或 `per_layer_per_head` |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH`                 | (无)        | 预计算统计文件路径                         |
| `TRIATTN_RUNTIME_PROTECT_PREFILL`                   | `false`    | 是否保护 prompt 头不被压缩                 |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION` | `true`     | 是否启用 in-place KV 压缩               |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM` | `true`     | 是否回收压缩后多余的物理 block                |
| `TRIATTENTION_QUIET`                                | `0`        | 是否静默一些 INFO 日志                    |


---

## 十、验证矩阵与失败模式对照表

### 10.1 验证项


| 验证                        | 命令                                                                                                                                                                               | 期望                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Entry point               | `python -c "import importlib.metadata as m; print([(e.name, e.value) for e in m.entry_points().get('vllm.general_plugins', [])])"`                                               | 两行                                                                           |
| Ascend 平台                 | `python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"`                                                                                | 含 `Ascend` / `NPU`                                                           |
| NPUWorker patched         | `python -c "from vllm_ascend.worker.worker import NPUWorker; print(getattr(NPUWorker, '_ensure_triattention_runner_proxy', None))"`                                              | `<staticmethod>`                                                             |
| Scheduler patched         | `python -c "import vllm.v1.core.sched.scheduler as s; print(getattr(s.Scheduler.__init__, '_triattention_patched', False))"`                                                     | `True`                                                                       |
| BalanceScheduler 继承       | `python -c "import vllm.v1.core.sched.scheduler as s; import vllm_ascend.patch.platform.patch_balance_schedule; print(s.Scheduler.__name__); print(s.Scheduler._build_signals)"` | `BalanceScheduler`、callable                                                  |
| AscendBlockTables patched | `python -c "from vllm_ascend.worker.v2.block_table import AscendBlockTables; print(getattr(AscendBlockTables.compute_slot_mappings, '_triattention_patched', False))"`           | `True`                                                                       |
| 启动期日志                     | `vllm serve ...` 前 100 行                                                                                                                                                         | 含 6 条 `[TriAttention-Ascend]` 日志                                             |
| 首次 signal 触发              | 超过 2048 tokens 的请求                                                                                                                                                               | 出现 `signal triggered` 日志                                                     |
| 压缩后                       | 长 prompt 推理                                                                                                                                                                      | 出现 `lazily installed runner proxy` + `update_from_output: received N events` |
| 内存释放                      | `npu-smi info` 内存曲线                                                                                                                                                              | 压缩后 KV 占用下降、free blocks 上升                                                   |


### 10.2 失败模式 → 排查步骤映射


| 失败现象                                                                                                | 排查步骤                                                                                                |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 启动日志里完全没有 `[TriAttention-Ascend]` 字样                                                                | 步骤 1（entry point）、步骤 2（ascend 平台）                                                                   |
| 启动日志里有 `[TriAttention-Ascend] plugin entry point invoked` 但 `first install complete` status 全 False | 步骤 3（AIM 异常）——通常是某个 import 失败，向上翻日志找 `ERROR` 行                                                      |
| 启动日志完整但第一个 signal step 始终不出现                                                                        | 步骤 7（算法核心没被调）——调小 `TRIATTN_RUNTIME_KV_BUDGET` 到 256 试试                                              |
| `signal triggered` 出现但 `lazily installed runner proxy` 没出现                                          | 步骤 4（NPUWorker 没 patch 好）——查 `_PATCHED_WORKER_ACTIVE` 状态                                            |
| `lazily installed runner proxy` 出现但 `compute_scores_triton` 仍没被调                                    | 算法链路某处 try/except 吞了异常——把 `TriAttentionModelRunner._execute_compression_actions` 里的 try/except 临时关掉 |
| `update_from_output: received N events` 出现但 `M applied` 是 0                                         | 物理 KV 压缩动作没真正成功——看 stats 文件的 `num_layers` / `num_kv_heads` 是不是当前模型的                                 |
| 内存没下降                                                                                               | `enable_experimental_block_reclaim=0` —— block 物理回收没开                                               |


---

## 附：适配前后调用链对比

### 适配前（broken）

```
vllm.plugins.load_general_plugins
  └─ triattention  (entry point) [唯一]
       └─ install_vllm_integration_monkeypatches
            └─ Worker = vllm.v1.worker.gpu_worker.Worker   ← 错类
                 ↑ NPUWorker 不是 Worker 的子类
                 ↑ 装不上去
                 ↑ scheduler_output.triattention_signals 永远无人接
                 ↑ TriAttentionModelRunner 永远没装
                 ↑ compute_scores_triton 永远不被调
                 ↑ 用户看到的：日志 "activated"，但断点调试完全失效
```

### 适配后（fixed）

```
vllm.plugins.load_general_plugins
  ├─ triattention          (entry point)  [_is_running_on_ascend() == True → 早退]
  └─ triattention_ascend   (entry point)  [AIM]
       └─ ensure_patches_installed
            ├─ Scheduler (upstream)            [__init__ / schedule / update_from_output + 7 helper]
            ├─ KVCacheManager (upstream)        [allocate_slots]
            ├─ EngineCore (upstream)            [step_with_batch_queue]
            ├─ NPUWorker (vllm_ascend)          [__init__ / init_device / execute_model + _ensure_triattention_runner_proxy]
            ├─ AscendBlockTables (vllm_ascend)  [compute_slot_mappings passthrough]
            ├─ kv_cache_utils (vllm upstream)   [check_enough_kv_cache_memory 放松]
            └─ meta-patch on scheduler module   [Scheduler rebind 时自动 re-attach helpers]

每个 step:
  BalanceScheduler.schedule()                   [MRO → patched Scheduler.schedule]
    └─ setattr(scheduler_output, "triattention_signals", signals)

  NPUWorker.execute_model(scheduler_output)     [patched wrapper]
    └─ if signals: install TriAttentionModelRunner proxy   [lazy]
    └─ proxy.execute_model(scheduler_output)
         └─ base_runner.triattention_apply_compression(req_id, signal, ...)
              └─ _select_keep_indices(...)       [algorithm core]
                   └─ compute_scores_triton(...)  [Triton kernel — 你打了一晚上断点就在这里]
              └─ compact_request_kv_in_place(...)
    └─ output = proxy.execute_model(...)
    └─ setattr(output, "triattention_compression_events", events)

  BalanceScheduler.update_from_output(...)     [MRO → patched Scheduler.update_from_output]
    └─ _apply_compression_events(events)
         └─ block_pool.free_blocks(reversed(removed))   [状态显式同步]
```

### 一句话总结

> **适配前**：vLLM 上游 `Worker` 被 patch，但 `NPUWorker` 没被 patch——整个 ascend 引擎根本不知道 TriAttention 存在。**适配后**：vllm-ascend 专用 `NPUWorker` / `BalanceScheduler` / `AscendBlockTables` 全部通过 `setattr` 在运行时被注入 TriAttention 逻辑，算法核心代码 100% 复用 CUDA 路径，不修改任何 vllm / vllm-ascend 源文件。

