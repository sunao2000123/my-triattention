# TriAttention-Ascend 断点 workflow（基于 vllm-ascend 0.18.0）

> 配 RUN.md 一起看：WORKFLOW.md 讲"代码怎么走"，RUN.md 讲"怎么
> 实际跑起来看到 triton 算子被调用" + "哪些是 verified、哪些是
> 假设性的"。

## 文件层级（理解整体结构）

```
triattention/
├── vllm/                                    ← 平台无关的算法核心 + 嫁接到 vllm 上游
│   ├── core/                                ←   纯算子（triton kernel），与 vllm 框架解耦
│   │   ├── scoring.py                       ←     compute_scores_triton
│   │   └── kernels/triton_scoring.py        ←     @triton.jit triattention_scoring_kernel
│   └── runtime/                             ←   框架嫁接层 + 算法包装
│       ├── integration_monkeypatch.py       ←     (RIM)  嫁接到 vllm 上游
│       ├── runner.py                        ←     TriAttentionModelRunner
│       ├── hook_impl.py                     ←     make_runner_compression_hook
│       ├── scheduler.py                     ←     planner, _apply_compression_events, etc.
│       ├── selector_hf.py                   ←     _select_keep_indices
│       ├── kv_compaction.py                 ←     compact_request_kv_in_place
│       ├── runner_output_bridge.py          ←     attach_execute_model_compression_events
│       ├── hook_group_pipeline.py           ←     run_group_compaction_pipeline
│       ├── executor.py                      ←     CompressionExecutor
│       ├── layout_engine.py                 ←     execute_group_compaction
│       ├── kv_allocation_sync.py            ←     effective_num_computed helpers
│       └── ... (40+ 个文件)
│
└── vllm_ascend/                              ← 嫁接到 vllm-ascend 的薄壳
    ├── plugin.py                             ←   triattention_ascend entry-point 函数
    └── runtime/                             ←   全部 thin wrapper / 嫁接代码
        ├── integration_monkeypatch.py       ←     (AIM)  嫁接到 vllm-ascend
        ├── scheduler_ascend.py               ←     8 个 helper method 装到 BalanceScheduler
        ├── worker_ascend.py                  ←     lazy runner proxy install
        ├── gpu_seq_len_patch.py              ←     input-prep hook（ascend 端是 no-op）
        ├── kv_allocation_sync.py             ←     re-export platform-agnostic
        ├── effective_len_tracker.py          ←     re-export
        ├── planner.py / signals.py / etc.    ←     re-export
        └── config.py                         ←     re-export
```

**关键命名澄清**：`vllm/core/` 不是"vllm 上游"，是"算法核心"——它和 vllm
框架同名纯属历史命名。`vllm/core/` 下的代码**不 import 任何 vllm 框架
代码**——就是纯 Triton 算子。如果去掉 `vllm/core/`，`vllm/runtime/`
下的 selector 也就空了，**整个 pipeline 断**。

## 两个 integration_monkeypatch 的分工（重点）

`triattention/vllm/runtime/integration_monkeypatch.py`（**RIM**）和
`triattention/vllm_ascend/runtime/integration_monkeypatch.py`（**AIM**）
**不是同层的东西**，是**分两路**：


|                 | RIM（CUDA 端）                                                          | AIM（Ascend 端）                                                                                            |
| --------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 目标 class        | `vllm.v1.worker.gpu_worker.Worker`                                   | `vllm_ascend.worker.worker.NPUWorker`                                                                    |
| 目标 Scheduler    | `vllm.v1.core.sched.scheduler.Scheduler`（上游原版）                       | 同一个模块符号——但 ascend 跑时**已被 vllm-ascend 替换成 `BalanceScheduler`**（subclass）                                  |
| 目标 block tables | `vllm.v1.worker.gpu.block_table.BlockTables`                         | `vllm_ascend.worker.v2.block_table.AscendBlockTables`（vllm-ascend 自己 subclass，重写了 compute_slot_mappings） |
| 入口              | `triattention.vllm.plugin:register_triattention_backend`             | `triattention.vllm_ascend.plugin:register_triattention_backend`                                          |
| 走 ascend 时      | **不跑**（`_is_running_on_ascend()` 返回 True → return，**让位**给 ascend 入口） | **跑**                                                                                                    |


**AIM 与 RIM 的 cross-import**（AIM 重用 RIM 的算法核心，不重写）：

```python
# triattention/vllm_ascend/runtime/worker_ascend.py
from triattention.vllm.runtime.hook_impl import install_runner_compression_hook
from triattention.vllm.runtime.runner import TriAttentionModelRunner

# triattention/vllm_ascend/runtime/scheduler_ascend.py
from triattention.vllm.runtime.scheduler import (
    _free_reclaimed_blocks, _resolve_full_prefill_len_from_request_like,
)
from triattention.vllm.runtime.request_key_compat import iter_scheduled_token_items

# 等等
```

**所以"两个 integration_monkeypatch 在 ascend 系统的协奏"是**：

- **AIM 跑**（被 `triattention_ascend` entry point 触发），它做三件事：
  1. 把 `Scheduler` / `NPUWorker` / `AscendBlockTables` 等 vllm-ascend 的 class 给 patch 上
  2. 把 8 个 helper method attach 到 `Scheduler` class（这些 helper method 的实现就在 AIM 的 `scheduler_ascend.py` 里，但它们的**算法核心**调的是从 `triattention.vllm.runtime` import 来的 planner / request_key_compat）
  3. 装 NPUWorker 上 `_ensure_triattention_runner_proxy`（这个 proxy 装上以后会**直接调用** `triattention.vllm.runtime.runner.TriAttentionModelRunner`）
- **RIM 也在同一进程跑**（被 `triattention` entry point 触发，但因为 `_is_running_on_ascend()` 返回 True 它**提前 return 了**），所以**它在 ascend 上什么都不做**。RIM 在 ascend 上的**唯一作用**就是"让位"——告诉 vllm "有 ascend 入口在后面，我不做事"。
- 当 NPUWorker.execute_model 被调时（已经 patch 过了），AIM 的 wrapper 触发 `_ensure_triattention_runner_proxy`，**这步**会从 `triattention.vllm.runtime.runner` import `TriAttentionModelRunner`——**这是 RIM 同 package 下的 class**。然后 TriAttentionModelRunner 自己按阶段 E 走：调 hook_impl、selector、kv_compaction **全部从 `triattention.vllm.runtime.*` 来**。

**为什么不合并成一个文件**：vllm-ascend 的 class 路径、模块路径、patch 时机都和 vllm 上游不一样。合并后 `if is_ascend: <patch NPUWorker> else: <patch Worker>` 每个点都得判断——既丑又容易出 bug。分开后 **CUDA 用户只看 `vllm/`**、**ascend 用户只看 `vllm_ascend/`**、**算法核心共享 `vllm/runtime/`**。

---

# 阶段 A：装 entry point（`vllm serve` 启动之前）

```
[你] pip install -e /Users/sunao2000/tri
     pip install -e /Users/sunao2000/tri/vllm-ascend-releases-v0.18.0
     # 两个都装到同一个 Python env（>=3.10）
     # 第一条命令在 setup.py:38-46 把这两个 entry point 写进 dist-info
```

```python
# setup.py:38-46
entry_points={
    "vllm.general_plugins": [
        "triattention       = triattention.vllm.plugin:register_triattention_backend",
        "triattention_ascend = triattention.vllm_ascend.plugin:register_triattention_backend",
    ],
}
```

**唯一目的**：让 `importlib.metadata.entry_points()` 能找到 `triattention_ascend`。

**验证**：

```bash
python -c "import importlib.metadata as m; print([(e.name, e.value) for e in m.entry_points().get('vllm.general_plugins', [])])"
# 必须看到两行
```

**RIM 在这里"让位"的代码**（`triattention/vllm/plugin.py:89`）：

```python
if _is_running_on_ascend():
    logger.info("[TriAttention] Detected vllm-ascend platform; ... Skipping CUDA plugin ...")
    return                                  # ← RIM 走 ascend 时提前 return
```

---

# 阶段 B：`vllm serve ...` 进程起来

vllm 0.18.0 的 main 进程第一件事是导入 plugins（`vllm/plugins/__init__.py:69 load_general_plugins`）：

```
vllm.plugins.load_general_plugins()
    ↓
load_plugins_by_group(group="vllm.general_plugins")
    ↓ 读取 importlib.metadata，返回 dict{name → 函数}
    ↓ 按 entry point name 排序（dict 是 insertion-ordered）
    ↓
for func in plugins.values():
    func()
```

`plugins.values()` 的 key 是 entry point name，**遍历顺序是 `triattention` 在前、`triattention_ascend` 在后**。`triattention` 那路跑，探测到 ascend → return；`triattention_ascend` 那路跑，**这是 ascend 路径的入口**：

```
triattention_ascend.register_triattention_backend()
    ├─ if not _is_running_on_ascend(): return        # 再确认
    ├─ if not ENABLE_TRIATTENTION: return
    ├─ ensure_patches_installed(...)
    │   └─ 见阶段 B.1
    └─ log "[TriAttention-Ascend] Runtime (V2) plugin activated: ..."
```

## 阶段 B.1：`ensure_patches_installed()`（AIM 的工作）

```
ensure_patches_installed(patch_scheduler=True, patch_worker=True, reason="load_general_plugins")
    ├─ import vllm.v1.core.sched.scheduler       # Scheduler 符号 = 上游 Scheduler（adapt_patch 还没跑）
    ├─ import vllm_ascend.worker.worker          # NPUWorker
    ├─ import vllm_ascend.worker.v2.block_table  # AscendBlockTables
    ├─ import vllm.v1.core.kv_cache_manager      # KVCacheManager
    ├─ import vllm.v1.core                       # kv_cache_utils
    │
    ├─ _patch_upstream_scheduler_class(Scheduler, helper_methods={...})
    │   ├─ Scheduler.__init__                  = _patched_scheduler_init            [patch 1/3]
    │   ├─ Scheduler.schedule                   = _patched_scheduler_schedule        [patch 2/3]
    │   ├─ Scheduler.update_from_output         = _patched_scheduler_update_from_output [patch 3/3]
    │   └─ attach 7 个 helper method:           _resolve_prefill_len, _compute_length_threshold,
    │                                            _sync_prefill_lens, _has_active_effective_len_overrides,
    │                                            _build_signals, _sync_effective_kv_offsets_before_schedule,
    │                                            _apply_compression_events
    │
    ├─ KVCacheManager.allocate_slots            = _patched_kv_cache_allocate_slots   [patch 4]
    │
    ├─ NPUWorker.__init__                       = _patched_npu_worker_init            [patch 5]
    ├─ NPUWorker.execute_model                  = _patched_npu_worker_execute_model  [patch 6]
    ├─ NPUWorker._ensure_triattention_runner_proxy  = TriAttentionAscendWorker._ensure_triattention_runner_proxy  [helper install]
    ├─ AscendBlockTables.compute_slot_mappings  = _patched_compute_slot_mappings     [patch 7]
    │
    ├─ kv_cache_utils._check_enough_kv_cache_memory    = _relaxed_legacy_check        [patch 8]
    ├─ kv_cache_utils.check_enough_kv_cache_memory      = _relaxed_public_check        [patch 9]
    │
    └─ _install_module_meta_patch()             # 见阶段 B.2
        ↓
        log "[TriAttention-Ascend] first install complete reason=load_general_plugins ..."
```

## 阶段 B.2：meta-patch（防御性兜底）

`_install_module_meta_patch()` 在 `vllm.v1.core.sched.scheduler` 模块
的 `__setattr__` 上装代理——**任何后续的 `Scheduler = X` rebind 都会
自动 fire 重新装 helper methods**（不重 patch `__init__` / `schedule`，
因为 patch 在 upstream class 对象上，subclass 通过 MRO 继承）。

```
_install_module_meta_patch()
    ├─ sched_mod.__class__ = type(sched_mod.__class__, ..., {"__setattr__": _meta_setattr})
    └─ log "[TriAttention-Ascend] meta-patch installed on vllm.v1.core.sched.scheduler; ..."
```

```
_meta_setattr(mod, name, value)
    ├─ original_setattr(mod, name, value)   # 先真的 rebind
    └─ if name == "Scheduler" and isinstance(value, type):
        _attach_helpers_only(value, helper_methods=...)   # 只挂 helper
        log "[TriAttention-Ascend] meta-patch: Scheduler rebound to %s; ..."
```

## 阶段 B.3：vllm-ascend 自己的 platform patch 触发

`current_platform.pre_register_and_update(parser)` 调
`NPUPlatform.pre_register_and_update()` → `adapt_patch(is_global_patch=True)`
→ 加载 `vllm_ascend/patch/platform/patch_balance_schedule.py`：

```
patch_balance_schedule.py:705
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

这一步只**改模块里 `Scheduler` 这个名字的指向**，**不动 upstream class
对象上的 `__init__` 属性**。`BalanceScheduler` 是 upstream Scheduler
的 subclass，通过 Python MRO 继承 `_patched_scheduler_init` 等。

```
vllm.v1.core.sched.scheduler.Scheduler
        ↓ 名字指向换了
BalanceScheduler (subclass of Scheduler)
        ↓ MRO 继承
Scheduler (upstream, 我们的 __init__ 还在上面)
```

**没有递归**——`_patched_scheduler_init` 内部调 `_ORIG_SCHED_INIT`（真
正的 upstream init），走 super().**init**() 链时已经是 BalanceScheduler
调 super（即 upstream Scheduler.**init** = 我们 patch 过的），**这个
就是 super chain 的正常行为**。

---

# 阶段 C：engine core 子进程——Scheduler 实例化

`vllm/v1/engine/core.py:126` 在 engine core 子进程里：

```python
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
# → 此时 Scheduler 符号 = BalanceScheduler（vllm-ascend 刚 rebind 过）
self.scheduler = Scheduler(vllm_config, kv_cache_config, structured_output_manager, block_size, ...)
```

```
BalanceScheduler()                              [engine core 调]
    ↓ BalanceScheduler.__init__ (MRO 找到的不是 patch 版的,
    ↓   因为 BalanceScheduler 自己 override 了 __init__)
    ├─ super().__init__(vllm_config, ...)       [走 MRO 到 upstream Scheduler.__init__]
    │   └─ _patched_scheduler_init (我们 patch 的)         [MRO 找到]
    │       ├─ _ORIG_SCHED_INIT(self, ...)                  [真正的 upstream init]
    │       │   └─ 装 vllm 的 state: self.requests, self.kv_cache_manager, ...
    │       │   └─ 装 self.max_num_scheduled_tokens, self.block_size, ...
    │       └─ _attach_triattention_scheduler_state
    │           └─ 装 self.triattention_config (from env: kv_budget=2048, divide_length=128, ...)
    │           └─ 装 self._planner = CompressionPlanner(cfg)
    │           └─ 装 self._effective_len_tracker = EffectiveCacheLenTracker()
    │           └─ 装 self._prefill_lens, self._length_threshold_cache, self._triattention_step
    │           └─ log "[TriAttention-Ascend] Scheduler initialized: type=BalanceScheduler budget=2048 ..."
    └─ self.balance_queue = [torch.tensor([0], ...) for _ in range(dp_size)]   [vllm-ascend 自己的]
```

**至此 `self.scheduler` 是 `BalanceScheduler` 实例**——既继承了 vllm-ascend
的 `balance_queue`，又挂上了 TriAttention 的 signal/decision state。**两边状态都齐了**。

---

# 阶段 D：scheduler 端每一步——决策

`_patched_scheduler_schedule`（installed on upstream Scheduler class，
BalanceScheduler 通过 MRO 继承）：

```
BalanceScheduler().schedule()                    [engine core 调]
    ↓ MRO 找到
    ↓ _patched_scheduler_schedule
    ├─ TriAttentionAscendScheduler._sync_effective_kv_offsets_before_schedule(self)
    │
    ├─ orig_max_scheduled = None
    ├─ if not cfg.disable_compression:
    │     max_chunk = TriAttentionAscendScheduler._compute_max_chunk_for_compression(self)
    │     if max_chunk < self.max_num_scheduled_tokens:
    │       orig_max_scheduled = self.max_num_scheduled_tokens
    │       self.max_num_scheduled_tokens = max_chunk
    │
    ├─ _ORIG_SCHED_SCHEDULE(self)                [真正的 upstream schedule（或 BalanceScheduler 的 schedule，super 调 upstream）]
    │
    ├─ if orig_max_scheduled is not None:
    │     self.max_num_scheduled_tokens = orig_max_scheduled
    │
    ├─ self._triattention_step += 1
    ├─ TriAttentionAscendScheduler._sync_prefill_lens(self, scheduler_output)
    │
    ├─ if (disable_compression AND not enable_kv_usage_trigger AND no effective_len_overrides):
    │     signals = {}                           [空 dict = 这个 step 不压]
    │ else:
    │     signals = TriAttentionAscendScheduler._build_signals(self, scheduler_output)  ← 算法入口
    │       └─ 对每个 scheduled req:
    │             ├─ effective_base_len = effective_len_tracker.observe_num_computed(req_id, ...)
    │             │                    或 request.num_computed_tokens（如果没 override）
    │             ├─ estimated_cache_len = effective_base_len + scheduled_tokens
    │             ├─ threshold = self._length_threshold_cache[req_id] or self._compute_length_threshold(prefill_len)
    │             │              (= kv_budget + divide_length [+ prefill_len])
    │             ├─ if estimated_cache_len < threshold: continue
    │             └─ signal = self._planner.build_signal(
    │                   req_id, estimated_cache_len, prefill_len,
    │                   step=self._triattention_step, kv_usage, scheduled_tokens
    │                 )
    │                 ├─ 比较 estimated_cache_len >= length_threshold
    │                 ├─ 或 kv_usage >= trigger
    │                 └─ return CompressionSignal(should_compress, reason, ...)
    │             └─ if signal.should_compress or has_override:
    │                   if signal.should_compress:
    │                     log "TriAttention-Ascend signal triggered req=... step=... estimated_cache_len=... reason=length_threshold"
    │                   signals[req_id] = signal
    │
    ├─ setattr(scheduler_output, "triattention_step", self._triattention_step)              [跨进程桥 1/4]
    ├─ setattr(scheduler_output, "triattention_signals", signals)                              [跨进程桥 2/4]
    └─ return scheduler_output
    ↓ pickle 跨进程
[worker 子进程]
```

scheduler_output 上的 `triattention_step` 和 `triattention_signals` 是
**vllm 看不到的属性**——vllm 的 pickle 协议自动序列化它们。

---

# 阶段 E：worker 端每一步——执行

## 阶段 E.1：lazy runner proxy install

`_patched_npu_worker_execute_model`（installed on `NPUWorker.execute_model`）：

```
NPUWorker.execute_model(scheduler_output)       [engine core 调]
    ↓ MRO 找到 wrapper
    ↓ _patched_npu_worker_execute_model
    ├─ signals = getattr(scheduler_output, "triattention_signals", None)
    │
    ├─ if signals:                                [不是每一步都装，只装一次]
    │     TriAttentionAscendWorker._ensure_triattention_runner_proxy(self)
    │       ├─ if self._triattention_runner_proxy_installed: return    [已装过 → no-op]
    │       ├─ if isinstance(self.model_runner, TriAttentionModelRunner): return
    │       ├─ config = self._triattention_runtime_config
    │       ├─ base_runner = self.model_runner    [NPUModelRunner]
    │       ├─ install_runner_compression_hook(base_runner, config)   ← 这个 hook 来自 triattention.vllm.runtime.hook_impl
    │       │   └─ setattr(base_runner, "triattention_apply_compression", make_runner_compression_hook(...))
    │       ├─ self.model_runner = TriAttentionModelRunner(base_runner=base_runner, config=config)
    │       │                    ↑ 来自 triattention.vllm.runtime.runner
    │       └─ log "TriAttention-Ascend lazily installed runner proxy: budget=... stats_path=..."
    │
    └─ _ORIG_NPU_WORKER_EXECUTE_MODEL(self, scheduler_output)
        └─ self.model_runner.execute_model(...)   [此时 model_runner 已经是 TriAttentionModelRunner]
            ↓ 见阶段 E.2
```

**为什么是 lazy**：vllm-ascend 的 `NPUModelRunner` 在第一次 forward 之前要
做 ACL graph 编译 / 权重加载 / device warmup——**这些不能被 wrapper 拦**。
所以在第一步纯 decode 之前不动 `model_runner`，等 scheduler 发过来的
scheduler_output 里**第一次**带 `triattention_signals` 时再换上去。

## 阶段 E.2：TriAttentionModelRunner.execute_model

`TriAttentionModelRunner.execute_model`（`triattention/vllm/runtime/runner.py:366`，**ascend 端直接复用**）：

```
TriAttentionModelRunner.execute_model(scheduler_output, ...)
    ├─ self._register_new_requests(scheduler_output)                        [新 req 的 state 入库]
    ├─ self._cleanup_finished_requests(scheduler_output)                    [清掉 finished 的 state]
    ├─ self._mark_preemptions(scheduler_output)                             [记录被抢占的 req]
    ├─ self._mark_resumed(scheduler_output)                                 [记录恢复的 req]
    ├─ signals = self._consume_signals(scheduler_output)                    [把跨进程桥上的 signals 拿出来]
    │     └─ getattr(scheduler_output, "triattention_signals", {})           [若空 → 空 dict]
    ├─ signals = self._supplement_worker_self_triggers(scheduler_output, signals)  [worker 端自触发补判]
    │
    ├─ self._execute_compression_actions(scheduler_output, signals)         ← 算法执行入口
    │   └─ execute_runner_compression_actions(
    │         executor=self.executor,        # CompressionExecutor(base_runner)
    │         state_store=...,
    │         scheduler_output=scheduler_output,
    │         signals=signals,
    │         ...,
    │     )
    │     └─ for 每个 signal.should_compress == True 的 (req_id, signal):
    │           base_runner.triattention_apply_compression(req_id, signal, scheduler_output)
    │             ↓
    │           _hook(req_id, signal, scheduler_output)                      [hook_impl.py:59]
    │             ├─ req_ctx = resolve_hook_request_context(...)
    │             ├─ runtime_ctx = build_hook_runtime_context(...)
    │             ├─ compaction_inputs = resolve_hook_compaction_inputs(base_runner, ...)
    │             └─ run_group_compaction_pipeline(
    │                   req_id, signal, config, ...,
    │                   select_keep_indices=_select_keep_indices,
    │                   select_keep_indices_for_group=_select_keep_indices_for_group_per_head,
    │                   shared_compact_fn=compact_request_kv_in_place,
    │                   per_head_compact_fn=compact_request_kv_in_place_per_head,
    │                   gather_dense_fn=gather_request_k_dense,
    │               )
    │                 └─ prepare_group_layer_compactions(...)              [hook_group_pipeline.py:81]
    │                       └─ _select_keep_indices(...)                    [selector_hf.py:679]
    │                             └─ _select_keep_indices_paged_streaming(...)
    │                                   └─ _compute_layer_scores(...)        [selector_hf.py]
    │                                         └─ _compute_layer_scores_raw(...)
    │                                               └─ from triattention.vllm.core.scoring import compute_scores_triton
    │                                                     ↑ 关键 cross-module 跳转
    │                                               └─ compute_scores_triton(...)
    │                                                     └─ @triton.jit triattention_scoring_kernel  ← 你打了一晚上断点就在这里
    │                                                           [triattention/vllm/core/kernels/triton_scoring.py:160]
    │                                                           [NPU 上并行算每个 token 的 score]
    │                             [回到 _select_keep_indices:726-732]
    │                             topk = torch.topk(scores, k=budget_total, dim=-1, largest=True)
    │                             keep_per_head = torch.sort(topk.indices, dim=-1).values.contiguous()
    │                             return {"mode": "per_head", "indices": keep_per_head}
    │                 [回到 run_group_compaction_pipeline:106-117]
    │                 └─ execute_group_compaction(...)                       [layout_engine.py]
    │                       └─ compact_request_kv_in_place(...)              [kv_compaction.py:355]
    │                             ├─ perm_tensor = cat([keep, dropped])         [重排]
    │                             ├─ gathered = kv_cache[src_blocks, src_off]   [读 K/V]
    │                             └─ kv_cache[dst_blocks, dst_off] = gathered   [写回 in-place]
    │
    ├─ self._apply_worker_block_reclaim_events()                            [把 step 内的事件应用到 worker 端 block tables]
    ├─ self._patch_scheduler_output_for_compressed_reqs(scheduler_output)   [scheduler_output 内的 block_ids 列表改写]
    │
    ├─ need_effective_overrides = self._needs_effective_input_overrides(...)
    ├─ self._ensure_runtime_input_patch_if_needed(need_effective_overrides)  [按需装 vllm 端 input patch]
    │     └─ 注：ascend 端的 gpu_seq_len_patch 是 no-op（详见 vllm_ascend/runtime/gpu_seq_len_patch.py
    │         头部 docstring）。物理 KV 已被压缩到同一 prefix，attention 输出仍正确。
    │
    ├─ output = execute_base_model_with_effective_overrides(                [真正的 NPU forward]
    │       base_runner=self._base_runner,
    │       state_store=self.state_store,
    │       scheduler_output=scheduler_output,
    │       intermediate_tensors=...,
    │       use_effective_overrides=need_effective_overrides,
    │   )
    │   └─ self._base_runner.execute_model(...)                             [NPUModelRunner 的 forward]
    │
    └─ output, self._pending_compression_events = attach_execute_model_compression_events(
          output=output,
          pending_events=self._pending_compression_events,
          scheduler_output=scheduler_output,
      )
        └─ setattr(output, "triattention_compression_events", events)       [跨进程桥 3/4]
    return output
    ↓ pickle 跨进程
[engine core 子进程]
```

---

# 阶段 F：scheduler 端每一步——物理回收

`update_from_output`（engine core 收到 `ModelRunnerOutput` 后调）：

```
BalanceScheduler.update_from_output(self, scheduler_output, model_runner_output)
    ↓ MRO 找到
    ↓ _patched_scheduler_update_from_output
    ├─ _ORIG_SCHED_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)  [原版 BalanceScheduler.update_from_output]
    │
    ├─ compression_events = getattr(model_runner_output, "triattention_compression_events", None)
    │   if not compression_events:
    │     compression_events = getattr(scheduler_output, "triattention_compression_events", None)   [跨进程桥 4/4, fallback]
    │
    ├─ if compression_events:
    │     TriAttentionAscendScheduler._apply_compression_events(self, compression_events)   [scheduler_ascend.py:237]
    │       └─ 对每条 event:
    │             ├─ if event["status"] != "applied": continue
    │             ├─ self._effective_len_tracker.apply_compression(req_id, cache_len_after, ...)
    │             │   └─ 记录 logical length 从 num_computed_tokens 改为 cache_len_after
    │             ├─ if not cfg.enable_experimental_block_reclaim: continue
    │             ├─ required_blocks = ceil(cache_len_after / block_size)
    │             ├─ 对每个 kv group (manager in single_type_managers):
    │             │     req_blocks = manager.req_to_blocks[req_id]
    │             │     if len(req_blocks) <= required_blocks: continue
    │             │     kept = req_blocks[:required_blocks]
    │             │     removed = req_blocks[required_blocks:]
    │             │     manager.req_to_blocks[req_id] = kept          [改写 req_to_blocks]
    │             │     manager.num_cached_block[req_id] = min(..., len(kept))   [如果有]
    │             │     _free_reclaimed_blocks(manager, removed)       [真释放]
    │             │       └─ block_pool.free_blocks(reversed(removed))   [→ block_pool 真的还 block]
    │             └─ update_request_effective_kv_offset(request=req, cache_len_after=cache_len_after)
    │                 └─ 通知 kv_allocation_sync：下一次 allocate_slots 时 request 的
    │                     effective_num_computed 已经变了，allocator 应按 post-compression
    │                     长度分配而不是按 logical num_computed 分配
    │
    ├─ for req_id in scheduler_output.finished_req_ids:
    │     self._prefill_lens.pop(req_id, None)
    │     self._effective_len_tracker.remove_request(req_id)
    │
    └─ return _ORIG_SCHED_UPDATE_FROM_OUTPUT 的结果（outputs dict 给 engine 串行回去）
```

**注意**：ascend 端 `_apply_compression_events` **多了一段**原生命令链没
提到的事——`manager.req_to_blocks[req_id] = kept` 这步同时改写了
**scheduler 端的 block table**（不只是 `block_pool.free_blocks`）。
`block_reclaim` 字典里的 `block_ids_before` / `block_ids_after` 列表
就是给这一步做 prefix-verify 用的（防止 block table 和物理 KV 错位）。

---

# 阶段 G：跨进程桥的属性清单（vllm 看不到的 4 个 attribute）


| 方向                 | 挂载点                                     | 属性名                               | 数据                                | 消费者                                     |
| ------------------ | --------------------------------------- | --------------------------------- | --------------------------------- | --------------------------------------- |
| scheduler → worker | `SchedulerOutput`（每次 schedule 后）        | `triattention_step`               | int                               | worker 端 perf/profile                   |
| scheduler → worker | `SchedulerOutput`                       | `triattention_signals`            | `dict[req_id, CompressionSignal]` | worker 端 `_consume_signals`             |
| worker → scheduler | `ModelRunnerOutput`（每次 execute_model 后） | `triattention_compression_events` | `list[dict]`                      | scheduler 端 `_apply_compression_events` |
| worker → scheduler | `SchedulerOutput`（fallback）             | `triattention_compression_events` | 同上                                | 同上（若 runner_output 上没附）                 |


---

# 阶段 H：理想日志序列

启 vllm 之后**第一次**出现 signal 触发，**这个顺序**的日志应该全部出现：

```
# 主进程：plugin discovery
[TriAttention-Ascend] plugin entry point invoked: reason=load_general_plugins ascend_detected=True ...
[TriAttention-Ascend] meta-patch installed on vllm.v1.core.sched.scheduler; ...
[TriAttention-Ascend] first install complete reason=load_general_plugins scheduler_class=Scheduler status={'scheduler': True, 'kv_cache_manager': True, 'npu_worker': True, 'ascend_block_tables': True, 'kv_utils': True}
[TriAttention-Ascend] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True status=...

# 主进程：current_platform resolve 触发 adapt_patch
[TriAttention-Ascend] meta-patch: Scheduler rebound to BalanceScheduler; re-attached TriAttention helper methods on the new class (__init__/schedule/update_from_output inherited via MRO)

# engine core 子进程：Scheduler 实例化
[TriAttention-Ascend] Scheduler initialized: type=BalanceScheduler budget=2048 divide_length=128 ...

# worker 子进程：NPUWorker.__init__ 跑完
[TriAttention-Ascend] NPUWorker initialized: type=NPUWorker budget=2048 stats_path=...

# 之后若干个 step（没 signal 的，纯 decode）
... 没有 TriAttention-Ascend 日志 ...

# 第一个有 signal 的 step：scheduler 端
[TriAttention-Ascend] signal triggered req=... step=N estimated_cache_len=M reason=length_threshold

# 同一个 step：worker 端 lazy proxy install
[TriAttention-Ascend] lazily installed runner proxy: budget=... stats_path=... model_path=...

# 下一个 step：如果有新的 signal 触发，又会有一行 "signal triggered"
[TriAttention-Ascend] signal triggered req=... step=N+1 estimated_cache_len=M' reason=...
```

**任何一个 log 没出现 = 对应链路断了**。debug 步骤：


| 缺的 log                          | 可能原因                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `plugin entry point invoked`    | `pip install -e .` 没跑（entry point 没注册）                                                                                                    |
| `first install complete`        | 某个 import 失败，看上面 ERROR 行                                                                                                                  |
| `meta-patch: Scheduler rebound` | adapt_patch 没跑（vllm-ascend 不是 0.18.0 或者 `VLLM_ASCEND_BALANCE_SCHEDULING` 是 0）                                                             |
| `Scheduler initialized`         | `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()` 返回的不是 BalanceScheduler，或 super().**init**() 走的不是 `_patched_scheduler_init` |
| `NPUWorker initialized`         | `NPUWorker.__init__` 没被 patch（看 `_PATCHED_WORKER_ACTIVE`）                                                                                 |
| `lazily installed runner proxy` | `TRIATTN_RUNTIME_SPARSE_STATS_PATH` 没设或文件不存在（`install_runner_compression_hook` raise 了 `RuntimeError`）                                    |
| `signal triggered` 但 tpot 没改善   | 压缩成功但 selector 的 keep 选错了（看 stats 文件的 `num_layers` / `num_kv_heads` 是不是当前模型的）                                                             |


---

# 阶段 I：跟原生 TriAttention 比，ascend 端独有的 4 个事情

1. **Meta-patch on `vllm.v1.core.sched.scheduler` module**：原生命令链没
  提到这个，因为原生只需要 patch 一次就够了。Ascend 端必须 patch 三次
   （三个进程）+ 还要对付 vllm-ascend 的 `Scheduler = BalanceScheduler` rebind，
   所以加了 `__setattr__` 代理。
2. `**_patched_npu_worker_init` 里的 `ensure_patches_installed(reason="npu_worker_post_init")` 防御性 re-apply**：原生
  Worker 也有 re-apply（CUDA 端的 `integration_monkeypatch.py` 也有
   `_reapply_`*），但 ascend 端的实现更简单——因为 ascend 没有 CUDA 那边的
   vllm.distributed KV connector 复杂借用关系，meta-patch 已经能 cover
   99% 的情况，剩下的就用 worker init hook 兜底。
3. **Lazy runner proxy install**：`TriAttentionModelRunner` 不在
  `NPUWorker.__init__` 里装，而是**第一次**有 `triattention_signals` 时
   才装上去。这是因为 vllm-ascend 在 `NPUModelRunner.__init__` 期间要做
   ACL graph 编译 / 设备初始化 / 权重 dispatch，这些都不能被我们的
   `TriAttentionModelRunner` 拦下来。原生 CUDA 端也有类似的考虑，但实现
   路径不同（CUDA 端是在 `gpu_worker.Worker.__init__` 里 patch input prep）。
4. `**gpu_seq_len_patch` 在 ascend 端是 no-op**：CUDA 端的 input prep
  patch 通过改 `prepare_pos_seq_lens` 把 logical length 同步给 vllm 的
   `attn_metadata.seq_lens`，让 attention kernel 读正确的 logical length。
   ascend 端 vllm-ascend 的 `AscendBlockTables.compute_slot_mappings`
   是用自家 Triton kernel 一次性 gather slot_mappings 的，没有"单独
   改 seq_lens"的钩子；ascend 端的实现是**不 patch seq_lens**（让
   `slot_mappings` 保持旧的 positions 也没事，因为底层 KV 已经被压缩到
   同一个 prefix）。详见 `vllm_ascend/runtime/gpu_seq_len_patch.py`
   头部 docstring。

---

# 阶段 J：Verified vs Unverified（**关键诚实声明**）


| 链节点                                                         | 状态                                    |
| ----------------------------------------------------------- | ------------------------------------- |
| Plugin 发现、entry point 注册                                    | ✅ verified（unit test）                 |
| Scheduler / NPUWorker patch 安装（Python 层面）                   | ✅ verified（7/7 unit test）             |
| 信号生成 + 跨进程桥                                                 | ✅ verified（unit test）                 |
| `_select_keep_indices` → `compute_scores_triton` Python 调用链 | ✅ verified（unit test）                 |
| `@triton.jit triattention_scoring_kernel` 在 NPU 上能跑         | ❓ **未在 ascend 上真机 verify（当前你不能实现这个）** |
| `compact_request_kv_in_place` 在 NPU 上能跑                     | ❓ **未在 ascend 上真机 verify（当前你不能实现这个）** |
| `block_pool.free_blocks` 在 NPU 上真释放                         | ❓ **未在 ascend 上真机 verify（当前你不能实现这个）** |


**"verified"** = 在 unit test 里 stub 出 vllm / vllm_ascend，跑了 mock
调度流程，证明调用链在 Python 层面是连通的。
**"未真机 verify"** = 在 ascend 机器 + 真 vllm-ascend 上没人跑过这个具体
step。**这条表是说实话的关键**：上次打了一晚上断点没走通，**很可
能就是这条链上 NPU 真实环境里某一环（compute_scores_triton / compact
_request_kv_in_place）有 ascend 不兼容的算子或 API**。我没法替
你跑真机；你跑的时候，把 NPU 上第一行的 traceback 贴给我。