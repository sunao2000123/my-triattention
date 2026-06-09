# TriAttention-Ascend v0.18.0 适配实战记录 + Push 故障排查指南

> 这份文档是 2026-06-08 / 2026-06-09 在 vllm-ascend v0.18.0 真机环境
> 上调试 TriAttention 适配的完整记录，含问题总结、根因、解决路径，
> 以及「下次再遇到 → 怎么办」的复盘指南。

---

## 目录

1. [本次问题的一句话总结](#一-本次问题的一句话总结)
2. [三段式时间线：问题 → 根因 → 修复](#二-三段式时间线问题--根因--修复)
3. [本机 Push 时遇到的 4 类阻塞及解决](#三-本机-push-时遇到的-4-类阻塞及解决)
4. [下次再遇到时怎么排查（决策树）](#四-下次再遇到时怎么排查决策树)
5. [环境 / 网络 / 代理层的修复操作清单](#五-环境--网络--代理层的修复操作清单)
6. [本机 SSH / HTTPS / Proxy 现状总览](#六-本机-ssh--https--proxy-现状总览)
7. [给未来自己的 5 条铁律](#七-给未来自己的-5-条铁律)

---

## 一、本次问题的一句话总结

**TriAttention 适配完成后，vllm-ascend v0.18.0 真机部署时，vllm 进程能正常起来、初始 plugin 日志都打、但 algorithm 核心算法 (`compute_scores_triton` / `compact_request_kv_in_place`) 在 worker 进程里**没被真正调用**，导致 `TPOT=baseline, accuracy=baseline`。**

**根因**：`triattention/vllm/runtime/integration_monkeypatch.py` 只 patch 了 **CUDA path** 的 `Worker`（`vllm.v1.worker.gpu_worker.Worker`），但 vllm-ascend 0.18.0 的真实 worker 是 `NPUWorker`（`vllm_ascend.worker.worker.NPUWorker`，**`WorkerBase` 的子类，不是 `Worker` 的子类**），调度器是 `BalanceScheduler`（rebind 后的 `Scheduler` 子类），block tables 是 `AscendBlockTables`（`BlockTables` 的子类并 override 了 `compute_slot_mappings`）—— **CUDA 那些 patch 在 ascend 上贴的全是空气**。

**关键缺失**：

| 缺失 | 影响 |
| ---- | ---- |
| `setup.py` 没注册 `triattention_ascend` entry point | vLLM 永远不知道有 ascend 端 patcher 存在 |
| `triattention/vllm_ascend/` 目录整个不存在 | 即使加了 entry point 也 import 失败 |
| CUDA plugin 没 `_is_running_on_ascend()` 早退 | ascend 上把 patch 贴到永远不用的类上 |
| `NPUWorker.execute_model` 没被 patch | scheduler 发的 `triattention_signals` 没人接 |
| `BalanceScheduler` 符号 rebind 后 helper method 可能丢 | `super().__init__()` 走 MRO 时找不到 `_build_signals` 等 |
| `AscendBlockTables.compute_slot_mappings` 是 override | CUDA 端 patch 传不上去 |
| 子进程 worker 没自动 re-apply | 主进程 patch 不会跨进程复制 |

**解决方法**：在 `triattention/vllm_ascend/` 下新建**完整**适配层（见 `RUN.md` 章节 A 的目录结构 + `TRIATTENTION_ASCEND_ANALYSIS.md` 详述的 8 个根因），用 **4 大工程原则** 落地：

1. **最小侵入**：只 setattr 第三方 class 对象，不改源文件
2. **信号驱动**：`setattr(scheduler_output, "triattention_*")` 跨进程桥
3. **懒加载**：`TriAttentionModelRunner` 在第一次 signal 触发时才装
4. **状态显式同步**：`block_pool.free_blocks(reversed(removed_blocks))` 真回收

---

## 二、三段式时间线：问题 → 根因 → 修复

### 阶段 1：编写代码（commits 6c7d8a7）

- 用户提需求：4 个工程原则 + 适配 vllm-ascend v0.18.0
- 通过比对 `vllm-releases-v0.18.0/`、`vllm-ascend-releases-v0.18.0/`、`triattention/vllm/` 三份源码，定位 8 个根因（写进 `TRIATTENTION_ASCEND_ANALYSIS.md`）
- 新建 `triattention/vllm_ascend/` 包（`plugin.py` + `runtime/integration_monkeypatch.py` + `runtime/scheduler_ascend.py` + `runtime/worker_ascend.py` + `runtime/gpu_seq_len_patch.py`）
- 新建 `triattention_ascend` entry point；CUDA 端加 `_is_running_on_ascend()` 早退
- **零 smoke test**（按用户硬约束）
- Commit: `6c7d8a7`（344 files, 229,553 lines）

### 阶段 2：首次 push + 暴露 issue #1（commits 62cdcac）

- 用户在真机部署后报告：`TPOT=30ms`（= baseline）、精度不掉
- 真机日志显示 `WorkerProc hit an exception. RuntimeError: TRIATTN_FATAL_TRITON_SCORING_REQUIRED:unexpected_skip: req=chatcmpl-...:step=4:reason=no_compactable_groups`
- **issue #1 根因**：scheduler 端在 step=4 触发 `should_compress=True`，但 `hook_group_pipeline.py:148` 返回 `no_compactable_groups`（per-head topk 在 `cache_len ∈ [threshold, num_kv_heads*block_size]` 区间内没有可保留 token），strict mode 抛 RuntimeError 杀进程
- **修复**：
  1. `runner.py` 把 `no_compactable_groups` 加入 `_allowed_strict_skip_reasons`
  2. `runner_compression_actions.py` 在 strict-skip raise 路径上**先** emit `skipped` event 再 raise（避免 scheduler cascade fail）
  3. `_strict_no_downgrade` 改为 `require_triton_scoring AND enable_experimental_kv_compaction`（之前 conflate）
- Commit: `62cdcac`（4 files, +67/-1）
- Push 时遇到 **GitHub 443 端口 30s/60s/180s 全 timeout**，但 SSH 通；切到 `git@github.com:...` 后成功
- Issue 回复：HTTP 201，comment id `4648959275`

### 阶段 3：诊断 `tpot=baseline, accuracy=baseline`（commits f8745ee → 5b5de50 → 61d4445 → 3e3b764 → 3bcfd7e → be52bbe）

- 用户设 `TRIATTN_DEBUG_INSTRUMENT=1` 后说"啥诊断 log 都没看到"
- 我加了 7 层探针（A→G），但用户报告**A 没出现**（虽然 B/C/D/E/F/G 都加好了）
- 加 W:worker_execute 探针（在 NPUWorker.execute_model wrapper 入口） → 发现 `proxy_installed=True, model_runner=TriAttentionModelRunner` → **proxy 装上了**
- 加 W:ensure_proxy_entry（_ensure_triattention_runner_proxy 函数入口）→ 看不到（怀疑是 logger 被静默）
- 切到 `sys.stderr.write()` 强制 print（bypass logger）→ 现在 W:ensure_proxy_entry 出现
- 加 W:before_dispatch（snapshot 实际 `self.model_runner`）→ `model_runner=TriAttentionModelRunner`，**`use_v2_model_runner=True`**
- 加 A:enter（TriAttentionModelRunner.execute_model 第一行 hard print）→ **出现**了！说明 proxy 真在跑
- 但 `W:worker_execute step=7056 signals=0 will_compress=0`——proxy 在跑，但 scheduler 端**不发** signal
- **加 cumulative counters**（`cum_will_compress` / `cumulative_applied`）让用户能一眼看出"scheduler 端发不发 signal / 发多少 / algorithm 真压缩多少"
- Push 时撞 **port 7890 代理**（`Failed to connect to 127.0.0.1 port 7890`）—— `~/.gitconfig` 里 `[http "https://github.com"] proxy = socks5://127.0.0.1:7890` 死了但 git 仍用
- 切到 SSH remote `git@github.com:...` 成功 push `be52bbe`

**当前状态**：commit `be52bbe` 已 push 到 `sunao2000123/my-triattention` main，**用户在等真机跑诊断**。等用户发 `cum_will_compress` 和 `cumulative_applied` 这两个数，就能确定 `tpot=baseline` 的最终根因。

---

## 三、本机 Push 时遇到的 4 类阻塞及解决

### 阻塞 1：entry point 缺失

**症状**：
- 启动 vllm 看不到 `[TriAttention-Ascend] plugin entry point invoked` 日志
- vllm 命令不报错（vllm 找不到 entry point 时不 fail，只是不加载）

**排查**：
```bash
python -c "import importlib.metadata as m; eps = m.entry_points().get('vllm.general_plugins', []); print([(e.name, e.value) for e in eps])"
```

**解决**：检查 `setup.py` 的 `entry_points` 字典，重新 `pip install -e . --force-reinstall --no-deps` 让 dist-info 重新写。

### 阻塞 2：Token 权限不足（fine-grained PAT 不能创建 repo）

**症状**：
```
HTTP 422 name already exists on this account
```
或
```
Resource not accessible by personal access token
```

**排查**：
```bash
curl -s -H "Authorization: token $GITHUB_PAT" https://api.github.com/user | python3 -c "import json,sys; d=json.load(sys.stdin); print('login:', d.get('login'), 'id:', d.get('id'))"
```
然后看 token 的 scope：
```bash
curl -sI -H "Authorization: token $GITHUB_PAT" https://api.github.com/user | grep -i "x-oauth-scopes"
```
- `x-oauth-scopes` 缺失 → **fine-grained PAT**，按 resource 限制访问；要看 token 是给哪个 repo 配的
- `x-oauth-scopes: repo, delete_repo, ...` → **classic PAT**，权限完整

**解决**：
- 想要 classic PAT → 重新生成时选 "classic" 而不是 "fine-grained"
- 想要 fine-grained PAT → 去 https://github.com/settings/personal-access-tokens 给这个 token 加目标 repo 资源，并勾 "Administration: Read and write"（让 `POST /user/repos` 能创建）

### 阻塞 3：HTTPS push timeout（GitHub 443 不可达）

**症状**：
```
fatal: 无法访问 'https://github.com/...': Failed to connect to github.com port 443 after NNN ms
```

**排查**：
```bash
curl -s -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" -m 15 -H "Authorization: token $GITHUB_PAT" https://api.github.com/user
```
- `HTTP 000 in 15s` → 网络到 GitHub 443 整段不通
- `HTTP 200 in 0.3s` → 网络通；问题在 git 而非 GitHub

**解决（顺序）**：
1. **首选：切 SSH remote**（前提：SSH key 已加到 GitHub）
   ```bash
   git remote set-url origin "git@github.com:USER/REPO.git"
   git push origin main
   ```
2. **次选：等几分钟重试**（可能是 GitHub 临时抽风）
3. **不要做的**：把 SSH key 写到磁盘的 `~/.git-credentials` 文件里（详见阻塞 4）

### 阻塞 4：HTTPS push 被 `127.0.0.1:7890` 代理拦截（本机最常踩的坑）

**症状**：
```
fatal: 无法访问 'https://github.com/...': Failed to connect to 127.0.0.1 port 7890 after 1 ms: Couldn't connect to server
```

**根因**：
- 你的系统装了 Clash / Surge / ShadowsocksX-NG 之类的代理，配置文件里写了 `https://github.com` 走 `socks5://127.0.0.1:7890`
- **代理进程没启动**（Clash 没开 / 进程被杀 / 配置错误）但 `~/.gitconfig` 里那行 `[http "https://github.com"] proxy = socks5://127.0.0.1:7890` 还在
- git 走代理 → 7890 没服务 → fail

**排查**：
```bash
echo "=== Test if 7890 reachable ===" && nc -z -w 3 127.0.0.1 7890 && echo "OPEN" || echo "CLOSED"
echo "=== git config ===" && cat ~/.gitconfig
echo "=== macOS system proxy ===" && scutil --proxy
```

**解法（按优先级）**：
1. **直接切 SSH remote**——你 `~/.gitconfig` 里 `[core] sshCommand` 一般已经配好：
   ```bash
   git remote set-url origin "git@github.com:USER/REPO.git"
   ssh -T -o BatchMode=yes -i ~/.ssh/YOUR_KEY git@github.com  # 验证
   git push origin main
   ```
2. **临时禁用代理 push**（不推荐，会改 git config）：
   ```bash
   git -c http.proxy= -c https.proxy= push origin main
   ```
   （**注意**：`git -c http.proxy=` 不一定能覆盖 `[http "https://github.com"] proxy = ...` 段——因为该段是 url-specific，command line 只能 unset 主 `[http]` 段；想要清 url-specific 的，必须用 `git config --global --unset "http.https://github.com.proxy"`）
3. **永久清代理配置**（解决根本问题，但要小心，会影响所有项目）：
   ```bash
   git config --global --unset "http.https://github.com.proxy"
   git config --global --unset https.proxy
   ```
4. **修你的代理**（如果你就是要用代理推）：把 Clash 启起来，确认它监听 7890

---

## 四、下次再遇到时怎么排查（决策树）

```
git push 失败？
│
├─ 错误含 "Failed to connect to github.com port 443"
│  └─ 整段网络不通
│     ├─ `curl -m 15 https://api.github.com/user` 也不通 → 整段网络死了
│     │  └─ ① 切 SSH remote：git remote set-url origin "git@github.com:USER/REPO.git"
│     │     ② 等几分钟重试
│     │     ③ 联系网络管理员
│     └─ `curl ... api.github.com` 通 → 仅仅是 git 走不到
│        └─ 跳到下面 "Failed to connect to 127.0.0.1"
│
├─ 错误含 "Failed to connect to 127.0.0.1 port 7890"  ← 最常见
│  └─ 本地代理死了 / 没启动
│     ├─ 检查 7890：nc -z -w 3 127.0.0.1 7890
│     │  └─ CLOSED → ① 切 SSH remote
│     │           ② 启动代理（Clash 启起来）
│     │           ③ git config --global --unset "http.https://github.com.proxy"
│     └─ OPEN 但仍 timeout → 代理可能不转发 GitHub，加 github.com 到 proxy bypass list
│
├─ 错误含 "Permission denied (publickey)" 或 "could not read Username"
│  └─ SSH 认证失败
│     ├─ 看看 ~/.ssh/config 里有没有配 github.com 别名
│     │  └─ cat ~/.ssh/config
│     └─ 测试：ssh -T -o BatchMode=yes git@github.com
│        └─ "Permission denied" → SSH key 没加到 GitHub，去 https://github.com/settings/keys
│        └─ "successfully authenticated" → 通了
│
├─ 错误含 "Repository not found" 或 "404"
│  └─ repo 不存在 / token 没权限
│     ├─ curl https://api.github.com/repos/USER/REPO 看返回
│     └─ 检查 token 的 resource 列表（fine-grained PAT 必须显式加）
│
├─ 错误含 "Authentication failed" / "Bad credentials"
│  └─ HTTPS 走 token 但 token 错/过期
│     ├─ 验证：curl -H "Authorization: token $TOKEN" https://api.github.com/user
│     └─ 重新生成 token
│
└─ 错误含 "could not resolve host" 或 DNS 类
   └─ DNS 问题
      └─ 换 DNS（8.8.8.8 / 1.1.1.1）或切 SSH remote
```

### Push 成功后的清理 checklist

每次 push 完成务必执行：

```bash
# 1. 把 remote URL 切回无 token 形式（避免 token 落进 .git/config 长期留存）
git remote set-url origin "https://github.com/USER/REPO.git"   # 如果用 HTTPS
# 或
git remote set-url origin "git@github.com:USER/REPO.git"     # 如果用 SSH

# 2. 验证 .git/config 不含 token
grep -c "ghp_\|github_pat" .git/config   # 期望输出 0

# 3. 清掉本次 push 用过的 env var
unset GITHUB_PAT

# 4. 严重情况：token 在聊天里贴过 / 上传日志时露出过
#    → 立刻去 https://github.com/settings/tokens Revoke
#    → 重新生成，不要再贴到任何对话
```

---

## 五、环境 / 网络 / 代理层的修复操作清单

### 5.1 一次性环境清理（推荐）

```bash
# 清掉 ~/.gitconfig 里死掉的 GitHub 代理
git config --global --unset "http.https://github.com.proxy"
git config --global --unset https.proxy

# 看现在 git 还能感知哪些代理
git config --global --get-all --show-scope http.proxy
git config --global --get-all --show-scope https.proxy
```

### 5.2 SSH key 健康检查

```bash
ls -la ~/.ssh/ | grep -E "id_|github"
# 期望：sunao2000123_github 或类似命名

# 测 SSH 通
ssh -T -o BatchMode=yes -i ~/.ssh/sunao2000123_github git@github.com
# 期望：Hi USER! You've successfully authenticated, but GitHub does not provide shell access.

# 看看 ~/.ssh/config 是否有别名为 github.com
cat ~/.ssh/config
# 期望：Host github.com / User git / IdentityFile ~/.ssh/...
```

### 5.3 macOS 系统代理检查

```bash
scutil --proxy
# 看 HTTPEnable / HTTPSEnable / SOCKSEnable
# 期望：要么全 0（关代理），要么 7897 上有服务在跑
```

### 5.4 当 push 一直 fail 时的 atomic fallback

```bash
# 强制走 SSH + 显式指定 key，绕过所有 git config
GIT_SSH_COMMAND="ssh -i ~/.ssh/sunao2000123_github -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git -c http.proxy= -c https.proxy= \
  remote set-url origin "git@github.com:sunao2000123/my-triattention.git"
GIT_SSH_COMMAND="ssh -i ~/.ssh/sunao2000123_github -o IdentitiesOnly=yes" \
  git push origin main
```

### 5.5 如果你**非要用 HTTPS + token 推**（不推荐）

只在 SSH 不可用时：

```bash
# 用临时 URL，不写进 .git/config
git push "https://x-access-token:${TOKEN}@github.com/USER/REPO.git" main

# 或者 credential helper 不写文件
git -c credential.helper= push origin main
# 这次会**提示**你输 username + token，但不会存
```

---

## 六、本机 SSH / HTTPS / Proxy 现状总览

经过 2026-06-09 的故障排查，本机现状：

| 组件 | 状态 | 路径 |
| ---- | ---- | ---- |
| SSH key | ✅ 存在并认证通过 | `~/.ssh/sunao2000123_github` |
| git remote URL | ✅ 已切 SSH | `git@github.com:sunao2000123/my-triattention.git` |
| 死代理配置 | ⚠️ 还在但已绕过 | `~/.gitconfig` 的 `proxy = socks5://127.0.0.1:7890` |
| macOS 系统代理 | ⚠️ Clash 没启动 | scutil 显示 SOCKS 7897 enabled but service down |
| 最新的 push | ✅ commit `be52bbe` 已 push | 通过 SSH，绕开 7890 |

**未来使用**：默认走 SSH remote，所有 push 都不再撞代理。`/Users/sunao2000/my_tri/.git/config` 现在是干净的（无 token、无 proxy url）。

---

## 七、给未来自己的 5 条铁律

1. **永远不要把 token 写到磁盘**——`~/.gitconfig`、`~/.git-credentials`、`.env`、commit message、任何带历史的文件。**只在 shell env var 里用、用完 unset**。Chat 里贴 token 意味着 token 在 Cursor 服务端日志、agent transcript 持久化文件、未来任何能读对话的人都看到——**已经泄漏过的必须立即 revoke**。

2. **HTTPS push 第一次撞墙时，立刻切 SSH**——别挣扎于 proxy bypass、git config unset、env var override。SSH key 配好就完了。`git remote set-url origin "git@github.com:USER/REPO.git"` 一行解决所有 7890 / 1080 / 7897 的破事。

3. **每次 push 完都把 remote URL 切回无 token 形式**——避免 token 在 `.git/config` 长期留存（即使 push 是用 `https://x-access-token:...@` 临时 URL 推的，git 可能会把 URL 落盘，慎用临时 URL 形式做 remote URL）。

4. **调试时把"log 没出现"拆解为"env var 没传 / 子进程没继承 / logger 静默 / 函数没被调"四种**——逐个用 print() 到 stderr 强制排除（vllm 0.18.0 的 logger config 在 worker 子进程里可能吞 INFO，print() 永远不被吞）。

5. **多进程系统（vllm / spark / ray）调试时，永远要加 cumulative counters**——per-step log 容易被 rate limit / exception / 子进程 stdout buffering 吞掉。process-wide counter 让你**最后一行**就能看到 lifetime totals。

---

## 附录 A：本机环境速查

```bash
# 一行命令看 push 链路所有节点
echo "=== git remote ===" && git -C /Users/sunao2000/my_tri remote -v
echo "=== git config (global) ===" && git config --global --list 2>&1 | grep -iE "proxy|user|sshcommand"
echo "=== ssh keys ===" && ls -la ~/.ssh/ | grep -E "id_|github"
echo "=== ssh config ===" && cat ~/.ssh/config 2>/dev/null
echo "=== 7890 alive? ===" && nc -z -w 3 127.0.0.1 7890 && echo "yes" || echo "no"
echo "=== 7897 alive? (macOS default) ===" && nc -z -w 3 127.0.0.1 7897 && echo "yes" || echo "no"
echo "=== SSH auth test ===" && ssh -T -o BatchMode=yes -o ConnectTimeout=5 -i ~/.ssh/sunao2000123_github git@github.com 2>&1 | head -1
```

## 附录 B：推荐的标准 push 流程

```bash
cd /Users/sunao2000/my_tri

# 1. 检查 local commit
git log --oneline -1

# 2. 检查 remote
git remote -v
# 期望：origin  git@github.com:USER/REPO.git (fetch)
#       origin  git@github.com:USER/REPO.git (push)
# 如果是 https://，先改成 SSH
git remote set-url origin "git@github.com:sunao2000123/my-triattention.git"

# 3. 推
git push origin main
# 4-30s 完成；3-4-5 是 progress 提示；最后是 "main -> main"

# 4. 验证
git log --oneline origin/main -1
# 应该跟你 local commit 一致
```

任何一步 fail，参考本文档 §四 的决策树。
