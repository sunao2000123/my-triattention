# Fix: Post-Compression `effective_len_regressed` Fatal on Ascend

Branch: `fix/effective-len-tracker-events-and-current-cache-len`
Commit: `919b07c`
Scope: `triattention/vllm/runtime/{state,hook_runtime_context}.py`,
`triattention/vllm_ascend/runtime/integration_monkeypatch.py`
(3 files, +39 / -10)

---

## 1. Problem Statement

After fixing the "20k → 4096 truncation" issue (so the selector now sees
the full 20k-token KV cache), long-video generation on Ascend crashed
one step after the first compression with:

```
RuntimeError: TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed:
  req=cmpl-...:effective_tokens=19790:num_computed_tokens=19790:guard_upper=2304
```

The fatal is raised by `triattention/vllm/runtime/hook_runtime_context.py`
inside `build_hook_runtime_context`. It's a defensive regression guard
that fires when, after a request has been compressed at least once, the
hook observes `effective_tokens` has jumped back to ~full
`num_computed_tokens`. Diagnostic instrumentation added to the same
branch revealed two interacting root causes.

### 1.1 Observed log timeline (single request)

| Step | What ran | `effective_tokens` | `current_cache_len` (worker state) |
|------|----------|--------------------|-----------------------------------|
| 1 (prefill) | Selector ran layers 0-35 with `total_tokens=19789`, applied compaction | (compact) | (start: 0) |
| 1 (post) | `mark_compressed(step=1, cache_len=2048, scheduled_tokens=19789)` | — | **21837** *(bug B)* |
| 2 | `batch_queue_dedup` skipped (last_step diff <=1, sched=1) | — | 21837 *(unchanged)* |
| 3 | New signal; `_resolve_estimated_effective_tokens` returns 21839; capped at `kv_upper=19790` | **19790** | 21837 |
| 3 | Regression guard: `19790 > 2304 + slack AND 19790 >= 0.9 * 19790` → **fatal** | — | — |

The crash is therefore a cascading effect of two bugs, not a bug in the
guard itself.

---

## 2. Root Causes

### Bug A — `_apply_compression_events` failure was silently swallowed

* **Where:** `triattention/vllm_ascend/runtime/integration_monkeypatch.py`,
  `_patched_scheduler_update_from_output`, the call to
  `TriAttentionAscendScheduler._apply_compression_events(self, compression_events)`.
* **Symptom (from diagnostics):** EngineCore log line
  `S:has_override=False req=... snapshot=(-1, 0)` was printed
  *every* step, including step 3 — meaning
  `scheduler._effective_len_tracker._effective_len[req]` was
  *still* empty even after step 1's compaction event had been received.
* **Effect:** With `_effective_len[req]` empty, `has_effective_len_override`
  returns `False`, so `_build_signals` takes the fast path at
  `scheduler.py:247` and uses `request.num_computed_tokens` as
  `effective_base_len`. Because `num_computed_tokens` is
  vLLM's physical prompt-progress (monotonic, never shrinks after
  prefill), `effective_base_len` stays at the full 19789–19790 forever,
  and the regression guard fires every step.

The reason `_effective_len[req]` never got populated: the call to
`_apply_compression_events` (which is what calls
`tracker.apply_compression(cache_len_after=2048, num_computed=...)`)
was wrapped in nothing. Any exception inside it would propagate out
of `update_from_output` — but at the vLLM EngineCore boundary that
call is also wrapped by framework code that *swallows* exceptions
from `Scheduler.update_from_output`. The exception never reached
Python logging, and the EngineCore kept running with a permanently
empty `_effective_len_tracker`.

### Bug B — `mark_compressed` wrote `cache_len + scheduled_tokens`

* **Where:** `triattention/vllm/runtime/state.py:116` (pre-fix),
  inside `RequestStateStore.mark_compressed`.
* **Symptom (from diagnostics):** the worker log line
  `STATE:mark_compressed_AFTER req=... cc=1 cur_len=21837` showed
  that the post-compaction `current_cache_len` was 21837 — not the
  actual KV length 2048.
* **Cause:** the line was
  `state.current_cache_len = cache_len + max(0, scheduled_tokens)`.
  In the prefill+compress step, `cache_len=2048` (post-compaction KV
  length) and `scheduled_tokens=19789` (the full prompt chunk that
  was just absorbed). The sum 21837 has no physical meaning.
* **Effect:** even if Bug A had been fixed, the worker-side
  `_resolve_estimated_effective_tokens` (in
  `hook_runtime_context.py`) would have returned `21837` and
  triggered the same regression guard one step later. Also, the
  worker's own `effective_kv` self-trigger path at
  `runner.py:218-219` (`effective_kv = actual_kv + max(1, scheduled)`)
  uses the same `state.current_cache_len` as `actual_kv`, so a wrong
  value there would have caused spurious self-triggers as well.

---

## 3. Solution

### 3.1 Fix Bug B at the source — write the correct `current_cache_len`

The post-compaction KV cache length is `cache_len`, not
`cache_len + scheduled_tokens`. The `scheduled_tokens` value was
already a poor substitute for "decode tokens absorbed during the
compression step's own forward pass", and that absorption is
already tracked by `state.last_absorbed_cache_len` (set
unconditionally in the same function), and by the scheduler-side
`EffectiveCacheLenTracker.observe_num_computed` path.

### 3.2 Fix Bug A — surface the swallowed exception

Wrap the call to
`TriAttentionAscendScheduler._apply_compression_events` in a
`try/except` that:
1. Writes the exception class + first 300 chars to **stderr** with
   the unique prefix `[TriAttention-Ascend][FATAL]`. vLLM's worker
   logger configuration suppresses `INFO`/`WARNING` in subprocesses
   but `sys.stderr` is always visible in process logs.
2. Calls `logger.exception(...)` so the full traceback is captured
   by the normal logger.
3. Re-raises so the failure is no longer silent — the EngineCore
   will surface the real cause instead of running with a broken
   tracker.

### 3.3 Defensive layer at the consumer

`_resolve_estimated_effective_tokens` (the *only* place worker-side
state converts to `effective_tokens`) now applies a sanity check:
if `state.current_cache_len` is more than 2× the
`signal.estimated_cache_len`, it falls back to
`signal.estimated_cache_len`. This bounds any future regression of
the same shape (e.g. a future caller writing wrong values into
`current_cache_len`) without altering the common path.

---

## 4. Implementation Detail (file / line)

### 4.1 `triattention/vllm/runtime/state.py` — fix `current_cache_len` formula

File: `triattention/vllm/runtime/state.py`
Function: `RequestStateStore.mark_compressed`
Lines: 101-128 (pre-fix line 116 was the bug; the fix is at line 120)

```python
# Before:
state.last_compression_step = step
# Include this step's scheduled decode tokens so that the next step's
# effective_base calculation accounts for the KV entries written by
# the compression step's own decode.  Without this, the next step
# computes the same effective slot as the compression step (off-by-1).
state.current_cache_len = cache_len + max(0, scheduled_tokens)
state.current_cache_len_semantics = "estimated_with_scheduled"

# After:
state.last_compression_step = step
# `current_cache_len` MUST equal the post-compaction KV length, not
# `cache_len + scheduled_tokens`. The prefill+compress path passes
# scheduled_tokens=19789 (full prompt) and cache_len=2048 (post-
# compaction KV length); adding them gives 21837, which is greater
# than the next step's threshold and immediately re-triggers a
# second compression. The KV cache length is `cache_len`; the
# scheduler-side effective-length tracker separately accounts for
# scheduled decode tokens via its own observe_num_computed path.
state.current_cache_len = max(0, int(cache_len))
state.current_cache_len_semantics = "post_compaction"
```

The semantics label was also renamed from
`"estimated_with_scheduled"` to `"post_compaction"` so that any
downstream assertion or log-grep that pinned on the old string
breaks loudly on the next change, rather than silently continuing
to describe a value that's no longer what it was.

`state.last_absorbed_cache_len` (set at line 123) is unchanged —
it already receives `int(cache_len)`, which is correct.

### 4.2 `triattention/vllm_ascend/runtime/integration_monkeypatch.py` — surface swallowed exception

File: `triattention/vllm_ascend/runtime/integration_monkeypatch.py`
Function: `_patched_scheduler_update_from_output`
Lines: 240-267 (pre-fix lines 240-251; the new `try/except` is at
lines 249-267)

```python
if compression_events:
    applied = [e for e in compression_events if e.get("status") == "applied"]
    logger.info(
        "[TriAttention-Ascend] update_from_output: received %d events "
        "(%d applied) via %s",
        len(compression_events),
        len(applied),
        source,
    )
    try:
        TriAttentionAscendScheduler._apply_compression_events(
            self, compression_events
        )
    except Exception as _ev_exc:
        # Previously this exception was silently swallowed by the
        # framework, which left the scheduler's
        # `_effective_len_tracker` permanently empty and caused the
        # next scheduling round to re-fire on full num_computed_tokens.
        import sys as _sys_ev
        _sys_ev.stderr.write(
            f"[TriAttention-Ascend][FATAL] _apply_compression_events "
            f"raised exc={type(_ev_exc).__name__}: {str(_ev_exc)[:300]}\n"
        )
        _sys_ev.stderr.flush()
        logger.exception(
            "[TriAttention-Ascend] _apply_compression_events failed"
        )
        raise
```

The plain `TriAttentionAscendScheduler._apply_compression_events(self, compression_events)`
call that existed pre-fix was a single statement, no `try` block.
The exception it could throw (e.g. from
`_free_reclaimed_blocks`, `manager.req_to_blocks` access, etc. inside
`_apply_compression_events`) was caught somewhere upstream by the
EngineCore, the scheduler kept running, but
`_effective_len_tracker._effective_len` was never written.

The fix preserves the existing behavior when no exception is thrown
(no log spam; the existing `logger.info` still fires once with the
event count). When an exception *does* throw, it now:
1. Goes to stderr (always visible, even with vLLM log filtering).
2. Goes to the normal Python logger with full traceback.
3. Re-raises so the EngineCore fails loudly instead of running in a
   broken state.

### 4.3 `triattention/vllm/runtime/hook_runtime_context.py` — defensive cap on consumer

File: `triattention/vllm/runtime/hook_runtime_context.py`
Function: `_resolve_estimated_effective_tokens`
Lines: 25-49

```python
def _resolve_estimated_effective_tokens(
    *,
    signal: CompressionSignal,
    req_runtime_state: Any,
) -> int:
    if req_runtime_state is not None:
        compression_count = getattr(req_runtime_state, "compression_count", None)
        current_cache_len = getattr(req_runtime_state, "current_cache_len", None)
        if (
            isinstance(compression_count, int)
            and compression_count > 0
            and isinstance(current_cache_len, int)
            and current_cache_len > 0
        ):
            # Defensive: a stale or wrong `current_cache_len` (e.g. the
            # pre-fix state.py bug that wrote `cache_len + scheduled_tokens`)
            # would balloon effective_tokens and trip the regression guard.
            # If current_cache_len is implausibly large relative to
            # `estimated_cache_len`, prefer the latter.
            est_cache_len = int(getattr(signal, "estimated_cache_len", 0) or 0)
            value = int(current_cache_len)
            if est_cache_len > 0 and value > 2 * max(est_cache_len, 1):
                return max(0, est_cache_len)
            return max(0, value)
    return max(0, int(getattr(signal, "estimated_cache_len", 0)))
```

The original function was 16 lines (25-40). The defensive branch
adds 6 lines inside the existing `if` block at lines 38-48, leaving
the outer `if` and the fallback `return` unchanged. The 2×
threshold is deliberately generous: in the normal path the
worker `current_cache_len` and the scheduler `estimated_cache_len`
should be within ~1 scheduled token of each other; a 2× ratio is
wide enough to never trigger on legitimate state while still
catching the `cache_len + scheduled_tokens` shape (which produces
roughly `1 + prefill_len / cache_len` ratio, ~10× for our case).

---

## 5. How to Verify

1. Apply the branch and run the 120-frame LongLive config with
   `TRIATTN_DEBUG_INSTRUMENT=1`.
2. Confirm `current_cache_len` is now ~2048 after the first
   compression (it was 21837 pre-fix):
   `STATE:mark_compressed_AFTER req=... cc=1 cur_len=2048`
3. Confirm `has_override=True` on the next scheduling round:
   `S:has_override=True req=... snapshot=(2048, ...)`
4. Confirm `effective_tokens` is now ~2048–2050 (it was 19790
   pre-fix): `HOOK:after_subtract ... effective_tokens=2048`
5. If `_apply_compression_events` ever throws, the EngineCore log
   will now contain `[TriAttention-Ascend][FATAL]` and a full
   traceback. (Pre-fix it would have been silently lost.)