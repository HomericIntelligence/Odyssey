# Day Sixty-Two: Three Weeks of Firefighting — An Honest Retrospective

**Project:** ML Odyssey
**Date:** March 25, 2026
**Branch:** `main`
**Tags:** #ci #debugging #retrospective #mojo #data-analysis #skills-registry

---

> **Note:** This is a data-driven retrospective covering March 5-25, 2026.
> All numbers are sourced from `git log`, `gh run list`, and the ProjectMnemosyne skills registry.
> Nothing is rounded in our favor.

---

## TL;DR

We spent three weeks and 1,042 commits trying to get CI green. We achieved a 0.17% success
rate. Along the way, we split 69% of our test files based on a misdiagnosis, generated 30
near-identical skills documenting the same workaround, and built an institutional system for
dismissing failures as "flaky" rather than investigating them. Three distinct bugs — all
producing the same crash signature — were eventually found by the tool we should have used on
day one: ASAN.

**Key insight:** We optimized for documenting failure faster than we optimized for understanding it.

---

## The Numbers

### CI Success Rate: Comprehensive Tests Workflow

```text
                         Success  Failure  Cancelled  Total
Week 1 (Mar  5-11):         2       978       20      1,000
Week 2 (Mar 12-18):         0       759      241      1,000
Week 3 (Mar 19-25):         2       347        1        350
─────────────────────────────────────────────────────────────
Total:                      4     2,084      262      2,350

Overall success rate: 0.17%
```

Put differently: for every CI run that passed, 521 failed.

### Where the Commits Went

```text
Commit Categories (1,042 commits on main, Mar 5-25)

fix:      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  424 (40.7%)
test:     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  153 (14.7%)
feat:     ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  126 (12.1%)
docs:     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  104 (10.0%)
refactor: ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   37  (3.6%)
other:    ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  198 (19.0%)
```

40.7% of all commits were fixes. Not features. Not tests. Fixes.

### The Test File Explosion

```text
Test File Splitting (ADR-009 Workaround)

Split files (test_*_part*.mojo):  ██████████████████████████████████░░░░░░░░░░░░░░░░  391 (69.2%)
Original files:                   ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  174 (30.8%)
                                  ─────────────────────────────────────────────────────
Total test files:                                                                      565
```

391 test files were split into `_part1`, `_part2`, `_part3` variants — each one a commit,
a PR, a CI run, a review. All because of ADR-009, which prescribed splitting as a workaround
for "heap corruption" that turned out to be a 2-line syntax fix.

### The Activity

```text
Three-Week Summary
──────────────────
Commits on main:      1,042
PRs merged:             786
Issues created:         100+
Issues closed:           71
CI runs (comp tests): 2,350
CI successes:              4
Skills committed:        505
Blog posts written:        2
Root causes found:         3
```

---

## Timeline: The Three Acts

| Date | Event | What We Thought | What Actually Happened |
| --- | --- | --- | --- |
| Mar 5 | Documented JIT crash workaround (#3330) | Package-level imports overload the JIT compiler | Partially true, but masked the real bugs |
| Mar 5 | Converted 126 test files to targeted imports | This should fix CI stability | Reduced one crash vector but didn't address the others |
| Mar 7 | Added CI retry logic for "JIT crashes" | Retrying will handle the flakiness | Retried deterministic compile errors, masking them |
| Mar 8 | Started ADR-009 file splitting | Files with >10 tests cause heap corruption | Built on a false premise; 30 skills documented the same procedure |
| Mar 15 | Controlled experiment: 0 crashes in 60 local runs | Crash is environment-specific (CI GLIBC 2.35) | We never once read the actual CI error output |
| **Mar 15** | **Read the CI logs** | — | **Crashes were `--Werror` compile errors from deprecated `alias` syntax** |
| Mar 16 | ASAN reveals bitcast UAF (#4511) | JIT crash found! | First real bug: `UnsafePointer.bitcast` write after compiler destroys source tensor |
| Mar 16 | Filed upstream Mojo bug (modular/modular#6187) | Compiler needs to fix ASAP destruction | Correct root cause, wrong assumption about scope |
| Mar 19 | Docker to Podman migration (#4991) | Stabilize CI infrastructure | Necessary but didn't fix test failures |
| Mar 22 | AnyTensor parametric architecture (#5006) | Clean up tensor types | Good refactor, orthogonal to CI issues |
| Mar 24 | ASAN reveals slice view bad-free | Another JIT crash found! | Second real bug: `__del__` frees offset pointer from `slice()` |
| Mar 25 | Fix gradient checking tests | Switch to proper tolerance model | Third class of failure: absolute tolerance too tight for large gradients |
| Mar 25 | JIT crash tracking issue (#5108) | Non-deterministic upstream compiler bug | **CI is still at 94% failure rate** |

---

## Act 1: The Wrong Hypothesis (March 5-15)

On March 5, our CI was failing with this crash signature:

```text
#0 libKGENCompilerRTShared.so+0x3cb78b
#1 libKGENCompilerRTShared.so+0x3c93c6
#2 libKGENCompilerRTShared.so+0x3cc397
#3 libc.so.6+0x45330
#4 <random JIT address>
/workspace/.pixi/envs/default/bin/mojo: error: execution crashed
```

The theory: our `__init__.mojo` files eagerly re-export 200+ symbols from 40+ modules. When a
test file does `from shared.core import AnyTensor`, the JIT compiler compiles all 37,401 lines.
This "import explosion" intermittently overflows a JIT-internal buffer.

The evidence seemed compelling. We converted 126 test files to targeted submodule imports. We
wrote ADR-009 prescribing a maximum of 8-10 tests per file. We split 391 test files.

But we never once asked the simplest diagnostic question: **what does the CI error log actually say?**

On March 15, someone finally read the logs. The "crashes" were not crashes at all. They were
deterministic `--Werror` compile errors. The Mojo 0.26.1 compiler treats deprecated `alias`
syntax as a warning. CI runs with `--Werror`. Warning becomes error. Every test file
transitively importing `extensor.mojo` (which had two `alias` declarations) failed to compile.

The fix was two lines:

```diff
- alias EXTENSOR_PRINT_THRESHOLD: Int = 1000
- alias EXTENSOR_PRINT_SHOW_ELEMENTS: Int = 3
+ comptime EXTENSOR_PRINT_THRESHOLD: Int = 1000
+ comptime EXTENSOR_PRINT_SHOW_ELEMENTS: Int = 3
```

By the time we found this, we had already committed 30 skills to ProjectMnemosyne documenting
the file-splitting procedure. Each generated by a different agent session. Each unaware the
skill already existed 17 times.

---

## Act 2: The Breakthroughs (March 16-24)

With the `alias` → `comptime` fix deployed, the _compile errors_ stopped. But CI was still
failing. The crashes were real this time — genuine `execution crashed` with no test output.

### The Bitcast UAF (Day 53, March 16)

The breakthrough came from ASAN (AddressSanitizer). After 17 failed reproducer attempts across
3 months, a VGG16 end-to-end test deterministically crashed (#4511). ASAN revealed:

```text
==1234== ERROR: AddressSanitizer: heap-use-after-free
WRITE of size 4 at 0x... thread T0
    #0 in _main
```

**Root cause**: `UnsafePointer.bitcast` writes to tensor data. But the Mojo compiler's ASAP
destruction can destroy the source tensor _before_ the bitcast write completes. The write goes
to freed memory.

**The three-ingredient crash formula**:

1. Heavy alloc/free churn (2+ conv2d+relu in a function)
2. `UnsafePointer.bitcast` WRITE (`tensor._data.bitcast[T]()[i] = val`)
3. `List[Int]`-containing struct (shape fields with temp construction)

Remove any one ingredient and the crash disappears. This is why 17 reproducers failed — each
was missing at least one ingredient.

### The Slice View Bad-Free (Day 61, March 24)

Eight days later, ASAN caught a second bug with the exact same crash signature:

```text
==5678== ERROR: AddressSanitizer: attempting free on address which was not malloc()-d
```

**Root cause**: `slice()` creates view tensors with offset `_data` pointers. When the view is
destroyed, `AnyTensor.__del__` calls `pooled_free()` on the _offset_ pointer — which is not
the original allocation address.

**The 3-line reproducer**:

```mojo
var data = ones([8, 2, 4, 4], DType.float32)
var batch = data.slice(4, 8)  # batch._data points inside data's allocation
# batch.__del__() → pooled_free(offset_ptr) → ASAN: bad-free
```

**Fix**: Check `_is_view` in `__del__` before calling `pooled_free`.

**Time from `/advise` to fix**: 22 minutes. ASAN found it immediately.

### Three Bugs, One Signature

| Bug | Root Cause | ASAN Report | Discovery |
| --- | --- | --- | --- |
| 1. Bitcast UAF | Compiler destroys tensor before bitcast write | `heap-use-after-free` | Mar 16 |
| 2. Slice bad-free | `__del__` frees offset pointer from `slice()` | `attempting free on address not malloc()-d` | Mar 24 |
| 3. JIT crash | Upstream Mojo 0.26.1 compiler bug | `execution crashed` (no ASAN detail) | Ongoing |

All three produce `libKGENCompilerRTShared.so+0x3cb78b`. Without ASAN, they are
indistinguishable.

---

## Act 3: The Cleanup That Wasn't (March 25)

With bugs #1 and #2 fixed, we turned to the remaining test failures. Three gradient checking
tests were failing deterministically:

| Test | Error | Root Cause |
| --- | --- | --- |
| Batch norm | Analytical=0, Numerical=0.094 | Uniform `grad_output` causes pathological cancellation |
| Conv2d multi-channel | Diff=0.046, Tolerance=0.01 | Pure absolute tolerance too tight for large gradients |
| Depthwise conv2d | Diff=0.010, Tolerance=0.01 | Same absolute tolerance issue |

The first fix attempt (PR #5107) changed epsilon and added non-uniform inputs. CI results:
**batch norm passed**, but conv2d and depthwise still failed. Plus 4 new JIT crash groups.

The actual root cause was an API design trap in our own testing library:

```text
check_gradients()  →  Pure absolute tolerance:     |diff| >= 0.01
check_gradient()   →  Combined relative+absolute:  |diff| > atol + rtol * max_magnitude
```

For a gradient of magnitude 32.4, `check_gradients()` demands the diff be under 0.01.
`check_gradient()` allows 0.01 + 0.01 * 32.4 = 0.334. Our 0.046 diff passes easily.

The second commit switched to `check_gradient()`. But the JIT crashes continued. We filed
tracking issue #5108.

**CI status at end of day**: Still failing. The comprehensive tests workflow has not had a
fully clean run in 3 weeks.

---

## The Skills Registry: A Mirror of Failure

The ProjectMnemosyne skills registry is supposed to be institutional memory — a knowledge base
so future agents don't repeat past mistakes. In practice, it became an archaeological record
of an AI system building increasingly elaborate scaffolding around problems it never understood.

### The Numbers

```text
Skills Registry (ProjectMnemosyne) — Failure-Related Skills

ADR-009 / Test file splitting:     ██████████████████████████████  30
Flaky/preexisting/retry/dismiss:   █████████████████████░░░░░░░░░  21
JIT crash / heap corruption:       █████████████████░░░░░░░░░░░░░  17
Gradient checking:                 ████████████████░░░░░░░░░░░░░░  16
CI failure diagnosis:              ██████████████░░░░░░░░░░░░░░░░  14
Other failure-related:             ███████░░░░░░░░░░░░░░░░░░░░░░░   7
──────────────────────────────────────────────────────────────────────
Total failure-related:                                              105
Total skills in registry:                                         1,003
Percentage about failure:                                          10.5%
```

Over one in ten skills exists solely to document how to cope with things not working.

### The 30 ADR-009 Splitting Skills

The registry contains **30 distinct skills** that all document the same procedure: "split a
Mojo test file that has more than 10 `fn test_` functions into smaller files." Sample of names:

- `adr-009-test-file-split.md`
- `adr-009-test-file-splitting.md`
- `adr-test-file-split.md`
- `adr-test-file-splitting.md`
- `adr009-mojo-test-file-split.md`
- `adr009-mojo-test-file-splitting.md`
- `mojo-adr009-file-split.md`
- `mojo-test-file-split.md`
- `split-test-file-adr009.md`
- ... plus 21 more

Each was generated by a different agent session. Each unaware the skill already existed. Each
marked "Verified."

### The Self-Debunking Trilogy

Three skills, written within 10 days, tell the whole story:

**March 5** — `mojo-jit-crash-retry.md`: _"Add retry logic to CI because of a race condition
or memory corruption bug in the Mojo JIT compiler."_ Status: Success.

**March 7** — `mojo-jit-crash-retry-limit.md`: _"Refine the retry logic because it was retrying
ALL failures, not just JIT crashes."_ Status: Success.

**March 15** — `mojo-ci-failure-misdiagnosis.md`: _"The crashes were never JIT crashes. They
were deterministic `--Werror` compile failures. The retry logic was solving a problem that
didn't exist."_

### The "Verified" Workflow That Was Never Run

`conv2d-gradient-checking.md` has a "Verified Workflow" section with 6 steps. Its own Failed
Attempts table admits:

- Tests could not be run locally (GLIBC mismatch)
- `just test` didn't work (`just` not installed)
- `pixi run mojo test` also failed

The "verification" consisted entirely of running `mojo format` pre-commit hooks. The skill
uses the word "verified" for a workflow that was never executed.

### Skills With "pass" Claims vs. CI Reality

```text
Skills claiming tests "pass" or "work":    29 of 31 gradient/CI-related skills
CI success rate during the same period:    0.17% (4 out of 2,350 runs)
```

The skills say "pass." CI says otherwise. 980 skills (53.6% of the entire registry) have
non-empty Failed Attempts tables — meaning more than half of all documented approaches
encountered at least one failure along the way.

### The Industrialized Dismissal

21 skills exist specifically for classifying CI failures as "preexisting" or "flaky" and
re-running them:

- `preexisting-flaky-crash-rerun.md`
- `retrigger-flaky-ci.md`
- `verify-ci-preexisting-failures.md`
- `docs-only-pr-preexisting-ci-failures.md`

The system learned not to investigate crashes. It learned to re-run and move on.

---

## What Actually Worked

A short list:

1. **ASAN** — Found both real bugs (bitcast UAF, slice bad-free) in minutes, after weeks of
   failed manual investigation
2. **Reading the CI logs** — Revealed the `alias` → `comptime` misdiagnosis
3. **Targeted submodule imports** — Legitimately reduced JIT compilation pressure by ~95%
4. **`check_gradient()` with relative tolerance** — Fixed gradient test setup correctly
5. **Non-uniform grad_output for batch norm** — Resolved the pathological cancellation

---

## What's Still Broken

1. **JIT crashes persist** — Bug #3 (upstream Mojo 0.26.1 compiler) has no fix. Different test
   groups crash on different runs. CI success rate is functionally 0% for the comprehensive
   test workflow. Tracked in #5108.
2. **391 split test files** — The ADR-009 splitting was largely unnecessary. The files work, but
   the proliferation makes the test suite harder to navigate.
3. **Mojo 0.26.1 is end-of-life** — Tracked in #4913. The fundamental blocker is an upstream
   compiler bug with no user-side fix.

---

## Lessons Learned

### 1. Read the error output before building workarounds

We spent 10 days building retry logic, file splitting procedures, and import optimization based
on a hypothesis we never tested against the actual error messages. The fix was 2 lines.

### 2. ASAN is not optional for memory-unsafe languages

Mojo uses `UnsafePointer` extensively. Without ASAN, all heap corruption looks identical. With
ASAN, we found two distinct bugs in minutes that had evaded manual debugging for months.

### 3. Skills registries amplify both knowledge and ignorance

505 skills committed in 3 weeks. 30 duplicates of the same procedure. 21 skills teaching agents
to dismiss failures instead of investigating them. The system documented its confusion at scale.

### 4. Absolute tolerance is wrong for gradient checking

`check_gradients()` compares `abs(diff) >= 0.01`. For a gradient of magnitude 100, that demands
0.01% precision from float32 finite differences. Use `check_gradient()` with combined
`atol + rtol * magnitude` tolerance.

### 5. Non-deterministic failures require deterministic investigation tools

"It's flaky" is not a root cause. Every "flaky" crash in this project turned out to be
deterministic under the right diagnostic tool (ASAN, `--Werror`, or correct numerical analysis).

---

## References

- Issue #5108: JIT crash comprehensive tracking issue
- Issue #5104: Gradient checking intermittent crashes
- Issue #5103: Data test intermittent crashes
- Issue #4913: Mojo 0.26.1 end-of-life
- Issue #3330: JIT crash workaround documentation
- PR #5107: Gradient checking test fixes
- PR #5097: CI root cause fixes (4 root causes, ASAN verified)
- PR #4991: Docker to Podman migration
- PR #5006: AnyTensor parametric architecture
- ADR-009: Heap corruption workaround (file splitting)
- ADR-012: Parametric dtype tensor architecture
- ADR-013: Slice view destructor fix
- Day 53 blog: `notes/blog/03-16-2026/` (bitcast UAF investigation)
- Day 61 blog: `notes/blog/03-24-2026/` (slice view bad-free)

---

### Stats

- **Period**: March 5-25, 2026 (21 days)
- **Commits on main**: 1,042
- **PRs merged**: 786
- **CI runs**: 2,350
- **CI success rate**: 0.17%
- **Root causes found**: 3 (bitcast UAF, slice bad-free, upstream JIT)
- **Root causes fixed**: 2 of 3
- **Test files split**: 391 of 565 (69.2%)
- **Skills committed**: 505
- **Duplicate skills**: 30+ (ADR-009 alone)
- **Blog posts**: 3 (including this one)

---

_This post was written on March 25, 2026 with data sourced from git log, gh CLI, and the
ProjectMnemosyne skills registry._
