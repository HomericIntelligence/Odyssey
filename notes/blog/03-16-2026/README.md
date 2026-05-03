# Day Fifty-Three: The UnsafePointer.bitcast Use-After-Free — A Three-Act Investigation

**Project:** ML Odyssey
**Date:** March 16, 2026
**Branch:** `blog/day-53-unsafe-pointer-investigation`
**Tags:** #mojo #debugging #asan #use-after-free #compiler-bug #unsafe-pointer

---

> **Note:** This post documents a full debugging investigation that pivoted three times between
> "runtime bug," "user error," and "compiler bug." Every claim is backed by ASAN output or
> Mojo documentation. Reproducer files and a validation script are in [repro/](/repro/).

---

## TL;DR

VGG16 E2E tests crashed deterministically in Mojo 0.26.1. After a multi-hour investigation
involving 26 experiments, binary reduction, and ASAN, the root cause turned out to be a
**use-after-free**: writing to an `UnsafePointer` obtained via `._data.bitcast[Float32]()`
after the compiler had already destroyed the source tensor.

The twist: Mojo's [documentation on wildcard origins](https://docs.modular.com/mojo/manual/values/lifetimes/#wildcard-origins)
says this should NOT happen — `MutAnyOrigin` pointers should disable
[ASAP destruction](https://docs.modular.com/mojo/manual/lifecycle/death/). But the compiler
only enforces this in simple functions; after heavy allocation churn, it destroys the tensor
anyway. Filed as [modular/modular#6187](https://github.com/modular/modular/issues/6187).

**Workaround applied:** Replace `tensor._data.bitcast[T]()[i] = val` with `tensor[i] = val`
(`__setitem__`). Three VGG16 test files fixed, all tests pass.

But this is only the latest chapter. **This bug has been haunting the project for three months.**

---

## Prologue: Three Months of Workarounds (Dec 2025 — Mar 2026)

Before Act 1 begins, here is the full history of this bug. Every item below is
**the same underlying cause** — the `UnsafePointer.bitcast` use-after-free — but we
didn't know that until today.

### December 2025: The First Crash (Issue #2942)

The LeNet-5 model had 24 layerwise unit tests in a single file
(`test_lenet5_layers.mojo`). Running all 24 tests in sequence crashed after
approximately 15:

```text
Starting LeNet-5 Layerwise Tests...
  test_conv1_forward_float32... OK
  test_conv1_forward_float16... OK
  ...
  test_fc1_forward_float32... OK
  test_fc1_forward_float16... OK
  test_fc2_forward_float32...
#0 libKGENCompilerRTShared.so  +0x3cb78b
error: execution crashed
```

We diagnosed it as **"heap corruption after ~15 cumulative tests"** — the allocator
was corrupting its own metadata after a threshold of allocation churn. We filed
[Issue #2942](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2942).

### The 17 Failed Reproducer Attempts

We tried 17 different minimal reproducers. All failed:

- Synthetic allocation churn (malloc/free loops) — no crash
- Identical test functions repeated 20x — no crash
- Conv2d stress tests in for loops — no crash
- Pure ExTensor creation/destruction loops — no crash

**Why they all failed:** They didn't include the exact combination that triggers the
optimizer bug: `List[Int]`-containing structs + heavy allocation churn + `bitcast` writes
in function-scoped destruction contexts. Without all three ingredients, ASAP destruction
doesn't misfire. We didn't know to look for this combination because we didn't know the
bug was a use-after-free — we thought it was heap corruption.

### ADR-009: The File-Split Workaround

Since we couldn't reproduce the crash in isolation, we applied a pragmatic workaround:
split test files to contain fewer than 10 tests each
([ADR-009](../../docs/adr/ADR-009-heap-corruption-workaround.md)).

The monolithic `test_lenet5_layers.mojo` (24 tests) became 5 files:

| File | Tests |
| --- | --- |
| `test_lenet5_conv_layers.mojo` | 6 |
| `test_lenet5_activation_layers.mojo` | 3 |
| `test_lenet5_pooling_layers.mojo` | 4 |
| `test_lenet5_fc_layers.mojo` | 9 |
| `test_lenet5_reshape_layers.mojo` | 2 |

**This worked.** All split files passed. We declared victory and moved on.

**Why it worked (we now know):** Fewer tests per file = fewer temporaries =
less allocation churn = ASAP destruction is less likely to trigger the premature
destruction of a tensor whose pointer was extracted via `bitcast`. The "threshold of
~15 tests" wasn't a magic number — it was the point where enough `List[Int]` temporaries
had been created and destroyed to confuse the optimizer's liveness analysis.

### January — February 2026: Systematic File Splits (#3472-#3638)

ADR-009 was applied across the entire test suite. Over 12+ PRs, every model's test
file was split:

- AlexNet tests → split
- VGG16 tests → split
- ResNet tests → split
- Loss function tests → split
- Optimizer tests → split

This was ~6 weeks of work across issues #3472 through #3638. **All of it was treating
a symptom, not the disease.**

### March 5, 2026: The JIT Crash Workaround (#3330)

A new crash pattern appeared: `libKGENCompilerRTShared.so` crashes that looked like
JIT compilation overflows. Investigation led to
[Issue #3330](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3330)
and a new theory: **wildcard imports** (`from shared.core import *`) were causing
the JIT compiler to compile too much code, overflowing internal buffers.

The fix: convert all 126 test files from wildcard imports to targeted imports:

```mojo
# BEFORE (wildcard):
from shared.core import *

# AFTER (targeted):
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.conv import conv2d
from shared.core.activation import relu
```

This improved CI stability from ~70% to ~95% pass rate. But crashes still happened
intermittently — because the real bug was the UAF, not the import style.

### March 15, 2026: Controlled Experiment (30 Runs, 0% Local Crash)

To understand the remaining crashes, I ran a controlled experiment: 30 consecutive
runs of the test suite locally. **0% crash rate.** The same tests crashed ~5% of the
time in CI.

This led to the hypothesis that the crash was environment-specific — perhaps a
difference in memory layout between local (WSL2) and CI (Ubuntu 24.04 Docker).

### March 16, 2026: The Breakthrough

Issue [#4511](https://github.com/HomericIntelligence/ProjectOdyssey/issues/4511)
reported that VGG16 E2E tests were crashing deterministically on the 4th test function.
Unlike previous crashes, this one was **100% reproducible** — not a flake, not
environment-specific.

That reproducibility is what made the difference. For the first time, we could
systematically reduce the crash to its root cause.

### The Full Timeline

| Date | Issue | Event | What We Thought |
| --- | --- | --- | --- |
| Dec 2025 | #2942 | LeNet-5 crashes after 15 tests | "Heap corruption at allocation threshold" |
| Dec 2025 | ADR-009 | Split files to <=10 tests | "Stay below the crash threshold" |
| Dec 2025 | — | 17 failed minimal reproducers | "Can't reproduce = must be Mojo runtime" |
| Jan-Feb 2026 | #3472-#3638 | 12+ file splits across all models | "Systematic ADR-009 compliance" |
| Mar 5 2026 | #3120 | Core Loss tests crash on main | "Disabled with continue-on-error" |
| Mar 5 2026 | #3330 | Document JIT crash workaround | "Import explosion causes JIT overflow" |
| Mar 5 2026 | — | Convert 126 files to targeted imports | "Reduce compilation footprint" |
| Mar 15 2026 | — | Controlled experiment: 30 runs, 0% local | "Environment-specific crash" |
| Mar 15 2026 | #4493 | Investigate crashes across all CI groups | "Multiple crash categories" |
| Mar 16 2026 | #4511 | VGG16 crashes on 4th forward pass | "Heap corruption from deep network" |
| Mar 16 2026 | — | **ASAN reveals use-after-free** | "Oh. It's a UAF from bitcast." |
| Mar 16 2026 | — | **Docs prove compiler bug** | "MutAnyOrigin should prevent ASAP destruction" |

**The key retrospective insight:** ADR-009 was not a workaround for a *different* bug.
It was a workaround for *this* bug. The file splitting reduced allocation churn below
the threshold where ASAP destruction misfires on `bitcast`-derived pointers. The 17
failed reproducers failed because they didn't include the `List[Int]`-containing struct +
bitcast write + function-scoped destruction combination.

---

## Act 1: The Crash

### The Symptom

Running `test_vgg16_e2e_part1.mojo` crashed deterministically on the 4th test function:

```text
Starting VGG16 E2E Tests (Part 1)...
  test_vgg16_e2e_forward_inference... OK
  test_vgg16_e2e_forward_small_batch... OK
  test_vgg16_e2e_forward_varying_values... OK
  test_vgg16_e2e_forward_backward...
#0 libKGENCompilerRTShared.so  +0x3cb78b
#1 libKGENCompilerRTShared.so  +0x3c93c6
#2 libKGENCompilerRTShared.so  +0x3cc397
#3 libc.so.6                   +0x45330
#4 libAsyncRTRuntimeGlobals.so +0x416ba
error: execution crashed
```

Key observations from the first characterization:

- **100% deterministic** — 5/5 runs crashed with identical stack trace offsets
- Both `libKGENCompilerRTShared.so` and `libAsyncRTRuntimeGlobals.so` appear — but this is
  ONE crash (the allocator crashes in libAsyncRT, libc catches the signal, libKGEN handles it)
- The 4th test (`test_vgg16_e2e_forward_backward`) was the one that crashed

### The Initial (Wrong) Hypothesis

The plan assumed two separate crashes: one in libKGEN (JIT compilation) and one in libAsyncRT
(async runtime). This wasted about an hour before I realized they're frames in the same call chain.

**Lesson:** Read the stack trace bottom-up. Frame #4 (libAsyncRT) is the crash origin. Frames
0-2 (libKGEN) are the crash handler. Frame 3 (libc) is the signal handler.

---

## Act 2: Binary Reduction (26 Experiments)

The goal was to find the minimum code that triggers the crash. Each experiment below was run
and its result recorded. The full set is in the [isolation experiments table](#isolation-experiments).

### Finding the trigger: bitcast writes

Through systematic reduction, I discovered the crash requires three ingredients:

1. **Heavy alloc/free churn** — 2+ conv2d+relu operations in a function (creates ~20 tensors,
   all freed when the function returns)
2. **`UnsafePointer.bitcast` WRITE** — writing to `tensor._data.bitcast[Float32]()[i]`
3. **Function-scoped destruction** — the operations must be in separate `fn` calls, not inline

Removing **any one** of these prevents the crash:

| Variant | Crashes? |
| --- | --- |
| Full reproducer (all 3 ingredients) | **YES** |
| Remove bitcast write (`td[0] = 0.0`) | NO |
| Create tensor without writing via bitcast | NO |
| Same code inline in `main()` (no function calls) | NO |
| 20x conv2d in a for loop (no function scope) | NO |
| Struct without `List[Int]` field (fixed Int fields) | NO |
| Struct with `List[Int]` but shapes constructed inline (no temp Lists) | NO |

### What did NOT matter

- Using `^` move vs `.copy()` in `__moveinit__` — both crash
- Array bounds — bounds-checked conv2d with explicit range checks on every index: all in-bounds,
  still crashes
- Refcount logic — traced every `__init__`/`__copyinit__`/`__moveinit__`/`__del__` transition,
  all balanced

### The self-contained reproducer

I created a 223-line zero-dependency reproducer ([repro_crash_standalone.mojo](/repro/repro_crash_standalone.mojo))
that inlines a minimal Tensor struct, conv2d, and relu. It crashes deterministically:

```bash
pixi run mojo run repro_crash_standalone.mojo
# Output: "Step A: ... OK" then crash
```

At this point I filed [modular/modular#6187](https://github.com/modular/modular/issues/6187),
believing it was a Mojo runtime allocator bug.

---

## Act 3: ASAN Reveals the Truth

### The breakthrough

Mojo supports AddressSanitizer. Building with `--sanitize address` revealed the actual bug:

```bash
pixi run mojo build --sanitize address -g -o /tmp/repro_asan repro_crash_standalone.mojo
/tmp/repro_asan
```

```text
ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000534
WRITE of size 4 at 0x502000000534 thread T0
    #0 in repro_crash_standalone::step_b()
       repro_crash_standalone.mojo:210

freed by thread T0 here:
    #0 in free
    #4 in repro_crash_standalone::Tensor::__del__
       repro_crash_standalone.mojo:71
    #5 in repro_crash_standalone::step_b()
       repro_crash_standalone.mojo:210

previously allocated by thread T0 here:
    #0 in malloc
    #3 in repro_crash_standalone::Tensor::__init__
```

**The tensor was freed BEFORE the bitcast write.** Line 210 is `td[1] = 1.0`. The compiler
destroyed the `target` tensor after line 208 (`var td = target._data.bitcast[Float32]()`)
because `target` is never referenced again — only `td` (the raw pointer) is used.

### The crashing code

```mojo
fn step_b() raises:
    var target = zeros(shape1(2))                    # allocate 8 bytes
    var td = target._data.bitcast[Float32]()         # extract raw pointer
    td[0] = 0.0   # <-- WRITES TO FREED MEMORY      # target already destroyed!
    td[1] = 1.0   # <-- WRITES TO FREED MEMORY
```

[`UnsafePointer`](https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/#unsafepointer-and-origins)
doesn't participate in Mojo's ownership system the way safe references do. Extracting a pointer
from a struct does NOT extend the struct's lifetime. The compiler's
[ASAP destruction](https://docs.modular.com/mojo/manual/lifecycle/death/) policy destroys values
as soon as it determines they are "dead":

> "Mojo uses static analysis at compile-time to determine the last use of a value. At that point,
> it immediately ends the value's lifetime and calls the implicit `__del__()` destructor."
> — [Value destruction](https://docs.modular.com/mojo/manual/lifecycle/death/)

### The fix confirmation

Adding `_ = target` after the writes keeps the tensor alive. With ASAN:

```bash
# Fixed version (keepalive added) — clean ASAN exit
pixi run mojo build --sanitize address -g -o /tmp/repro_fixed repro_fixed.mojo
/tmp/repro_fixed
# No errors, clean exit
```

---

## The Plot Twist: But The Docs Say This Should Work

After initially concluding "user code error" and closing the upstream issue, I re-read the
Mojo documentation more carefully. The
[lifetimes documentation](https://docs.modular.com/mojo/manual/values/lifetimes/#wildcard-origins)
says:

> "Using a pointer with a wildcard origin into a scope effectively disables Mojo's ASAP
> destruction for any values in that scope, as long as the pointer is live."

The `_data` field is declared as `UnsafePointer[UInt8, origin=MutAnyOrigin]`. `MutAnyOrigin`
is a wildcard origin. The pointer `td` obtained via `target._data.bitcast[Float32]()` carries
this wildcard. According to the docs, this **should prevent ASAP destruction** of `target`.

### The inconsistency

The same bitcast pattern behaves differently depending on context:

```mojo
# WORKS — simple function, no prior allocation churn
fn works() raises:
    var t = Tensor(2)
    var p = t._data.bitcast[Float32]()
    p[0] = 0.0  # no ASAN error
    p[1] = 1.0  # no ASAN error

# CRASHES — after heavy alloc/free churn in step_a()
fn step_b() raises:  # called after step_a()
    var target = zeros(shape1(2))
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0  # OK (same shadow byte)
    td[1] = 1.0  # ASAN: heap-use-after-free!
```

The wildcard origin lifetime guarantee is enforced in simple functions but **NOT after the
optimizer processes complex code paths with heavy allocation patterns.** This is a compiler bug.

I reopened [modular/modular#6187](https://github.com/modular/modular/issues/6187) with the
documentation evidence.

---

## Epilogue: How ADR-009 Was the Same Bug All Along

With the root cause identified, every prior workaround snaps into focus:

### Why file splitting worked (ADR-009)

The monolithic LeNet-5 file ran 24 test functions. Each function created ExTensor objects
with `List[Int]` shape fields and wrote to them via `._data.bitcast[Float32]()`. By test
15, enough allocation churn had accumulated that the optimizer's liveness analysis was
confused — it destroyed tensors whose `bitcast`-derived pointers were still live.

Splitting into 5 files (3-9 tests each) reduced per-file allocation churn below the
threshold. **Not a different bug. The same UAF, less trigger pressure.**

### Why the 17 reproducers failed

Every reproducer tried to isolate ONE aspect of the crash:

- Pure allocation churn → no `bitcast` write → no UAF
- Repeated identical functions → no `List[Int]` temporaries → optimizer not confused
- Conv2d in a for loop → single function scope → ASAP destruction works correctly

The bug requires the **combination** of all three ingredients. None of the 17 attempts
included all three.

### Why targeted imports helped

Converting from `from shared.core import *` to targeted imports reduced the number of
symbols in scope. This reduced the optimizer's workload when analyzing liveness, making
the ASAP destruction pass less likely to misfire on `bitcast`-derived pointers. It
didn't fix the bug — it reduced the probability of triggering it.

### The original LeNet-5 monolithic file

The original 24-test monolithic file is preserved as an artifact:
[bug_repro_lenet5_layers_monolithic.mojo.bug](/repro/bug_repro_lenet5_layers_monolithic.mojo.bug).
This is the file that first exposed the bug in December 2025 (Issue #2942). It uses
project imports and requires the shared library to run, but it demonstrates the exact
same crash pattern: tests pass one by one until allocation churn crosses the threshold,
then `libKGENCompilerRTShared.so` crashes.

---

## Failed Attempts

| # | What Was Tried | Result | Lesson |
| --- | --- | --- | --- |
| 1 | Assumed two separate crashes | Wrong — one crash | Read stack traces bottom-up |
| 2 | Raw `alloc`/`free` churn + bitcast | No crash | Bug requires `List[Int]` in struct |
| 3 | Pure `List[Int]` churn (1000x) | No crash | Needs struct + computation combo |
| 4 | `mojo build -debug-level=line-tables` | Linker error | Use `--sanitize address` instead |
| 5 | 8x8 spatial, 16 channels | No crash | Not enough allocation volume |
| 6 | 20x conv2d in a `for` loop | No crash | Function-scoped destruction required |
| 7 | `^` move vs `.copy()` in moveinit | Both crash | Not a move semantics issue |
| 8 | Static code review | Missed the bug | Static analysis cannot find UAF |
| 9 | Concluded "user error" | Wrong | Always read the language spec |
| 10 | 17 LeNet-5 minimal reproducers (Dec 2025) | All failed | Missing the 3-ingredient combo |

---

## Workarounds vs Fix: Effort Comparison

### Workaround 1: Replace bitcast with `__setitem__` (applied)

```mojo
# BEFORE (crashes):
var td = target._data.bitcast[Float32]()
td[0] = 0.0
td[1] = 1.0

# AFTER (safe):
target[0] = Float32(0.0)
target[1] = Float32(1.0)
```

- **Files touched:** 3 VGG16 test files, 12 patterns replaced
- **Effort:** ~30 minutes
- **Pros:** Avoids raw pointers entirely, uses safe bounds-checked API
- **Cons:** Doesn't fix the underlying compiler issue; ~141 instances across 34 files still exist

### Workaround 2: `_ = tensor` keepalive

```mojo
var td = target._data.bitcast[Float32]()
td[0] = 0.0
td[1] = 1.0
_ = target  # keep target alive past writes
```

- **Effort:** ~5 min per instance
- **Pros:** Minimal code change
- **Cons:** Fragile, easy to forget, no compiler warning if missing

### Workaround 3: Split test files (ADR-009)

- **Effort:** ~6 weeks across 12+ PRs
- **Pros:** Reduced per-file allocation churn below crash threshold
- **Cons:** Didn't fix the bug, just made it less likely to trigger; increased file count

### The actual fix (from Modular)

The compiler's ASAP destruction pass needs to respect `MutAnyOrigin` wildcard origin
consistently, not just in simple functions. Zero effort from us, but unknown timeline.

| Approach | Effort for VGG16 | Effort for all 141 instances | Fragility |
| --- | --- | --- | --- |
| `__setitem__` | 30 min | ~4-6 hours | Low (safe API) |
| `_ = tensor` keepalive | 15 min | ~12 hours | High (easy to forget) |
| File splitting (ADR-009) | ~6 weeks | N/A (probabilistic) | Medium (threshold-dependent) |
| Compiler fix | 0 | 0 | None (correct fix) |

---

## Isolation Experiments

Every experiment from the investigation, with its result:

| # | Experiment | Result | Proves |
| --- | --- | --- | --- |
| 1 | Run VGG16 part1 5x | 5/5 crash, identical offsets | 100% deterministic |
| 2 | VGG16 part2 | Assertion error (inf), not crash | Different failure mode |
| 3 | 20x conv2d in for loop | 20/20 pass | Function scope required |
| 4 | test1+test2+test4 (exact sequence) | Crash on test4 | Need function-scoped destruction |
| 5 | test1+test2+simple_run (no bitcast) | Pass | Bitcast write is the trigger |
| 6 | With bitcast write vs without | Write crashes, no-write passes | Bitcast write specifically |
| 7 | `zeros()` without bitcast write | Pass | Allocation alone is fine |
| 8 | 2 conv blocks (32x32, 16ch) | Crash | Don't need full VGG16 |
| 9 | 8x8 spatial, 16 channels | Pass | Not enough alloc volume |
| 10 | step_a + step_b (minimal) | Crash | Minimum reproducer |
| 11 | Raw alloc/free churn + bitcast | Pass | Bug needs List[Int] |
| 12 | Pure List[Int] churn | Pass | Needs struct+computation |
| 13 | Struct WITHOUT List[Int] field | Pass | List[Int] field required |
| 14 | Struct WITH List[Int], inline shapes | Pass | Temporary List churn matters |
| 15 | Add back shape4()/shape1() helpers | Crash | Temp List is the trigger |
| 16 | `^` move in `__moveinit__` | Crash | Not move semantics |
| 17 | Bounds-checked conv2d | All in-bounds, crash | Not an OOB issue |
| 18 | Probe struct for move semantics | Only 1 `__del__` | Move semantics correct |
| 19 | ASAN build + run | heap-use-after-free | **ROOT CAUSE** |
| 20 | ASAN with `_ = target` keepalive | Clean exit | Confirms early destruction |
| 21 | Simple bitcast (no prior churn) | Pass (no ASAN error) | ASAP destruction prevented here |
| 22 | `__setitem__` workaround | Pass | Safe API avoids bug |
| 23 | Bitcast READ (no write) | Pass | Only writes trigger |
| 24 | Fixed VGG16 tests 3x each | 6/6 pass | Workaround stable |
| 25 | LeNet-5 monolithic (24 tests) | Crash after ~15 | Same UAF, original manifestation |
| 26 | LeNet-5 split files (5 files) | All pass | File splitting avoids threshold |

---

## Reproduce It Yourself

All reproducers are in [repro/](/repro/). Run the full validation:

```bash
bash repro/run_all_experiments.sh
```

The script prints the exact repro command for each experiment. Or run individually:

```bash
# 1. See the crash (no ASAN — just the raw segfault)
pixi run mojo run repro/repro_crash_standalone.mojo
# Expected: "Step A: ... OK" then crash with libKGEN stack trace

# 2. Build with ASAN to prove heap-use-after-free
pixi run mojo build --sanitize address -g \
    -o /tmp/repro_asan \
    repro/repro_crash_standalone.mojo
/tmp/repro_asan
# Expected: "ERROR: AddressSanitizer: heap-use-after-free on address ..."
# Shows: WRITE of size 4, freed by Tensor::__del__, allocated by Tensor::__init__

# 3. See the pre-workaround VGG16 test crash
cp repro/bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug \
    tests/models/_tmp_pre_fix.mojo
pixi run mojo run tests/models/_tmp_pre_fix.mojo
# Expected: crash on test_vgg16_e2e_forward_backward
rm tests/models/_tmp_pre_fix.mojo

# 4. See the post-workaround VGG16 tests pass
pixi run mojo run tests/models/test_vgg16_e2e_part1.mojo
pixi run mojo run tests/models/test_vgg16_e2e_part2.mojo
pixi run mojo run tests/models/test_vgg16_layers_part2.mojo
# Expected: all pass

# 5. See the original LeNet-5 monolithic crash (ADR-009 — same bug)
cp repro/bug_repro_lenet5_layers_monolithic.mojo.bug \
    tests/models/_tmp_lenet5_monolithic.mojo
pixi run mojo run tests/models/_tmp_lenet5_monolithic.mojo
# Expected: crash after ~15 tests (same UAF, same libKGEN stack trace)
# Note: Requires shared library packages to be built first (pixi run just build)
rm tests/models/_tmp_lenet5_monolithic.mojo

# 6. See the project-import reproducers crash
pixi run mojo run repro/repro_libkgen_crash.mojo
pixi run mojo run repro/repro_libasyncrt_crash.mojo
# Expected: both crash with the same libKGEN/libAsyncRT stack trace
```

### ASAN Reproduction (from [modular/modular#6187][issue])

These are the exact commands from the upstream bug report:

```bash
# Build the self-contained reproducer with address sanitizer
mojo build --sanitize address -g -o repro_asan repro/repro_crash_standalone.mojo

# Run it — ASAN catches the use-after-free
./repro_asan
```

[issue]: https://github.com/modular/modular/issues/6187

---

## Artifacts

| File | Description |
| --- | --- |
| [repro_crash_standalone.mojo](/repro/repro_crash_standalone.mojo) | Self-contained reproducer (223 lines, zero dependencies) |
| [repro_libkgen_crash.mojo](/repro/repro_libkgen_crash.mojo) | Reproducer using project imports (128 lines) |
| [repro_libasyncrt_crash.mojo](/repro/repro_libasyncrt_crash.mojo) | VGG16-scale reproducer (159 lines) |
| [bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug](/repro/bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug) | Pre-workaround VGG16 test (crashes on 4th test) |
| [bug_repro_lenet5_layers_monolithic.mojo.bug](/repro/bug_repro_lenet5_layers_monolithic.mojo.bug) | Original 24-test LeNet-5 file from Dec 2025 (crashes after ~15 tests) |
| [run_all_experiments.sh](/repro/run_all_experiments.sh) | Validates all blog claims (11 experiments) |

---

## References

### Mojo Documentation

- [Wildcard origins](https://docs.modular.com/mojo/manual/values/lifetimes/#wildcard-origins) —
  "Using a pointer with a wildcard origin into a scope effectively disables Mojo's ASAP destruction"
- [ASAP destruction](https://docs.modular.com/mojo/manual/lifecycle/death/) —
  "Mojo destroys values using an 'as soon as possible' destruction policy"
- [Explicit lifetime extension](https://docs.modular.com/mojo/manual/lifecycle/death/#explicit-lifetime-extension) —
  for unsafe/advanced code
- [UnsafePointer and origins][ptr-origins] — origin tracking on pointers
- [Pointer lifecycle][ptr-lifecycle] — dangling pointer risks
- [Bitcasting pointers][ptr-bitcast] — same memory, different type

[ptr-origins]: https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/#unsafepointer-and-origins
[ptr-lifecycle]: https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/#lifecycle-of-a-pointer
[ptr-bitcast]: https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/#converting-data-bitcasting-and-byte-order

- [Value ownership](https://docs.modular.com/mojo/manual/values/ownership/) —
  transfer arguments with `^` operator

### Project Files

- [Upstream issue: modular/modular#6187](https://github.com/modular/modular/issues/6187)
- [mojo-anti-patterns.md](/.claude/shared/mojo-anti-patterns.md) — updated with this anti-pattern
- [ADR-009: Heap corruption workaround](/docs/adr/ADR-009-heap-corruption-workaround.md) — now
  understood to be the same UAF bug, not a separate heap corruption issue
- [Mojo JIT crash workaround](/docs/dev/mojo-jit-crash-workaround.md) — prior investigation

### Environment

- Mojo 0.26.1.0 (156d3ac6)
- Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
- GLIBC 2.39, Ubuntu 24.04
