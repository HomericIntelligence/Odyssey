# Day One Hundred Sixty-Five: The 3.6 GB Virtual Ghost

**Project:** ML Odyssey
**Date:** April 20, 2026
**Branch:** `fix-ci-failures-asan-circular-benchmark`
**Tags:** #mojo #debugging #ci #virtual-memory #jit #modular-bug

---

> **Note:** This post documents a three-pivot investigation that started as "fix flaky CI
> crashes" and ended with a filed upstream bug report and a one-command reproducer. Every
> measured value below came from `/proc/$PID/status` polling and `ulimit -v` binary search
> inside the container. Reproducer files are in [repro/](/repro/).

---

## TL;DR

CI was crashing with `libKGENCompilerRTShared.so` signals on tests that passed locally.
After two wrong theories — import chain explosion, then monomorphization overflow — the
real cause turned out to be something nobody expected: **the Mojo JIT compiler
unconditionally reserves ~3.6 GB of virtual address space per invocation, regardless of
what the source file does.** Even `def main(): print("hello")` peaks at 3.6 GB virtual.

On GitHub Actions free-tier runners (7 GB total), two concurrent `mojo` processes
compete for virtual address space → one of them loses → crash in
`libKGENCompilerRTShared.so` → CI reports "JIT crash."

Physical RAM usage? ~330 MB. The crash has nothing to do with memory pressure. It is
pure **virtual address space exhaustion**.

**Reproducible command sequence** (requires Podman or Docker):

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey && REPO_ROOT="$(pwd)"
pixi install
podman build -t projectodyssey:repro .
# Crash:
podman run --rm -v "$REPO_ROOT:$REPO_ROOT:z" --user "$(id -u):$(id -g)" projectodyssey:repro bash -c "ulimit -v 3500000 && MODULAR_HOME=$REPO_ROOT/.pixi/envs/default/share/max $REPO_ROOT/.pixi/envs/default/bin/mojo run $REPO_ROOT/repro/repro_hello.mojo"
# Pass:
podman run --rm -v "$REPO_ROOT:$REPO_ROOT:z" --user "$(id -u):$(id -g)" projectodyssey:repro bash -c "ulimit -v 4000000 && MODULAR_HOME=$REPO_ROOT/.pixi/envs/default/share/max $REPO_ROOT/.pixi/envs/default/bin/mojo run $REPO_ROOT/repro/repro_hello.mojo"
```

Filed upstream: [modular/modular#6433](https://github.com/modular/modular/issues/6433)

---

## Prologue: Sixteen PRs Stuck in BLOCKED

The investigation started with a different problem: 16 open PRs, all in
`MERGEABLE | BLOCKED`, none able to auto-merge. The same required CI checks had been
failing on `main` for 10 consecutive pushes. Even a documentation-only PR was stuck.

The pattern pointed to a systemic root cause, not 16 individual bugs. After reading the
logs and running `/mnemosyne:advise` for prior skills, three crash categories emerged:

| Category | Crash | Status |
| --- | --- | --- |
| 1 | ASAP destruction + `UnsafePointer.bitcast` UAF | Fixed (ADR-013, March 2026) |
| 2 | Docker UID mismatch → container permission fault | Fixed (PR #5252) |
| 3 | JIT compilation volume crash | Under investigation |

Categories 1 and 2 had clean fixes. Category 3 kept evading a definitive root cause.

The Mnemosyne skill `mojo-jit-crash-retry` (v3.2.0) was explicit: "v3.x **reverses**
v2.2.0 — do not bump retry count, do import audit instead." ADR-015 was already written
documenting this decision. The import audit had been executed. And CI was still failing.

That contradiction was the starting signal for a deeper investigation.

---

## Act 1: The Import Chain Theory (Wrong)

### The Original Hypothesis

The CI failures all looked the same: a crash in `libKGENCompilerRTShared.so` before any
test output was produced. This timing — before output — is the hallmark of a compilation
failure rather than a runtime failure. The working theory was:

> Long module-level import chains force the JIT to compile a large transitive closure of
> symbols before executing any test. Files with `from projectodyssey.core.loss_utils import ...`
> at module level trigger `elementwise.mojo` (1650 lines) and `dtype_dispatch.mojo`
> (1520 lines) to be compiled upfront. This compilation volume overwhelms the JIT.

The fix was per-function imports: move all `from projectodyssey.core import ...` statements into
function bodies so they are compiled on-demand rather than all at once.

This fix was applied to `reduction.mojo`, `conv.mojo`, `pooling.mojo`, `matrix.mojo`,
and `loss_utils.mojo` in PR #5259.

### The Problem

After the import audit was complete and CI ran on the fixed branch, the failures
continued. Not with exactly the same frequency, but they still occurred.

The import theory couldn't explain something fundamental: **the crashes were
non-deterministic.** Sometimes the same test file passed. Sometimes it crashed. Module-
level import chains would cause deterministic failures — the same file always triggers
the same compilation path. But these crashes appeared and disappeared across CI runs.

Non-determinism is a race condition signal. Something outside the test file itself was
varying between runs.

---

## Act 2: The Monomorphization Theory (Also Wrong)

### Parametric Functions × DType Variants

The next theory was that the crash came from the number of JIT monomorphizations. Mojo
compiles parametric functions (`fn foo[dtype: DType](...) -> ...`) separately for each
concrete type combination they're called with. A function called with 10 DType variants
produces 10 compiled specializations. If the test file had many parametric functions
and many type variants, the total monomorphization count could push the JIT past some
internal limit.

This theory was testable with a self-contained reproducer. No external libraries needed.

```mojo
# 14 parametric functions × 10 DType variants = 140 monomorphizations
fn generic_op_0[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Scalar[dtype]:
    var acc = Scalar[dtype](0)
    for i in range(n):
        acc += ptr[i]
    return acc
# ... repeated for fn generic_op_1 through fn generic_op_13

def main() raises:
    var a_0 = UnsafePointer[Float32].alloc(8)
    # ... call each fn × each dtype
```

Result: **all PASS.** 140 monomorphizations compiled and ran without incident.

Pushed to 300 monomorphizations (30 functions × 10 dtypes). Still PASS.

### Line Count

Maybe raw compilation work — not structure — was the trigger. Generated 1000 concrete
(non-parametric) functions, approximately 8000 lines total.

```mojo
def concrete_fn_1(x: Float32) -> Float32:
    var a = x * Float32(1)
    var b = a + Float32(2)
    var c = b - Float32(2)
    var d = c * Float32(0.5)
    var e = d + Float32(0.1)
    return e
# ... repeated for concrete_fn_2 through concrete_fn_1000
```

Result: **all PASS.** 8000 lines compiled without incident.

The data was accumulating in an uncomfortable direction: nothing that could vary between
test files seemed to matter.

---

## Act 3: Following the Non-Determinism

### The Shift in Thinking

If the content of the test file doesn't determine whether the crash happens, what does?

The crashes were non-deterministic across CI runs. The same test file crashed sometimes
and passed other times. This isn't what you see when compilation volume overflows an
internal buffer — that would be deterministic. This is what you see when something
*external* to the test file varies: runner load, job scheduling, concurrent process
count.

GitHub Actions free-tier runners have 7 GB RAM and 2 cores. CI workflows use a matrix
strategy — multiple test groups run in parallel. When two or more `mojo run` processes
overlap in time on the same runner, they compete for shared resources.

The question was which shared resource they were competing for.

### Measuring What mojo Actually Uses

The investigation moved from "what does the test file contain" to "what does the mojo
process consume." Set up a `/proc/$PID/status` polling loop inside the container:

```bash
export MODULAR_HOME=/workspace/.pixi/envs/default/share/max
export MOJO=/workspace/.pixi/envs/default/bin/mojo

# Poll memory metrics while mojo runs
mojo run /tmp/test_hello.mojo &
PID=$!
while kill -0 $PID 2>/dev/null; do
    grep -E "VmPeak|VmRSS|VmSize" /proc/$PID/status 2>/dev/null
    sleep 0.1
done
wait $PID
```

The output was unexpected:

```text
VmPeak:  3583024 kB   ← 3.5 GB virtual peak
VmRSS:    321408 kB   ← 314 MB physical RAM
VmSize:  2847660 kB   ← 2.8 GB virtual current
```

For `def main(): print("hello")`.

Not a complex test file. Not a file with imports. Just hello world.

**3.5 GB virtual, 314 MB physical.**

### The ulimit Binary Search

Virtual address space has a hard per-process limit. If the `mojo` process tries to reserve
more virtual space than the limit allows, the kernel kills it with a signal — and
`libKGENCompilerRTShared.so` is the library that was executing when the kill arrives.

`ulimit -v` sets the virtual address limit. Binary search on the threshold:

| ulimit -v (kB) | Virtual Limit | Result |
| --- | --- | --- |
| 2929000 | 2.86 GB | CRASH |
| 3200000 | 3.12 GB | CRASH |
| 3417000 | 3.34 GB | CRASH |
| 3613000 | 3.53 GB | PASS |
| 4000000 | 3.91 GB | PASS |

The crash threshold is between 3.34 GB and 3.53 GB. The process needs approximately
**3.5 GB of virtual address space** to initialize — before it compiles a single function.

This is the minimal reproducer (see `repro/issues/jit-virtual-memory-exhaustion.md` for
the full self-contained command sequence including clone, pixi install, and container build):

```bash
ulimit -v 3500000 && mojo run repro/repro_hello.mojo  # crash
```

Works on any Mojo file. Even an empty one.

---

## The Root Cause

The Mojo JIT compiler (via `libKGENCompilerRTShared.so` and related LLVM infrastructure)
uses **eager virtual page reservation**: it maps a large region of virtual address space
at startup, before any compilation begins. The mapped pages are not populated with
physical memory — `VmRSS` stays low — but the virtual reservation happens immediately
and cannot be avoided.

This is a known pattern in LLVM-based compilers: the JIT arena is pre-mapped to avoid
fragmentation during code emission. The question is *how large* that arena is and whether
it is tunable.

For Mojo 0.26.3, the answer is: **~3.6 GB, not tunable from outside the binary.**

### Why It Causes CI Crashes

GitHub Actions free-tier runner:

- Total RAM: 7 GB
- OS + runner agent + pixi env: ~2-3 GB consumed at baseline
- Available virtual address space for user processes: ~4-5 GB
- `mojo` processes needed simultaneously: 2 (matrix jobs run in parallel)
- Virtual space per `mojo` invocation: ~3.6 GB
- Combined demand at peak overlap: **7.2 GB → exceeds 7 GB → OOM → crash**

The crash is non-deterministic because it depends on whether two `mojo` processes happen
to be in their startup phase at the same time. If one finishes before the other starts,
both fit. If they overlap, one is killed.

### Why Per-Function Imports Help (Indirectly)

Per-function imports were not wrong. They still help — but for a different reason than
originally believed.

Module-level imports force the JIT to compile a large transitive closure of symbols
*before any test output*, extending the duration of the `mojo` process's lifetime.
Per-function imports reduce this upfront compilation, shortening the window during which
the process holds 3.6 GB of virtual address space.

On CI with parallel matrix jobs, shorter process lifetime means fewer overlapping
processes at peak. The fix reduces collision probability, it does not eliminate the
root cause.

---

## What Was Falsified

| Variable | Effect on VmPeak | Effect on crash threshold |
| --- | --- | --- |
| Monomorphization count (1–300) | None (~3.5 GB always) | None |
| Function count (1–1000 concrete fns) | None | None |
| Module-level vs per-function imports | None | None |
| Source file line count | None | None |
| Cross-module import chain depth | None | None |

Only one variable correlated with the crash: **whether the available virtual address
space on the runner was below ~3.5 GB at the moment the `mojo` process started.**

---

## The Fix

### Immediate: max-parallel Constraint

The CI matrix runs test groups in parallel. Adding `max-parallel: 2` limits simultaneous
`mojo` processes on a single runner, keeping the combined virtual demand within the 7 GB
runner budget:

```yaml
strategy:
  fail-fast: false
  max-parallel: 2  # Prevent concurrent mojo processes from exhausting virtual address space.
                   # Each mojo invocation maps ~3.6 GB virtual (see repro/issues/jit-virtual-memory-exhaustion.md).
                   # With 7 GB total on free-tier runners, two concurrent processes fit (7.2 GB demand).
                   # Three or more concurrent processes cause OOM crashes in libKGENCompilerRTShared.so.
  matrix:
    test-group:
```

This is a conservative cap. Two processes × 3.6 GB = 7.2 GB, which is tight but within
the 7 GB bound when accounting for the fact that VmPeak is a brief spike during
initialization and not sustained throughout the entire process lifetime.

### Upstream: Filed modular/modular#6433

The real fix is in Mojo's JIT initialization. Either the virtual pre-mapping size should
be reduced, or it should be documented as a minimum system requirement. The issue
includes:

- Minimal reproducer (`ulimit -v 3500000 && mojo run hello.mojo`)
- VmPeak / VmRSS measurement data
- Binary search table for crash threshold
- Relationship to closed issue #6187 (different crash, same library)
- Suggested fix: reduce pre-mapped arena or expose an environment variable

---

## Epilogue: The Three-Month Pattern

Day 53 documented the `UnsafePointer.bitcast` use-after-free: a compiler bug where ASAP
destruction freed a tensor before a `bitcast` pointer finished using it.

Day 61 documented the `AnyTensor.__del__` bad-free: a one-line bug where slice views
were freed at the wrong address.

Today's crash lives in the same library — `libKGENCompilerRTShared.so` — and produces
the same signal. But it is a third distinct problem: not a use-after-free, not a bad-free,
but an OOM kill during JIT initialization on memory-constrained runners.

Every investigation of this library has ended the same way: the crash was never what the
first theory said it was. The pattern now has a name:

> When `libKGENCompilerRTShared.so` crashes, check the environment before assuming the
> code is wrong. Three of the last four crashes in this library had nothing to do with
> the test file.

The first conclusion should always be: the crash is real, the cause is probably not
what you think it is, and the way to find out is to measure.

---

## Reproducers and Links

- **Upstream bug:** [modular/modular#6433](https://github.com/modular/modular/issues/6433)
- **Issue template:** [`repro/issues/jit-virtual-memory-exhaustion.md`](/repro/issues/jit-virtual-memory-exhaustion.md)
- **Investigation script:** [`repro/investigate_import_threshold.sh`](/repro/investigate_import_threshold.sh)
- **Repro files:** [`repro/repro_parametric_monomorphization_crash.mojo`](/repro/repro_parametric_monomorphization_crash.mojo),
  [`repro/repro_module_import_crash.mojo`](/repro/repro_module_import_crash.mojo)
- **CI fix PR:** [#5260](https://github.com/HomericIntelligence/Odyssey/pull/5260)
