# Day Sixty-One: The Slice View Bad-Free -- When "Flaky JIT Crashes" Were Our Bug All Along

**Project:** ML Odyssey
**Date:** March 24, 2026
**Tags:** #mojo #debugging #asan #bad-free #slice #view #ci-flakiness

---

> **Note:** This post documents how a systematic CI investigation revealed that a "flaky JIT crash"
> was actually a one-line bug in AnyTensor.\_\_del\_\_. The 3-line reproducer and ASAN output are in
> [artifacts](../03-16-2026/artifacts/).

---

## TL;DR

`test_training_loop.mojo` crashed "flakily" in CI with the same `libKGENCompilerRTShared.so`
abort as Day 53's bitcast UAF. ASAN revealed a completely different bug: `slice()` creates
views with offset `_data` pointers (line 754), but `__del__` (line 491) calls `pooled_free`
unconditionally -- freeing an address that was never returned by `malloc`. Fix: check
`_is_view` before freeing. Not a Mojo bug -- our bug.

---

## Prologue: The Investigation That Started It All

The user asked to investigate flaky CI. The first step was `/advise`, which returned prior
skills about JIT crash patterns, misdiagnosis warnings, and retry strategies. The skills
registry had learned from Day 53 that crashes labeled "JIT" are often something else
entirely.

Three parallel exploration sub-agents launched:

1. **CI workflow agent** -- examined GitHub Actions runs, identified failure categories
2. **Recent failures agent** -- pulled logs from the last 10 CI runs, classified crash types
3. **Test isolation agent** -- checked which test files were involved in each failure

The results came back in about five minutes. Four distinct CI failures, but only one matched
the "JIT crash" signature. The other three were entirely different problems.

---

## Act 1: The Infrastructure Layer

Before touching the JIT crash, the investigation uncovered that CI was 100% blocked by an
infrastructure problem: **Docker Hub was returning 504 Gateway Timeout errors**, which broke
the Podman image cache that all CI jobs depend on. Every workflow was failing before it could
even pull the container image.

### The Docker Fix

The broken Podman cache was replaced with a tar-based image cache strategy. This unblocked
CI entirely and was unrelated to any test crash -- but it had to be fixed first, because
no CI job could run at all.

### Three Deterministic Test Bugs

With CI running again, the parallel agents identified three test files with deterministic
failures that had nothing to do with JIT crashes:

| Bug | Root Cause | Fix |
| --- | --- | --- |
| f-string syntax error | Mojo f-string interpolation using unsupported expression | Replaced with string concatenation |
| Tuple destructuring | Incorrect unpacking of multi-return function | Fixed destructuring pattern |
| Fast-path ignoring step | Slice fast-path skipped the `step` parameter | Added step to fast-path logic |

These were real bugs, each with a clear fix. None of them were flaky.

### One Remaining Failure

After fixing the infrastructure and the three deterministic bugs, exactly one CI failure
remained: `test_training_loop.mojo` crashing with the `libKGENCompilerRTShared.so` abort
signal. This was the one that had been classified as "flaky JIT crash."

---

## Act 2: Reclassifying the Crash

### The Initial (Wrong) Classification

The crash had been classified as a "JIT compilation overflow" -- the theory was that the
Mojo JIT compiler was running out of internal buffers before producing any test output.
This classification came from the assumption that `libKGENCompilerRTShared.so` crashes
always indicate compiler-level failures.

### Reading the Actual CI Log

Re-reading the actual CI log revealed a critical detail that contradicted the classification:

```text
Starting Training Loop Tests...
  test_single_step_train... OK
  test_multi_step_convergence... OK
  test_batch_gradient_accumulation... OK
  ...
  test_checkpoint_save_restore... OK
  test_learning_rate_schedule...
#0 libKGENCompilerRTShared.so  +0x3cb78b
error: execution crashed
```

The crash was **AFTER 17 tests passed**, not before any output. If this were a JIT
compilation failure, it would crash before the first line of output -- the JIT compiles the
entire file before executing any code. Seventeen tests passing means the JIT succeeded. The
crash was at runtime.

### The Familiar Offset

The crash offset `+0x3cb78b` was the exact same offset from Day 53's bitcast UAF. Same
library, same offset, same crash output format. The initial hypothesis: this is the same
bitcast UAF pattern, manifesting in the training loop tests.

---

## Act 3: ASAN Reveals a Different Bug

### Expected: heap-use-after-free

Based on the Day 53 pattern, the ASAN build was expected to produce a `heap-use-after-free`
report -- the compiler destroying a tensor whose bitcast-derived pointer was still live.

### Actual: bad-free

```bash
pixi run mojo build --sanitize address -g -o /tmp/train_asan test_training_loop.mojo
/tmp/train_asan
```

```text
ERROR: AddressSanitizer: attempting free on address which was not malloc()-d
    #0 in free
    #1 in AnyTensor::__del__
       src/projectodyssey/tensor/any_tensor.mojo:491

0x50200000ab00 is located 512 bytes inside of 1024-byte region
    [0x502000000900, 0x502000000d00)
allocated by thread T0 here:
    #0 in malloc
    #1 in pooled_alloc
```

**"Attempting free on address which was not malloc()-d."** Not a use-after-free. Not a
double-free. A **bad-free**: calling `free()` on a pointer that points into the middle of an
allocation, not to its start.

The freed address `0x50200000ab00` was **512 bytes inside** a 1024-byte allocation that
started at `0x502000000900`. Something was calling `pooled_free` on an offset pointer.

### The Key Experiment: Isolation

Day 53's bitcast UAF required "allocation churn" -- it only triggered after multiple test
functions had created and destroyed enough temporaries to confuse the optimizer's liveness
analysis. The file-splitting workaround from ADR-009 worked precisely because it reduced
per-file churn below the trigger threshold.

If this were the same bug, the crash test should pass when run alone (no prior churn).

**It did not pass.** The crash test, extracted into its own file with zero prior test
functions, triggered ASAN immediately:

```bash
# Single test, no prior churn, still crashes
pixi run mojo build --sanitize address -g -o /tmp/single_asan single_crash_test.mojo
/tmp/single_asan
# ERROR: AddressSanitizer: attempting free on address which was not malloc()-d
```

This **disproved** the bitcast UAF hypothesis. The Day 53 bug required accumulated churn to
trigger. This bug triggers on a single operation with zero history. Different bug, same crash
signature.

---

## The Root Cause

### slice() Creates Views With Offset Pointers

In `src/projectodyssey/tensor/any_tensor.mojo`, the `slice()` method creates a view into an existing
tensor by offsetting the `_data` pointer:

```mojo
# any_tensor.mojo, line 741-754
var offset_elements = start * self._strides[axis]
var dtype_size = self._get_dtype_size()
var offset_bytes = offset_elements * dtype_size

var result = self.copy()
result._is_view = True

# ...

result._data = self._data + offset_bytes  # <-- offset pointer
```

The result tensor's `_data` now points 512 bytes (or whatever `offset_bytes` is) into the
parent tensor's allocation. This is correct for reading and writing elements through the
view -- the view shares the parent's memory.

### \_\_del\_\_ Frees Unconditionally

The destructor at line 478-492:

```mojo
fn __del__(deinit self):
    if self._refcount:
        self._refcount[] -= 1
        if self._refcount[] == 0:
            pooled_free(self._data, self._allocated_size)  # line 491
            self._refcount.free()
```

When the last reference is destroyed, `__del__` calls `pooled_free(self._data, ...)`. But
if `self` is a view created by `slice()`, then `self._data` is an **offset pointer** -- it
points into the middle of the parent's allocation, not to the start. `pooled_free` (which
wraps `free()`) requires the exact pointer that `malloc()` returned. Passing an interior
pointer is undefined behavior.

### The Undefined Behavior Chain

1. `var data = ones([8, 2, 4, 4], DType.float32)` -- allocates 1024 bytes at `0x900`
2. `var batch = data.slice(4, 8)` -- creates view, `batch._data = 0x900 + 512 = 0xb00`
3. `batch.__del__()` -- calls `pooled_free(0xb00, ...)` -- **bad-free: 0xb00 was never malloc'd**

Without ASAN, `free()` on an invalid pointer silently corrupts the heap allocator's internal
metadata. The corruption accumulates across test functions until the metadata is damaged
enough that a subsequent `malloc()` or `free()` call triggers the abort handler in
`libKGENCompilerRTShared.so`.

---

## The 3-Line Reproducer

```mojo
var data = ones([8, 2, 4, 4], DType.float32)
var batch = data.slice(4, 8)
# batch.__del__() -> pooled_free(offset_ptr) -> ASAN: bad-free
```

That is it. Three lines. The view's destructor frees an interior pointer. Every time
`slice()` is used and the resulting view is destroyed, the heap takes silent damage.

---

## Why It Appeared Flaky

Without ASAN, `free()` on an invalid pointer does not crash immediately. It corrupts a few
bytes of heap metadata -- the allocator's free-list pointers, chunk size fields, or boundary
tags. The corruption is invisible until a later allocation or deallocation operation reads
the corrupted metadata and finds it inconsistent.

The delay between the bad-free and the crash depends on:

- **Heap layout** -- which metadata bytes were corrupted
- **Subsequent allocation pattern** -- which operations first touch the corrupted region
- **Allocator implementation** -- glibc's ptmalloc2 checks metadata at chunk boundaries

Different CI runners have different heap layouts (ASLR, different prior allocations from
container setup). The same test file might crash after test 15 on one runner and after test
19 on another -- or pass entirely if the corrupted metadata is never read before the process
exits.

This is why the crash appeared "flaky":

- **Same test, same code, different runners** -- different heap layout, different crash point
- **File splitting (ADR-009)** -- fewer tests per file = fewer slice operations = less
  metadata corruption = lower probability of hitting the abort threshold
- **Local vs CI** -- WSL2 and Ubuntu 24.04 Docker have different allocator configurations

ASAN catches the bad-free at the exact moment it happens, regardless of heap layout. With
ASAN, the crash is 100% reproducible, 100% deterministic, and points directly to the buggy
line.

---

## The Fix

One-line change in `__del__`: check `_is_view` before calling `pooled_free`. Views share
their parent's data allocation. Only non-views (tensors that own their memory) should free
the data pointer:

```mojo
fn __del__(deinit self):
    if self._refcount:
        self._refcount[] -= 1
        if self._refcount[] == 0:
            if not self._is_view:
                pooled_free(self._data, self._allocated_size)
            self._refcount.free()
```

Views still participate in reference counting -- they decrement the refcount when destroyed.
But they never free the data pointer, because they hold an offset pointer that was never
returned by `malloc`. The parent tensor (the non-view) will free the data when its own
refcount reaches zero.

---

## Lessons Learned

1. **Read the actual CI logs** -- the crash was labeled "before any output" but the log
   clearly showed 17 tests passing. The classification was wrong, and everything downstream
   of it was wrong too.

2. **ASAN catches what "flaky" hides** -- the crash is 100% reproducible with ASAN. The
   "flakiness" was just heap corruption manifesting at different points depending on
   allocator state. ASAN catches the root cause at the moment it happens.

3. **"Same crash signature" does not mean "same root cause"** -- the same
   `libKGENCompilerRTShared.so+0x3cb78b` offset appeared in Day 53's bitcast UAF and
   today's bad-free. Same library, same offset, completely different bugs. The offset is
   just the crash handler entry point.

4. **Test the crash in isolation** -- if the crash triggers with a single test function and
   zero prior churn, then accumulated allocation churn is not the cause. This one experiment
   disproved the bitcast UAF hypothesis in seconds.

5. **The simplest explanation is often right** -- before blaming the compiler, check
   `__del__`. A destructor freeing a pointer it doesn't own is one of the most common
   memory bugs in systems programming. ASAN's "bad-free" message made it obvious.

---

## Timeline

| Step | Finding | Time |
| --- | --- | --- |
| /advise | Prior skills: JIT crash patterns, misdiagnosis warnings | 0 min |
| 3 parallel explore agents | CI blocked by Docker 504, 3 deterministic bugs, 1 "JIT" crash | +5 min |
| Fix Docker cache | Tar-based image cache, CI unblocked | +7 min |
| Fix 3 deterministic bugs | f-string, tuple destructuring, fast-path step | +8 min |
| Re-read CI log | Crash AFTER 17 tests, not before -- same offset as Day 53 | +10 min |
| ASAN build + run | "bad-free" not "heap-use-after-free" -- different bug | +15 min |
| Test crash alone | Still crashes -- no prior churn needed | +17 min |
| Read slice() code | `_data` offset without `__del__` check -- found it | +20 min |
| 3-line reproducer | 100% reproducible, zero dependencies | +22 min |

---

## Relationship to Day 53

This is the **third distinct bug** producing identical `libKGENCompilerRTShared.so` crash
signatures:

| # | Bug | Root Cause | ASAN Report | Discovery |
| --- | --- | --- | --- | --- |
| 1 | Bitcast UAF (Day 53, Act 1) | Compiler destroys tensor before bitcast write completes | `heap-use-after-free` | March 16, 2026 |
| 2 | ADR-009 threshold crashes | Same bitcast UAF, masked by file splitting | (same as above) | December 2025 |
| 3 | Slice view bad-free (today) | `__del__` frees offset pointer from `slice()` | `attempting free on address not malloc()-d` | March 24, 2026 |

All three produce the same crash output:

```text
#0 libKGENCompilerRTShared.so  +0x3cb78b
#1 libKGENCompilerRTShared.so  +0x3c93c6
#2 libKGENCompilerRTShared.so  +0x3cc397
error: execution crashed
```

Only ASAN distinguishes them. Without ASAN, every heap corruption bug in Mojo looks
identical -- the crash handler offset is a property of the Mojo runtime, not the bug.

### How ADR-009 Masked Both Bugs

ADR-009 split test files to fewer than 10 tests each. This workaround was designed for the
bitcast UAF (reducing allocation churn below the trigger threshold), but it also
accidentally reduced the number of `slice()` operations per file -- meaning fewer bad-frees
per process, meaning less heap corruption per file, meaning lower probability of hitting the
abort threshold.

The file-splitting was treating two diseases without knowing either diagnosis.

---

## References

### Day 53 Blog Post

- [Day Fifty-Three: The UnsafePointer.bitcast Use-After-Free](../03-16-2026/README.md) --
  the bitcast UAF investigation that established the ASAN methodology used here

### Project Files

- `src/projectodyssey/tensor/any_tensor.mojo` lines 478-492 -- the `__del__` destructor
- `src/projectodyssey/tensor/any_tensor.mojo` lines 741-754 -- the `slice()` view creation
- [ADR-009: Heap corruption workaround](/docs/adr/ADR-009-heap-corruption-workaround.md) --
  file-splitting workaround that masked both bugs

### Mojo Documentation

- [ASAP destruction](https://docs.modular.com/mojo/manual/lifecycle/death/) --
  Mojo's eager value destruction policy
- [UnsafePointer](https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/) --
  raw pointer semantics and origin tracking

### Environment

- Mojo 0.26.1.0 (156d3ac6)
- Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
- GLIBC 2.39, Ubuntu 24.04
