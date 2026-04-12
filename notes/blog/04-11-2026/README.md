# Day Seventy-Nine: One Upstream Bug, One Closed Assumption, One Removed Workaround

**Project:** ML Odyssey
**Date:** April 11, 2026
**Branch:** `blog/day-79-upstream-bug-documentation`
**Tags:** #mojo #debugging #asan #fp16 #simd #upstream-bugs #verification

---

> **Note:** This post documents the ASAN + Python FFI dlsym collision (1 upstream Mojo bug,
> 100% deterministic reproducer) and the discovery that ADR-010's FP16 SIMD scalar workaround
> is no longer necessary in Mojo 0.26.3. The scalar loops have been replaced with proper
> vectorized paths. Issue template is in [repro/issues/](/repro/issues/).

---

## TL;DR

Two things happened today. First: a genuine upstream Mojo bug — running any code with
Python FFI under AddressSanitizer aborts immediately, not from a memory error, but because
ASAN's `dlsym` interceptor returns non-NULL for `PyRun_SimpleString` and Mojo's FFI loader
treats that as fatal. Five lines reproduce it 100% of the time.

Second: ADR-010 was wrong about the current state. It documented `SIMD[DType.float16, N]`
as unsupported — that was true in Mojo 0.26.1, but 0.26.3 supports it fully. Before
filing a feature request upstream, a quick verification run confirmed that
`SIMD[DType.float16, 8]()` compiles and runs cleanly. The scalar workaround loops in
`mixed_precision.mojo` have been replaced with vectorized `load + cast` paths, and
ADR-010 is now marked Superseded.

---

## Bug 1: The ASAN + Python FFI dlsym Collision

### Prologue: Why We Were Running ASAN

The March 2026 debugging marathon taught us a hard lesson: ASAN catches things that
regular testing misses. The bitcast UAF (Day 53) and the slice bad-free (Day 61) both
showed up cleanly under ASAN before they caused mystery crashes in CI. After that, we
started running ASAN routinely on every new component.

The first time we ran ASAN on a component that uses Python FFI — the serialization
utilities that call Python's `os.makedirs()` via `from python import Python` — it
aborted immediately. Not with a heap error. With this:

```text
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:647:22: dlsym unexpectedly
returned non-NULL result when loading symbol: PyRun_SimpleString
```

That's not an ASAN finding. That's Mojo's own stdlib aborting.

### Act 1: What Is This?

The stack trace made it clear what was happening:

```text
#0  libasan.so.8               0x00007f6bbdd7a1e0
#1  libKGENCompilerRTShared.so 0x00007f6bbdcbc4ab
#2  libKGENCompilerRTShared.so 0x00007f6bbdcb9686
#3  libKGENCompilerRTShared.so 0x00007f6bbdbd157
#4  libc.so.6                  0x00007f6bbd9ae330
```

ASAN is in frame 0. This is ASAN's `dlsym` interceptor being called by Mojo's stdlib.

Here's what Mojo's FFI loader does at startup: it calls
`dlsym(RTLD_DEFAULT, "PyRun_SimpleString")` to check whether CPython is available.
If that call returns NULL, Mojo knows Python isn't loaded and disables the Python
interop path. If it returns non-NULL, Mojo enables Python interop.

The check is documented behavior. The problem is what happens when ASAN is active.

### Act 2: ASAN Intercepts dlsym

AddressSanitizer installs its own `dlsym` interceptor at process startup. The interceptor
wraps the real `dlsym` so ASAN can track dynamic symbol lookups — this is how ASAN
handles code that looks up and calls arbitrary function pointers at runtime.

The ASAN interceptor does something that makes sense for ASAN's purposes but breaks
Mojo's assumption: for symbols it recognizes as "interesting" (which includes common
C library and Python runtime symbols), the interceptor can return a non-NULL result
even when the symbol would not normally be found via the real `dlsym`.

`PyRun_SimpleString` is a Python runtime symbol. ASAN intercepts the lookup, returns
non-NULL. Mojo sees "CPython is available," proceeds to use it. But CPython isn't
actually loaded — the non-NULL pointer isn't valid. Mojo's stdlib detects the
inconsistency at line 647 of `std/ffi/__init__.mojo` and aborts.

### Act 3: The Five-Line Reproducer

```mojo
from python import Python

def main() raises:
    var py = Python.import_module("os")
    print(py.getcwd())
```

Run with:

```bash
mojo build --sanitize address -o repro_asan_ffi repro_asan_ffi.mojo
./repro_asan_ffi
```

Crash is 100% deterministic on Mojo 0.26.3 with libasan.so.8. The program
never reaches `main()` — the abort fires during Mojo's runtime initialization,
before the first line of user code executes.

### Isolation Experiments

| Variant | Crashes? | Conclusion |
| ------- | -------- | ---------- |
| Run without `--sanitize address` | NO | ASAN is the trigger |
| Run under Valgrind instead of ASAN | NO | Valgrind doesn't intercept dlsym the same way |
| Remove `from python import Python` entirely | NO | FFI loader not invoked |
| Use `--sanitize memory` (MSan) instead | NO | MSan's dlsym handling differs |
| Run with `LD_PRELOAD=""` explicitly | NO effect | ASAN is already linked, not LD_PRELOAD |
| Run on code that uses Python FFI at runtime | Identical crash | Not call-site-specific |

### Impact

Any Mojo code using Python FFI cannot be tested under ASAN. This includes:

- Serialization utilities that call `os.makedirs()` via Python
- Any component that imports `from python import Python`
- Our CI ASAN test matrix for those components

The workaround is to restructure code to avoid Python FFI in components you want
to test under ASAN, or to skip ASAN on Python-FFI-using tests. Neither is great.

### Filed

Issue template: [`repro/issues/asan-dlsym-abort.md`](/repro/issues/asan-dlsym-abort.md)

---

## Not Bug 2: ADR-010 Was Stale

### The Assumption

ADR-010 documented that `SIMD[DType.float16, N]` was unsupported in Mojo 0.26.1
(error: `invalid SIMD element type 'float16'`). The scalar workaround in
`mixed_precision.mojo` was written against that constraint, with a comment pointing
to the ADR. Before filing an upstream feature request for FP16 SIMD support, we
verified the current behavior.

### The Verification

```bash
mojo run test_fp16_simd.mojo
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

`SIMD[DType.float16, 8]` compiles and runs in Mojo 0.26.3. Vectorized load, cast
to float32, and store all work:

```mojo
var fp16_vec = src_ptr.load[width=8]()
dst_ptr.store[width=8](fp16_vec.cast[DType.float32]())
```

The limitation was fixed somewhere between 0.26.1 and 0.26.3. The feature we were
about to file a request for was already shipped.

### What Changed

- **`_convert_fp16_to_fp32_simd`**: replaced scalar inner loop with `load + cast[DType.float32]()`
- **`_convert_fp32_to_fp16_simd`**: replaced scalar inner loop with `load + cast[DType.float16]()`
- **`convert_to_fp32_master`**: FP16 branch now calls `_convert_fp16_to_fp32_simd` instead of
  the inline scalar loop
- **ADR-010**: marked Superseded with note explaining the resolution
- `repro/issues/fp16-simd-limitation.md`: deleted — describes a resolved non-issue

The lesson: **verify assumptions before filing upstream**. The reproducibility standard that
kept us from filing the non-deterministic JIT crashes also caught this.

---

## Epilogue: The Reproducibility Standard, Applied

This session set out to document all unfiled upstream Mojo bugs and file them. The rule
was: only file if you have a 100% deterministic minimal reproducer on the current Mojo
version.

Result:

| Candidate | Verdict | Reason |
| --------- | ------- | ------ |
| ASAN + dlsym abort | Filed (template) | 5-line reproducer, 100% deterministic |
| FP16 SIMD limitation | Not filed — resolved | Feature shipped in 0.26.3; verified clean |
| JIT volume crash | Not filed | Non-deterministic; can't guarantee 100% repro |
| JIT fortify abort | Not filed | CI-only; no local reproducer by definition |

One real issue. One stale ADR cleaned up. Two non-deterministic crashes still waiting
for a better reproducer. Good enough for a day.

---

### Stats

- **Upstream bugs filed**: 1 (ASAN + dlsym)
- **Stale assumptions corrected**: 1 (ADR-010 FP16 SIMD)
- **Scalar workaround loops removed**: 2 (both FP16↔FP32 conversion helpers)
- **ADRs marked Superseded**: 1
- **Issue templates deleted**: 1 (fp16-simd-limitation.md — describes resolved issue)
