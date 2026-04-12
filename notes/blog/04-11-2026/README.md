# Day Seventy-Nine: Two More Mojo Bugs — ASAN Lies and FP16 SIMD Silence

**Project:** ML Odyssey
**Date:** April 11, 2026
**Branch:** `blog/day-79-upstream-bug-documentation`
**Tags:** #mojo #debugging #asan #fp16 #simd #upstream-bugs #compiler-limitations

---

> **Note:** This post documents two separate Mojo upstream bugs discovered during
> ProjectOdyssey development. Both have minimal reproducers. One causes an abort
> when ASAN is active; the other silently blocks FP16 vectorization with no
> workaround message. Issue templates are in [repro/issues/](/repro/issues/).

---

## TL;DR

Two more Mojo bugs worth filing. First: running any Mojo code with Python FFI under
AddressSanitizer causes an immediate abort — not from a memory error, but because
ASAN's `dlsym` interceptor returns non-NULL for `PyRun_SimpleString` when CPython
isn't loaded, and Mojo's FFI loader treats that as a fatal inconsistency. Five lines
of code reproduce it 100% of the time.

Second: `SIMD[DType.float16, N]` doesn't compile in Mojo 0.26.x. No error message
tells you there's a workaround or that this type is planned. You just get a type error
and have to fall back to scalar loops, taking a 10-15x performance hit on FP16
conversion paths.

Neither is a showstopper. Both are annoying enough to document.

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

This is one of the shortest reproducers in the project:

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

## Bug 2: FP16 SIMD — The Silent Missing Type

### Prologue: Mixed-Precision Training

Mixed-precision training (FP16 model weights, FP32 master weights, FP32 optimizer state)
is a standard technique for training on hardware with FP16 tensor cores. The implementation
in `shared/training/mixed_precision.mojo` requires two conversion functions:

- `convert_to_fp32_master()`: copies FP16 model params → FP32 master weights before the
  optimizer step
- `update_model_from_master()`: copies FP32 master weights → FP16 model params after update

The natural implementation uses SIMD vectorization for ~4x throughput. Every other dtype
path in the project uses vectorized load/store. FP16 should be no different.

### Act 1: The Compile Error

```mojo
alias simd_width = simdwidthof[DType.float16]()
var vec: SIMD[DType.float16, simd_width]  # <-- compile error
```

Error:

```text
error: invalid SIMD element type 'float16'
note: SIMD element types must be numeric but float16 is not supported
```

The compiler rejects `SIMD[DType.float16, N]` with a type error. Not "unimplemented" —
"not supported." There's no documentation note, no migration guide, no "use this instead"
message.

### Act 2: Searching for a Workaround

The Mojo documentation for `SIMD` lists supported element types. `float16` is not in the
list. The `DType` enum includes `float16` as a valid value (it's a valid element type for
`UnsafePointer` and scalar operations), but the SIMD type system doesn't support it as a
lane type.

Experiments:

| Approach | Result |
| -------- | ------ |
| `SIMD[DType.float16, simd_width]` | Compile error: invalid element type |
| `SIMD[DType.float16, 1]` | Compile error: same error even with width=1 |
| `Float16` scalar ops | Works — element-by-element conversion is valid |
| `DType.bfloat16` in SIMD | Also fails: same constraint |
| `DType.float32` SIMD (for comparison) | Works — FP32 SIMD path confirmed correct |

The scalar path works: `dst_ptr[i] = Float32(src_ptr[i])` element by element. It's
10-15x slower than the SIMD path would be for FP32, but correct.

### Act 3: The One-Line Insight

The limitation isn't a bug in Mojo's SIMD implementation — it's a missing feature in
the hardware abstraction layer. Modern x86 CPUs support `_mm256_cvtph_ps` (F16C
extension) for converting 8 FP16 values to FP32 in a single instruction. AVX-512FP16
goes further with direct FP16 arithmetic.

Mojo's SIMD type system doesn't expose `float16` as a lane type yet. The scalar path
works correctly and uses the CPU's scalar FP16→FP32 conversion instructions, just one
element at a time.

This is a feature request, not a bug. The behavior is consistent (always fails,
never works partially). The workaround (scalar loop) is well-defined and correct.
The performance penalty is bounded and acceptable for training (conversion happens
once per optimizer step, not per forward pass).

### Minimal Reproducer

```mojo
def main():
    # This line fails to compile in Mojo 0.26.3
    var v = SIMD[DType.float16, 8]()
    print(v)
```

```bash
mojo run repro_fp16_simd.mojo
# error: invalid SIMD element type 'float16'
```

### The Workaround

```mojo
# FP16 SIMD blocked by Mojo v0.26.x limitation; using scalar loop.
# See docs/adr/ADR-010-fp16-simd-mojo-limitation.md
var src_ptr = params._data.bitcast[Float16]()
var dst_ptr = result._data.bitcast[Float32]()
for i in range(size):
    dst_ptr[i] = Float32(src_ptr[i])
```

Performance: ~10-15x slower than the equivalent SIMD FP32→FP32 path. Acceptable for
optimizer steps, which run once per batch, not once per element.

### Filed

Feature request template: [`repro/issues/fp16-simd-limitation.md`](/repro/issues/fp16-simd-limitation.md)

---

## Epilogue: The Reproducibility Standard

The March 2026 retrospective was unsparing about how many bugs were dismissed or
misclassified over three months. One lesson from that: **only file upstream issues
when you have a deterministic, minimal reproducer that works on the current Mojo version.**

The JIT volume crash and the CI-only fortify abort don't meet that bar yet. They're
documented in `repro/issues/` but not filed — the reproducers are non-deterministic or
environment-specific. If we can isolate them to something smaller and more reliable,
we'll file them.

These two meet the bar:

- ASAN + dlsym: 5-line reproducer, 100% deterministic, crashes on every run
- FP16 SIMD: one-line compile error, 100% deterministic, fails on every compile

Short investigation. Two clean issues. That's a good day.

---

### Stats

- **Bugs documented**: 2
- **Issue templates created/updated**: 2
- **Lines of reproducer code**: 5 (ASAN) + 3 (FP16)
- **Time to minimal reproducer**: ~15 minutes each (both were already documented)
- **Bugs fixed by us**: 0 (both are upstream; workarounds already in codebase)
