# Mojo 1.0.0 Regressions: Validated Failure Modes

**Status**: Investigation complete — failures root-caused, each validated by running the
**identical reproducer** under both `mojo==1.0.0b2` (release beta, `2cf4d08a`) and
`mojo==1.0.0` (stable, `ed45d567`) in the odyssey-dev container.

**Headline**: every failure below **passes on 1.0.0b2 and fails on 1.0.0 stable**.

Reproducers live in `docs/dev/reproducers/`. Each is self-contained, uses only
the public stdlib, and is the smallest code shape that demonstrates the bug.

---

## Failure Mode 1: Return-move runs the moved-from source's `__deinit__` → use-after-free

**Severity**: Critical (silent memory corruption; wrong tensor values + crashes across the
core test suite).

### Symptom

A struct that owns a buffer, has a custom `deinit move` constructor, and is **returned by
value** (`return r^`) has its *moved-from source's* field deinit executed **after the
function returns but before the caller reads the destination**. The source's deinit frees
the buffer that the destination now owns → use-after-free.

Simple shapes (parameter moves, `var x = src`) are NOT affected — only **return moves**.

### Cross-version evidence (identical file `docs/dev/reproducers/repro_uaf_return_move.mojo`)

```text
$ mojo run repro_uaf_return_move.mojo          # 1.0.0b2
values: 0 1                                     # ✅ correct — moved-from deinit skipped

$ mojo run repro_uaf_return_move.mojo          # 1.0.0 stable
    deinit freeing ptr                          # ❌ moved-from deinit RUNS before the read
values: 0 1                                     #    (values correct only by luck — freed
                                                #     block not yet reused by allocator)
```

AddressSanitizer makes the UAF unambiguous on stable; on b2 the same binary reports no
use-after-free:

```text
$ mojo build --sanitize address repro_uaf_return_move.mojo -o /tmp/r && /tmp/r   # 1.0.0b2
SUMMARY: AddressSanitizer: 16 byte(s) leaked in 1 allocation(s).                  # no UAF

$ mojo build --sanitize address repro_uaf_return_move.mojo -o /tmp/r && /tmp/r   # 1.0.0 stable
==ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000138
READ of size 8 ... in main
```

Instrumented run (1.0.0 stable) shows the deinit interleaved *before* the destination read:

```text
main: calling make
main: got p, p.a.ptr = ...
main: reading via p
    deinit freeing ptr     <- moved-from r's deinit (WRONG: should be skipped)
    deinit freeing ptr
main: values 0 1 0 1       <- UAF read
```

### Root cause

In 1.0.0 the compiler does not reliably skip `__deinit__` of a moved-from value for the
return-move shape. `AnyTensor`'s shared-refcount ownership model (`_refcount` cell shared
between copies) then decrements the refcount twice for one transfer — once by the
moved-from source's deinit, once by the destination's eventual deinit — so the buffer is
freed while the destination still references it.

### Impact on Odyssey

This is the root cause of the core-suite corruption seen during the 1.0.0 migration:

- `tests/odyssey/core/test_arithmetic.mojo` — `test_multiply_backward` reads garbage at
  element 1 (`5.757e-42 !≈ 5.0`-class failures, value varies run-to-run)
- `tests/odyssey/core/test_pooling.mojo` — pooled output bytes corrupted
- `tests/odyssey/core/test_activations.mojo` — gradient-check diffs (`1.9904632568`-class)
- `tests/odyssey/core/test_conv.mojo` — deterministic crash in `List._realloc` during a
  fresh `zeros()` (heap corruption from earlier premature frees)
- Any function returning `GradientPair(grad_a^, grad_b^)` or `Tensor` by value

All of these pass on 1.0.0b2.

### Reproducer

`docs/dev/reproducers/repro_uaf_return_move.mojo` (also the simpler
`docs/dev/reproducers/repro_min.mojo`). Compiles and runs unchanged on both versions.

---

## Failure Mode 2: `Scalar[dt] ** 0.5` fails to compile for non-float64 dtypes

**Severity**: High (compile failure — `mojo: error: failed to run the pass manager`).

### Symptom

Raising a `Scalar[dt]` to the power `0.5` fails to instantiate for `float16`,
`bfloat16`, and `float32`. `float64` works.

### Cross-version evidence (identical file `docs/dev/reproducers/repro_scalar_pow.mojo`)

```text
$ mojo run repro_scalar_pow.mojo        # 1.0.0b2
float16 => 2.0
bfloat16 => 2.0
float32 => 2.0
float64 => 2.000000000000565

$ mojo run repro_scalar_pow.mojo        # 1.0.0 stable
error: failed to run the pass manager   # float16/bfloat16/float32 instantiation fails
note: constraint failed: unsupported type combination   (std/builtin/simd.mojo)
```

### Root cause

The `**` (pow) operator on `Scalar[dt]` routes through a SIMD implementation that rejects
the dtype combination for non-float64 types in 1.0.0. The `std.math.sqrt` function
handles all float dtypes, which is the workaround used in `_sqrt_typed`
(`src/odyssey/core/normalization.mojo`) to keep the suite compiling.

### Impact on Odyssey

`normalization._sqrt_typed` originally used `x**0.5`; this is the compile error that
forced the `std.math.sqrt` change. It does not affect float64-only paths.

### Reproducer

`docs/dev/reproducers/repro_scalar_pow.mojo`. Compiles and runs unchanged on both versions.

---

## Non-bugs ruled out during investigation

These were suspected but verified NOT to be regressions (identical behavior on b2 and
stable):

| Pattern | Result |
| --- | --- |
| `ptr.unsafe_bitcast[T]()[unsafe_offset=i]` read | ✅ works on both |
| `ptr.unsafe_offset(off).unsafe_bitcast[T]()[]` read/write | ✅ works on both |
| Parameter moves (`consume(x)`), implicit last-use moves | ✅ source deinit skipped on both |
| `List[Int]`-field struct return-move | ✅ works on both (List tracks its own moved-from state) |
| `SIMD[dt,1] ** 0.5` (non-Scalar) | ✅ works on both |

The garbage values originally attributed to a "bitcast read miscompile" in the earlier
session are explained by Failure Mode 1 (reads of already-freed buffers), not by a
separate miscompile.

---

## Validation of modular/modular#6445 (KGEN JIT buffer overflow) — NOT A REGRESSION

**Status: does NOT reproduce on 1.0.0b2 or 1.0.0 stable — appears fixed** (consistent with
upstream #6413, the JIT-stability fix that landed in `1.0.0b2.dev2026052506`).

Issue #6445 reported a KGEN JIT **compile-time** crash (`__fortify_fail_abort` in
`libKGENCompilerRTShared.so`) on mojo 0.26.3, CI-only, from: module-level `std.python`
import + struct with `List[String]` field + 6 overloaded `__init__`s + `Dict[String, Value]`.
The reporter (Odyssey) had already seen 10/10 clean on `1.0.0b2.dev2026052506` and suspected
a duplicate of #6413.

Validation performed on the issue's reproducer adapted to 1.0.0 syntax
(`docs/dev/reproducers/repro_kgen_6445.mojo`) and on the actual `tests/configs/` group the
issue claimed to block:

| Check | 1.0.0b2 | 1.0.0 stable |
| --- | --- | --- |
| Reproducer, 10 consecutive runs | ✅ 10/10 | ✅ 10/10 |
| Reproducer, 7 GB `ulimit -v` (CI-like) | ✅ | ✅ prints "KGEN crash did NOT occur" |
| Reproducer, `mojo build` under 2 GB cap | — | ✅ compile succeeds (exit 0) |
| `tests/configs/` group (5 files, `--Werror`) | — | ✅ all pass |
| `test_jit_crash_6413.mojo` canary (200 Python imports) | — | ✅ PASS |

Under an artificial 2 GB `ulimit -v` the program still compiles and starts (prints the
first line) and only then aborts with a **runtime** `alloc.mojo:602 alloc failed: returned
a null pointer` (tcmalloc cannot map under the cap) — not the compile-time
`__fortify_fail_abort` signature from #6445. No `__fortify_fail_abort` /
`libKGENCompilerRTShared` frames appeared in any constrained run.

**Conclusion**: the #6445 crash mode is fixed in current releases; the issue is a candidate
for closure (with a note that it may be a duplicate of #6413). Do NOT file a new upstream
issue for it.

---

## Upstream issue status

- **modular/modular#6187** (heap corruption, `0.26.1`) — closed COMPLETED; different
  signature (crash at alloc, not move semantics).
- **modular/modular#6707** (stale read through `UnsafePointer[MutUntrackedOrigin]`,
  closed NOT_PLANNED "expected behavior") — different mechanism: origin-tracking
  limitation, not a move-semantics violation.
- **modular/modular#6475** (bitcast read in struct method vs external, OPEN) — different
  mechanism (no moves involved).

All failure modes have been filed upstream (2026-08-20/21):

- **Failure Mode 1 (UAF)**: <https://github.com/modular/modular/issues/6939>
- **Failure Mode 2 (compile)**: <https://github.com/modular/modular/issues/6940>
- **Failure Mode 3 (virtual-memory limit aborts instead of degrading gracefully)**:
  <https://github.com/modular/modular/issues/6941>
- **Failure Mode 4 (KGEN JIT runtime crash)**: <https://github.com/modular/modular/issues/6958>

Each issue embeds the full reproducer and the b2-vs-stable evidence. Labels are added by
maintainers during triage (repo restricts label permissions).

**On the 2 GB `ulimit -v` crash (FM3)**: the official Mojo system requirements document a
**8 GiB minimum RAM for Mojo development** (mojolang.org/docs/requirements/), so the
crash itself is a below-minimum condition. The upstream root cause of the large virtual
reservation is #6433 (closed COMPLETED, reservation reduced ~3.6 GB → ~2.78 GB measured);
FM3 (#6941) is the separate failure-mode problem: a hard `abort()` with opaque tcmalloc
output at the limit instead of a clear, actionable error. Measured thresholds (stable
1.0.0, identical on b2): `mojo run` reliably passes at ≥~2.75 GB virtual (non-monotonic
crash bands below, e.g. 2 GB passes but 2.5 GB crashes — tcmalloc 1 GB-aligned mmap
placement), built binary ≥~1.5 GB. Reproducer:
`docs/dev/reproducers/repro_vm_limit_hello.mojo`.

---

## Cross-version validation (full suite)

Complete cross-version test run performed 2026-08-22, comparing **Mojo 1.0.0 stable
(ed45d567)** against **Mojo 1.0.0b2 (2cf4d08a)** on the identical test suite
(437 test/example files).

### Compilation

| Scope | Result |
| --- | --- |
| `mojo precompile src/odyssey --Werror` | ✅ 0 errors (src compiles clean) |
| Eager build all tests (`mojo build --Werror`) | ✅ 380 PASS / 0 FAIL (tests only) |
| Eager build all examples (`mojo build --Werror`) | ❌ 57 FAIL (all deprecated syntax in examples/) |

### Runtime (JIT)

| Compiler | Pass | Fail | Total |
| --- | --- | --- | --- |
| **Mojo 1.0.0 stable** | 320 | 117 | 437 |
| **Mojo 1.0.0b2** | 391 | 45 | 436 |

### Cross-version regressions (pass on b2, fail on stable)

67 tests fail on stable but pass on b2. Each was individually diagnosed with full
stdout/stderr capture. Classification:

| Failure mode | Count | Upstream issue | Root cause |
| --- | --- | --- | --- |
| **FM-A: Memory corruption** | 49 tests | [#6939](https://github.com/modular/modular/issues/6939) | Return-move `__deinit__` UAF — garbage values (`~1.75`, `~4e-41`, `0.0`) or NaN in forward/backward passes |
| **FM-B: KGEN JIT crash** | 4 tests | [#6958](https://github.com/modular/modular/issues/6958) | Runtime segfault in `libKGENCompilerRTShared.so` after successful compilation |
| **FM-C: Atomic regression** | 1 test | [#6959](https://github.com/modular/modular/issues/6959) | Premature `__deinit__` UAF (same class as [#6707](https://github.com/modular/modular/issues/6707)) — the compiler hoists `unsafe_free` to right after the last syntactic use of a struct whose raw `Atomic` pointer escaped (e.g. `SpinLock._as_atomic`, `AtomicStats._counter` held in a local), so subsequent `Atomic` ops hit freed memory; the counter value is whatever the freed chunk holds. **WAR applied** (no pointer escapes from `SpinLock` or `AtomicStats`); repro still fails on stable |
| **FM-D: Timeout/OOM** | 4 tests | N/A (resource limit) | Heavy model tests (AlexNet/VGG16 224×224, MobileNet train) timeout on 4-core container |
| **FM-E: Non-deterministic** | 6 tests | (same as FM-A) | Pass on re-run; failed in full suite due to FM-A non-determinism |
| **FM-F: Pre-existing** | 1 test | N/A (already disabled) | `DISABLED_test_batchnorm` — SIMD type constraint, not a regression |
| **Unclassified** | 2 tests | N/A (timeout on re-run) | `googlenet_e2e`, `gradient_checking_batch_norm` — intermittent timeout or downstream corruption |

### Non-deterministic behavior

FM-A failures are non-deterministic — the same test sometimes passes and sometimes fails,
with different garbage values each run (`~1.75`, `~4e-41`, `0.0`). This is consistent
with use-after-free where the freed block may or may not be reused before the read.

6 tests that failed in the full suite passed on individual re-run (FM-E), confirming
the non-deterministic nature of FM-A.

---

## Reproducer inventory

| File | Failure mode | Passes b2 | Fails stable |
| --- | --- | --- | --- |
| `docs/dev/reproducers/repro_uaf_return_move.mojo` | FM-A (UAF) | ✅ | ❌ (ASAN-confirmed) |
| `docs/dev/reproducers/repro_min.mojo` | FM-A (UAF, minimal) | ✅ | ❌ (ASAN-confirmed) |
| `docs/dev/reproducers/repro_scalar_pow.mojo` | FM-2 (compile) | ✅ | ❌ |
| `repro/fmd_kgen_jit_crash_min.mojo` | FM-B (KGEN JIT) | ✅ | ❌ |
| `tests/odyssey/tensor/test_typed_batchnorm.mojo` | FM-B (KGEN JIT) | ✅ | ❌ |
| `tests/models/test_mobilenetv1_e2e.mojo` | FM-B (KGEN JIT) | ✅ | ❌ |
| `repro/fmd_kgen_jit_crash_min.mojo` (atomic stdlib) | FM-C (Atomic) | ✅ | ❌ |
| `repro/repro_6959_inline.mojo` | FM-C (Atomic UAF) | n/a (1.0.0 APIs) | ❌ (5/5) |
| `repro/repro_6959_inline_b2.mojo` | FM-C (b2-flavored mirror) | ✅ (5/5) | n/a |

## Complete issue tracker

| Issue | Failure mode | Status | URL |
| --- | --- | --- | --- |
| FM-1 / FM-A | Return-move UAF | OPEN | [#6939](https://github.com/modular/modular/issues/6939) |
| FM-2 | Scalar pow compile | OPEN | [#6940](https://github.com/modular/modular/issues/6940) |
| FM-3 | VM limit abort | OPEN | [#6941](https://github.com/modular/modular/issues/6941) |
| FM-B | KGEN JIT runtime crash | OPEN | [#6958](https://github.com/modular/modular/issues/6958) |
| FM-C | Premature `__deinit__` UAF from escaping `Atomic` pointers (`SpinLock._as_atomic`, `AtomicStats._counter`) | OPEN — WAR applied (no-escape API on both) | [#6959](https://github.com/modular/modular/issues/6959) |

---

## Raw-pointer-escape audit (premature `__deinit__` UAF — #6959/#6707 class)

An audit of every raw-pointer escape from heap-owning structs was performed on 1.0.0 stable
(after the `SpinLock`/`AtomicStats` WARs). **Same root cause everywhere**: a struct's heap
storage is freed by `__deinit__` that the compiler hoists to right after the last *syntactic*
use of the struct, so any raw pointer derived from that storage and held in a local reads
freed memory.

### Confirmed live (probe, 3/3 fail on stable; control 3/3 pass)

`repro/repro_tensor_data_ptr_uaf.mojo` — owned-local `AnyTensor`, `data_ptr` captured into a
local, then a loop reads through the pointer (the shape in `evaluation.mojo`/model e2e tests):

```text
acc      = 10222.0   (expected 10225.0 — 3 elements clobbered by tcmalloc freelist reuse)
MISMATCH: escaped data_ptr read freed memory
```

`repro/repro_tensor_data_ptr_control.mojo` — identical but reads via `t.load[i]` → `control: OK` 3/3,
proving the write path is correct and the mismatch is the escaped pointer.

### Inventory (classification)

| Class | Pattern | Sites | Risk |
| --- | --- | --- | --- |
| **A — WAR'd** | no-escape API, no pointer leaves the struct | `SpinLock`, `AtomicStats` (`memory_pool.mojo`) | ✅ fixed |
| **B — latent, owned-local escape** | owned local tensor → `data_ptr`/`_data` into a local → loop | `evaluation.mojo` (`logits_data`), `gradient_checker.mojo` (`f_plus_ptr`/`out_plus_ptr` etc.), model e2e tests (`test_vgg16_e2e.mojo`), `examples/mobilenetv1_cifar10/train.mojo`, and any user code | ⚠️ **UAF — passes today only by heap/codegen luck** (probe proves the class is live) |
| **C — borrowed params (safe)** | kernel funcs taking `tensor: AnyTensor`/`Tensor[dtype]` params, escaping `_data` into locals (~221 `._data` sites across `tensor_ops`, `typed/*`, `dtype_conv`, `gradient_clipping`, `inference_utils`, …) | owner lives in the caller frame, callee cannot destroy → deinit cannot be hoisted *in the callee* | ✅ safe from this bug (owner's own frame is the caller's responsibility) |
| **D — by-design (low risk)** | pool APIs returning blocks (`FreeList.pop`, `TensorMemoryPool.allocate`), `examples/mojo_patterns/trait_example.mojo` (no pointer returns from storage) | the returned pointer is the *object itself*, not a view into the struct's persistent storage; owner kept alive by caller | ✅ low |

### Fix path (not yet applied — API blast radius)

Per modular/modular#6707, the correct fix is to tie the returned pointer's origin to the
owner: `AnyTensor.data_ptr()` would return `Pointer[Scalar[dtype], origin=origin_of(self)]`
via `unsafe_origin_cast[origin_of(self)]()`. Validated on the `SpinLock` shape
(`/tmp/origin_mod_direct.mojo` + import test, 3/3). For tensors this requires `mut self`
on `data_ptr`, cascading `mut` through ~15-20 callers (`evaluate_logits_batch`,
`compute_accuracy_on_batch`, the `mixed_precision` `_simd` helpers, `gradient_checker`
helpers, `evaluation.mojo`, model e2e tests). Not applied pending decision — the latent class
is documented here and tracked under [#6959](https://github.com/modular/modular/issues/6959).
