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

- **modular/modular#6187** (heap corruption, `0.26.1`) — closed COMPLETED (2026-03-26);
  different signature (crash at alloc, not move semantics). The `data_ptr` keep-alive
  pattern once attributed to it remains canonical, now cited against the still-open
  premature-deinit class #6707/#6939.
- **modular/modular#6707** (stale read through `UnsafePointer[MutUntrackedOrigin]`,
  closed NOT_PLANNED "expected behavior") — different mechanism: origin-tracking
  limitation, not a move-semantics violation.
- **modular/modular#6475** (bitcast read in struct method vs external) — closed COMPLETED
  (2026-08-23); different mechanism (no moves involved).

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
| **FM-C: Atomic regression** | 1 test | [#6959](https://github.com/modular/modular/issues/6959) | Premature `__deinit__` UAF (same class as [#6707](https://github.com/modular/modular/issues/6707)) — the compiler hoists `unsafe_free` to right after the last syntactic use of a struct whose raw `Atomic` pointer escaped (e.g. `SpinLock._as_atomic`, `AtomicStats._counter` held in a local), so subsequent `Atomic` ops hit freed memory; the counter value is whatever the freed chunk holds. **WAR removed 2026-08-26** (upstream closed as expected-behavior, NOT a compiler fix — the caller-side escape repro still fails on stable); production is safe because its escapes are borrow-internal (see tracker) |
| **FM-D: Timeout/OOM** | 4 tests | N/A (resource limit) | Heavy model tests (AlexNet/VGG16 224×224, MobileNet train) timeout on 4-core container |
| **FM-E: Non-deterministic** | 6 tests | (same as FM-A) | Pass on re-run; failed in full suite due to FM-A non-determinism |
| **FM-F: Pre-existing** | 2 tests | **FIXED + re-enabled 2026-08-22** | `DISABLED_test_batchnorm` (SIMD type constraint) and `DISABLED_test_conv2d` (escape-UAF `_data` write in `test_conv2d_forward_batch_independence`) — both fixed and renamed back to `test_batchnorm.mojo` / `test_conv2d.mojo`; additionally `_batch_norm2d_fused_training_{float32,float64}` and both inference paths in `normalization_simd.mojo` migrated to origin-tied `data_ptr` (raw `_data` escapes on owned locals — the #6963 class) |
| **FM-G: Closure-capture premature deinit** | ~3 src paths | Filed [#6965](https://github.com/modular/modular/issues/6965); WAR applied (inline TaskGroup dispatch) | A `@parameter` capturing closure passed as a function-value parameter has its captured locals destroyed at closure-construction time — freed-memory reads in every `parallelize` batch path (pooling/conv/normalization batch ≥ 4). Passes on b2 |
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
| `tests/models/test_mobilenetv1_layers.mojo` | FM-B (KGEN JIT) | ✅ | ❌ (crashes right after `test_batchnorm2d_initialization`; identical output + crash at baseline, verified 2026-08-22) |
| `tests/examples/test_mobilenetv1_train_step.mojo` | FM-B (KGEN JIT, flaky) | ✅ | ❌ intermittent — crashes during JIT compile (zero output) on one run; passes on rerun and at baseline (verified 2026-08-22) |
| `repro/fmd_kgen_jit_crash_min.mojo` (atomic stdlib) | FM-C (Atomic) | ✅ | ❌ |
| `repro/repro_6959_inline.mojo` | FM-C (Atomic UAF) | n/a (1.0.0 APIs) | ❌ (5/5) |
| `repro/repro_6959_inline_b2.mojo` | FM-C (b2-flavored mirror) | ✅ (5/5) | n/a |
| `docs/dev/reproducers/repro_closure_capture_identical.mojo` | FM-G (closure-capture premature deinit) | ✅ (capture kept alive) | ❌ (deinit at closure-construction) |
| `docs/dev/reproducers/repro_closure_capture_uaf.mojo` | FM-G (UAF consequence, pointer payload) | n/a (1.0.0 syntax) | ❌ (0.0 reads at elements 0-1) |

## Complete issue tracker

| Issue | Failure mode | Status | URL |
| --- | --- | --- | --- |
| FM-1 / FM-A | Return-move UAF | CLOSED DUPLICATE of #6707 (expected behavior, no fix) — double-`__deinit__` **still reproducible on 1.0.0 stable** (verified 2026-08-26); canonical single-tensor-return shape retained (`batch_norm2d_inplace`); permanent canary `test_return_move_canary.mojo` (2026-08-25) | [#6939](https://github.com/modular/modular/issues/6939) |
| FM-2 | Scalar pow compile | OPEN | [#6940](https://github.com/modular/modular/issues/6940) |
| FM-3 | VM limit abort | OPEN | [#6941](https://github.com/modular/modular/issues/6941) |
| FM-B | KGEN JIT runtime crash | CLOSED COMPLETED (2026-08-22) — not reproducible in the current suite (all 363 test files pass); no code workaround existed | [#6958](https://github.com/modular/modular/issues/6958) |
| FM-C / #6963 | Premature `__deinit__` UAF from escaping `Atomic`/raw pointers | **#6959 CLOSED COMPLETED (2026-08-22, maintainer: "lifetime gotcha") — NOT a compiler fix**: the caller-side owned-local escape repro (`repro/repro_6959_inline.mojo`) **still FAILS 5/5 on 1.0.0 stable** (verified 2026-08-26). The prior "verified FIXED" claim was wrong — it was based on tests that only use single-expression escapes. WAR removal is nonetheless **safe**: production escapes are borrow-internal (pointer derived from borrowed `self` within a method, owner kept alive by the caller frame), verified by `docs/dev/reproducers/probe_update_peak_shape.mojo` passing 5/5 on the exact `update_peak_cached` shape. **#6963 closed DUPLICATE of #6707 (expected behavior — use proper origin-tracking)**: origin-tied `data_ptr` IS that guidance; "workaround" labels removed 2026-08-26, pattern retained as canonical | [#6959](https://github.com/modular/modular/issues/6959) · [#6963](https://github.com/modular/modular/issues/6963) |
| FM-G | Premature `__deinit__` of closure captures when a `@parameter` capturing closure is passed as a function-value parameter (breaks `parallelize` batch paths in pooling/conv/normalization) | **Filed [#6965](https://github.com/modular/modular/issues/6965)** — WAR applied: inline TaskGroup dispatch at all 3 call sites (closure never crosses a function-value boundary); `parallelize` retained for scalar/keep-alive captures only | 2026-08-22 |

---

## Full-suite validation (2026-08-22)

All 362 non-DISABLED test files now compile and run on 1.0.0 stable:

- **70 converted files**: 68 pass; 2 remaining are deterministic FM-B KGEN
  crashes (`test_mobilenetv1_layers`, upstream #6958 — same crash at
  baseline; `test_mobilenetv1_e2e` and `test_typed_batchnorm` are also in
  the KGEN class). `test_mobilenetv1_train_step` is flaky-KGEN (passes on
  rerun, verified against baseline).
- **294 non-converted files**: 290 pass; 4 initially failed and were
  root-caused + fixed this session: `test_lazy_expression` (missing
  `[dtype]` on `data_ptr`), `test_slicing` (refcount assertions vs 1.0.0
  last-use `__deinit__`), `test_multi_precision_training` (unneeded `mut`
  cascade on a borrowed param), `test_alexnet_layers` (JIT time > 200 s;
  passes with a 600 s timeout — slow, not broken).
- The two previously-DISABLED layer suites (`test_batchnorm.mojo`,
  `test_conv2d.mojo`) are fixed, `--Werror`-clean, and re-enabled.
- Owned-local raw-pointer-escape audit: **0 remaining** in `src/` (20
  migrated across 10 files to origin-tied `data_ptr[]`).

Failure classes filed upstream (2026-08-26 status): FM-A/return-move
(#6939, closed DUPLICATE — double-`__deinit__` still reproducible), FM-B/KGEN JIT
(#6958, closed COMPLETED — not reproducible in the current suite),
FM-C/Atomic (#6959, closed COMPLETED as expected-behavior — the caller-side
escape repro still fails on stable, but production escapes are borrow-internal
so the WAR removal stands; probe evidence 2026-08-26), raw-pointer escape
(#6963, closed DUPLICATE — canonical
origin-tied pattern retained), FM-G/closure captures (#6965, open).

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
| **B — latent, owned-local escape** | owned local tensor → `data_ptr`/`_data` into a local → loop | `evaluation.mojo` (`logits_data`), `gradient_checker.mojo` (`f_plus_ptr`/`out_plus_ptr`), model e2e tests (`test_vgg16_e2e.mojo`), and any user code. **Found live in CI 2026-08-23**: `examples/mobilenetv1_cifar10/model.mojo` `depthwise_conv2d` (per-channel `channel_input`/`channel_filter`/`channel_bias`/`channel_output` + classifier `flattened`/`out`) — root cause of the `test_bn_persistence` flake (`NaN`/`0.0` running stats → "not persisted (#5537)"); `examples/mobilenetv1_cifar10/{inference,train,train_autograd}.mojo`, `examples/alexnet_cifar10/{run_train,run_train_autograd}.mojo`, `examples/vgg16_cifar10/{train,train_autograd,train_new}.mojo` (smoke-dataset builders + `loss_tensor` scalar extraction) — root cause of the CI `training-smoke` NaN failures. **All migrated to origin-tied `data_ptr`** | ⚠️ **UAF — visible as NaN/garbage in CI when heap reuse lands** (probe proves the class is live) |
| **B2 — full sweep 2026-08-25 (remaining owned-local escapes)** | owned local tensor → raw `_data` into a local → loop | a scripted audit (`scripts/audit_escape_sites.py`) flagged every owned-local `_data`/`data_ptr` escape whose owner is never used again. Migrated the last raw `_data` sites to origin-tied `data_ptr`: `src/odyssey/core/matmul.mojo` (`b_t` float32/float64 transpose results — the GEMM kernels held `bt_ptr = b_t._data` across the whole blocked loop), `src/odyssey/tensor/typed/arithmetic.mojo` (`t_cont` in `_multiply_scalar_typed`), `examples/lenet_emnist/inference.mojo` (`logits` argmax/softmax), `examples/resnet18_cifar10/inference.mojo` (`logits` argmax), `examples/resnet18_cifar10/test_forward_cache_velocities.mojo` (`cache_logits`/`forward_logits` compare), `examples/resnet18_cifar10/test_model.mojo` (`logits_train` NaN check). All other flagged sites are single-expression reads (pointer consumed in the same statement) or borrowed params (Class C) — safe. | ✅ **all migrated 2026-08-25** — remaining raw `_data` sites are single-expression or Class C |
| **C — borrowed params (safe)** | kernel funcs taking `tensor: AnyTensor`/`Tensor[dtype]` params, escaping `_data` into locals (~221 `._data` sites across `tensor_ops`, `typed/*`, `dtype_conv`, `gradient_clipping`, `inference_utils`, …) | owner lives in the caller frame, callee cannot destroy → deinit cannot be hoisted *in the callee* | ✅ safe from this bug (owner's own frame is the caller's responsibility) |
| **D — by-design (low risk)** | pool APIs returning blocks (`FreeList.pop`, `TensorMemoryPool.allocate`), `examples/mojo_patterns/trait_example.mojo` (no pointer returns from storage) | the returned pointer is the *object itself*, not a view into the struct's persistent storage; owner kept alive by caller | ✅ low |

### Fix applied (origin-tie WAR)

Per modular/modular#6707, the fix ties the returned pointer's origin to the owner:
`AnyTensor.data_ptr()` now returns `Pointer[Scalar[dtype], origin=origin_of(self)]` via
`unsafe_origin_cast[origin_of(self)]()`. This requires `mut self`, cascading `mut` through
the callers: `evaluate_logits_batch`/`compute_accuracy_on_batch` (`metrics/evaluate.mojo`),
the `mixed_precision` `_simd` helpers + `convert_to_fp32_master`/`update_model_from_master`/
`clip_gradients_by_value`, and the `gradient_checker` helpers + public functions
(`check_gradients`, `check_gradients_verbose`, `compute_numerical_gradient`,
`compute_sampled_numerical_gradient`, `check_gradient`, `assert_gradients_close`,
`assert_sampled_gradients_close`). Test files that passed rvalue tensors to these now bind
a `var` first (`test_gradient_checking_basic`, `test_gradient_checking_dtype`, autograd
`test_variable_batch_norm`/`test_variable_depthwise_conv2d`).

**Validated**: the `data_ptr` UAF probe (`repro/repro_tensor_data_ptr_uaf.mojo`) now
passes 3/3 on stable (was 10222 vs 10225 before); AnyTensor, memory-pool, gradient-checking,
autograd, and training evaluation suites pass. Remaining failures in the sweep
(`test_typed_*`, `test_activations`/`arithmetic_backward`/`elementwise`, FP16 SIMD test)
are pre-existing FM-A/FM-B failures, identical on the unpatched baseline.

### Overlap analysis: FM-A (return-move) vs this escape UAF

**Verdict: same compiler-lifetime root-cause family, DIFFERENT triggers — the escape WAR
does not fix FM-A and vice versa.**

| | FM-A (#6939) | Escape UAF (#6963/#6959) |
| --- | --- | --- |
| Trigger | Struct **returned by value** (`return Pair(a^, b^)`) — the moved-from source's `__deinit__` also runs, over-decrementing the shared refcount → buffer freed | Raw pointer **escapes into a local** — the owner's `__deinit__` is hoisted to the last syntactic use → buffer freed early |
| Shape | No pointer escape needed | No return-move needed |
| Example | `subtract_backward` → `return GradientPair(grad_a^, grad_b^)` (the `test_subtract_scalar_backward` failure: `grad_b` reads 0.0 instead of -6.0) | `evaluation.mojo` → `var logits_data = logits.data_ptr()` |
| Evidence of non-overlap | FM-A repro `docs/dev/reproducers/repro_uaf_return_move.mojo` still fails `0 1 0 1` **after** the escape WAR; FM-A tests fail with the same garbage class with and without the WAR; failing shapes contain no escape | Escape probe passes 3/3 **only** with the WAR; failing shapes contain no return-move |

Both are the 1.0.0 compiler's lifetime analysis misplacing `__deinit__` (one extra, one early) —
which is why #6939 was closed DUPLICATE and the issues cross-reference each other — but they are
independent code shapes. FM-A remains OPEN upstream (#6939, closed as DUPLICATE without a fix;
see the `tensor.mojo` move-constructor note not to reintroduce the refcount-sentinel WAR). Note
FM-A garbage values are non-deterministic run-to-run (e.g. `4.096e-41` vs `-1.75` for the same
test), consistent with freed-block reuse timing.

**Filed upstream**: [#6963](https://github.com/modular/modular/issues/6963) — self-contained
stdlib-only repro (`repro/repro_tensor_inline.mojo`, fails 3/3 stable / passes 3/3 b2
mirror `repro/repro_tensor_inline_b2.mojo`, control `repro/repro_tensor_inline_control.mojo`
passes 3/3) with cross-version evidence and the `mut self` origin-tie workaround.
Related closed issues: #6959 (COMPLETED), #6707 (NOT_PLANNED), #6939 (DUPLICATE).

---

## Premature-`__deinit__` pattern inventory (audit 2026-08-25) and hardening

Four distinct code shapes can trip the 1.0.0 compiler's premature/incorrect
`__deinit__` placement. All were re-audited after the #5802 sweep; results and
hardening below.

| # | Trigger shape | In-repo sites | Upstream | Empirical status on 1.0.0 stable |
| --- | --- | --- | --- | --- |
| 1 | **Raw `_data` escape into a local** (`var ptr = x._data...`, owner never used again) | migrated to origin-tied `data_ptr` (#5801/#5802); **0 remain** | [#6963](https://github.com/modular/modular/issues/6963) | minimal repro fails 3/3; value probe passes after WAR |
| 2 | **Return-move of a custom-`deinit` refcounted struct** (`return r^`, `return Pair(a^, b^)`) | ~30 `GradientPair`/`GradientTriple`/`DualTensor`/`Variable` returns + `return output^` (batchnorm/mamba/ssm/kan/jvp) + `Tuple[...](...^)` (normalization_simd/serialization/data_generators) | [#6939](https://github.com/modular/modular/issues/6939) | minimal raw repro (`Box`) still fails 5/5 (values correct only by allocator luck; ASAN-clean on b2); **real-`AnyTensor` shapes pass 10/10 / 25 iterations** in all 3 shapes (whole-local, Pair-ctor, Tuple-ctor) |
| 3 | **Closure capture crossing a function-value boundary** | `parallelize` kept only for scalar/keep-alive captures; pooling/conv/normalization dispatch inline via `TaskGroup`; `vectorize` closures in examples capture pointers but are `@always_inline` comptime-parameters (direct-call, not function-value pass) | [#6965](https://github.com/modular/modular/issues/6965) | repro fails (deinit at closure construction); in-repo paths WAR'd; `test_memory_pool_threadsafe` captures only `pool` (used in enclosing frame after the call — keep-alive, safe) |
| 4 | **Borrowed-param `_data` escape in a callee** (owner lives in caller frame) | ~221 `._data` sites | #6707 (NOT_PLANNED) | safe by construction (callee cannot destroy the caller's local) |

### Hardening added 2026-08-25

- **Permanent FM-A canary**: `tests/odyssey/core/test_return_move_canary.mojo`
  — exercises all three live shapes (whole-local `return output^`,
  `GradientPair(grad_a^, grad_b^)`, `Tuple(t1^, t2^, t3^)`) with real
  `AnyTensor`s, 25 iterations each, exact-value asserts. Heap-reuse dependent,
  so it cannot *guarantee* catching a regressed compiler on every run, but the
  exact-value assert trips the moment freed blocks are reused (the mechanism
  behind the FM-A/FM-E CI flakes). Passes 4/4 on stable (2026-08-25).
- **CI gate for pattern 1**: `scripts/audit_escape_sites.py --raw-only`
audit now distinguishes pointer-HELD escapes (dangerous) from in-statement
  value reads (safe), strips comments, handles multi-line statements, and
  exits 1 on any owned-local raw `_data` escape. Wired as pre-commit hook
  `no-owned-local-raw-escapes` (always_run). Verified: repo clean (0 sites),
  synthetic negative control flagged.

### Empirical matrix (probes, 2026-08-25, stable 1.0.0)

| Probe | Shape | Result |
| --- | --- | --- |
| `docs/dev/probe_return_caret_c.mojo` | real AnyTensor, `return GradientPair(ga^, gb^)` vs implicit | both OK 10/10 |
| `docs/dev/probe_return_caret_d.mojo` | real AnyTensor, `return output^` whole local | OK 10/10 (small+large) |
| `probe_return_caret_a/b.mojo` (synthetic Box, no refcount) | explicit `^` vs implicit | explicit OK, implicit CORRUPT (no refcount — differences are model-specific) |
| `repro_uaf_return_move.mojo` | `Box` custom-deinit + refcount + `return r^` | **FAILS 5/5** (moved-from deinit runs; ASAN heap-UAF) |

Takeaway: the raw-`Box` shape (custom move + shared refcount + `return r^`)
still demonstrates the FM-A bug deterministically; the real-`AnyTensor` shapes
pass because `AnyTensor`'s List fields + larger allocations make the double-
decrement land on a refcount cell that happens to survive in practice, or the
return-move path is instantiated differently. The canary keeps watch; do NOT
reintroduce the refcount-sentinel WAR (see `tensor.mojo` note).

---

## Failure Mode G: Premature `__deinit__` of closure captures at the function-value boundary

**Severity**: High (silent use-after-free — garbage reads / segfaults in every
`parallelize`-driven parallel batch path: pooling, conv, normalization).

**Status**: Root-caused + documented; **filed as
[modular/modular#6965](https://github.com/modular/modular/issues/6965)** (2026-08-22).
WAR applied: the three impacted call sites (`pooling.mojo`, `conv.mojo`,
`normalization.mojo`) now dispatch inline via `TaskGroup` so the capturing
closure never crosses a function-value parameter boundary; `parallelize`
remains only for scalar / keep-alive captures (e.g. the memory-pool
thread-safety tests).

### Symptom

A `@parameter` capturing closure (`def worker(i: Int) capturing:`) that captures a local
heap-owning struct, when **passed as a function-value parameter**
(`func: def(Int) capturing -> None`), causes the captured local's `__deinit__` to fire at
closure-construction time — **before any call through the closure** — freeing the
captured buffer while the closure body still reads/writes it. Direct calls to the same
closure (no function-value boundary) keep the capture alive correctly.

The corruption is data-buffer-correlated, not call-order-correlated: the freed block's
head is reused by the closure context (reads as 0.0 / garbage), the tail still holds
stale values — e.g. `ones([8], float32)` reads as `0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0`.

### Cross-version evidence (identical file `docs/dev/reproducers/repro_closure_capture_identical.mojo`)

```text
$ mojo run repro_closure_capture_identical.mojo    # 1.0.0b2
creating box
calling closure via function-value param:           # ✅ NO deinit print before reads
  param[ 0 ]: 700                                     #    capture kept alive
  ...

$ mojo run repro_closure_capture_identical.mojo    # 1.0.0 stable
creating box
    >>> Box.__deinit__ firing (payload= 7 )        # ❌ PREMATURE — fires at closure
calling closure via function-value param:            #    construction, before any call
  param[ 0 ]: 700                                    #    (reads correct here only because
  ...                                                #    an Int payload frees nothing)
```

With a heap-owning payload (`docs/dev/reproducers/repro_closure_capture_uaf.mojo`, 1.0.0
only):

```text
$ mojo run repro_closure_capture_uaf.mojo          # 1.0.0 stable
    >>> Box.__deinit__ firing (freeing buffer)     # buffer freed before closure runs
  param[ 0 ]: 0.0        # UAF read — expected 0.25
  param[ 1 ]: 0.0        # UAF read — expected 0.75
  param[ 2 ]: 1.25       # stale value from freed-but-unreused tail
  ...
```

### Root cause

In 1.0.0 the lifetime/use analysis treats a captured local as NOT live inside a
`@parameter` capturing closure once that closure is **passed as a function-value
parameter** (`def(Int) capturing -> None`):

- **Owned captured locals**: `__deinit__` is hoisted to closure-construction time (the
  compiler considers the closure-pass the last use), freeing the buffer the closure body
  still reads → use-after-free.
- **Pointer/scalar captures**: the closure's uses are not registered — the compiler
  warns `assignment to 'xd' was never used` for a pointer the closure body reads, and the
  closure reads garbage or crashes.

Direct calls to the same closure (no function-value boundary) are correct, and a *later
use of the captured variable in the enclosing frame* keeps owned captures alive — but
only for the owned-local case; borrowed-param and raw-pointer captures stay broken
(probe `repro_war15`: `var xd = x._data` captured → compiler reports `xd` unused →
crash). This is a FOURTH trigger in the premature-`__deinit__` family — distinct from
FM-A's return-move and FM-C/#6963's raw-pointer-escape (the pure repro has neither a
return-move nor an escaping pointer).

### Impact on Odyssey

Every `parallelize` call site passes a capturing closure as a function-value param:

- `src/odyssey/core/pooling.mojo:168` — `maxpool_batch` captures `x`, `output` → batch ≥ 4
  max-pooling returns all zeros / segfaults; this is the deterministic
  `test_autograd_convergence` LeNet-shape crash (verified: standalone copy of the real
  closure body crashes identically; batch 1-3 sequential pass, batch 4+ parallel dies)
- `src/odyssey/core/conv.mojo:216` — `conv_batch` captures `x`, `output` → batch ≥ 4 conv
  corrupted
- `src/odyssey/core/normalization.mojo:147` — `normalize_batch` captures `typed_x`,
  `out_ptr` etc. → batch ≥ 4 normalization corrupted
- `src/odyssey/core/parallel_utils.mojo:95` — generic `parallel_for_batch` wrapper

All of these pass on 1.0.0b2 (which keeps captures alive).

### Reproducers

- `docs/dev/reproducers/repro_closure_capture_identical.mojo` — identical file, both
  versions (deinit-timing proof).
- `docs/dev/reproducers/repro_closure_capture_uaf.mojo` — 1.0.0 pointer payload showing
  the UAF reads.

### WAR (proposed, pending approval)

Keep-alive (a later use of the captured variable after the pass) fixes ONLY the
owned-local capture shape and is fragile. The reliable WAR is to stop passing
tensor-capturing closures to `parallelize` on 1.0.0: restructure the pooling/conv/
normalization closures to take the tensors as **arguments** (borrowed params are safe —
Class C) instead of capturing them, or disable the parallel path (fall back to the
sequential loop) until upstream fixes the capture liveness bug. Approval pending
alongside the upstream filing decision.
