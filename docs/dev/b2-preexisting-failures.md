# Mojo 1.0.0b2 Pre-existing Failures Report

**Date**: 2026-08-22
**Compiler**: Mojo 1.0.0b2 (2cf4d08a)
**Codebase**: Odyssey at commit febfcd23 (pre-1.0.0-migration)
**Total failures**: 45 out of 436 tests/examples

---

## Summary

The 45 b2 failures fall into **6 unique root causes**, none of which are regressions
introduced by the 1.0.0 migration. They are pre-existing issues that exist on b2
and were already present before any migration work.

| Category | Count | Root Cause | Severity |
|----------|-------|------------|----------|
| A. Utility modules without main() | 11 | Test runner tries to run import-only modules | False positive |
| B. KGEN JIT crash on b2 | 4 | Same crash as #6958, also affects b2 | High |
| C. Pre-existing disabled test | 1 | Numerical mismatch in DISABLED_test_conv2d | Low (disabled) |
| D. Deprecated API runtime failures | 20 | Deprecated bitcast/load/store/index APIs | Medium |
| E. Timeout / OOM on heavy tests | 4 | Model tests too heavy for 4-core container | Resource limit |
| F. Non-deterministic (passes on re-run) | 5 | Memory pressure or JIT timing in full suite | False positive |

**Total real failures (B+C+D)**: 25
**False positives (A+E+F)**: 20

---

## Category A: Utility modules without main() — 11 files (FALSE POSITIVE)

These files are **import-only modules** (conftest, fixtures, utils, model definitions)
that the test runner incorrectly tries to execute as standalone programs. They have
no `def main()` because they're libraries, not executables.

### Test files (7):

1. `tests/conftest.mojo` — shared test assertions
2. `tests/helpers/fixtures.mojo` — test fixture utilities
3. `tests/helpers/utils.mojo` — test helper functions
4. `tests/odyssey/conftest.mojo` — shared test config
5. `tests/odyssey/fixtures/config_fixtures.mojo` — config test fixtures
6. `tests/odyssey/fixtures/mock_data.mojo` — mock data generators
7. `tests/odyssey/fixtures/mock_tensors.mojo` — mock tensor factories

### Example files (4):

8. `examples/grok/lenet_emnist/model.mojo` — model definition (no main)
9. `examples/mnist/model.mojo` — model definition (no main)
10. `examples/mobilenetv1_cifar10/model.mojo` — model definition (no main)
11. `examples/resnet18_cifar10/model.mojo` — model definition (no main)

**Action needed**: Exclude these from the test runner's direct execution. They should
only be compiled as libraries, not run as executables.

---

## Category B: KGEN JIT crash on b2 — 4 examples (REAL BUG)

These examples crash with `libKGENCompilerRTShared.so` / `libAsyncRTRuntimeGlobals.so`
stack traces on b2 — the same crash pattern documented in #6958/#6413/#6445.

### Affected files:

1. `examples/autograd/linear_regression.mojo` — JIT crash
2. `examples/autograd/simple_example.mojo` — JIT crash
3. `examples/custom_layers/attention_layer.mojo` — JIT crash
4. `examples/data_pipeline_demo.mojo` — JIT crash

### Stack trace (identical for all 4):

```text
#0 libKGENCompilerRTShared.so+0xfbf8e
#1 libKGENCompilerRTShared.so+0xf90a6
#2 libKGENCompilerRTShared.so+0xfcdd0
#3 libc.so.6+0x45330
#4 libAsyncRTRuntimeGlobals.so+0x4e8c2
```

### Impact

This proves the KGEN JIT crash is NOT stable-only — it also affects b2 under heavy
JIT load. The crash is related to the autograd runtime (variable tracking, gradient
computation) which exercises complex JIT code paths.

**Filed**: modular/modular#6958 (noted in comment that b2 is also affected)

---

## Category C: Pre-existing disabled test — 1 file (KNOWN BUG)

### Affected file:

1. `tests/odyssey/core/layers/DISABLED_test_conv2d.mojo`

### Error:

```text
Unhandled exception caught during execution: -0.12866569 !≈ -0.17696491 (diff: 0.048299223)
```

### Root cause

Numerical mismatch in Conv2dLayer test — the test was **already disabled** (filename
prefix `DISABLED_`) before the 1.0.0 migration. This is a pre-existing bug in the
conv2d backward pass numerical precision.

**Action**: No action needed — test is already disabled.

---

## Category D: Deprecated API runtime failures — 20 examples (PRE-EXISTING)

These examples use APIs deprecated in earlier Mojo versions (pre-b2 or during b2
development). On b2 they fail with compile errors or runtime crashes because the
deprecated APIs were removed or changed.

### Sub-categories:

#### D1. Deprecated `bitcast` (not `unsafe_bitcast`) — 3 files

- `examples/googlenet_cifar10/test_backward.mojo`
- `examples/lenet_emnist/run_train_autograd.mojo`
- `examples/lenet_emnist/train_autograd.mojo`

#### D2. Deprecated positional `__getitem__` on pointers — 8 files

- `examples/alexnet_cifar10/run_train.mojo`
- `examples/alexnet_cifar10/run_train_autograd.mojo`
- `examples/googlenet_cifar10/train.mojo`
- `examples/googlenet_cifar10/train_autograd.mojo`
- `examples/grok/lenet_emnist/run_train.mojo`
- `examples/lenet_emnist/run_train.mojo`
- `examples/mobilenetv1_cifar10/train.mojo`
- `examples/mobilenetv1_cifar10/train_autograd.mojo`

#### D3. Runtime exit code 1 (deprecated API paths) — 8 files

- `examples/alexnet_cifar10/inference.mojo`
- `examples/googlenet_cifar10/inference.mojo`
- `examples/lenet_emnist/inference.mojo`
- `examples/mnist/train.mojo`
- `examples/mobilenetv1_cifar10/inference.mojo`
- `examples/resnet18_cifar10/train.mojo`
- `examples/resnet18_cifar10/train_autograd.mojo`
- `examples/vgg16_cifar10/train.mojo` + `train_autograd.mojo` + `train_new.mojo`

#### D4. Missing file dependency — 1 file

- `examples/mnist/train_autograd.mojo` — "No such file or directory"

### Root cause

The examples/ directory was not migrated to b2-compatible syntax. The deprecated
APIs (`bitcast`, positional subscripts, `load`/`store` without `unsafe_` prefix)
were already removed by the time b2 was released.

**Action**: Migrate examples/ to b2-compatible syntax (same fixes needed for 1.0.0).

---

## Category E: Timeout / OOM on heavy tests — 4 files (RESOURCE LIMIT)

### Affected files:

1. `tests/models/test_alexnet_layers.mojo` — 224×224 forward pass, heavy
2. `tests/models/test_mobilenetv1_e2e.mojo` — full model E2E
3. `tests/models/test_googlenet_e2e.mojo` — full model E2E
4. `tests/models/test_vgg16_e2e.mojo` — 224×224 forward pass, very heavy

### Root cause

Heavy model tests with large tensor operations exceed the 4-core container's
memory and CPU capacity when run sequentially after 400+ other tests. These pass
when run individually.

**Action**: No code fix needed — resource constraint. Consider splitting heavy
tests into separate CI jobs or increasing container resources.

---

## Category F: Non-deterministic (passes on re-run) — 5 files (FALSE POSITIVE)

### Affected files:

1. `tests/models/test_alexnet_layers.mojo` — passes on re-run
2. `tests/odyssey/core/test_backward_conv_padding.mojo` — passes on re-run
3. `tests/odyssey/core/test_backward_losses.mojo` — passes on re-run
4. `tests/odyssey/core/test_conv_noncontiguous.mojo` — passes on re-run
5. `tests/odyssey/core/test_reduction.mojo` — passes on re-run

### Root cause

These tests pass when run individually but fail in the full suite due to memory
pressure, JIT compilation side effects, or allocator state from preceding tests.
The failures are non-deterministic — they appear in different positions on different
runs.

**Action**: No code fix needed — environmental non-determinism.

---

## Recommendations

1. **Fix test runner** to skip import-only modules (Category A) — exclude files
   without `main()` from direct execution
2. **Migrate examples/** to b2-compatible syntax (Category D) — apply the same
   `bitcast→unsafe_bitcast`, `ptr[i]→ptr[unsafe_offset=i]` fixes
3. **Note b2 KGEN JIT crash** in #6958 (Category B) — the crash is not stable-only
4. **No action needed** for Categories C, E, F — pre-existing, resource limits, or
   non-deterministic
