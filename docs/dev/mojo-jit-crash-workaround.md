# Mojo JIT Crash: `libKGENCompilerRTShared.so` (Mojo v0.26.1)

**Tracking**: Issue #3330, follow-up from #3120

**Status**: The heap-corruption file-splitting workaround is **RESOLVED** (2026-03-20, bitcast UAF fix).
The test file splitting workaround is no longer necessary. A per-file JIT crash
retry mechanism was added (2026-03-25, ADR-014) but has since been **REMOVED** to force root cause
investigation. Persistent JIT crashes now fail visibly instead of being masked by retries.

The JIT crash described in this document (`libKGENCompilerRTShared.so`) is a separate upstream
Mojo 0.26.1 compiler bug that is mitigated by targeted submodule imports (see below).

## Root Cause

The JIT crash is triggered by **compilation footprint**, not random instability. When a test
file does `from shared.core import AnyTensor, zeros`, the Mojo JIT must compile **all 37,401
lines** across 60+ source files, because `shared/core/__init__.mojo` eagerly re-exports 200+
symbols from 40+ modules. This compilation volume intermittently overflows a JIT-internal
buffer, triggering glibc's `__fortify_fail_abort`.

**Evidence**:

- `__fortify_fail_abort` fires on buffer overflow detection -- this is a **compilation-time**
  overflow, not a runtime bug
- `shared/core/__init__.mojo` imports from 40+ submodules including `dtype_dispatch.mojo`
  (176+ monomorphizations) and `elementwise.mojo` (154+ monomorphizations)
- Files using `from shared.core import` (package-level) -> compile all 37K lines per test
- Files using `from shared.core.any_tensor import` (targeted) -> compile ~500-2000 lines per test
- The crash is non-deterministic because ASLR, memory layout, and JIT caching vary per run

## Fix Applied

**Convert package-level imports to targeted submodule imports** in all test files.

```mojo
# BEFORE: compiles all 37,401 lines via __init__.mojo
from shared.core import AnyTensor, zeros, ones, matmul, relu

# AFTER: compiles only the needed modules (~500-2000 lines)
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.matrix import matmul
from shared.core.activation import relu
```

This fix was applied to 126 test files in the commit that introduced this note. See the
symbol-to-submodule mapping below.

## Symbol-to-Submodule Mapping

| Symbol(s) | Submodule |
| --- | --- |
| AnyTensor, zeros, ones, full, empty, arange, eye, linspace, ones_like, zeros_like, full_like, nan_tensor, inf_tensor, neg_inf_tensor, clone, item, diff, randn | `shared.core.any_tensor` |
| reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack, split, tile, repeat, permute, is_contiguous, as_contiguous, view, broadcast_to, ... | `shared.core.shape` |
| add, subtract, multiply, divide, floor_divide, modulo, power, multiply_scalar, \*\_backward | `shared.core.arithmetic` |
| matmul, transpose, transpose_view, dot, outer, \*\_backward | `shared.core.matrix` |
| relu, sigmoid, tanh, softmax, gelu, swish, mish, elu, selu, hard\_\*, \*\_backward | `shared.core.activation` |
| linear, linear_no_bias, \*\_backward | `shared.core.linear` |
| conv2d, depthwise_conv2d, depthwise_separable_conv2d, \*\_backward | `shared.core.conv` |
| maxpool2d, avgpool2d, global_avgpool2d, \*\_backward | `shared.core.pooling` |
| dropout, dropout2d, \*\_backward | `shared.core.dropout` |
| batch_norm2d, layer_norm, group_norm, instance_norm, \*\_backward | `shared.core.normalization` |
| abs, sign, exp, log, sqrt, sin, cos, clip, ceil, floor, round, trunc, logical\_\*, \*\_backward | `shared.core.elementwise` |
| equal, not_equal, less, less_equal, greater, greater_equal | `shared.core.comparison` |
| broadcast_shapes, are_shapes_broadcastable, compute_broadcast_strides, BroadcastIterator | `shared.core.broadcasting` |
| xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, he_uniform, he_normal, uniform, normal, constant | `shared.core.initializers` |
| binary_cross_entropy, mean_squared_error, cross_entropy, smooth_l1_loss, hinge_loss, focal_loss, kl_divergence, \*\_backward | `shared.core.loss` |
| has_nan, has_inf, count_nan, count_inf, check_tensor_safety, tensor_min, tensor_max, ... | `shared.core.numerical_safety` |
| dispatch_unary, dispatch_binary, dispatch_softmax, dispatch_gelu, ... | `shared.core.dtype_dispatch` |
| sum, mean, max_reduce, min_reduce, variance, std, median, percentile, \*\_backward | `shared.core.reduction` |
| argmax, top_k_indices, top_k, argsort | `shared.core.utils` |
| validate_tensor_shape, validate_tensor_dtype, validate_matching_tensors, ... | `shared.core.validation` |
| TensorMemoryPool, PoolConfig, PoolStats, get_global_pool, pooled_alloc, pooled_free | `shared.core.memory_pool` |
| GradientPair, GradientTriple, GradientQuad, Conv2dNoBiasGradient, ... | `shared.core.gradient_types` |
| BF16, FP8, BF8, FP4, E8M0 | `shared.core.types.dtype_aliases` |
| Module | `shared.core.module` |
| Sequential2, Sequential3 | `shared.core.sequential` |

## Diagnosis: Compiler Flake vs. Test Bug

The key diagnostic is **where the crash appears relative to test output**:

| Symptom | Cause |
| --- | --- |
| `execution crashed` appears **before any test output** | Import explosion crash -- check import style |
| `execution crashed` or segfault appears **after test output** | Likely a real test bug |
| Specific assertion failure message | Real test bug -- investigate |

### Sample: Import Explosion Crash

```text
execution crashed
```

No test names printed. Nothing ran. This is the JIT crash from compiling too many modules.

### Sample: Real Test Failure (investigate code)

```text
test_forward ... PASS
test_backward ... FAIL: assertion failed at line 42
  left:  0.5
  right: 0.6
```

Tests started running, then a specific test failed with a meaningful message.

## New Test Files: Use Targeted Imports

When writing new test files, always use targeted submodule imports:

```mojo
# CORRECT: only compiles what you need
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.activation import relu, sigmoid

# WRONG: compiles all 37K lines, risks JIT crash
from shared.core import AnyTensor, zeros, ones, relu, sigmoid
```

## Relationship to Heap Corruption Bug (RESOLVED)

The heap corruption bug was **resolved on 2026-03-20** via a bitcast UAF fix. The test file
splitting workaround is no longer necessary. The `continue-on-error: true` workaround has been
removed from `comprehensive-tests.yml`.

| | JIT Crash (this doc) | Heap Corruption Workaround |
| --- | --- | --- |
| **Trigger** | Package-level import compilation overflow | After exactly ~15 cumulative tests in one file |
| **Output** | `execution crashed` before any test runs | Crash mid-run after test output |
| **Root cause** | `__init__.mojo` import explosion -> monomorphization overflow | Cumulative allocations exceed JIT heap limit |
| **Fix** | Targeted submodule imports (applied) | Bitcast UAF fix (resolved 2026-03-20) |
| **CI workarounds** | None -- every failure is immediately visible | None -- `continue-on-error` removed |

## Controlled Experiment (2026-03-15)

Two synthetic test files were created to isolate the import-style variable:

- `tests/shared/core/test_jit_crash_heavy_import.mojo` -- uses `from shared.core import` (package-level)
- `tests/shared/core/test_jit_crash_light_import.mojo` -- uses `from shared.core.any_tensor import` (targeted)

### Local Results (GLIBC 2.39, Mojo 0.26.1, WSL2 Linux 6.6.87)

| Test | Runs | Pass | Crash | Crash Rate |
| --- | --- | --- | --- | --- |
| Heavy (package-level) | 30 | 30 | 0 | 0% |
| Light (targeted) | 30 | 30 | 0 | 0% |

**Conclusion**: The JIT crash was **not reproducible locally**. This suggests the crash is
environment-specific (likely CI's GLIBC 2.35 or Docker memory constraints) or requires
higher memory pressure than a single test file produces. The targeted import fix is still
the correct defensive measure -- it reduces compilation footprint by ~95% per test file,
which lowers the probability of hitting the JIT buffer overflow regardless of environment.

## References

- [Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108) -- JIT crash comprehensive tracking
- [Issue #3330](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3330) -- Document JIT crash workaround
- [Issue #3120](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3120) --
  Core Loss test crashes (follow-up context)
- [ADR-014](../adr/ADR-014-jit-crash-retry-mitigation.md) -- JIT crash retry mitigation (SUPERSEDED)
- [ADR-015](../adr/ADR-015-flaky-required-checks-jit-crash.md) -- Flaky required checks and corrective actions (2026-04-12)

## 0.26.3 Required Checks Impact (2026-04-12)

The JIT compilation volume crash described in this document persists in **Mojo 0.26.3**
(dev2026040705, Ubuntu 24.04, GLIBC 2.39) and manifests in CI as non-deterministic failures
of the two required status checks -- `Core Types & Fuzz` (path: `tests/core/types/`) and
`Integration Tests` (path: `tests/shared/integration/`). Three consecutive `main` runs on
2026-04-12 all failed in a different random subset of groups, confirming the classic
non-deterministic JIT overflow pattern. See
[`repro/issues/jit-compilation-volume-crash.md`](https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/repro/issues/jit-compilation-volume-crash.md)
for the 0.26.3 minimal reproducer.

[ADR-015](../adr/ADR-015-flaky-required-checks-jit-crash.md) formalizes the corrective
actions: (1) audit and convert any remaining package-level `from shared.core import` statements
in the two failing test groups to targeted submodule imports using the Symbol-to-Submodule
Mapping table above; (2) remove `scripts/test-with-retry.sh` and its justfile wiring to make
the retry mitigation's SUPERSEDED status true in the tree. Both actions are tracked under
[Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108).
