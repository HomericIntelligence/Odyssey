# Mojo JIT Crash: `libKGENCompilerRTShared.so` (Mojo v0.26.1)

**Tracking**: Issue #3330, follow-up from #3120

## Root Cause

The JIT crash is triggered by **compilation footprint**, not random instability. When a test
file does `from shared.core import ExTensor, zeros`, the Mojo JIT must compile **all 37,401
lines** across 60+ source files, because `shared/core/__init__.mojo` eagerly re-exports 200+
symbols from 40+ modules. This compilation volume intermittently overflows a JIT-internal
buffer, triggering glibc's `__fortify_fail_abort`.

**Evidence**:

- `__fortify_fail_abort` fires on buffer overflow detection — this is a **compilation-time**
  overflow, not a runtime bug
- `shared/core/__init__.mojo` imports from 40+ submodules including `dtype_dispatch.mojo`
  (176+ monomorphizations) and `elementwise.mojo` (154+ monomorphizations)
- Files using `from shared.core import` (package-level) → compile all 37K lines per test
- Files using `from shared.core.extensor import` (targeted) → compile ~500-2000 lines per test
- The crash is non-deterministic because ASLR, memory layout, and JIT caching vary per run

## Fix Applied

**Convert package-level imports to targeted submodule imports** in all test files.

```mojo
# BEFORE: compiles all 37,401 lines via __init__.mojo
from shared.core import ExTensor, zeros, ones, matmul, relu

# AFTER: compiles only the needed modules (~500-2000 lines)
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.matrix import matmul
from shared.core.activation import relu
```

This fix was applied to 126 test files in the commit that introduced this note. See the
symbol-to-submodule mapping below.

## Symbol-to-Submodule Mapping

| Symbol(s) | Submodule |
|-----------|-----------|
| ExTensor, zeros, ones, full, empty, arange, eye, linspace, ones_like, zeros_like, full_like, nan_tensor, inf_tensor, neg_inf_tensor, clone, item, diff, randn | `shared.core.extensor` |
| reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack, split, tile, repeat, permute, is_contiguous, as_contiguous, view, broadcast_to, ... | `shared.core.shape` |
| add, subtract, multiply, divide, floor_divide, modulo, power, multiply_scalar, *_backward | `shared.core.arithmetic` |
| matmul, transpose, transpose_view, dot, outer, *_backward | `shared.core.matrix` |
| relu, sigmoid, tanh, softmax, gelu, swish, mish, elu, selu, hard_*, *_backward | `shared.core.activation` |
| linear, linear_no_bias, *_backward | `shared.core.linear` |
| conv2d, depthwise_conv2d, depthwise_separable_conv2d, *_backward | `shared.core.conv` |
| maxpool2d, avgpool2d, global_avgpool2d, *_backward | `shared.core.pooling` |
| dropout, dropout2d, *_backward | `shared.core.dropout` |
| batch_norm2d, layer_norm, group_norm, instance_norm, *_backward | `shared.core.normalization` |
| abs, sign, exp, log, sqrt, sin, cos, clip, ceil, floor, round, trunc, logical_*, *_backward | `shared.core.elementwise` |
| equal, not_equal, less, less_equal, greater, greater_equal | `shared.core.comparison` |
| broadcast_shapes, are_shapes_broadcastable, compute_broadcast_strides, BroadcastIterator | `shared.core.broadcasting` |
| xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, he_uniform, he_normal, uniform, normal, constant | `shared.core.initializers` |
| binary_cross_entropy, mean_squared_error, cross_entropy, smooth_l1_loss, hinge_loss, focal_loss, kl_divergence, *_backward | `shared.core.loss` |
| has_nan, has_inf, count_nan, count_inf, check_tensor_safety, tensor_min, tensor_max, ... | `shared.core.numerical_safety` |
| dispatch_unary, dispatch_binary, dispatch_softmax, dispatch_gelu, ... | `shared.core.dtype_dispatch` |
| sum, mean, max_reduce, min_reduce, variance, std, median, percentile, *_backward | `shared.core.reduction` |
| argmax, top_k_indices, top_k, argsort | `shared.core.utils` |
| validate_tensor_shape, validate_tensor_dtype, validate_matching_tensors, ... | `shared.core.validation` |
| TensorMemoryPool, PoolConfig, PoolStats, get_global_pool, pooled_alloc, pooled_free | `shared.core.memory_pool` |
| GradientPair, GradientTriple, GradientQuad, Conv2dNoBiasGradient, ... | `shared.core.gradient_types` |
| BF16, FP8, BF8, FP4, E8M0 | `shared.core.types.dtype_aliases` |
| Module | `shared.core.module` |
| Sequential2, Sequential3 | `shared.core.sequential` |

## Problem

An intermittent crash in the Mojo JIT compiler causes `mojo test` to output `execution crashed`
and exit non-zero. The crash originates in `libKGENCompilerRTShared.so`, the Mojo runtime/compiler
shared library.

Sample crash output:

```text
execution crashed
```

This single line is the entire output. No test names, no stack trace, no assertion failures.

## Diagnosis: Compiler Flake vs. Test Bug

The key diagnostic is **where the crash appears relative to test output**:

| Symptom | Cause |
|---------|-------|
| `execution crashed` appears **before any test output** | Import explosion crash — check import style |
| `execution crashed` or segfault appears **after test output** | Likely a real test bug |
| Specific assertion failure message | Real test bug — investigate |

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
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.activation import relu, sigmoid

# WRONG: compiles all 37K lines, risks JIT crash
from shared.core import ExTensor, zeros, ones, relu, sigmoid
```

## Relationship to Heap Corruption Bug (ADR-009)

This crash is **distinct** from the deterministic heap corruption crash described in
[ADR-009](../adr/ADR-009-heap-corruption-workaround.md):

| | JIT Crash (this doc) | Heap Corruption (ADR-009) |
|-|---------------------|--------------------------|
| **Trigger** | Package-level import compilation overflow | After exactly ~15 cumulative tests in one file |
| **Output** | `execution crashed` before any test runs | Crash mid-run after test output |
| **Root cause** | `__init__.mojo` import explosion → monomorphization overflow | Cumulative allocations exceed JIT heap limit |
| **Fix** | Targeted submodule imports (applied) | Split files to ≤10 tests each (applied) |
| **CI fix** | `continue-on-error` removed once imports converted | File splitting (already applied) |

Both originate in `libKGENCompilerRTShared.so` but are separate bugs with different behaviors and
different workarounds.

## Long-Term Resolution

The targeted submodule import fix (applied in 126 test files) addresses the root cause. The
`continue-on-error: true` workarounds in `comprehensive-tests.yml` can be removed once the
converted tests are confirmed stable across multiple CI runs.

When removing `continue-on-error`:

1. Remove it from the matrix entries: Core Gradient, Data, Shared Infra & Testing, Models,
   Core Types & Fuzz
2. Remove it from standalone jobs: Configs, Benchmarks
3. Verify by running each test group 5+ times in a row — they should pass consistently

## References

- [Issue #3330](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3330) — Document JIT crash workaround
- [Issue #3120](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3120) — Core Loss test crashes (follow-up context)
- [ADR-009](../adr/ADR-009-heap-corruption-workaround.md) — Heap corruption workaround (related but distinct)
- `.github/workflows/comprehensive-tests.yml` — `continue-on-error` mitigations (remove after fix confirmed)
