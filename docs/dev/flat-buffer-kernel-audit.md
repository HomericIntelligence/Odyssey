# Flat-Buffer Kernel Contiguity Audit

**Issue**: #3800 - Add is_contiguous() guard to all flat-buffer kernels
**Date**: 2026-03-15
**Status**: Partial audit with recommendations

## Executive Summary

This audit examines all Mojo kernels that use `_data.bitcast[T]()[i]` for raw pointer arithmetic
to identify which ones need `as_contiguous()` guards before calling the kernels. Raw pointer
arithmetic is unsafe on non-contiguous views (transposed matrices, slices, permuted axes) as it
ignores memory strides and will produce silently incorrect results.

## Key Finding

A reference pattern exists in `src/odyssey/core/matrix.mojo` (lines 604-611) that demonstrates the
correct guard pattern. This should be replicated across other modules.

## Contiguity Guard Pattern

**Correct Pattern** (from matrix.mojo):

```mojo
var input_cont: AnyTensor
if input.is_contiguous():
    input_cont = input
else:
    input_cont = as_contiguous(input)
```

Alternative (conditional application):

```mojo
if not a.is_contiguous():
    return False  # Skip fast path
if not b.is_contiguous():
    return False  # Skip fast path
```

## Files Using `_data.bitcast` (758 total usages across 35 files)

### Critical Path: Reduction Operations

**File**: `src/odyssey/core/reduction.mojo`

Functions using raw pointer arithmetic:

- `_reduce_all_impl()` (line 40) - Generic reduction kernel
- Other dispatch functions that call into dtype-specialized kernels

**Issue**: Reduction operations do not validate contiguity before bitcasting.

**Status**: Requires audit - check all callers of `reduce_sum()`, `reduce_mean()`, etc.

### Critical Path: Elementwise Operations

**File**: `src/odyssey/core/elementwise.mojo`

Elementwise operations (element-by-element operations like `add_elementwise()`, `multiply()`,
`relu()`, etc.) use bitcast for fast indexing.

**Issue**: Many elementwise functions don't check contiguity.

**Status**: Requires audit - check which elementwise functions have fast paths needing guards.

### Critical Path: Convolution Operations

**File**: `src/odyssey/core/conv.mojo`

Functions:

- `_conv2d_kernel()` (complex stride calculations)
- `conv2d()` and related functions

**Issue**: Conv operations use raw pointer arithmetic for kernel application.

**Status**: Requires audit - conv2d is performance-critical and may need conditional checks.

### Critical Path: Activation Functions

**File**: `src/odyssey/core/activation.mojo` and `src/odyssey/core/activation_simd.mojo`

Functions:

- `relu()`, `sigmoid()`, `tanh()`, etc.
- SIMD-vectorized variants

**Status**: Requires audit - activation functions are in the forward pass critical path.

### Already Protected: Matrix Operations

**File**: `src/odyssey/core/matrix.mojo` ✓

Lines 604-611 show the pattern:

```mojo
if a.is_contiguous():
    a_cont = a
else:
    a_cont = as_contiguous(a)
```

**Status**: Already protected in the main matrix multiplication entry point.

### Already Protected: Arithmetic Operations

**File**: `src/odyssey/core/arithmetic_contiguous.mojo` ✓

Lines 82-85 validate contiguity for each input:

```mojo
if not a.is_contiguous():
    return False
if not b.is_contiguous():
    return False
```

**Status**: Already has guards - returns False for non-contiguous inputs, falling back to other
implementations.

### Other Files Needing Review

- `src/odyssey/core/pooling.mojo` - Pooling kernels use raw pointer access
- `src/odyssey/core/normalization_simd.mojo` - Batch/layer normalization
- `src/odyssey/core/comparison.mojo` - Comparison operations
- `src/odyssey/core/dtype_dispatch.mojo` - Generic dispatch templates
- `src/odyssey/training/gradient_ops.mojo` - Backward pass operations
- `src/odyssey/training/loops/training_loop.mojo` - Training loop kernels

## Recommended Approach

### Phase 1: Identify Fast Paths (High Priority)

Search for patterns where bitcast is used directly without stride awareness:

```bash
grep -n "_data\.bitcast\[.*\]()\[i\]" src/odyssey/core/*.mojo
```

Identify fast-path patterns that:

1. Assume contiguous memory layout
2. Use simple index-based access
3. Are called from high-level functions without guards

### Phase 2: Add Guards or Skip Checks

For each identified kernel, choose one of:

**Option A: Add input guard** (preferred for widely-used kernels)

```mojo
fn my_kernel_impl(tensor: AnyTensor) raises:
    var t_cont: AnyTensor
    if tensor.is_contiguous():
        t_cont = tensor
    else:
        t_cont = as_contiguous(tensor)
    # Fast path using bitcast on t_cont
```

**Option B: Skip fast path** (preferred for already-protected code)

```mojo
fn my_kernel_fast(a: AnyTensor, b: AnyTensor) raises -> Optional[AnyTensor]:
    if not a.is_contiguous() or not b.is_contiguous():
        return None  # Skip fast path, use generic implementation
    # Fast path using bitcast
```

**Option C: Caller responsibility** (if caller already has guard)

- Leave kernel as-is if documented that it requires contiguous input
- Ensure all callers are protected

### Phase 3: Test Coverage

Existing test files for non-contiguous support:

- `tests/odyssey/testing/test_gradient_checker_noncont_tensors.mojo` (new, #3801)

Additional test requirements:

- Each guarded kernel should have a transposed/non-contiguous test case
- Verify correctness matches contiguous execution
- Benchmark contiguous vs as_contiguous() performance impact

## Implementation Notes

### `as_contiguous()` Performance Cost

From `src/odyssey/core/shape.mojo`:

- Always copies data (no zero-copy optimization)
- Uses memcpy for bulk copy - O(n) where n = tensor.numel()
- Should only be called for non-contiguous inputs

### Memory Layout Facts

- Contiguous tensor: C-order layout, strides = [d1*d2*..., d2*d3*..., ..., 1]
- Non-contiguous view: Arbitrary strides, may skip elements
- Raw `ptr[i]` access only works for contiguous layouts
- Use `_get_float64()` / `_set_float64()` for stride-aware access (slower but correct)

### Stride-Aware Access

The gradient_checker already demonstrates stride-aware access pattern:

- `tensor._get_float64(i)` - Handles strides correctly
- `tensor._set_float64(i, val)` - Handles strides correctly

See `src/odyssey/testing/gradient_checker.mojo` for examples of correct non-contiguous handling.

## Risk Assessment

**Latent Bugs**: Operations that accept non-contiguous inputs and silently produce wrong results.

- Affects: Transposed weight matrices, sliced tensors, permuted axes
- Impact: Training produces incorrect gradients without error
- Detection: Only visible via gradient checking with non-contiguous inputs (now available in #3801)

**Examples**:

- `relu(transpose(x))` would compute wrong activation
- `matmul(a, transpose(b))` would compute wrong matrix product
- `sum(slice(x))` would compute wrong reduction

## Testing Strategy

1. For each guarded kernel, create a test with non-contiguous input
2. Use transpose_view() or custom permutation to create non-contiguous tensors
3. Compare output against reference contiguous implementation
4. Verify performance impact of contiguity guards

Example test pattern:

```mojo
fn test_relu_noncont() raises:
    var x = full([3, 4], 0.5, DType.float32)
    var x_nc = transpose_view(x)  # Non-contiguous

    var y1 = relu(x)
    var y2 = relu(x_nc)

    assert_equal(y1, y2, "relu should produce same output")
```

## References

- Issue #3236: Original fix that added matmul guard (reference implementation)
- Issue #3801: Non-contiguous gradient checker tests (now available)
- `docs/dev/extensor-view-contract.md`: View semantics documentation
- `src/odyssey/core/matrix.mojo`: Line 604-611 - Guard pattern reference

## Next Steps

1. Complete audit by categorizing all 35 files using bitcast
2. Prioritize critical path files (reduction, elementwise, activation)
3. Apply guards following the matrix.mojo pattern
4. Add non-contiguous test for each guarded kernel
5. Measure performance impact of contiguity checks
