"""Tests for high-dimensional tensor operations (Part 3: large tensors, precision, numerical behavior).

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_high_dimensional.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests large tensor operations, precision with reductions, and numerical behavior.
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.arithmetic import add, multiply, subtract
from shared.core.reduction import sum, mean, max_reduce, min_reduce

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test large tensor operations
# ============================================================================


fn test_large_multidimensional() raises:
    """Large multidimensional tensor [100, 100, 100, 100]."""
    var shape = List[Int]()
    shape.append(100)
    shape.append(100)
    shape.append(100)
    shape.append(100)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 4, "Tensor should be 4D")
    assert_numel(t, 100000000, "100^4 = 100M elements")


# ============================================================================
# Test precision with high-dimensional reductions
# ============================================================================


fn test_5d_sum_precision() raises:
    """High-dimensional sum should maintain precision."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    shape.append(3)
    shape.append(3)
    shape.append(3)
    var t = full(shape, 1.0, DType.float32)
    var result = sum(t)

    # 3^5 = 243
    assert_value_at(result, 0, 243.0, 1e-3, "Sum preserves precision")


fn test_5d_mean_precision() raises:
    """High-dimensional mean should maintain precision."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 7.5, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 7.5, 1e-5, "Mean of constant tensor")


fn test_6d_max_reduce() raises:
    """Max reduction on 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.float32)
    var result = max_reduce(t)

    assert_value_at(result, 0, 5.0, 1e-6, "Max of constant tensor")


fn test_6d_min_reduce() raises:
    """Min reduction on 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.float32)
    var result = min_reduce(t)

    assert_value_at(result, 0, 5.0, 1e-6, "Min of constant tensor")


# ============================================================================
# Test numerical behavior with many dimensions
# ============================================================================


fn test_5d_accumulation() raises:
    """Accumulation in 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    # Create tensor with values 1.0 each
    var t = ones(shape, DType.float32)

    # Sum should accumulate correctly
    var result = sum(t)
    assert_value_at(result, 0, 32.0, 1e-4, "Accumulation preserves precision")


fn test_6d_mixed_arithmetic() raises:
    """Mixed arithmetic operations on 6D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    # a + b = 3
    var result1 = add(a, b)
    # result1 * 2 = 6
    var result2 = multiply(result1, full(shape, 2.0, DType.float32))

    assert_all_values(result2, 6.0, 1e-5, "Mixed arithmetic correct")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run Part 3 high-dimensional tensor tests (large tensors, precision, numerical behavior).
    """
    print(
        "Running high-dimensional tensor tests (Part 3: large tensors,"
        " precision, numerical behavior)..."
    )

    print("  Testing large tensor operations...")
    test_large_multidimensional()

    print("  Testing precision with reductions...")
    test_5d_sum_precision()
    test_5d_mean_precision()
    test_6d_max_reduce()
    test_6d_min_reduce()

    print("  Testing numerical behavior...")
    test_5d_accumulation()
    test_6d_mixed_arithmetic()

    print("All Part 3 high-dimensional tensor tests completed!")
