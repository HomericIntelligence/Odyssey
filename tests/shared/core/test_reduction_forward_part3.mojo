"""Tests for AnyTensor reduction operations - Part 3: min_reduce() and axis-specific sum/mean.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests reduction operations following the Array API Standard:
min_reduce (all-elements) and sum/mean with axis parameter.
"""

# Import AnyTensor and reduction operations
from shared.tensor.any_tensor import AnyTensor, full, ones, zeros, arange
from shared.core.reduction import sum, mean, max_reduce, min_reduce

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
)


# ============================================================================
# Test min_reduce() continued
# ============================================================================


fn test_min_arange() raises:
    """Test min of range [0, 1, 2, 3, 4]."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)
    var b = min_reduce(a)

    assert_value_at(b, 0, 0.0, 1e-6, "Min of [0,1,2,3,4] should be 0.0")


fn test_min_negative_values() raises:
    """Test min with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -7.0, DType.float32)
    var b = min_reduce(a)

    assert_value_at(b, 0, -7.0, 1e-6, "Min of all -7s should be -7.0")


fn test_min_with_keepdims() raises:
    """Test min with keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 2.5, DType.float32)
    var b = min_reduce(a, keepdims=True)

    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 2.5, 1e-6, "Min should be 2.5")


fn test_min_preserves_dtype() raises:
    """Test that min preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = arange(1.0, 6.0, 1.0, DType.float64)
    var b = min_reduce(a)

    assert_dtype(b, DType.float64, "Min should preserve float64 dtype")
    assert_value_at(b, 0, 1.0, 1e-10, "Min should be 1.0")


# ============================================================================
# Test axis-specific reductions
# ============================================================================


fn test_sum_axis_0() raises:
    """Test sum along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 2.0, DType.float32)  # 3x4 matrix of 2s
    var b = sum(a, axis=0)  # Sum along rows -> shape (4,)

    # Should sum 3 values (each 2.0) per column
    assert_dim(b, 1, "Sum along axis 0 should be 1D")
    assert_numel(b, 4, "Sum along axis 0 should have 4 elements")
    assert_value_at(b, 0, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 1, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 2, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 3, 6.0, 1e-6, "Each column sum should be 6.0")


fn test_sum_axis_1() raises:
    """Test sum along axis 1."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 2.0, DType.float32)  # 3x4 matrix of 2s
    var b = sum(a, axis=1)  # Sum along columns -> shape (3,)

    # Should sum 4 values (each 2.0) per row
    assert_dim(b, 1, "Sum along axis 1 should be 1D")
    assert_numel(b, 3, "Sum along axis 1 should have 3 elements")
    assert_value_at(b, 0, 8.0, 1e-6, "Each row sum should be 8.0")
    assert_value_at(b, 1, 8.0, 1e-6, "Each row sum should be 8.0")
    assert_value_at(b, 2, 8.0, 1e-6, "Each row sum should be 8.0")


fn test_sum_axis_keepdims() raises:
    """Test sum with axis and keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = sum(a, axis=0, keepdims=True)

    # Should be shape (1, 4) instead of (4,)
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_numel(b, 4, "Should have 4 elements")


fn test_mean_axis_0() raises:
    """Test mean along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 6.0, DType.float32)  # 3x4 matrix of 6s
    var b = mean(a, axis=0)  # Mean along rows -> shape (4,)

    # Should average 3 values (each 6.0) per column
    assert_dim(b, 1, "Mean along axis 0 should be 1D")
    assert_numel(b, 4, "Mean along axis 0 should have 4 elements")
    assert_value_at(b, 0, 6.0, 1e-6, "Each column mean should be 6.0")
    assert_value_at(b, 1, 6.0, 1e-6, "Each column mean should be 6.0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run min_reduce and axis-specific sum/mean tests."""
    print("Running AnyTensor reduction forward tests - Part 3...")

    # min_reduce() tests
    print("  Testing min_reduce()...")
    test_min_arange()
    test_min_negative_values()
    test_min_with_keepdims()
    test_min_preserves_dtype()

    # Axis-specific sum tests
    print("  Testing axis-specific sum()...")
    test_sum_axis_0()
    test_sum_axis_1()
    test_sum_axis_keepdims()

    # Axis-specific mean tests
    print("  Testing axis-specific mean()...")
    test_mean_axis_0()

    print("All Part 3 reduction forward tests completed!")
