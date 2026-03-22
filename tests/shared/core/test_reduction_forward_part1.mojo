"""Tests for AnyTensor reduction operations - Part 1: sum() basics.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests reduction operations following the Array API Standard:
sum (all-elements and basic mean) with all-elements reduction (axis=-1).
"""

# Import AnyTensor and reduction operations
from shared.core.any_tensor import AnyTensor, full, ones, zeros, arange
from shared.core.reduction import sum, mean, max_reduce, min_reduce

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
)


# ============================================================================
# Test sum()
# ============================================================================


fn test_sum_all_ones() raises:
    """Test sum of all ones."""
    var shape = List[Int]()
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = sum(a)  # Sum all elements

    assert_dim(b, 0, "Sum should return scalar (0D)")
    assert_numel(b, 1, "Scalar should have 1 element")
    assert_value_at(b, 0, 10.0, 1e-6, "Sum of 10 ones should be 10.0")


fn test_sum_2d_tensor() raises:
    """Test sum of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 2.0, DType.float32)
    var b = sum(a)  # Sum all 12 elements

    assert_dim(b, 0, "Sum should return scalar")
    assert_value_at(b, 0, 24.0, 1e-6, "Sum of 12 twos should be 24.0")


fn test_sum_arange() raises:
    """Test sum of range [0, 1, 2, 3, 4]."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)
    var b = sum(a)

    # 0 + 1 + 2 + 3 + 4 = 10
    assert_value_at(b, 0, 10.0, 1e-6, "Sum of [0,1,2,3,4] should be 10.0")


fn test_sum_with_keepdims() raises:
    """Test sum with keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = sum(a, keepdims=True)

    # Should be (1, 1) shape instead of scalar
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 12.0, 1e-6, "Sum should still be 12.0")


fn test_sum_preserves_dtype() raises:
    """Test that sum preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = sum(a)

    assert_dtype(b, DType.float64, "Sum should preserve float64 dtype")
    assert_value_at(b, 0, 5.0, 1e-10, "Sum of 5 ones should be 5.0")


# ============================================================================
# Test mean() basics
# ============================================================================


fn test_mean_all_ones() raises:
    """Test mean of all ones."""
    var shape = List[Int]()
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = mean(a)

    assert_dim(b, 0, "Mean should return scalar")
    assert_value_at(b, 0, 1.0, 1e-6, "Mean of ones should be 1.0")


fn test_mean_2d_tensor() raises:
    """Test mean of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 5.0, DType.float32)
    var b = mean(a)

    assert_dim(b, 0, "Mean should return scalar")
    assert_value_at(b, 0, 5.0, 1e-6, "Mean of all 5s should be 5.0")


fn test_mean_arange() raises:
    """Test mean of range [0, 1, 2, 3, 4]."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)
    var b = mean(a)

    # (0 + 1 + 2 + 3 + 4) / 5 = 10 / 5 = 2.0
    assert_value_at(b, 0, 2.0, 1e-6, "Mean of [0,1,2,3,4] should be 2.0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run sum and basic mean reduction tests."""
    print("Running AnyTensor reduction forward tests - Part 1...")

    # sum() tests
    print("  Testing sum()...")
    test_sum_all_ones()
    test_sum_2d_tensor()
    test_sum_arange()
    test_sum_with_keepdims()
    test_sum_preserves_dtype()

    # mean() basic tests
    print("  Testing mean() basics...")
    test_mean_all_ones()
    test_mean_2d_tensor()
    test_mean_arange()

    print("All Part 1 reduction forward tests completed!")
