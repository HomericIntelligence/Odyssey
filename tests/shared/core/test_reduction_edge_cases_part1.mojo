"""Tests for reduction operation edge cases (Part 1).

Tests edge cases for sum, mean, max_reduce, min_reduce operations including:
- Empty tensor reductions
- Scalar tensor reductions
- Single-element tensor reductions
- Reductions with all same values

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange
from shared.core.reduction import sum, mean, max_reduce, min_reduce

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_true,
)


# ============================================================================
# Test empty tensor reductions
# ============================================================================


fn test_sum_empty_tensor() raises:
    """Sum of empty tensor should return 0 (identity element)."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)
    var result = sum(t)

    assert_numel(result, 1, "Sum of empty should return scalar")
    assert_value_at(result, 0, 0.0, 1e-6, "Sum of empty should be 0")


fn test_mean_empty_tensor() raises:
    """Mean of empty tensor should handle division by 0."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)
    var result = mean(t)

    # Mean of empty: 0/0 = NaN or special handling
    # Verify it doesn't crash
    assert_dim(result, 0, "Mean of empty should return scalar")


# ============================================================================
# Test scalar tensor reductions
# ============================================================================


fn test_sum_scalar_tensor() raises:
    """Sum of 0D scalar tensor should return the scalar itself."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var result = sum(t)

    assert_numel(result, 1, "Sum of scalar should return 1 element")
    assert_value_at(
        result, 0, 42.0, 1e-6, "Sum of scalar should equal the scalar"
    )


fn test_mean_scalar_tensor() raises:
    """Mean of 0D scalar tensor should return the scalar itself."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var result = mean(t)

    assert_numel(result, 1, "Mean of scalar should return 1 element")
    assert_value_at(
        result, 0, 42.0, 1e-6, "Mean of scalar should equal the scalar"
    )


# ============================================================================
# Test single-element tensor reductions
# ============================================================================


fn test_sum_single_element() raises:
    """Sum of [x] should equal x."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 7.5, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 7.5, 1e-6, "Sum of [7.5] should be 7.5")


fn test_mean_single_element() raises:
    """Mean of [x] should equal x."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 7.5, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 7.5, 1e-6, "Mean of [7.5] should be 7.5")


# ============================================================================
# Test reductions with all same values
# ============================================================================


fn test_sum_all_ones() raises:
    """Sum of tensor with all ones."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 10.0, 1e-5, "Sum of 10 ones should be 10")


fn test_mean_all_zeros() raises:
    """Mean of all zeros should be 0."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 0.0, 1e-8, "Mean of zeros should be 0")


fn test_mean_all_same() raises:
    """Mean of tensor with all same values."""
    var shape = List[Int]()
    shape.append(10)
    var t = full(shape, 3.14159, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 3.14159, 1e-5, "Mean of constant tensor")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run reduction edge case tests (Part 1)."""
    print("Running reduction edge case tests (Part 1)...")

    # Empty tensor reductions
    print("  Testing empty tensor reductions...")
    test_sum_empty_tensor()
    test_mean_empty_tensor()

    # Scalar tensor reductions
    print("  Testing scalar tensor reductions...")
    test_sum_scalar_tensor()
    test_mean_scalar_tensor()

    # Single-element tensor reductions
    print("  Testing single-element tensor reductions...")
    test_sum_single_element()
    test_mean_single_element()

    # Reductions with same values
    print("  Testing reductions with same values...")
    test_sum_all_ones()
    test_mean_all_zeros()
    test_mean_all_same()

    print("All reduction edge case tests (Part 1) completed!")
