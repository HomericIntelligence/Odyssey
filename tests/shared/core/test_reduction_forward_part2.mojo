"""Tests for AnyTensor reduction operations - Part 2: mean() and max_reduce() basics.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests reduction operations following the Array API Standard:
mean (keepdims/dtype) and max_reduce (all-elements) with axis=-1.
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
# Test mean() keepdims and dtype
# ============================================================================


fn test_mean_with_keepdims() raises:
    """Test mean with keepdims=True."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 6.0, DType.float32)
    var b = mean(a, keepdims=True)

    # Should be (1, 1) shape instead of scalar
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 6.0, 1e-6, "Mean should be 6.0")


fn test_mean_preserves_dtype() raises:
    """Test that mean preserves dtype."""
    var shape = List[Int]()
    shape.append(4)
    var a = full(shape, 8.0, DType.float64)
    var b = mean(a)

    assert_dtype(b, DType.float64, "Mean should preserve float64 dtype")
    assert_value_at(b, 0, 8.0, 1e-10, "Mean of all 8s should be 8.0")


# ============================================================================
# Test max_reduce()
# ============================================================================


fn test_max_all_same() raises:
    """Test max of all same values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 7.0, DType.float32)
    var b = max_reduce(a)

    assert_dim(b, 0, "Max should return scalar")
    assert_value_at(b, 0, 7.0, 1e-6, "Max of all 7s should be 7.0")


fn test_max_arange() raises:
    """Test max of range [0, 1, 2, 3, 4]."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)
    var b = max_reduce(a)

    assert_value_at(b, 0, 4.0, 1e-6, "Max of [0,1,2,3,4] should be 4.0")


fn test_max_negative_values() raises:
    """Test max with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -3.0, DType.float32)
    var b = max_reduce(a)

    assert_value_at(b, 0, -3.0, 1e-6, "Max of all -3s should be -3.0")


fn test_max_with_keepdims() raises:
    """Test max with keepdims=True."""
    var shape2d = List[Int]()
    shape2d.append(3)
    shape2d.append(4)
    var a2d = full(shape2d, 9.0, DType.float32)
    var b = max_reduce(a2d, keepdims=True)

    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 9.0, 1e-6, "Max should be 9.0")


fn test_max_preserves_dtype() raises:
    """Test that max preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = arange(0.0, 5.0, 1.0, DType.float64)
    var b = max_reduce(a)

    assert_dtype(b, DType.float64, "Max should preserve float64 dtype")
    assert_value_at(b, 0, 4.0, 1e-10, "Max should be 4.0")


fn test_min_all_same() raises:
    """Test min of all same values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 3.0, DType.float32)
    var b = min_reduce(a)

    assert_dim(b, 0, "Min should return scalar")
    assert_value_at(b, 0, 3.0, 1e-6, "Min of all 3s should be 3.0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run mean (keepdims/dtype) and max_reduce basic tests."""
    print("Running AnyTensor reduction forward tests - Part 2...")

    # mean() keepdims/dtype tests
    print("  Testing mean() keepdims and dtype...")
    test_mean_with_keepdims()
    test_mean_preserves_dtype()

    # max_reduce() tests
    print("  Testing max_reduce()...")
    test_max_all_same()
    test_max_arange()
    test_max_negative_values()
    test_max_with_keepdims()
    test_max_preserves_dtype()

    # first min_reduce test
    print("  Testing min_reduce() basics...")
    test_min_all_same()

    print("All Part 2 reduction forward tests completed!")
