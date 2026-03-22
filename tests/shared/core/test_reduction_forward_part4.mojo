"""Tests for AnyTensor reduction operations - Part 4: axis-specific max/min and consistency.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests reduction operations following the Array API Standard:
mean axis=1, max_reduce/min_reduce with axis parameter, and cross-reduction consistency.
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
# Test axis-specific reductions continued
# ============================================================================


fn test_mean_axis_1() raises:
    """Test mean along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(5)
    var a = full(shape, 10.0, DType.float32)  # 2x5 matrix of 10s
    var b = mean(a, axis=1)  # Mean along columns -> shape (2,)

    # Should average 5 values (each 10.0) per row
    assert_dim(b, 1, "Mean along axis 1 should be 1D")
    assert_numel(b, 2, "Mean along axis 1 should have 2 elements")
    assert_value_at(b, 0, 10.0, 1e-6, "Each row mean should be 10.0")
    assert_value_at(b, 1, 10.0, 1e-6, "Each row mean should be 10.0")


fn test_max_axis_0() raises:
    """Test max along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.0, DType.float32)
    var b = max_reduce(a, axis=0)

    # Should find max of 3 values per column
    assert_dim(b, 1, "Max along axis 0 should be 1D")
    assert_numel(b, 4, "Max along axis 0 should have 4 elements")
    assert_value_at(b, 0, 7.0, 1e-6, "Max should be 7.0")


fn test_max_axis_1() raises:
    """Test max along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 9.0, DType.float32)
    var b = max_reduce(a, axis=1)

    # Should find max of 3 values per row
    assert_dim(b, 1, "Max along axis 1 should be 1D")
    assert_numel(b, 2, "Max along axis 1 should have 2 elements")
    assert_value_at(b, 0, 9.0, 1e-6, "Max should be 9.0")
    assert_value_at(b, 1, 9.0, 1e-6, "Max should be 9.0")


fn test_min_axis_0() raises:
    """Test min along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 3.5, DType.float32)
    var b = min_reduce(a, axis=0)

    # Should find min of 3 values per column
    assert_dim(b, 1, "Min along axis 0 should be 1D")
    assert_numel(b, 4, "Min along axis 0 should have 4 elements")
    assert_value_at(b, 0, 3.5, 1e-6, "Min should be 3.5")


fn test_min_axis_1() raises:
    """Test min along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 2.5, DType.float32)
    var b = min_reduce(a, axis=1)

    # Should find min of 3 values per row
    assert_dim(b, 1, "Min along axis 1 should be 1D")
    assert_numel(b, 2, "Min along axis 1 should have 2 elements")
    assert_value_at(b, 0, 2.5, 1e-6, "Min should be 2.5")
    assert_value_at(b, 1, 2.5, 1e-6, "Min should be 2.5")


# ============================================================================
# Test reduction combinations
# ============================================================================


fn test_reductions_consistent() raises:
    """Test that reductions are consistent with each other."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 5.0, DType.float32)

    var sum_result = sum(a)
    var mean_result = mean(a)
    var max_result = max_reduce(a)
    var min_result = min_reduce(a)

    # For all same values:
    # sum = n * value
    # mean = value
    # max = value
    # min = value
    assert_value_at(sum_result, 0, 50.0, 1e-6, "Sum should be 10 * 5 = 50")
    assert_value_at(mean_result, 0, 5.0, 1e-6, "Mean should be 5")
    assert_value_at(max_result, 0, 5.0, 1e-6, "Max should be 5")
    assert_value_at(min_result, 0, 5.0, 1e-6, "Min should be 5")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run axis-specific max/min and consistency tests."""
    print("Running AnyTensor reduction forward tests - Part 4...")

    # Axis-specific mean test
    print("  Testing axis-specific mean()...")
    test_mean_axis_1()

    # Axis-specific max_reduce tests
    print("  Testing axis-specific max_reduce()...")
    test_max_axis_0()
    test_max_axis_1()

    # Axis-specific min_reduce tests
    print("  Testing axis-specific min_reduce()...")
    test_min_axis_0()
    test_min_axis_1()

    # Combination tests
    print("  Testing reduction consistency...")
    test_reductions_consistent()

    print("All Part 4 reduction forward tests completed!")
