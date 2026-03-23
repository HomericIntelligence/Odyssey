# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor creation operations - Part 1: zeros() and ones().

Tests zeros() and ones() creation functions with various shapes and dtypes.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import AnyTensor and creation operations
from shared.tensor.any_tensor import AnyTensor, zeros, ones

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test zeros()
# ============================================================================


fn test_zeros_1d_float32() raises:
    """Test creating 1D tensor of zeros with float32."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 1, "zeros 1D should have 1 dimension")
    assert_numel(t, 5, "zeros 1D should have 5 elements")
    assert_dtype(t, DType.float32, "zeros should have float32 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_2d_int64() raises:
    """Test creating 2D tensor of zeros with int64."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.int64)

    assert_dim(t, 2, "zeros 2D should have 2 dimensions")
    assert_numel(t, 12, "zeros 2D(3,4) should have 12 elements")
    assert_dtype(t, DType.int64, "zeros should have int64 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_3d_float64() raises:
    """Test creating 3D tensor of zeros with float64."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float64)

    assert_dim(t, 3, "zeros 3D should have 3 dimensions")
    assert_numel(t, 24, "zeros 3D(2,3,4) should have 24 elements")
    assert_dtype(t, DType.float64, "zeros should have float64 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_empty_shape() raises:
    """Test creating 0D scalar tensor of zeros."""
    var shape = List[Int]()
    var t = zeros(shape, DType.float32)

    assert_dim(t, 0, "zeros 0D should have 0 dimensions")
    assert_numel(t, 1, "zeros 0D should have 1 element")
    assert_dtype(t, DType.float32, "zeros should have float32 dtype")
    assert_value_at(t, 0, 0.0, 1e-8, "zeros 0D should be 0.0")


fn test_zeros_large_shape() raises:
    """Test creating zeros with very large shape."""
    var shape = List[Int]()
    shape.append(10000)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 10000, "zeros large should have 10000 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "zeros first element should be 0.0")
    assert_value_at(t, 5000, 0.0, 1e-8, "zeros middle element should be 0.0")
    assert_value_at(t, 9999, 0.0, 1e-8, "zeros last element should be 0.0")


# ============================================================================
# Test ones()
# ============================================================================


fn test_ones_1d_float32() raises:
    """Test creating 1D tensor of ones with float32."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    assert_dim(t, 1, "ones 1D should have 1 dimension")
    assert_numel(t, 5, "ones 1D should have 5 elements")
    assert_dtype(t, DType.float32, "ones should have float32 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


fn test_ones_2d_int32() raises:
    """Test creating 2D tensor of ones with int32."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.int32)

    assert_dim(t, 2, "ones 2D should have 2 dimensions")
    assert_numel(t, 12, "ones 2D(3,4) should have 12 elements")
    assert_dtype(t, DType.int32, "ones should have int32 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


fn test_ones_3d_float64() raises:
    """Test creating 3D tensor of ones with float64."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float64)

    assert_dim(t, 3, "ones 3D should have 3 dimensions")
    assert_numel(t, 24, "ones 3D(2,3,4) should have 24 elements")
    assert_dtype(t, DType.float64, "ones should have float64 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run zeros() and ones() creation tests."""
    print("Running AnyTensor creation tests - Part 1: zeros() and ones()...")

    # zeros() tests
    test_zeros_1d_float32()
    test_zeros_2d_int64()
    test_zeros_3d_float64()
    test_zeros_empty_shape()
    test_zeros_large_shape()

    # ones() tests
    test_ones_1d_float32()
    test_ones_2d_int32()
    test_ones_3d_float64()

    print("All Part 1 creation tests completed!")
