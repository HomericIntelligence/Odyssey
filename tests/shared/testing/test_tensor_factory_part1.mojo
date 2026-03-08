"""Tests for shared.testing.tensor_factory module - Part 1: zeros_tensor and ones_tensor.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_factory.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from shared.testing.tensor_factory import (
    zeros_tensor,
    ones_tensor,
)
from shared.testing.assertions import (
    assert_shape_equal,
    assert_dtype_equal,
    assert_almost_equal,
)


# ============================================================================
# Test zeros_tensor
# ============================================================================


fn test_zeros_tensor_float32() raises:
    """Test zeros_tensor creates float32 tensor with all zeros."""
    var shape = [10, 5]
    var tensor = zeros_tensor(shape, DType.float32)

    # Check shape
    assert_shape_equal(tensor.shape(), shape)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Check all values are zero
    for i in range(50):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, 0.0, tolerance=1e-6)


fn test_zeros_tensor_int32() raises:
    """Test zeros_tensor creates int32 tensor with all zeros."""
    var shape = [5, 4]
    var tensor = zeros_tensor(shape, DType.int32)

    # Check shape
    assert_shape_equal(tensor.shape(), shape)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.int32)

    # Check all values are zero
    for i in range(20):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, 0.0, tolerance=1e-6)


fn test_zeros_tensor_1d() raises:
    """Test zeros_tensor with 1D shape."""
    var shape = [10]
    var tensor = zeros_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


fn test_zeros_tensor_3d() raises:
    """Test zeros_tensor with 3D shape."""
    var shape = [2, 3, 4]
    var tensor = zeros_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


# ============================================================================
# Test ones_tensor
# ============================================================================


fn test_ones_tensor_float32() raises:
    """Test ones_tensor creates float32 tensor with all ones."""
    var shape = [10, 5]
    var tensor = ones_tensor(shape, DType.float32)

    # Check shape
    assert_shape_equal(tensor.shape(), shape)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Check all values are one
    for i in range(50):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, 1.0, tolerance=1e-6)


fn test_ones_tensor_int32() raises:
    """Test ones_tensor creates int32 tensor with all ones."""
    var shape = [5, 4]
    var tensor = ones_tensor(shape, DType.int32)

    # Check shape
    assert_shape_equal(tensor.shape(), shape)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.int32)

    # Check all values are one
    for i in range(20):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, 1.0, tolerance=1e-6)


fn test_ones_tensor_1d() raises:
    """Test ones_tensor with 1D shape."""
    var shape = [10]
    var tensor = ones_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


fn test_ones_tensor_3d() raises:
    """Test ones_tensor with 3D shape."""
    var shape = [2, 3, 4]
    var tensor = ones_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


fn main() raises:
    """Run all tests."""
    test_zeros_tensor_float32()
    test_zeros_tensor_int32()
    test_zeros_tensor_1d()
    test_zeros_tensor_3d()

    test_ones_tensor_float32()
    test_ones_tensor_int32()
    test_ones_tensor_1d()
    test_ones_tensor_3d()
