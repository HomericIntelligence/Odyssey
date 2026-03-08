"""Tests for shared.testing.tensor_factory module - Part 2: full_tensor and random_tensor.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_factory.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from shared.testing.tensor_factory import (
    zeros_tensor,
    full_tensor,
    random_tensor,
)
from shared.testing.assertions import (
    assert_shape_equal,
    assert_dtype_equal,
    assert_almost_equal,
    assert_true as custom_assert_true,
)


# ============================================================================
# Test full_tensor
# ============================================================================


fn test_full_tensor_float32_positive() raises:
    """Test full_tensor creates float32 tensor with specified positive value."""
    var shape = [10, 5]
    var fill_value = 3.14
    var tensor = full_tensor(shape, fill_value, DType.float32)

    # Check shape
    assert_shape_equal(tensor.shape(), shape)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Check all values match fill_value
    for i in range(50):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, fill_value, tolerance=1e-4)


fn test_full_tensor_float32_negative() raises:
    """Test full_tensor creates float32 tensor with specified negative value."""
    var shape = [5, 4]
    var fill_value = -2.71
    var tensor = full_tensor(shape, fill_value, DType.float32)

    # Check all values match fill_value
    for i in range(20):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, fill_value, tolerance=1e-4)


fn test_full_tensor_int32() raises:
    """Test full_tensor creates int32 tensor with specified value."""
    var shape = [5, 4]
    var fill_value = 42.0
    var tensor = full_tensor(shape, fill_value, DType.int32)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.int32)

    # Check all values match fill_value (as int)
    for i in range(20):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, 42.0, tolerance=1e-6)


# ============================================================================
# Test random_tensor
# ============================================================================


fn test_random_tensor_uniform_bounds_float32() raises:
    """Test random_tensor generates values within specified bounds."""
    var shape = [100, 50]
    var low = -1.0
    var high = 1.0
    var tensor = random_tensor(shape, DType.float32, low, high)

    # Check shape and dtype
    assert_shape_equal(tensor.shape(), shape)
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Check all values are within bounds
    for i in range(5000):
        var val = tensor._get_float64(i)
        custom_assert_true(val >= low, "Value below low bound")
        custom_assert_true(val < high, "Value at or above high bound")


fn test_random_tensor_default_bounds() raises:
    """Test random_tensor with default bounds [0, 1)."""
    var shape = [50, 50]
    var tensor = random_tensor(shape)

    # Check all values are in [0, 1)
    for i in range(2500):
        var val = tensor._get_float64(i)
        custom_assert_true(val >= 0.0, "Value below 0")
        custom_assert_true(val < 1.0, "Value at or above 1")


fn test_random_tensor_int32() raises:
    """Test random_tensor with int32 dtype."""
    var shape = [100]
    var low = 0.0
    var high = 10.0
    var tensor = random_tensor(shape, DType.int32, low, high)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.int32)

    # Check values are integers in range
    for i in range(100):
        var val = tensor._get_float64(i)
        var int_val = Int(val)
        custom_assert_true(int_val >= 0, "Int value below low")
        custom_assert_true(int_val < 10, "Int value at or above high")


fn test_random_tensor_1d() raises:
    """Test random_tensor with 1D shape."""
    var shape = [100]
    var tensor = random_tensor(shape, DType.float32, 0.0, 1.0)
    assert_shape_equal(tensor.shape(), shape)


fn test_random_tensor_3d() raises:
    """Test random_tensor with 3D shape."""
    var shape = [10, 10, 10]
    var tensor = random_tensor(shape, DType.float32, 0.0, 1.0)
    assert_shape_equal(tensor.shape(), shape)


fn main() raises:
    """Run all tests."""
    test_full_tensor_float32_positive()
    test_full_tensor_float32_negative()
    test_full_tensor_int32()

    test_random_tensor_uniform_bounds_float32()
    test_random_tensor_default_bounds()
    test_random_tensor_int32()
    test_random_tensor_1d()
    test_random_tensor_3d()
