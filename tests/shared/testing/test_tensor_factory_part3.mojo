"""Tests for shared.testing.tensor_factory module - Part 3: random_normal_tensor and set_tensor_value (first half).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_factory.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from math import sqrt
from shared.testing.tensor_factory import (
    zeros_tensor,
    ones_tensor,
    random_normal_tensor,
    set_tensor_value,
)
from shared.testing.assertions import (
    assert_shape_equal,
    assert_dtype_equal,
    assert_almost_equal,
    assert_true as custom_assert_true,
)


# ============================================================================
# Test random_normal_tensor
# ============================================================================


fn test_random_normal_tensor_default_params() raises:
    """Test random_normal_tensor with default mean=0, std=1."""
    var shape = [1000]
    var tensor = random_normal_tensor(shape, DType.float32)

    # Check shape and dtype
    assert_shape_equal(tensor.shape(), shape)
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Calculate empirical mean and std (rough check)
    var sum_val = 0.0
    var sum_sq = 0.0
    for i in range(1000):
        var val = tensor._get_float64(i)
        sum_val += val
        sum_sq += val * val

    var empirical_mean = sum_val / 1000.0
    var empirical_var = (sum_sq / 1000.0) - (empirical_mean * empirical_mean)
    var empirical_std = sqrt(empirical_var)

    # Check mean is roughly 0 (loose tolerance due to random sampling)
    assert_almost_equal(empirical_mean, 0.0, tolerance=0.1)

    # Check std is roughly 1 (loose tolerance due to random sampling)
    assert_almost_equal(empirical_std, 1.0, tolerance=0.3)


fn test_random_normal_tensor_custom_mean_std() raises:
    """Test random_normal_tensor with custom mean and std."""
    var shape = [500]
    var mean = 5.0
    var std = 2.0
    var tensor = random_normal_tensor(shape, DType.float32, mean, std)

    # Check shape and dtype
    assert_shape_equal(tensor.shape(), shape)
    assert_dtype_equal(tensor.dtype(), DType.float32)

    # Calculate empirical mean and std
    var sum_val = 0.0
    var sum_sq = 0.0
    for i in range(500):
        var val = tensor._get_float64(i)
        sum_val += val
        sum_sq += val * val

    var empirical_mean = sum_val / 500.0
    var empirical_var = (sum_sq / 500.0) - (empirical_mean * empirical_mean)
    var empirical_std = sqrt(empirical_var)

    # Check mean is roughly as specified
    assert_almost_equal(empirical_mean, mean, tolerance=0.5)

    # Check std is roughly as specified
    assert_almost_equal(empirical_std, std, tolerance=0.5)


fn test_random_normal_tensor_int32() raises:
    """Test random_normal_tensor with int32 dtype."""
    var shape = [100]
    var tensor = random_normal_tensor(shape, DType.int32, mean=0.0, std=1.0)

    # Check dtype
    assert_dtype_equal(tensor.dtype(), DType.int32)

    # Values should be reasonable integers (some negatives, some positive)
    var has_positive = False
    var has_negative = False
    for i in range(100):
        var val = tensor._get_float64(i)
        if val > 0.0:
            has_positive = True
        if val < 0.0:
            has_negative = True

    custom_assert_true(
        has_positive or has_negative, "Should have some non-zero values"
    )


fn test_random_normal_tensor_1d() raises:
    """Test random_normal_tensor with 1D shape."""
    var shape = [100]
    var tensor = random_normal_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


fn test_random_normal_tensor_3d() raises:
    """Test random_normal_tensor with 3D shape."""
    var shape = [5, 5, 5]
    var tensor = random_normal_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


# ============================================================================
# Test set_tensor_value (first half)
# ============================================================================


fn test_set_tensor_value_float32() raises:
    """Test set_tensor_value with float32 dtype."""
    var shape = [10, 5]
    var tensor = zeros_tensor(shape, DType.float32)

    # Set specific values
    set_tensor_value(tensor, 0, 1.5, DType.float32)
    set_tensor_value(tensor, 10, -2.5, DType.float32)
    set_tensor_value(tensor, 49, 3.14, DType.float32)

    # Verify values were set correctly
    var val0 = tensor._get_float64(0)
    var val10 = tensor._get_float64(10)
    var val49 = tensor._get_float64(49)

    assert_almost_equal(val0, 1.5, tolerance=1e-4)
    assert_almost_equal(val10, -2.5, tolerance=1e-4)
    assert_almost_equal(val49, 3.14, tolerance=1e-4)


fn test_set_tensor_value_int32() raises:
    """Test set_tensor_value with int32 dtype."""
    var shape = [10]
    var tensor = zeros_tensor(shape, DType.int32)

    # Set specific values
    set_tensor_value(tensor, 0, 42.0, DType.int32)
    set_tensor_value(tensor, 5, -10.0, DType.int32)
    set_tensor_value(tensor, 9, 99.0, DType.int32)

    # Verify values were set correctly
    var val0 = tensor._get_float64(0)
    var val5 = tensor._get_float64(5)
    var val9 = tensor._get_float64(9)

    assert_almost_equal(val0, 42.0, tolerance=1e-6)
    assert_almost_equal(val5, -10.0, tolerance=1e-6)
    assert_almost_equal(val9, 99.0, tolerance=1e-6)


fn test_set_tensor_value_overwrite() raises:
    """Test set_tensor_value overwrites previous values."""
    var shape = [5]
    var tensor = ones_tensor(shape, DType.float32)

    # Verify all ones initially
    var initial_val = tensor._get_float64(0)
    assert_almost_equal(initial_val, 1.0, tolerance=1e-6)

    # Overwrite with new value
    set_tensor_value(tensor, 0, 2.0, DType.float32)

    # Verify overwritten value
    var new_val = tensor._get_float64(0)
    assert_almost_equal(new_val, 2.0, tolerance=1e-6)

    # Verify other values unchanged
    var other_val = tensor._get_float64(1)
    assert_almost_equal(other_val, 1.0, tolerance=1e-6)


fn main() raises:
    """Run all tests."""
    test_random_normal_tensor_default_params()
    test_random_normal_tensor_custom_mean_std()
    test_random_normal_tensor_int32()
    test_random_normal_tensor_1d()
    test_random_normal_tensor_3d()

    test_set_tensor_value_float32()
    test_set_tensor_value_int32()
    test_set_tensor_value_overwrite()
