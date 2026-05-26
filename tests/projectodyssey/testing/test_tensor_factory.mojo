"""Tests for projectodyssey.testing.tensor_factory module.

"""


from std.testing import assert_true, assert_equal
from projectodyssey.testing.tensor_factory import (
    full_tensor,
    ones_tensor,
    random_normal_tensor,
    random_tensor,
    set_tensor_value,
    zeros_tensor,
)
from projectodyssey.testing.assertions import (
    assert_almost_equal,
    assert_dtype_equal,
    assert_shape_equal,
    assert_true as custom_assert_true,
)
from std.math import sqrt


def test_zeros_tensor_float32() raises:
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


def test_zeros_tensor_int32() raises:
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


def test_zeros_tensor_1d() raises:
    """Test zeros_tensor with 1D shape."""
    var shape = [10]
    var tensor = zeros_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_zeros_tensor_3d() raises:
    """Test zeros_tensor with 3D shape."""
    var shape = [2, 3, 4]
    var tensor = zeros_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_ones_tensor_float32() raises:
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


def test_ones_tensor_int32() raises:
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


def test_ones_tensor_1d() raises:
    """Test ones_tensor with 1D shape."""
    var shape = [10]
    var tensor = ones_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_ones_tensor_3d() raises:
    """Test ones_tensor with 3D shape."""
    var shape = [2, 3, 4]
    var tensor = ones_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_full_tensor_float32_positive() raises:
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


def test_full_tensor_float32_negative() raises:
    """Test full_tensor creates float32 tensor with specified negative value."""
    var shape = [5, 4]
    var fill_value = -2.71
    var tensor = full_tensor(shape, fill_value, DType.float32)

    # Check all values match fill_value
    for i in range(20):
        var val = tensor._get_float64(i)
        assert_almost_equal(val, fill_value, tolerance=1e-4)


def test_full_tensor_int32() raises:
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


def test_random_tensor_uniform_bounds_float32() raises:
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


def test_random_tensor_default_bounds() raises:
    """Test random_tensor with default bounds [0, 1)."""
    var shape = [50, 50]
    var tensor = random_tensor(shape)

    # Check all values are in [0, 1)
    for i in range(2500):
        var val = tensor._get_float64(i)
        custom_assert_true(val >= 0.0, "Value below 0")
        custom_assert_true(val < 1.0, "Value at or above 1")


def test_random_tensor_int32() raises:
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


def test_random_tensor_1d() raises:
    """Test random_tensor with 1D shape."""
    var shape = [100]
    var tensor = random_tensor(shape, DType.float32, 0.0, 1.0)
    assert_shape_equal(tensor.shape(), shape)


def test_random_tensor_3d() raises:
    """Test random_tensor with 3D shape."""
    var shape = [10, 10, 10]
    var tensor = random_tensor(shape, DType.float32, 0.0, 1.0)
    assert_shape_equal(tensor.shape(), shape)


def test_random_normal_tensor_default_params() raises:
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


def test_random_normal_tensor_custom_mean_std() raises:
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


def test_random_normal_tensor_int32() raises:
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


def test_random_normal_tensor_1d() raises:
    """Test random_normal_tensor with 1D shape."""
    var shape = [100]
    var tensor = random_normal_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_random_normal_tensor_3d() raises:
    """Test random_normal_tensor with 3D shape."""
    var shape = [5, 5, 5]
    var tensor = random_normal_tensor(shape, DType.float32)
    assert_shape_equal(tensor.shape(), shape)


def test_set_tensor_value_float32() raises:
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


def test_set_tensor_value_int32() raises:
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


def test_set_tensor_value_overwrite() raises:
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


def test_set_tensor_value_float64() raises:
    """Test set_tensor_value with float64 dtype."""
    var shape = [5]
    var tensor = zeros_tensor(shape, DType.float64)

    set_tensor_value(tensor, 2, 6.28, DType.float64)

    var val = tensor._get_float64(2)
    assert_almost_equal(val, 6.28, tolerance=1e-10)


def test_set_tensor_value_multiple_indices() raises:
    """Test setting multiple values in same tensor."""
    var shape = [10]
    var tensor = zeros_tensor(shape, DType.float32)

    # Set multiple values
    for i in range(10):
        set_tensor_value(tensor, i, Float64(i) * 1.5, DType.float32)

    # Verify all values
    for i in range(10):
        var val = tensor._get_float64(i)
        var expected = Float64(i) * 1.5
        assert_almost_equal(val, expected, tolerance=1e-4)


def test_tensor_factory_workflow() raises:
    """Test typical workflow using multiple factory functions."""
    # Create tensors for a simple test scenario
    var shape = [10, 10]

    # Create various tensors
    var zeros = zeros_tensor(shape, DType.float32)
    var ones = ones_tensor(shape, DType.float32)
    var fives = full_tensor(shape, 5.0, DType.float32)
    var random = random_tensor(shape, DType.float32, -1.0, 1.0)
    var normal = random_normal_tensor(shape, DType.float32, 0.0, 1.0)

    # Verify all have correct shape
    assert_shape_equal(zeros.shape(), shape)
    assert_shape_equal(ones.shape(), shape)
    assert_shape_equal(fives.shape(), shape)
    assert_shape_equal(random.shape(), shape)
    assert_shape_equal(normal.shape(), shape)

    # Verify all have correct dtype
    assert_dtype_equal(zeros.dtype(), DType.float32)
    assert_dtype_equal(ones.dtype(), DType.float32)
    assert_dtype_equal(fives.dtype(), DType.float32)
    assert_dtype_equal(random.dtype(), DType.float32)
    assert_dtype_equal(normal.dtype(), DType.float32)


def test_tensor_factory_all_dtypes() raises:
    """Test tensor factories work with multiple dtypes."""
    var shape = [5]
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)

    # Test each dtype
    for dtype in dtypes:
        var zeros = zeros_tensor(shape, dtype)
        var ones = ones_tensor(shape, dtype)
        var full = full_tensor(shape, 3.0, dtype)

        assert_dtype_equal(zeros.dtype(), dtype)
        assert_dtype_equal(ones.dtype(), dtype)
        assert_dtype_equal(full.dtype(), dtype)


def main() raises:
    """Run all test_tensor_factory tests."""
    print("Running test_tensor_factory tests...")

    test_zeros_tensor_float32()
    print("✓ test_zeros_tensor_float32")

    test_zeros_tensor_int32()
    print("✓ test_zeros_tensor_int32")

    test_zeros_tensor_1d()
    print("✓ test_zeros_tensor_1d")

    test_zeros_tensor_3d()
    print("✓ test_zeros_tensor_3d")

    test_ones_tensor_float32()
    print("✓ test_ones_tensor_float32")

    test_ones_tensor_int32()
    print("✓ test_ones_tensor_int32")

    test_ones_tensor_1d()
    print("✓ test_ones_tensor_1d")

    test_ones_tensor_3d()
    print("✓ test_ones_tensor_3d")

    test_full_tensor_float32_positive()
    print("✓ test_full_tensor_float32_positive")

    test_full_tensor_float32_negative()
    print("✓ test_full_tensor_float32_negative")

    test_full_tensor_int32()
    print("✓ test_full_tensor_int32")

    test_random_tensor_uniform_bounds_float32()
    print("✓ test_random_tensor_uniform_bounds_float32")

    test_random_tensor_default_bounds()
    print("✓ test_random_tensor_default_bounds")

    test_random_tensor_int32()
    print("✓ test_random_tensor_int32")

    test_random_tensor_1d()
    print("✓ test_random_tensor_1d")

    test_random_tensor_3d()
    print("✓ test_random_tensor_3d")

    test_random_normal_tensor_default_params()
    print("✓ test_random_normal_tensor_default_params")

    test_random_normal_tensor_custom_mean_std()
    print("✓ test_random_normal_tensor_custom_mean_std")

    test_random_normal_tensor_int32()
    print("✓ test_random_normal_tensor_int32")

    test_random_normal_tensor_1d()
    print("✓ test_random_normal_tensor_1d")

    test_random_normal_tensor_3d()
    print("✓ test_random_normal_tensor_3d")

    test_set_tensor_value_float32()
    print("✓ test_set_tensor_value_float32")

    test_set_tensor_value_int32()
    print("✓ test_set_tensor_value_int32")

    test_set_tensor_value_overwrite()
    print("✓ test_set_tensor_value_overwrite")

    test_set_tensor_value_float64()
    print("✓ test_set_tensor_value_float64")

    test_set_tensor_value_multiple_indices()
    print("✓ test_set_tensor_value_multiple_indices")

    test_tensor_factory_workflow()
    print("✓ test_tensor_factory_workflow")

    test_tensor_factory_all_dtypes()
    print("✓ test_tensor_factory_all_dtypes")

    print("\nAll test_tensor_factory tests passed!")
