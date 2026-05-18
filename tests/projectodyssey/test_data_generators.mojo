"""Tests for data generators module.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests random_tensor function:
- Shape creation (1D, 2D, 3D)
- Dtype handling (float32, float64, int32)
- Value ranges
- Default dtype
"""


from projectodyssey.testing import (
    random_normal,
    random_tensor,
    random_uniform,
    synthetic_classification_data,
)
from tests.projectodyssey.conftest import (
    assert_dim,
    assert_dtype,
    assert_equal_int,
    assert_numel,
    assert_shape,
    assert_true,
)


def test_random_tensor_shape_1d() raises:
    """Test random_tensor creates correct 1D shape."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 1, "random_tensor 1D should have 1 dimension")
    assert_numel(tensor, 10, "random_tensor 1D should have 10 elements")


def test_random_tensor_shape_2d() raises:
    """Test random_tensor creates correct 2D shape."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(8)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 2, "random_tensor 2D should have 2 dimensions")
    assert_numel(tensor, 40, "random_tensor 2D(5,8) should have 40 elements")


def test_random_tensor_shape_3d() raises:
    """Test random_tensor creates correct 3D shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 3, "random_tensor 3D should have 3 dimensions")
    assert_numel(tensor, 24, "random_tensor 3D(2,3,4) should have 24 elements")


def test_random_tensor_dtype_float32() raises:
    """Test random_tensor with float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.float32)

    assert_dtype(tensor, DType.float32, "random_tensor should respect dtype")


def test_random_tensor_dtype_float64() raises:
    """Test random_tensor with float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.float64)

    assert_dtype(tensor, DType.float64, "random_tensor should respect dtype")


def test_random_tensor_dtype_int32() raises:
    """Test random_tensor with int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.int32)

    assert_dtype(tensor, DType.int32, "random_tensor should respect dtype")


def test_random_tensor_values_in_range() raises:
    """Test random_tensor values are in [0, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_tensor(shape, DType.float32)

    # Check a sample of values are in valid range
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= 0.0, "random_tensor value should be >= 0.0")
        assert_true(val < 1.0, "random_tensor value should be < 1.0")


def test_random_tensor_default_dtype() raises:
    """Test random_tensor uses float32 as default dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape)  # No dtype specified

    assert_dtype(
        tensor, DType.float32, "random_tensor default dtype should be float32"
    )


def test_random_uniform_shape() raises:
    """Test random_uniform creates correct shape."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(5)
    var tensor = random_uniform(shape, low=0.0, high=1.0)

    assert_numel(tensor, 50, "random_uniform(10,5) should have 50 elements")


def test_random_uniform_range_0_to_1() raises:
    """Test random_uniform with default range [0, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_uniform(shape)

    # Check sample values are in [0, 1)
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= 0.0, "uniform value should be >= 0.0")
        assert_true(val < 1.0, "uniform value should be < 1.0")


def test_random_uniform_range_negative_to_positive() raises:
    """Test random_uniform with custom range [-1, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_uniform(shape, low=-1.0, high=1.0)

    # Check sample values are in [-1, 1)
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= -1.0, "uniform value should be >= -1.0")
        assert_true(val < 1.0, "uniform value should be < 1.0")


def test_random_uniform_dtype() raises:
    """Test random_uniform respects dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_uniform(shape, dtype=DType.float64)

    assert_dtype(tensor, DType.float64, "random_uniform should respect dtype")


def test_random_normal_shape() raises:
    """Test random_normal creates correct shape."""
    var shape = List[Int]()
    shape.append(20)
    shape.append(30)
    var tensor = random_normal(shape)

    assert_numel(tensor, 600, "random_normal(20,30) should have 600 elements")


def test_random_normal_dtype_float32() raises:
    """Test random_normal with float32 dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_normal(shape, dtype=DType.float32)

    assert_dtype(tensor, DType.float32, "random_normal should respect dtype")


def test_random_normal_dtype_float64() raises:
    """Test random_normal with float64 dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_normal(shape, dtype=DType.float64)

    assert_dtype(tensor, DType.float64, "random_normal should respect dtype")


def test_random_normal_mean_and_std() raises:
    """Test random_normal respects mean and std parameters.

    Note: With small samples (10 elements), we only do sanity checks.
    Statistical tests would require larger samples.
    """
    var shape = List[Int]()
    shape.append(10)

    # Generate with mean=5.0, std=1.0
    var tensor = random_normal(shape, mean=5.0, std=1.0)

    # Get a sample value - should be roughly around mean (with tolerance)
    var val = tensor._get_float64(0)

    # With mean=5.0, values should typically be in [3, 7] (mean ± 2*std)
    assert_true(
        val >= 1.0 and val <= 9.0,
        "random_normal with mean=5.0 should produce values roughly around 5.0",
    )


def test_random_normal_distribution_sanity() raises:
    """Test random_normal produces varied values (not all same).

    With a normal distribution, consecutive values should be different.
    """
    var shape = List[Int]()
    shape.append(50)
    var tensor = random_normal(shape)

    # Get first two values
    var val1 = tensor._get_float64(0)
    var val2 = tensor._get_float64(1)

    # They should not be identical (probability of identical floats is essentially 0)
    assert_true(
        val1 != val2,
        "random_normal should produce different values for different samples",
    )


def test_synthetic_classification_data_shape_features() raises:
    """Test synthetic_classification_data produces correct feature shape."""
    var result1 = synthetic_classification_data(100, 20, 3)
    var X = result1[0]

    assert_shape(X, [100, 20], "Features should have shape [100, 20]")


def test_synthetic_classification_data_shape_labels() raises:
    """Test synthetic_classification_data produces correct label shape."""
    var result2 = synthetic_classification_data(100, 20, 3)
    var y = result2[1]

    assert_shape(y, [100], "Labels should have shape [100]")


def test_synthetic_classification_data_dtype_features() raises:
    """Test synthetic_classification_data feature dtype."""
    var result3 = synthetic_classification_data(100, 20, 3, DType.float32)
    var X = result3[0]

    assert_dtype(X, DType.float32, "Features should have specified dtype")


def test_synthetic_classification_data_dtype_labels() raises:
    """Test synthetic_classification_data labels are int32."""
    var result2 = synthetic_classification_data(100, 20, 3)
    var y = result2[1]

    assert_dtype(y, DType.int32, "Labels should be int32")


def test_synthetic_classification_data_label_values() raises:
    """Test synthetic_classification_data labels are in valid range."""
    var result4 = synthetic_classification_data(50, 10, 3)
    var y = result4[1]

    # Check that labels are in [0, num_classes)
    for i in range(50):
        var label_val = y._get_int64(i)
        assert_true(label_val >= 0, "Label should be >= 0")
        assert_true(label_val < 3, "Label should be < num_classes")


def test_synthetic_classification_data_single_sample() raises:
    """Test synthetic_classification_data with minimal size."""
    var result5 = synthetic_classification_data(1, 1, 1)
    var X = result5[0]
    var y = result5[1]

    assert_shape(X, [1, 1], "Minimal features should have shape [1, 1]")
    assert_shape(y, [1], "Minimal labels should have shape [1]")


def test_synthetic_classification_data_many_classes() raises:
    """Test synthetic_classification_data with many classes."""
    var result6 = synthetic_classification_data(100, 10, 10)
    var y = result6[1]

    # Check all labels are in valid range
    for i in range(100):
        var label_val = y._get_int64(i)
        assert_true(label_val >= 0, "Label should be >= 0")
        assert_true(label_val < 10, "Label should be < num_classes")


def test_synthetic_classification_data_high_dimensions() raises:
    """Test synthetic_classification_data with high-dimensional features."""
    var result7 = synthetic_classification_data(50, 100, 5)
    var X = result7[0]

    assert_shape(X, [50, 100], "Should handle high-dimensional features")


def test_integration_random_tensor_and_normal() raises:
    """Test combining random tensor and normal generation."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)

    var t1 = random_tensor(shape)
    var t2 = random_normal(shape)

    assert_numel(t1, 25, "First tensor should have 25 elements")
    assert_numel(t2, 25, "Second tensor should have 25 elements")


def test_integration_classification_data_shapes_match() raises:
    """Test features and labels shapes are consistent."""
    var num_samples = 100
    var result8 = synthetic_classification_data(num_samples, 15, 4)
    var X = result8[0]
    var y = result8[1]

    # Number of samples should match
    assert_equal_int(
        X.shape()[0],
        num_samples,
        "Features first dimension should match num_samples",
    )
    assert_equal_int(
        y.shape()[0], num_samples, "Labels should match num_samples"
    )


def main() raises:
    """Run all test_data_generators tests."""
    print("Running test_data_generators tests...")

    test_random_tensor_shape_1d()
    print("✓ test_random_tensor_shape_1d")

    test_random_tensor_shape_2d()
    print("✓ test_random_tensor_shape_2d")

    test_random_tensor_shape_3d()
    print("✓ test_random_tensor_shape_3d")

    test_random_tensor_dtype_float32()
    print("✓ test_random_tensor_dtype_float32")

    test_random_tensor_dtype_float64()
    print("✓ test_random_tensor_dtype_float64")

    test_random_tensor_dtype_int32()
    print("✓ test_random_tensor_dtype_int32")

    test_random_tensor_values_in_range()
    print("✓ test_random_tensor_values_in_range")

    test_random_tensor_default_dtype()
    print("✓ test_random_tensor_default_dtype")

    test_random_uniform_shape()
    print("✓ test_random_uniform_shape")

    test_random_uniform_range_0_to_1()
    print("✓ test_random_uniform_range_0_to_1")

    test_random_uniform_range_negative_to_positive()
    print("✓ test_random_uniform_range_negative_to_positive")

    test_random_uniform_dtype()
    print("✓ test_random_uniform_dtype")

    test_random_normal_shape()
    print("✓ test_random_normal_shape")

    test_random_normal_dtype_float32()
    print("✓ test_random_normal_dtype_float32")

    test_random_normal_dtype_float64()
    print("✓ test_random_normal_dtype_float64")

    test_random_normal_mean_and_std()
    print("✓ test_random_normal_mean_and_std")

    test_random_normal_distribution_sanity()
    print("✓ test_random_normal_distribution_sanity")

    test_synthetic_classification_data_shape_features()
    print("✓ test_synthetic_classification_data_shape_features")

    test_synthetic_classification_data_shape_labels()
    print("✓ test_synthetic_classification_data_shape_labels")

    test_synthetic_classification_data_dtype_features()
    print("✓ test_synthetic_classification_data_dtype_features")

    test_synthetic_classification_data_dtype_labels()
    print("✓ test_synthetic_classification_data_dtype_labels")

    test_synthetic_classification_data_label_values()
    print("✓ test_synthetic_classification_data_label_values")

    test_synthetic_classification_data_single_sample()
    print("✓ test_synthetic_classification_data_single_sample")

    test_synthetic_classification_data_many_classes()
    print("✓ test_synthetic_classification_data_many_classes")

    test_synthetic_classification_data_high_dimensions()
    print("✓ test_synthetic_classification_data_high_dimensions")

    test_integration_random_tensor_and_normal()
    print("✓ test_integration_random_tensor_and_normal")

    test_integration_classification_data_shapes_match()
    print("✓ test_integration_classification_data_shapes_match")

    print("\nAll test_data_generators tests passed!")
