"""Tests for shared.testing.fixtures module

Tests SimpleCNN and LinearModel structs plus the create_test_cnn factory function.
"""


from testing import assert_true, assert_equal
from shared.testing.models import SimpleCNN, LinearModel
from shared.testing.fixtures import (
    create_test_cnn,
)
from shared.tensor.any_tensor import ones
from shared.testing.fixtures import (
    create_linear_model,
    create_test_input,
    create_test_targets,
    assert_tensor_shape,
    assert_tensor_dtype,
)
from shared.testing.fixtures import (
    assert_tensor_dtype,
    assert_tensor_all_finite,
    assert_tensor_not_all_zeros,
)
from shared.tensor.any_tensor import ones, zeros


fn test_simple_cnn_initialization() raises:
    """Test SimpleCNN struct initialization."""
    var model = SimpleCNN(1, 8, 10)
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)


fn test_simple_cnn_default_initialization() raises:
    """Test SimpleCNN with default parameters."""
    var model = SimpleCNN()
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)


fn test_simple_cnn_get_output_shape() raises:
    """Test SimpleCNN output shape computation."""
    var model = SimpleCNN(1, 8, 10)
    var shape = model.get_output_shape(32)
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 32)
    assert_equal(shape[1], 10)


fn test_simple_cnn_forward() raises:
    """Test SimpleCNN forward pass produces correct output shape."""
    var model = SimpleCNN(1, 8, 10)
    var batch_size = 32
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(1)
    input_shape.append(28)
    input_shape.append(28)

    var input = ones(input_shape, DType.float32)
    var output = model.forward(input)

    # Check output shape
    assert_equal(output._shape[0], batch_size)
    assert_equal(output._shape[1], 10)

    # Check all values are 0.1
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_true(val > 0.099 and val < 0.101, "Output should be 0.1")


fn test_linear_model_initialization() raises:
    """Test LinearModel struct initialization."""
    var model = LinearModel(784, 10)
    assert_equal(model.in_features, 784)
    assert_equal(model.out_features, 10)


fn test_linear_model_get_output_shape() raises:
    """Test LinearModel output shape computation."""
    var model = LinearModel(784, 10)
    var shape = model.get_output_shape(32)
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 32)
    assert_equal(shape[1], 10)


fn test_linear_model_forward() raises:
    """Test LinearModel forward pass produces correct output shape."""
    var model = LinearModel(784, 10)
    var batch_size = 32
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(784)

    var input = ones(input_shape, DType.float32)
    var output = model.forward(input)

    # Check output shape
    assert_equal(output._shape[0], batch_size)
    assert_equal(output._shape[1], 10)

    # Check all values are zeros
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_equal(val, 0.0)


fn test_create_test_cnn() raises:
    """Test create_test_cnn factory function."""
    var model = create_test_cnn()
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)

    var custom_model = create_test_cnn(3, 32, 1000)
    assert_equal(custom_model.in_channels, 3)
    assert_equal(custom_model.out_channels, 32)
    assert_equal(custom_model.num_classes, 1000)


fn test_create_linear_model() raises:
    """Test create_linear_model factory function."""
    var model = create_linear_model()
    assert_equal(model.in_features, 784)
    assert_equal(model.out_features, 10)

    var custom_model = create_linear_model(2048, 1024)
    assert_equal(custom_model.in_features, 2048)
    assert_equal(custom_model.out_features, 1024)


fn test_create_test_input() raises:
    """Test create_test_input utility function."""
    var input = create_test_input(32, 784)
    assert_equal(input._shape[0], 32)
    assert_equal(input._shape[1], 784)

    # Check all values are 1.0
    for i in range(input.numel()):
        var val = input._get_float64(i)
        assert_equal(val, 1.0)


fn test_create_test_input_custom_dtype() raises:
    """Test create_test_input with custom dtype."""
    var input = create_test_input(16, 512, DType.float64)
    assert_equal(input._shape[0], 16)
    assert_equal(input._shape[1], 512)
    assert_true(input._dtype == DType.float64)


fn test_create_test_targets() raises:
    """Test create_test_targets utility function."""
    var targets = create_test_targets(32, 10)
    assert_equal(targets._shape[0], 32)

    # Check all values are 0
    for i in range(targets.numel()):
        var val = targets._get_float64(i)
        assert_equal(val, 0.0)


fn test_assert_tensor_shape_valid() raises:
    """Test assert_tensor_shape with matching shapes."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [32, 10]
    assert_true(assert_tensor_shape(tensor, expected))


fn test_assert_tensor_shape_invalid_dimensions() raises:
    """Test assert_tensor_shape with wrong number of dimensions."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [32, 10, 5]
    assert_true(not assert_tensor_shape(tensor, expected))


fn test_assert_tensor_shape_invalid_size() raises:
    """Test assert_tensor_shape with wrong dimension sizes."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [64, 10]
    assert_true(not assert_tensor_shape(tensor, expected))


fn test_assert_tensor_dtype_valid() raises:
    """Test assert_tensor_dtype with matching dtype."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_dtype(tensor, DType.float32))


fn test_assert_tensor_dtype_invalid() raises:
    """Test assert_tensor_dtype with mismatched dtype."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(not assert_tensor_dtype(tensor, DType.float64))


fn test_assert_tensor_all_finite_valid() raises:
    """Test assert_tensor_all_finite with finite values."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_all_finite(tensor))


fn test_assert_tensor_not_all_zeros_valid() raises:
    """Test assert_tensor_not_all_zeros with non-zero values."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_not_all_zeros(tensor))


fn test_assert_tensor_not_all_zeros_invalid() raises:
    """Test assert_tensor_not_all_zeros with all zeros."""
    var tensor = zeros([32, 10], DType.float32)
    assert_true(not assert_tensor_not_all_zeros(tensor))


fn main() raises:
    """Run all test_fixtures tests."""
    print("Running test_fixtures tests...")

    test_simple_cnn_initialization()
    print("✓ test_simple_cnn_initialization")

    test_simple_cnn_default_initialization()
    print("✓ test_simple_cnn_default_initialization")

    test_simple_cnn_get_output_shape()
    print("✓ test_simple_cnn_get_output_shape")

    test_simple_cnn_forward()
    print("✓ test_simple_cnn_forward")

    test_linear_model_initialization()
    print("✓ test_linear_model_initialization")

    test_linear_model_get_output_shape()
    print("✓ test_linear_model_get_output_shape")

    test_linear_model_forward()
    print("✓ test_linear_model_forward")

    test_create_test_cnn()
    print("✓ test_create_test_cnn")

    test_create_linear_model()
    print("✓ test_create_linear_model")

    test_create_test_input()
    print("✓ test_create_test_input")

    test_create_test_input_custom_dtype()
    print("✓ test_create_test_input_custom_dtype")

    test_create_test_targets()
    print("✓ test_create_test_targets")

    test_assert_tensor_shape_valid()
    print("✓ test_assert_tensor_shape_valid")

    test_assert_tensor_shape_invalid_dimensions()
    print("✓ test_assert_tensor_shape_invalid_dimensions")

    test_assert_tensor_shape_invalid_size()
    print("✓ test_assert_tensor_shape_invalid_size")

    test_assert_tensor_dtype_valid()
    print("✓ test_assert_tensor_dtype_valid")

    test_assert_tensor_dtype_invalid()
    print("✓ test_assert_tensor_dtype_invalid")

    test_assert_tensor_all_finite_valid()
    print("✓ test_assert_tensor_all_finite_valid")

    test_assert_tensor_not_all_zeros_valid()
    print("✓ test_assert_tensor_not_all_zeros_valid")

    test_assert_tensor_not_all_zeros_invalid()
    print("✓ test_assert_tensor_not_all_zeros_invalid")

    print("\nAll test_fixtures tests passed!")
