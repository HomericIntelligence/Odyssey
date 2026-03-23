"""Test fixtures and utilities for ML Odyssey.

Provides common test setup utilities, model fixtures, and assertion helpers
for validating neural network implementations.

Features:
    - Simple test models (CNN, Linear, MLP) for basic functionality testing
    - Model factory functions for consistent initialization
    - Common test setup patterns
    - Assertion helpers for tensor validation

This module re-exports the consolidated test models from models.mojo
for backward compatibility and provides additional fixture utilities.

Example:
    from shared.testing.fixtures import create_test_cnn, create_linear_model
    from shared.core import zeros, ones

    fn test_forward_pass():
        var model = create_test_cnn(1, 8, 10)
        var batch_size = 32
        var input_shape : List[Int] = [batch_size, 1, 28, 28]
        var input = ones(input_shape, DType.float32)
        # Test forward pass...

    fn test_linear_output_shape():
        var model = create_linear_model(784, 10)
        assert_equal(model.in_features, 784)
        assert_equal(model.out_features, 10)
    ```
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, zeros_like
from shared.testing.models import SimpleCNN, LinearModel


fn create_test_cnn(
    in_channels: Int = 1, out_channels: Int = 8, num_classes: Int = 10
) -> SimpleCNN:
    """Factory function to create test CNN models.

        Useful for consistent model initialization across tests

    Args:
            in_channels: Number of input channels (default: 1 for MNIST).
            out_channels: Number of output channels (default: 8).
            num_classes: Number of output classes (default: 10).

    Returns:
            Test CNN instance

        Example:
            ```mojo
            var model = create_test_cnn(1, 8, 10)
            assert_equal(model.num_classes, 10)

            # Create custom channel configuration
            var custom_model = create_test_cnn(3, 32, 1000)
            assert_equal(custom_model.in_channels, 3)
            ```
    """
    return SimpleCNN(in_channels, out_channels, num_classes)


fn create_linear_model(
    in_features: Int = 784, out_features: Int = 10
) -> LinearModel:
    """Create a simple linear test model.

        Useful for testing linear layers, loss functions, and optimization.

    Args:
            in_features: Input dimension (default: 784 for MNIST flattened).
            out_features: Output dimension (default: 10 for MNIST classes).

    Returns:
            Linear model instance

        Example:
            ```mojo
            var model = create_linear_model(784, 10)
            assert_equal(model.out_features, 10)

            # Create custom dimensions
            var large_model = create_linear_model(2048, 1024)
            assert_equal(large_model.in_features, 2048)
            ```
    """
    return LinearModel(in_features, out_features)


fn create_test_input(
    batch_size: Int, in_features: Int, dtype: DType = DType.float32
) raises -> AnyTensor:
    """Create a test input tensor for linear models.

        Creates a simple tensor filled with ones for testing

    Args:
            batch_size: Number of samples.
            in_features: Input dimension.
            dtype: Data type (default: float32).

    Returns:
            Input tensor (batch_size, in_features) filled with 1.0

    Raises:
            Error: If tensor creation fails.

        Example:
            ```mojo
            var input = create_test_input(32, 784)
            assert_equal(input._shape[0], 32)
            assert_equal(input._shape[1], 784)
            ```
    """
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(in_features)
    return ones(shape, dtype)


fn create_test_targets(
    batch_size: Int, num_classes: Int, dtype: DType = DType.int32
) raises -> AnyTensor:
    """Create a test target tensor for classification.

        Creates a tensor filled with zeros for testing

    Args:
            batch_size: Number of samples.
            num_classes: Number of classes.
            dtype: Data type (default: int32).

    Returns:
            Target tensor (batch_size,) filled with 0

    Raises:
            Error: If tensor creation fails.

        Example:
            ```mojo
            var targets = create_test_targets(32, 10)
            assert_equal(targets._shape[0], 32)
            ```
    """
    var shape = List[Int]()
    shape.append(batch_size)
    return zeros(shape, dtype)


fn assert_tensor_shape(tensor: AnyTensor, expected_shape: List[Int]) -> Bool:
    """Validate tensor has expected shape.

    Args:
            tensor: Tensor to check.
            expected_shape: Expected shape dimensions.

    Returns:
            True if shapes match, False otherwise

        Example:
            ```mojo
            var tensor = ones([32, 10], DType.float32)
            assert_true(assert_tensor_shape(tensor, [32, 10]))
            ```
    """
    if len(tensor._shape) != len(expected_shape):
        return False

    for i in range(len(expected_shape)):
        if tensor._shape[i] != expected_shape[i]:
            return False

    return True


fn assert_tensor_dtype(tensor: AnyTensor, expected_dtype: DType) -> Bool:
    """Validate tensor has expected data type.

    Args:
            tensor: Tensor to check.
            expected_dtype: Expected DType.

    Returns:
            True if dtypes match, False otherwise

        Example:
            ```mojo
            var tensor = ones([32, 10], DType.float32)
            assert_true(assert_tensor_dtype(tensor, DType.float32))
            ```
    """
    return tensor._dtype == expected_dtype


fn assert_tensor_all_finite(tensor: AnyTensor) -> Bool:
    """Check if all tensor values are finite (no NaN or Inf).

    Args:
            tensor: Tensor to check.

    Returns:
            True if all values are finite, False if any NaN/Inf found

        Example:
            ```mojo
            var tensor = ones([32, 10], DType.float32)
            assert_true(assert_tensor_all_finite(tensor))
            ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        # Check for NaN
        if val != val:  # NaN != NaN is true
            return False
        # Check for Inf
        if val > 1e308 or val < -1e308:
            return False

    return True


fn assert_tensor_not_all_zeros(tensor: AnyTensor) -> Bool:
    """Check if tensor contains at least one non-zero value.

        Useful for verifying weights are initialized and gradients are flowing.

    Args:
            tensor: Tensor to check.

    Returns:
            True if at least one non-zero value exists, False if all zeros.

    Example:
        ```mojo
        var tensor = ones([32, 10], DType.float32)
        assert_true(assert_tensor_not_all_zeros(tensor))
        ```
    """
    for i in range(tensor.numel()):
        if tensor._get_float64(i) != 0.0:
            return True

    return False
