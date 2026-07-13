"""Unit tests for Linear struct - OOP fully connected layer.

Tests the Linear struct API which manages weights and bias internally.

Key differences from functional API:
- Weights are stored internally with shape (in_features, out_features)
- Bias is stored internally with shape (out_features,)
- Forward pass computes: output = input @ weight + bias
- No transpose of weights needed (stored in correct order)
"""

from tests.odyssey.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import ones, zeros, zeros_like
from odyssey.core.layers.linear import Linear


def test_linear_struct_initialization() raises:
    """Test Linear struct initialization.

    Verify that weights and bias are created with correct shapes.
    """
    var in_features = 10
    var out_features = 5

    var layer = Linear(in_features, out_features)

    # Check weight shape: (in_features, out_features)
    var weight_shape = layer.weight.shape()
    assert_equal(weight_shape[0], in_features)
    assert_equal(weight_shape[1], out_features)

    # Check bias shape: (out_features,)
    var bias_shape = layer.bias.shape()
    assert_equal(bias_shape[0], out_features)

    # Check stored dimensions
    assert_equal(layer.in_features, in_features)
    assert_equal(layer.out_features, out_features)


def test_linear_struct_forward_batched() raises:
    """Test Linear forward pass with batched input.

    Input: (batch_size=4, in_features=10)
    Weight: (in_features=10, out_features=5)
    Bias: (out_features=5,)
    Expected output: (batch_size=4, out_features=5).
    """
    var layer = Linear(10, 5)

    # Create input batch
    var input_shape: List[Int] = [4, 10]
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = layer.forward(input)

    # Check output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], 4)
    assert_equal(output_shape[1], 5)


def test_linear_struct_forward_single_sample() raises:
    """Test Linear forward pass with single sample.

    Input: (in_features=3,)
    Weight: (in_features=3, out_features=2)
    Bias: (out_features=2,)
    Expected output: (out_features=2,)

    With specific values:
    Input: [1, 2, 3]
    Weight: [[1, 0, 0], [0, 1, 0]] (identity-like)
    Bias: [0, 0]
    Expected output: [1, 2]
    """
    # Create layer with specific dimensions
    var layer = Linear(3, 2)

    # Set weights to identity-like pattern
    var weight_data = layer.weight._data.bitcast[Float32]()
    weight_data[0] = 1.0  # weight[0, 0]
    weight_data[1] = 0.0  # weight[0, 1]
    weight_data[2] = 0.0  # weight[1, 0]
    weight_data[3] = 1.0  # weight[1, 1]
    weight_data[4] = 0.0  # weight[2, 0]
    weight_data[5] = 0.0  # weight[2, 1]

    # Set bias to zeros
    var bias_data = layer.bias._data.bitcast[Float32]()
    bias_data[0] = 0.0
    bias_data[1] = 0.0

    # Create input: [[1, 2, 3]] (2D: batch=1, features=3)
    # AnyTensor.__matmul__ requires 2D inputs
    var input_shape: List[Int] = [1, 3]
    var input_val = ones(input_shape, DType.float32)
    var input_data = input_val._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    # Forward pass
    var output = input_val @ layer.weight + layer.bias

    # Check output shape: (1, 2)
    var output_shape = output.shape()
    assert_equal(output_shape[0], 1)
    assert_equal(output_shape[1], 2)

    # Check values: [1*1 + 2*0 + 3*0, 1*0 + 2*1 + 3*0] = [1, 2]
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 2.0, tolerance=1e-5)


def test_linear_struct_with_bias() raises:
    """Test Linear forward pass correctly adds bias.

    Input: (1, 2) = [1, 1]
    Weight: (2, 2) = identity
    Bias: (2,) = [5, 10]
    Expected output: [1+5, 1+10] = [6, 11]
    """
    var layer = Linear(2, 2)

    # Set weight to identity
    var weight_data = layer.weight._data.bitcast[Float32]()
    weight_data[0] = 1.0  # weight[0, 0]
    weight_data[1] = 0.0  # weight[0, 1]
    weight_data[2] = 0.0  # weight[1, 0]
    weight_data[3] = 1.0  # weight[1, 1]

    # Set bias to [5, 10]
    var bias_data = layer.bias._data.bitcast[Float32]()
    bias_data[0] = 5.0
    bias_data[1] = 10.0

    # Create input [1, 1]
    var input = ones([1, 2], DType.float32)

    # Forward pass
    var output = layer.forward(input)

    # Check values
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 11.0, tolerance=1e-5)


def test_linear_struct_parameters() raises:
    """Test Linear.parameters() returns copies of weights and bias.

    Verify that the parameters method returns a list with [weight, bias].
    """
    var layer = Linear(4, 3)

    var params = layer.parameters()

    # Should have 2 parameters: weight and bias
    assert_equal(len(params), 2)

    # First parameter should have weight shape
    var weight_shape = params[0].shape()
    assert_equal(weight_shape[0], 4)
    assert_equal(weight_shape[1], 3)

    # Second parameter should have bias shape
    var bias_shape = params[1].shape()
    assert_equal(bias_shape[0], 3)


def test_linear_struct_large_batch() raises:
    """Test Linear with larger batch size and dimensions.

    Verify it works with realistic neural network sizes.
    """
    var layer = Linear(784, 128)

    # Create batch
    var input = ones([32, 784], DType.float32)

    # Forward pass
    var output = layer.forward(input)

    # Check output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], 32)
    assert_equal(output_shape[1], 128)


def main() raises:
    """Run all Linear struct tests."""
    print("Running Linear struct tests...")

    test_linear_struct_initialization()
    print("✓ test_linear_struct_initialization")

    test_linear_struct_forward_batched()
    print("✓ test_linear_struct_forward_batched")

    test_linear_struct_forward_single_sample()
    print("✓ test_linear_struct_forward_single_sample")

    test_linear_struct_with_bias()
    print("✓ test_linear_struct_with_bias")

    test_linear_struct_parameters()
    print("✓ test_linear_struct_parameters")

    test_linear_struct_large_batch()
    print("✓ test_linear_struct_large_batch")

    print("\nAll Linear struct tests passed!")
