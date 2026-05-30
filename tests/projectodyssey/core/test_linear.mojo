"""Unit tests for linear layer (fully connected) forward operations.

Tests cover:
- linear: Forward pass with bias y = xW^T + b
- linear_no_bias: Forward pass without bias y = xW^T
- linear_backward: Backward pass computing gradients (shape and single sample)
- Shape computations and dimension handling
- Numerical correctness

All tests use pure functional API - no internal state.

"""


from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full, ones, zeros
from projectodyssey.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)
from projectodyssey.core.matrix import transpose


def test_linear_initialization() raises:
    """Test that linear layer parameters can be created with correct shapes.

    Functional API Note:
        Caller creates weights (out_features, in_features) and bias (out_features,).
        This test verifies parameters can be created.
    """
    var batch_size = 32
    var in_features = 784
    var out_features = 128

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias: (out_features,)
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var input_s = input.shape()
    var weights_s = weights.shape()
    var bias_s = bias.shape()
    assert_equal(input_s[0], batch_size)
    assert_equal(input_s[1], in_features)
    assert_equal(weights_s[0], out_features)
    assert_equal(weights_s[1], in_features)
    assert_equal(bias_s[0], out_features)


def test_linear_output_shape() raises:
    """Test linear layer output shape computation.

    Formula: output_shape = (batch_size, out_features)

    Test case: batch=32, in_features=784, out_features=128
    Expected: output shape (32, 128).
    """
    var batch_size = 32
    var in_features = 784
    var out_features = 128

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias: (out_features,)
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Compute linear: y = xW^T + b
    var output = linear(input, weights, bias)

    # Check output shape: (batch_size, out_features)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)


def test_linear_single_sample() raises:
    """Test linear layer with single sample.

    Single sample: (1, 3) @ (2, 3)^T + (2,) = (1, 2)

    Input: [1, 2, 3]
    Weights: [[1, 0, 0],
              [0, 1, 0]]
    Bias: [0, 0]

    Expected output: [[1, 2]]
    """
    # Input: (1, 3)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    # Weights: (2, 3) - identity-like pattern
    var weights_shape = List[Int]()
    weights_shape.append(2)
    weights_shape.append(3)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0] = 1.0
    weights_data[4] = 1.0  # weights[1, 1] = 1.0

    # Bias: (2,) - all zeros
    var bias_shape = List[Int]()
    bias_shape.append(2)
    var bias = zeros(bias_shape, DType.float32)

    # Compute linear
    var output = linear(input, weights, bias)

    # Check shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 2)

    # Check values: output should be [1, 2]
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 2.0, tolerance=1e-5)


def test_linear_with_bias() raises:
    """Test linear layer correctly adds bias.

    Single sample: [1, 1] @ [[1, 0], [0, 1]]^T + [5, 10] = [6, 11]
    """
    # Input: (1, 2)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    var input = ones(input_shape, DType.float32)

    # Weights: (2, 2) - identity
    var weights_shape = List[Int]()
    weights_shape.append(2)
    weights_shape.append(2)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0]
    weights_data[3] = 1.0  # weights[1, 1]

    # Bias: (2,) - [5, 10]
    var bias_shape = List[Int]()
    bias_shape.append(2)
    var bias = zeros(bias_shape, DType.float32)
    var bias_data = bias._data.bitcast[Float32]()
    bias_data[0] = 5.0
    bias_data[1] = 10.0

    # Compute linear
    var output = linear(input, weights, bias)

    # Check values: [1 + 5, 1 + 10] = [6, 11]
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 11.0, tolerance=1e-5)


def test_linear_no_bias_output_shape() raises:
    """Test linear_no_bias output shape computation.

    Formula: output_shape = (batch_size, out_features)

    Test case: batch=32, in_features=784, out_features=128
    Expected: output shape (32, 128).
    """
    var batch_size = 32
    var in_features = 784
    var out_features = 128

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Compute linear_no_bias: y = xW^T
    var output = linear_no_bias(input, weights)

    # Check output shape: (batch_size, out_features)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)


def test_linear_no_bias_single_sample() raises:
    """Test linear_no_bias with single sample.

    Single sample: (1, 3) @ (2, 3)^T = (1, 2)

    Input: [1, 2, 3]
    Weights: [[1, 0, 0],
              [0, 1, 0]]

    Expected output: [[1, 2]]
    """
    # Input: (1, 3)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    # Weights: (2, 3) - identity-like pattern
    var weights_shape = List[Int]()
    weights_shape.append(2)
    weights_shape.append(3)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0] = 1.0
    weights_data[4] = 1.0  # weights[1, 1] = 1.0

    # Compute linear_no_bias
    var output = linear_no_bias(input, weights)

    # Check shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 2)

    # Check values: output should be [1, 2]
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 2.0, tolerance=1e-5)


def test_linear_backward_output_shape() raises:
    """Test linear_backward produces correct gradient shapes.

    Given:
        grad_output: (batch_size, out_features)
        x: (batch_size, in_features)
        weights: (out_features, in_features)

    Expected gradients:
        grad_input: (batch_size, in_features)
        grad_kernel: (out_features, in_features)
        grad_bias: (out_features,).
    """
    var batch_size = 32
    var in_features = 784
    var out_features = 128

    # Create grad_output: (batch_size, out_features)
    var grad_output_shape = List[Int]()
    grad_output_shape.append(batch_size)
    grad_output_shape.append(out_features)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Compute backward
    var result = linear_backward(grad_output, input, weights)

    # Check gradient shapes
    var grad_input_shape = result.grad_input.shape()
    var grad_weights_shape = result.grad_weights.shape()
    var grad_bias_shape = result.grad_bias.shape()

    assert_equal(grad_input_shape[0], batch_size)
    assert_equal(grad_input_shape[1], in_features)
    assert_equal(grad_weights_shape[0], out_features)
    assert_equal(grad_weights_shape[1], in_features)
    assert_equal(grad_bias_shape[0], out_features)


def test_linear_backward_single_sample() raises:
    """Test linear_backward with single sample.

    Simple case with small tensors to verify math.

    Forward: y = xW^T + b
    x: (1, 2) = [1, 2]
    W: (2, 2) = [[1, 0], [0, 1]] (identity)
    b: (2,) = [0, 0]
    y = [1, 2]

    Backward: grad_output = [1, 1]
    grad_input = grad_output @ W = [1, 1] @ [[1, 0], [0, 1]] = [1, 1]
    grad_kernel = grad_output^T @ x = [1, 1]^T @ [1, 2] = [[1, 2], [1, 2]]
    grad_bias = sum(grad_output, axis=0) = [1, 1]
    """
    # Input: (1, 2)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    var input = ones(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0

    # Weights: (2, 2) - identity
    var weights_shape = List[Int]()
    weights_shape.append(2)
    weights_shape.append(2)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0]
    weights_data[3] = 1.0  # weights[1, 1]

    # grad_output: (1, 2) = [1, 1]
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(2)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Compute backward
    var result = linear_backward(grad_output, input, weights)

    # Check grad_input: should be [1, 1]
    var grad_input_data = result.grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_input_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(grad_input_data[1], 1.0, tolerance=1e-5)

    # Check grad_weights: should be [[1, 2], [1, 2]]
    var grad_weights_data = result.grad_weights._data.bitcast[Float32]()
    assert_almost_equal(grad_weights_data[0], 1.0, tolerance=1e-5)  # [0, 0]
    assert_almost_equal(grad_weights_data[1], 2.0, tolerance=1e-5)  # [0, 1]
    assert_almost_equal(grad_weights_data[2], 1.0, tolerance=1e-5)  # [1, 0]
    assert_almost_equal(grad_weights_data[3], 2.0, tolerance=1e-5)  # [1, 1]

    # Check grad_bias: should be [1, 1]
    var grad_bias_data = result.grad_bias._data.bitcast[Float32]()
    assert_almost_equal(grad_bias_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(grad_bias_data[1], 1.0, tolerance=1e-5)


def test_linear_backward_multiple_samples() raises:
    """Test linear_backward accumulates gradients across batch.

    With batch_size=2:
    grad_output = [[1, 1], [2, 2]]
    grad_bias should sum to [3, 3]
    """
    var batch_size = 2
    var in_features = 2
    var out_features = 2

    # Input: (2, 2)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Weights: (2, 2) - identity
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0]
    weights_data[3] = 1.0  # weights[1, 1]

    # grad_output: (2, 2) = [[1, 1], [2, 2]]
    var grad_output_shape = List[Int]()
    grad_output_shape.append(batch_size)
    grad_output_shape.append(out_features)
    var grad_output = ones(grad_output_shape, DType.float32)
    var grad_output_data = grad_output._data.bitcast[Float32]()
    grad_output_data[0] = 1.0  # grad_output[0, 0]
    grad_output_data[1] = 1.0  # grad_output[0, 1]
    grad_output_data[2] = 2.0  # grad_output[1, 0]
    grad_output_data[3] = 2.0  # grad_output[1, 1]

    # Compute backward
    var result = linear_backward(grad_output, input, weights)

    # Check grad_bias: should be [1+2, 1+2] = [3, 3]
    var grad_bias_data = result.grad_bias._data.bitcast[Float32]()
    assert_almost_equal(grad_bias_data[0], 3.0, tolerance=1e-5)
    assert_almost_equal(grad_bias_data[1], 3.0, tolerance=1e-5)


def test_linear_no_bias_backward_output_shape() raises:
    """Test linear_no_bias_backward produces correct gradient shapes.

    Given:
        grad_output: (batch_size, out_features)
        x: (batch_size, in_features)
        weights: (out_features, in_features)

    Expected gradients:
        grad_input: (batch_size, in_features)
        grad_kernel: (out_features, in_features).
    """
    var batch_size = 32
    var in_features = 784
    var out_features = 128

    # Create grad_output: (batch_size, out_features)
    var grad_output_shape = List[Int]()
    grad_output_shape.append(batch_size)
    grad_output_shape.append(out_features)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Compute backward
    var result = linear_no_bias_backward(grad_output, input, weights)

    # Check gradient shapes (GradientPair has grad_a=grad_input, grad_b=grad_weights)
    var grad_input_shape = result.grad_a.shape()
    var grad_weights_shape = result.grad_b.shape()

    assert_equal(grad_input_shape[0], batch_size)
    assert_equal(grad_input_shape[1], in_features)
    assert_equal(grad_weights_shape[0], out_features)
    assert_equal(grad_weights_shape[1], in_features)


def test_linear_no_bias_backward_single_sample() raises:
    """Test linear_no_bias_backward with single sample.

    Forward: y = xW^T (no bias)
    x: (1, 2) = [1, 2]
    W: (2, 2) = [[1, 0], [0, 1]]
    y = [1, 2]

    Backward: grad_output = [1, 1]
    grad_input = grad_output @ W = [1, 1]
    grad_kernel = grad_output^T @ x = [[1, 2], [1, 2]]
    """
    # Input: (1, 2)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    var input = ones(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0

    # Weights: (2, 2) - identity
    var weights_shape = List[Int]()
    weights_shape.append(2)
    weights_shape.append(2)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 1.0  # weights[0, 0]
    weights_data[3] = 1.0  # weights[1, 1]

    # grad_output: (1, 2) = [1, 1]
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(2)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Compute backward
    var result = linear_no_bias_backward(grad_output, input, weights)

    # Check grad_input (grad_a): should be [1, 1]
    var grad_input_data = result.grad_a._data.bitcast[Float32]()
    assert_almost_equal(grad_input_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(grad_input_data[1], 1.0, tolerance=1e-5)

    # Check grad_weights (grad_b): should be [[1, 2], [1, 2]]
    var grad_weights_data = result.grad_b._data.bitcast[Float32]()
    assert_almost_equal(grad_weights_data[0], 1.0, tolerance=1e-5)  # [0, 0]
    assert_almost_equal(grad_weights_data[1], 2.0, tolerance=1e-5)  # [0, 1]
    assert_almost_equal(grad_weights_data[2], 1.0, tolerance=1e-5)  # [1, 0]
    assert_almost_equal(grad_weights_data[3], 2.0, tolerance=1e-5)  # [1, 1]


def test_linear_batch_processing() raises:
    """Test linear layer processes batches correctly.

    Multiple samples with different values should be processed independently
    in terms of batch dimension but combined in output.
    """
    var batch_size = 3
    var in_features = 2
    var out_features = 1

    # Input: (3, 2)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = zeros(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 1.0  # Sample 1: [1, 1]
    input_data[2] = 2.0
    input_data[3] = 2.0  # Sample 2: [2, 2]
    input_data[4] = 3.0
    input_data[5] = 3.0  # Sample 3: [3, 3]

    # Weights: (1, 2) = [[2, 3]]
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = zeros(weights_shape, DType.float32)
    var weights_data = weights._data.bitcast[Float32]()
    weights_data[0] = 2.0
    weights_data[1] = 3.0

    # Bias: (1,) = [1]
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = ones(bias_shape, DType.float32)

    # Compute linear
    var output = linear(input, weights, bias)

    # Expected outputs:
    # [1, 1] @ [2, 3]^T + 1 = [1*2 + 1*3] + 1 = 6
    # [2, 2] @ [2, 3]^T + 1 = [2*2 + 2*3] + 1 = 11
    # [3, 3] @ [2, 3]^T + 1 = [3*2 + 3*3] + 1 = 16
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 11.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 16.0, tolerance=1e-5)


def test_linear_forward_backward_consistency() raises:
    """Test that forward and backward passes are mathematically consistent.

    Verify that gradient shapes match corresponding forward activations.
    """
    var batch_size = 4
    var in_features = 8
    var out_features = 6

    # Create forward inputs
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)
    var output_shape = output.shape()

    # Create gradient with same shape as output
    var grad_output_shape = List[Int]()
    grad_output_shape.append(output_shape[0])
    grad_output_shape.append(output_shape[1])
    var grad_output = ones(grad_output_shape, DType.float32)

    # Backward pass
    var result = linear_backward(grad_output, input, weights)

    # Verify gradient shapes match forward activation shapes
    assert_equal(result.grad_input.shape()[0], input.shape()[0])
    assert_equal(result.grad_input.shape()[1], input.shape()[1])
    assert_equal(result.grad_weights.shape()[0], weights.shape()[0])
    assert_equal(result.grad_weights.shape()[1], weights.shape()[1])
    assert_equal(result.grad_bias.shape()[0], bias.shape()[0])


def test_linear_large_dimensions() raises:
    """Test linear layer with larger realistic dimensions.

    This ensures the layer works with typical neural network sizes.
    """
    var batch_size = 64
    var in_features = 1024
    var out_features = 512

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass should not raise
    var output = linear(input, weights, bias)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)

    # Backward should also work
    var grad_output_shape = List[Int]()
    grad_output_shape.append(batch_size)
    grad_output_shape.append(out_features)
    var grad_output = ones(grad_output_shape, DType.float32)

    var result = linear_backward(grad_output, input, weights)
    var grad_input_shape = result.grad_input.shape()
    var grad_weights_shape = result.grad_weights.shape()
    var grad_bias_shape = result.grad_bias.shape()

    assert_equal(grad_input_shape[0], batch_size)
    assert_equal(grad_input_shape[1], in_features)
    assert_equal(grad_weights_shape[0], out_features)
    assert_equal(grad_weights_shape[1], in_features)
    assert_equal(grad_bias_shape[0], out_features)


def main() raises:
    """Run all test_linear tests."""
    print("Running test_linear tests...")

    test_linear_initialization()
    print("✓ test_linear_initialization")

    test_linear_output_shape()
    print("✓ test_linear_output_shape")

    test_linear_single_sample()
    print("✓ test_linear_single_sample")

    test_linear_with_bias()
    print("✓ test_linear_with_bias")

    test_linear_no_bias_output_shape()
    print("✓ test_linear_no_bias_output_shape")

    test_linear_no_bias_single_sample()
    print("✓ test_linear_no_bias_single_sample")

    test_linear_backward_output_shape()
    print("✓ test_linear_backward_output_shape")

    test_linear_backward_single_sample()
    print("✓ test_linear_backward_single_sample")

    test_linear_backward_multiple_samples()
    print("✓ test_linear_backward_multiple_samples")

    test_linear_no_bias_backward_output_shape()
    print("✓ test_linear_no_bias_backward_output_shape")

    test_linear_no_bias_backward_single_sample()
    print("✓ test_linear_no_bias_backward_single_sample")

    test_linear_batch_processing()
    print("✓ test_linear_batch_processing")

    test_linear_forward_backward_consistency()
    print("✓ test_linear_forward_backward_consistency")

    test_linear_large_dimensions()
    print("✓ test_linear_large_dimensions")

    print("\nAll test_linear tests passed!")
