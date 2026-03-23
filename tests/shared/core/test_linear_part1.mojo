"""Unit tests for linear layer (fully connected) forward operations - Part 1.

Tests cover:
- linear: Forward pass with bias y = xW^T + b
- linear_no_bias: Forward pass without bias y = xW^T
- linear_backward: Backward pass computing gradients (shape and single sample)
- Shape computations and dimension handling
- Numerical correctness

All tests use pure functional API - no internal state.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_linear.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)
from shared.core.matrix import transpose


# ============================================================================
# Linear Forward Tests
# ============================================================================


fn test_linear_initialization() raises:
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


fn test_linear_output_shape() raises:
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


fn test_linear_single_sample() raises:
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


fn test_linear_with_bias() raises:
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


fn test_linear_no_bias_output_shape() raises:
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


fn test_linear_no_bias_single_sample() raises:
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


# ============================================================================
# Linear Backward Tests (shape and single sample)
# ============================================================================


fn test_linear_backward_output_shape() raises:
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


fn test_linear_backward_single_sample() raises:
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


fn main() raises:
    """Run linear layer forward tests (Part 1)."""
    print("Running linear layer tests (Part 1)...")

    # Forward tests
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

    # Backward shape and single sample tests
    test_linear_backward_output_shape()
    print("✓ test_linear_backward_output_shape")

    test_linear_backward_single_sample()
    print("✓ test_linear_backward_single_sample")

    print("\nAll linear layer tests (Part 1) passed!")
