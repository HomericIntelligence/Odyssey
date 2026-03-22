# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for neural network layers - Part 1: Linear and Conv2D (initialization/shape).

Tests cover:
- Linear (fully connected) layers: initialization, forward, no-bias, backward
- Convolutional layers (Conv2D): initialization, output shape, stride

Split from test_layers.mojo per ADR-009 (≤10 fn test_ per file).
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
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.linear import linear, linear_no_bias
from shared.core.activation import relu, sigmoid, tanh, softmax


# ============================================================================
# Linear Layer Tests
# ============================================================================


fn test_linear_initialization() raises:
    """Test Linear layer parameter creation.

    Functional API Note:
        Pure functional design - no layer class initialization.
        Caller creates weight matrix (out_features, in_features) and bias vector.
        This test verifies parameters can be created with correct shapes.
    """
    # Create parameters for a linear transformation: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights shape: (out_features, in_features) = (5, 10)
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    # Bias shape: (out_features,) = (5,)
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var w_shape = weights.shape()
    var b_shape = bias.shape()
    assert_equal(w_shape[0], out_features)
    assert_equal(w_shape[1], in_features)
    assert_equal(b_shape[0], out_features)


fn test_linear_forward() raises:
    """Test Linear layer forward pass computation.

    Functional API:
        linear(x, weights, bias) -> output
        - Input shape: (batch_size, in_features)
        - Weights shape: (out_features, in_features)
        - Bias shape: (out_features,)
        - Output shape: (batch_size, out_features)
        - Computation: output = x @ weights.T + bias.
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5
    var batch_size = 2

    # Weights: (5, 10) filled with 0.1
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    # Fill with 0.1
    for i in range(out_features):
        for j in range(in_features):
            weights._data.bitcast[Float32]()[i * in_features + j] = 0.1

    # Bias: (5,) filled with 0.0
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Input: (2, 10) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum of weights = 10 * 0.1 = 1.0
    var expected_value = Float32(10.0 * 0.1)
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5
    )


fn test_linear_no_bias() raises:
    """Test Linear layer without bias term.

    Functional API:
        linear_no_bias(x, weights) -> output
        - No bias parameter required
        - Computation: output = x @ weights.T.
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights: (5, 10) filled with 0.5
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    for i in range(out_features * in_features):
        weights._data.bitcast[Float32]()[i] = 0.5

    # Input: (1, 10) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Forward pass without bias
    var output = linear_no_bias(input, weights)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum = 10 * 0.5 = 5.0 (no bias added)
    var expected_value = Float32(10.0 * 0.5)
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5
    )


fn test_linear_backward() raises:
    """Test Linear layer backward pass (gradient computation).

    Deferred - backward pass implementations are not yet available.
    Will be implemented when autograd/backpropagation system is added.
    """
    pass  # Deferred - backward pass not yet implemented


# ============================================================================
# Conv2D Layer Tests
# ============================================================================


fn test_conv2d_initialization() raises:
    """Test Conv2D layer initialization.

    API Contract:
        Conv2D(
            in_channels: Int,
            out_channels: Int,
            kernel_size: Int,
            stride: Int = 1,
            padding: Int = 0,
            bias: Bool = True
        ).
    """
    # TODO(#1538): Implement when Conv2D is available
    # var layer = Conv2D(
    #     in_channels=3,
    #     out_channels=16,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1
    # )
    # assert_equal(layer.in_channels, 3)
    # assert_equal(layer.out_channels, 16)
    # assert_equal(layer.kernel_size, 3)
    pass


fn test_conv2d_output_shape() raises:
    """Test Conv2D computes correct output shape.

    Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1

    API Contract:
        layer.forward(input: Tensor) -> Tensor
        - Input: (batch, in_channels, height, width)
        - Output: (batch, out_channels, out_height, out_width).
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (batch=1, channels=3, height=32, width=32)
    # # Conv2D: out_channels=16, kernel=3, stride=1, padding=1
    # # Expected output: (1, 16, 32, 32) - same spatial size due to padding
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 32, 32))
    pass


fn test_conv2d_stride() raises:
    """Test Conv2D with stride > 1 downsamples correctly.

    API Contract:
        Conv2D with stride=2 should halve spatial dimensions.
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=3, stride=2, padding=1
    # # Expected output: (1, 16, 16, 16) - halved spatial size
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=2, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run Linear and Conv2D (initialization/shape) layer tests."""
    print("Running Linear layer tests...")
    test_linear_initialization()
    test_linear_forward()
    test_linear_no_bias()
    test_linear_backward()

    print("Running Conv2D layer tests (initialization/shape)...")
    test_conv2d_initialization()
    test_conv2d_output_shape()
    test_conv2d_stride()

    print("\nAll part1 layer tests passed! ✓")
