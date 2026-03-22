# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for neural network layers - Part 2: Conv2D (padding), Activations, and Pooling.

Tests cover:
- Convolutional layers (Conv2D): valid padding
- Activation layers (ReLU, Sigmoid, Tanh)
- Pooling layers (MaxPool2D)

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
# Conv2D Layer Tests (continued)
# ============================================================================


fn test_conv2d_valid_padding() raises:
    """Test Conv2D with no padding (valid convolution).

    API Contract:
        Conv2D with padding=0 reduces spatial dimensions.
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=5, stride=1, padding=0
    # # Expected output: (1, 16, 28, 28) - reduced by kernel_size-1
    #
    # var layer = Conv2D(3, 16, kernel_size=5, stride=1, padding=0)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 28, 28))
    pass


# ============================================================================
# Activation Layer Tests
# ============================================================================


fn test_relu_activation() raises:
    """Test ReLU zeros negative values and preserves positive values.

    Functional API:
        relu(x) -> output
        - For each element: output = max(0, input).
    """
    # Test with known values: [-2.0, -1.0, 0.0, 1.0, 2.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -2.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 2.0

    # Apply ReLU
    var output = relu(input)

    # Expected: [0.0, 0.0, 0.0, 1.0, 2.0]
    assert_almost_equal(output._data.bitcast[Float32]()[0], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[3], 1.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[4], 2.0, tolerance=1e-6)


fn test_relu_in_place() raises:
    """Test ReLU can modify input in-place for memory efficiency.

    Not applicable to pure functional design - functional operations
    always return new tensors and never mutate inputs.
    """
    pass  # Not applicable - pure functional design


fn test_sigmoid_range() raises:
    """Test Sigmoid outputs values in range [0, 1].

    Functional API:
        sigmoid(x) -> output
        - For each element: output = 1 / (1 + exp(-input))
        - Output range: (0, 1).
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -10.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 10.0

    # Apply sigmoid
    var output = sigmoid(input)

    # All outputs should be in (0, 1)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(Float32(0.0) < val, "Value must be greater than 0")
        assert_true(val < Float32(1.0), "Value must be less than 1")

    # Check sigmoid(0) = 0.5
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.5, tolerance=1e-6)


fn test_tanh_range() raises:
    """Test Tanh outputs values in range [-1, 1].

    Functional API:
        tanh(x) -> output
        - For each element: output = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        - Output range: (-1, 1).
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -10.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 10.0

    # Apply tanh
    var output = tanh(input)

    # All outputs should be in [-1, 1] (inclusive due to FP precision)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(Float32(-1.0) <= val, "Value must be >= -1")
        assert_true(val <= Float32(1.0), "Value must be <= 1")

    # Check tanh(0) = 0.0
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)


# ============================================================================
# Pooling Layer Tests
# ============================================================================


fn test_maxpool2d_downsampling() raises:
    """Test MaxPool2D downsamples spatial dimensions.

    API Contract:
        MaxPool2D(kernel_size: Int, stride: Int = None, padding: Int = 0)
        - Reduces spatial dimensions by kernel_size (if stride=kernel_size).
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # # Input: (1, 16, 32, 32)
    # # MaxPool2D: kernel=2, stride=2
    # # Expected output: (1, 16, 16, 16)
    #
    # var pool = MaxPool2D(kernel_size=2, stride=2)
    # var input = Tensor.randn(1, 16, 32, 32)
    # var output = pool.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


fn test_maxpool2d_max_selection() raises:
    """Test MaxPool2D selects maximum value in each window.

    API Contract:
        MaxPool2D selects max over kernel_size x kernel_size window.
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # var pool = MaxPool2D(kernel_size=2)
    #
    # # Create input with known values
    # # [[1, 2], [3, 4]] -> max = 4
    # var data = [1.0, 2.0, 3.0, 4.0]
    # var input = AnyTensor([1, 1, 2, 2], DType.float32)
    # # Fill with data
    # var output = pool.forward(input)
    #
    # # Output should be single value: 4.0
    # assert_almost_equal(output._get_float64(0), 4.0)
    pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run Conv2D (padding), Activation, and Pooling layer tests."""
    print("Running Conv2D valid padding test...")
    test_conv2d_valid_padding()

    print("Running activation layer tests...")
    test_relu_activation()
    test_relu_in_place()
    test_sigmoid_range()
    test_tanh_range()

    print("Running pooling layer tests...")
    test_maxpool2d_downsampling()
    test_maxpool2d_max_selection()

    print("\nAll part2 layer tests passed! ✓")
