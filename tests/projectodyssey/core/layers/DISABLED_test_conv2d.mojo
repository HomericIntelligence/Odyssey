"""Unit tests for Conv2dLayer class.

Tests cover:
- Conv2dLayer initialization with correct parameter shapes
- Forward pass computation
- Backward pass gradient computation
- Parameter extraction and management
- Different kernel sizes, strides, and padding

Following TDD principles - these tests define the expected API for Conv2dLayer.
"""

from projectodyssey.testing.assertions import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, randn
from projectodyssey.core.layers.conv2d import Conv2dLayer


# ============================================================================
# Conv2dLayer Initialization Tests
# ============================================================================


def test_conv2d_initialization() raises:
    """Test Conv2dLayer parameter creation with correct shapes.

    Verifies that weights and biases are initialized with correct dimensions.
    """
    var in_channels = 3
    var out_channels = 16
    var kernel_h = 3
    var kernel_w = 3

    var layer = Conv2dLayer(in_channels, out_channels, kernel_h, kernel_w)

    # Check weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    var weight_shape = layer.weight.shape()
    assert_equal_int(len(weight_shape), 4)
    assert_equal(weight_shape[0], out_channels)
    assert_equal(weight_shape[1], in_channels)
    assert_equal(weight_shape[2], kernel_h)
    assert_equal(weight_shape[3], kernel_w)

    # Check bias shape: (out_channels,)
    var bias_shape = layer.bias.shape()
    assert_equal_int(len(bias_shape), 1)
    assert_equal(bias_shape[0], out_channels)


def test_conv2d_initialization_with_stride_padding() raises:
    """Test Conv2dLayer initialization with stride and padding parameters.

    Verifies that stride and padding are stored correctly.
    """
    var in_channels = 3
    var out_channels = 32
    var kernel_h = 5
    var kernel_w = 5
    var stride = 2
    var padding = 2

    var layer = Conv2dLayer(
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride=stride,
        padding=padding,
    )

    # Check parameters stored
    assert_equal(layer.in_channels, in_channels)
    assert_equal(layer.out_channels, out_channels)
    assert_equal(layer.kernel_h, kernel_h)
    assert_equal(layer.kernel_w, kernel_w)
    assert_equal(layer.stride, stride)
    assert_equal(layer.padding, padding)


def test_conv2d_weight_initialization_scale() raises:
    """Test that weights are initialized with reasonable scale.

    He initialization should scale weights by sqrt(2 / (in_channels * kH * kW))
    We verify that weights are not too large or too small.
    """
    var in_channels = 3
    var out_channels = 16
    var kernel_h = 3
    var kernel_w = 3

    var layer = Conv2dLayer(in_channels, out_channels, kernel_h, kernel_w)

    # He scale should be sqrt(2 / (3 * 3 * 3)) ≈ 0.264
    # We check that weights are roughly in this scale (not zeros, not too large)
    var weight_data = layer.weight._data.bitcast[Float32]()
    var num_weights = layer.weight.numel()

    var sum_abs = Float32(0.0)
    for i in range(num_weights):
        var abs_val = Float32(0.0)
        if weight_data[i] > 0.0:
            abs_val = weight_data[i]
        else:
            abs_val = -weight_data[i]
        sum_abs += abs_val

    var mean_abs = sum_abs / Float32(num_weights)

    # Mean absolute value should be reasonably scaled (not 0, not > 1)
    assert_true(mean_abs > Float32(0.01), "Weights too small - not initialized")
    assert_true(mean_abs < Float32(1.0), "Weights too large - bad scaling")


def test_conv2d_bias_initialized_to_zero() raises:
    """Test that bias is initialized to zero."""
    var layer = Conv2dLayer(3, 16, 3, 3)

    var bias_data = layer.bias._data.bitcast[Float32]()
    for i in range(layer.bias.numel()):
        assert_almost_equal(bias_data[i], 0.0, tolerance=1e-6)


# ============================================================================
# Conv2dLayer Forward Pass Tests
# ============================================================================


def test_conv2d_forward_output_shape() raises:
    """Test Conv2dLayer forward pass produces correct output shape.

    Formula: out_size = (in_size + 2*padding - kernel_size) / stride + 1
    """
    var in_channels = 3
    var out_channels = 16
    var kernel_h = 3
    var kernel_w = 3
    var stride = 1
    var padding = 1

    var layer = Conv2dLayer(
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride=stride,
        padding=padding,
    )

    # Input: (batch=2, channels=3, height=32, width=32)
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(in_channels)
    input_shape.append(32)
    input_shape.append(32)
    var input = randn(input_shape, DType.float32)

    var output = layer.forward(input)

    # Expected output shape: (2, 16, 32, 32)
    # height: (32 + 2*1 - 3) / 1 + 1 = 32
    # width: (32 + 2*1 - 3) / 1 + 1 = 32
    var output_shape = output.shape()
    assert_equal(output_shape[0], 2)  # batch
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], 32)  # height
    assert_equal(output_shape[3], 32)  # width


def test_conv2d_forward_with_stride() raises:
    """Test Conv2dLayer forward pass with stride > 1.

    Stride downsamples spatial dimensions by stride factor.
    """
    var in_channels = 3
    var out_channels = 16
    var stride = 2
    var padding = 0

    var layer = Conv2dLayer(
        in_channels, out_channels, 3, 3, stride=stride, padding=padding
    )

    # Input: (1, 3, 32, 32)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_channels)
    input_shape.append(32)
    input_shape.append(32)
    var input = randn(input_shape, DType.float32)

    var output = layer.forward(input)

    # Expected output shape: (1, 16, 15, 15)
    # height: (32 + 2*0 - 3) / 2 + 1 = 15
    # width: (32 + 2*0 - 3) / 2 + 1 = 15
    var output_shape = output.shape()
    assert_equal(output_shape[0], 1)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], 15)
    assert_equal(output_shape[3], 15)


def test_conv2d_forward_no_padding() raises:
    """Test Conv2dLayer forward pass with no padding (valid convolution).

    No padding reduces spatial dimensions by (kernel_size - 1).
    """
    var in_channels = 3
    var out_channels = 16
    var kernel_size = 5
    var padding = 0

    var layer = Conv2dLayer(
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
        stride=1,
        padding=padding,
    )

    # Input: (1, 3, 32, 32)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_channels)
    input_shape.append(32)
    input_shape.append(32)
    var input = randn(input_shape, DType.float32)

    var output = layer.forward(input)

    # Expected output shape: (1, 16, 28, 28)
    # height: (32 + 0 - 5) / 1 + 1 = 28
    var output_shape = output.shape()
    assert_equal(output_shape[2], 28)
    assert_equal(output_shape[3], 28)


def test_conv2d_forward_batch_independence() raises:
    """Property: Conv2dLayer processes batch elements independently.

    Processing a batch should give same results as processing individually.
    """
    var layer = Conv2dLayer(3, 16, 3, 3, stride=1, padding=1)

    # Create batch input: (2, 3, 16, 16)
    var batch_input_shape = List[Int]()
    batch_input_shape.append(2)
    batch_input_shape.append(3)
    batch_input_shape.append(16)
    batch_input_shape.append(16)
    var batch_input = randn(batch_input_shape, DType.float32)

    # Process as batch
    var batch_output = layer.forward(batch_input)

    # Process first element individually: (1, 3, 16, 16)
    var single_input_shape = List[Int]()
    single_input_shape.append(1)
    single_input_shape.append(3)
    single_input_shape.append(16)
    single_input_shape.append(16)
    var single_input = zeros(single_input_shape, DType.float32)

    # Copy first batch element to single input
    # Use proper tensor indexing with channel/height/width dimensions
    for c in range(3):  # channels
        for h in range(16):  # height
            for w in range(16):  # width
                var idx = c * 16 * 16 + h * 16 + w
                var src_idx = 0 * (3 * 16 * 16) + idx  # batch 0
                single_input._data[idx] = batch_input._data[src_idx]

    var single_output = layer.forward(single_input)

    # First batch element output should match individual processing
    var batch_out_spatial = 16 * 16 * 16  # out_channels * height * width
    for i in range(batch_out_spatial):
        assert_almost_equal(
            batch_output._data.bitcast[Float32]()[i],
            single_output._data.bitcast[Float32]()[i],
            tolerance=1e-4,
        )


# ============================================================================
# Conv2dLayer Backward Pass Tests
# ============================================================================


def test_conv2d_backward_gradient_shapes() raises:
    """Test Conv2dLayer backward pass returns gradients with correct shapes.

    Backward should return (grad_input, grad_weight, grad_bias) with matching
    shapes to forward inputs/parameters.
    """
    var in_channels = 3
    var out_channels = 16
    var kernel_h = 3
    var kernel_w = 3

    var layer = Conv2dLayer(
        in_channels, out_channels, kernel_h, kernel_w, stride=1, padding=1
    )

    # Input and forward pass
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(in_channels)
    input_shape.append(32)
    input_shape.append(32)
    var input = randn(input_shape, DType.float32)
    var output = layer.forward(input)

    # Create gradient w.r.t. output
    var grad_output = randn(output.shape(), DType.float32)

    # Backward pass
    var grads = layer.backward(grad_output, input)
    var grad_input = grads[0]
    var grad_weight = grads[1]
    var grad_bias = grads[2]

    # Check gradient shapes match input/parameter shapes
    var grad_input_shape = grad_input.shape()
    assert_equal(grad_input_shape[0], 2)
    assert_equal(grad_input_shape[1], in_channels)
    assert_equal(grad_input_shape[2], 32)
    assert_equal(grad_input_shape[3], 32)

    var grad_weight_shape = grad_weight.shape()
    assert_equal(grad_weight_shape[0], out_channels)
    assert_equal(grad_weight_shape[1], in_channels)
    assert_equal(grad_weight_shape[2], kernel_h)
    assert_equal(grad_weight_shape[3], kernel_w)

    var grad_bias_shape = grad_bias.shape()
    assert_equal(grad_bias_shape[0], out_channels)


# ============================================================================
# Conv2dLayer Parameters Tests
# ============================================================================


def test_conv2d_parameters_list() raises:
    """Test Conv2dLayer.parameters() returns weight and bias tensors."""
    var layer = Conv2dLayer(3, 16, 3, 3)

    var params = layer.parameters()

    # Should return [weight, bias]
    assert_equal(len(params), 2)

    # First parameter is weight
    var weight = params[0]
    var weight_shape = weight.shape()
    assert_equal(weight_shape[0], 16)
    assert_equal(weight_shape[1], 3)
    assert_equal(weight_shape[2], 3)
    assert_equal(weight_shape[3], 3)

    # Second parameter is bias
    var bias = params[1]
    var bias_shape = bias.shape()
    assert_equal(bias_shape[0], 16)


# ============================================================================
# Test Main
# ============================================================================


def main() raises:
    """Run all Conv2dLayer tests."""
    print("Running Conv2dLayer initialization tests...")
    test_conv2d_initialization()
    test_conv2d_initialization_with_stride_padding()
    test_conv2d_weight_initialization_scale()
    test_conv2d_bias_initialized_to_zero()

    print("Running Conv2dLayer forward pass tests...")
    test_conv2d_forward_output_shape()
    test_conv2d_forward_with_stride()
    test_conv2d_forward_no_padding()
    test_conv2d_forward_batch_independence()

    print("Running Conv2dLayer backward pass tests...")
    test_conv2d_backward_gradient_shapes()

    print("Running Conv2dLayer parameters tests...")
    test_conv2d_parameters_list()

    print("\nAll Conv2dLayer tests passed! ✓")
