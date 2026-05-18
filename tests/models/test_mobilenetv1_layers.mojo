"""Unit tests for MobileNetV1 depthwise and pointwise convolution layers.

Tests cover:
- Depthwise convolution: each channel convolved independently
- Pointwise convolution: 1x1 convolution for channel projection

"""


from tests.projectodyssey.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
)
from projectodyssey.core.conv import (
    depthwise_conv2d,
    depthwise_conv2d_backward,
    depthwise_conv2d_no_bias,
    conv2d,
    conv2d_backward,
)
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.layers.batchnorm import BatchNorm2dLayer
from projectodyssey.core.pooling import (
    global_avgpool2d,
    global_avgpool2d_backward,
)


def test_depthwise_conv2d_initialization() raises:
    """Test that depthwise conv2d parameters can be created with correct shapes.

    Depthwise kernel shape: (channels, 1, kH, kW) - one filter per input channel
    This differs from standard conv: (out_channels, in_channels, kH, kW)
    """
    var batch_size = 2
    var channels = 32
    var in_height = 28
    var in_width = 28
    var kH = 3
    var kW = 3

    # Create input: (batch, channels, height, width)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create depthwise kernel: (channels, 1, kH, kW)
    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (channels,)
    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var input_s = input.shape()
    var kernel_s = kernel.shape()
    var bias_s = bias.shape()
    assert_equal(input_s[0], batch_size)
    assert_equal(input_s[1], channels)
    assert_equal(kernel_s[0], channels)
    assert_equal(kernel_s[1], 1)
    assert_equal(kernel_s[2], kH)
    assert_equal(kernel_s[3], kW)
    assert_equal(bias_s[0], channels)


def test_depthwise_conv2d_forward_shape() raises:
    """Test depthwise conv2d output shape computation.

    Depthwise conv: channels stay the same, only spatial dims change
    Input: (batch, channels, H, W)
    Output: (batch, channels, H', W')
    """
    var batch_size = 1
    var channels = 32
    var in_height = 8
    var in_width = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Create input: (1, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create depthwise kernel: (32, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (32,)
    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute depthwise conv2d
    var output = depthwise_conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 32, 8, 8)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], 8)  # (8 + 2*1 - 3) // 1 + 1 = 8
    assert_equal(out_shape[3], 8)  # (8 + 2*1 - 3) // 1 + 1 = 8


def test_depthwise_conv2d_stride2() raises:
    """Test depthwise conv2d with stride 2 (downsampling).

    MobileNetV1 uses stride=2 in several blocks for downsampling.
    """
    var batch_size = 1
    var channels = 64
    var in_height = 14
    var in_width = 14
    var kH = 3
    var kW = 3
    var stride = 2
    var padding = 1

    # Create input: (1, 64, 14, 14)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create depthwise kernel: (64, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (64,)
    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute depthwise conv2d with stride=2
    var output = depthwise_conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 64, 7, 7)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], 7)  # (14 + 2*1 - 3) // 2 + 1 = 7
    assert_equal(out_shape[3], 7)


def test_depthwise_conv2d_backward() raises:
    """Test depthwise conv2d backward pass (gradient computation).

    Verifies that gradients are computed correctly for:
    - grad_input: gradients w.r.t. input
    - grad_kernel: gradients w.r.t. depthwise filters
    - grad_bias: gradients w.r.t. bias terms
    """
    var batch_size = 1
    var channels = 4
    var in_height = 4
    var in_width = 4
    var kH = 3
    var kW = 3

    # Create small tensors for testing
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = depthwise_conv2d(input, kernel, bias, stride=1, padding=1)
    var out_shape = output.shape()

    # Create grad_output
    var grad_output = ones(out_shape, DType.float32)

    # Backward pass
    var result = depthwise_conv2d_backward(
        grad_output, input, kernel, stride=1, padding=1
    )

    # Verify gradient shapes
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    var grad_input_shape = grad_input.shape()
    assert_equal(grad_input_shape[0], batch_size)
    assert_equal(grad_input_shape[1], channels)
    assert_equal(grad_input_shape[2], in_height)
    assert_equal(grad_input_shape[3], in_width)

    var grad_kernel_shape = grad_kernel.shape()
    assert_equal(grad_kernel_shape[0], channels)
    assert_equal(grad_kernel_shape[1], 1)

    var grad_bias_shape = grad_bias.shape()
    assert_equal(grad_bias_shape[0], channels)


def test_pointwise_conv2d_1x1_initialization() raises:
    """Test that pointwise (1x1) convolution parameters are created correctly.

    Pointwise convolution is a standard 1x1 convolution used for:
    - Channel dimension projection
    - Feature transformation without spatial mixing
    """
    var batch_size = 2
    var in_channels = 32
    var out_channels = 64
    var in_height = 8
    var in_width = 8

    # Create input: (batch, in_channels, height, width)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create pointwise kernel: (out_channels, in_channels, 1, 1)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(1)
    kernel_shape.append(1)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (out_channels,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var kernel_s = kernel.shape()
    assert_equal(kernel_s[0], out_channels)
    assert_equal(kernel_s[1], in_channels)
    assert_equal(kernel_s[2], 1)
    assert_equal(kernel_s[3], 1)


def test_pointwise_conv2d_forward() raises:
    """Test pointwise (1x1) convolution forward pass.

    1x1 convolution should preserve spatial dimensions while transforming channels.
    """
    var batch_size = 1
    var in_channels = 32
    var out_channels = 64
    var height = 8
    var width = 8

    # Create input: (1, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create 1x1 kernel: (64, 32, 1, 1)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(1)
    kernel_shape.append(1)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (64,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute pointwise conv
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 64, 8, 8)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)

    # Check output values: 1x1 conv with all ones should produce 32.0
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], Float32(in_channels), tolerance=1e-5)


def test_pointwise_conv2d_backward() raises:
    """Test pointwise (1x1) convolution backward pass.

    Verify gradient computation for channel projection.
    """
    var batch_size = 1
    var in_channels = 16
    var out_channels = 32
    var height = 4
    var width = 4

    # Create input: (1, 16, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create 1x1 kernel: (32, 16, 1, 1)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(1)
    kernel_shape.append(1)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (32,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Create grad_output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(
        grad_output, input, kernel, stride=1, padding=0
    )

    # Verify gradient shapes
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights

    var grad_input_shape = grad_input.shape()
    assert_equal(grad_input_shape[0], batch_size)
    assert_equal(grad_input_shape[1], in_channels)

    var grad_kernel_shape = grad_kernel.shape()
    assert_equal(grad_kernel_shape[0], out_channels)
    assert_equal(grad_kernel_shape[1], in_channels)


def test_depthwise_separable_block_basic() raises:
    """Test depthwise separable block: complete forward pass.

    Block structure:
    1. Depthwise Conv 3x3 + ReLU
    2. Pointwise Conv 1x1 + ReLU

    This tests the combined operation without BatchNorm to isolate the core logic.
    """
    var batch_size = 1
    var in_channels = 32
    var out_channels = 64
    var height = 8
    var width = 8

    # Create input: (1, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Stage 1: Depthwise conv
    var depthwise_kernel_shape = List[Int]()
    depthwise_kernel_shape.append(in_channels)
    depthwise_kernel_shape.append(1)
    depthwise_kernel_shape.append(3)
    depthwise_kernel_shape.append(3)
    var depthwise_kernel = ones(depthwise_kernel_shape, DType.float32)

    var depthwise_bias_shape = List[Int]()
    depthwise_bias_shape.append(in_channels)
    var depthwise_bias = zeros(depthwise_bias_shape, DType.float32)

    var depthwise_output = depthwise_conv2d(
        input, depthwise_kernel, depthwise_bias, stride=1, padding=1
    )

    # Apply ReLU
    var depthwise_relu = relu(depthwise_output)

    # Stage 2: Pointwise conv
    var pointwise_kernel_shape = List[Int]()
    pointwise_kernel_shape.append(out_channels)
    pointwise_kernel_shape.append(in_channels)
    pointwise_kernel_shape.append(1)
    pointwise_kernel_shape.append(1)
    var pointwise_kernel = ones(pointwise_kernel_shape, DType.float32)

    var pointwise_bias_shape = List[Int]()
    pointwise_bias_shape.append(out_channels)
    var pointwise_bias = zeros(pointwise_bias_shape, DType.float32)

    var output = conv2d(
        depthwise_relu, pointwise_kernel, pointwise_bias, stride=1, padding=0
    )

    # Apply ReLU
    var output_relu = relu(output)

    # Verify final output shape
    var out_shape = output_relu.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


def test_depthwise_separable_block_with_stride() raises:
    """Test depthwise separable block with stride 2 for downsampling.

    MobileNetV1 uses stride=2 in blocks 2, 4, and 6 for spatial reduction.
    """
    var batch_size = 1
    var in_channels = 64
    var out_channels = 128
    var in_height = 14
    var in_width = 14
    var stride = 2

    # Create input: (1, 64, 14, 14)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Depthwise conv with stride 2
    var depthwise_kernel_shape = List[Int]()
    depthwise_kernel_shape.append(in_channels)
    depthwise_kernel_shape.append(1)
    depthwise_kernel_shape.append(3)
    depthwise_kernel_shape.append(3)
    var depthwise_kernel = ones(depthwise_kernel_shape, DType.float32)

    var depthwise_bias_shape = List[Int]()
    depthwise_bias_shape.append(in_channels)
    var depthwise_bias = zeros(depthwise_bias_shape, DType.float32)

    var depthwise_output = depthwise_conv2d(
        input, depthwise_kernel, depthwise_bias, stride=stride, padding=1
    )

    # Verify depthwise output shape after stride
    var dw_shape = depthwise_output.shape()
    assert_equal(dw_shape[2], 7)  # (14 + 2*1 - 3) // 2 + 1 = 7
    assert_equal(dw_shape[3], 7)

    # Pointwise conv
    var pointwise_kernel_shape = List[Int]()
    pointwise_kernel_shape.append(out_channels)
    pointwise_kernel_shape.append(in_channels)
    pointwise_kernel_shape.append(1)
    pointwise_kernel_shape.append(1)
    var pointwise_kernel = ones(pointwise_kernel_shape, DType.float32)

    var pointwise_bias_shape = List[Int]()
    pointwise_bias_shape.append(out_channels)
    var pointwise_bias = zeros(pointwise_bias_shape, DType.float32)

    var output = conv2d(
        depthwise_output, pointwise_kernel, pointwise_bias, stride=1, padding=0
    )

    # Verify final output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 7)
    assert_equal(out_shape[3], 7)


def test_batchnorm2d_initialization() raises:
    """Test BatchNorm2D layer initialization.

    Gamma (scale) should be initialized to 1.0
    Beta (shift) should be initialized to 0.0
    Running mean/variance for inference mode
    """
    var num_channels = 64

    # Initialize BatchNorm2D layer
    var bn = BatchNorm2dLayer(num_channels, momentum=0.1, eps=1e-5)

    # Verify parameters shape
    var gamma_shape = bn.gamma.shape()
    var beta_shape = bn.beta.shape()
    var running_mean_shape = bn.running_mean.shape()
    var running_var_shape = bn.running_var.shape()

    assert_equal(gamma_shape[0], num_channels)
    assert_equal(beta_shape[0], num_channels)
    assert_equal(running_mean_shape[0], num_channels)
    assert_equal(running_var_shape[0], num_channels)

    # Verify gamma is initialized to 1.0
    var gamma_data = bn.gamma._data.bitcast[Float32]()
    assert_almost_equal(gamma_data[0], 1.0, tolerance=1e-5)

    # Verify beta is initialized to 0.0
    var beta_data = bn.beta._data.bitcast[Float32]()
    assert_almost_equal(beta_data[0], 0.0, tolerance=1e-5)


def test_batchnorm2d_forward_training() raises:
    """Test BatchNorm2D forward pass in training mode.

    In training mode:
    - Uses batch statistics (mean, variance)
    - Updates running statistics with exponential moving average
    """
    var batch_size = 4
    var num_channels = 32
    var height = 8
    var width = 8

    # Create input: (4, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(num_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Initialize BatchNorm2D
    var bn = BatchNorm2dLayer(num_channels)

    # Forward pass in training mode
    var output = bn.forward(input, training=True)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], num_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


def test_batchnorm2d_forward_inference() raises:
    """Test BatchNorm2D forward pass in inference mode.

    In inference mode:
    - Uses running statistics (mean, variance)
    - Does not update statistics
    """
    var batch_size = 4
    var num_channels = 32
    var height = 8
    var width = 8

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(num_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Initialize BatchNorm2D
    var bn = BatchNorm2dLayer(num_channels)

    # Forward pass in inference mode
    var output = bn.forward(input, training=False)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], num_channels)


def test_relu_activation_basic() raises:
    """Test ReLU activation: clamps negative values to 0.

    ReLU(x) = max(0, x)
    Used after every convolution in MobileNetV1.
    """
    var batch_size = 1
    var channels = 32
    var height = 4
    var width = 4

    # Create input with mixed positive and negative values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set some values to negative
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 0.5
    input_data[2] = -0.5
    input_data[3] = 2.0

    # Apply ReLU
    var output = relu(input)

    # Verify shape is preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)

    # Verify values: negatives become 0, positives stay same
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)  # -1.0 -> 0.0
    assert_almost_equal(output_data[1], 0.5, tolerance=1e-5)  # 0.5 -> 0.5
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)  # -0.5 -> 0.0
    assert_almost_equal(output_data[3], 2.0, tolerance=1e-5)  # 2.0 -> 2.0


def test_relu_multiple_applications() raises:
    """Test ReLU applied multiple times in a forward pass sequence.

    Verifies that ReLU can be applied repeatedly without issues.
    """
    var batch_size = 1
    var channels = 4
    var height = 2
    var width = 2

    # Create input with mixed values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set some values negative
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 0.5
    input_data[2] = 2.0
    input_data[3] = -0.5

    # First ReLU
    var output1 = relu(input)

    # Second ReLU (idempotent after first ReLU)
    var output2 = relu(output1)

    # Verify shapes are preserved
    var shape1 = output1.shape()
    var shape2 = output2.shape()
    assert_equal(shape1[0], batch_size)
    assert_equal(shape2[0], batch_size)

    # After ReLU, all negative values should be 0
    var out_data = output2._data.bitcast[Float32]()
    assert_almost_equal(out_data[0], 0.0, tolerance=1e-5)  # -1.0 -> 0.0
    assert_almost_equal(out_data[1], 0.5, tolerance=1e-5)  # 0.5 -> 0.5
    assert_almost_equal(out_data[2], 2.0, tolerance=1e-5)  # 2.0 -> 2.0
    assert_almost_equal(out_data[3], 0.0, tolerance=1e-5)  # -0.5 -> 0.0


def test_global_avgpool2d_forward() raises:
    """Test global average pooling forward pass.

    Reduces spatial dimensions to 1x1 by averaging all spatial positions per channel.
    Input: (batch, channels, H, W)
    Output: (batch, channels, 1, 1)

    MobileNetV1 uses this before the final FC layer.
    """
    var batch_size = 1
    var channels = 32
    var height = 8
    var width = 8

    # Create input: (1, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Apply global average pooling
    var output = global_avgpool2d(input)

    # Verify output shape: (1, 32, 1, 1)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], 1)
    assert_equal(out_shape[3], 1)

    # Verify output values: average of all ones is 1.0
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 1.0, tolerance=1e-5)


def test_global_avgpool2d_backward() raises:
    """Test global average pooling backward pass.

    Distributes gradients equally to all spatial positions.
    """
    var batch_size = 1
    var channels = 4
    var height = 4
    var width = 4

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = global_avgpool2d(input)

    # Create grad_output: (1, 4, 1, 1)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var grad_input = global_avgpool2d_backward(grad_output, input)

    # Verify grad_input shape
    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], batch_size)
    assert_equal(grad_shape[1], channels)
    assert_equal(grad_shape[2], height)
    assert_equal(grad_shape[3], width)

    # Verify gradient distribution: each spatial position gets 1/(H*W)
    var grad_data = grad_input._data.bitcast[Float32]()
    var expected_grad = 1.0 / Float32(height * width)
    assert_almost_equal(grad_data[0], expected_grad, tolerance=1e-5)


def test_mobilenetv1_block1_32to64() raises:
    """Test MobileNetV1 Block 1: 32->64 channels, stride=1."""
    var batch_size = 1
    var in_channels = 32
    var out_channels = 64
    var height = 28
    var width = 28

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Depthwise conv
    var dw_kernel_shape = List[Int]()
    dw_kernel_shape.append(in_channels)
    dw_kernel_shape.append(1)
    dw_kernel_shape.append(3)
    dw_kernel_shape.append(3)
    var dw_kernel = ones(dw_kernel_shape, DType.float32)
    var dw_bias = zeros([in_channels], DType.float32)

    var dw_output = depthwise_conv2d(
        input, dw_kernel, dw_bias, stride=1, padding=1
    )

    # Pointwise conv
    var pw_kernel_shape = List[Int]()
    pw_kernel_shape.append(out_channels)
    pw_kernel_shape.append(in_channels)
    pw_kernel_shape.append(1)
    pw_kernel_shape.append(1)
    var pw_kernel = ones(pw_kernel_shape, DType.float32)
    var pw_bias = zeros([out_channels], DType.float32)

    var output = conv2d(dw_output, pw_kernel, pw_bias, stride=1, padding=0)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


def test_mobilenetv1_block2_64to128_stride2() raises:
    """Test MobileNetV1 Block 2: 64->128 channels, stride=2 (downsampling)."""
    var batch_size = 1
    var in_channels = 64
    var out_channels = 128
    var in_height = 28
    var in_width = 28

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Depthwise conv with stride 2
    var dw_kernel_shape = List[Int]()
    dw_kernel_shape.append(in_channels)
    dw_kernel_shape.append(1)
    dw_kernel_shape.append(3)
    dw_kernel_shape.append(3)
    var dw_kernel = ones(dw_kernel_shape, DType.float32)
    var dw_bias = zeros([in_channels], DType.float32)

    var dw_output = depthwise_conv2d(
        input, dw_kernel, dw_bias, stride=2, padding=1
    )

    # Verify spatial downsampling
    var dw_shape = dw_output.shape()
    assert_equal(dw_shape[2], 14)
    assert_equal(dw_shape[3], 14)

    # Pointwise conv
    var pw_kernel_shape = List[Int]()
    pw_kernel_shape.append(out_channels)
    pw_kernel_shape.append(in_channels)
    pw_kernel_shape.append(1)
    pw_kernel_shape.append(1)
    var pw_kernel = ones(pw_kernel_shape, DType.float32)
    var pw_bias = zeros([out_channels], DType.float32)

    var output = conv2d(dw_output, pw_kernel, pw_bias, stride=1, padding=0)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 14)
    assert_equal(out_shape[3], 14)


def test_mobilenetv1_block5_512_repeat() raises:
    """Test MobileNetV1 Block 5: 512->512 channels repeated 5x with stride=1.

    This block is repeated 5 times without changing dimensions.
    """
    var batch_size = 1
    var channels = 512
    var height = 7
    var width = 7

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Depthwise conv
    var dw_kernel_shape = List[Int]()
    dw_kernel_shape.append(channels)
    dw_kernel_shape.append(1)
    dw_kernel_shape.append(3)
    dw_kernel_shape.append(3)
    var dw_kernel = ones(dw_kernel_shape, DType.float32)
    var dw_bias = zeros([channels], DType.float32)

    var dw_output = depthwise_conv2d(
        input, dw_kernel, dw_bias, stride=1, padding=1
    )

    # Pointwise conv (no channel change)
    var pw_kernel_shape = List[Int]()
    pw_kernel_shape.append(channels)
    pw_kernel_shape.append(channels)
    pw_kernel_shape.append(1)
    pw_kernel_shape.append(1)
    var pw_kernel = ones(pw_kernel_shape, DType.float32)
    var pw_bias = zeros([channels], DType.float32)

    var output = conv2d(dw_output, pw_kernel, pw_bias, stride=1, padding=0)

    # Verify output shape unchanged
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


def main() raises:
    """Run all test_mobilenetv1_layers tests."""
    print("Running test_mobilenetv1_layers tests...")

    test_depthwise_conv2d_initialization()
    print("✓ test_depthwise_conv2d_initialization")

    test_depthwise_conv2d_forward_shape()
    print("✓ test_depthwise_conv2d_forward_shape")

    test_depthwise_conv2d_stride2()
    print("✓ test_depthwise_conv2d_stride2")

    test_depthwise_conv2d_backward()
    print("✓ test_depthwise_conv2d_backward")

    test_pointwise_conv2d_1x1_initialization()
    print("✓ test_pointwise_conv2d_1x1_initialization")

    test_pointwise_conv2d_forward()
    print("✓ test_pointwise_conv2d_forward")

    test_pointwise_conv2d_backward()
    print("✓ test_pointwise_conv2d_backward")

    test_depthwise_separable_block_basic()
    print("✓ test_depthwise_separable_block_basic")

    test_depthwise_separable_block_with_stride()
    print("✓ test_depthwise_separable_block_with_stride")

    test_batchnorm2d_initialization()
    print("✓ test_batchnorm2d_initialization")

    test_batchnorm2d_forward_training()
    print("✓ test_batchnorm2d_forward_training")

    test_batchnorm2d_forward_inference()
    print("✓ test_batchnorm2d_forward_inference")

    test_relu_activation_basic()
    print("✓ test_relu_activation_basic")

    test_relu_multiple_applications()
    print("✓ test_relu_multiple_applications")

    test_global_avgpool2d_forward()
    print("✓ test_global_avgpool2d_forward")

    test_global_avgpool2d_backward()
    print("✓ test_global_avgpool2d_backward")

    test_mobilenetv1_block1_32to64()
    print("✓ test_mobilenetv1_block1_32to64")

    test_mobilenetv1_block2_64to128_stride2()
    print("✓ test_mobilenetv1_block2_64to128_stride2")

    test_mobilenetv1_block5_512_repeat()
    print("✓ test_mobilenetv1_block5_512_repeat")

    print("\nAll test_mobilenetv1_layers tests passed!")
