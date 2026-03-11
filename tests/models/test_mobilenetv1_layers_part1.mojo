# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mobilenetv1_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for MobileNetV1 depthwise and pointwise convolution layers.

Tests cover:
- Depthwise convolution: each channel convolved independently
- Pointwise convolution: 1x1 convolution for channel projection

Split from test_mobilenetv1_layers.mojo per ADR-009 (≤10 fn test_ per file).
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
)
from shared.core.conv import (
    depthwise_conv2d,
    depthwise_conv2d_backward,
    depthwise_conv2d_no_bias,
    conv2d,
    conv2d_backward,
)
from shared.core.activation import relu, relu_backward
from shared.core.layers.batchnorm import BatchNorm2dLayer
from shared.core.pooling import global_avgpool2d, global_avgpool2d_backward


# ============================================================================
# Depthwise Convolution Tests
# ============================================================================


fn test_depthwise_conv2d_initialization() raises:
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


fn test_depthwise_conv2d_forward_shape() raises:
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


fn test_depthwise_conv2d_stride2() raises:
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


fn test_depthwise_conv2d_backward() raises:
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


# ============================================================================
# Pointwise Convolution Tests (1x1 Conv)
# ============================================================================


fn test_pointwise_conv2d_1x1_initialization() raises:
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


fn test_pointwise_conv2d_forward() raises:
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


fn test_pointwise_conv2d_backward() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print(
        "Starting MobileNetV1 Layerwise Tests Part 1 (Depthwise + Pointwise"
        " Conv)..."
    )

    print("  test_depthwise_conv2d_initialization...", end="")
    test_depthwise_conv2d_initialization()
    print(" OK")

    print("  test_depthwise_conv2d_forward_shape...", end="")
    test_depthwise_conv2d_forward_shape()
    print(" OK")

    print("  test_depthwise_conv2d_stride2...", end="")
    test_depthwise_conv2d_stride2()
    print(" OK")

    print("  test_depthwise_conv2d_backward...", end="")
    test_depthwise_conv2d_backward()
    print(" OK")

    print("  test_pointwise_conv2d_1x1_initialization...", end="")
    test_pointwise_conv2d_1x1_initialization()
    print(" OK")

    print("  test_pointwise_conv2d_forward...", end="")
    test_pointwise_conv2d_forward()
    print(" OK")

    print("  test_pointwise_conv2d_backward...", end="")
    test_pointwise_conv2d_backward()
    print(" OK")

    print("All MobileNetV1 layerwise tests part 1 passed!")
