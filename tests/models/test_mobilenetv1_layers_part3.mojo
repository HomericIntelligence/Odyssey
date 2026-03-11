# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mobilenetv1_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for MobileNetV1 global average pooling and unique channel configurations.

Tests cover:
- Global average pooling: spatial dimension reduction
- Unique channel configurations: Block 1, Block 2, Block 5

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
# Global Average Pooling Tests
# ============================================================================


fn test_global_avgpool2d_forward() raises:
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


fn test_global_avgpool2d_backward() raises:
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


# ============================================================================
# Unique Channel Configuration Tests
# ============================================================================


fn test_mobilenetv1_block1_32to64() raises:
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


fn test_mobilenetv1_block2_64to128_stride2() raises:
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


fn test_mobilenetv1_block5_512_repeat() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print(
        "Starting MobileNetV1 Layerwise Tests Part 3 (Global AvgPool + Channel"
        " Configs)..."
    )

    print("  test_global_avgpool2d_forward...", end="")
    test_global_avgpool2d_forward()
    print(" OK")

    print("  test_global_avgpool2d_backward...", end="")
    test_global_avgpool2d_backward()
    print(" OK")

    print("  test_mobilenetv1_block1_32to64...", end="")
    test_mobilenetv1_block1_32to64()
    print(" OK")

    print("  test_mobilenetv1_block2_64to128_stride2...", end="")
    test_mobilenetv1_block2_64to128_stride2()
    print(" OK")

    print("  test_mobilenetv1_block5_512_repeat...", end="")
    test_mobilenetv1_block5_512_repeat()
    print(" OK")

    print("All MobileNetV1 layerwise tests part 3 passed!")
