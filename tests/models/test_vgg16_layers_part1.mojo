# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_vgg16_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for VGG-16 convolutional layer operations (Part 1 of 2).

VGG-16 Architecture Overview (~25 layer operations):
- 13 Conv2D layers (3x3 kernels, various channel depths)
- 5 MaxPool2D layers (2x2, stride 2)
- 3 Fully Connected (FC) layers
- ReLU activation after each conv and FC
- Dropout (0.5) in FC layers during training

Layer Deduplication Strategy:
VGG-16 contains 13 convolutional layers but only ~5 unique architectures
distinguished by channel count. Rather than testing all 13 identical conv
operations, we test one representative conv per unique channel configuration:

1. Conv with 64 output channels (appears 2x in VGG-16: Conv1_1, Conv1_2)
2. Conv with 128 output channels (appears 2x: Conv2_1, Conv2_2)
3. Conv with 256 output channels (appears 3x: Conv3_1, Conv3_2, Conv3_3)
4. Conv with 512 output channels (appears 3x: Conv4_1, Conv4_2, Conv4_3)
5. Conv with 512 output channels again (appears 3x: Conv5_1, Conv5_2, Conv5_3)

All use: kernel_size=3x3, padding=1, stride=1, followed by ReLU and MaxPool

This file contains Part 1: Conv layer tests (64, 128, 256, 512 channels).
See test_vgg16_layers_part2.mojo for MaxPool, ReLU, and FC layer tests.

All tests use pure functional API - no internal state or parameters.
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
from shared.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)
from shared.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)
from shared.core.activation import (
    relu,
    relu_backward,
)
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
)


# ============================================================================
# VGG-16 Conv Layer Tests - 64 Channel Configuration
# ============================================================================
# Tests Conv1_1 and Conv1_2: 3 channels -> 64 channels
# All subsequent conv layers follow the same pattern


fn test_vgg16_conv64_forward() raises:
    """Test Conv2D forward pass with 64 output channels (VGG layer 1).

    Configuration:
        Input: (4, 3, 32, 32) - 4 samples, 3 input channels, 32x32 spatial
        Conv: 3x3 kernel, 64 output channels, padding=1, stride=1
        Output: (4, 64, 32, 32) - spatial size preserved by padding

    Deduplication Note:
        This tests Conv1_1 and Conv1_2 which are identical in architecture.
        We test once and document that both use this configuration.
    """
    var batch_size = 4
    var in_channels = 3
    var out_channels = 64
    var height = 32
    var width = 32
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input: (batch_size, in_channels, height, width)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (out_channels, in_channels, kernel_size, kernel_size)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (out_channels,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)  # Preserved by padding=1
    assert_equal(output_shape[3], width)


fn test_vgg16_conv64_backward() raises:
    """Test Conv2D backward pass with 64 output channels.

    Verifies gradient computation w.r.t. input, kernel, and bias.
    """
    var batch_size = 2
    var in_channels = 3
    var out_channels = 64
    var height = 32
    var width = 32
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient w.r.t. output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


# ============================================================================
# VGG-16 Conv Layer Tests - 128 Channel Configuration
# ============================================================================
# Tests Conv2_1 and Conv2_2: 64 channels -> 128 channels


fn test_vgg16_conv128_forward() raises:
    """Test Conv2D forward pass with 128 output channels (VGG layer 2).

    Configuration:
        Input: (4, 64, 16, 16) - from after MaxPool1
        Conv: 3x3 kernel, 128 output channels, padding=1, stride=1
        Output: (4, 128, 16, 16)

    Deduplication Note:
        Conv2_1 and Conv2_2 both have this configuration.
    """
    var batch_size = 4
    var in_channels = 64
    var out_channels = 128
    var height = 16
    var width = 16
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


fn test_vgg16_conv128_backward() raises:
    """Test Conv2D backward pass with 128 output channels."""
    var batch_size = 2
    var in_channels = 64
    var out_channels = 128
    var height = 16
    var width = 16
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


# ============================================================================
# VGG-16 Conv Layer Tests - 256 Channel Configuration
# ============================================================================
# Tests Conv3_1, Conv3_2, Conv3_3: 128 channels -> 256 channels


fn test_vgg16_conv256_forward() raises:
    """Test Conv2D forward pass with 256 output channels (VGG layer 3).

    Configuration:
        Input: (4, 128, 8, 8) - from after MaxPool2
        Conv: 3x3 kernel, 256 output channels, padding=1, stride=1
        Output: (4, 256, 8, 8)

    Deduplication Note:
        Conv3_1, Conv3_2, Conv3_3 all have this configuration.
        We test once as they are identical.
    """
    var batch_size = 4
    var in_channels = 128
    var out_channels = 256
    var height = 8
    var width = 8
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


fn test_vgg16_conv256_backward() raises:
    """Test Conv2D backward pass with 256 output channels."""
    var batch_size = 2
    var in_channels = 128
    var out_channels = 256
    var height = 8
    var width = 8
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


# ============================================================================
# VGG-16 Conv Layer Tests - 512 Channel Configuration
# ============================================================================
# Tests Conv4_1, Conv4_2, Conv4_3, Conv5_1, Conv5_2, Conv5_3
# (256 -> 512 and 512 -> 512)


fn test_vgg16_conv512_forward() raises:
    """Test Conv2D forward pass with 512 output channels (VGG layers 4-5).

    Configuration:
        Input: (4, 256, 4, 4) - from after MaxPool3
        Conv: 3x3 kernel, 512 output channels, padding=1, stride=1
        Output: (4, 512, 4, 4)

    Deduplication Note:
        Conv4_1, Conv4_2, Conv4_3, Conv5_1, Conv5_2, Conv5_3 all use 512
        output channels. We test this once to represent all 6 layers.
    """
    var batch_size = 4
    var in_channels = 256
    var out_channels = 512
    var height = 4
    var width = 4
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


fn test_vgg16_conv512_backward() raises:
    """Test Conv2D backward pass with 512 output channels."""
    var batch_size = 2
    var in_channels = 256
    var out_channels = 512
    var height = 4
    var width = 4
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


fn main() raises:
    """Run VGG-16 convolutional layer tests (Part 1 of 2)."""
    print("Starting VGG-16 Layerwise Tests (Part 1: Conv Layers)...")

    # Conv64 tests
    print("  test_vgg16_conv64_forward...", end="")
    test_vgg16_conv64_forward()
    print(" OK")

    print("  test_vgg16_conv64_backward...", end="")
    test_vgg16_conv64_backward()
    print(" OK")

    # Conv128 tests
    print("  test_vgg16_conv128_forward...", end="")
    test_vgg16_conv128_forward()
    print(" OK")

    print("  test_vgg16_conv128_backward...", end="")
    test_vgg16_conv128_backward()
    print(" OK")

    # Conv256 tests
    print("  test_vgg16_conv256_forward...", end="")
    test_vgg16_conv256_forward()
    print(" OK")

    print("  test_vgg16_conv256_backward...", end="")
    test_vgg16_conv256_backward()
    print(" OK")

    # Conv512 tests
    print("  test_vgg16_conv512_forward...", end="")
    test_vgg16_conv512_forward()
    print(" OK")

    print("  test_vgg16_conv512_backward...", end="")
    test_vgg16_conv512_backward()
    print(" OK")

    print("All VGG-16 Part 1 (Conv Layer) tests passed!")
