"""Unit tests for 2D convolution layer - Part 1: Initialization and Shape Tests.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_conv.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- conv2d: Parameter initialization and output shapes
- Various kernel sizes (1x1, 3x3)
- Various stride values (1, 2)
- Various padding values (0, 1)
- Multi-channel input/output shape checks
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


# ============================================================================
# Conv2D Forward Pass Tests - Initialization and Shapes
# ============================================================================


fn test_conv2d_initialization() raises:
    """Test that conv2d layer parameters can be created with correct shapes.

    Functional API Note:
        Caller creates kernel (out_channels, in_channels, kH, kW) and bias (out_channels,).
        This test verifies parameters can be created.
    """
    var batch_size = 4
    var in_channels = 3
    var out_channels = 16
    var in_height = 32
    var in_width = 32
    var kH = 3
    var kW = 3

    # Create input: (batch, in_channels, height, width)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (out_channels, in_channels, kH, kW)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (out_channels,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var input_s = input.shape()
    var kernel_s = kernel.shape()
    var bias_s = bias.shape()
    assert_equal(input_s[0], batch_size)
    assert_equal(input_s[1], in_channels)
    assert_equal(input_s[2], in_height)
    assert_equal(input_s[3], in_width)
    assert_equal(kernel_s[0], out_channels)
    assert_equal(kernel_s[1], in_channels)
    assert_equal(kernel_s[2], kH)
    assert_equal(kernel_s[3], kW)
    assert_equal(bias_s[0], out_channels)


fn test_conv2d_output_shape_no_padding() raises:
    """Test conv2d output shape with no padding.

    Formula: out_height = (height - kH) // stride + 1
             out_width = (width - kW) // stride + 1

    Test case: batch=1, in_channels=1, height=4, width=4, kH=3, kW=3, stride=1, padding=0
    Expected: output shape (1, 1, 2, 2).
    """
    var batch_size = 1
    var in_channels = 1
    var out_channels = 1
    var in_height = 4
    var in_width = 4
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    # Create input: (1, 1, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 1, 2, 2)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 2)  # (4 - 3) // 1 + 1 = 2
    assert_equal(out_shape[3], 2)  # (4 - 3) // 1 + 1 = 2


fn test_conv2d_output_shape_with_padding() raises:
    """Test conv2d output shape with padding.

    Formula: out_height = (height + 2*padding - kH) // stride + 1
             out_width = (width + 2*padding - kW) // stride + 1

    Test case: batch=1, in_channels=1, height=4, width=4, kH=3, kW=3, stride=1, padding=1
    Expected: output shape (1, 1, 4, 4) - same as input due to padding.
    """
    var batch_size = 1
    var in_channels = 1
    var out_channels = 1
    var in_height = 4
    var in_width = 4
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Create input: (1, 1, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d with padding=1
    var output = conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 1, 4, 4)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 4)  # (4 + 2*1 - 3) // 1 + 1 = 4
    assert_equal(out_shape[3], 4)  # (4 + 2*1 - 3) // 1 + 1 = 4


fn test_conv2d_output_shape_with_stride() raises:
    """Test conv2d output shape with stride > 1.

    Test case: batch=1, in_channels=1, height=8, width=8, kH=3, kW=3, stride=2, padding=1
    Expected: output shape (1, 1, 4, 4).
    """
    var batch_size = 1
    var in_channels = 1
    var out_channels = 1
    var in_height = 8
    var in_width = 8
    var kH = 3
    var kW = 3
    var stride = 2
    var padding = 1

    # Create input: (1, 1, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d with stride=2
    var output = conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 1, 4, 4)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 4)  # (8 + 2*1 - 3) // 2 + 1 = 4
    assert_equal(out_shape[3], 4)  # (8 + 2*1 - 3) // 2 + 1 = 4


fn test_conv2d_1x1_kernel() raises:
    """Test conv2d with 1x1 kernel (special case).

    1x1 kernels are commonly used in modern architectures (e.g., ResNets).
    Should act as a channel-wise linear transformation without spatial mixing.

    Test case: batch=1, in_channels=2, out_channels=3, height=4, width=4, kH=1, kW=1
    """
    var batch_size = 1
    var in_channels = 2
    var out_channels = 3
    var in_height = 4
    var in_width = 4
    var kH = 1
    var kW = 1

    # Create input: (1, 2, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (3, 2, 1, 1)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (3,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 3, 4, 4) - spatial dimensions unchanged
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 4)
    assert_equal(out_shape[3], 4)

    # Check output values: for 1x1 kernel, output should be sum of input channels * kernel
    # With all ones: output = 1.0 * 2 (in_channels) = 2.0
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 2.0, tolerance=1e-5)


fn test_conv2d_multichannel() raises:
    """Test conv2d with multiple input and output channels.

    Verify that the function correctly handles multi-channel convolution.
    """
    var batch_size = 1
    var in_channels = 3
    var out_channels = 2
    var in_height = 5
    var in_width = 5
    var kH = 3
    var kW = 3

    # Create input: (1, 3, 5, 5)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (2, 3, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (2,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride=1, padding=1)

    # Check output shape: (1, 2, 5, 5)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(
        out_shape[2], in_height
    )  # padding=1 preserves spatial dimensions
    assert_equal(out_shape[3], in_width)


fn test_conv2d_multi_channel() raises:
    """Test conv2d with multiple input and output channels.

    Input: (1, 2, 3, 3) - 2 input channels
    Kernel: (3, 2, 2, 2) - 3 output channels, 2 input channels
    Output: (1, 3, 2, 2) - 3 output channels.
    """
    var batch = 1
    var in_channels = 2
    var out_channels = 3
    var in_h = 3
    var in_w = 3

    # Create input: (1, 2, 3, 3)
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (3, 2, 2, 2)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(2)
    kernel_shape.append(2)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (3,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 3, 2, 2)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 3)  # 3 output channels
    assert_equal(out_shape[2], 2)  # (3 - 2) + 1 = 2
    assert_equal(out_shape[3], 2)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Conv2D Part 1 tests: initialization and shape tests."""
    print("Running Conv2D Part 1 tests (initialization and shapes)...")

    test_conv2d_initialization()
    print("✓ test_conv2d_initialization")

    test_conv2d_output_shape_no_padding()
    print("✓ test_conv2d_output_shape_no_padding")

    test_conv2d_output_shape_with_padding()
    print("✓ test_conv2d_output_shape_with_padding")

    test_conv2d_output_shape_with_stride()
    print("✓ test_conv2d_output_shape_with_stride")

    test_conv2d_1x1_kernel()
    print("✓ test_conv2d_1x1_kernel")

    test_conv2d_multichannel()
    print("✓ test_conv2d_multichannel")

    test_conv2d_multi_channel()
    print("✓ test_conv2d_multi_channel")

    print("\nAll Conv2D Part 1 tests passed!")
