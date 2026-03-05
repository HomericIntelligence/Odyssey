"""Unit tests for 2D convolution layer operations.

Tests cover:
- conv2d: Forward pass with bias y = conv2d(x, kernel, bias, stride, padding)
- conv2d_no_bias: Forward pass without bias y = conv2d(x, kernel, stride, padding)
- conv2d_backward: Backward pass computing gradients w.r.t. input, kernel, and bias
- conv2d_no_bias_backward: Backward pass without bias
- Shape computations and dimension handling
- Various kernel sizes (1x1, 3x3, 5x5)
- Various stride values (1, 2)
- Various padding values (0, 1, 2)
- Multi-channel input/output
- Numerical correctness
- Gradient shape and value correctness

All tests use pure functional API - no internal state.
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
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)


# ============================================================================
# Conv2D Forward Pass Tests
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


fn test_conv2d_single_sample_simple() raises:
    """Test conv2d with single sample and simple known values.

    Test a 3x3 input with 3x3 kernel to verify correct computation.
    This is a simple case where we can manually compute expected output.
    """
    var batch_size = 1
    var in_channels = 1
    var out_channels = 1
    var in_height = 3
    var in_width = 3
    var kH = 3
    var kW = 3

    # Create input: (1, 1, 3, 3) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3) filled with 1.0
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,) filled with 0.0
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 1, 1, 1) - 3x3 input, 3x3 kernel, no padding, stride 1
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 1)
    assert_equal(out_shape[3], 1)

    # Check output value: sum of 3x3 ones * 1.0 = 9.0
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 9.0, tolerance=1e-5)


fn test_conv2d_with_bias() raises:
    """Test conv2d correctly adds bias term.

    Verify that bias is properly added to the convolution output.
    """
    var batch_size = 1
    var in_channels = 1
    var out_channels = 1
    var in_height = 3
    var in_width = 3
    var kH = 3
    var kW = 3

    # Create input: (1, 1, 3, 3) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3) filled with 1.0
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,) with value 5.0
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)
    var bias_data = bias._data.bitcast[Float32]()
    bias_data[0] = 5.0

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output value: sum of 3x3 ones * 1.0 + bias(5.0) = 9.0 + 5.0 = 14.0
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 14.0, tolerance=1e-5)


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


fn test_conv2d_numerical_correctness() raises:
    """Test conv2d produces correct numerical output.

    Simple case:
    - 1x1 input: [[1.0]]
    - 1x1 kernel: [[2.0]]
    - bias: [0.5]
    - Expected output: 1.0 * 2.0 + 0.5 = 2.5
    """
    # Create input: (1, 1, 1, 1) with value 1.0
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(1)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 1, 1) with value 2.0
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(1)
    var kernel = ones(kernel_shape, DType.float32)
    kernel._data.bitcast[Float32]()[0] = 2.0

    # Create bias: (1,) with value 0.5
    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)
    bias._data.bitcast[Float32]()[0] = 0.5

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output: 1.0 * 2.0 + 0.5 = 2.5
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(2.5), tolerance=1e-5)


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


fn test_conv2d_no_bias() raises:
    """Test conv2d_no_bias produces correct output without bias.

    Should be equivalent to conv2d with zero bias.
    """
    # Create input: (1, 1, 3, 3)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(3)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 2, 2)
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(2)
    kernel_shape.append(2)
    var kernel = ones(kernel_shape, DType.float32)

    # Compute without bias
    var output = conv2d_no_bias(input, kernel, stride=1, padding=0)

    # Check output shape: (1, 1, 2, 2)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)

    # Check numerical value: 4 ones summed = 4.0
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(4.0), tolerance=1e-5)


fn test_conv2d_batched() raises:
    """Test conv2d with batch size > 1.

    Verify that convolution is applied independently to each batch element.
    """
    var batch = 4
    var in_channels = 1
    var out_channels = 1

    # Create input: (4, 1, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 2, 2)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(2)
    kernel_shape.append(2)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (4, 1, 3, 3) - batch size preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], 4)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 3)
    assert_equal(out_shape[3], 3)


# ============================================================================
# Conv2D Backward Pass Tests
# ============================================================================


fn test_conv2d_backward_output_shape() raises:
    """Test that conv2d_backward returns gradients with correct shapes.

    Input: (1, 1, 4, 4), Kernel: (1, 1, 3, 3), Bias: (1,)
    Forward output: (1, 1, 2, 2) -> grad_output shape: (1, 1, 2, 2)
    Expected gradient shapes:
        grad_input: (1, 1, 4, 4) - same as input
        grad_weights: (1, 1, 3, 3) - same as kernel
        grad_bias: (1,) - same as bias
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
    var x = ones(input_shape, DType.float32)

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

    # Forward pass to get output shape: (1, 1, 2, 2)
    var output = conv2d(x, kernel, bias, stride, padding)

    # grad_output = ones with same shape as output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride, padding)

    # Check grad_input shape: (1, 1, 4, 4)
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch_size)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_height)
    assert_equal(gi_shape[3], in_width)

    # Check grad_weights shape: (1, 1, 3, 3)
    var gw_shape = grads.grad_weights.shape()
    assert_equal(gw_shape[0], out_channels)
    assert_equal(gw_shape[1], in_channels)
    assert_equal(gw_shape[2], kH)
    assert_equal(gw_shape[3], kW)

    # Check grad_bias shape: (1,)
    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


fn test_conv2d_backward_single_sample() raises:
    """Test conv2d_backward with analytically-verifiable values.

    Setup: 1x1 kernel = [[1.0]], input = [[1.0, 2.0], [3.0, 4.0]] (1,1,2,2)
    Forward: 1x1 kernel on 2x2 input -> 2x2 output (stride=1, padding=0)
    Each output element = corresponding input element * 1.0
    grad_output = ones (1, 1, 2, 2)

    Expected:
        grad_input = ones (1x1 kernel acts as identity for gradients)
        grad_weights = sum(input) = 1+2+3+4 = 10.0
        grad_bias = sum(grad_output) = 4.0
    """
    # Create input: (1, 1, 2, 2) with values [[1, 2], [3, 4]]
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 1.0
    x_data[1] = 2.0
    x_data[2] = 3.0
    x_data[3] = 4.0

    # Create 1x1 kernel: (1, 1, 1, 1) with value 1.0
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(1)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,) = 0.0
    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    # grad_output = ones (1, 1, 2, 2)
    var go_shape = List[Int]()
    go_shape.append(1)
    go_shape.append(1)
    go_shape.append(2)
    go_shape.append(2)
    var grad_output = ones(go_shape, DType.float32)

    var grads = conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    # grad_input should be all 1.0 (1x1 kernel = 1.0, grad_output all 1.0)
    var gi_data = grads.grad_input._data.bitcast[Float32]()
    assert_almost_equal(gi_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(gi_data[1], 1.0, tolerance=1e-5)
    assert_almost_equal(gi_data[2], 1.0, tolerance=1e-5)
    assert_almost_equal(gi_data[3], 1.0, tolerance=1e-5)

    # grad_weights = sum of (grad_output * input patches) = 1+2+3+4 = 10.0
    var gw_data = grads.grad_weights._data.bitcast[Float32]()
    assert_almost_equal(gw_data[0], 10.0, tolerance=1e-5)

    # grad_bias = sum(grad_output) = 4.0
    var gb_data = grads.grad_bias._data.bitcast[Float32]()
    assert_almost_equal(gb_data[0], 4.0, tolerance=1e-5)


fn test_conv2d_backward_stride_padding() raises:
    """Test conv2d_backward returns correct gradient shapes with stride and padding.

    Input: (1, 1, 8, 8), Kernel: (1, 1, 3, 3), stride=2, padding=1
    Forward output: (1, 1, 4, 4)
    Expected gradient shapes match input and kernel shapes.
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
    var x = ones(input_shape, DType.float32)

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

    # Forward to get output shape
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride, padding)

    # grad_input shape must match input shape
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch_size)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_height)
    assert_equal(gi_shape[3], in_width)

    # grad_weights shape must match kernel shape
    var gw_shape = grads.grad_weights.shape()
    assert_equal(gw_shape[0], out_channels)
    assert_equal(gw_shape[1], in_channels)
    assert_equal(gw_shape[2], kH)
    assert_equal(gw_shape[3], kW)

    # grad_bias shape must match bias shape
    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


fn test_conv2d_no_bias_backward_output_shape() raises:
    """Test conv2d_no_bias_backward returns GradientPair with correct shapes.

    Input: (1, 1, 4, 4), Kernel: (1, 1, 3, 3), stride=1, padding=0
    Forward output: (1, 1, 2, 2)
    Expected:
        grad_a (grad_input): (1, 1, 4, 4)
        grad_b (grad_kernel): (1, 1, 3, 3)
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
    var x = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Forward to get output shape
    var output = conv2d_no_bias(x, kernel, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass (no bias)
    var grads = conv2d_no_bias_backward(grad_output, x, kernel, stride, padding)

    # grad_a = grad_input: (1, 1, 4, 4)
    var ga_shape = grads.grad_a.shape()
    assert_equal(ga_shape[0], batch_size)
    assert_equal(ga_shape[1], in_channels)
    assert_equal(ga_shape[2], in_height)
    assert_equal(ga_shape[3], in_width)

    # grad_b = grad_kernel: (1, 1, 3, 3)
    var gb_shape = grads.grad_b.shape()
    assert_equal(gb_shape[0], out_channels)
    assert_equal(gb_shape[1], in_channels)
    assert_equal(gb_shape[2], kH)
    assert_equal(gb_shape[3], kW)

# ============================================================================
# Integration Tests
# ============================================================================


fn test_conv2d_forward_backward_consistency() raises:
    """Test that forward pass works correctly with various configurations.

    Verifies forward pass produces correct output shapes for batch processing.
    """
    var batch_size = 2
    var in_channels = 3
    var out_channels = 8
    var in_height = 16
    var in_width = 16
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Create forward inputs
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var out_height = (in_height + 2 * padding - kH) // stride + 1
    var out_width = (in_width + 2 * padding - kW) // stride + 1

    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], out_height)
    assert_equal(output_shape[3], out_width)


fn test_conv2d_batch_processing() raises:
    """Test conv2d processes batches correctly.

    Multiple samples should be processed independently but combined in output.
    """
    var batch_size = 4
    var in_channels = 1
    var out_channels = 1
    var in_height = 5
    var in_width = 5
    var kH = 3
    var kW = 3

    # Create input: (4, 1, 5, 5)
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
    var output = conv2d(input, kernel, bias, stride=1, padding=1)

    # Check output shape: (4, 1, 5, 5)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], in_height)
    assert_equal(out_shape[3], in_width)


fn test_conv2d_5x5_kernel() raises:
    """Test conv2d with larger 5x5 kernel.

    Ensures the function works with different kernel sizes.
    """
    var batch_size = 1
    var in_channels = 3
    var out_channels = 16
    var in_height = 32
    var in_width = 32
    var kH = 5
    var kW = 5
    var stride = 1
    var padding = 2

    # Create input: (1, 3, 32, 32)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (16, 3, 5, 5)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (16,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Compute conv2d
    var output = conv2d(input, kernel, bias, stride, padding)

    # Check output shape: (1, 16, 32, 32) with padding=2 preserves dimensions
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], in_height)
    assert_equal(out_shape[3], in_width)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all Conv2D layer tests."""
    print("Running Conv2D layer tests...")

    # Forward pass tests
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

    test_conv2d_single_sample_simple()
    print("✓ test_conv2d_single_sample_simple")

    test_conv2d_with_bias()
    print("✓ test_conv2d_with_bias")

    test_conv2d_multichannel()
    print("✓ test_conv2d_multichannel")

    test_conv2d_numerical_correctness()
    print("✓ test_conv2d_numerical_correctness")

    test_conv2d_multi_channel()
    print("✓ test_conv2d_multi_channel")

    test_conv2d_no_bias()
    print("✓ test_conv2d_no_bias")

    test_conv2d_batched()
    print("✓ test_conv2d_batched")

    # Backward pass tests
    test_conv2d_backward_output_shape()
    print("✓ test_conv2d_backward_output_shape")

    test_conv2d_backward_single_sample()
    print("✓ test_conv2d_backward_single_sample")

    test_conv2d_backward_stride_padding()
    print("✓ test_conv2d_backward_stride_padding")

    test_conv2d_no_bias_backward_output_shape()
    print("✓ test_conv2d_no_bias_backward_output_shape")

    # Integration tests
    test_conv2d_forward_backward_consistency()
    print("✓ test_conv2d_forward_backward_consistency")

    test_conv2d_batch_processing()
    print("✓ test_conv2d_batch_processing")

    test_conv2d_5x5_kernel()
    print("✓ test_conv2d_5x5_kernel")

    print("\nAll Conv2D layer tests passed!")
