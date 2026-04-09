"""Unit tests for 2D convolution layer

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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, ones_like
from shared.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)
from shared.testing import (
    compute_numerical_gradient,
    assert_gradients_close,
)
from shared.core.reduction import sum as reduce_sum
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full


def test_conv2d_initialization() raises:
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


def test_conv2d_output_shape_no_padding() raises:
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


def test_conv2d_output_shape_with_padding() raises:
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


def test_conv2d_output_shape_with_stride() raises:
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


def test_conv2d_1x1_kernel() raises:
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


def test_conv2d_single_sample_simple() raises:
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


def test_conv2d_with_bias() raises:
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


def test_conv2d_multichannel() raises:
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


def test_conv2d_numerical_correctness() raises:
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
    kernel.set(0, Float32(2.0))

    # Create bias: (1,) with value 0.5
    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)
    bias.set(0, Float32(0.5))

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output: 1.0 * 2.0 + 0.5 = 2.5
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(2.5), tolerance=1e-5)


def test_conv2d_multi_channel() raises:
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


def test_conv2d_no_bias() raises:
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


def test_conv2d_batched() raises:
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


def test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 1
    var in_channels = 1
    var in_height = 4
    var in_width = 4
    var out_channels = 1
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

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
    var output = conv2d(x, kernel, bias, stride, padding)

    # Gradient w.r.t. output (ones_like)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # grad_input should match input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_kernel should match kernel shape
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)

    # grad_bias should match bias shape
    assert_equal(grad_bias.shape()[0], out_channels)


def test_conv2d_backward_bias_gradient() raises:
    """Test that conv2d_backward computes correct gradient w.r.t. bias.

    grad_bias[oc] = sum of grad_output over all (batch, height, width) positions
    for that output channel.
    """
    var batch = 2
    var in_channels = 1
    var in_height = 4
    var in_width = 4
    var out_channels = 2
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

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
    var output = conv2d(x, kernel, bias, stride, padding)

    # Output shape: (2, 2, 2, 2) - batch=2, out_channels=2, out_h=2, out_w=2
    # grad_output = ones
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_bias = result.grad_bias

    # grad_bias[oc] = sum of grad_output for that output channel
    # With ones gradient and output shape (2, 2, 2, 2):
    # Each channel has 2*2*2 = 8 output elements
    # grad_bias should be 8.0 for each channel
    var out_h = output.shape()[2]
    var out_w = output.shape()[3]
    var expected_grad_bias = Float32(batch * out_h * out_w)

    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias._data.bitcast[Float32]()[oc],
            expected_grad_bias,
            tolerance=1e-4,
        )


def test_conv2d_no_bias_backward_shapes() raises:
    """Test that conv2d_no_bias_backward returns correct gradient shapes."""
    var batch = 1
    var in_channels = 2
    var in_height = 5
    var in_width = 5
    var out_channels = 3
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var output = conv2d_no_bias(x, kernel, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    var result = conv2d_no_bias_backward(
        grad_output, x, kernel, stride, padding
    )
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights

    # grad_input should match input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_kernel should match kernel shape
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)


def test_conv2d_backward_multichannel_shapes() raises:
    """Test conv2d_backward returns correct gradient shapes for multi-channel config.

    Tests in_channels=3, out_channels=8 — typical first conv layer configuration.
    Verifies grad_weights shape is (out_channels, in_channels, kH, kW) and
    grad_input shape is (batch, in_channels, H, W).
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var in_height = 6
    var in_width = 6
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 6, 6) with padding=1
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # grad_input should match input shape: (batch, in_channels, H, W)
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_weights should match kernel shape: (out_channels, in_channels, kH, kW)
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)

    # grad_bias should match bias shape: (out_channels,)
    assert_equal(grad_bias.shape()[0], out_channels)


def test_conv2d_backward_multichannel_values() raises:
    """Test conv2d_backward computes correct gradients for multi-channel config.

    Uses analytically tractable configuration (all ones, single spatial output)
    to verify gradient accumulation across input channels and output channels.

    Config: batch=1, in_channels=3, out_channels=8
    Input: (1, 3, 3, 3) all ones, kernel: (8, 3, 3, 3) all ones
    stride=1, padding=0 -> output shape: (1, 8, 1, 1)
    grad_output = ones((1, 8, 1, 1))

    Analytical expected values:
    - grad_weights[oc, ic, kh, kw] = sum over (batch, oh, ow) of
        grad_output[b, oc, oh, ow] * x[b, ic, oh+kh, ow+kw]
        = 1.0 * 1.0 = 1.0 for every weight position
    - grad_input[b, ic, ih, iw] = sum over oc of
        grad_output[b, oc, oh, ow] * kernel[oc, ic, kh, kw]
        = 8 output channels * 1.0 * 1.0 = 8.0 for every input position
    - grad_bias[oc] = sum over (batch, oh, ow) of grad_output[b, oc, oh, ow]
        = 1.0 (single spatial position, single batch item)
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    # Input: (1, 3, 3, 3) all ones
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = ones(input_shape, DType.float32)

    # Kernel: (8, 3, 3, 3) all ones
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 1, 1) - 3x3 input, 3x3 kernel, no padding
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # Verify grad_weights values: each weight gradient = 1.0
    # (single spatial output position, all ones input and grad_output)
    var n_weights = out_channels * in_channels * kH * kW
    var grad_weights_data = grad_kernel._data.bitcast[Float32]()
    for i in range(n_weights):
        assert_almost_equal(
            grad_weights_data[i],
            Float32(1.0),
            tolerance=1e-4,
        )

    # Verify grad_input values: each input gradient = 8.0
    # (8 output channels each contributing kernel=1.0 * grad_output=1.0)
    var n_inputs = batch * in_channels * 3 * 3
    var grad_input_data = grad_input._data.bitcast[Float32]()
    for i in range(n_inputs):
        assert_almost_equal(
            grad_input_data[i],
            Float32(out_channels),
            tolerance=1e-4,
        )

    # Verify grad_bias values: each bias gradient = 1.0
    # (1 batch * 1 spatial position * grad_output=1.0)
    var grad_bias_data = grad_bias._data.bitcast[Float32]()
    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias_data[oc],
            Float32(1.0),
            tolerance=1e-4,
        )


def test_conv2d_backward_multichannel_batch_gt_1() raises:
    """Test conv2d_backward with batch > 1 verifies grad accumulation across batch dimension.

    When batch > 1, grad_bias and grad_weights must accumulate over all batch items.
    grad_bias[oc] = sum over (batch, oh, ow) of grad_output[b, oc, oh, ow]
    grad_weights[oc, ic, kh, kw] = sum over (batch, oh, ow) of grad_output[b, oc, oh, ow] * x[b, ic, oh+kh, ow+kw]

    Config: batch=2, in_channels=3, out_channels=8
    Input: (2, 3, 3, 3) all ones, kernel: (8, 3, 3, 3) all ones
    stride=1, padding=0 -> output shape: (2, 8, 1, 1)
    grad_output = ones((2, 8, 1, 1))

    Expected values:
    - grad_weights[oc, ic, kh, kw] = batch * 1.0 * 1.0 = 2.0 for every weight position
      (2 batch items, each contributing 1.0)
    - grad_input[b, ic, ih, iw] = 8.0 (same as batch=1, doesn't depend on batch size)
    - grad_bias[oc] = batch * 1.0 = 2.0 (2 batch items, each contributing 1.0)
    """
    var batch = 2
    var in_channels = 3
    var out_channels = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    # Input: (2, 3, 3, 3) all ones
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = ones(input_shape, DType.float32)

    # Kernel: (8, 3, 3, 3) all ones
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (2, 8, 1, 1)
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # Verify grad_weights values: each weight gradient = 2.0
    # (2 batch items each contributing 1.0)
    var n_weights = out_channels * in_channels * kH * kW
    var grad_weights_data = grad_kernel._data.bitcast[Float32]()
    for i in range(n_weights):
        assert_almost_equal(
            grad_weights_data[i],
            Float32(2.0),
            tolerance=1e-4,
        )

    # Verify grad_input values: each input gradient = 8.0
    # (8 output channels, independent of batch size)
    var n_inputs = batch * in_channels * 3 * 3
    var grad_input_data = grad_input._data.bitcast[Float32]()
    for i in range(n_inputs):
        assert_almost_equal(
            grad_input_data[i],
            Float32(out_channels),
            tolerance=1e-4,
        )

    # Verify grad_bias values: each bias gradient = 2.0
    # (2 batch items, each with 1 spatial position)
    var grad_bias_data = grad_bias._data.bitcast[Float32]()
    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias_data[oc],
            Float32(2.0),
            tolerance=1e-4,
        )


def test_conv2d_backward_gradient_input() raises:
    """Numerical gradient check for grad_input computed by conv2d_backward.

    Compares analytical gradient w.r.t. input against finite differences.
    Uses a small (1, 1, 4, 4) input with a (1, 1, 3, 3) kernel so the
    test runs quickly while exercising the transposed convolution path.
    """
    var stride = 1
    var padding = 0

    # Input: (1, 1, 4, 4) with simple FP-representable values
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    for i in range(16):
        x_data[i] = Float32(i) * Float32(0.1)

    # Kernel: (1, 1, 3, 3) with simple FP-representable values
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    var k_data = kernel._data.bitcast[Float32]()
    for i in range(9):
        k_data[i] = Float32(i + 1) * Float32(0.5)

    # Forward pass to get output shape for grad_output
    var output = conv2d_no_bias(x, kernel, stride, padding)

    # Analytical gradient: use ones_like as grad_output (matches the sum reduction below)
    var grad_output = ones_like(output)
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var analytical_grad = result.grad_input

    # Numerical gradient: forward_fn sums conv output (equivalent to ones grad_output)
    def forward_for_input(inp: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(inp, kernel, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    var numerical_grad = compute_numerical_gradient(
        forward_for_input, x, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. input",
    )


def test_conv2d_backward_gradient_kernel() raises:
    """Numerical gradient check for grad_weights computed by conv2d_backward.

    Compares analytical gradient w.r.t. kernel against finite differences.
    Uses the same small dimensions as test_conv2d_backward_gradient_input.
    """
    var stride = 1
    var padding = 0

    # Input: (1, 1, 4, 4)
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    for i in range(16):
        x_data[i] = Float32(i) * Float32(0.1)

    # Kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    var k_data = kernel._data.bitcast[Float32]()
    for i in range(9):
        k_data[i] = Float32(i + 1) * Float32(0.5)

    # Forward pass to get output shape for grad_output
    var output = conv2d_no_bias(x, kernel, stride, padding)

    # Analytical gradient: use ones_like as grad_output (matches sum reduction below)
    var grad_output = ones_like(output)
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var analytical_grad = result.grad_weights

    # Numerical gradient: forward_fn sums conv output (equivalent to ones grad_output)
    def forward_for_kernel(k: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(x, k, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    var numerical_grad = compute_numerical_gradient(
        forward_for_kernel, kernel, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. kernel",
    )


def test_conv2d_backward_gradient_input_with_padding() raises:
    """Numerical gradient check for grad_input with padding > 0.

    Tests the transposed convolution path in conv2d_backward with boundary
    handling (padding=1 and padding=2). The grad_input computation is most
    complex when padding > 0 due to the padded transposed convolution.
    This verifies mathematical correctness of the padded backward path.

    Note: Input values are kept small (0.01 scale) to avoid Float32
    precision loss in the numerical gradient's reduce_sum chain. With
    larger values the output sum exceeds ~1000, and central-difference
    finite differences lose significant digits when subtracting two
    close Float32 sums.
    """
    var stride = 1

    # Input: (1, 1, 6, 6) with padding=1 or padding=2
    # Use 0.01 scale to keep output sum small (~120) for Float32 precision
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(6)
    input_shape.append(6)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    for i in range(36):
        x_data[i] = Float32(i) * Float32(0.01)

    # Kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    var k_data = kernel._data.bitcast[Float32]()
    for i in range(9):
        k_data[i] = Float32(i + 1) * Float32(0.5)

    # Test padding=1
    var padding = 1
    var output = conv2d_no_bias(x, kernel, stride, padding)
    var grad_output = ones_like(output)
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var analytical_grad = result.grad_input

    def forward_for_input_p1(inp: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(inp, kernel, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    var numerical_grad = compute_numerical_gradient(
        forward_for_input_p1, x, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. input (padding=1)",
    )

    # Test padding=2
    padding = 2
    output = conv2d_no_bias(x, kernel, stride, padding)
    grad_output = ones_like(output)
    result = conv2d_backward(grad_output, x, kernel, stride, padding)
    analytical_grad = result.grad_input

    def forward_for_input_p2(inp: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(inp, kernel, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    numerical_grad = compute_numerical_gradient(
        forward_for_input_p2, x, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. input (padding=2)",
    )


def test_conv2d_backward_gradient_kernel_with_padding() raises:
    """Numerical gradient check for grad_weights with padding > 0.

    Ensures kernel gradient computation is correct when input padding is used.
    Tests with padding=1 and padding=2 to exercise the padded forward path
    during weight gradient accumulation.

    Note: Input values use 0.01 scale to keep output sum small for Float32
    numerical gradient precision (see input gradient test docstring).
    """
    var stride = 1

    # Input: (1, 1, 6, 6) - use 0.01 scale for Float32 precision
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(6)
    input_shape.append(6)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    for i in range(36):
        x_data[i] = Float32(i) * Float32(0.01)

    # Kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    var k_data = kernel._data.bitcast[Float32]()
    for i in range(9):
        k_data[i] = Float32(i + 1) * Float32(0.5)

    # Test padding=1
    var padding = 1
    var output = conv2d_no_bias(x, kernel, stride, padding)
    var grad_output = ones_like(output)
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var analytical_grad = result.grad_weights

    def forward_for_kernel_p1(k: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(x, k, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    var numerical_grad = compute_numerical_gradient(
        forward_for_kernel_p1, kernel, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. kernel (padding=1)",
    )

    # Test padding=2
    padding = 2
    output = conv2d_no_bias(x, kernel, stride, padding)
    grad_output = ones_like(output)
    result = conv2d_backward(grad_output, x, kernel, stride, padding)
    analytical_grad = result.grad_weights

    def forward_for_kernel_p2(k: AnyTensor) raises -> AnyTensor:
        var out = conv2d_no_bias(x, k, stride, padding)
        var reduced = out
        while reduced.dim() > 0:
            reduced = reduce_sum(reduced, axis=0, keepdims=False)
        return reduced

    numerical_grad = compute_numerical_gradient(
        forward_for_kernel_p2, kernel, epsilon=3e-4
    )

    assert_gradients_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-2,
        atol=1e-4,
        message="conv2d_backward gradient w.r.t. kernel (padding=2)",
    )


def test_conv2d_forward_backward_consistency() raises:
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


def test_conv2d_batch_processing() raises:
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


def test_conv2d_5x5_kernel() raises:
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


def test_conv2d_backward_multichannel_padding1_values() raises:
    """Test conv2d_backward computes correct gradient values with padding=1.

    Verifies that border pixels receive fewer gradient contributions than interior
    pixels — a correctness property unique to padded convolutions.

    Config: batch=1, in_channels=3, out_channels=8, spatial=5x5, kH=kW=3,
    stride=1, padding=1 -> output shape (1, 8, 5, 5) (same-padding).

    With all-ones input, all-ones kernel, all-ones grad_output:
    grad_input[0, ic, ih, iw] = out_channels * overlap_count(ih, iw)

    Where overlap_count is the number of output positions (oh, ow) whose
    receptive field covers (ih, iw). With padding=1 and 5x5 spatial:
    - Input row/col 0 or 4: covered by 2 output positions in that dimension
    - Input row/col 1, 2, 3: covered by 3 output positions in that dimension

    Expected grad_input values (same for all in_channels):
    - Corner (0,0): 2*2 = 4 covering outputs -> 4 * 8 = 32.0
    - Edge non-corner (e.g., 0,1): 2*3 = 6 -> 6 * 8 = 48.0
    - Interior (e.g., 1,1): 3*3 = 9 -> 9 * 8 = 72.0
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var in_height = 5
    var in_width = 5
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Input: (1, 3, 5, 5) all ones
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    # Kernel: (8, 3, 3, 3) all ones
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 5, 5) — same-padding preserves spatial dims
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input

    # Verify grad_input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # Helper to index flat grad_input: shape (1, in_channels, 5, 5)
    # flat index = ic * 5*5 + ih * 5 + iw
    var grad_input_data = grad_input._data.bitcast[Float32]()

    # Corner pixel (0, 0): overlap = 2 * 2 = 4 -> expected 4 * 8 = 32.0
    var expected_corner = Float32(32.0)
    # Edge non-corner pixel (0, 1): overlap = 2 * 3 = 6 -> expected 6 * 8 = 48.0
    var expected_edge = Float32(48.0)
    # Interior pixel (1, 1): overlap = 3 * 3 = 9 -> expected 9 * 8 = 72.0
    var expected_interior = Float32(72.0)

    # Check all in_channels have the same pattern (symmetry: all-ones kernel)
    for ic in range(in_channels):
        var base = ic * in_height * in_width

        # Corner (ih=0, iw=0)
        assert_almost_equal(
            grad_input_data[base + 0 * in_width + 0],
            expected_corner,
            tolerance=1e-3,
        )

        # Edge non-corner (ih=0, iw=1)
        assert_almost_equal(
            grad_input_data[base + 0 * in_width + 1],
            expected_edge,
            tolerance=1e-3,
        )

        # Interior (ih=1, iw=1)
        assert_almost_equal(
            grad_input_data[base + 1 * in_width + 1],
            expected_interior,
            tolerance=1e-3,
        )

    # Verify border < interior: corner < edge < interior
    var corner_val = grad_input_data[0]
    var edge_val = grad_input_data[1]
    var interior_val = grad_input_data[1 * in_width + 1]
    assert_true(corner_val < edge_val)
    assert_true(edge_val < interior_val)


def main() raises:
    """Run all test_conv tests."""
    print("Running test_conv tests...")

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

    test_conv2d_backward_shapes()
    print("✓ test_conv2d_backward_shapes")

    test_conv2d_backward_bias_gradient()
    print("✓ test_conv2d_backward_bias_gradient")

    test_conv2d_no_bias_backward_shapes()
    print("✓ test_conv2d_no_bias_backward_shapes")

    test_conv2d_backward_multichannel_shapes()
    print("✓ test_conv2d_backward_multichannel_shapes")

    test_conv2d_backward_multichannel_values()
    print("✓ test_conv2d_backward_multichannel_values")

    test_conv2d_backward_multichannel_batch_gt_1()
    print("✓ test_conv2d_backward_multichannel_batch_gt_1")

    test_conv2d_backward_gradient_input()
    print("✓ test_conv2d_backward_gradient_input")

    test_conv2d_backward_gradient_kernel()
    print("✓ test_conv2d_backward_gradient_kernel")

    test_conv2d_backward_gradient_input_with_padding()
    print("✓ test_conv2d_backward_gradient_input_with_padding")

    test_conv2d_backward_gradient_kernel_with_padding()
    print("✓ test_conv2d_backward_gradient_kernel_with_padding")

    test_conv2d_forward_backward_consistency()
    print("✓ test_conv2d_forward_backward_consistency")

    test_conv2d_batch_processing()
    print("✓ test_conv2d_batch_processing")

    test_conv2d_5x5_kernel()
    print("✓ test_conv2d_5x5_kernel")

    test_conv2d_backward_multichannel_padding1_values()
    print("✓ test_conv2d_backward_multichannel_padding1_values")

    print("\nAll test_conv tests passed!")
