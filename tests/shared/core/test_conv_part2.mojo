"""Unit tests for 2D convolution layer - Part 2: Numerical Correctness and Batch Tests.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_conv.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- conv2d: Numerical correctness verification
- conv2d_no_bias: Forward pass without bias
- Bias term addition
- Simple known-value computations
- Batch processing
- 5x5 kernel support
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
# Conv2D Forward Pass Tests - Numerical Correctness
# ============================================================================


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
    """Run Conv2D Part 2 tests: numerical correctness and batch tests."""
    print("Running Conv2D Part 2 tests (numerical correctness and batching)...")

    test_conv2d_single_sample_simple()
    print("✓ test_conv2d_single_sample_simple")

    test_conv2d_with_bias()
    print("✓ test_conv2d_with_bias")

    test_conv2d_numerical_correctness()
    print("✓ test_conv2d_numerical_correctness")

    test_conv2d_no_bias()
    print("✓ test_conv2d_no_bias")

    test_conv2d_batched()
    print("✓ test_conv2d_batched")

    test_conv2d_batch_processing()
    print("✓ test_conv2d_batch_processing")

    test_conv2d_5x5_kernel()
    print("✓ test_conv2d_5x5_kernel")

    print("\nAll Conv2D Part 2 tests passed!")
