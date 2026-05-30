"""Unit tests for pooling layer operations.

Tests cover:
- maxpool2d: Forward pass with various kernel sizes and strides
- maxpool2d_backward: Backward pass for max pooling
- avgpool2d: Forward pass (basic tests)

All tests use pure functional API - no internal state.
"""


from tests.projectodyssey.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full, ones, zeros
from projectodyssey.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
    global_avgpool2d,
    global_avgpool2d_backward,
)


def test_maxpool2d_output_shape() raises:
    """Test maxpool2d output shape computation.

    Formula: out_height = (in_height - kernel_size) / stride + 1

    Test case: 4x4 input, 2x2 kernel, stride 2, padding 0
    Expected: 2x2 output.
    """
    var input_shape = List[Int]()
    input_shape.append(1)  # batch
    input_shape.append(1)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width
    var input = ones(input_shape, DType.float32)

    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[0], 1, "Batch size mismatch")
    assert_equal(out_shape[1], 1, "Channels mismatch")
    assert_equal(out_shape[2], 2, "Output height should be 2")
    assert_equal(out_shape[3], 2, "Output width should be 2")


def test_maxpool2d_basic_4x4() raises:
    """Test basic max pooling 2x2 kernel on 4x4 input."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(16):
        input_data[i] = Float32(i + 1)

    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 8.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 14.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 16.0, tolerance=1e-5)


def test_maxpool2d_stride_1() raises:
    """Test maxpool2d with stride=1 (overlapping windows)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(3)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(9):
        input_data[i] = Float32(i + 1)

    var output = maxpool2d(input, kernel_size=2, stride=1, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[2], 2, "Output height should be 2")
    assert_equal(out_shape[3], 2, "Output width should be 2")

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 5.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 8.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 9.0, tolerance=1e-5)


def test_maxpool2d_batch_processing() raises:
    """Test maxpool2d processes multiple samples in batch correctly."""
    var input_shape = List[Int]()
    input_shape.append(2)  # batch_size
    input_shape.append(1)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width
    var input = ones(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(16):
        input_data[i] = 1.0

    for i in range(16, 32):
        input_data[i] = 2.0

    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[0], 2, "Batch size should be preserved")
    assert_equal(out_shape[2], 2, "Output height should be 2")
    assert_equal(out_shape[3], 2, "Output width should be 2")

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 2.0, tolerance=1e-5)


def test_maxpool2d_backward_output_shape() raises:
    """Test maxpool2d_backward produces correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(2)
    grad_output_shape.append(2)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = maxpool2d_backward(
        grad_output, input, kernel_size=2, stride=2, padding=0
    )

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1, "Batch size mismatch")
    assert_equal(grad_shape[1], 1, "Channels mismatch")
    assert_equal(grad_shape[2], 4, "Gradient height mismatch")
    assert_equal(grad_shape[3], 4, "Gradient width mismatch")


def test_avgpool2d_output_shape() raises:
    """Test avgpool2d output shape computation."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)


def test_avgpool2d_basic_4x4() raises:
    """Test basic average pooling 2x2 kernel on 4x4 input."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(16):
        input_data[i] = Float32(i + 1)

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 3.5, tolerance=1e-5)
    assert_almost_equal(output_data[1], 5.5, tolerance=1e-5)
    assert_almost_equal(output_data[2], 11.5, tolerance=1e-5)
    assert_almost_equal(output_data[3], 13.5, tolerance=1e-5)


def test_avgpool2d_all_ones() raises:
    """Test avgpool2d with all-ones input."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var output_data = output._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(output_data[i], 1.0, tolerance=1e-5)


def test_avgpool2d_stride_1() raises:
    """Test avgpool2d with stride=1 (overlapping windows)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(3)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(9):
        input_data[i] = Float32(i + 1)

    var output = avgpool2d(input, kernel_size=2, stride=1, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 3.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 4.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 6.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 7.0, tolerance=1e-5)


def test_avgpool2d_batch_processing() raises:
    """Test avgpool2d processes batches correctly."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(16):
        input_data[i] = 2.0

    for i in range(16, 32):
        input_data[i] = 4.0

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var out_shape = output.shape()
    assert_equal(out_shape[0], 2)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 2.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 4.0, tolerance=1e-5)


def test_global_avgpool2d_basic() raises:
    """Test global average pooling reduces spatial dimensions to 1x1."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = global_avgpool2d(input)

    var out_shape = output.shape()
    assert_equal(out_shape[0], 2, "Batch size preserved")
    assert_equal(out_shape[1], 3, "Channels preserved")
    assert_equal(out_shape[2], 1, "Height reduced to 1")
    assert_equal(out_shape[3], 1, "Width reduced to 1")


def test_pooling_dtype_preservation() raises:
    """Test that pooling preserves input dtype."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output_max = maxpool2d(input, kernel_size=2, stride=2, padding=0)
    var output_avg = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    assert_true(output_max.dtype() == DType.float32, "MaxPool dtype preserved")
    assert_true(output_avg.dtype() == DType.float32, "AvgPool dtype preserved")


def test_global_avgpool2d_backward_output_shape() raises:
    """Test global_avgpool2d_backward produces correct gradient shape.

    Forward: (B, C, H, W) -> (B, C, 1, 1)
    Backward: (B, C, 1, 1) -> (B, C, H, W).
    """
    var input_shape = List[Int]()
    input_shape.append(2)  # batch
    input_shape.append(3)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width
    var input = ones(input_shape, DType.float32)

    var output = global_avgpool2d(input)

    var grad_output_shape = output.shape()
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 2, "Batch size mismatch")
    assert_equal(grad_shape[1], 3, "Channels mismatch")
    assert_equal(grad_shape[2], 4, "Gradient height mismatch")
    assert_equal(grad_shape[3], 4, "Gradient width mismatch")


def test_global_avgpool2d_backward_uniform_distribution() raises:
    """Test global_avgpool2d_backward distributes gradient uniformly.

    For input of shape (1, 1, 4, 4) and grad_output of 1.0,
    each position in grad_input should receive 1.0 / (4 * 4) = 1.0 / 16
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = global_avgpool2d(input)

    var grad_output_shape = output.shape()
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    var expected_value = Float32(1.0 / 16.0)

    for i in range(16):
        assert_almost_equal(grad_data[i], expected_value, Float32(1e-6))


def test_global_avgpool2d_backward_batch_independence() raises:
    """Test that gradients for different batch samples are independent.

    Two batch samples with grad_output values [2.0, 3.0] should produce
    different gradients for each batch element.
    """
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var input = ones(input_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(2)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, DType.float32)

    var grad_out_data = grad_output._data.bitcast[Float32]()
    grad_out_data[0] = 2.0  # First batch
    grad_out_data[1] = 3.0  # Second batch

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    var spatial_size = 4  # 2 * 2

    # First batch: 2.0 / 4 = 0.5
    var expected_batch0 = 2.0 / Float32(spatial_size)
    for i in range(4):
        assert_almost_equal(grad_data[i], expected_batch0, tolerance=1e-6)

    # Second batch: 3.0 / 4 = 0.75
    var expected_batch1 = 3.0 / Float32(spatial_size)
    for i in range(4, 8):
        assert_almost_equal(grad_data[i], expected_batch1, tolerance=1e-6)


def test_global_avgpool2d_backward_channel_independence() raises:
    """Test that gradients for different channels are independent.

    Multiple channels should each receive their own gradients distributed
    uniformly across spatial positions.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    input_shape.append(2)
    var input = ones(input_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(2)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, DType.float32)

    var grad_out_data = grad_output._data.bitcast[Float32]()
    grad_out_data[0] = 2.0  # Channel 0
    grad_out_data[1] = 4.0  # Channel 1

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    var spatial_size = 4  # 2 * 2

    # Channel 0: 2.0 / 4 = 0.5
    var expected_ch0 = 2.0 / Float32(spatial_size)
    for i in range(4):
        assert_almost_equal(grad_data[i], expected_ch0, tolerance=1e-6)

    # Channel 1: 4.0 / 4 = 1.0
    var expected_ch1 = 4.0 / Float32(spatial_size)
    for i in range(4, 8):
        assert_almost_equal(grad_data[i], expected_ch1, tolerance=1e-6)


def test_global_avgpool2d_backward_zero_gradient() raises:
    """Test that zero gradient produces zero output."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(3)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    for i in range(9):
        assert_almost_equal(grad_data[i], 0.0, tolerance=1e-6)


def test_global_avgpool2d_backward_forward_backward_consistency() raises:
    """Test consistency between forward and backward passes.

    Forward: x -> global_avgpool2d -> y
    Backward: grad_y -> global_avgpool2d_backward -> grad_x

    The sum of grad_x should equal grad_y (gradient conservation).
    """
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(2)
    input_shape.append(2)
    var input = full(input_shape, 2.0, DType.float32)

    var output = global_avgpool2d(input)

    var grad_output_shape = output.shape()
    var grad_output = full(grad_output_shape, 1.0, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()

    # Each spatial position should get 1.0 / (2*2) = 0.25
    var expected_per_position = Float32(1.0 / 4.0)
    var total_elements = 2 * 3 * 2 * 2

    for i in range(total_elements):
        assert_almost_equal(grad_data[i], expected_per_position, Float32(1e-6))


def test_avgpool2d_backward_output_shape() raises:
    """Test avgpool2d_backward produces correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var grad_output_shape = output.shape()
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = avgpool2d_backward(
        grad_output, input, kernel_size=2, stride=2, padding=0
    )

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1, "Batch size mismatch")
    assert_equal(grad_shape[1], 1, "Channels mismatch")
    assert_equal(grad_shape[2], 4, "Gradient height mismatch")
    assert_equal(grad_shape[3], 4, "Gradient width mismatch")


def main() raises:
    """Run all test_pooling tests."""
    print("Running test_pooling tests...")

    test_maxpool2d_output_shape()
    print("✓ test_maxpool2d_output_shape")

    test_maxpool2d_basic_4x4()
    print("✓ test_maxpool2d_basic_4x4")

    test_maxpool2d_stride_1()
    print("✓ test_maxpool2d_stride_1")

    test_maxpool2d_batch_processing()
    print("✓ test_maxpool2d_batch_processing")

    test_maxpool2d_backward_output_shape()
    print("✓ test_maxpool2d_backward_output_shape")

    test_avgpool2d_output_shape()
    print("✓ test_avgpool2d_output_shape")

    test_avgpool2d_basic_4x4()
    print("✓ test_avgpool2d_basic_4x4")

    test_avgpool2d_all_ones()
    print("✓ test_avgpool2d_all_ones")

    test_avgpool2d_stride_1()
    print("✓ test_avgpool2d_stride_1")

    test_avgpool2d_batch_processing()
    print("✓ test_avgpool2d_batch_processing")

    test_global_avgpool2d_basic()
    print("✓ test_global_avgpool2d_basic")

    test_pooling_dtype_preservation()
    print("✓ test_pooling_dtype_preservation")

    test_global_avgpool2d_backward_output_shape()
    print("✓ test_global_avgpool2d_backward_output_shape")

    test_global_avgpool2d_backward_uniform_distribution()
    print("✓ test_global_avgpool2d_backward_uniform_distribution")

    test_global_avgpool2d_backward_batch_independence()
    print("✓ test_global_avgpool2d_backward_batch_independence")

    test_global_avgpool2d_backward_channel_independence()
    print("✓ test_global_avgpool2d_backward_channel_independence")

    test_global_avgpool2d_backward_zero_gradient()
    print("✓ test_global_avgpool2d_backward_zero_gradient")

    test_global_avgpool2d_backward_forward_backward_consistency()
    print("✓ test_global_avgpool2d_backward_forward_backward_consistency")

    test_avgpool2d_backward_output_shape()
    print("✓ test_avgpool2d_backward_output_shape")

    print("\nAll test_pooling tests passed!")
