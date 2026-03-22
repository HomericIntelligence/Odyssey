# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_pooling.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for pooling layer operations - Part 1.

Tests cover:
- maxpool2d: Forward pass with various kernel sizes and strides
- maxpool2d_backward: Backward pass for max pooling
- avgpool2d: Forward pass (basic tests)

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
    global_avgpool2d,
    global_avgpool2d_backward,
)


# ============================================================================
# MaxPool2D Forward Tests
# ============================================================================


fn test_maxpool2d_output_shape() raises:
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


fn test_maxpool2d_basic_4x4() raises:
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


fn test_maxpool2d_stride_1() raises:
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


fn test_maxpool2d_batch_processing() raises:
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


fn test_maxpool2d_backward_output_shape() raises:
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


# ============================================================================
# AvgPool2D Forward Tests (basic)
# ============================================================================


fn test_avgpool2d_output_shape() raises:
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


fn test_avgpool2d_basic_4x4() raises:
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


fn test_avgpool2d_all_ones() raises:
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


fn main() raises:
    """Run pooling tests part 1."""
    print("Running pooling tests part 1...")

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

    print("\nAll pooling tests part 1 passed!")
