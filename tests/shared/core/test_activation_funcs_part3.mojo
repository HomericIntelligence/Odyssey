"""Unit tests for activation functions - Part 3: Softmax (axis) and Integration.

Tests cover:
- softmax: Axis-specific and multi-dimensional tests
- Integration: Shape/dtype preservation and gradient masking

All tests use pure functional API - no internal state.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activation_funcs.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    tanh,
    softmax,
)


# ============================================================================
# Softmax Tests (Axis-specific)
# ============================================================================


fn test_softmax_axis_0() raises:
    """Test softmax along first axis (axis=0)."""
    var input_shape = List[Int]()
    input_shape.append(3)
    input_shape.append(2)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    # Shape: (3, 2)
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]
    input_data[0] = 1.0  # [0, 0]
    input_data[1] = 2.0  # [0, 1]
    input_data[2] = 3.0  # [1, 0]
    input_data[3] = 4.0  # [1, 1]
    input_data[4] = 5.0  # [2, 0]
    input_data[5] = 6.0  # [2, 1]

    var output = softmax(input, axis=0)

    var output_data = output._data.bitcast[Float32]()
    # For each column, softmax should sum to 1
    # Column 0: softmax([1, 3, 5])
    # Column 1: softmax([2, 4, 6])
    var col0_sum = output_data[0] + output_data[2] + output_data[4]
    var col1_sum = output_data[1] + output_data[3] + output_data[5]
    assert_almost_equal(col0_sum, 1.0, tolerance=1e-5)
    assert_almost_equal(col1_sum, 1.0, tolerance=1e-5)


fn test_softmax_axis_1() raises:
    """Test softmax along second axis (axis=1)."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    # Shape: (2, 3)
    # [[1, 2, 3],
    #  [4, 5, 6]]
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0
    input_data[3] = 4.0
    input_data[4] = 5.0
    input_data[5] = 6.0

    var output = softmax(input, axis=1)

    var output_data = output._data.bitcast[Float32]()
    # For each row, softmax should sum to 1
    var row0_sum = output_data[0] + output_data[1] + output_data[2]
    var row1_sum = output_data[3] + output_data[4] + output_data[5]
    assert_almost_equal(row0_sum, 1.0, tolerance=1e-5)
    assert_almost_equal(row1_sum, 1.0, tolerance=1e-5)


fn test_softmax_axis_negative_indexing() raises:
    """Test softmax with negative axis indexing."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    # Test axis=-1 (last axis)
    var output_neg1 = softmax(input, axis=-1)
    var output_data_neg1 = output_neg1._data.bitcast[Float32]()

    # For axis=-1, each group of 4 elements should sum to 1
    # Since input is all ones, each softmax output should be 0.25
    for i in range(2 * 3):
        var sum_val = Float32(0.0)
        for j in range(4):
            sum_val += output_data_neg1[i * 4 + j]
        assert_almost_equal(sum_val, 1.0, tolerance=1e-5)

    # Test axis=-2 (second-to-last axis) on 2D tensor
    var input_shape_2d = List[Int]()
    input_shape_2d.append(3)
    input_shape_2d.append(2)
    var input_2d = ones(input_shape_2d, DType.float32)
    var output_neg2 = softmax(input_2d, axis=-2)
    var output_data_neg2 = output_neg2._data.bitcast[Float32]()

    # axis=-2 on 2D is equivalent to axis=0
    for j in range(2):
        var sum_val = Float32(0.0)
        for i in range(3):
            sum_val += output_data_neg2[i * 2 + j]
        assert_almost_equal(sum_val, 1.0, tolerance=1e-5)


fn test_softmax_3d_axis_middle() raises:
    """Test softmax on 3D tensor along middle axis."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    # Fill with sequential values
    var input_data = input._data.bitcast[Float32]()
    for i in range(24):
        input_data[i] = Float32(i + 1)

    var output = softmax(input, axis=1)

    var output_data = output._data.bitcast[Float32]()
    # For axis=1 on shape (2, 3, 4):
    # For each (i, k) pair, sum over j should be 1
    for i in range(2):
        for k in range(4):
            var sum_val = Float32(0.0)
            for j in range(3):
                var idx = i * (3 * 4) + j * 4 + k
                sum_val += output_data[idx]
            assert_almost_equal(sum_val, 1.0, tolerance=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_activation_output_shape_preservation() raises:
    """Test that activations preserve input shape."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)

    var relu_out = relu(input)
    var sig_out = sigmoid(input)
    var tanh_out = tanh(input)

    var relu_shape = relu_out.shape()
    var sig_shape = sig_out.shape()
    var tanh_shape = tanh_out.shape()

    assert_equal(relu_shape[0], 2)
    assert_equal(relu_shape[1], 3)
    assert_equal(sig_shape[0], 2)
    assert_equal(sig_shape[1], 3)
    assert_equal(tanh_shape[0], 2)
    assert_equal(tanh_shape[1], 3)


fn test_activation_dtype_preservation() raises:
    """Test that activations preserve input dtype."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = ones(input_shape, DType.float32)

    var relu_out = relu(input)
    var sig_out = sigmoid(input)

    assert_true(relu_out.dtype() == DType.float32)
    assert_true(sig_out.dtype() == DType.float32)


fn test_relu_gradient_mask() raises:
    """Test that ReLU gradient acts as a mask for positive values."""
    var input_shape = List[Int]()
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 2.0
    input_data[2] = -3.0
    input_data[3] = 4.0

    var grad_output_shape = List[Int]()
    grad_output_shape.append(4)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = relu_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.0, tolerance=1e-5)  # Masked
    assert_almost_equal(grad_data[1], 1.0, tolerance=1e-5)  # Passed through
    assert_almost_equal(grad_data[2], 0.0, tolerance=1e-5)  # Masked
    assert_almost_equal(grad_data[3], 1.0, tolerance=1e-5)  # Passed through


fn main() raises:
    """Run activation tests - Part 3: Softmax (axis) and Integration."""
    print("Running activation tests - Part 3: Softmax (axis) and Integration...")

    test_softmax_axis_0()
    print("✓ test_softmax_axis_0")

    test_softmax_axis_1()
    print("✓ test_softmax_axis_1")

    test_softmax_axis_negative_indexing()
    print("✓ test_softmax_axis_negative_indexing")

    test_softmax_3d_axis_middle()
    print("✓ test_softmax_3d_axis_middle")

    test_activation_output_shape_preservation()
    print("✓ test_activation_output_shape_preservation")

    test_activation_dtype_preservation()
    print("✓ test_activation_dtype_preservation")

    test_relu_gradient_mask()
    print("✓ test_relu_gradient_mask")

    print("\nAll Part 3 activation tests passed!")
