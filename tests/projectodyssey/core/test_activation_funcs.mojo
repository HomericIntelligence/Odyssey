"""Unit tests for activation functions.

Tests cover:
- relu: Forward and backward passes
- sigmoid: Forward and backward passes with numerical stability
- leaky_relu: Parametric activation function

All tests use pure functional API - no internal state.

"""


from tests.projectodyssey.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, full
from projectodyssey.core.activation import (
    leaky_relu,
    leaky_relu_backward,
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    softmax,
    softmax_backward,
    tanh,
    tanh_backward,
)


def test_relu_positive_values() raises:
    """Test ReLU preserves positive values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = ones(input_shape, DType.float32)

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_almost_equal(output_data[i], 1.0, tolerance=1e-5)


def test_relu_negative_values() raises:
    """Test ReLU zeros out negative values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(5):
        input_data[i] = -Float32(i + 1)

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_almost_equal(output_data[i], 0.0, tolerance=1e-5)


def test_relu_mixed_values() raises:
    """Test ReLU on mixed positive and negative values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0  # Should become 0
    input_data[1] = -1.0  # Should become 0
    input_data[2] = 0.0  # Should become 0
    input_data[3] = 1.0  # Should stay 1
    input_data[4] = 2.0  # Should stay 2

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 2.0, tolerance=1e-5)


def test_relu_backward() raises:
    """Test ReLU backward pass computes correct gradients."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0  # Negative
    input_data[1] = -1.0  # Negative
    input_data[2] = 0.0  # Zero
    input_data[3] = 1.0  # Positive
    input_data[4] = 2.0  # Positive

    var grad_output_shape = List[Int]()
    grad_output_shape.append(5)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = relu_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    # Negative inputs -> grad is 0
    assert_almost_equal(grad_data[0], 0.0, tolerance=1e-5)
    assert_almost_equal(grad_data[1], 0.0, tolerance=1e-5)
    assert_almost_equal(grad_data[2], 0.0, tolerance=1e-5)
    # Positive inputs -> grad passes through
    assert_almost_equal(grad_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(grad_data[4], 1.0, tolerance=1e-5)


def test_leaky_relu_forward() raises:
    """Test Leaky ReLU preserves negative values with slope."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 2.0

    var output = leaky_relu(input, alpha=0.1)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], -0.2, tolerance=1e-5)  # -2 * 0.1
    assert_almost_equal(output_data[1], -0.1, tolerance=1e-5)  # -1 * 0.1
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 2.0, tolerance=1e-5)


def test_sigmoid_range() raises:
    """Test sigmoid output is in [0, 1] range."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -10.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 10.0

    var output = sigmoid(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_greater_or_equal(output_data[i], 0.0, "Sigmoid lower bound")
        assert_less_or_equal(output_data[i], 1.0, "Sigmoid upper bound")


def test_sigmoid_at_zero() raises:
    """Test sigmoid(0) = 0.5."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var input = zeros(input_shape, DType.float32)

    var output = sigmoid(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.5, tolerance=1e-4)


def test_sigmoid_symmetry() raises:
    """Test sigmoid(-x) + sigmoid(x) = 1."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 2.0

    var neg_x = zeros(input_shape, DType.float32)
    var neg_x_data = neg_x._data.bitcast[Float32]()
    neg_x_data[0] = -2.0

    var sig_x = sigmoid(x)
    var sig_neg_x = sigmoid(neg_x)

    var sig_x_data = sig_x._data.bitcast[Float32]()
    var sig_neg_x_data = sig_neg_x._data.bitcast[Float32]()

    var sum_val = sig_x_data[0] + sig_neg_x_data[0]
    assert_almost_equal(sum_val, 1.0, tolerance=1e-4)


def test_sigmoid_backward() raises:
    """Test sigmoid backward pass computes correct gradients."""
    var output_shape = List[Int]()
    output_shape.append(1)
    var output = full(output_shape, 0.5, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = sigmoid_backward(grad_output, output)

    # For sigmoid output = 0.5, gradient = 1 * 0.5 * (1 - 0.5) = 0.25
    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.25, tolerance=1e-4)


def test_tanh_range() raises:
    """Test tanh output is in [-1, 1] range."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -10.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 10.0

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_greater_or_equal(output_data[i], -1.0, "Tanh lower bound")
        assert_less_or_equal(output_data[i], 1.0, "Tanh upper bound")


def test_tanh_at_zero() raises:
    """Test tanh(0) = 0."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var input = zeros(input_shape, DType.float32)

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)


def test_tanh_antisymmetry() raises:
    """Test tanh(-x) = -tanh(x)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 2.0

    var neg_x = zeros(input_shape, DType.float32)
    var neg_x_data = neg_x._data.bitcast[Float32]()
    neg_x_data[0] = -2.0

    var tanh_x = tanh(x)
    var tanh_neg_x = tanh(neg_x)

    var tanh_x_data = tanh_x._data.bitcast[Float32]()
    var tanh_neg_x_data = tanh_neg_x._data.bitcast[Float32]()

    assert_almost_equal(tanh_x_data[0], -tanh_neg_x_data[0], tolerance=1e-4)


def test_tanh_backward() raises:
    """Test tanh backward pass computes correct gradients."""
    var output_shape = List[Int]()
    output_shape.append(1)
    var output = zeros(output_shape, DType.float32)
    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.5

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = tanh_backward(grad_output, output)

    # For tanh output = 0.5, gradient = 1 * (1 - 0.5^2) = 0.75
    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.75, tolerance=1e-4)


def test_softmax_output_sum() raises:
    """Test softmax outputs sum to 1 along last axis."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    var sum_val = output_data[0] + output_data[1] + output_data[2]
    assert_almost_equal(sum_val, 1.0, tolerance=1e-5)


def test_softmax_positive_outputs() raises:
    """Test softmax outputs are all positive."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -5.0
    input_data[1] = 0.0
    input_data[2] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(3):
        assert_greater_or_equal(output_data[i], 0.0, "Softmax positive")


def test_softmax_uniform() raises:
    """Test softmax on uniform input gives uniform output."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(4):
        input_data[i] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(output_data[i], 0.25, tolerance=1e-5)


def test_softmax_backward() raises:
    """Test softmax backward pass."""
    var output_shape = List[Int]()
    output_shape.append(1)
    output_shape.append(3)
    var output = zeros(output_shape, DType.float32)

    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.1
    output_data[1] = 0.6
    output_data[2] = 0.3

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = softmax_backward(grad_output, output, axis=-1)

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1)
    assert_equal(grad_shape[1], 3)


def test_softmax_axis_0() raises:
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


def test_softmax_axis_1() raises:
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


def test_softmax_axis_negative_indexing() raises:
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


def test_softmax_3d_axis_middle() raises:
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


def test_activation_output_shape_preservation() raises:
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


def test_activation_dtype_preservation() raises:
    """Test that activations preserve input dtype."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = ones(input_shape, DType.float32)

    var relu_out = relu(input)
    var sig_out = sigmoid(input)

    assert_true(relu_out.dtype() == DType.float32)
    assert_true(sig_out.dtype() == DType.float32)


def test_relu_gradient_mask() raises:
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


def main() raises:
    """Run all test_activation_funcs tests."""
    print("Running test_activation_funcs tests...")

    test_relu_positive_values()
    print("✓ test_relu_positive_values")

    test_relu_negative_values()
    print("✓ test_relu_negative_values")

    test_relu_mixed_values()
    print("✓ test_relu_mixed_values")

    test_relu_backward()
    print("✓ test_relu_backward")

    test_leaky_relu_forward()
    print("✓ test_leaky_relu_forward")

    test_sigmoid_range()
    print("✓ test_sigmoid_range")

    test_sigmoid_at_zero()
    print("✓ test_sigmoid_at_zero")

    test_sigmoid_symmetry()
    print("✓ test_sigmoid_symmetry")

    test_sigmoid_backward()
    print("✓ test_sigmoid_backward")

    test_tanh_range()
    print("✓ test_tanh_range")

    test_tanh_at_zero()
    print("✓ test_tanh_at_zero")

    test_tanh_antisymmetry()
    print("✓ test_tanh_antisymmetry")

    test_tanh_backward()
    print("✓ test_tanh_backward")

    test_softmax_output_sum()
    print("✓ test_softmax_output_sum")

    test_softmax_positive_outputs()
    print("✓ test_softmax_positive_outputs")

    test_softmax_uniform()
    print("✓ test_softmax_uniform")

    test_softmax_backward()
    print("✓ test_softmax_backward")

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

    print("\nAll test_activation_funcs tests passed!")
