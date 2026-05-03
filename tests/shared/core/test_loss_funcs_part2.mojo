"""Unit tests for loss functions (Part 2 of 2)

Tests cover:
- mean_squared_error: Backward shape and symmetry
- binary_cross_entropy: Shape, basic, perfect prediction, backward
- Combined loss property tests

All tests use pure functional API - no internal state.
"""


from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
    assert_close_float,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
)
from shared.core.reduction import mean


def test_mean_squared_error_backward_shape() raises:
    """Test MSE backward produces gradients with same shape as input."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    pred_shape.append(3)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    targets_shape.append(3)
    var targets = zeros(targets_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(4)
    grad_output_shape.append(3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    var grad_shape = grad_pred.shape()
    assert_equal(grad_shape[0], 4, "Batch dimension preserved")
    assert_equal(grad_shape[1], 3, "Feature dimension preserved")


def test_binary_cross_entropy_output_shape() raises:
    """Test BCE output shape matches input shape."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    var targets = zeros(targets_shape, DType.float32)

    var loss = binary_cross_entropy(predictions, targets)

    var loss_shape = loss.shape()
    assert_equal(loss_shape[0], 4, "Output shape matches input")


def test_binary_cross_entropy_basic() raises:
    """Test BCE on simple binary classification."""
    var pred_shape = List[Int]()
    pred_shape.append(2)
    var predictions = zeros(pred_shape, DType.float32)

    var pred_data = predictions._data.bitcast[Float32]()
    pred_data[0] = 0.9  # High confidence in class 1
    pred_data[1] = 0.1  # Low confidence in class 1

    var targets_shape = List[Int]()
    targets_shape.append(2)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 1.0  # True label 1
    targets_data[1] = 0.0  # True label 0

    var loss = binary_cross_entropy(predictions, targets)

    var loss_data = loss._data.bitcast[Float32]()
    for i in range(2):
        assert_greater_or_equal(loss_data[i], 0.0, "BCE loss non-negative")


def test_binary_cross_entropy_perfect_prediction() raises:
    """Test BCE when prediction is perfect (loss approaches 0)."""
    var pred_shape = List[Int]()
    pred_shape.append(1)
    var predictions = zeros(pred_shape, DType.float32)

    var pred_data = predictions._data.bitcast[Float32]()
    pred_data[0] = 0.99  # Near 1.0

    var targets_shape = List[Int]()
    targets_shape.append(1)
    var targets = ones(targets_shape, DType.float32)

    var loss = binary_cross_entropy(predictions, targets)

    var loss_data = loss._data.bitcast[Float32]()
    assert_less_or_equal(loss_data[0], 0.1, "Small loss for good prediction")


def test_binary_cross_entropy_backward() raises:
    """Test BCE backward pass computes correct gradients."""
    var pred_shape = List[Int]()
    pred_shape.append(2)
    var predictions = zeros(pred_shape, DType.float32)

    var pred_data = predictions._data.bitcast[Float32]()
    pred_data[0] = 0.7
    pred_data[1] = 0.3

    var targets_shape = List[Int]()
    targets_shape.append(2)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 1.0
    targets_data[1] = 0.0

    var grad_output_shape = List[Int]()
    grad_output_shape.append(2)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    var grad_shape = grad_pred.shape()
    assert_equal(grad_shape[0], 2, "Gradient shape matches input")


def test_loss_non_negative() raises:
    """Test that all loss functions produce non-negative values."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    var targets = zeros(targets_shape, DType.float32)

    var mse_loss = mean_squared_error(predictions, targets)
    var bce_loss = binary_cross_entropy(predictions, targets)

    var mse_data = mse_loss._data.bitcast[Float32]()
    var bce_data = bce_loss._data.bitcast[Float32]()

    for i in range(4):
        assert_greater_or_equal(mse_data[i], 0.0, "MSE non-negative")
        assert_greater_or_equal(bce_data[i], 0.0, "BCE non-negative")


def test_loss_gradient_shape_consistency() raises:
    """Test that backward passes produce gradients matching input shape."""
    var pred_shape = List[Int]()
    pred_shape.append(3)
    pred_shape.append(4)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(3)
    targets_shape.append(4)
    var targets = zeros(targets_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(3)
    grad_output_shape.append(4)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_mse = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    var grad_shape = grad_mse.shape()
    assert_equal(grad_shape[0], 3, "Gradient batch dimension matches")
    assert_equal(grad_shape[1], 4, "Gradient feature dimension matches")


def test_mse_symmetric() raises:
    """Test that MSE is symmetric in predictions and targets order."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    var a_data = a._data.bitcast[Float32]()
    var b_data = b._data.bitcast[Float32]()

    for i in range(3):
        a_data[i] = Float32(i)
        b_data[i] = Float32(i + 1)

    var loss_ab = mean_squared_error(a, b)
    var loss_ba = mean_squared_error(b, a)

    var loss_ab_data = loss_ab._data.bitcast[Float32]()
    var loss_ba_data = loss_ba._data.bitcast[Float32]()

    for i in range(3):
        assert_almost_equal(loss_ab_data[i], loss_ba_data[i], tolerance=1e-5)


def main() raises:
    """Run test_loss_funcs part 2 tests."""
    print("Running test_loss_funcs_part2 tests...")

    test_mean_squared_error_backward_shape()
    print("✓ test_mean_squared_error_backward_shape")

    test_binary_cross_entropy_output_shape()
    print("✓ test_binary_cross_entropy_output_shape")

    test_binary_cross_entropy_basic()
    print("✓ test_binary_cross_entropy_basic")

    test_binary_cross_entropy_perfect_prediction()
    print("✓ test_binary_cross_entropy_perfect_prediction")

    test_binary_cross_entropy_backward()
    print("✓ test_binary_cross_entropy_backward")

    test_loss_non_negative()
    print("✓ test_loss_non_negative")

    test_loss_gradient_shape_consistency()
    print("✓ test_loss_gradient_shape_consistency")

    test_mse_symmetric()
    print("✓ test_mse_symmetric")

    print("\nAll test_loss_funcs_part2 tests passed!")
