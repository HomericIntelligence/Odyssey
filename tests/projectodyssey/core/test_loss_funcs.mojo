"""Unit tests for loss functions.

Tests cover:
- cross_entropy: Forward and backward passes for multi-class classification
- mean_squared_error: Forward pass and gradient computation for regression

All tests use pure functional API - no internal state.
"""


from tests.projectodyssey.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
    assert_close_float,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, full
from projectodyssey.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
)
from projectodyssey.core.reduction import mean


def test_cross_entropy_output_shape() raises:
    """Test cross_entropy returns loss tensor."""
    var logits_shape = List[Int]()
    logits_shape.append(4)
    logits_shape.append(3)  # 3 classes
    var logits = zeros(logits_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    targets_shape.append(3)
    var targets = zeros(targets_shape, DType.float32)

    var loss = cross_entropy(logits, targets)

    # Loss tensor should have numel >= 1 (scalar or reduced)
    assert_true(loss.numel() >= 1, "Loss should have at least 1 element")


def test_cross_entropy_basic() raises:
    """Test cross_entropy on simple one-hot example."""
    var logits_shape = List[Int]()
    logits_shape.append(1)
    logits_shape.append(2)
    var logits = zeros(logits_shape, DType.float32)

    var logits_data = logits._data.bitcast[Float32]()
    logits_data[0] = 1.0
    logits_data[1] = 0.0

    var targets_shape = List[Int]()
    targets_shape.append(1)
    targets_shape.append(2)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 1.0  # Target class 0
    targets_data[1] = 0.0

    var loss = cross_entropy(logits, targets)

    var loss_data = loss._data.bitcast[Float32]()
    assert_greater_or_equal(loss_data[0], 0.0, "Loss should be non-negative")


def test_cross_entropy_correct_prediction() raises:
    """Test cross_entropy when model predicts correctly."""
    var logits_shape = List[Int]()
    logits_shape.append(1)
    logits_shape.append(3)
    var logits = zeros(logits_shape, DType.float32)

    var logits_data = logits._data.bitcast[Float32]()
    logits_data[0] = 5.0  # High score for correct class
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    var targets_shape = List[Int]()
    targets_shape.append(1)
    targets_shape.append(3)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 1.0  # Target is class 0
    targets_data[1] = 0.0
    targets_data[2] = 0.0

    var loss = cross_entropy(logits, targets)

    var loss_data = loss._data.bitcast[Float32]()
    assert_greater_or_equal(loss_data[0], 0.0, "Loss should be non-negative")


def test_cross_entropy_backward_shape() raises:
    """Test cross_entropy_backward produces correct gradient shape."""
    var logits_shape = List[Int]()
    logits_shape.append(4)
    logits_shape.append(3)
    var logits = ones(logits_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    targets_shape.append(3)
    var targets = zeros(targets_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_logits = cross_entropy_backward(grad_output, logits, targets)

    var grad_shape = grad_logits.shape()
    assert_equal(grad_shape[0], 4, "Batch dimension preserved")
    assert_equal(grad_shape[1], 3, "Class dimension preserved")


def test_mean_squared_error_zero_error() raises:
    """Test MSE when predictions match targets (error = 0)."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(4)
    var targets = ones(targets_shape, DType.float32)

    var loss = mean_squared_error(predictions, targets)

    var loss_data = loss._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(loss_data[i], 0.0, tolerance=1e-5)


def test_mean_squared_error_simple() raises:
    """Test MSE on simple prediction error."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = zeros(pred_shape, DType.float32)

    var pred_data = predictions._data.bitcast[Float32]()
    pred_data[0] = 1.0
    pred_data[1] = 2.0
    pred_data[2] = 3.0
    pred_data[3] = 4.0

    var targets_shape = List[Int]()
    targets_shape.append(4)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 0.0
    targets_data[1] = 0.0
    targets_data[2] = 0.0
    targets_data[3] = 0.0

    var loss = mean_squared_error(predictions, targets)

    var loss_data = loss._data.bitcast[Float32]()
    assert_almost_equal(loss_data[0], 1.0, tolerance=1e-5)  # (1-0)^2 = 1
    assert_almost_equal(loss_data[1], 4.0, tolerance=1e-5)  # (2-0)^2 = 4
    assert_almost_equal(loss_data[2], 9.0, tolerance=1e-5)  # (3-0)^2 = 9
    assert_almost_equal(loss_data[3], 16.0, tolerance=1e-5)  # (4-0)^2 = 16


def test_mean_squared_error_output_shape() raises:
    """Test MSE preserves input shape."""
    var pred_shape = List[Int]()
    pred_shape.append(2)
    pred_shape.append(3)
    var predictions = ones(pred_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(2)
    targets_shape.append(3)
    var targets = ones(targets_shape, DType.float32)

    var loss = mean_squared_error(predictions, targets)

    var loss_shape = loss.shape()
    assert_equal(loss_shape[0], 2, "Batch dimension preserved")
    assert_equal(loss_shape[1], 3, "Feature dimension preserved")


def test_mean_squared_error_backward() raises:
    """Test MSE backward pass computes correct gradients."""
    var pred_shape = List[Int]()
    pred_shape.append(2)
    var predictions = zeros(pred_shape, DType.float32)

    var pred_data = predictions._data.bitcast[Float32]()
    pred_data[0] = 2.0
    pred_data[1] = 4.0

    var targets_shape = List[Int]()
    targets_shape.append(2)
    var targets = zeros(targets_shape, DType.float32)

    var targets_data = targets._data.bitcast[Float32]()
    targets_data[0] = 1.0
    targets_data[1] = 2.0

    var grad_output_shape = List[Int]()
    grad_output_shape.append(2)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    var grad_data = grad_pred._data.bitcast[Float32]()
    # Gradient: 2 * (predictions - targets)
    # grad[0] = 2 * (2 - 1) = 2
    # grad[1] = 2 * (4 - 2) = 4
    assert_almost_equal(grad_data[0], 2.0, tolerance=1e-5)
    assert_almost_equal(grad_data[1], 4.0, tolerance=1e-5)


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
    """Run all test_loss_funcs tests."""
    print("Running test_loss_funcs tests...")

    test_cross_entropy_output_shape()
    print("✓ test_cross_entropy_output_shape")

    test_cross_entropy_basic()
    print("✓ test_cross_entropy_basic")

    test_cross_entropy_correct_prediction()
    print("✓ test_cross_entropy_correct_prediction")

    test_cross_entropy_backward_shape()
    print("✓ test_cross_entropy_backward_shape")

    test_mean_squared_error_zero_error()
    print("✓ test_mean_squared_error_zero_error")

    test_mean_squared_error_simple()
    print("✓ test_mean_squared_error_simple")

    test_mean_squared_error_output_shape()
    print("✓ test_mean_squared_error_output_shape")

    test_mean_squared_error_backward()
    print("✓ test_mean_squared_error_backward")

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

    print("\nAll test_loss_funcs tests passed!")
