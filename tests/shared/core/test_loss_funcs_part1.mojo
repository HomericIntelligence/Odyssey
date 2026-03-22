# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loss_funcs.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for loss functions - Part 1.

Tests cover:
- cross_entropy: Forward and backward passes for multi-class classification
- mean_squared_error: Forward pass and gradient computation for regression

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
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
)
from shared.core.reduction import mean


# ============================================================================
# Cross Entropy Tests
# ============================================================================


fn test_cross_entropy_output_shape() raises:
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


fn test_cross_entropy_basic() raises:
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


fn test_cross_entropy_correct_prediction() raises:
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


fn test_cross_entropy_backward_shape() raises:
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


# ============================================================================
# Mean Squared Error Tests
# ============================================================================


fn test_mean_squared_error_zero_error() raises:
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


fn test_mean_squared_error_simple() raises:
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


fn test_mean_squared_error_output_shape() raises:
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


fn test_mean_squared_error_backward() raises:
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


fn main() raises:
    """Run loss function tests - Part 1."""
    print("Running loss function tests (Part 1)...")

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

    print("\nAll loss function tests (Part 1) passed!")
