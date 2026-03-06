"""Tests for loss function backward passes (cross-entropy, BCE, MSE).

Note: Split from test_backward.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, ones_like
from shared.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
)
from shared.testing import check_gradient


fn test_cross_entropy_backward_shapes() raises:
    """Test that cross_entropy_backward returns correct gradient shape."""
    var batch = 4
    var num_classes = 10

    var logits_shape = List[Int]()
    logits_shape.append(batch)
    logits_shape.append(num_classes)
    var logits = ones(logits_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(batch)
    targets_shape.append(num_classes)
    var targets = zeros(targets_shape, DType.float32)
    for i in range(batch):
        targets._data.bitcast[Float32]()[i * num_classes] = 1.0

    var loss = cross_entropy(logits, targets)
    var grad_output = ones_like(loss)
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)

    var gl_shape = grad_logits.shape()
    assert_equal(gl_shape[0], batch)
    assert_equal(gl_shape[1], num_classes)


fn test_binary_cross_entropy_backward_shapes() raises:
    """Test that binary_cross_entropy_backward returns correct gradient shape."""
    var batch = 32
    var features = 1

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = zeros(pred_shape, DType.float32)

    for i in range(batch):
        predictions._data.bitcast[Float32]()[i] = Float32(i) / Float32(batch)

    var targets = zeros(pred_shape, DType.float32)
    for i in range(batch // 2, batch):
        targets._data.bitcast[Float32]()[i] = 1.0

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(grad_output, predictions, targets)

    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


fn test_binary_cross_entropy_backward_edge_cases() raises:
    """Test BCE backward with edge case values near 0 and 1."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = zeros(pred_shape, DType.float32)
    predictions._data.bitcast[Float32]()[0] = 0.001
    predictions._data.bitcast[Float32]()[1] = 0.999
    predictions._data.bitcast[Float32]()[2] = 0.5
    predictions._data.bitcast[Float32]()[3] = 0.1

    var targets = zeros(pred_shape, DType.float32)
    targets._data.bitcast[Float32]()[0] = 0.0
    targets._data.bitcast[Float32]()[1] = 1.0
    targets._data.bitcast[Float32]()[2] = 0.0
    targets._data.bitcast[Float32]()[3] = 1.0

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(grad_output, predictions, targets)

    for i in range(4):
        var grad = grad_pred._data.bitcast[Float32]()[i]
        assert_true(grad == grad, "Gradient should not be NaN")
        assert_true(grad > -1e10 and grad < 1e10, "Gradient should not be Inf")


fn test_mean_squared_error_backward_shapes() raises:
    """Test that mean_squared_error_backward returns correct gradient shape."""
    var batch = 16
    var features = 10

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = ones(pred_shape, DType.float32)

    var targets = zeros(pred_shape, DType.float32)
    for i in range(batch * features):
        targets._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(grad_output, predictions, targets)

    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


fn test_mean_squared_error_backward_zero_diff() raises:
    """Test MSE backward when predictions equal targets (zero gradient)."""
    var pred_shape = List[Int]()
    pred_shape.append(5)
    var predictions = zeros(pred_shape, DType.float32)
    for i in range(5):
        predictions._data.bitcast[Float32]()[i] = Float32(i)

    var targets = zeros(pred_shape, DType.float32)
    for i in range(5):
        targets._data.bitcast[Float32]()[i] = Float32(i)

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(grad_output, predictions, targets)

    for i in range(5):
        assert_almost_equal(
            grad_pred._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-6
        )


fn main() raises:
    """Run loss backward tests."""
    print("Running loss backward tests...")
    test_cross_entropy_backward_shapes()
    test_binary_cross_entropy_backward_shapes()
    test_binary_cross_entropy_backward_edge_cases()
    test_mean_squared_error_backward_shapes()
    test_mean_squared_error_backward_zero_diff()
    print("All loss backward tests passed!")
