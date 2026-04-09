# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_backward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for loss function backward passes (cross-entropy, BCE, MSE)."""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, ones_like
from shared.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
)
from shared.testing import check_gradient, NumericalForward, NumericalBackward


def test_cross_entropy_backward_shapes() raises:
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
        targets.set(i * num_classes, Float64(1.0))

    var loss = cross_entropy(logits, targets)
    var grad_output = ones_like(loss)
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)

    var gl_shape = grad_logits.shape()
    assert_equal(gl_shape[0], batch)
    assert_equal(gl_shape[1], num_classes)


def test_binary_cross_entropy_backward_shapes() raises:
    """Test that binary_cross_entropy_backward returns correct gradient shape.
    """
    var batch = 32
    var features = 1

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = zeros(pred_shape, DType.float32)

    for i in range(batch):
        predictions.set(i, Float64(Float32(i) / Float32(batch)))

    var targets = zeros(pred_shape, DType.float32)
    for i in range(batch // 2, batch):
        targets.set(i, Float64(1.0))

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


def test_binary_cross_entropy_backward_edge_cases() raises:
    """Test BCE backward with edge case values near 0 and 1."""
    var pred_shape = List[Int]()
    pred_shape.append(4)
    var predictions = zeros(pred_shape, DType.float32)
    predictions.set(0, Float64(0.001))
    predictions.set(1, Float64(0.999))
    predictions.set(2, Float64(0.5))
    predictions.set(3, Float64(0.1))

    var targets = zeros(pred_shape, DType.float32)
    targets.set(0, Float64(0.0))
    targets.set(1, Float64(1.0))
    targets.set(2, Float64(0.0))
    targets.set(3, Float64(1.0))

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    for i in range(4):
        var grad = grad_pred._data.bitcast[Float32]()[i]
        assert_true(grad == grad, "Gradient should not be NaN")
        assert_true(grad > -1e10 and grad < 1e10, "Gradient should not be Inf")


def test_mean_squared_error_backward_shapes() raises:
    """Test that mean_squared_error_backward returns correct gradient shape."""
    var batch = 16
    var features = 10

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = ones(pred_shape, DType.float32)

    var targets = zeros(pred_shape, DType.float32)
    for i in range(batch * features):
        targets.set(i, Float64(Float32(i) * 0.1))

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


def test_mean_squared_error_backward_zero_diff() raises:
    """Test MSE backward when predictions equal targets (zero gradient)."""
    var pred_shape = List[Int]()
    pred_shape.append(5)
    var predictions = zeros(pred_shape, DType.float32)
    for i in range(5):
        predictions.set(i, Float64(Float32(i)))

    var targets = zeros(pred_shape, DType.float32)
    for i in range(5):
        targets.set(i, Float64(Float32(i)))

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    for i in range(5):
        assert_almost_equal(
            grad_pred._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-6
        )


@fieldwise_init
struct _CEFwd(NumericalForward):
    var targets: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return cross_entropy(inp, self.targets)


@fieldwise_init
struct _CEBwd(NumericalBackward):
    var targets: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return cross_entropy_backward(grad_out, inp, self.targets)


def test_cross_entropy_backward_gradient() raises:
    """Test cross-entropy backward with numerical gradient checking."""
    var batch = 2
    var num_classes = 3

    var logits_shape = List[Int]()
    logits_shape.append(batch)
    logits_shape.append(num_classes)
    var logits = zeros(logits_shape, DType.float32)
    logits.set(0, Float64(0.5))
    logits.set(1, Float64(-0.3))
    logits.set(2, Float64(1.2))
    logits.set(3, Float64(-0.8))
    logits.set(4, Float64(0.1))
    logits.set(5, Float64(0.7))

    var targets_shape = List[Int]()
    targets_shape.append(batch)
    targets_shape.append(num_classes)
    var targets = zeros(targets_shape, DType.float32)
    targets.set(0, Float64(1.0))
    targets.set(4, Float64(1.0))

    var loss = cross_entropy(logits, targets)
    var grad_output = ones_like(loss)
    check_gradient(_CEFwd(targets), _CEBwd(targets), logits, grad_output, rtol=1e-3, atol=1e-3)


@fieldwise_init
struct _BCEBLFwd(NumericalForward):
    var targets: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return binary_cross_entropy(inp, self.targets)


@fieldwise_init
struct _BCEBLBwd(NumericalBackward):
    var targets: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return binary_cross_entropy_backward(grad_out, inp, self.targets)


def test_binary_cross_entropy_backward_gradient() raises:
    """Test binary cross-entropy backward with numerical gradient checking."""
    var batch = 8

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    var predictions = zeros(pred_shape, DType.float32)
    predictions.set(0, Float64(0.1))
    predictions.set(1, Float64(0.3))
    predictions.set(2, Float64(0.5))
    predictions.set(3, Float64(0.7))
    predictions.set(4, Float64(0.9))
    predictions.set(5, Float64(0.2))
    predictions.set(6, Float64(0.6))
    predictions.set(7, Float64(0.8))

    var targets = zeros(pred_shape, DType.float32)
    targets.set(0, Float64(0.0))
    targets.set(1, Float64(1.0))
    targets.set(2, Float64(0.0))
    targets.set(3, Float64(1.0))
    targets.set(4, Float64(0.0))
    targets.set(5, Float64(1.0))
    targets.set(6, Float64(0.0))
    targets.set(7, Float64(1.0))

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    check_gradient(
        _BCEBLFwd(targets), _BCEBLBwd(targets), predictions, grad_output, rtol=1e-3, atol=1e-6
    )


@fieldwise_init
struct _MSEBLFwd(NumericalForward):
    var targets: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return mean_squared_error(inp, self.targets)


@fieldwise_init
struct _MSEBLBwd(NumericalBackward):
    var targets: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return mean_squared_error_backward(grad_out, inp, self.targets)


def test_mean_squared_error_backward_gradient() raises:
    """Test mean squared error backward with numerical gradient checking."""
    var batch = 4
    var features = 3

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = zeros(pred_shape, DType.float32)
    predictions.set(0, Float64(0.5))
    predictions.set(1, Float64(-0.3))
    predictions.set(2, Float64(1.2))
    predictions.set(3, Float64(-0.8))
    predictions.set(4, Float64(0.1))
    predictions.set(5, Float64(0.7))
    predictions.set(6, Float64(2.0))
    predictions.set(7, Float64(-1.5))
    predictions.set(8, Float64(0.0))
    predictions.set(9, Float64(1.0))
    predictions.set(10, Float64(-0.5))
    predictions.set(11, Float64(0.3))

    var targets = zeros(pred_shape, DType.float32)
    targets.set(0, Float64(0.2))
    targets.set(1, Float64(0.4))
    targets.set(2, Float64(0.8))
    targets.set(3, Float64(-0.3))
    targets.set(4, Float64(0.5))
    targets.set(5, Float64(1.0))
    targets.set(6, Float64(1.5))
    targets.set(7, Float64(-1.0))
    targets.set(8, Float64(0.3))
    targets.set(9, Float64(0.7))
    targets.set(10, Float64(0.0))
    targets.set(11, Float64(0.6))

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    check_gradient(
        _MSEBLFwd(targets), _MSEBLBwd(targets), predictions, grad_output, rtol=1e-3, atol=1e-6
    )


def main() raises:
    """Run loss backward tests."""
    print("Running loss backward tests...")
    test_cross_entropy_backward_shapes()
    test_cross_entropy_backward_gradient()
    test_binary_cross_entropy_backward_shapes()
    test_binary_cross_entropy_backward_edge_cases()
    test_binary_cross_entropy_backward_gradient()
    test_mean_squared_error_backward_shapes()
    test_mean_squared_error_backward_zero_diff()
    test_mean_squared_error_backward_gradient()
    print("All loss backward tests passed!")
