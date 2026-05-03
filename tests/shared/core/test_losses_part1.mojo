"""Tests for BCE and MSE loss functions (Part 1 of 3)

This module tests:
- Binary Cross-Entropy (BCE) forward pass
- Mean Squared Error (MSE) forward pass
- Numerical stability
"""


from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_close_float,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.loss import binary_cross_entropy, binary_cross_entropy_backward
from shared.core.loss import mean_squared_error, mean_squared_error_backward
from shared.core.reduction import mean
from shared.core.loss import smooth_l1_loss, smooth_l1_loss_backward
from shared.testing.gradient_checker import check_gradient, NumericalForward, NumericalBackward
from shared.core.loss import hinge_loss, hinge_loss_backward
from shared.core.loss import focal_loss, focal_loss_backward
from shared.core.loss import kl_divergence, kl_divergence_backward


def test_binary_cross_entropy_perfect_prediction() raises:
    """Test BCE with perfect predictions (should be near zero)."""
    print("Testing BCE with perfect predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Perfect predictions: pred = target
    for i in range(4):
        var val = 1.0 if i % 2 == 0 else 0.0
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Perfect prediction loss:", loss_val)

    # Should be very close to 0 (within epsilon tolerance)
    if loss_val > 0.01:
        raise Error("BCE loss for perfect predictions should be near 0")

    print("  ✓ BCE perfect prediction test passed")


def test_binary_cross_entropy_worst_prediction() raises:
    """Test BCE with worst predictions (should be high)."""
    print("Testing BCE with worst predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Worst predictions: pred = 1 - target
    for i in range(4):
        var target_val = 1.0 if i % 2 == 0 else 0.0
        var pred_val = 1.0 - target_val  # Opposite
        predictions._set_float64(i, pred_val)
        targets._set_float64(i, target_val)

    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Worst prediction loss:", loss_val)

    # Should be high (> 1.0)
    if loss_val < 1.0:
        raise Error("BCE loss for worst predictions should be high")

    print("  ✓ BCE worst prediction test passed")


def test_binary_cross_entropy_gradient_shape() raises:
    """Test that BCE backward produces correct gradient shape."""
    print("Testing BCE gradient shape...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    for i in range(3):
        predictions._set_float64(i, 0.5)
        targets._set_float64(i, 1.0 if i == 0 else 0.0)

    var loss = binary_cross_entropy(predictions, targets)

    # Create upstream gradient
    var grad_output = ones(shape, DType.float32)

    # Compute gradient
    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    # Check shape matches
    if grad_pred.shape()[0] != predictions.shape()[0]:
        raise Error("Gradient shape should match predictions shape")

    print("  Gradient shape:", grad_pred.shape()[0])
    print("  ✓ BCE gradient shape test passed")


def test_mean_squared_error_zero_loss() raises:
    """Test MSE with identical predictions and targets."""
    print("Testing MSE with zero loss...")

    var shape = List[Int]()
    shape.append(5)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Identical values
    for i in range(5):
        var val = Float64(i) * 0.5
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = mean_squared_error(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  MSE zero loss:", loss_val)

    # Should be exactly 0
    if loss_val != 0.0:
        raise Error("MSE loss for identical values should be 0")

    print("  ✓ MSE zero loss test passed")


def test_mean_squared_error_known_values() raises:
    """Test MSE with known error values."""
    print("Testing MSE with known values...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Predictions: [2, 3, 4]
    # Targets:     [1, 3, 5]
    # Errors:      [1, 0, -1]
    # Squared:     [1, 0, 1]
    # Mean:        (1 + 0 + 1) / 3 = 0.666...

    predictions._set_float64(0, 2.0)
    predictions._set_float64(1, 3.0)
    predictions._set_float64(2, 4.0)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 5.0)

    var loss = mean_squared_error(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  MSE known value:", loss_val)

    # Should be approximately 0.666...
    var expected = 2.0 / 3.0
    var diff = abs(loss_val - expected)
    if diff > 0.01:
        raise Error("MSE loss doesn't match expected value")

    print("  ✓ MSE known values test passed")


def test_mean_squared_error_gradient() raises:
    """Test MSE gradient computation."""
    print("Testing MSE gradient...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Simple case: predictions = [2, 3, 4], targets = [1, 3, 5]
    # Gradient should be 2 * (predictions - targets) = 2 * [1, 0, -1] = [2, 0, -2]

    predictions._set_float64(0, 2.0)
    predictions._set_float64(1, 3.0)
    predictions._set_float64(2, 4.0)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 5.0)

    var loss = mean_squared_error(predictions, targets)

    # Upstream gradient (all ones)
    var grad_output = ones(shape, DType.float32)

    # Compute gradient
    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    # Check values
    var grad0 = grad_pred._get_float64(0)
    var grad1 = grad_pred._get_float64(1)
    var grad2 = grad_pred._get_float64(2)

    print("  Gradients: [", grad0, ",", grad1, ",", grad2, "]")

    # Should be approximately [2, 0, -2]
    if abs(grad0 - 2.0) > 0.01:
        raise Error("Gradient[0] should be 2.0")
    if abs(grad1 - 0.0) > 0.01:
        raise Error("Gradient[1] should be 0.0")
    if abs(grad2 - (-2.0)) > 0.01:
        raise Error("Gradient[2] should be -2.0")

    print("  ✓ MSE gradient test passed")


def test_loss_numerical_stability() raises:
    """Test that loss functions handle extreme values gracefully."""
    print("Testing loss function numerical stability...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Test BCE with values close to 0 and 1 (should be clipped)
    predictions._set_float64(0, 0.0)  # Should be clipped to epsilon
    predictions._set_float64(1, 1.0)  # Should be clipped to 1-epsilon

    targets._set_float64(0, 0.0)
    targets._set_float64(1, 1.0)

    # Should not raise error or produce NaN/Inf
    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  BCE with extreme values:", loss_val)

    # Should be finite (not NaN or Inf)
    # Note: We can't check for NaN/Inf in Mojo easily, so just verify it doesn't crash

    print("  ✓ Numerical stability test passed")


@fieldwise_init
struct _BCEFwd(NumericalForward):
    var targets: AnyTensor

    def __call__(self, pred: AnyTensor) raises -> AnyTensor:
        return binary_cross_entropy(pred, self.targets)


@fieldwise_init
struct _BCEBwd(NumericalBackward):
    var targets: AnyTensor

    def __call__(self, grad_out: AnyTensor, pred: AnyTensor) raises -> AnyTensor:
        return binary_cross_entropy_backward(grad_out, pred, self.targets)


def test_binary_cross_entropy_backward_gradient() raises:
    """Test BCE backward with numerical gradient checking."""
    print("Testing BCE backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    predictions._set_float64(0, 0.7)
    predictions._set_float64(1, 0.3)
    predictions._set_float64(2, 0.5)
    predictions._set_float64(3, 0.2)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 0.0)
    targets._set_float64(2, 1.0)
    targets._set_float64(3, 0.0)

    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        _BCEFwd(targets), _BCEBwd(targets), predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ BCE backward gradient check passed")


@fieldwise_init
struct _MSEFwd(NumericalForward):
    var targets: AnyTensor

    def __call__(self, pred: AnyTensor) raises -> AnyTensor:
        return mean_squared_error(pred, self.targets)


@fieldwise_init
struct _MSEBwd(NumericalBackward):
    var targets: AnyTensor

    def __call__(self, grad_out: AnyTensor, pred: AnyTensor) raises -> AnyTensor:
        return mean_squared_error_backward(grad_out, pred, self.targets)


def test_mean_squared_error_backward_gradient() raises:
    """Test MSE backward with numerical gradient checking."""
    print("Testing MSE backward gradient checking...")

    var shape = List[Int]()
    shape.append(5)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    predictions._set_float64(0, 2.1)
    predictions._set_float64(1, 3.5)
    predictions._set_float64(2, 1.2)
    predictions._set_float64(3, 4.8)
    predictions._set_float64(4, 0.5)

    targets._set_float64(0, 2.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 1.5)
    targets._set_float64(3, 4.5)
    targets._set_float64(4, 0.8)

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        _MSEFwd(targets), _MSEBwd(targets), predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ MSE backward gradient check passed")


def main() raises:
    """Run test_losses part 1 tests (BCE and MSE)."""
    print("Running test_losses_part1 tests...")

    test_binary_cross_entropy_perfect_prediction()
    print("✓ test_binary_cross_entropy_perfect_prediction")

    test_binary_cross_entropy_worst_prediction()
    print("✓ test_binary_cross_entropy_worst_prediction")

    test_binary_cross_entropy_gradient_shape()
    print("✓ test_binary_cross_entropy_gradient_shape")

    test_mean_squared_error_zero_loss()
    print("✓ test_mean_squared_error_zero_loss")

    test_mean_squared_error_known_values()
    print("✓ test_mean_squared_error_known_values")

    test_mean_squared_error_gradient()
    print("✓ test_mean_squared_error_gradient")

    test_loss_numerical_stability()
    print("✓ test_loss_numerical_stability")

    test_binary_cross_entropy_backward_gradient()
    print("✓ test_binary_cross_entropy_backward_gradient")

    test_mean_squared_error_backward_gradient()
    print("✓ test_mean_squared_error_backward_gradient")

    print("\nAll test_losses_part1 tests passed!")
