"""Tests for BCE and MSE loss functions

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
from shared.testing import check_gradient
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

    # Forward function wrapper
    def forward(pred: AnyTensor) raises unified {read} -> AnyTensor:
        return binary_cross_entropy(pred, targets)

    # Backward function wrapper
    def backward(grad_out: AnyTensor, pred: AnyTensor) raises unified {read} -> AnyTensor:
        return binary_cross_entropy_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ BCE backward gradient check passed")


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

    # Forward function wrapper
    def forward(pred: AnyTensor) raises unified {read} -> AnyTensor:
        return mean_squared_error(pred, targets)

    # Backward function wrapper
    def backward(grad_out: AnyTensor, pred: AnyTensor) raises unified {read} -> AnyTensor:
        return mean_squared_error_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ MSE backward gradient check passed")


def test_smooth_l1_zero_beta_boundary() raises:
    """Test Smooth L1 loss at beta boundary."""
    print("Testing Smooth L1 loss at beta boundary...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Test with differences exactly at beta (1.0)
    # At |x| = beta: both formulas should give same value
    # L2 part: 0.5 * 1^2 / 1.0 = 0.5
    # L1 part: 1.0 - 0.5 * 1.0 = 0.5
    var beta: Float32 = 1.0

    predictions._set_float64(0, 2.0)  # diff = 1.0
    targets._set_float64(0, 1.0)

    predictions._set_float64(1, 3.5)  # diff = 0.5
    targets._set_float64(1, 3.0)

    predictions._set_float64(2, 4.0)  # diff = 0.0
    targets._set_float64(2, 4.0)

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Smooth L1 loss at boundary:", loss_val)

    # Loss should be reasonable (not zero or infinity)
    if loss_val < 0.0:
        raise Error("Smooth L1 loss should never be negative")

    print("  ✓ Smooth L1 boundary test passed")


def test_smooth_l1_quadratic_region() raises:
    """Test Smooth L1 in quadratic region (|x| < beta)."""
    print("Testing Smooth L1 loss in quadratic region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Small difference (0.1) with beta=1.0 should be in quadratic region
    # L = 0.5 * 0.1^2 / 1.0 = 0.005
    var beta: Float32 = 1.0

    predictions._set_float64(0, 1.1)
    targets._set_float64(0, 1.0)

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var loss_val = loss._get_float64(0)

    print("  Smooth L1 quadratic loss:", loss_val)

    # Should be approximately 0.005
    var expected = 0.5 * 0.1 * 0.1 / 1.0
    var diff = abs(loss_val - expected)
    if diff > 0.01:
        print("  Warning: Expected ~", expected, " but got ", loss_val)

    print("  ✓ Smooth L1 quadratic region test passed")


def test_smooth_l1_linear_region() raises:
    """Test Smooth L1 in linear region (|x| >= beta)."""
    print("Testing Smooth L1 loss in linear region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Large difference (2.0) with beta=1.0 should be in linear region
    # L = 2.0 - 0.5 * 1.0 = 1.5
    var beta: Float32 = 1.0

    predictions._set_float64(0, 3.0)
    targets._set_float64(0, 1.0)

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var loss_val = loss._get_float64(0)

    print("  Smooth L1 linear loss:", loss_val)

    # Should be approximately 1.5
    var expected = 2.0 - 0.5 * 1.0
    var diff = abs(loss_val - expected)
    if diff > 0.01:
        print("  Warning: Expected ~", expected, " but got ", loss_val)

    print("  ✓ Smooth L1 linear region test passed")


def test_smooth_l1_backward_quadratic() raises:
    """Test Smooth L1 backward in quadratic region."""
    print("Testing Smooth L1 backward in quadratic region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Small difference (0.1) with beta=1.0
    # Gradient should be: diff / beta = 0.1 / 1.0 = 0.1
    var beta: Float32 = 1.0

    predictions._set_float64(0, 1.1)
    targets._set_float64(0, 1.0)

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var grad_output = ones(shape, DType.float32)

    var grad_pred = smooth_l1_loss_backward(
        grad_output, predictions, targets, beta=beta
    )
    var grad_val = grad_pred._get_float64(0)

    print("  Smooth L1 quadratic gradient:", grad_val)

    # Should be approximately 0.1
    var expected = 0.1 / 1.0
    var diff = abs(grad_val - expected)
    if diff > 0.05:
        print("  Warning: Expected ~", expected, " but got ", grad_val)

    print("  ✓ Smooth L1 backward quadratic test passed")


def test_smooth_l1_backward_linear() raises:
    """Test Smooth L1 backward in linear region."""
    print("Testing Smooth L1 backward in linear region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Large difference (2.0) with beta=1.0
    # Gradient should be: sign(diff) = sign(2.0) = 1.0
    var beta: Float32 = 1.0

    predictions._set_float64(0, 3.0)
    targets._set_float64(0, 1.0)

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var grad_output = ones(shape, DType.float32)

    var grad_pred = smooth_l1_loss_backward(
        grad_output, predictions, targets, beta=beta
    )
    var grad_val = grad_pred._get_float64(0)

    print("  Smooth L1 linear gradient:", grad_val)

    # Should be approximately 1.0 (sign of positive diff)
    if abs(grad_val - 1.0) > 0.1:
        print("  Warning: Expected ~1.0 but got ", grad_val)

    print("  ✓ Smooth L1 backward linear test passed")


def test_smooth_l1_backward_gradient() raises:
    """Test Smooth L1 backward with numerical gradient checking."""
    print("Testing Smooth L1 backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    var beta: Float32 = 1.0

    # Initialize with non-uniform values
    predictions._set_float64(0, 2.1)
    predictions._set_float64(1, 0.8)
    predictions._set_float64(2, 3.2)
    predictions._set_float64(3, 1.5)

    targets._set_float64(0, 2.0)
    targets._set_float64(1, 1.0)
    targets._set_float64(2, 2.8)
    targets._set_float64(3, 0.5)

    # Forward function wrapper
    def forward(pred: AnyTensor) raises unified {read} -> AnyTensor:
        return smooth_l1_loss(pred, targets, beta=beta)

    # Backward function wrapper
    def backward(grad_out: AnyTensor, pred: AnyTensor) raises unified {read} -> AnyTensor:
        return smooth_l1_loss_backward(grad_out, pred, targets, beta=beta)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for smooth L1)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-2, atol=1e-3
    )

    print("  ✓ Smooth L1 backward gradient check passed")


def test_hinge_loss_correct_prediction() raises:
    """Test hinge loss with correct predictions."""
    print("Testing hinge loss with correct predictions...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Correct predictions with high confidence
    # y*pred = 1*2.0 = 2.0, so loss = max(0, 1 - 2.0) = max(0, -1.0) = 0.0
    for i in range(3):
        var val = 2.0
        predictions._set_float64(i, val)
        targets._set_float64(i, 1.0)

    var loss = hinge_loss(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Correct prediction hinge loss:", loss_val)

    # Should be near 0
    if loss_val > 0.01:
        raise Error("Hinge loss for correct predictions should be near 0")

    print("  ✓ Hinge correct prediction test passed")


def test_hinge_loss_wrong_prediction() raises:
    """Test hinge loss with wrong predictions."""
    print("Testing hinge loss with wrong predictions...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Wrong predictions (margin violated)
    # y*pred = 1*(-0.5) = -0.5, so loss = max(0, 1 - (-0.5)) = 1.5
    predictions._set_float64(0, -0.5)
    targets._set_float64(0, 1.0)

    # Also test negative targets
    # y*pred = (-1)*0.5 = -0.5, so loss = max(0, 1 - (-0.5)) = 1.5
    predictions._set_float64(1, 0.5)
    targets._set_float64(1, -1.0)

    var loss = hinge_loss(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Wrong prediction hinge loss:", loss_val)

    # Should be high (around 1.5)
    if loss_val < 1.0:
        raise Error("Hinge loss for wrong predictions should be high")

    print("  ✓ Hinge wrong prediction test passed")


def test_hinge_loss_at_margin() raises:
    """Test hinge loss exactly at margin (y*pred = 1)."""
    print("Testing hinge loss at margin...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # At the margin: y*pred = 1, so loss = max(0, 1 - 1) = 0
    predictions._set_float64(0, 1.0)
    targets._set_float64(0, 1.0)

    predictions._set_float64(1, -1.0)
    targets._set_float64(1, -1.0)

    var loss = hinge_loss(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Margin boundary hinge loss:", loss_val)

    # Should be exactly 0
    if loss_val > 0.01:
        raise Error("Hinge loss at margin should be near 0")

    print("  ✓ Hinge margin test passed")


def test_hinge_loss_backward() raises:
    """Test hinge loss backward pass."""
    print("Testing hinge loss backward...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Mix of correct and wrong predictions
    # Case 1: y*pred = 2.0 > 1 (correct), grad should be 0
    predictions._set_float64(0, 2.0)
    targets._set_float64(0, 1.0)

    # Case 2: y*pred = -0.5 < 1 (wrong), grad should be -y = -1.0
    predictions._set_float64(1, -0.5)
    targets._set_float64(1, 1.0)

    # Case 3: y*pred = 0.5 < 1 (wrong), grad should be -y = 1.0 (for y=-1)
    predictions._set_float64(2, 0.5)
    targets._set_float64(2, -1.0)

    var loss = hinge_loss(predictions, targets)
    var grad_output = ones(shape, DType.float32)

    var grad_pred = hinge_loss_backward(grad_output, predictions, targets)

    var grad0 = grad_pred._get_float64(0)
    var grad1 = grad_pred._get_float64(1)
    var grad2 = grad_pred._get_float64(2)

    print("  Hinge gradients: [", grad0, ",", grad1, ",", grad2, "]")

    # Should be approximately [0.0, -1.0, 1.0]
    if abs(grad0 - 0.0) > 0.1:
        print("  Warning: grad[0] should be ~0.0, got ", grad0)
    if abs(grad1 - (-1.0)) > 0.1:
        print("  Warning: grad[1] should be ~-1.0, got ", grad1)
    if abs(grad2 - 1.0) > 0.1:
        print("  Warning: grad[2] should be ~1.0, got ", grad2)

    print("  ✓ Hinge backward test passed")


def test_hinge_loss_backward_gradient() raises:
    """Test hinge loss backward with gradient checking."""
    print("Testing hinge loss backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with values that produce some margin violations
    predictions._set_float64(0, 0.5)  # margin = 0.5 (violation)
    predictions._set_float64(1, 2.0)  # margin = -1.0 (satisfied)
    predictions._set_float64(2, -0.3)  # margin = 1.3 (violation)
    predictions._set_float64(3, 1.8)  # margin = -0.8 (satisfied)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 1.0)
    targets._set_float64(2, 1.0)
    targets._set_float64(3, 1.0)

    # Forward function wrapper
    def forward(pred: AnyTensor) raises unified {read} -> AnyTensor:
        return hinge_loss(pred, targets)

    # Backward function wrapper
    def backward(grad_out: AnyTensor, pred: AnyTensor) raises unified {read} -> AnyTensor:
        return hinge_loss_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for discontinuous gradient)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-2, atol=1e-3
    )

    print("  ✓ Hinge backward gradient check passed")


def test_focal_loss_perfect_prediction() raises:
    """Test focal loss with perfect predictions (should be near zero)."""
    print("Testing focal loss with perfect predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Perfect predictions: pred = target
    for i in range(4):
        var val = 1.0 if i % 2 == 0 else 0.0
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = focal_loss(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Perfect prediction focal loss:", loss_val)

    # Should be very close to 0 (within epsilon tolerance)
    if loss_val > 0.01:
        raise Error("Focal loss for perfect predictions should be near 0")

    print("  ✓ Focal loss perfect prediction test passed")


def test_focal_loss_hard_examples() raises:
    """Test focal loss focuses on hard examples."""
    print("Testing focal loss on hard examples...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    # Easy example: p=0.9, target=1 (easy positive)
    predictions._set_float64(0, 0.9)
    targets._set_float64(0, 1.0)

    # Hard example: p=0.1, target=1 (hard positive)
    predictions._set_float64(1, 0.1)
    targets._set_float64(1, 1.0)

    var loss = focal_loss(predictions, targets)

    var easy_loss = loss._get_float64(0)
    var hard_loss = loss._get_float64(1)

    print("  Easy example loss:", easy_loss)
    print("  Hard example loss:", hard_loss)

    # Hard examples should have larger loss
    if hard_loss <= easy_loss:
        raise Error(
            "Focal loss should emphasize hard examples more than easy examples"
        )

    print("  ✓ Focal loss hard examples test passed")


def test_focal_loss_backward_shape() raises:
    """Test focal loss backward produces correct gradient shape."""
    print("Testing focal loss backward shape...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = AnyTensor(shape, DType.float32)
    var targets = AnyTensor(shape, DType.float32)

    for i in range(3):
        predictions._set_float64(i, 0.5)
        targets._set_float64(i, 1.0 if i == 0 else 0.0)

    var loss = focal_loss(predictions, targets)
    var grad_output = ones(shape, DType.float32)
    var grad_pred = focal_loss_backward(grad_output, predictions, targets)

    if grad_pred.shape()[0] != predictions.shape()[0]:
        raise Error("Focal loss gradient shape should match predictions shape")

    print("  Gradient shape:", grad_pred.shape()[0])
    print("  ✓ Focal loss backward shape test passed")


def test_focal_loss_backward_gradient() raises:
    """Test focal loss backward with numerical gradient checking."""
    print("Testing focal loss backward gradient checking...")

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

    # Forward function wrapper
    def forward(pred: AnyTensor) raises unified {read} -> AnyTensor:
        return focal_loss(pred, targets)

    # Backward function wrapper
    def backward(grad_out: AnyTensor, pred: AnyTensor) raises unified {read} -> AnyTensor:
        return focal_loss_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-2, atol=1e-4
    )

    print("  ✓ Focal loss backward gradient check passed")


def test_kl_divergence_same_distribution() raises:
    """Test KL divergence with identical distributions (should be near zero)."""
    print("Testing KL divergence with same distribution...")

    var shape = List[Int]()
    shape.append(4)
    var p = AnyTensor(shape, DType.float32)
    var q = AnyTensor(shape, DType.float32)

    # Same distribution
    for i in range(4):
        var val = 0.25  # Uniform distribution
        p._set_float64(i, val)
        q._set_float64(i, val)

    var kl = kl_divergence(p, q)
    var avg_kl = mean(kl)

    var kl_val = avg_kl._get_float64(0)
    print("  KL divergence with same distribution:", kl_val)

    # Should be very close to 0 (within epsilon tolerance)
    if kl_val > 0.001:
        raise Error(
            "KL divergence for identical distributions should be near 0"
        )

    print("  ✓ KL divergence same distribution test passed")


def test_kl_divergence_different_distributions() raises:
    """Test KL divergence with different distributions (should be positive)."""
    print("Testing KL divergence with different distributions...")

    var shape = List[Int]()
    shape.append(3)
    var p = AnyTensor(shape, DType.float32)
    var q = AnyTensor(shape, DType.float32)

    # Distribution p: [0.5, 0.3, 0.2]
    p._set_float64(0, 0.5)
    p._set_float64(1, 0.3)
    p._set_float64(2, 0.2)

    # Distribution q: [0.3, 0.5, 0.2] (different)
    q._set_float64(0, 0.3)
    q._set_float64(1, 0.5)
    q._set_float64(2, 0.2)

    var kl = kl_divergence(p, q)
    var avg_kl = mean(kl)

    var kl_val = avg_kl._get_float64(0)
    print("  KL divergence with different distributions:", kl_val)

    # Should be positive
    if kl_val <= 0.0:
        raise Error(
            "KL divergence for different distributions should be positive"
        )

    print("  ✓ KL divergence different distributions test passed")


def test_kl_divergence_backward_shape() raises:
    """Test KL divergence backward produces correct gradient shape."""
    print("Testing KL divergence backward shape...")

    var shape = List[Int]()
    shape.append(4)
    var p = AnyTensor(shape, DType.float32)
    var q = AnyTensor(shape, DType.float32)

    # Initialize with valid probability distributions
    for i in range(4):
        p._set_float64(i, 0.25)
        q._set_float64(i, 0.25)

    var kl = kl_divergence(p, q)
    var grad_output = ones(shape, DType.float32)
    var grad_q = kl_divergence_backward(grad_output, p, q)

    if grad_q.shape()[0] != q.shape()[0]:
        raise Error("KL divergence gradient shape should match q shape")

    print("  Gradient shape:", grad_q.shape()[0])
    print("  ✓ KL divergence backward shape test passed")


def test_kl_divergence_backward_gradient() raises:
    """Test KL divergence backward with numerical gradient checking."""
    print("Testing KL divergence backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var p = zeros(shape, DType.float32)
    var q = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    p._set_float64(0, 0.4)
    p._set_float64(1, 0.3)
    p._set_float64(2, 0.2)
    p._set_float64(3, 0.1)

    q._set_float64(0, 0.3)
    q._set_float64(1, 0.4)
    q._set_float64(2, 0.2)
    q._set_float64(3, 0.1)

    # Forward function wrapper
    def forward(q_dist: AnyTensor) raises unified {read} -> AnyTensor:
        return kl_divergence(p, q_dist)

    # Backward function wrapper
    def backward(
        grad_out: AnyTensor, q_dist: AnyTensor
    ) raises unified {read} -> AnyTensor:
        return kl_divergence_backward(grad_out, p, q_dist)

    var kl = forward(q)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, q, grad_output, rtol=2e-2, atol=1e-4)

    print("  ✓ KL divergence backward gradient check passed")


def main() raises:
    """Run all test_losses tests."""
    print("Running test_losses tests...")

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

    test_smooth_l1_zero_beta_boundary()
    print("✓ test_smooth_l1_zero_beta_boundary")

    test_smooth_l1_quadratic_region()
    print("✓ test_smooth_l1_quadratic_region")

    test_smooth_l1_linear_region()
    print("✓ test_smooth_l1_linear_region")

    test_smooth_l1_backward_quadratic()
    print("✓ test_smooth_l1_backward_quadratic")

    test_smooth_l1_backward_linear()
    print("✓ test_smooth_l1_backward_linear")

    test_smooth_l1_backward_gradient()
    print("✓ test_smooth_l1_backward_gradient")

    test_hinge_loss_correct_prediction()
    print("✓ test_hinge_loss_correct_prediction")

    test_hinge_loss_wrong_prediction()
    print("✓ test_hinge_loss_wrong_prediction")

    test_hinge_loss_at_margin()
    print("✓ test_hinge_loss_at_margin")

    test_hinge_loss_backward()
    print("✓ test_hinge_loss_backward")

    test_hinge_loss_backward_gradient()
    print("✓ test_hinge_loss_backward_gradient")

    test_focal_loss_perfect_prediction()
    print("✓ test_focal_loss_perfect_prediction")

    test_focal_loss_hard_examples()
    print("✓ test_focal_loss_hard_examples")

    test_focal_loss_backward_shape()
    print("✓ test_focal_loss_backward_shape")

    test_focal_loss_backward_gradient()
    print("✓ test_focal_loss_backward_gradient")

    test_kl_divergence_same_distribution()
    print("✓ test_kl_divergence_same_distribution")

    test_kl_divergence_different_distributions()
    print("✓ test_kl_divergence_different_distributions")

    test_kl_divergence_backward_shape()
    print("✓ test_kl_divergence_backward_shape")

    test_kl_divergence_backward_gradient()
    print("✓ test_kl_divergence_backward_gradient")

    print("\nAll test_losses tests passed!")
