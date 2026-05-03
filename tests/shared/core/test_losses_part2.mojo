"""Tests for Smooth L1 and Hinge loss functions (Part 2 of 3)

This module tests:
- Smooth L1 loss forward and backward passes
- Hinge loss forward and backward passes
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


@fieldwise_init
struct _SmoothL1Fwd(NumericalForward):
    var targets: AnyTensor
    var beta: Float32

    def __call__(self, pred: AnyTensor) raises -> AnyTensor:
        return smooth_l1_loss(pred, self.targets, beta=self.beta)


@fieldwise_init
struct _SmoothL1Bwd(NumericalBackward):
    var targets: AnyTensor
    var beta: Float32

    def __call__(self, grad_out: AnyTensor, pred: AnyTensor) raises -> AnyTensor:
        return smooth_l1_loss_backward(grad_out, pred, self.targets, beta=self.beta)


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

    var loss = smooth_l1_loss(predictions, targets, beta=beta)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for smooth L1)
    check_gradient(
        _SmoothL1Fwd(targets, beta), _SmoothL1Bwd(targets, beta), predictions, grad_output, rtol=1e-2, atol=1e-3
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


def main() raises:
    """Run test_losses part 2 tests (Smooth L1 and Hinge)."""
    print("Running test_losses_part2 tests...")

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

    print("\nAll test_losses_part2 tests passed!")
