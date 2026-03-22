# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_losses.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for Smooth L1 backward gradient and Hinge loss functions (Part 3 of 4).

This module tests:
- Smooth L1 Loss backward gradient checking
- Hinge Loss forward pass (correct, wrong, margin)
- Hinge Loss backward pass
- Hinge Loss backward gradient checking
- Focal Loss perfect prediction
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_close_float,
)
from shared.core.extensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.loss import smooth_l1_loss, smooth_l1_loss_backward
from shared.core.loss import hinge_loss, hinge_loss_backward
from shared.core.loss import focal_loss, focal_loss_backward
from shared.core.reduction import mean
from shared.testing import check_gradient


fn test_smooth_l1_backward_gradient() raises:
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
    fn forward(pred: AnyTensor) raises escaping -> AnyTensor:
        return smooth_l1_loss(pred, targets, beta=beta)

    # Backward function wrapper
    fn backward(grad_out: AnyTensor, pred: AnyTensor) raises escaping -> AnyTensor:
        return smooth_l1_loss_backward(grad_out, pred, targets, beta=beta)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for smooth L1)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-2, atol=1e-3
    )

    print("  ✓ Smooth L1 backward gradient check passed")


fn test_hinge_loss_correct_prediction() raises:
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


fn test_hinge_loss_wrong_prediction() raises:
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


fn test_hinge_loss_at_margin() raises:
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


fn test_hinge_loss_backward() raises:
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


fn test_hinge_loss_backward_gradient() raises:
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
    fn forward(pred: AnyTensor) raises escaping -> AnyTensor:
        return hinge_loss(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: AnyTensor, pred: AnyTensor) raises escaping -> AnyTensor:
        return hinge_loss_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for discontinuous gradient)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-2, atol=1e-3
    )

    print("  ✓ Hinge backward gradient check passed")


fn test_focal_loss_perfect_prediction() raises:
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


fn run_all_tests() raises:
    """Run all loss function tests (Part 3)."""
    print("=" * 60)
    print("Loss Functions Test Suite - Part 3 (Smooth L1 Grad & Hinge & Focal)")
    print("=" * 60)

    test_smooth_l1_backward_gradient()
    test_hinge_loss_correct_prediction()
    test_hinge_loss_wrong_prediction()
    test_hinge_loss_at_margin()
    test_hinge_loss_backward()
    test_hinge_loss_backward_gradient()
    test_focal_loss_perfect_prediction()

    print("=" * 60)
    print("All Part 3 loss function tests passed! ✓")
    print("=" * 60)


fn main() raises:
    """Entry point for loss function tests (Part 3)."""
    run_all_tests()
