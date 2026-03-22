# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_losses.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for Focal Loss and KL Divergence functions (Part 4 of 4).

This module tests:
- Focal Loss hard examples emphasis
- Focal Loss backward pass (shape, gradient checking)
- KL Divergence forward pass (same/different distributions)
- KL Divergence backward pass (shape, gradient checking)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_close_float,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.loss import focal_loss, focal_loss_backward
from shared.core.loss import kl_divergence, kl_divergence_backward
from shared.core.reduction import mean
from shared.testing import check_gradient


fn test_focal_loss_hard_examples() raises:
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


fn test_focal_loss_backward_shape() raises:
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


fn test_focal_loss_backward_gradient() raises:
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
    fn forward(pred: AnyTensor) raises escaping -> AnyTensor:
        return focal_loss(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: AnyTensor, pred: AnyTensor) raises escaping -> AnyTensor:
        return focal_loss_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-2, atol=1e-4
    )

    print("  ✓ Focal loss backward gradient check passed")


fn test_kl_divergence_same_distribution() raises:
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


fn test_kl_divergence_different_distributions() raises:
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


fn test_kl_divergence_backward_shape() raises:
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


fn test_kl_divergence_backward_gradient() raises:
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
    fn forward(q_dist: AnyTensor) raises escaping -> AnyTensor:
        return kl_divergence(p, q_dist)

    # Backward function wrapper
    fn backward(
        grad_out: AnyTensor, q_dist: AnyTensor
    ) raises escaping -> AnyTensor:
        return kl_divergence_backward(grad_out, p, q_dist)

    var kl = forward(q)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, q, grad_output, rtol=2e-2, atol=1e-4)

    print("  ✓ KL divergence backward gradient check passed")


fn run_all_tests() raises:
    """Run all loss function tests (Part 4)."""
    print("=" * 60)
    print("Loss Functions Test Suite - Part 4 (Focal Loss & KL Divergence)")
    print("=" * 60)

    test_focal_loss_hard_examples()
    test_focal_loss_backward_shape()
    test_focal_loss_backward_gradient()
    test_kl_divergence_same_distribution()
    test_kl_divergence_different_distributions()
    test_kl_divergence_backward_shape()
    test_kl_divergence_backward_gradient()

    print("=" * 60)
    print("All Part 4 loss function tests passed! ✓")
    print("=" * 60)


fn main() raises:
    """Entry point for loss function tests (Part 4)."""
    run_all_tests()
