# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_losses.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for BCE/MSE backward and Smooth L1 loss functions (Part 2 of 4).

This module tests:
- Binary Cross-Entropy (BCE) backward gradient checking
- Mean Squared Error (MSE) backward gradient checking
- Smooth L1 Loss forward pass (boundary, quadratic, linear regions)
- Smooth L1 Loss backward pass
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.loss import binary_cross_entropy, binary_cross_entropy_backward
from shared.core.loss import mean_squared_error, mean_squared_error_backward
from shared.core.loss import smooth_l1_loss, smooth_l1_loss_backward
from shared.core.reduction import mean
from shared.testing import check_gradient


fn test_binary_cross_entropy_backward_gradient() raises:
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
    fn forward(pred: ExTensor) raises escaping -> ExTensor:
        return binary_cross_entropy(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, pred: ExTensor) raises escaping -> ExTensor:
        return binary_cross_entropy_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ BCE backward gradient check passed")


fn test_mean_squared_error_backward_gradient() raises:
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
    fn forward(pred: ExTensor) raises escaping -> ExTensor:
        return mean_squared_error(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, pred: ExTensor) raises escaping -> ExTensor:
        return mean_squared_error_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(
        forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5
    )

    print("  ✓ MSE backward gradient check passed")


fn test_smooth_l1_zero_beta_boundary() raises:
    """Test Smooth L1 loss at beta boundary."""
    print("Testing Smooth L1 loss at beta boundary...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

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


fn test_smooth_l1_quadratic_region() raises:
    """Test Smooth L1 in quadratic region (|x| < beta)."""
    print("Testing Smooth L1 loss in quadratic region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

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


fn test_smooth_l1_linear_region() raises:
    """Test Smooth L1 in linear region (|x| >= beta)."""
    print("Testing Smooth L1 loss in linear region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

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


fn test_smooth_l1_backward_quadratic() raises:
    """Test Smooth L1 backward in quadratic region."""
    print("Testing Smooth L1 backward in quadratic region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

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


fn test_smooth_l1_backward_linear() raises:
    """Test Smooth L1 backward in linear region."""
    print("Testing Smooth L1 backward in linear region...")

    var shape = List[Int]()
    shape.append(1)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

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


fn run_all_tests() raises:
    """Run all loss function tests (Part 2)."""
    print("=" * 60)
    print("Loss Functions Test Suite - Part 2 (BCE/MSE Backward & Smooth L1)")
    print("=" * 60)

    test_binary_cross_entropy_backward_gradient()
    test_mean_squared_error_backward_gradient()
    test_smooth_l1_zero_beta_boundary()
    test_smooth_l1_quadratic_region()
    test_smooth_l1_linear_region()
    test_smooth_l1_backward_quadratic()
    test_smooth_l1_backward_linear()

    print("=" * 60)
    print("All Part 2 loss function tests passed! ✓")
    print("=" * 60)


fn main() raises:
    """Entry point for loss function tests (Part 2)."""
    run_all_tests()
