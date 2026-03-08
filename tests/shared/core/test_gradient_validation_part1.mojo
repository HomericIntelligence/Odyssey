# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Gradient validation tests for activation function backward passes (Part 1).

Systematically validates analytical gradients against numerical gradients
using finite differences. Ensures backward implementations are mathematically correct.

Test Coverage:
- Activation functions: ReLU (5 cases), Sigmoid (3 cases)

All tests use small tensors (2×3) to ensure fast runtime (<10 seconds total).

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #2644: Add Numerical Stability Tests for Gradients
    - Issue #3626: Split test_gradient_validation.mojo per ADR-009
"""

from shared.core.activation import relu, sigmoid
from shared.core.activation import relu_backward, sigmoid_backward
from shared.core.extensor import ExTensor, full
from shared.testing.gradient_checker import check_gradients
from shared.testing.special_values import (
    create_seeded_random_tensor,
)
from shared.testing.assertions import assert_true


# ============================================================================
# ReLU Gradient Tests
# ============================================================================


fn test_relu_gradient_positive_values() raises:
    """Test ReLU gradient with positive inputs (gradient should be 1)."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=0.1, high=2.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return relu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad_out, inp)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "ReLU gradient check failed for positive values")


fn test_relu_gradient_negative_values() raises:
    """Test ReLU gradient with negative inputs (gradient should be 0)."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=123, low=-2.0, high=-0.1
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return relu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad_out, inp)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "ReLU gradient check failed for negative values")


fn test_relu_gradient_mixed_values() raises:
    """Test ReLU gradient with mixed positive/negative inputs."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=999, low=-1.0, high=1.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return relu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad_out, inp)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "ReLU gradient check failed for mixed values")


fn test_relu_gradient_near_zero() raises:
    """Test ReLU gradient near zero (boundary region).

    Note: ReLU is not differentiable exactly at x=0 (corner point).
    Numerical gradient gives 0.5 (average of left/right limits).
    We test very close to zero instead to avoid this discontinuity.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=555, low=-0.01, high=0.01
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return relu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad_out, inp)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "ReLU gradient check failed near zero")


fn test_relu_gradient_large_values() raises:
    """Test ReLU gradient with moderately large positive values.

    Gradient should still be 1.0 (ReLU is linear for x > 0).
    Using realistic neural network activation values (10-20).
    Note: Wider tolerance due to numerical precision with larger values.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=10.0, high=20.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return relu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad_out, inp)

    # Use wider tolerance (5%) for large values due to numerical precision
    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=0.05
    )
    assert_true(passed, "ReLU gradient check failed for large values")


# ============================================================================
# Sigmoid Gradient Tests
# ============================================================================


fn test_sigmoid_gradient_normal_range() raises:
    """Test Sigmoid gradient in normal range (-2 to 2).

    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return sigmoid(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var output = sigmoid(inp)  # Compute sigmoid(x) first
        return sigmoid_backward(grad_out, output)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "Sigmoid gradient check failed")


fn test_sigmoid_gradient_saturation_positive() raises:
    """Test sigmoid gradient in saturation region (x >> 0).

    At x = 10.0, sigmoid(x) ≈ 1.0, gradient ≈ 0.0.
    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 10.0, DType.float32)

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return sigmoid(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var output = sigmoid(inp)  # Compute sigmoid(x) first
        return sigmoid_backward(grad_out, output)

    # Use tighter tolerance for near-zero gradients
    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-4, tolerance=1e-3
    )
    assert_true(passed, "Sigmoid gradient check failed in positive saturation")


fn test_sigmoid_gradient_saturation_negative() raises:
    """Test sigmoid gradient in saturation region (x << 0).

    At x = -10.0, sigmoid(x) ≈ 0.0, gradient ≈ 0.0.
    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, -10.0, DType.float32)

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return sigmoid(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var output = sigmoid(inp)  # Compute sigmoid(x) first
        return sigmoid_backward(grad_out, output)

    # Use tighter tolerance for near-zero gradients
    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-4, tolerance=1e-3
    )
    assert_true(passed, "Sigmoid gradient check failed in negative saturation")


# ============================================================================
# Main Test Function
# ============================================================================


fn main() raises:
    """Run all gradient validation tests (Part 1: ReLU and Sigmoid)."""
    print("Running Gradient Validation Tests - Part 1...")
    print("=" * 60)

    # ReLU tests
    print("\n[1/8] Testing ReLU gradient (positive values)...")
    test_relu_gradient_positive_values()
    print("✓ PASSED")

    print("[2/8] Testing ReLU gradient (negative values)...")
    test_relu_gradient_negative_values()
    print("✓ PASSED")

    print("[3/8] Testing ReLU gradient (mixed values)...")
    test_relu_gradient_mixed_values()
    print("✓ PASSED")

    print("[4/8] Testing ReLU gradient (near zero)...")
    test_relu_gradient_near_zero()
    print("✓ PASSED")

    print("[5/8] Testing ReLU gradient (large values)...")
    test_relu_gradient_large_values()
    print("✓ PASSED")

    # Sigmoid tests
    print("[6/8] Testing Sigmoid gradient (normal range)...")
    test_sigmoid_gradient_normal_range()
    print("✓ PASSED")

    print("[7/8] Testing Sigmoid gradient (positive saturation)...")
    test_sigmoid_gradient_saturation_positive()
    print("✓ PASSED")

    print("[8/8] Testing Sigmoid gradient (negative saturation)...")
    test_sigmoid_gradient_saturation_negative()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 8 gradient validation tests (Part 1) PASSED! ✓")
    print("Analytical gradients match numerical gradients within tolerance.")
