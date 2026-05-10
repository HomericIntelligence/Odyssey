"""Gradient validation tests for ReLU and Sigmoid activation backward passes.

Systematically validates analytical gradients against numerical gradients
using finite differences. Ensures backward implementations are mathematically correct.

Test Coverage:
- ReLU: positive, negative, mixed, near-zero, large values
- Sigmoid: normal range, positive saturation, negative saturation

All tests use small tensors (2×3) to ensure fast runtime.

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #2644: Add Numerical Stability Tests for Gradients
"""

from shared.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
)
from shared.tensor.any_tensor import AnyTensor, full, zeros_like
from shared.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from shared.testing.special_values import create_seeded_random_tensor


# ---- ReLU (no captures) ----


@fieldwise_init
struct _ReluFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return relu(inp)


@fieldwise_init
struct _ReluBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return relu_backward(grad_out, inp)


# ---- Sigmoid (no captures) ----


@fieldwise_init
struct _SigmoidFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return sigmoid(inp)


@fieldwise_init
struct _SigmoidBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var output = sigmoid(inp)  # Compute sigmoid(x) first
        return sigmoid_backward(grad_out, output)


def test_relu_gradient_positive_values() raises:
    """Test ReLU gradient with positive inputs (gradient should be 1)."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=0.1, high=2.0
    )
    var fwd = _ReluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _ReluBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_relu_gradient_negative_values() raises:
    """Test ReLU gradient with negative inputs (gradient should be 0)."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=123, low=-2.0, high=-0.1
    )
    var fwd = _ReluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _ReluBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_relu_gradient_mixed_values() raises:
    """Test ReLU gradient with mixed positive/negative inputs."""
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=999, low=-1.0, high=1.0
    )
    var fwd = _ReluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _ReluBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_relu_gradient_near_zero() raises:
    """Test ReLU gradient near zero (boundary region).

    Note: ReLU is not differentiable exactly at x=0 (corner point).
    Numerical gradient gives 0.5 (average of left/right limits).
    We test very close to zero instead to avoid this discontinuity.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=555, low=-0.01, high=0.01
    )
    var fwd = _ReluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _ReluBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_relu_gradient_large_values() raises:
    """Test ReLU gradient with moderately large positive values.

    Gradient should still be 1.0 (ReLU is linear for x > 0).
    Using realistic neural network activation values (10-20).
    Combined relative+absolute tolerance handles large-magnitude gradients correctly.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=10.0, high=20.0
    )
    var fwd = _ReluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _ReluBwd(), x, grad_output, rtol=5e-2, atol=1e-4)


def test_sigmoid_gradient_normal_range() raises:
    """Test Sigmoid gradient in normal range (-2 to 2).

    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )
    var fwd = _SigmoidFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _SigmoidBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_sigmoid_gradient_saturation_positive() raises:
    """Test sigmoid gradient in saturation region (x >> 0).

    At x = 10.0, sigmoid(x) ≈ 1.0, gradient ≈ 0.0.
    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 10.0, DType.float32)
    var fwd = _SigmoidFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    # Near-zero gradients: use combined tolerance with small atol floor
    check_gradient(fwd, _SigmoidBwd(), x, grad_output, rtol=1e-2, atol=1e-3)


def test_sigmoid_gradient_saturation_negative() raises:
    """Test sigmoid gradient in saturation region (x << 0).

    At x = -10.0, sigmoid(x) ≈ 0.0, gradient ≈ 0.0.
    Note: sigmoid_backward takes output (sigmoid(x)), not input x.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, -10.0, DType.float32)
    var fwd = _SigmoidFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    # Near-zero gradients: use combined tolerance with small atol floor
    check_gradient(fwd, _SigmoidBwd(), x, grad_output, rtol=1e-2, atol=1e-3)


def main() raises:
    """Run ReLU and Sigmoid gradient validation tests."""
    print("Running ReLU Gradient Validation Tests...")

    print("[1/8] Testing ReLU gradient (positive values)...")
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

    print("Running Sigmoid Gradient Validation Tests...")

    print("[6/8] Testing Sigmoid gradient (normal range)...")
    test_sigmoid_gradient_normal_range()
    print("✓ PASSED")

    print("[7/8] Testing Sigmoid gradient (positive saturation)...")
    test_sigmoid_gradient_saturation_positive()
    print("✓ PASSED")

    print("[8/8] Testing Sigmoid gradient (negative saturation)...")
    test_sigmoid_gradient_saturation_negative()
    print("✓ PASSED")

    print("All 8 activation gradient validation tests PASSED! ✓")
