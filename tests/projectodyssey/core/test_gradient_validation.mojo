"""Gradient validation tests for activation function and parametric layer backward passes.

Systematically validates analytical gradients against numerical gradients
using finite differences. Ensures backward implementations are mathematically correct.

Test Coverage:
- Activation functions: ReLU (5 cases), Sigmoid (3 cases), Tanh, GELU
- Parametric layers: Conv2D, Linear

All tests use small tensors (2×3, 8×8) to ensure fast runtime (<10 seconds total).

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #2644: Add Numerical Stability Tests for Gradients
"""

from odyssey.core.activation import (
    gelu,
    gelu_backward,
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
)
from odyssey.core.conv import conv2d, conv2d_backward
from odyssey.core.linear import linear, linear_backward
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full, zeros, zeros_like
from odyssey.core.initializers import kaiming_uniform
from odyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from odyssey.testing.special_values import (
    create_seeded_random_tensor,
)


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


# ---- Tanh (no captures) ----


@fieldwise_init
struct _TanhFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return tanh(inp)


@fieldwise_init
struct _TanhBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var output = tanh(inp)  # Compute tanh(x) first
        return tanh_backward(grad_out, output)


# ---- GELU (no captures) ----


@fieldwise_init
struct _GeluFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return gelu(inp)


@fieldwise_init
struct _GeluBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return gelu_backward(grad_out, inp)


# ---- Conv2D input gradient (captures kernel, bias) ----


@fieldwise_init
struct _Conv2dInputFwd(NumericalForward):
    var kernel: AnyTensor
    var bias: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return conv2d(inp, self.kernel, self.bias, stride=1, padding=1)


@fieldwise_init
struct _Conv2dInputBwd(NumericalBackward):
    var kernel: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var result = conv2d_backward(
            grad_out, inp, self.kernel, stride=1, padding=1
        )
        return result.grad_input


# ---- Linear input gradient (captures weights, bias) ----


@fieldwise_init
struct _LinearFwd(NumericalForward):
    var weights: AnyTensor
    var bias: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return linear(inp, self.weights, self.bias)


@fieldwise_init
struct _LinearBwd(NumericalBackward):
    var weights: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var result = linear_backward(grad_out, inp, self.weights)
        return result.grad_input


# ============================================================================
# ReLU Gradient Tests
# ============================================================================


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


# ============================================================================
# Sigmoid Gradient Tests
# ============================================================================


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


# ============================================================================
# Activation Function Gradients (Part 2)
# ============================================================================


def test_tanh_gradient() raises:
    """Test Tanh gradient.

    Note: tanh_backward takes output (tanh(x)), not input x.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )
    var fwd = _TanhFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _TanhBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


def test_gelu_gradient() raises:
    """Test GELU gradient.

    Note: gelu_backward takes input x (not output).
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )
    var fwd = _GeluFwd()
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(fwd, _GeluBwd(), x, grad_output, rtol=1e-2, atol=1e-2)


# ============================================================================
# Parametric Layer Gradients
# ============================================================================


def test_conv2d_gradient_input() raises:
    """Test Conv2D gradient w.r.t. input."""
    # Create small conv layer: 3 input channels, 8 output channels, 3x3 kernel
    var in_channels = 3
    var out_channels = 8
    var kernel_size = 3

    # Create kernel and bias
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(
        fan_in, fan_out, kernel_shape, dtype=DType.float32
    )

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Create small input: batch=1, 3 channels, 8x8 image
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_channels)
    input_shape.append(8)
    input_shape.append(8)
    var x = create_seeded_random_tensor(input_shape, DType.float32, seed=42)

    var fwd = _Conv2dInputFwd(kernel, bias)
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(
        fwd, _Conv2dInputBwd(kernel), x, grad_output, rtol=1e-2, atol=1e-2
    )


def test_linear_gradient_input() raises:
    """Test Linear gradient w.r.t. input.

    Note: Combined relative+absolute tolerance handles accumulated matrix op errors.
    """
    # Create small linear layer: 16 input features, 10 output features
    var in_features = 16
    var out_features = 10

    # Create weights and bias
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=DType.float32
    )

    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Create small input: batch=2, 16 features
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(in_features)
    var x = create_seeded_random_tensor(input_shape, DType.float32, seed=42)

    var fwd = _LinearFwd(weights, bias)
    var grad_output = zeros_like(fwd(x))
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    check_gradient(
        fwd, _LinearBwd(weights), x, grad_output, rtol=1.5e-2, atol=1e-4
    )


# ============================================================================
# Main Test Function
# ============================================================================


def main() raises:
    """Run all gradient validation tests."""
    print("Running Gradient Validation Tests...")
    print("=" * 60)

    # ReLU tests
    print("\n[1/12] Testing ReLU gradient (positive values)...")
    test_relu_gradient_positive_values()
    print("✓ PASSED")

    print("[2/12] Testing ReLU gradient (negative values)...")
    test_relu_gradient_negative_values()
    print("✓ PASSED")

    print("[3/12] Testing ReLU gradient (mixed values)...")
    test_relu_gradient_mixed_values()
    print("✓ PASSED")

    print("[4/12] Testing ReLU gradient (near zero)...")
    test_relu_gradient_near_zero()
    print("✓ PASSED")

    print("[5/12] Testing ReLU gradient (large values)...")
    test_relu_gradient_large_values()
    print("✓ PASSED")

    # Sigmoid tests
    print("[6/12] Testing Sigmoid gradient (normal range)...")
    test_sigmoid_gradient_normal_range()
    print("✓ PASSED")

    print("[7/12] Testing Sigmoid gradient (positive saturation)...")
    test_sigmoid_gradient_saturation_positive()
    print("✓ PASSED")

    print("[8/12] Testing Sigmoid gradient (negative saturation)...")
    test_sigmoid_gradient_saturation_negative()
    print("✓ PASSED")

    # Activation tests (Part 2)
    print("[9/12] Testing Tanh gradient...")
    test_tanh_gradient()
    print("✓ PASSED")

    print("[10/12] Testing GELU gradient...")
    test_gelu_gradient()
    print("✓ PASSED")

    # Parametric layer tests
    print("[11/12] Testing Conv2D gradient (input)...")
    test_conv2d_gradient_input()
    print("✓ PASSED")

    print("[12/12] Testing Linear gradient (input)...")
    test_linear_gradient_input()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 12 gradient validation tests PASSED! ✓")
    print("Analytical gradients match numerical gradients within tolerance.")
