"""Gradient validation tests for Tanh, GELU, Conv2D, and Linear backward passes.

Systematically validates analytical gradients against numerical gradients
using finite differences. Ensures backward implementations are mathematically correct.

Test Coverage:
- Tanh and GELU activation functions
- Parametric layers: Conv2D, Linear

All tests use small tensors (2×3, 8×8) to ensure fast runtime.

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #2644: Add Numerical Stability Tests for Gradients
"""

from odyssey.core.activation import (
    gelu,
    gelu_backward,
    tanh,
    tanh_backward,
)
from odyssey.core.conv import conv2d, conv2d_backward
from odyssey.core.linear import linear, linear_backward
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, zeros_like
from odyssey.core.initializers import kaiming_uniform
from odyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from odyssey.testing.special_values import create_seeded_random_tensor


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


def main() raises:
    """Run Tanh, GELU, Conv2D, and Linear gradient validation tests."""
    print("Running Layer Gradient Validation Tests...")
    print("=" * 60)

    print("[1/4] Testing Tanh gradient...")
    test_tanh_gradient()
    print("✓ PASSED")

    print("[2/4] Testing GELU gradient...")
    test_gelu_gradient()
    print("✓ PASSED")

    print("[3/4] Testing Conv2D gradient (input)...")
    test_conv2d_gradient_input()
    print("✓ PASSED")

    print("[4/4] Testing Linear gradient (input)...")
    test_linear_gradient_input()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 4 layer gradient validation tests PASSED! ✓")
    print("Analytical gradients match numerical gradients within tolerance.")
