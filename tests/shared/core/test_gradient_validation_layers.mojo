# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

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

from shared.core.activation import tanh, gelu
from shared.core.activation import tanh_backward, gelu_backward
from shared.core.conv import conv2d, conv2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.extensor import ExTensor, zeros
from shared.core.initializers import kaiming_uniform
from shared.testing.gradient_checker import check_gradients
from shared.testing.special_values import create_seeded_random_tensor
from shared.testing.assertions import assert_true


fn test_tanh_gradient() raises:
    """Test Tanh gradient.

    Note: tanh_backward takes output (tanh(x)), not input x.
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return tanh(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var output = tanh(inp)  # Compute tanh(x) first
        return tanh_backward(grad_out, output)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "Tanh gradient check failed")


fn test_gelu_gradient() raises:
    """Test GELU gradient.

    Note: gelu_backward takes input x (not output).
    """
    var x = create_seeded_random_tensor(
        [2, 3], DType.float32, seed=42, low=-2.0, high=2.0
    )

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return gelu(inp)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        return gelu_backward(grad_out, inp)

    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "GELU gradient check failed")


fn test_conv2d_gradient_input() raises:
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

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=1)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var result = conv2d_backward(grad_out, inp, kernel, stride=1, padding=1)
        return result.grad_input

    # Use slightly larger epsilon for conv (more complex operation)
    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-4, tolerance=1e-2
    )
    assert_true(passed, "Conv2D input gradient check failed")


fn test_linear_gradient_input() raises:
    """Test Linear gradient w.r.t. input.

    Note: Slightly wider tolerance due to accumulated numerical errors in matrix operations.
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

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return linear(inp, weights, bias)

    fn backward_fn(
        grad_out: ExTensor, inp: ExTensor
    ) raises escaping -> ExTensor:
        var result = linear_backward(grad_out, inp, weights)
        return result.grad_input

    # Wider tolerance (1.5%) for matrix operations
    var passed = check_gradients(
        forward, backward_fn, x, epsilon=1e-5, tolerance=0.015
    )
    assert_true(passed, "Linear input gradient check failed")


fn main() raises:
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
