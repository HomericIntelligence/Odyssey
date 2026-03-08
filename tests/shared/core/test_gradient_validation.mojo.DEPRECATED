"""Gradient validation tests for backward passes.

Systematically validates analytical gradients against numerical gradients
using finite differences. Ensures backward implementations are mathematically correct.

Test Coverage:
- Activation functions: ReLU, Sigmoid, Tanh, GELU
- Parametric layers: Conv2D, Linear
- Edge cases: Saturation, boundary values, large values

All tests use small tensors (2×3, 8×8) to ensure fast runtime (<10 seconds total).

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #2644: Add Numerical Stability Tests for Gradients
"""

from shared.core.activation import relu, sigmoid, tanh, gelu
from shared.core.activation import (
    relu_backward,
    sigmoid_backward,
    tanh_backward,
    gelu_backward,
)
from shared.core.conv import conv2d, conv2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.extensor import ExTensor, zeros, ones_like, full
from shared.core.initializers import kaiming_uniform
from shared.testing.gradient_checker import check_gradients
from shared.testing.special_values import (
    create_seeded_random_tensor,
    create_special_value_tensor,
)
from shared.testing.assertions import assert_true


# ============================================================================
# Activation Function Gradients
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


# ============================================================================
# Parametric Layer Gradients
# ============================================================================


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


# ============================================================================
# Main Test Function
# ============================================================================


fn main() raises:
    """Run all gradient validation tests."""
    print("Running Gradient Validation Tests...")
    print("=" * 60)

    # Activation function tests
    print("\n[1/13] Testing ReLU gradient (positive values)...")
    test_relu_gradient_positive_values()
    print("✓ PASSED")

    print("[2/13] Testing ReLU gradient (negative values)...")
    test_relu_gradient_negative_values()
    print("✓ PASSED")

    print("[3/13] Testing ReLU gradient (mixed values)...")
    test_relu_gradient_mixed_values()
    print("✓ PASSED")

    print("[4/13] Testing ReLU gradient (near zero)...")
    test_relu_gradient_near_zero()
    print("✓ PASSED")

    print("[5/13] Testing ReLU gradient (large values)...")
    test_relu_gradient_large_values()
    print("✓ PASSED")

    print("[6/13] Testing Sigmoid gradient (normal range)...")
    test_sigmoid_gradient_normal_range()
    print("✓ PASSED")

    print("[7/13] Testing Sigmoid gradient (positive saturation)...")
    test_sigmoid_gradient_saturation_positive()
    print("✓ PASSED")

    print("[8/13] Testing Sigmoid gradient (negative saturation)...")
    test_sigmoid_gradient_saturation_negative()
    print("✓ PASSED")

    print("[9/13] Testing Tanh gradient...")
    test_tanh_gradient()
    print("✓ PASSED")

    print("[10/13] Testing GELU gradient...")
    test_gelu_gradient()
    print("✓ PASSED")

    # Parametric layer tests
    print("[11/13] Testing Conv2D gradient (input)...")
    test_conv2d_gradient_input()
    print("✓ PASSED")

    print("[12/13] Testing Linear gradient (input)...")
    test_linear_gradient_input()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 13 gradient validation tests PASSED! ✓")
    print("Analytical gradients match numerical gradients within tolerance.")
