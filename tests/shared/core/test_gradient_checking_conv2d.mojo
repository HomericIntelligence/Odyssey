# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Gradient checking tests for Conv2D backward pass outputs.

Validates analytical gradients against numerical gradients for all three
outputs of conv2d_backward: grad_input, grad_kernel (grad_weights), and grad_bias.

Test Coverage:
- grad_input: Input gradient (already tested in test_gradient_validation_part2.mojo)
- grad_kernel: Kernel/weight gradient
- grad_bias: Bias gradient

All tests use small tensors to ensure fast runtime (<10 seconds total).
Each test checks a single Conv2D output dimension to stay under ADR-009 limit.

References:
    - CS231n Gradient Checking: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Issue #3774: Add gradient checking for grad_kernel and grad_bias
"""

from shared.core.conv import conv2d, conv2d_backward
from shared.core.extensor import ExTensor, zeros
from shared.core.initializers import kaiming_uniform
from shared.testing.gradient_checker import check_gradients
from shared.testing.special_values import create_seeded_random_tensor
from shared.testing.assertions import assert_true


# ============================================================================
# Conv2D Kernel Gradient Checking
# ============================================================================


fn test_conv2d_gradient_kernel_basic() raises:
    """Test Conv2D gradient w.r.t. kernel (basic 3x3 same-padding).

    Validates grad_kernel output from conv2d_backward using finite differences.
    Uses small 8x8 input with 3 input channels and 2 output channels.
    """
    var in_channels = 3
    var out_channels = 2
    var kernel_size = 3

    # Create kernel
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

    # For gradient w.r.t. kernel, we treat kernel as variable and input/bias as fixed
    fn forward(k: ExTensor) raises escaping -> ExTensor:
        return conv2d(x, k, bias, stride=1, padding=1)

    fn backward_fn(
        grad_out: ExTensor, k: ExTensor
    ) raises escaping -> ExTensor:
        var result = conv2d_backward(grad_out, x, k, stride=1, padding=1)
        return result.grad_kernel

    # Use larger epsilon for high-order operations
    var passed = check_gradients(
        forward, backward_fn, kernel, epsilon=1e-4, tolerance=1e-2
    )
    assert_true(passed, "Conv2D kernel gradient check failed")


fn test_conv2d_gradient_bias_basic() raises:
    """Test Conv2D gradient w.r.t. bias (basic 3x3 same-padding).

    Validates grad_bias output from conv2d_backward using finite differences.
    Uses small 8x8 input with 3 input channels and 2 output channels.
    """
    var in_channels = 3
    var out_channels = 2
    var kernel_size = 3

    # Create kernel
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

    # For gradient w.r.t. bias, we treat bias as variable and input/kernel as fixed
    fn forward(b: ExTensor) raises escaping -> ExTensor:
        return conv2d(x, kernel, b, stride=1, padding=1)

    fn backward_fn(
        grad_out: ExTensor, b: ExTensor
    ) raises escaping -> ExTensor:
        var result = conv2d_backward(grad_out, x, kernel, stride=1, padding=1)
        return result.grad_bias

    # Bias gradient should be more stable
    var passed = check_gradients(
        forward, backward_fn, bias, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "Conv2D bias gradient check failed")


fn test_conv2d_gradient_kernel_strided() raises:
    """Test Conv2D kernel gradient with stride=2.

    Validates grad_kernel with strided convolution to ensure gradient
    computation handles non-unit strides correctly.
    """
    var in_channels = 2
    var out_channels = 2
    var kernel_size = 3

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

    # Create input: batch=1, 2 channels, 8x8 image
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_channels)
    input_shape.append(8)
    input_shape.append(8)
    var x = create_seeded_random_tensor(input_shape, DType.float32, seed=42)

    fn forward(k: ExTensor) raises escaping -> ExTensor:
        return conv2d(x, k, bias, stride=2, padding=1)

    fn backward_fn(
        grad_out: ExTensor, k: ExTensor
    ) raises escaping -> ExTensor:
        var result = conv2d_backward(grad_out, x, k, stride=2, padding=1)
        return result.grad_kernel

    var passed = check_gradients(
        forward, backward_fn, kernel, epsilon=1e-4, tolerance=0.015
    )
    assert_true(passed, "Conv2D strided kernel gradient check failed")


# ============================================================================
# Main Test Function
# ============================================================================


fn main() raises:
    """Run Conv2D gradient checking tests (kernel and bias)."""
    print("Running Conv2D Gradient Checking Tests...")
    print("=" * 60)

    print("\n[1/3] Testing Conv2D kernel gradient (basic)...")
    test_conv2d_gradient_kernel_basic()
    print("✓ PASSED")

    print("[2/3] Testing Conv2D bias gradient (basic)...")
    test_conv2d_gradient_bias_basic()
    print("✓ PASSED")

    print("[3/3] Testing Conv2D kernel gradient (strided)...")
    test_conv2d_gradient_kernel_strided()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 3 Conv2D gradient checking tests PASSED! ✓")
    print("Kernel and bias gradients match numerical gradients within tolerance.")
