# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checking.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Gradient checking tests for depthwise_conv2d backward pass.

Uses check_gradient() (not check_gradients()) for combined relative+absolute
tolerance, and non-uniform input values to avoid degenerate gradient patterns.
See issue #3282 and the depthwise-conv2d-gradient-checking skill for rationale.

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.testing.gradient_checker import check_gradient
from shared.tensor.any_tensor import AnyTensor, zeros, zeros_like
from shared.core.conv import depthwise_conv2d, depthwise_conv2d_backward


fn _make_ones_grad_output(output: AnyTensor) raises -> AnyTensor:
    """Create ones grad_output matching output shape."""
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)
    return grad_output^


fn test_depthwise_conv2d_gradient_kernel_basic() raises:
    """Test depthwise_conv2d kernel gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = zeros(shape, DType.float32)
    for i in range(input.numel()):
        input.set(i, Float32(Float32(i) * Float32(0.1)))

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(kernel.numel()):
        kernel.set(i, Float32(Float32(i) * Float32(0.05) + Float32(0.1)))

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = zeros(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, k, bias, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, k, stride=1, padding=1)
        return result.grad_weights

    var output = forward(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(forward, backward, kernel, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_gradient_bias_basic() raises:
    """Test depthwise_conv2d bias gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = zeros(shape, DType.float32)
    for i in range(input.numel()):
        input.set(i, Float32(Float32(i) * Float32(0.1)))

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(kernel.numel()):
        kernel.set(i, Float32(Float32(i) * Float32(0.05) + Float32(0.1)))

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = zeros(bias_shape, DType.float32)

    fn forward(b: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, kernel, b, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, b: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, kernel, stride=1, padding=1)
        return result.grad_bias

    var output = forward(bias)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(forward, backward, bias, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_gradient_input_basic() raises:
    """Test depthwise_conv2d input gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = zeros(shape, DType.float32)
    for i in range(input.numel()):
        input.set(i, Float32(Float32(i) * Float32(0.1)))

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(kernel.numel()):
        kernel.set(i, Float32(Float32(i) * Float32(0.05) + Float32(0.1)))

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = zeros(bias_shape, DType.float32)

    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(x, kernel, bias, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, x, kernel, stride=1, padding=1)
        return result.grad_input

    var output = forward(input)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(forward, backward, input, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_gradient_kernel_strided() raises:
    """Test depthwise_conv2d kernel gradient with stride=2."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(6)  # height
    shape.append(6)  # width
    var input = zeros(shape, DType.float32)
    for i in range(input.numel()):
        input.set(i, Float32(Float32(i) * Float32(0.1)))

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(kernel.numel()):
        kernel.set(i, Float32(Float32(i) * Float32(0.05) + Float32(0.1)))

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = zeros(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, k, bias, stride=2, padding=1)

    fn backward(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, k, stride=2, padding=1)
        return result.grad_weights

    var output = forward(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(forward, backward, kernel, grad_output, rtol=1e-2, atol=1e-2)


fn main() raises:
    print("Running depthwise conv2d gradient checking tests...")
    test_depthwise_conv2d_gradient_kernel_basic()
    test_depthwise_conv2d_gradient_bias_basic()
    test_depthwise_conv2d_gradient_input_basic()
    test_depthwise_conv2d_gradient_kernel_strided()
    print("All depthwise conv2d gradient checking tests passed!")
