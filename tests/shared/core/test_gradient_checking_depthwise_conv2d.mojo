# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checking.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Gradient checking tests for depthwise_conv2d backward pass.

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import assert_true
from shared.testing import check_gradients
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.conv import depthwise_conv2d, depthwise_conv2d_backward


fn test_depthwise_conv2d_gradient_kernel_basic() raises:
    """Test depthwise_conv2d kernel gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = ones(shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = ones(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, k, bias, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, k, stride=1, padding=1)
        return result.grad_weights

    var passed = check_gradients(forward, backward, kernel)
    assert_true(passed, "depthwise_conv2d kernel gradient check failed")


fn test_depthwise_conv2d_gradient_bias_basic() raises:
    """Test depthwise_conv2d bias gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = ones(shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = ones(bias_shape, DType.float32)

    fn forward(b: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, kernel, b, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, b: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, kernel, stride=1, padding=1)
        return result.grad_bias

    var passed = check_gradients(forward, backward, bias)
    assert_true(passed, "depthwise_conv2d bias gradient check failed")


fn test_depthwise_conv2d_gradient_input_basic() raises:
    """Test depthwise_conv2d input gradient using finite differences."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var input = ones(shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = ones(bias_shape, DType.float32)

    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(x, kernel, bias, stride=1, padding=1)

    fn backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, x, kernel, stride=1, padding=1)
        return result.grad_input

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "depthwise_conv2d input gradient check failed")


fn test_depthwise_conv2d_gradient_kernel_strided() raises:
    """Test depthwise_conv2d kernel gradient with stride=2."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(6)  # height
    shape.append(6)  # width
    var input = ones(shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(2)  # channels
    kernel_shape.append(1)  # 1 for depthwise
    kernel_shape.append(3)  # kernel height
    kernel_shape.append(3)  # kernel width
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(2)  # channels
    var bias = ones(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return depthwise_conv2d(input, k, bias, stride=2, padding=1)

    fn backward(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = depthwise_conv2d_backward(grad_out, input, k, stride=2, padding=1)
        return result.grad_weights

    var passed = check_gradients(forward, backward, kernel)
    assert_true(passed, "depthwise_conv2d kernel gradient check with stride=2 failed")
