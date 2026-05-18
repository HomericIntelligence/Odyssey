# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

"""Gradient checking tests for depthwise_conv2d backward pass.

Uses check_gradient() (not check_gradients()) for combined relative+absolute
tolerance, and non-uniform input values to avoid degenerate gradient patterns.
See issue #3282 and the depthwise-conv2d-gradient-checking skill for rationale.

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests.
"""

from projectodyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from projectodyssey.core.conv import depthwise_conv2d, depthwise_conv2d_backward


# ---- Depthwise conv2d: perturb kernel (captures input, bias, stride, padding) ----


@fieldwise_init
struct _DepthwiseConv2dKernelFwd(NumericalForward):
    var input: AnyTensor
    var bias: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, k: AnyTensor) raises -> AnyTensor:
        return depthwise_conv2d(
            self.input, k, self.bias, stride=self.stride, padding=self.padding
        )


@fieldwise_init
struct _DepthwiseConv2dKernelBwd(NumericalBackward):
    var input: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, k: AnyTensor) raises -> AnyTensor:
        var result = depthwise_conv2d_backward(
            grad_out, self.input, k, stride=self.stride, padding=self.padding
        )
        return result.grad_weights


# ---- Depthwise conv2d: perturb bias (captures input, kernel, stride, padding) ----


@fieldwise_init
struct _DepthwiseConv2dBiasFwd(NumericalForward):
    var input: AnyTensor
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, b: AnyTensor) raises -> AnyTensor:
        return depthwise_conv2d(
            self.input, self.kernel, b, stride=self.stride, padding=self.padding
        )


@fieldwise_init
struct _DepthwiseConv2dBiasBwd(NumericalBackward):
    var input: AnyTensor
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, b: AnyTensor) raises -> AnyTensor:
        var result = depthwise_conv2d_backward(
            grad_out,
            self.input,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
        )
        return result.grad_bias


# ---- Depthwise conv2d: perturb input (captures kernel, bias, stride, padding) ----


@fieldwise_init
struct _DepthwiseConv2dInputFwd(NumericalForward):
    var kernel: AnyTensor
    var bias: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return depthwise_conv2d(
            x, self.kernel, self.bias, stride=self.stride, padding=self.padding
        )


@fieldwise_init
struct _DepthwiseConv2dInputBwd(NumericalBackward):
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var result = depthwise_conv2d_backward(
            grad_out, x, self.kernel, stride=self.stride, padding=self.padding
        )
        return result.grad_input


def _make_ones_grad_output(output: AnyTensor) raises -> AnyTensor:
    """Create ones grad_output matching output shape."""
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)
    return grad_output^


def test_depthwise_conv2d_gradient_kernel_basic() raises:
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

    var fwd = _DepthwiseConv2dKernelFwd(input, bias, 1, 1)
    var bwd = _DepthwiseConv2dKernelBwd(input, 1, 1)
    var output = fwd(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, kernel, grad_output, rtol=1e-2, atol=1e-2)


def test_depthwise_conv2d_gradient_bias_basic() raises:
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

    var fwd = _DepthwiseConv2dBiasFwd(input, kernel, 1, 1)
    var bwd = _DepthwiseConv2dBiasBwd(input, kernel, 1, 1)
    var output = fwd(bias)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, bias, grad_output, rtol=1e-2, atol=1e-2)


def test_depthwise_conv2d_gradient_input_basic() raises:
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

    var fwd = _DepthwiseConv2dInputFwd(kernel, bias, 1, 1)
    var bwd = _DepthwiseConv2dInputBwd(kernel, 1, 1)
    var output = fwd(input)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, input, grad_output, rtol=1e-2, atol=1e-2)


def test_depthwise_conv2d_gradient_kernel_strided() raises:
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

    var fwd = _DepthwiseConv2dKernelFwd(input, bias, 2, 1)
    var bwd = _DepthwiseConv2dKernelBwd(input, 2, 1)
    var output = fwd(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, kernel, grad_output, rtol=1e-2, atol=1e-2)


def main() raises:
    print("Running depthwise conv2d gradient checking tests...")
    test_depthwise_conv2d_gradient_kernel_basic()
    test_depthwise_conv2d_gradient_bias_basic()
    test_depthwise_conv2d_gradient_input_basic()
    test_depthwise_conv2d_gradient_kernel_strided()
    print("All depthwise conv2d gradient checking tests passed!")
