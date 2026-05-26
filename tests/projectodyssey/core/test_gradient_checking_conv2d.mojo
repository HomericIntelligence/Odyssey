"""Gradient checking tests for conv2d backward: grad_input, grad_weights, grad_bias.

Verifies all three conv2d backward outputs via finite-difference gradient
checking across three configurations: same-padding, strided, and multi-channel.

Uses check_gradient() (not check_gradients()) because it applies combined
relative+absolute tolerance: |diff| <= atol + rtol * max_magnitude. This is
essential for multi-channel configs where gradient magnitudes reach ~30-45,
making pure absolute tolerance of 0.01 too tight. See issue #2704.

Test Coverage:
- Config A (same-padding): stride=1, padding=1, input (1,1,4,4), kernel (1,1,3,3)
- Config B (strided): stride=2, padding=0, input (1,1,7,7), kernel (1,1,3,3)
- Config C (multi-channel): stride=1, padding=1, in_channels=2, out_channels=3,
    input (1,2,5,5), kernel (3,2,3,3)

References:
    - Issue #3774: Add gradient checking for grad_kernel and grad_bias in conv2d
    - Follow-up from #3233
"""

from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from projectodyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)


# ---- Conv2d: perturb input (captures kernel, bias, stride, padding) ----


@fieldwise_init
struct _Conv2dInputFwd(NumericalForward):
    var kernel: AnyTensor
    var bias: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return conv2d(
            inp,
            self.kernel,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


@fieldwise_init
struct _Conv2dInputBwd(NumericalBackward):
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var result = conv2d_backward(
            grad_out, inp, self.kernel, stride=self.stride, padding=self.padding
        )
        return result.grad_input


# ---- Conv2d: perturb kernel/weights (captures x, bias, stride, padding) ----


@fieldwise_init
struct _Conv2dKernelFwd(NumericalForward):
    var x: AnyTensor
    var bias: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, k: AnyTensor) raises -> AnyTensor:
        return conv2d(
            self.x, k, self.bias, stride=self.stride, padding=self.padding
        )


@fieldwise_init
struct _Conv2dKernelBwd(NumericalBackward):
    var x: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, k: AnyTensor) raises -> AnyTensor:
        var result = conv2d_backward(
            grad_out, self.x, k, stride=self.stride, padding=self.padding
        )
        return result.grad_weights


# ---- Conv2d: perturb bias (captures x, kernel, stride, padding) ----


@fieldwise_init
struct _Conv2dBiasFwd(NumericalForward):
    var x: AnyTensor
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, b: AnyTensor) raises -> AnyTensor:
        return conv2d(
            self.x, self.kernel, b, stride=self.stride, padding=self.padding
        )


@fieldwise_init
struct _Conv2dBiasBwd(NumericalBackward):
    var x: AnyTensor
    var kernel: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, grad_out: AnyTensor, b: AnyTensor) raises -> AnyTensor:
        var result = conv2d_backward(
            grad_out,
            self.x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
        )
        return result.grad_bias


def _make_ones_grad_output(output: AnyTensor) raises -> AnyTensor:
    """Create ones grad_output matching output shape."""
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)
    return grad_output^


def test_conv2d_same_padding_grad_input() raises:
    """Test conv2d grad_input with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dInputFwd(kernel, bias, 1, 1)
    var bwd = _Conv2dInputBwd(kernel, 1, 1)
    var output = fwd(x)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, x, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_same_padding_grad_weights() raises:
    """Test conv2d grad_weights with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dKernelFwd(x, bias, 1, 1)
    var bwd = _Conv2dKernelBwd(x, 1, 1)
    var output = fwd(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, kernel, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_same_padding_grad_bias() raises:
    """Test conv2d grad_bias with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dBiasFwd(x, kernel, 1, 1)
    var bwd = _Conv2dBiasBwd(x, kernel, 1, 1)
    var output = fwd(bias)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, bias, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_strided_grad_input() raises:
    """Test conv2d grad_input with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dInputFwd(kernel, bias, 2, 0)
    var bwd = _Conv2dInputBwd(kernel, 2, 0)
    var output = fwd(x)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, x, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_strided_grad_weights() raises:
    """Test conv2d grad_weights with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dKernelFwd(x, bias, 2, 0)
    var bwd = _Conv2dKernelBwd(x, 2, 0)
    var output = fwd(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, kernel, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_strided_grad_bias() raises:
    """Test conv2d grad_bias with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dBiasFwd(x, kernel, 2, 0)
    var bwd = _Conv2dBiasBwd(x, kernel, 2, 0)
    var output = fwd(bias)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, bias, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_multichannel_grad_input() raises:
    """Test conv2d grad_input with in_channels=2, out_channels=3, input (1,2,5,5).
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dInputFwd(kernel, bias, 1, 1)
    var bwd = _Conv2dInputBwd(kernel, 1, 1)
    var output = fwd(x)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, x, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_multichannel_grad_weights() raises:
    """Test conv2d grad_weights with in_channels=2, out_channels=3, kernel (3,2,3,3).
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dKernelFwd(x, bias, 1, 1)
    var bwd = _Conv2dKernelBwd(x, 1, 1)
    var output = fwd(kernel)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, kernel, grad_output, rtol=1e-2, atol=1e-2)


def test_conv2d_multichannel_grad_bias() raises:
    """Test conv2d grad_bias with out_channels=3, bias shape (3,)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    var fwd = _Conv2dBiasFwd(x, kernel, 1, 1)
    var bwd = _Conv2dBiasBwd(x, kernel, 1, 1)
    var output = fwd(bias)
    var grad_output = _make_ones_grad_output(output)
    check_gradient(fwd, bwd, bias, grad_output, rtol=1e-2, atol=1e-2)


def main() raises:
    """Run all 9 conv2d gradient checking tests."""
    print("Running Conv2D Gradient Checking Tests...")
    print("=" * 60)

    print("[1/9] Config A — same-padding grad_input...")
    test_conv2d_same_padding_grad_input()
    print("✓ PASSED")

    print("[2/9] Config A — same-padding grad_weights...")
    test_conv2d_same_padding_grad_weights()
    print("✓ PASSED")

    print("[3/9] Config A — same-padding grad_bias...")
    test_conv2d_same_padding_grad_bias()
    print("✓ PASSED")

    print("[4/9] Config B — strided grad_input...")
    test_conv2d_strided_grad_input()
    print("✓ PASSED")

    print("[5/9] Config B — strided grad_weights...")
    test_conv2d_strided_grad_weights()
    print("✓ PASSED")

    print("[6/9] Config B — strided grad_bias...")
    test_conv2d_strided_grad_bias()
    print("✓ PASSED")

    print("[7/9] Config C — multi-channel grad_input...")
    test_conv2d_multichannel_grad_input()
    print("✓ PASSED")

    print("[8/9] Config C — multi-channel grad_weights...")
    test_conv2d_multichannel_grad_weights()
    print("✓ PASSED")

    print("[9/9] Config C — multi-channel grad_bias...")
    test_conv2d_multichannel_grad_bias()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 9 conv2d gradient checking tests PASSED! ✓")
    print("grad_input, grad_weights, grad_bias verified for 3 configs.")
