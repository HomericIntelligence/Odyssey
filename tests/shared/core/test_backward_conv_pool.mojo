# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Tests for conv2d and pooling backward passes."""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
)
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
)
from shared.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)


# ============================================================================
# Helper Functions
# ============================================================================


def _make_test_conv2d_tensors() raises -> (
    Tuple[AnyTensor, AnyTensor, AnyTensor]
):
    """Create and initialize test tensors for conv2d backward tests.

    Returns:
        (x, kernel, bias) - input, kernel, and bias tensors with non-uniform
        initialization values for meaningful gradient checking.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values for meaningful gradient check
    for i in range(16):
        x.set(i, Float64(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    # Initialize kernel with non-uniform values
    for i in range(9):
        kernel.set(i, Float64(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    return (x, kernel, bias)


def test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 2
    var in_channels = 3
    var out_channels = 4
    var in_h = 8
    var in_w = 8
    var kh = 3
    var kw = 3

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    var grads = conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_h)
    assert_equal(gi_shape[3], in_w)

    var gk_shape = grads.grad_weights.shape()
    assert_equal(gk_shape[0], out_channels)
    assert_equal(gk_shape[1], in_channels)
    assert_equal(gk_shape[2], kh)
    assert_equal(gk_shape[3], kw)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


def test_conv2d_backward_with_stride() raises:
    """Test conv2d_backward with stride > 1."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var output = conv2d(x, kernel, bias, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grads = conv2d_backward(grad_output, x, kernel, stride=2, padding=0)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 1)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


@fieldwise_init
struct _Conv2dInpFwd(NumericalForward):
    var kernel: AnyTensor
    var bias: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return conv2d(inp, self.kernel, self.bias, stride=1, padding=0)


@fieldwise_init
struct _Conv2dInpBwd(NumericalBackward):
    var kernel: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(
            grad_out, inp, self.kernel, stride=1, padding=0
        )
        return grads.grad_input


def test_conv2d_backward_grad_input_numerical() raises:
    """Test conv2d_backward grad_input values via numerical gradient checking.

    Uses small input (1,1,4,4) with kernel (1,1,3,3) as specified in issue #3281.
    Verifies mathematical correctness of grad_input, not just shape.
    """
    var (x, kernel, bias) = _make_test_conv2d_tensors()

    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    check_gradient(
        _Conv2dInpFwd(kernel, bias),
        _Conv2dInpBwd(kernel),
        x,
        grad_output,
        rtol=1e-2,
        atol=1e-2,
    )


@fieldwise_init
struct _Conv2dWgtFwd(NumericalForward):
    var x: AnyTensor
    var bias: AnyTensor

    def __call__(self, k: AnyTensor) raises -> AnyTensor:
        return conv2d(self.x, k, self.bias, stride=1, padding=0)


@fieldwise_init
struct _Conv2dWgtBwd(NumericalBackward):
    var x: AnyTensor

    def __call__(self, grad_out: AnyTensor, k: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(grad_out, self.x, k, stride=1, padding=0)
        return grads.grad_weights


def test_conv2d_backward_grad_weights_numerical() raises:
    """Test conv2d_backward grad_weights values via numerical gradient checking.

    Uses small input (1,1,4,4) with kernel (1,1,3,3) as specified in issue #3281.
    Verifies mathematical correctness of grad_weights, not just shape.
    """
    var (x, kernel, bias) = _make_test_conv2d_tensors()

    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    check_gradient(
        _Conv2dWgtFwd(x, bias),
        _Conv2dWgtBwd(x),
        kernel,
        grad_output,
        rtol=1e-2,
        atol=1e-2,
    )


def test_maxpool2d_backward_shapes() raises:
    """Test that maxpool2d_backward returns correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


def test_maxpool2d_backward_gradient_routing() raises:
    """Test that maxpool2d_backward routes gradients only to max positions."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = zeros(input_shape, DType.float32)
    x.set(0, Float64(1.0))
    x.set(1, Float64(2.0))
    x.set(2, Float64(3.0))
    x.set(3, Float64(4.0))

    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


def test_avgpool2d_backward_shapes() raises:
    """Test that avgpool2d_backward returns correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = avgpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


def test_avgpool2d_backward_gradient_distribution() raises:
    """Test that avgpool2d_backward distributes gradients equally."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = ones(input_shape, DType.float32)

    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = avgpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(0.25), tolerance=1e-5
    )


@fieldwise_init
struct _Conv2dFwd(NumericalForward):
    var kernel: AnyTensor
    var bias: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return conv2d(inp, self.kernel, self.bias, stride=1, padding=0)


@fieldwise_init
struct _Conv2dBwd(NumericalBackward):
    var kernel: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(
            grad_out, inp, self.kernel, stride=1, padding=0
        )
        return grads.grad_input


def test_conv2d_backward_gradient() raises:
    """Test conv2d backward with numerical gradient checking."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(1 * 2 * 5 * 5):
        x.set(i, Float64(Float32(i) * 0.1 - 2.5))

    var kernel_shape = List[Int]()
    kernel_shape.append(2)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(2 * 2 * 3 * 3):
        kernel.set(i, Float64(Float32(i) * 0.05 - 0.5))

    var bias_shape = List[Int]()
    bias_shape.append(2)
    var bias = zeros(bias_shape, DType.float32)

    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    check_gradient(
        _Conv2dFwd(kernel, bias),
        _Conv2dBwd(kernel),
        x,
        grad_output,
        rtol=1e-2,
        atol=1e-2,
    )


@fieldwise_init
struct _MaxPool2dFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return maxpool2d(inp, kernel_size=2, stride=2, padding=0)


@fieldwise_init
struct _MaxPool2dBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return maxpool2d_backward(
            grad_out, inp, kernel_size=2, stride=2, padding=0
        )


def test_maxpool2d_backward_gradient() raises:
    """Test maxpool2d backward with numerical gradient checking."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(1 * 2 * 4 * 4):
        x.set(i, Float64(Float32(i) * 0.1 - 1.6))

    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    check_gradient(
        _MaxPool2dFwd(), _MaxPool2dBwd(), x, grad_output, rtol=1e-3, atol=5e-4
    )


@fieldwise_init
struct _AvgPool2dFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return avgpool2d(inp, kernel_size=2, stride=2, padding=0)


@fieldwise_init
struct _AvgPool2dBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return avgpool2d_backward(
            grad_out, inp, kernel_size=2, stride=2, padding=0
        )


def test_avgpool2d_backward_gradient() raises:
    """Test avgpool2d backward with numerical gradient checking."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(1 * 2 * 4 * 4):
        x.set(i, Float64(Float32(i) * 0.1 - 1.6))

    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    check_gradient(
        _AvgPool2dFwd(), _AvgPool2dBwd(), x, grad_output, rtol=1e-3, atol=5e-4
    )


def main() raises:
    """Run conv2d and pooling backward tests."""
    print("Running conv2d and pooling backward tests...")
    test_conv2d_backward_shapes()
    test_conv2d_backward_with_stride()
    test_conv2d_backward_grad_input_numerical()
    test_conv2d_backward_grad_weights_numerical()
    test_maxpool2d_backward_shapes()
    test_maxpool2d_backward_gradient_routing()
    test_avgpool2d_backward_shapes()
    test_avgpool2d_backward_gradient_distribution()
    test_conv2d_backward_gradient()
    test_maxpool2d_backward_gradient()
    test_avgpool2d_backward_gradient()
    print("All conv2d and pooling backward tests passed!")
