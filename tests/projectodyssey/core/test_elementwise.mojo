"""Tests for elementwise operations.

Tests cover:
- Mathematical functions: abs, sign
- Backward passes for differentiable functions
- Numerical correctness and edge cases

All tests use pure functional API.
"""


from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    zeros,
    ones,
    zeros_like,
    ones_like,
)
from projectodyssey.core.elementwise import (
    abs,
    sign,
    exp,
    log,
    sqrt,
    sin,
    cos,
    clip,
    ceil,
    floor,
    round,
    trunc,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    log10,
    log2,
    exp_backward,
    log_backward,
    sqrt_backward,
    abs_backward,
    clip_backward,
    log10_backward,
    log2_backward,
    sin_backward,
    cos_backward,
)
from projectodyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from std.math import sqrt as math_sqrt, pi


@fieldwise_init
struct _AbsFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return abs(inp)


@fieldwise_init
struct _AbsBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return abs_backward(grad, inp)


@fieldwise_init
struct _ExpFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return exp(inp)


@fieldwise_init
struct _ExpBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return exp_backward(grad, inp)


@fieldwise_init
struct _LogFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return log(inp)


@fieldwise_init
struct _LogBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return log_backward(grad, inp)


@fieldwise_init
struct _Log10Fwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return log10(inp)


@fieldwise_init
struct _Log10Bwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return log10_backward(grad, inp)


@fieldwise_init
struct _Log2Fwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return log2(inp)


@fieldwise_init
struct _Log2Bwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return log2_backward(grad, inp)


@fieldwise_init
struct _SqrtFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return sqrt(inp)


@fieldwise_init
struct _SqrtBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return sqrt_backward(grad, inp)


@fieldwise_init
struct _SinFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return sin(inp)


@fieldwise_init
struct _SinBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return sin_backward(grad, inp)


@fieldwise_init
struct _CosFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return cos(inp)


@fieldwise_init
struct _CosBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return cos_backward(grad, inp)


@fieldwise_init
struct _ClipFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return clip(inp, min_val=-1.0, max_val=1.0)


@fieldwise_init
struct _ClipBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return clip_backward(grad, inp, min_val=-1.0, max_val=1.0)


def test_abs_shapes() raises:
    """Test that abs returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = abs(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


def test_abs_values() raises:
    """Test that abs computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-5.0))
    x.set(1, Float32(-2.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(3.0))
    x.set(4, Float32(7.0))

    var result = abs(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(7.0), tolerance=1e-5
    )


def test_abs_backward() raises:
    """Test abs backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(-2.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(3.0))

    var grad_input = abs_backward(grad_output, x)

    # Gradient: -1 for x < 0, +1 for x > 0, 0 for x == 0
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


def test_abs_backward_gradient() raises:
    """Test abs backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use non-zero values to avoid discontinuity at x=0
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.2))
    x.set(2, Float32(1.5))

    var y = abs(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_AbsFwd(), _AbsBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_sign_values() raises:
    """Test that sign returns correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-5.0))
    x.set(1, Float32(-0.1))
    x.set(2, Float32(0.0))
    x.set(3, Float32(0.1))
    x.set(4, Float32(7.0))

    var result = sign(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(1.0), tolerance=1e-5
    )


def test_logical_xor_basic() raises:
    """Test logical_xor basic functionality. Closes #4145."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a: [0, 1, 0, 1], b: [0, 0, 1, 1]
    a.set(1, Float32(1.0))
    a.set(3, Float32(1.0))
    b.set(2, Float32(1.0))
    b.set(3, Float32(1.0))

    var result = logical_xor(a, b)

    # XOR: [0^0=0, 1^0=1, 0^1=1, 1^1=0]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(0.0), tolerance=1e-5
    )


def test_logical_xor_same_inputs() raises:
    """Test logical_xor with identical inputs returns all zeros. Closes #4145.
    """
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)

    var result = logical_xor(a, a)

    # XOR of identical inputs should be all 0
    for i in range(3):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            Float32(0.0),
            tolerance=1e-5,
        )


def test_exp_shapes() raises:
    """Test that exp returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = exp(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


def test_exp_values() raises:
    """Test that exp computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

    var result = exp(x)

    # exp(0) = 1, exp(1) ≈ 2.718, exp(2) ≈ 7.389
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(2.718), tolerance=0.01
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(7.389), tolerance=0.01
    )


def test_exp_backward() raises:
    """Test exp backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(1.0))

    var grad_input = exp_backward(grad_output, x)

    # d/dx[exp(x)] = exp(x)
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(2.718), tolerance=0.01
    )


def test_exp_backward_gradient() raises:
    """Test exp backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = exp(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_ExpFwd(), _ExpBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_log_shapes() raises:
    """Test that log returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = log(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


def test_log_values() raises:
    """Test that log computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(2.718))
    x.set(2, Float32(7.389))

    var result = log(x)

    # log(1) = 0, log(e) ≈ 1, log(e^2) ≈ 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=0.01
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=0.01
    )


def test_log_backward() raises:
    """Test log backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))

    var grad_input = log_backward(grad_output, x)

    # d/dx[log(x)] = 1/x
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5
    )


def test_log_backward_gradient() raises:
    """Test log backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

    var y = log(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_LogFwd(), _LogBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_log10_values() raises:
    """Test that log10 computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(10.0))
    x.set(2, Float32(100.0))

    var result = log10(x)

    # log10(1) = 0, log10(10) = 1, log10(100) = 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


def test_log10_backward_gradient() raises:
    """Test log10 backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

    var y = log10(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_Log10Fwd(), _Log10Bwd(), x, grad_out, rtol=5e-3, atol=1e-5)


def test_log2_values() raises:
    """Test that log2 computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(8.0))

    var result = log2(x)

    # log2(1) = 0, log2(2) = 1, log2(8) = 3
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(3.0), tolerance=1e-5
    )


def test_log2_backward_gradient() raises:
    """Test log2 backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

    var y = log2(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_Log2Fwd(), _Log2Bwd(), x, grad_out, rtol=5e-3, atol=1e-5)


def test_sqrt_shapes() raises:
    """Test that sqrt returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = sqrt(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


def test_sqrt_values() raises:
    """Test that sqrt computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(1.0))
    x.set(2, Float32(4.0))
    x.set(3, Float32(9.0))

    var result = sqrt(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(3.0), tolerance=1e-5
    )


def test_sqrt_backward() raises:
    """Test sqrt backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(4.0))

    var grad_input = sqrt_backward(grad_output, x)

    # d/dx[sqrt(x)] = 1/(2*sqrt(x))
    # x=1: 1/(2*1) = 0.5
    # x=4: 1/(2*2) = 0.25
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )


def test_sqrt_backward_gradient() raises:
    """Test sqrt backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

    var y = sqrt(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_SqrtFwd(), _SqrtBwd(), x, grad_out, rtol=5e-3, atol=1e-5)


def test_sin_values() raises:
    """Test that sin computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(pi / 2.0))
    x.set(2, Float32(pi))

    var result = sin(x)

    # sin(0) = 0, sin(π/2) = 1, sin(π) = 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )


def test_cos_values() raises:
    """Test that cos computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(pi / 2.0))
    x.set(2, Float32(pi))

    var result = cos(x)

    # cos(0) = 1, cos(π/2) = 0, cos(π) = -1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(-1.0), tolerance=1e-5
    )


def test_sin_backward() raises:
    """Test sin backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(pi / 2.0))
    x.set(2, Float32(pi))

    var grad_input = sin_backward(grad_output, x)

    # d/dx[sin(x)] = cos(x)
    # cos(0) = 1, cos(π/2) = 0, cos(π) = -1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(-1.0), tolerance=1e-5
    )


def test_sin_backward_gradient() raises:
    """Test sin backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = sin(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_SinFwd(), _SinBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_cos_backward() raises:
    """Test cos backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(pi / 2.0))
    x.set(2, Float32(pi))

    var grad_input = cos_backward(grad_output, x)

    # d/dx[cos(x)] = -sin(x)
    # -sin(0) = 0, -sin(π/2) = -1, -sin(π) = 0
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )


def test_cos_backward_gradient() raises:
    """Test cos backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = cos(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_CosFwd(), _CosBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_clip_shapes() raises:
    """Test that clip returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = clip(x, min_val=-1.0, max_val=1.0)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


def test_clip_values() raises:
    """Test that clip computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-5.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(5.0))

    var result = clip(x, min_val=-2.0, max_val=2.0)

    # Clip to [-2, 2]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


def test_clip_backward() raises:
    """Test clip backward pass."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x.set(0, Float32(-5.0))  # Below min
    x.set(1, Float32(-1.0))  # Within range
    x.set(2, Float32(0.0))  # Within range
    x.set(3, Float32(1.0))  # Within range
    x.set(4, Float32(5.0))  # Above max

    var grad_input = clip_backward(grad_output, x, min_val=-2.0, max_val=2.0)

    # Gradient is 0 outside range, 1 inside range
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[4], Float32(0.0), tolerance=1e-5
    )


def test_clip_backward_gradient() raises:
    """Test clip backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values within the clipping range
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = clip(x, min_val=-1.0, max_val=1.0)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_ClipFwd(), _ClipBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_ceil_values() raises:
    """Test that ceil computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-2.5))
    x.set(1, Float32(-1.1))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.1))
    x.set(4, Float32(2.5))

    var result = ceil(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(3.0), tolerance=1e-5
    )


def test_floor_values() raises:
    """Test that floor computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-2.5))
    x.set(1, Float32(-1.1))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.1))
    x.set(4, Float32(2.5))

    var result = floor(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


def test_round_values() raises:
    """Test that round computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-2.5))
    x.set(1, Float32(-1.4))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.4))
    x.set(4, Float32(2.5))

    var result = round(x)

    # Round to nearest even (banker's rounding)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


def test_logical_and_values() raises:
    """Test that logical_and computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Test all combinations: (0, 0), (0, 1), (1, 0), (1, 1)
    a.set(0, Float32(0.0))
    a.set(1, Float32(0.0))
    a.set(2, Float32(1.0))
    a.set(3, Float32(1.0))

    b.set(0, Float32(0.0))
    b.set(1, Float32(1.0))
    b.set(2, Float32(0.0))
    b.set(3, Float32(1.0))

    var result = logical_and(a, b)

    # AND truth table: 0, 0, 0, 1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


def test_logical_or_values() raises:
    """Test that logical_or computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(0.0))
    a.set(1, Float32(0.0))
    a.set(2, Float32(1.0))
    a.set(3, Float32(1.0))

    b.set(0, Float32(0.0))
    b.set(1, Float32(1.0))
    b.set(2, Float32(0.0))
    b.set(3, Float32(1.0))

    var result = logical_or(a, b)

    # OR truth table: 0, 1, 1, 1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


def test_logical_not_values() raises:
    """Test that logical_not computes correct values."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(1.0))

    var result = logical_not(x)

    # NOT truth table: 1, 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )


def main() raises:
    """Run all test_elementwise tests."""
    print("Running test_elementwise tests...")

    test_abs_shapes()
    print("✓ test_abs_shapes")

    test_abs_values()
    print("✓ test_abs_values")

    test_abs_backward()
    print("✓ test_abs_backward")

    test_abs_backward_gradient()
    print("✓ test_abs_backward_gradient")

    test_sign_values()
    print("✓ test_sign_values")

    test_logical_xor_basic()
    print("✓ test_logical_xor_basic")

    test_logical_xor_same_inputs()
    print("✓ test_logical_xor_same_inputs")

    test_exp_shapes()
    print("✓ test_exp_shapes")

    test_exp_values()
    print("✓ test_exp_values")

    test_exp_backward()
    print("✓ test_exp_backward")

    test_exp_backward_gradient()
    print("✓ test_exp_backward_gradient")

    test_log_shapes()
    print("✓ test_log_shapes")

    test_log_values()
    print("✓ test_log_values")

    test_log_backward()
    print("✓ test_log_backward")

    test_log_backward_gradient()
    print("✓ test_log_backward_gradient")

    test_log10_values()
    print("✓ test_log10_values")

    test_log10_backward_gradient()
    print("✓ test_log10_backward_gradient")

    test_log2_values()
    print("✓ test_log2_values")

    test_log2_backward_gradient()
    print("✓ test_log2_backward_gradient")

    test_sqrt_shapes()
    print("✓ test_sqrt_shapes")

    test_sqrt_values()
    print("✓ test_sqrt_values")

    test_sqrt_backward()
    print("✓ test_sqrt_backward")

    test_sqrt_backward_gradient()
    print("✓ test_sqrt_backward_gradient")

    test_sin_values()
    print("✓ test_sin_values")

    test_cos_values()
    print("✓ test_cos_values")

    test_sin_backward()
    print("✓ test_sin_backward")

    test_sin_backward_gradient()
    print("✓ test_sin_backward_gradient")

    test_cos_backward()
    print("✓ test_cos_backward")

    test_cos_backward_gradient()
    print("✓ test_cos_backward_gradient")

    test_clip_shapes()
    print("✓ test_clip_shapes")

    test_clip_values()
    print("✓ test_clip_values")

    test_clip_backward()
    print("✓ test_clip_backward")

    test_clip_backward_gradient()
    print("✓ test_clip_backward_gradient")

    test_ceil_values()
    print("✓ test_ceil_values")

    test_floor_values()
    print("✓ test_floor_values")

    test_round_values()
    print("✓ test_round_values")

    test_logical_and_values()
    print("✓ test_logical_and_values")

    test_logical_or_values()
    print("✓ test_logical_or_values")

    test_logical_not_values()
    print("✓ test_logical_not_values")

    print("\nAll test_elementwise tests passed!")
