"""Tests for arithmetic backward passes — Part 1: element-wise and scalar operations.

Tests cover:
- Element-wise operations: add, subtract, multiply, divide backward
- Scalar (broadcast) variants of each operation
- Tests 1–8 of the original test_arithmetic_backward.mojo
"""


from tests.odyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import (
    zeros,
    ones,
    ones_like,
    zeros_like,
    full,
)
from odyssey.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)
from odyssey.testing.gradient_checker import (
    check_gradient,
    compute_numerical_gradient,
    NumericalForward,
    NumericalBackward,
)


def create_shape_vec(*dims: Int) -> List[Int]:
    """Create a List[Int] from variadic arguments.

    Args:
        dims: Variable number of dimension sizes.

    Returns:
        List[Int] with specified dimensions.
    """
    var shape = List[Int]()
    for i in range(len(dims)):
        shape.append(dims[i])
    return shape^


def fill_tensor_sequential(
    mut tensor: AnyTensor, start_val: Float32 = 1.0
) raises -> None:
    """Fill tensor with sequential values starting from start_val.

    Args:
        tensor: AnyTensor to fill.
        start_val: Starting value for sequence.
    """
    for i in range(tensor.numel()):
        tensor.set(i, Float32(start_val + Float32(i)))


def test_add_backward() raises:
    """Test add_backward with same-shaped tensors.

    Tests that ∂L/∂A and ∂L/∂B equal 1.0 for C = A + B when upstream
    gradient is 1.0. This is because ∂(A+B)/∂A = 1 and ∂(A+B)/∂B = 1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # Call add_backward - passing shapes as per function signature
    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # For addition, gradients should just pass through (equal to grad_output)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


def test_add_scalar_backward() raises:
    """Test add_backward with scalar (broadcast) addition.

    Tests gradient computation when one operand broadcasts to the other.
    Broadcasting case: [2, 3] + [1] -> [2, 3]
    The gradient for [1] should sum over the broadcast dimensions.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    # Create gradient matching output shape [2, 3]
    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Call add_backward
    var grads = add_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)

    # grad_b should contain sum of 6 ones = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


def test_subtract_backward() raises:
    """Test subtract_backward with same-shaped tensors.

    Tests that ∂L/∂A = 1.0 and ∂L/∂B = -1.0 for C = A - B when upstream
    gradient is 1.0. This is because ∂(A-B)/∂A = 1 and ∂(A-B)/∂B = -1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should be positive (1.0)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )

    # grad_b should be negative (-1.0)
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-5
        )


def test_subtract_scalar_backward() raises:
    """Test subtract_backward with scalar (broadcast) subtraction.

    Broadcasting case: [2, 3] - [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1] with value -6.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 1)
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-6.0), tolerance=1e-5
    )


def test_multiply_backward() raises:
    """Test multiply_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C * B and ∂L/∂B = ∂L/∂C * A for C = A * B.
    Uses product rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Fill b with value 2.0
    for i in range(b.numel()):
        b.set(i, Float32(2.0))

    var grad_output = ones(shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b = grad_output * a = 1.0 * 1.0 = 1.0
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


def test_multiply_scalar_backward() raises:
    """Test multiply_backward with scalar (broadcast) multiplication.

    Broadcasting case: [2, 3] * [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)
    b_scalar.set(0, Float32(2.0))

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(grad_output * a) = sum(1.0 * 1.0) = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


def test_divide_backward() raises:
    """Test divide_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C / B and ∂L/∂B = -∂L/∂C * A / B² for C = A / B.
    Uses quotient rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a.set(i, Float32(2.0))
        b.set(i, Float32(2.0))

    var grad_output = ones(shape, DType.float32)

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b = -grad_output * a / b² = -1.0 * 2.0 / 4.0 = -0.5
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-0.5), tolerance=1e-4
        )


def test_divide_scalar_backward() raises:
    """Test divide_backward with scalar (broadcast) division.

    Broadcasting case: [2, 3] / [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = zeros(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)

    for i in range(6):
        a.set(i, Float32(2.0))
    b_scalar.set(0, Float32(2.0))

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = divide_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) = 6 * (-0.5) = -3.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-4
    )


def test_add_broadcast() raises:
    """Test add_backward with broadcasting.

    Broadcasting case: [2, 3] + [3] -> [2, 3]
    The gradient for the [3] tensor must be summed over the broadcast dimension.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)

    # grad_b should contain sum over first dimension (2 ones = 2.0)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


def test_subtract_broadcast() raises:
    """Test subtract_backward with broadcasting.

    Broadcasting case: [2, 3] - [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3] with value -2.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 3)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-2.0), tolerance=1e-5
        )


def test_multiply_broadcast() raises:
    """Test multiply_backward with broadcasting.

    Broadcasting case: [2, 3] * [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    for i in range(3):
        b.set(i, Float32(Float32(i + 1)))

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(grad_output * a) = sum(1.0 * 1.0) over first dim = 2.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


def test_divide_broadcast() raises:
    """Test divide_backward with broadcasting.

    Broadcasting case: [2, 3] / [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a.set(i, Float32(2.0))
    for i in range(3):
        b.set(i, Float32(2.0))

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) over first dim
    #           = 2 * (-0.5) = -1.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-4
        )


@fieldwise_init
struct _AddFwd(NumericalForward):
    var b: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return add(inp, self.b)


@fieldwise_init
struct _AddBwd(NumericalBackward):
    var b: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, inp, self.b)
        return grads.grad_a


def test_add_backward_gradient() raises:
    """Test add_backward with numerical gradient checking.

    Validates that analytical gradients match numerical gradients computed
    via central differences, confirming correct implementation of the
    backward pass for addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.1 - 1.2))
        b.set(i, Float32(Float32(i) * 0.15 - 0.8))

    var output = add(a, b)
    var grad_output = ones_like(output)

    check_gradient(_AddFwd(b), _AddBwd(b), a, grad_output, rtol=5e-3, atol=1e-5)


@fieldwise_init
struct _SubtractFwd(NumericalForward):
    var b: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return subtract(inp, self.b)


@fieldwise_init
struct _SubtractBwd(NumericalBackward):
    var b: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = subtract_backward(grad_out, inp, self.b)
        return grads.grad_a


def test_subtract_backward_gradient() raises:
    """Test subtract_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for subtraction operation.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.1 + 0.5))
        b.set(i, Float32(Float32(i) * 0.2 - 1.5))

    var output = subtract(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _SubtractFwd(b), _SubtractBwd(b), a, grad_output, rtol=5e-3, atol=1e-5
    )


@fieldwise_init
struct _MultiplyFwd(NumericalForward):
    var b: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return multiply(inp, self.b)


@fieldwise_init
struct _MultiplyBwd(NumericalBackward):
    var b: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, inp, self.b)
        return grads.grad_a


def test_multiply_backward_gradient() raises:
    """Test multiply_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for multiplication operation.
    Tests product rule: ∂(A*B)/∂A = B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero to test product properly)
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.1 + 0.1))
        b.set(i, Float32(Float32(i) * 0.15 + 0.2))

    var output = multiply(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _MultiplyFwd(b), _MultiplyBwd(b), a, grad_output, rtol=5e-3, atol=1e-5
    )


@fieldwise_init
struct _DivideFwd(NumericalForward):
    var b: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return divide(inp, self.b)


@fieldwise_init
struct _DivideBwd(NumericalBackward):
    var b: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, inp, self.b)
        return grads.grad_a


def test_divide_backward_gradient() raises:
    """Test divide_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for division operation.
    Tests quotient rule: ∂(A/B)/∂A = 1/B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero denominator)
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.2 + 0.5))
        b.set(i, Float32(Float32(i) * 0.1 + 1.0))  # Ensure b > 0

    var output = divide(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _DivideFwd(b), _DivideBwd(b), a, grad_output, rtol=1e-2, atol=1e-5
    )


@fieldwise_init
struct _AddFwdB(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return add(self.a, inp)


@fieldwise_init
struct _AddBwdB(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, self.a, inp)
        return grads.grad_b


def test_add_backward_b_gradient() raises:
    """Test add_backward gradient w.r.t. second operand (B).

    Validates gradient computation for the second input of addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.1 - 0.5))
        b.set(i, Float32(Float32(i) * 0.12 + 0.3))

    var output = add(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _AddFwdB(a), _AddBwdB(a), b, grad_output, rtol=5e-3, atol=1e-5
    )


@fieldwise_init
struct _SubtractFwdB(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return subtract(self.a, inp)


@fieldwise_init
struct _SubtractBwdB(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = subtract_backward(grad_out, self.a, inp)
        return grads.grad_b


def test_subtract_backward_b_gradient() raises:
    """Test subtract_backward gradient w.r.t. second operand (B).

    Validates that gradient for B is negated: ∂(A-B)/∂B = -1
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.15 + 0.2))
        b.set(i, Float32(Float32(i) * 0.1 - 1.0))

    var output = subtract(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _SubtractFwdB(a), _SubtractBwdB(a), b, grad_output, rtol=5e-3, atol=1e-5
    )


@fieldwise_init
struct _MultiplyFwdB(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return multiply(self.a, inp)


@fieldwise_init
struct _MultiplyBwdB(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, self.a, inp)
        return grads.grad_b


def test_multiply_backward_b_gradient() raises:
    """Test multiply_backward gradient w.r.t. second operand (B).

    Validates product rule for second operand: ∂(A*B)/∂B = A.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.2 + 0.1))
        b.set(i, Float32(Float32(i) * 0.15 + 0.15))

    var output = multiply(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _MultiplyFwdB(a), _MultiplyBwdB(a), b, grad_output, rtol=1e-2, atol=1e-5
    )


@fieldwise_init
struct _DivideFwdB(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return divide(self.a, inp)


@fieldwise_init
struct _DivideBwdB(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, self.a, inp)
        return grads.grad_b


def test_divide_backward_b_gradient() raises:
    """Test divide_backward gradient w.r.t. second operand (B).

    Validates quotient rule for denominator: ∂(A/B)/∂B = -A/B²
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0 to avoid division by zero)
    for i in range(12):
        a.set(i, Float32(Float32(i) * 0.2 + 0.5))
        b.set(i, Float32(Float32(i) * 0.1 + 1.5))  # b > 0

    var output = divide(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _DivideFwdB(a), _DivideBwdB(a), b, grad_output, rtol=1e-2, atol=1e-5
    )


@fieldwise_init
struct _AddBroadcastFwd(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return add(self.a, inp)


@fieldwise_init
struct _AddBroadcastBwd(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, self.a, inp)
        return grads.grad_b


@fieldwise_init
struct _MultiplyBroadcastFwd(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return multiply(self.a, inp)


@fieldwise_init
struct _MultiplyBroadcastBwd(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, self.a, inp)
        return grads.grad_b


@fieldwise_init
struct _DivideBroadcastFwd(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return divide(self.a, inp)


@fieldwise_init
struct _DivideBroadcastBwd(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, self.a, inp)
        return grads.grad_b


def test_add_backward_broadcast_gradient() raises:
    """Test add_backward with broadcasting and numerical gradient checking.

    Validates gradient computation when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(6):
        a.set(i, Float32(Float32(i) * 0.1 + 0.2))
    for i in range(3):
        b.set(i, Float32(Float32(i) * 0.15 - 0.3))

    var output = add(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _AddBroadcastFwd(a),
        _AddBroadcastBwd(a),
        b,
        grad_output,
        rtol=5e-3,
        atol=1e-5,
    )


def test_multiply_backward_broadcast_gradient() raises:
    """Test multiply_backward with broadcasting and numerical gradient checking.

    Validates product rule when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(6):
        a.set(i, Float32(Float32(i) * 0.1 + 0.1))
    for i in range(3):
        b.set(i, Float32(Float32(i) * 0.2 + 0.2))

    var output = multiply(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _MultiplyBroadcastFwd(a),
        _MultiplyBroadcastBwd(a),
        b,
        grad_output,
        rtol=5e-3,
        atol=1e-5,
    )


def test_divide_backward_broadcast_gradient() raises:
    """Test divide_backward with broadcasting and numerical gradient checking.

    Validates quotient rule when denominator broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0)
    for i in range(6):
        a.set(i, Float32(Float32(i) * 0.2 + 0.5))
    for i in range(3):
        b.set(i, Float32(Float32(i) * 0.1 + 1.0))  # b > 0

    var output = divide(a, b)
    var grad_output = ones_like(output)

    check_gradient(
        _DivideBroadcastFwd(a),
        _DivideBroadcastBwd(a),
        b,
        grad_output,
        rtol=5e-3,
        atol=1e-5,
    )


def main() raises:
    """Run all test_arithmetic_backward tests."""
    print("Running test_arithmetic_backward tests...")

    test_add_backward()
    print("✓ test_add_backward")

    test_add_scalar_backward()
    print("✓ test_add_scalar_backward")

    test_subtract_backward()
    print("✓ test_subtract_backward")

    test_subtract_scalar_backward()
    print("✓ test_subtract_scalar_backward")

    test_multiply_backward()
    print("✓ test_multiply_backward")

    test_multiply_scalar_backward()
    print("✓ test_multiply_scalar_backward")

    test_divide_backward()
    print("✓ test_divide_backward")

    test_divide_scalar_backward()
    print("✓ test_divide_scalar_backward")

    test_add_broadcast()
    print("✓ test_add_broadcast")

    test_subtract_broadcast()
    print("✓ test_subtract_broadcast")

    test_multiply_broadcast()
    print("✓ test_multiply_broadcast")

    test_divide_broadcast()
    print("✓ test_divide_broadcast")

    test_add_backward_gradient()
    print("✓ test_add_backward_gradient")

    test_subtract_backward_gradient()
    print("✓ test_subtract_backward_gradient")

    test_multiply_backward_gradient()
    print("✓ test_multiply_backward_gradient")

    test_divide_backward_gradient()
    print("✓ test_divide_backward_gradient")

    test_add_backward_b_gradient()
    print("✓ test_add_backward_b_gradient")

    test_subtract_backward_b_gradient()
    print("✓ test_subtract_backward_b_gradient")

    test_multiply_backward_b_gradient()
    print("✓ test_multiply_backward_b_gradient")

    test_divide_backward_b_gradient()
    print("✓ test_divide_backward_b_gradient")

    test_add_backward_broadcast_gradient()
    print("✓ test_add_backward_broadcast_gradient")

    test_multiply_backward_broadcast_gradient()
    print("✓ test_multiply_backward_broadcast_gradient")

    test_divide_backward_broadcast_gradient()
    print("✓ test_divide_backward_broadcast_gradient")

    print("\nAll test_arithmetic_backward tests passed!")
