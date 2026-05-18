"""Gradient checking tests for activation, arithmetic, and edge cases."""

from projectodyssey.testing.gradient_checker import (
    check_gradient,
    NumericalForward,
    NumericalBackward,
)
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    zeros_like,
)
from projectodyssey.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
)
from projectodyssey.core.arithmetic import (
    add,
    multiply,
    add_backward,
    multiply_backward,
)


# ---- ReLU (no captures) ----


@fieldwise_init
struct _ReluFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return relu(x)


@fieldwise_init
struct _ReluBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return relu_backward(grad_out, x)


# ---- Sigmoid (no captures) ----


@fieldwise_init
struct _SigmoidFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return sigmoid(x)


@fieldwise_init
struct _SigmoidBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var output = sigmoid(x)
        return sigmoid_backward(grad_out, output)


# ---- Tanh (no captures) ----


@fieldwise_init
struct _TanhFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return tanh(x)


@fieldwise_init
struct _TanhBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var output = tanh(x)
        return tanh_backward(grad_out, output)


# ---- Add (captures input_b) ----


@fieldwise_init
struct _AddFwd(NumericalForward):
    var input_b: AnyTensor

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return add(x, self.input_b)


@fieldwise_init
struct _AddBwd(NumericalBackward):
    var input_b: AnyTensor

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, x, self.input_b)
        return grads.grad_a


# ---- Multiply (captures input_b) ----


@fieldwise_init
struct _MultiplyFwd(NumericalForward):
    var input_b: AnyTensor

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return multiply(x, self.input_b)


@fieldwise_init
struct _MultiplyBwd(NumericalBackward):
    var input_b: AnyTensor

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, x, self.input_b)
        return grads.grad_a


def _ones_grad(output: AnyTensor) raises -> AnyTensor:
    """Create ones grad_output matching output shape."""
    var grad_output = zeros_like(output)
    for i in range(grad_output.numel()):
        grad_output._set_float64(i, 1.0)
    return grad_output^


def test_relu_gradient() raises:
    """Test ReLU backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 2.0, DType.float32)
    var fwd = _ReluFwd()
    check_gradient(fwd, _ReluBwd(), input, _ones_grad(fwd(input)))


def test_relu_negative_inputs() raises:
    """Test ReLU gradient with negative inputs (zero gradient region)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, -2.0, DType.float32)
    var fwd = _ReluFwd()
    check_gradient(fwd, _ReluBwd(), input, _ones_grad(fwd(input)))


def test_relu_mixed_inputs() raises:
    """Test ReLU gradient with mixed positive/negative inputs."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var input = zeros(shape, DType.float32)
    # Set some positive, some negative (avoid 0.0 at ReLU discontinuity)
    input._set_float64(0, 1.0)
    input._set_float64(1, -1.0)
    input._set_float64(2, 2.0)
    input._set_float64(3, -2.0)
    input._set_float64(4, 1.5)
    input._set_float64(5, -1.5)
    input._set_float64(6, 0.5)
    input._set_float64(7, -0.5)
    input._set_float64(8, 3.0)
    input._set_float64(9, -3.0)
    input._set_float64(10, 0.1)
    input._set_float64(11, -0.1)

    var fwd = _ReluFwd()
    check_gradient(fwd, _ReluBwd(), input, _ones_grad(fwd(input)))


def test_sigmoid_gradient() raises:
    """Test Sigmoid backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 0.5, DType.float32)
    var fwd = _SigmoidFwd()
    check_gradient(fwd, _SigmoidBwd(), input, _ones_grad(fwd(input)))


def test_tanh_gradient() raises:
    """Test Tanh backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 0.5, DType.float32)
    var fwd = _TanhFwd()
    check_gradient(fwd, _TanhBwd(), input, _ones_grad(fwd(input)))


def test_add_gradient() raises:
    """Test addition backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input_a = ones(shape, DType.float32)
    var input_b = ones(shape, DType.float32)
    var fwd = _AddFwd(input_b)
    check_gradient(fwd, _AddBwd(input_b), input_a, _ones_grad(fwd(input_a)))


def test_multiply_gradient() raises:
    """Test multiplication backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input_a = full(shape, 2.0, DType.float32)
    var input_b = full(shape, 3.0, DType.float32)
    var fwd = _MultiplyFwd(input_b)
    check_gradient(
        fwd, _MultiplyBwd(input_b), input_a, _ones_grad(fwd(input_a))
    )


def test_gradient_at_zero() raises:
    """Test gradient checking near zero."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var input = full(shape, 0.01, DType.float32)
    var fwd = _ReluFwd()
    check_gradient(fwd, _ReluBwd(), input, _ones_grad(fwd(input)))


def test_gradient_small_tensor() raises:
    """Test gradient checking on very small tensors (1x1)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)
    var input = full(shape, 2.0, DType.float32)
    var fwd = _ReluFwd()
    check_gradient(fwd, _ReluBwd(), input, _ones_grad(fwd(input)))


def main() raises:
    """Run basic gradient checking tests."""
    print("Running activation gradient tests...")
    test_relu_gradient()
    test_relu_negative_inputs()
    test_relu_mixed_inputs()
    test_sigmoid_gradient()
    test_tanh_gradient()

    print("Running arithmetic gradient tests...")
    test_add_gradient()
    test_multiply_gradient()

    print("Running edge case gradient tests...")
    test_gradient_at_zero()
    test_gradient_small_tensor()

    print("All basic gradient checking tests passed!")
