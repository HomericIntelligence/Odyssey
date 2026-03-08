# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checking.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Gradient checking tests for activation, arithmetic, and edge cases.

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import assert_true
from shared.testing import check_gradients
from shared.core import ExTensor, zeros, ones, full
from shared.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
)
from shared.core.arithmetic import (
    add,
    multiply,
    add_backward,
    multiply_backward,
)


fn test_relu_gradient() raises:
    """Test ReLU backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 2.0, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed")


fn test_relu_negative_inputs() raises:
    """Test ReLU gradient with negative inputs (zero gradient region)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, -2.0, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed for negative inputs")


fn test_relu_mixed_inputs() raises:
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

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed for mixed inputs")


fn test_sigmoid_gradient() raises:
    """Test Sigmoid backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 0.5, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return sigmoid(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var output = sigmoid(x)
        return sigmoid_backward(grad_out, output)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Sigmoid gradient check failed")


fn test_tanh_gradient() raises:
    """Test Tanh backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input = full(shape, 0.5, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return tanh(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var output = tanh(x)
        return tanh_backward(grad_out, output)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Tanh gradient check failed")


fn test_add_gradient() raises:
    """Test addition backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input_a = ones(shape, DType.float32)
    var input_b = ones(shape, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return add(x, input_b)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var grads = add_backward(grad_out, x, input_b)
        return grads.grad_a

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Add gradient check failed")


fn test_multiply_gradient() raises:
    """Test multiplication backward pass using gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input_a = full(shape, 2.0, DType.float32)
    var input_b = full(shape, 3.0, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return multiply(x, input_b)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var grads = multiply_backward(grad_out, x, input_b)
        return grads.grad_a

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Multiply gradient check failed")


fn test_gradient_at_zero() raises:
    """Test gradient checking near zero."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var input = full(shape, 0.01, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Gradient near zero check failed")


fn test_gradient_small_tensor() raises:
    """Test gradient checking on very small tensors (1x1)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)
    var input = full(shape, 2.0, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Small tensor gradient check failed")


fn main() raises:
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
