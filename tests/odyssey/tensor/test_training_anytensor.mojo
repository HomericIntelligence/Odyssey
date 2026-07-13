"""Tests for training infrastructure with AnyTensor.

Tests verify that Variable wraps AnyTensor correctly and that the autograd
infrastructure (Variable creation, backward, detach) works with AnyTensor
as the underlying data type.

Tests cover:
- Variable.data is AnyTensor
- Variable preserves shape, numel, dtype through AnyTensor
- Variable backward with AnyTensor data
- Variable detach returns AnyTensor
- Multiple Variables with AnyTensor in same tape
"""

from std.testing import assert_true
from tests.odyssey.conftest import assert_almost_equal
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones
from odyssey.autograd import Variable, GradientTape
from odyssey.autograd.variable import variable_add, variable_sum


def test_variable_data_is_anytensor() raises:
    """Variable.data stores AnyTensor correctly."""
    var data: AnyTensor = zeros([4, 3], DType.float32)
    var tape = GradientTape()
    tape.enable()
    var v = Variable(data, True, tape)
    assert_true(v.data.numel() == 12, "variable data has 12 elements")
    assert_true(
        v.data.dtype() == DType.float32, "variable data dtype preserved"
    )
    print("PASS: test_variable_data_is_anytensor")


def test_variable_shape_from_anytensor() raises:
    """Variable.shape() returns shape from underlying AnyTensor."""
    var data: AnyTensor = zeros([2, 3, 4], DType.float32)
    var tape = GradientTape()
    tape.enable()
    var v = Variable(data, True, tape)
    var s = v.shape()
    assert_true(len(s) == 3, "3D shape")
    assert_true(s[0] == 2, "dim 0")
    assert_true(s[1] == 3, "dim 1")
    assert_true(s[2] == 4, "dim 2")
    print("PASS: test_variable_shape_from_anytensor")


def test_variable_detach_returns_anytensor() raises:
    """Variable.detach() returns the underlying AnyTensor."""
    var data: AnyTensor = zeros([3], DType.float32)
    data._set_float64(0, 1.0)
    data._set_float64(1, 0.5)
    data._set_float64(2, -0.5)
    var tape = GradientTape()
    tape.enable()
    var v = Variable(data, True, tape)
    var detached = v.detach()
    assert_true(detached.numel() == 3, "detached has 3 elements")
    assert_almost_equal(Float64(detached._get_float64(0)), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(detached._get_float64(1)), 0.5, tolerance=1e-6)
    print("PASS: test_variable_detach_returns_anytensor")


def test_variable_backward_with_anytensor() raises:
    """Variable backward computes gradients with AnyTensor data."""
    var tape = GradientTape()
    tape.enable()

    var x_data: AnyTensor = zeros([2], DType.float32)
    x_data._set_float64(0, 1.0)
    x_data._set_float64(1, 0.5)
    var x = Variable(x_data, True, tape)

    var y_data: AnyTensor = zeros([2], DType.float32)
    y_data._set_float64(0, 0.5)
    y_data._set_float64(1, 1.0)
    var y = Variable(y_data, True, tape)

    # z = x + y, loss = sum(z)
    var z = variable_add(x, y, tape)
    var loss = variable_sum(z, tape)

    loss.backward(tape)

    # d(sum(x+y))/dx = [1, 1]
    var grad_x = tape.registry.get_grad(x.id)
    assert_true(grad_x.numel() == 2, "grad_x has 2 elements")
    assert_almost_equal(Float64(grad_x._get_float64(0)), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(grad_x._get_float64(1)), 1.0, tolerance=1e-6)
    print("PASS: test_variable_backward_with_anytensor")


def test_multiple_variables_anytensor() raises:
    """Multiple Variables with AnyTensor data in same tape."""
    var tape = GradientTape()
    tape.enable()

    var a_data: AnyTensor = ones([3], DType.float32)
    var b_data: AnyTensor = zeros([3], DType.float32)
    var c_data: AnyTensor = ones([2, 2], DType.float32)

    var a = Variable(a_data, True, tape)
    var b = Variable(b_data, False, tape)
    var c = Variable(c_data, True, tape)

    assert_true(a.data.numel() == 3, "a has 3 elements")
    assert_true(b.data.numel() == 3, "b has 3 elements")
    assert_true(c.data.numel() == 4, "c has 4 elements")
    assert_true(a.requires_grad == True, "a requires grad")
    assert_true(b.requires_grad == False, "b does not require grad")
    assert_true(c.requires_grad == True, "c requires grad")
    # Each variable should have a unique ID
    assert_true(a.id != b.id, "a and b have different ids")
    assert_true(b.id != c.id, "b and c have different ids")
    print("PASS: test_multiple_variables_anytensor")


def test_variable_numel_dtype_from_anytensor() raises:
    """Variable.numel() and Variable.dtype() delegate to AnyTensor."""
    var data: AnyTensor = zeros([5, 3], DType.float32)
    var tape = GradientTape()
    tape.enable()
    var v = Variable(data, True, tape)
    assert_true(v.numel() == 15, "numel delegates to AnyTensor")
    assert_true(v.dtype() == DType.float32, "dtype delegates to AnyTensor")
    print("PASS: test_variable_numel_dtype_from_anytensor")


def main() raises:
    test_variable_data_is_anytensor()
    test_variable_shape_from_anytensor()
    test_variable_detach_returns_anytensor()
    test_variable_backward_with_anytensor()
    test_multiple_variables_anytensor()
    test_variable_numel_dtype_from_anytensor()
    print("\n6 training AnyTensor tests passed\n")
