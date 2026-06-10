"""Tests for Variable.backward() method for backpropagation.

This module tests the backward() API for automatic differentiation.

Tests cover:
- Basic scalar loss backward pass
- Gradient flow to input variables
- Gradient accumulation across multiple operations
- Chain rule gradient computation
"""

from tests.projectodyssey.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, ones_like
from projectodyssey.autograd.variable import (
    Variable,
    variable_add,
    variable_multiply,
    variable_sum,
)
from projectodyssey.autograd.tape import GradientTape


# ============================================================================
# Test Helpers
# ============================================================================


def create_shape_1d(size: Int) -> List[Int]:
    """Create a 1D shape list."""
    var shape = List[Int]()
    shape.append(size)
    return shape^


def create_shape_2d(rows: Int, cols: Int) -> List[Int]:
    """Create a 2D shape list."""
    var shape = List[Int]()
    shape.append(rows)
    shape.append(cols)
    return shape^


# ============================================================================
# Test Cases
# ============================================================================


def test_variable_backward_scalar_loss() raises:
    """Test backward() on a scalar loss."""
    # Create tape and enable recording
    var tape = GradientTape()
    tape.enable()

    # Create input variables
    var shape = create_shape_1d(1)
    var x_data = ones(shape, DType.float32)
    x_data.set(0, Float32(2.0))

    var x = Variable(x_data, True, tape)

    # Forward pass: compute loss = x * x
    var y = variable_multiply(x, x, tape)
    var loss = variable_sum(y, tape)

    # Backward pass
    loss.backward(tape)

    # Check gradients were computed
    var grad_x = tape.get_grad(x.id)
    assert_equal_int(grad_x.numel(), 1)

    # For loss = (x*x).sum() where x=2, grad_x = 2*x*2 = 8
    var grad_val = grad_x._data.bitcast[Float32]()[0]
    assert_almost_equal(grad_val, 4.0, tolerance=1e-5)


def test_variable_backward_multiple_variables() raises:
    """Test gradient flow to multiple input variables."""
    # Create tape and enable recording
    var tape = GradientTape()
    tape.enable()

    # Create two input variables
    var shape = create_shape_1d(2)
    var x_data = ones(shape, DType.float32)
    x_data.set(0, Float32(1.0))
    x_data.set(1, Float32(2.0))

    var y_data = ones(shape, DType.float32)
    y_data.set(0, Float32(3.0))
    y_data.set(1, Float32(4.0))

    var x = Variable(x_data, True, tape)
    var y = Variable(y_data, True, tape)

    # Forward pass: loss = (x + y).sum()
    var z = variable_add(x, y, tape)
    var loss = variable_sum(z, tape)

    # Backward pass
    loss.backward(tape)

    # Check gradients for both inputs
    var grad_x = tape.get_grad(x.id)
    var grad_y = tape.get_grad(y.id)

    # For loss = (x+y).sum(), grad_x and grad_y should be 1 for each element
    assert_equal_int(grad_x.numel(), 2)
    assert_equal_int(grad_y.numel(), 2)

    # Check first element
    var grad_x_0 = grad_x._data.bitcast[Float32]()[0]
    var grad_y_0 = grad_y._data.bitcast[Float32]()[0]
    assert_almost_equal(grad_x_0, 1.0, tolerance=1e-5)
    assert_almost_equal(grad_y_0, 1.0, tolerance=1e-5)


def test_variable_backward_chain_rule() raises:
    """Test gradient computation via chain rule."""
    # Create tape and enable recording
    var tape = GradientTape()
    tape.enable()

    # Create input variable
    var shape = create_shape_1d(1)
    var x_data = ones(shape, DType.float32)
    x_data.set(0, Float32(3.0))

    var x = Variable(x_data, True, tape)

    # Forward pass: loss = (x * x) * (x * x) = x^4
    var y = variable_multiply(x, x, tape)
    var z = variable_multiply(y, y, tape)
    var loss = variable_sum(z, tape)

    # Backward pass
    loss.backward(tape)

    # Check gradient at x
    var grad_x = tape.get_grad(x.id)
    assert_equal_int(grad_x.numel(), 1)

    # loss = ((x*x)^2) = x^4, so d(loss)/dx = 4*x^3 = 4*27 = 108
    var grad_val = grad_x._data.bitcast[Float32]()[0]
    assert_almost_equal(grad_val, 108.0, tolerance=1e-3)


def test_variable_backward_independent_tapes() raises:
    """Test that independent tapes produce consistent results."""
    # Test case 1: first tape
    var tape1 = GradientTape()
    tape1.enable()

    var shape = create_shape_1d(2)
    var x1_data = ones(shape, DType.float32)
    x1_data.set(0, Float32(2.0))
    x1_data.set(1, Float32(3.0))

    var x1 = Variable(x1_data, True, tape1)
    var y1 = variable_multiply(x1, x1, tape1)
    var loss1 = variable_sum(y1, tape1)

    # Backward with tape1
    loss1.backward(tape1)

    # Test case 2: second tape with same computation
    var tape2 = GradientTape()
    tape2.enable()

    var x2_data = ones(shape, DType.float32)
    x2_data.set(0, Float32(2.0))
    x2_data.set(1, Float32(3.0))

    var x2 = Variable(x2_data, True, tape2)
    var y2 = variable_multiply(x2, x2, tape2)
    var loss2 = variable_sum(y2, tape2)

    # Backward with tape2
    loss2.backward(tape2)

    # Compare gradients - should be identical
    var grad1_x1 = tape1.get_grad(x1.id)
    var grad2_x2 = tape2.get_grad(x2.id)

    # Both should have same values
    assert_equal_int(grad1_x1.numel(), grad2_x2.numel())

    var grad1_0 = grad1_x1._data.bitcast[Float32]()[0]
    var grad2_0 = grad2_x2._data.bitcast[Float32]()[0]

    assert_almost_equal(grad1_0, grad2_0, tolerance=1e-5)


def test_variable_backward_no_gradients_required() raises:
    """Test backward pass with some variables not requiring gradients."""
    # Create tape and enable recording
    var tape = GradientTape()
    tape.enable()

    # Create variables
    var shape = create_shape_1d(1)
    var x_data = ones(shape, DType.float32)
    x_data.set(0, Float32(2.0))

    var c_data = ones(shape, DType.float32)
    c_data.set(0, Float32(5.0))

    # x requires gradients, c does not
    var x = Variable(x_data, True, tape)
    var c = Variable(c_data, False, tape)

    # Forward pass: loss = (x * c).sum()
    var y = variable_multiply(x, c, tape)
    var loss = variable_sum(y, tape)

    # Backward pass
    loss.backward(tape)

    # Check gradient for x exists
    var grad_x = tape.get_grad(x.id)
    assert_equal_int(grad_x.numel(), 1)

    # Gradient should be c = 5
    var grad_val = grad_x._data.bitcast[Float32]()[0]
    assert_almost_equal(grad_val, 5.0, tolerance=1e-5)


# ============================================================================
# Main
# ============================================================================


def main() raises:
    """Run all Variable backward tests."""
    print("Running Variable backward tests...")
    print("")

    test_variable_backward_scalar_loss()
    print("✓ test_variable_backward_scalar_loss")

    test_variable_backward_multiple_variables()
    print("✓ test_variable_backward_multiple_variables")

    test_variable_backward_chain_rule()
    print("✓ test_variable_backward_chain_rule")

    test_variable_backward_independent_tapes()
    print("✓ test_variable_backward_independent_tapes")

    test_variable_backward_no_gradients_required()
    print("✓ test_variable_backward_no_gradients_required")

    print("")
    print("All Variable backward tests passed!")
