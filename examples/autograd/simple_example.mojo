"""Simple autograd example demonstrating basic usage.

This example shows how to:
1. Create Variables with gradient tracking
2. Enable gradient tape recording
3. Perform operations (will be recorded when fully implemented)
4. Compute gradients via backward pass
5. Update parameters using an optimizer

Note:
    This is a minimal example showing the API design. Full autograd functionality
    (automatic operation recording and backward pass) is still being implemented.
    Currently, this demonstrates the interface and manual gradient computation.
"""

from projectodyssey.autograd import Variable, GradientTape, SGD
from projectodyssey.autograd.variable import (
    variable_add,
    variable_multiply,
    variable_subtract,
    variable_mean,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones


def simple_linear_regression() raises:
    """Train a simple linear model using manual gradients.

    Model: y = w * x + b

    This example demonstrates the training loop structure that will work
    automatically once autograd operation recording is fully implemented.
    """
    print("=== Simple Linear Regression with Autograd ===\n")

    # Hyperparameters
    var learning_rate: Float64 = 0.01
    var num_epochs: Int = 10

    # Create gradient tape (needed for Variable creation)
    var tape = GradientTape()
    tape.enable()

    # Create parameters
    var w_data = zeros(List[Int](), DType.float32)
    w_data._set_float64(0, 0.5)  # Initialize w = 0.5
    var w = Variable(w_data, requires_grad=True, tape=tape)

    var b_data = zeros(List[Int](), DType.float32)
    b_data._set_float64(0, 0.0)  # Initialize b = 0.0
    var b = Variable(b_data, requires_grad=True, tape=tape)

    print("Initial parameters:")
    print("  w =", w.data._get_float64(0))
    print("  b =", b.data._get_float64(0))
    print()

    # Create simple training data: y = 2*x + 1
    # X = [1, 2, 3, 4, 5]
    # Y = [3, 5, 7, 9, 11]
    var X = AnyTensor(List[Int](), DType.float32)
    for i in range(5):
        X._set_float64(i, Float64(i + 1))

    var Y = AnyTensor(List[Int](), DType.float32)
    for i in range(5):
        Y._set_float64(i, Float64(2 * (i + 1) + 1))

    # Create optimizer
    var optimizer = SGD(learning_rate)

    print("Training for", num_epochs, "epochs...")
    print()

    # Training loop
    for epoch in range(num_epochs):
        # Clear tape for new forward pass
        tape.clear()
        tape.enable()

        # Forward pass: predictions = w * X + b
        # Expand scalar parameters to match X shape for broadcasting
        var w_expanded_data = AnyTensor(List[Int](), DType.float32)
        for i in range(5):
            w_expanded_data._set_float64(i, w.data._get_float64(0))
        var w_expanded = Variable(w_expanded_data, True, tape)

        var b_expanded_data = AnyTensor(List[Int](), DType.float32)
        for i in range(5):
            b_expanded_data._set_float64(i, b.data._get_float64(0))
        var b_expanded = Variable(b_expanded_data, True, tape)

        # Wrap X and Y as Variables
        var X_var = Variable(X, False, tape)
        var Y_var = Variable(Y, False, tape)

        # Use Variable operations (these are recorded in tape)
        var wx = variable_multiply(w_expanded, X_var, tape)
        var predictions = variable_add(wx, b_expanded, tape)

        # Compute loss: MSE = mean((predictions - targets)^2)
        var diff = variable_subtract(predictions, Y_var, tape)
        var squared = variable_multiply(diff, diff, tape)
        var loss = variable_mean(squared, tape, axis=-1)

        # Backward pass using autograd tape
        loss.backward(tape)

        # Extract gradients for w and b from expanded versions
        var grad_w_expanded = tape.get_grad(w_expanded.id)
        var grad_b_expanded = tape.get_grad(b_expanded.id)

        # Sum gradients across batch dimension to get parameter gradients
        var grad_w_sum_data = AnyTensor(List[Int](), DType.float32)
        var grad_w_val = 0.0
        for i in range(5):
            grad_w_val += grad_w_expanded._get_float64(i)
        grad_w_sum_data._set_float64(0, grad_w_val)

        var grad_b_sum_data = AnyTensor(List[Int](), DType.float32)
        var grad_b_val = 0.0
        for i in range(5):
            grad_b_val += grad_b_expanded._get_float64(i)
        grad_b_sum_data._set_float64(0, grad_b_val)

        # Set gradients in tape for actual parameters
        tape.registry.set_grad(w.id, grad_w_sum_data)
        tape.registry.set_grad(b.id, grad_b_sum_data)

        # Update parameters using optimizer
        var params: List[Variable] = []
        params.append(w.copy())
        params.append(b.copy())
        optimizer.step(params, tape)

        # Reset gradients
        optimizer.zero_grad(tape)

        # Print progress
        var loss_value = loss.data._get_float64(0)
        var w_value = w.data._get_float64(0)
        var b_value = b.data._get_float64(0)

        print("Epoch", epoch + 1, "/", num_epochs)
        print("  Loss:", loss_value)
        print("  w:", w_value)
        print("  b:", b_value)
        print()

    print("Training complete!")
    print()
    print("Final parameters (target: w=2.0, b=1.0):")
    print("  w =", w.data._get_float64(0))
    print("  b =", b.data._get_float64(0))
    print()


def main() raises:
    """Run autograd examples."""
    simple_linear_regression()
