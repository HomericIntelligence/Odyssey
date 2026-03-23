"""Linear regression example using functional autograd helpers.

This example demonstrates the clean, concise API enabled by:
1. mse_loss_and_grad() - Compute loss and gradient in one call
2. multiply_scalar() - Efficient scalar-tensor multiplication
3. apply_gradient() - Clean parameter updates

Model: y = w * x + b

Training: Learn w and b to fit the data y = 2x + 1
"""

from shared.autograd import mse_loss_and_grad, apply_gradient, multiply_scalar
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.core.arithmetic import add, multiply
from shared.core.reduction import sum as tensor_sum


fn linear_regression_functional() raises:
    """Train a simple linear model using functional autograd helpers.

    This demonstrates the recommended approach: use loss_and_grad helpers
    rather than manual gradient computation.
    """
    print("=== Linear Regression with Functional Autograd ===\n")

    # Hyperparameters
    var learning_rate: Float64 = 0.01
    var num_epochs: Int = 50

    # Create parameters: w and b
    var w = AnyTensor(List[Int](), DType.float32)
    w._set_float64(0, 0.5)  # Initialize w = 0.5

    var b = AnyTensor(List[Int](), DType.float32)
    b._set_float64(0, 0.0)  # Initialize b = 0.0

    print("Initial parameters:")
    print("  w =", w._get_float64(0))
    print("  b =", b._get_float64(0))
    print()

    # Create training data: y = 2*x + 1
    # X = [1, 2, 3, 4, 5]
    # Y = [3, 5, 7, 9, 11]
    var X = AnyTensor(List[Int](), DType.float32)
    for i in range(5):
        X._set_float64(i, Float64(i + 1))

    var Y = AnyTensor(List[Int](), DType.float32)
    for i in range(5):
        Y._set_float64(i, Float64(2 * (i + 1) + 1))

    print("Training for", num_epochs, "epochs...")
    print()

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass: predictions = w * X + b
        var w_val = w._get_float64(0)
        var b_val = b._get_float64(0)

        var predictions = AnyTensor(List[Int](), DType.float32)
        for i in range(5):
            var pred = w_val * X._get_float64(i) + b_val
            predictions._set_float64(i, pred)

        # Compute loss and gradient in one call!
        var result = mse_loss_and_grad(predictions, Y)
        var loss = result.loss
        var grad_predictions = result.grad

        # Compute gradients w.r.t. parameters
        # ∂loss/∂w = sum(grad_predictions * X)
        var grad_w_elements = multiply(grad_predictions, X)
        var grad_w = tensor_sum(grad_w_elements, axis=-1, keepdims=False)

        # ∂loss/∂b = sum(grad_predictions)
        var grad_b = tensor_sum(grad_predictions, axis=-1, keepdims=False)

        # Update parameters using helper functions
        w = apply_gradient(w, grad_w, learning_rate)
        b = apply_gradient(b, grad_b, learning_rate)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            var loss_value = loss._get_float64(0)
            var w_value = w._get_float64(0)
            var b_value = b._get_float64(0)

            print("Epoch", epoch + 1, "/", num_epochs)
            print("  Loss:", loss_value)
            print("  w:", w_value, " (target: 2.0)")
            print("  b:", b_value, " (target: 1.0)")
            print()

    print("Training complete!")
    print()
    print("Final parameters (target: w=2.0, b=1.0):")
    print("  w =", w._get_float64(0))
    print("  b =", b._get_float64(0))
    print()

    # Test predictions
    print("Test predictions:")
    for i in range(5):
        var x_val = X._get_float64(i)
        var y_true = Y._get_float64(i)
        var y_pred = w._get_float64(0) * x_val + b._get_float64(0)
        print("  X =", x_val, "  Y_true =", y_true, "  Y_pred =", y_pred)


fn main() raises:
    """Run autograd examples."""
    linear_regression_functional()
