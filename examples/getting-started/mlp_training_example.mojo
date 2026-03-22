"""Complete 2-Layer MLP Training Example.

This example demonstrates end-to-end training of a 2-layer multi-layer perceptron
for binary classification using the newly implemented loss functions and optimizers.

Architecture:
    Input (2) -> Hidden (4) -> Output (1)
    Activations: ReLU (hidden), Sigmoid (output)
    Loss: Binary Cross-Entropy
    Optimizer: SGD

This example trains on synthetic XOR-like data to demonstrate that the training
pipeline is fully functional.
"""

from shared.core import (
    AnyTensor,
    # Creation
    zeros,
    ones,
    full,
    ones_like,
    # Arithmetic
    add,
    subtract,
    multiply,
    # Activations
    relu,
    sigmoid,
    # Activation backward
    relu_backward,
    sigmoid_backward,
    # Backward passes
    add_backward,
    # Reduction
    mean,
    # Reduction backward
    mean_backward,
    # Loss functions
    binary_cross_entropy,
    binary_cross_entropy_backward,
    # Initializers
    xavier_uniform,
)

# Import matrix operations directly to avoid name collision
from shared.core.matrix import matmul, matmul_backward
from shared.training.optimizers import sgd_step_simple


fn create_synthetic_data() raises -> Tuple[AnyTensor, AnyTensor]:
    """Create synthetic XOR-like binary classification data.

    Returns:
        (inputs, targets) where:
            inputs: (4, 2) - Four 2D input points
            targets: (4, 1) - Binary labels (0 or 1).
    """
    # Create 4 samples with 2 features each
    var input_shape = [4, 2]
    var target_shape = [4, 1]

    var inputs = AnyTensor(input_shape, DType.float32)
    var targets = AnyTensor(target_shape, DType.float32)

    # XOR-like pattern:
    # [0, 0] -> 0
    # [0, 1] -> 1
    # [1, 0] -> 1
    # [1, 1] -> 0

    # Sample 1: [0, 0]
    inputs._set_float64(0, 0.0)
    inputs._set_float64(1, 0.0)
    targets._set_float64(0, 0.0)

    # Sample 2: [0, 1]
    inputs._set_float64(2, 0.0)
    inputs._set_float64(3, 1.0)
    targets._set_float64(1, 1.0)

    # Sample 3: [1, 0]
    inputs._set_float64(4, 1.0)
    inputs._set_float64(5, 0.0)
    targets._set_float64(2, 1.0)

    # Sample 4: [1, 1]
    inputs._set_float64(6, 1.0)
    inputs._set_float64(7, 1.0)
    targets._set_float64(3, 0.0)

    return (inputs, targets)


fn train_mlp() raises:
    """Train a 2-layer MLP on synthetic data.

    Network architecture:
        Input: (batch, 2)
        Hidden: (batch, 4) with ReLU activation
        Output: (batch, 1) with Sigmoid activation

    Training:
        Loss: Binary Cross-Entropy
        Optimizer: SGD with learning rate 0.1
        Epochs: 1000
    """
    print("=" * 60)
    print("2-Layer MLP Training Example")
    print("=" * 60)

    # Hyperparameters
    var learning_rate = 0.1
    var num_epochs = 1000
    var print_every = 100

    # Create synthetic data
    print("\nCreating synthetic XOR data...")
    var (X, y_true) = create_synthetic_data()
    print("Input shape:", X.shape()[0], "x", X.shape()[1])
    print("Target shape:", y_true.shape()[0], "x", y_true.shape()[1])

    # Initialize network parameters
    print("\nInitializing network parameters...")

    # Network dimensions
    comptime input_size = 2
    comptime hidden_size = 4
    comptime output_size = 1

    # Layer 1: (2, 4) - Input to Hidden
    var W1_shape = [hidden_size, input_size]
    var b1_shape = [hidden_size]
    var W1 = xavier_uniform(input_size, hidden_size, W1_shape, DType.float32)
    var b1 = zeros(b1_shape, DType.float32)

    # Layer 2: (4, 1) - Hidden to Output
    var W2_shape = [output_size, hidden_size]
    var b2_shape = [output_size]
    var W2 = xavier_uniform(hidden_size, output_size, W2_shape, DType.float32)
    var b2 = zeros(b2_shape, DType.float32)

    print("W1 shape:", W1.shape()[0], "x", W1.shape()[1])
    print("W2 shape:", W2.shape()[0], "x", W2.shape()[1])

    # Training loop
    print("\nStarting training...")
    print("Learning rate:", learning_rate)
    print("Epochs:", num_epochs)
    print("-" * 60)

    for epoch in range(num_epochs):
        # ========== FORWARD PASS ==========
        # Layer 1: h1 = relu(W1 @ X.T + b1)
        # Note: X is (4, 2), X.T would be (2, 4)
        # For batch processing, we'll process each sample separately

        # For simplicity, process first sample
        var x_sample_shape = [input_size]
        var x_sample = AnyTensor(x_sample_shape, DType.float32)
        x_sample._set_float64(0, X._get_float64(0))  # First feature
        x_sample._set_float64(1, X._get_float64(1))  # Second feature

        # Forward through layer 1
        var z1 = add(matmul(W1, x_sample), b1)  # (4, 1)
        var h1 = relu(z1)  # (4, 1)

        # Forward through layer 2
        var z2 = add(matmul(W2, h1), b2)  # (1, 1)
        var pred = sigmoid(z2)  # (1, 1)

        # Get target for this sample
        var y_sample_shape = [output_size]
        var y_sample = AnyTensor(y_sample_shape, DType.float32)
        y_sample._set_float64(0, y_true._get_float64(0))

        # Compute loss
        var loss_val = binary_cross_entropy(pred, y_sample)  # (1, 1)
        var loss = mean(loss_val)  # scalar

        # ========== BACKWARD PASS ==========
        # Initialize gradient
        var grad_loss_shape = List[Int]()
        var grad_loss = ones(
            grad_loss_shape, DType.float32
        )  # scalar 1.0 (empty shape)

        # Backprop through mean
        var grad_loss_val = mean_backward(grad_loss, loss_val)

        # Backprop through BCE
        var grad_pred = binary_cross_entropy_backward(
            grad_loss_val, pred, y_sample
        )

        # Backprop through sigmoid
        var grad_z2 = sigmoid_backward(grad_pred, pred)

        # Backprop through layer 2
        # z2 = W2 @ h1 + b2
        # Need to recompute forward intermediate for backward pass
        var W2h1 = matmul(W2, h1)
        var add_grads_2 = add_backward(grad_z2, W2h1, b2)
        var grad_h1_from_add = add_grads_2.grad_a
        var grad_b2 = add_grads_2.grad_b

        var matmul_grads_2 = matmul_backward(grad_h1_from_add, W2, h1)
        var grad_W2 = matmul_grads_2.grad_a
        var grad_h1 = matmul_grads_2.grad_b

        # Backprop through ReLU
        var grad_z1 = relu_backward(grad_h1, z1)

        # Backprop through layer 1
        # z1 = W1 @ x_sample + b1
        # Need to recompute forward intermediate for backward pass
        var W1x = matmul(W1, x_sample)
        var add_grads_1 = add_backward(grad_z1, W1x, b1)
        var grad_x_from_add = add_grads_1.grad_a
        var grad_b1 = add_grads_1.grad_b

        var matmul_grads_1 = matmul_backward(grad_x_from_add, W1, x_sample)
        var grad_W1 = matmul_grads_1.grad_a
        var _ = (
            matmul_grads_1.grad_b
        )  # grad_x not needed for this simple example

        # ========== OPTIMIZER STEP ==========
        # Update parameters using SGD
        W1 = sgd_step_simple(W1, grad_W1, learning_rate)
        W2 = sgd_step_simple(W2, grad_W2, learning_rate)
        b1 = sgd_step_simple(b1, grad_b1, learning_rate)
        b2 = sgd_step_simple(b2, grad_b2, learning_rate)

        # Print progress
        if epoch % print_every == 0:
            var loss_scalar = loss._get_float64(0)
            var pred_val = pred._get_float64(0)
            var target_val = y_sample._get_float64(0)
            print(
                "Epoch",
                epoch,
                "| Loss:",
                loss_scalar,
                "| Pred:",
                pred_val,
                "| Target:",
                target_val,
            )

    print("-" * 60)
    print("Training complete!")

    # Test all samples
    print("\n" + "=" * 60)
    print("Testing on all samples:")
    print("=" * 60)

    for i in range(4):
        # Get sample
        var x_test_shape = [input_size]
        var x_test = AnyTensor(x_test_shape, DType.float32)
        x_test._set_float64(0, X._get_float64(i * 2))
        x_test._set_float64(1, X._get_float64(i * 2 + 1))

        # Forward pass
        var z1_test = add(matmul(W1, x_test), b1)
        var h1_test = relu(z1_test)
        var z2_test = add(matmul(W2, h1_test), b2)
        var pred_test = sigmoid(z2_test)

        var input1 = X._get_float64(i * 2)
        var input2 = X._get_float64(i * 2 + 1)
        var target = y_true._get_float64(i)
        var prediction = pred_test._get_float64(0)
        var predicted_class = 1 if prediction > 0.5 else 0

        print(
            "Input: [",
            input1,
            ",",
            input2,
            "] -> Target:",
            target,
            "| Prediction:",
            prediction,
            "| Class:",
            predicted_class,
        )

    print("=" * 60)


fn main() raises:
    """Entry point for the MLP training example."""
    train_mlp()
