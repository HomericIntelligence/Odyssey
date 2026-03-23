"""Training Script for AlexNet on CIFAR-10

Implements training with manual backward passes (no autograd).
Uses SGD optimizer with momentum and dropout regularization.

Usage:
    mojo run examples/alexnet-cifar10/train.mojo --epochs 100 --batch-size 128 --lr 0.01 --momentum 0.9

Requirements:
    - CIFAR-10 dataset downloaded (run: python examples/alexnet-cifar10/download_cifar10.py)
    - Dataset location: datasets/cifar10/

References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
      ImageNet classification with deep convolutional neural networks.
      Advances in Neural Information Processing Systems, 25, 1097-1105.
"""

from model import AlexNet
from shared.data.datasets import CIFAR10Dataset
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.loss import cross_entropy, cross_entropy_backward
from shared.training.schedulers import step_lr
from shared.utils.arg_parser import create_training_parser
from shared.training.metrics import evaluate_with_predict
from shared.data import DatasetInfo
from collections import List


fn parse_args() raises -> Tuple[Int, Int, Float32, Float32, String, String]:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        Tuple of (epochs, batch_size, learning_rate, momentum, data_dir, weights_dir).
    """
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "alexnet_weights")

    var args = parser.parse()

    var epochs = args.get_int("epochs", 100)
    var batch_size = args.get_int("batch-size", 128)
    var learning_rate = Float32(args.get_float("lr", 0.01))
    var momentum = Float32(args.get_float("momentum", 0.9))
    var data_dir = args.get_string("data-dir", "datasets/cifar10")
    var weights_dir = args.get_string("weights-dir", "alexnet_weights")

    return (epochs, batch_size, learning_rate, momentum, data_dir, weights_dir)


fn compute_gradients(
    mut model: AlexNet,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually through all 8 layers.

    Args:
        model: AlexNet model.
        input: Batch of images (batch, 3, 32, 32).
        labels: Batch of labels (batch,).
        learning_rate: Learning rate for SGD.
        momentum: Momentum factor for SGD.
        velocities: Momentum velocities for each parameter (16 tensors).

    Returns:
        Loss value for this batch.
    """
    # ========== Forward Pass (with caching for backward) ==========

    # Conv1 + ReLU + MaxPool
    var conv1_out = conv2d(
        input, model.conv1_kernel, model.conv1_bias, stride=4, padding=2
    )
    var relu1_out = relu(conv1_out)
    var pool1_out = maxpool2d(relu1_out, kernel_size=3, stride=2, padding=0)

    # Conv2 + ReLU + MaxPool
    var conv2_out = conv2d(
        pool1_out, model.conv2_kernel, model.conv2_bias, stride=1, padding=2
    )
    var relu2_out = relu(conv2_out)
    var pool2_out = maxpool2d(relu2_out, kernel_size=3, stride=2, padding=0)

    # Conv3 + ReLU
    var conv3_out = conv2d(
        pool2_out, model.conv3_kernel, model.conv3_bias, stride=1, padding=1
    )
    var relu3_out = relu(conv3_out)

    # Conv4 + ReLU
    var conv4_out = conv2d(
        relu3_out, model.conv4_kernel, model.conv4_bias, stride=1, padding=1
    )
    var relu4_out = relu(conv4_out)

    # Conv5 + ReLU + MaxPool
    var conv5_out = conv2d(
        relu4_out, model.conv5_kernel, model.conv5_bias, stride=1, padding=1
    )
    var relu5_out = relu(conv5_out)
    var pool3_out = maxpool2d(relu5_out, kernel_size=3, stride=2, padding=0)

    # Flatten
    var pool3_shape = pool3_out.shape()
    var batch_size = pool3_shape[0]
    var flattened_size = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]
    var flatten_shape = List[Int]()
    flatten_shape.append(batch_size)
    flatten_shape.append(flattened_size)
    var flattened = pool3_out.reshape(flatten_shape)

    # FC1 + ReLU + Dropout
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu6_out = relu(fc1_out)
    var drop1_result = dropout(
        relu6_out, Float64(model.dropout_rate), training=True
    )
    var drop1_out = drop1_result[0]  # Dropout output
    var drop1_mask = drop1_result[1]  # Dropout mask for backward

    # FC2 + ReLU + Dropout
    var fc2_out = linear(drop1_out, model.fc2_weights, model.fc2_bias)
    var relu7_out = relu(fc2_out)
    var drop2_result = dropout(
        relu7_out, Float64(model.dropout_rate), training=True
    )
    var drop2_out = drop2_result[0]  # Dropout output
    var drop2_mask = drop2_result[1]  # Dropout mask for backward

    # FC3 (logits)
    var logits = linear(drop2_out, model.fc3_weights, model.fc3_bias)

    # Compute loss
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ========== Backward Pass ==========

    # Start with gradient from loss
    # For cross-entropy, the initial gradient is 1.0
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output._data.bitcast[Float32]()[0] = Float32(1.0)
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    # FC3 backward
    var fc3_grads = linear_backward(grad_logits, drop2_out, model.fc3_weights)
    var grad_drop2_out = fc3_grads.grad_input
    var grad_fc3_weights = fc3_grads.grad_weights
    var grad_fc3_bias = fc3_grads.grad_bias

    # Dropout2 backward
    var grad_relu7_out = dropout_backward(
        grad_drop2_out, drop2_mask, Float64(model.dropout_rate)
    )

    # ReLU7 backward
    var grad_fc2_out = relu_backward(grad_relu7_out, fc2_out)

    # FC2 backward
    var fc2_grads = linear_backward(grad_fc2_out, drop1_out, model.fc2_weights)
    var grad_drop1_out = fc2_grads.grad_input
    var grad_fc2_weights = fc2_grads.grad_weights
    var grad_fc2_bias = fc2_grads.grad_bias

    # Dropout1 backward
    var grad_relu6_out = dropout_backward(
        grad_drop1_out, drop1_mask, Float64(model.dropout_rate)
    )

    # ReLU6 backward
    var grad_fc1_out = relu_backward(grad_relu6_out, fc1_out)

    # FC1 backward
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)
    var grad_flattened = fc1_grads.grad_input
    var grad_fc1_weights = fc1_grads.grad_weights
    var grad_fc1_bias = fc1_grads.grad_bias

    # Unflatten gradient
    var grad_pool3_out = grad_flattened.reshape(pool3_shape)

    # MaxPool3 backward
    var grad_relu5_out = maxpool2d_backward(
        grad_pool3_out, relu5_out, kernel_size=3, stride=2, padding=0
    )

    # ReLU5 backward
    var grad_conv5_out = relu_backward(grad_relu5_out, conv5_out)

    # Conv5 backward
    var conv5_grads = conv2d_backward(
        grad_conv5_out, relu4_out, model.conv5_kernel, stride=1, padding=1
    )
    var grad_relu4_out = conv5_grads.grad_input
    var grad_conv5_kernel = conv5_grads.grad_weights
    var grad_conv5_bias = conv5_grads.grad_bias

    # ReLU4 backward
    var grad_conv4_out = relu_backward(grad_relu4_out, conv4_out)

    # Conv4 backward
    var conv4_grads = conv2d_backward(
        grad_conv4_out, relu3_out, model.conv4_kernel, stride=1, padding=1
    )
    var grad_relu3_out = conv4_grads.grad_input
    var grad_conv4_kernel = conv4_grads.grad_weights
    var grad_conv4_bias = conv4_grads.grad_bias

    # ReLU3 backward
    var grad_conv3_out = relu_backward(grad_relu3_out, conv3_out)

    # Conv3 backward
    var conv3_grads = conv2d_backward(
        grad_conv3_out, pool2_out, model.conv3_kernel, stride=1, padding=1
    )
    var grad_pool2_out = conv3_grads.grad_input
    var grad_conv3_kernel = conv3_grads.grad_weights
    var grad_conv3_bias = conv3_grads.grad_bias

    # MaxPool2 backward
    var grad_relu2_out = maxpool2d_backward(
        grad_pool2_out, relu2_out, kernel_size=3, stride=2, padding=0
    )

    # ReLU2 backward
    var grad_conv2_out = relu_backward(grad_relu2_out, conv2_out)

    # Conv2 backward
    var conv2_grads = conv2d_backward(
        grad_conv2_out, pool1_out, model.conv2_kernel, stride=1, padding=2
    )
    var grad_pool1_out = conv2_grads.grad_input
    var grad_conv2_kernel = conv2_grads.grad_weights
    var grad_conv2_bias = conv2_grads.grad_bias

    # MaxPool1 backward
    var grad_relu1_out = maxpool2d_backward(
        grad_pool1_out, relu1_out, kernel_size=3, stride=2, padding=0
    )

    # ReLU1 backward
    var grad_conv1_out = relu_backward(grad_relu1_out, conv1_out)

    # Conv1 backward
    var conv1_grads = conv2d_backward(
        grad_conv1_out, input, model.conv1_kernel, stride=4, padding=2
    )
    _ = conv1_grads.grad_input  # Not used (no input gradient needed)
    var grad_conv1_kernel = conv1_grads.grad_weights
    var grad_conv1_bias = conv1_grads.grad_bias

    # ========== Parameter Update (SGD with Momentum) ==========
    # Extract velocities from list to avoid aliasing issues
    var vel_conv1_kernel = velocities[0]
    var vel_conv1_bias = velocities[1]
    var vel_conv2_kernel = velocities[2]
    var vel_conv2_bias = velocities[3]
    var vel_conv3_kernel = velocities[4]
    var vel_conv3_bias = velocities[5]
    var vel_conv4_kernel = velocities[6]
    var vel_conv4_bias = velocities[7]
    var vel_conv5_kernel = velocities[8]
    var vel_conv5_bias = velocities[9]
    var vel_fc1_weights = velocities[10]
    var vel_fc1_bias = velocities[11]
    var vel_fc2_weights = velocities[12]
    var vel_fc2_bias = velocities[13]
    var vel_fc3_weights = velocities[14]
    var vel_fc3_bias = velocities[15]

    model.update_parameters(
        learning_rate,
        momentum,
        grad_conv1_kernel^,
        grad_conv1_bias^,
        grad_conv2_kernel^,
        grad_conv2_bias^,
        grad_conv3_kernel^,
        grad_conv3_bias^,
        grad_conv4_kernel^,
        grad_conv4_bias^,
        grad_conv5_kernel^,
        grad_conv5_bias^,
        grad_fc1_weights^,
        grad_fc1_bias^,
        grad_fc2_weights^,
        grad_fc2_bias^,
        grad_fc3_weights^,
        grad_fc3_bias^,
        vel_conv1_kernel,
        vel_conv1_bias,
        vel_conv2_kernel,
        vel_conv2_bias,
        vel_conv3_kernel,
        vel_conv3_bias,
        vel_conv4_kernel,
        vel_conv4_bias,
        vel_conv5_kernel,
        vel_conv5_bias,
        vel_fc1_weights,
        vel_fc1_bias,
        vel_fc2_weights,
        vel_fc2_bias,
        vel_fc3_weights,
        vel_fc3_bias,
    )

    # Update velocities list with modified values
    velocities[0] = vel_conv1_kernel^
    velocities[1] = vel_conv1_bias^
    velocities[2] = vel_conv2_kernel^
    velocities[3] = vel_conv2_bias^
    velocities[4] = vel_conv3_kernel^
    velocities[5] = vel_conv3_bias^
    velocities[6] = vel_conv4_kernel^
    velocities[7] = vel_conv4_bias^
    velocities[8] = vel_conv5_kernel^
    velocities[9] = vel_conv5_bias^
    velocities[10] = vel_fc1_weights^
    velocities[11] = vel_fc1_bias^
    velocities[12] = vel_fc2_weights^
    velocities[13] = vel_fc2_bias^
    velocities[14] = vel_fc3_weights^
    velocities[15] = vel_fc3_bias^

    return loss


fn train_epoch(
    mut model: AlexNet,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    epoch: Int,
    total_epochs: Int,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """Train for one epoch with manual batch iteration.

    Note: Uses manual batch iteration instead of closures to avoid capturing
    non-copyable AlexNet struct in a closure (Mojo limitation).

    Args:
        model: AlexNet model.
        train_images: Training images (50000, 3, 32, 32).
        train_labels: Training labels (50000,).
        batch_size: Mini-batch size.
        learning_rate: Learning rate for SGD.
        momentum: Momentum factor.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        velocities: Momentum velocities for each parameter.

    Returns:
        Average loss for the epoch.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    var total_loss = Float32(0.0)
    var log_interval = 100

    print("Epoch [", epoch, "/", total_epochs, "]")

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)

        # Extract batch slice
        var batch_data = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels = train_labels.slice(start_idx, end_idx, axis=0)

        # Compute gradients and update parameters
        var batch_loss = compute_gradients(
            model,
            batch_data,
            batch_labels,
            learning_rate,
            momentum,
            velocities,
        )
        total_loss += batch_loss

        # Print progress every log_interval batches
        if (batch_idx + 1) % log_interval == 0:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            print(
                "  Batch [",
                batch_idx + 1,
                "/",
                num_batches,
                "] - Loss: ",
                avg_loss,
            )

    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: ", avg_loss)

    return avg_loss


fn evaluate(
    mut model: AlexNet, test_images: AnyTensor, test_labels: AnyTensor
) raises -> Float32:
    """Evaluate model on test set.

    Args:
        model: AlexNet model.
        test_images: Test images (10000, 3, 32, 32).
        test_labels: Test labels (10000,).

    Returns:
        Test accuracy (0.0 to 1.0).
    """
    var num_samples = test_images.shape()[0]
    var correct = 0

    print("Evaluating...")

    # Evaluate on first 100 samples for efficiency
    var eval_samples = min(100, num_samples)

    for i in range(eval_samples):
        # Extract single sample using slice (axis=0 is batch dimension)
        var sample = test_images.slice(i, i + 1, axis=0)
        var pred_class = model.predict(sample)
        var true_label = Int(test_labels[i])

        if pred_class == true_label:
            correct += 1

    var accuracy = Float32(correct) / Float32(eval_samples)
    print(
        "  Test Accuracy: ",
        accuracy * 100.0,
        "% (",
        correct,
        "/",
        eval_samples,
        ")",
    )

    return accuracy


fn initialize_velocities(model: AlexNet) raises -> List[AnyTensor]:
    """Initialize momentum velocities for all parameters (16 tensors).

    Args:
        model: AlexNet model.

    Returns:
        DynamicVector of zero-initialized velocity tensors matching parameter shapes.
    """
    var velocities: List[AnyTensor] = []

    # Initialize velocities for all 16 parameters (conv1-5 + fc1-3, weights + bias)
    velocities.append(zeros(model.conv1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv4_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv5_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc1_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc1_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc2_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc2_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc3_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc3_bias.shape(), DType.float32))

    return velocities^


fn main() raises:
    """Main training loop."""
    print("=" * 60)
    print("AlexNet Training on CIFAR-10 Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config[0]
    var batch_size = config[1]
    var learning_rate = config[2]
    var momentum = config[3]
    var data_dir = config[4]
    var weights_dir = config[5]

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Momentum: ", momentum)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    print()

    # Initialize model
    print("Initializing AlexNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = AlexNet(
        num_classes=dataset_info.num_classes(), dropout_rate=0.5
    )
    print("  Model initialized with", model.num_classes, "classes")
    print("  Dropout rate:", model.dropout_rate)
    print()

    # Initialize momentum velocities
    print("Initializing momentum velocities...")
    var velocities = initialize_velocities(model)
    print("  Velocities initialized for 16 parameters")
    print()

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    var dataset = CIFAR10Dataset(data_dir)
    var train_data = dataset.get_train_data()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Training loop with learning rate decay
    print("Starting training...")
    print("Learning rate schedule: step decay every 30 epochs by 0.1x")
    print()

    for epoch in range(1, epochs + 1):
        # Apply learning rate decay (step every 30 epochs, gamma=0.1)
        var current_lr = step_lr(
            learning_rate, epoch - 1, step_size=30, gamma=Float32(0.1)
        )

        if epoch == 1 or epoch % 30 == 1:
            print("Epoch", epoch, "- Learning rate:", current_lr)

        var train_loss = train_epoch(
            model,
            train_images,
            train_labels,
            batch_size,
            current_lr,
            momentum,
            epoch,
            epochs,
            velocities,
        )

        # Evaluate every epoch
        var test_acc = evaluate(model, test_images, test_labels)
        print()

    # Save model
    print("Saving model weights...")
    model.save_weights(weights_dir)
    print("  Model saved to", weights_dir)
    print()

    print("Training complete!")
    print(
        "\nNote: This implementation demonstrates the full training structure."
    )
    print(
        "Batch processing will be more efficient when tensor slicing is"
        " optimized."
    )
