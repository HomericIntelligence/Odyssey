"""Training Script for LeNet-5 on EMNIST.

Implements training with manual backward passes (no autograd).
Uses SGD optimizer with simple training loop.

Usage:
    mojo run examples/lenet_emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01

Requirements:
    - EMNIST dataset downloaded (run: python scripts/download_emnist.py)
    - Dataset location: datasets/emnist/

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import (
    LeNet5,
    # Architecture constants for consistent forward/backward passes
    CONV1_STRIDE,
    CONV1_PADDING,
    CONV2_STRIDE,
    CONV2_PADDING,
    POOL1_KERNEL_SIZE,
    POOL1_STRIDE,
    POOL1_PADDING,
    POOL2_KERNEL_SIZE,
    POOL2_STRIDE,
    POOL2_PADDING,
)
from projectodyssey.data.formats import (
    load_idx_images,
    load_idx_labels,
    normalize_images,
    one_hot_encode,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import maxpool2d, maxpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.utils.arg_parser import create_training_parser
from projectodyssey.training.loops import TrainingLoop
from projectodyssey.training.optimizers import (
    sgd_step_simple,
    lion_step,
    shampoo_step,
)
from projectodyssey.training.metrics import (
    top1_accuracy,
    AccuracyMetric,
    LossTracker,
)
from projectodyssey.training.evaluation import evaluate_model_simple
from std.collections import List

# Default number of classes for EMNIST Balanced dataset
comptime DEFAULT_NUM_CLASSES = 47


struct TrainConfig:
    """Training configuration from command line arguments."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var data_dir: String
    var weights_dir: String
    var optimizer_name: String

    def __init__(
        out self,
        epochs: Int,
        batch_size: Int,
        learning_rate: Float32,
        data_dir: String,
        weights_dir: String,
        optimizer_name: String = "sgd",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.weights_dir = weights_dir
        self.optimizer_name = optimizer_name


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using enhanced argument parser.

    Note: Early stopping arguments are parsed but not used in this example.
    The create_training_parser() includes early stopping flags:
        --use-early-stopping
        --early-stopping-monitor (default: val_loss)
        --early-stopping-patience (default: 5)
        --early-stopping-min-delta (default: 0.0)
        --early-stopping-mode (default: min)

    Returns:
        TrainConfig with parsed arguments.
    """
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "lenet5_weights")
    parser.add_argument("data-dir", "string", "datasets/emnist")

    var args = parser.parse()

    var epochs = args.get_int("epochs", 10)
    var batch_size = args.get_int("batch-size", 32)
    var learning_rate = Float32(args.get_float("lr", 0.001))
    var data_dir = args.get_string("data-dir", "datasets/emnist")
    var weights_dir = args.get_string("weights-dir", "lenet5_weights")
    var optimizer_name = args.get_string("optimizer", "sgd")

    return TrainConfig(
        epochs, batch_size, learning_rate, data_dir, weights_dir, optimizer_name
    )


def compute_gradients(
    mut model: LeNet5,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    optimizer_name: String = "sgd",
    t: Int = 1,
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually.

    Args:
        model: LeNet-5 model.
        input: Batch of images (batch, 1, 28, 28).
        labels: One-hot encoded batch of labels (batch, num_classes).
        learning_rate: Learning rate for the optimizer.
        optimizer_name: Which optimizer to use ("sgd", "lion", "shampoo").
        t: Global training step counter (1-indexed, continuous across epochs).

    Returns:
        Loss value for this batch.
    """
    # ========== Forward Pass (with caching) ==========

    # Conv1 + ReLU + MaxPool
    var conv1_out = conv2d(
        input,
        model.conv1_kernel,
        model.conv1_bias,
        stride=CONV1_STRIDE,
        padding=CONV1_PADDING,
    )
    var relu1_out = relu(conv1_out)
    var pool1_out = maxpool2d(
        relu1_out,
        kernel_size=POOL1_KERNEL_SIZE,
        stride=POOL1_STRIDE,
        padding=POOL1_PADDING,
    )

    # Conv2 + ReLU + MaxPool
    var conv2_out = conv2d(
        pool1_out,
        model.conv2_kernel,
        model.conv2_bias,
        stride=CONV2_STRIDE,
        padding=CONV2_PADDING,
    )
    var relu2_out = relu(conv2_out)
    var pool2_out = maxpool2d(
        relu2_out,
        kernel_size=POOL2_KERNEL_SIZE,
        stride=POOL2_STRIDE,
        padding=POOL2_PADDING,
    )

    # Flatten
    var pool2_shape = pool2_out.shape()
    var batch_size = pool2_shape[0]
    var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    var flatten_shape: List[Int] = [batch_size, flattened_size]
    var flattened = pool2_out.reshape(flatten_shape)

    # FC1 + ReLU
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu3_out = relu(fc1_out)

    # FC2 + ReLU
    var fc2_out = linear(relu3_out, model.fc2_weights, model.fc2_bias)
    var relu4_out = relu(fc2_out)

    # FC3 (logits)
    var logits = linear(relu4_out, model.fc3_weights, model.fc3_bias)

    # Compute loss
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ========== Backward Pass ==========

    # Start with gradient from loss
    # For cross-entropy with mean reduction, the initial gradient is 1.0 / batch_size
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output.set(0, Float32(1.0))
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    # FC3 backward
    var fc3_grads = linear_backward(grad_logits, relu4_out, model.fc3_weights)

    # ReLU4 backward
    var grad_fc2_out = relu_backward(fc3_grads.grad_input, fc2_out)

    # FC2 backward
    var fc2_grads = linear_backward(grad_fc2_out, relu3_out, model.fc2_weights)

    # ReLU3 backward
    var grad_fc1_out = relu_backward(fc2_grads.grad_input, fc1_out)

    # FC1 backward
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)

    # Unflatten gradient
    var grad_pool2_out = fc1_grads.grad_input.reshape(pool2_shape)

    # MaxPool2 backward
    var grad_relu2_out = maxpool2d_backward(
        grad_pool2_out,
        relu2_out,
        kernel_size=POOL2_KERNEL_SIZE,
        stride=POOL2_STRIDE,
        padding=POOL2_PADDING,
    )

    # ReLU2 backward
    var grad_conv2_out = relu_backward(grad_relu2_out, conv2_out)

    # Conv2 backward
    var conv2_grads = conv2d_backward(
        grad_conv2_out,
        pool1_out,
        model.conv2_kernel,
        stride=CONV2_STRIDE,
        padding=CONV2_PADDING,
    )

    # MaxPool1 backward
    var grad_relu1_out = maxpool2d_backward(
        conv2_grads.grad_input,
        relu1_out,
        kernel_size=POOL1_KERNEL_SIZE,
        stride=POOL1_STRIDE,
        padding=POOL1_PADDING,
    )

    # ReLU1 backward
    var grad_conv1_out = relu_backward(grad_relu1_out, conv1_out)

    # Conv1 backward
    var conv1_grads = conv2d_backward(
        grad_conv1_out,
        input,
        model.conv1_kernel,
        stride=CONV1_STRIDE,
        padding=CONV1_PADDING,
    )

    # ========== Parameter Update ==========
    var lr64 = Float64(learning_rate)
    if optimizer_name == "lion":
        var res = lion_step(
            model.conv1_kernel,
            conv1_grads.grad_weights,
            model.conv1_kernel_m,
            lr64,
        )
        model.conv1_kernel = res[0]
        model.conv1_kernel_m = res[1]
        res = lion_step(
            model.conv1_bias, conv1_grads.grad_bias, model.conv1_bias_m, lr64
        )
        model.conv1_bias = res[0]
        model.conv1_bias_m = res[1]
        res = lion_step(
            model.conv2_kernel,
            conv2_grads.grad_weights,
            model.conv2_kernel_m,
            lr64,
        )
        model.conv2_kernel = res[0]
        model.conv2_kernel_m = res[1]
        res = lion_step(
            model.conv2_bias, conv2_grads.grad_bias, model.conv2_bias_m, lr64
        )
        model.conv2_bias = res[0]
        model.conv2_bias_m = res[1]
        res = lion_step(
            model.fc1_weights, fc1_grads.grad_weights, model.fc1_weights_m, lr64
        )
        model.fc1_weights = res[0]
        model.fc1_weights_m = res[1]
        res = lion_step(
            model.fc1_bias, fc1_grads.grad_bias, model.fc1_bias_m, lr64
        )
        model.fc1_bias = res[0]
        model.fc1_bias_m = res[1]
        res = lion_step(
            model.fc2_weights, fc2_grads.grad_weights, model.fc2_weights_m, lr64
        )
        model.fc2_weights = res[0]
        model.fc2_weights_m = res[1]
        res = lion_step(
            model.fc2_bias, fc2_grads.grad_bias, model.fc2_bias_m, lr64
        )
        model.fc2_bias = res[0]
        model.fc2_bias_m = res[1]
        res = lion_step(
            model.fc3_weights, fc3_grads.grad_weights, model.fc3_weights_m, lr64
        )
        model.fc3_weights = res[0]
        model.fc3_weights_m = res[1]
        res = lion_step(
            model.fc3_bias, fc3_grads.grad_bias, model.fc3_bias_m, lr64
        )
        model.fc3_bias = res[0]
        model.fc3_bias_m = res[1]
    elif optimizer_name == "shampoo":
        var sres = shampoo_step(
            model.conv1_kernel,
            conv1_grads.grad_weights,
            model.conv1_kernel_m,
            t,
            lr64,
        )
        model.conv1_kernel = sres[0]
        model.conv1_kernel_m = sres[1]
        sres = shampoo_step(
            model.conv1_bias, conv1_grads.grad_bias, model.conv1_bias_m, t, lr64
        )
        model.conv1_bias = sres[0]
        model.conv1_bias_m = sres[1]
        sres = shampoo_step(
            model.conv2_kernel,
            conv2_grads.grad_weights,
            model.conv2_kernel_m,
            t,
            lr64,
        )
        model.conv2_kernel = sres[0]
        model.conv2_kernel_m = sres[1]
        sres = shampoo_step(
            model.conv2_bias, conv2_grads.grad_bias, model.conv2_bias_m, t, lr64
        )
        model.conv2_bias = sres[0]
        model.conv2_bias_m = sres[1]
        sres = shampoo_step(
            model.fc1_weights,
            fc1_grads.grad_weights,
            model.fc1_weights_m,
            t,
            lr64,
        )
        model.fc1_weights = sres[0]
        model.fc1_weights_m = sres[1]
        sres = shampoo_step(
            model.fc1_bias, fc1_grads.grad_bias, model.fc1_bias_m, t, lr64
        )
        model.fc1_bias = sres[0]
        model.fc1_bias_m = sres[1]
        sres = shampoo_step(
            model.fc2_weights,
            fc2_grads.grad_weights,
            model.fc2_weights_m,
            t,
            lr64,
        )
        model.fc2_weights = sres[0]
        model.fc2_weights_m = sres[1]
        sres = shampoo_step(
            model.fc2_bias, fc2_grads.grad_bias, model.fc2_bias_m, t, lr64
        )
        model.fc2_bias = sres[0]
        model.fc2_bias_m = sres[1]
        sres = shampoo_step(
            model.fc3_weights,
            fc3_grads.grad_weights,
            model.fc3_weights_m,
            t,
            lr64,
        )
        model.fc3_weights = sres[0]
        model.fc3_weights_m = sres[1]
        sres = shampoo_step(
            model.fc3_bias, fc3_grads.grad_bias, model.fc3_bias_m, t, lr64
        )
        model.fc3_bias = sres[0]
        model.fc3_bias_m = sres[1]
    else:
        # Default: SGD via existing in-place update
        model.update_parameters(
            learning_rate,
            conv1_grads.grad_weights^,
            conv1_grads.grad_bias^,
            conv2_grads.grad_weights^,
            conv2_grads.grad_bias^,
            fc1_grads.grad_weights^,
            fc1_grads.grad_bias^,
            fc2_grads.grad_weights^,
            fc2_grads.grad_bias^,
            fc3_grads.grad_weights^,
            fc3_grads.grad_bias^,
        )

    return loss


def train_epoch(
    mut model: LeNet5,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    epoch: Int,
    total_epochs: Int,
    optimizer_name: String = "sgd",
    global_step_start: Int = 1,
) raises -> Tuple[Float32, Int]:
    """Train for one epoch using manual batch processing.

    Args:
        model: LeNet-5 model.
        train_images: Training images (num_samples, 1, 28, 28).
        train_labels: Integer training labels (num_samples,).
        batch_size: Mini-batch size.
        learning_rate: Learning rate for the optimizer.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        optimizer_name: Which optimizer to use ("sgd", "lion", "shampoo").
        global_step_start: Starting value for the global step counter this epoch.

    Returns:
        Tuple of (average loss for the epoch, final global step after epoch).
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    var total_loss = Float32(0.0)
    var t = global_step_start

    # Manual batch processing
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)
        var actual_batch_size = end_idx - start_idx

        # Extract batch from training data
        var batch_images_shape = List[Int]()
        batch_images_shape.append(actual_batch_size)
        batch_images_shape.append(train_images.shape()[1])
        batch_images_shape.append(train_images.shape()[2])
        batch_images_shape.append(train_images.shape()[3])
        var batch_images = zeros(batch_images_shape, train_images.dtype())

        var batch_labels_int_shape = List[Int]()
        batch_labels_int_shape.append(actual_batch_size)
        var batch_labels_int = zeros(
            batch_labels_int_shape, train_labels.dtype()
        )

        # Copy data into batch tensors
        for i in range(actual_batch_size):
            var sample_idx = start_idx + i
            # Copy image
            for c in range(train_images.shape()[1]):
                for h in range(train_images.shape()[2]):
                    for w in range(train_images.shape()[3]):
                        var src_idx = (
                            sample_idx
                            * train_images.shape()[1]
                            * train_images.shape()[2]
                            * train_images.shape()[3]
                            + c
                            * train_images.shape()[2]
                            * train_images.shape()[3]
                            + h * train_images.shape()[3]
                            + w
                        )
                        var dst_idx = (
                            i
                            * train_images.shape()[1]
                            * train_images.shape()[2]
                            * train_images.shape()[3]
                            + c
                            * train_images.shape()[2]
                            * train_images.shape()[3]
                            + h * train_images.shape()[3]
                            + w
                        )
                        (batch_images._data + dst_idx).store(
                            (train_images._data + src_idx).load()
                        )
            # Copy label
            (batch_labels_int._data + i).store(
                (train_labels._data + sample_idx).load()
            )

        # Convert batch labels to one-hot encoding
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=DEFAULT_NUM_CLASSES
        )

        # Compute gradients and update parameters
        var batch_loss = compute_gradients(
            model, batch_images, batch_labels, learning_rate, optimizer_name, t
        )
        total_loss += batch_loss
        t += 1

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            var avg_loss_so_far = total_loss / Float32(batch_idx + 1)
            print(
                "  Epoch [",
                epoch,
                "/",
                total_epochs,
                "] Batch [",
                batch_idx + 1,
                "/",
                num_batches,
                "] Loss:",
                avg_loss_so_far,
            )

    return (total_loss / Float32(num_batches), t)


def main() raises:
    """Main training loop."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config.epochs
    var batch_size = config.batch_size
    var learning_rate = config.learning_rate
    var data_dir = config.data_dir
    var weights_dir = config.weights_dir
    var optimizer_name = config.optimizer_name

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Optimizer: ", optimizer_name)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=DEFAULT_NUM_CLASSES)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load dataset
    print("Loading EMNIST dataset...")
    var train_images_path = (
        data_dir + "/emnist-balanced-train-images-idx3-ubyte"
    )
    var train_labels_path = (
        data_dir + "/emnist-balanced-train-labels-idx1-ubyte"
    )
    var test_images_path = data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    var test_labels_path = data_dir + "/emnist-balanced-test-labels-idx1-ubyte"

    var train_images_raw = load_idx_images(train_images_path)
    var train_labels = load_idx_labels(train_labels_path)
    var test_images_raw = load_idx_images(test_images_path)
    var test_labels = load_idx_labels(test_labels_path)

    # Normalize images to [0, 1]
    var train_images = normalize_images(train_images_raw)
    var test_images = normalize_images(test_images_raw)

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Training loop
    print("Starting training...")
    var global_step = 1
    for epoch in range(1, epochs + 1):
        var epoch_result = train_epoch(
            model,
            train_images,
            train_labels,
            batch_size,
            learning_rate,
            epoch,
            epochs,
            optimizer_name,
            global_step,
        )
        var train_loss = epoch_result[0]
        global_step = epoch_result[1]

        # Evaluate every epoch using shared evaluation module
        var test_acc = evaluate_model_simple(
            model,
            test_images,
            test_labels,
            batch_size=100,
            num_classes=DEFAULT_NUM_CLASSES,
            verbose=True,
        )
        print("  Test Accuracy: ", test_acc * 100.0, "%")
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
