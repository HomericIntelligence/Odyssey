"""Training Script for LeNet-5 on EMNIST using Tape-Based Autograd.

Implements training with automatic differentiation using Variable/GradientTape substrate.
Uses SGD optimizer with automatic backward pass.

Usage:
    mojo run examples/lenet_emnist/train_autograd.mojo --epochs 1 --batch-size 32 --lr 0.01 --max-batches 10

Requirements:
    - EMNIST dataset downloaded (run: python scripts/download_emnist.py)
    - Dataset location: datasets/emnist/

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Autograd Substrate: docs/dev/autograd-phase2-design.md
"""

from model import (
    LeNet5,
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
from odyssey.data.formats import (
    load_idx_images,
    load_idx_labels,
    normalize_images,
    one_hot_encode,
)
from odyssey.autograd import (
    Variable,
    GradientTape,
    SGD,
    variable_conv2d,
    variable_relu,
    variable_maxpool2d,
    variable_flatten,
    variable_linear,
    variable_cross_entropy,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.training.evaluation import evaluate_model_simple
from odyssey.utils.arg_parser import create_training_parser
from std.collections import List

# Default number of classes for EMNIST Balanced dataset
comptime DEFAULT_NUM_CLASSES = 47


struct TrainConfig:
    """Training configuration from command line arguments."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var data_dir: String
    var weights_dir: String
    var max_batches: Int

    def __init__(
        out self,
        epochs: Int,
        batch_size: Int,
        learning_rate: Float64,
        data_dir: String,
        weights_dir: String,
        max_batches: Int = -1,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.weights_dir = weights_dir
        self.max_batches = max_batches


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        TrainConfig with parsed arguments.
    """
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "lenet5_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/emnist")
    parser.add_argument("max-batches", "int", "-1")

    var args = parser.parse()

    var epochs = args.get_int("epochs", 1)
    var batch_size = args.get_int("batch-size", 32)
    var learning_rate = args.get_float("lr", 0.001)
    var data_dir = args.get_string("data-dir", "datasets/emnist")
    var weights_dir = args.get_string("weights-dir", "lenet5_weights_autograd")
    var max_batches = args.get_int("max-batches", -1)

    return TrainConfig(
        epochs, batch_size, learning_rate, data_dir, weights_dir, max_batches
    )


def train_batch(
    mut model: LeNet5,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: SGD,
    mut tape: GradientTape,
) raises -> Float32:
    """Compute gradients and update parameters for one batch using autograd.

    This implements the forward pass using variable_* ops that record to the tape,
    then calls loss.backward(tape) for automatic gradient computation.

    Args:
        model: LeNet-5 model.
        input: Batch of images (batch, 1, 28, 28).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: SGD optimizer.
        tape: GradientTape for recording operations.

    Returns:
        Loss value for this batch.
    """
    # ========== Wrap model parameters and inputs as Variables ==========
    # Note: Variable constructor takes ownership of the tensor.
    # Since model.conv1_kernel etc. are AnyTensors (copyable), the default
    # parameter passing creates a copy. We use move (^) to transfer ownership.
    # Actually, for safety and clarity with Mojo's ownership semantics,
    # we create Variables from the model's current tensors.
    var conv1_kernel = Variable(model.conv1_kernel, True, tape)
    var conv1_bias = Variable(model.conv1_bias, True, tape)
    var conv2_kernel = Variable(model.conv2_kernel, True, tape)
    var conv2_bias = Variable(model.conv2_bias, True, tape)
    var fc1_weights = Variable(model.fc1_weights, True, tape)
    var fc1_bias = Variable(model.fc1_bias, True, tape)
    var fc2_weights = Variable(model.fc2_weights, True, tape)
    var fc2_bias = Variable(model.fc2_bias, True, tape)
    var fc3_weights = Variable(model.fc3_weights, True, tape)
    var fc3_bias = Variable(model.fc3_bias, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Forward Pass (recorded to tape) ==========

    # Conv1 + ReLU + MaxPool
    var conv1_out = variable_conv2d(
        input_var,
        conv1_kernel,
        conv1_bias,
        tape,
        stride=CONV1_STRIDE,
        padding=CONV1_PADDING,
    )
    var relu1_out = variable_relu(conv1_out, tape)
    var pool1_out = variable_maxpool2d(
        relu1_out,
        tape,
        kernel_size=POOL1_KERNEL_SIZE,
        stride=POOL1_STRIDE,
        padding=POOL1_PADDING,
    )

    # Conv2 + ReLU + MaxPool
    var conv2_out = variable_conv2d(
        pool1_out,
        conv2_kernel,
        conv2_bias,
        tape,
        stride=CONV2_STRIDE,
        padding=CONV2_PADDING,
    )
    var relu2_out = variable_relu(conv2_out, tape)
    var pool2_out = variable_maxpool2d(
        relu2_out,
        tape,
        kernel_size=POOL2_KERNEL_SIZE,
        stride=POOL2_STRIDE,
        padding=POOL2_PADDING,
    )

    # Flatten
    var flattened = variable_flatten(pool2_out, tape)

    # FC1 + ReLU
    var fc1_out = variable_linear(flattened, fc1_weights, fc1_bias, tape)
    var relu3_out = variable_relu(fc1_out, tape)

    # FC2 + ReLU
    var fc2_out = variable_linear(relu3_out, fc2_weights, fc2_bias, tape)
    var relu4_out = variable_relu(fc2_out, tape)

    # FC3 (output logits)
    var logits = variable_linear(relu4_out, fc3_weights, fc3_bias, tape)

    # Compute cross-entropy loss
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract loss value for logging before consuming `loss` below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update (SGD) ==========
    # Move all trainable Variables into the parameters list (Variable is not
    # ImplicitlyCopyable, so we transfer ownership with `^`).
    var parameters: List[Variable] = []
    parameters.append(conv1_kernel^)
    parameters.append(conv1_bias^)
    parameters.append(conv2_kernel^)
    parameters.append(conv2_bias^)
    parameters.append(fc1_weights^)
    parameters.append(fc1_bias^)
    parameters.append(fc2_weights^)
    parameters.append(fc2_bias^)
    parameters.append(fc3_weights^)
    parameters.append(fc3_bias^)

    optimizer.step(parameters, tape)

    # ========== Update model parameters ==========
    # Copy updated Variable.data back into the model in the same order they
    # were appended above.
    model.conv1_kernel = parameters[0].data
    model.conv1_bias = parameters[1].data
    model.conv2_kernel = parameters[2].data
    model.conv2_bias = parameters[3].data
    model.fc1_weights = parameters[4].data
    model.fc1_bias = parameters[5].data
    model.fc2_weights = parameters[6].data
    model.fc2_bias = parameters[7].data
    model.fc3_weights = parameters[8].data
    model.fc3_bias = parameters[9].data

    # Zero gradients for next batch
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch(
    mut model: LeNet5,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float64,
    epoch: Int,
    total_epochs: Int,
    max_batches: Int,
) raises -> Float32:
    """Train for one epoch using autograd.

    Args:
        model: LeNet-5 model.
        train_images: Training images (num_samples, 1, 28, 28).
        train_labels: Integer training labels (num_samples,).
        batch_size: Mini-batch size.
        learning_rate: Learning rate for SGD.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        max_batches: Maximum number of batches to process (-1 = all).

    Returns:
        Average loss for the epoch.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    if max_batches > 0:
        num_batches = min(num_batches, max_batches)

    var total_loss = Float32(0.0)

    # Create optimizer for this epoch
    var optimizer = SGD(learning_rate=learning_rate)

    # Manual batch processing
    for batch_idx in range(num_batches):
        # Create fresh tape for each batch to avoid operation accumulation
        var tape = GradientTape()
        tape.enable()
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

        # Copy data into batch tensors using dtype-agnostic accessors.
        # NOTE: the previous version used `(_data + i).load()` directly on the
        # UInt8 byte pointer, which copied only 1 byte per element regardless
        # of dtype — silently corrupting float32 training data. _get_float64 /
        # _set_float64 round-trip through Float64 and preserve the value at
        # whatever precision the tensor was allocated with.
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
                        batch_images._set_float64(
                            dst_idx, train_images._get_float64(src_idx)
                        )
            # Copy label
            batch_labels_int._set_float64(
                i, train_labels._get_float64(sample_idx)
            )

        # Convert batch labels to one-hot encoding
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=DEFAULT_NUM_CLASSES
        )

        # Compute gradients and update parameters
        var batch_loss = train_batch(
            model, batch_images, batch_labels, optimizer, tape
        )
        total_loss += batch_loss

        # Log progress every 100 batches (or every batch if batch count is small)
        if (batch_idx + 1) % 100 == 0 or num_batches <= 10:
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

    return total_loss / Float32(num_batches)


def main() raises:
    """Main training loop."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset (Autograd)")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config.epochs
    var batch_size = config.batch_size
    var learning_rate = config.learning_rate
    var data_dir = config.data_dir
    var weights_dir = config.weights_dir
    var max_batches = config.max_batches

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    if max_batches > 0:
        print("  Max Batches: ", max_batches)
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
    for epoch in range(1, epochs + 1):
        var train_loss = train_epoch(
            model,
            train_images,
            train_labels,
            batch_size,
            learning_rate,
            epoch,
            epochs,
            max_batches,
        )

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
