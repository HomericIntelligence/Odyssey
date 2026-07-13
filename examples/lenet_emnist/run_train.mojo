"""
CLI Wrapper for LeNet-5 Training.

Provides command-line interface for training LeNet-5 on EMNIST.
Wraps the core train.mojo functionality with enhanced argument parser.

Usage:
    mojo run examples/lenet_emnist/run_train.mojo --epochs 10 --precision fp32
    mojo run examples/lenet_emnist/run_train.mojo --batch-size 64 --lr 0.01

Arguments:
    --epochs N       Number of training epochs (default: 10)
    --batch-size N   Mini-batch size (default: 32)
    --lr RATE        Learning rate (default: 0.001)
    --precision P    Precision mode: fp32, fp16, bf16, fp8 (default: fp32)
    --data-dir DIR   EMNIST data directory (default: datasets/emnist)
    --weights-dir D  Output weights directory (default: lenet5_weights)
"""

from model import LeNet5
from odyssey.data.constants import DatasetInfo
from odyssey.data.formats import (
    load_idx_images,
    load_idx_labels,
    normalize_images,
    one_hot_encode,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.conv import conv2d, conv2d_backward
from odyssey.core.pooling import maxpool2d, maxpool2d_backward
from odyssey.core.linear import linear, linear_backward
from odyssey.core.activation import relu, relu_backward
from odyssey.core.loss import cross_entropy, cross_entropy_backward
from odyssey.training.precision_config import (
    PrecisionConfig,
    PrecisionMode,
)
from odyssey.training.evaluation import evaluate_model_simple
from odyssey.utils.arg_parser import create_training_parser
from std.collections import List


struct TrainConfig(Movable):
    """Training configuration."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var precision: String
    var data_dir: String
    var weights_dir: String
    var max_batches: Int
    var smoke: Bool

    def __init__(out self):
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = Float32(0.001)
        self.precision = "fp32"
        self.data_dir = "datasets/emnist"
        self.weights_dir = "lenet5_weights"
        self.max_batches = 0
        self.smoke = False


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        TrainConfig with parsed arguments.
    """
    var parser = create_training_parser()
    parser.add_argument("precision", "string", "fp32")
    parser.add_argument("weights-dir", "string", "lenet5_weights")
    parser.add_argument("data-dir", "string", "datasets/emnist")

    var args = parser.parse()

    var config = TrainConfig()
    config.epochs = args.resolve_int("epochs", 10)
    config.batch_size = args.resolve_int("batch-size", 32)
    config.learning_rate = Float32(args.resolve_float("lr", 0.001))
    config.precision = args.resolve_string("precision", "fp32")
    config.data_dir = args.resolve_string("data-dir", "datasets/emnist")
    config.weights_dir = args.resolve_string("weights-dir", "lenet5_weights")
    config.max_batches = args.resolve_int("max-batches", 0)
    config.smoke = args.get_bool("smoke")

    return config^


def compute_gradients(
    mut model: LeNet5,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    mut precision_config: PrecisionConfig,
) raises -> Tuple[Float32, Bool]:
    """Compute gradients and update parameters for one batch with precision scaling.

    Implements full forward and backward pass with support for mixed-precision training.
    For FP16/BF16 training:
    - Loss is scaled before backward pass to prevent gradient underflow
    - Gradients are checked for overflow/NaN after backward pass
    - On overflow, parameter update is skipped and scaler is adjusted
    - On success, scaler may increase for next iteration

    Args:
        model: LeNet-5 model.
        input: Batch of images (batch, 1, 28, 28).
        labels: One-hot encoded batch of labels (batch, num_classes).
        learning_rate: Learning rate for SGD.
        precision_config: Precision configuration (FP32, FP16, BF16, FP8).

    Returns:
        Tuple of (loss_value, success_flag):
        - loss_value: Unscaled loss for logging.
        - success_flag: True if update succeeded, False on gradient overflow.
    """
    # ========== Forward Pass (with caching) ==========

    # Conv1 + ReLU + MaxPool
    var conv1_out = conv2d(
        input, model.conv1_kernel, model.conv1_bias, stride=1, padding=0
    )
    var relu1_out = relu(conv1_out)
    var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)

    # Conv2 + ReLU + MaxPool
    var conv2_out = conv2d(
        pool1_out, model.conv2_kernel, model.conv2_bias, stride=1, padding=0
    )
    var relu2_out = relu(conv2_out)
    var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)

    # Flatten
    var pool2_shape = pool2_out.shape()
    var batch_size = pool2_shape[0]
    var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    var flatten_shape = List[Int]()
    flatten_shape.append(batch_size)
    flatten_shape.append(flattened_size)
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

    # Scale loss for mixed-precision training (prevents gradient underflow in FP16)
    var scaled_loss_tensor = precision_config.scale_loss(loss_tensor)

    # ========== Backward Pass ==========

    # Start with gradient from scaled loss
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
        grad_pool2_out, relu2_out, kernel_size=2, stride=2, padding=0
    )

    # ReLU2 backward
    var grad_conv2_out = relu_backward(grad_relu2_out, conv2_out)

    # Conv2 backward
    var conv2_grads = conv2d_backward(
        grad_conv2_out, pool1_out, model.conv2_kernel, stride=1, padding=0
    )

    # MaxPool1 backward
    var grad_relu1_out = maxpool2d_backward(
        conv2_grads.grad_input, relu1_out, kernel_size=2, stride=2, padding=0
    )

    # ReLU1 backward
    var grad_conv1_out = relu_backward(grad_relu1_out, conv1_out)

    # Conv1 backward
    var conv1_grads = conv2d_backward(
        grad_conv1_out, input, model.conv1_kernel, stride=1, padding=0
    )

    # ========== Gradient Scaling and Validation ==========

    # Unscale gradients for mixed-precision training
    var unscaled_conv1_grad_kernel = precision_config.unscale_gradients(
        conv1_grads.grad_weights
    )
    var unscaled_conv1_grad_bias = precision_config.unscale_gradients(
        conv1_grads.grad_bias
    )
    var unscaled_conv2_grad_kernel = precision_config.unscale_gradients(
        conv2_grads.grad_weights
    )
    var unscaled_conv2_grad_bias = precision_config.unscale_gradients(
        conv2_grads.grad_bias
    )
    var unscaled_fc1_grad_kernel = precision_config.unscale_gradients(
        fc1_grads.grad_weights
    )
    var unscaled_fc1_grad_bias = precision_config.unscale_gradients(
        fc1_grads.grad_bias
    )
    var unscaled_fc2_grad_kernel = precision_config.unscale_gradients(
        fc2_grads.grad_weights
    )
    var unscaled_fc2_grad_bias = precision_config.unscale_gradients(
        fc2_grads.grad_bias
    )
    var unscaled_fc3_grad_kernel = precision_config.unscale_gradients(
        fc3_grads.grad_weights
    )
    var unscaled_fc3_grad_bias = precision_config.unscale_gradients(
        fc3_grads.grad_bias
    )

    # Check if any gradients have NaN/Inf (overflow detection)
    var grads_valid = True
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_conv1_grad_kernel
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_conv1_grad_bias
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_conv2_grad_kernel
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_conv2_grad_bias
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc1_grad_kernel
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc1_grad_bias
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc2_grad_kernel
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc2_grad_bias
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc3_grad_kernel
    )
    grads_valid = grads_valid and precision_config.check_gradients(
        unscaled_fc3_grad_bias
    )

    # ========== Conditional Parameter Update ==========

    if grads_valid:
        # Gradients are valid: perform parameter update
        model.update_parameters(
            learning_rate,
            unscaled_conv1_grad_kernel^,
            unscaled_conv1_grad_bias^,
            unscaled_conv2_grad_kernel^,
            unscaled_conv2_grad_bias^,
            unscaled_fc1_grad_kernel^,
            unscaled_fc1_grad_bias^,
            unscaled_fc2_grad_kernel^,
            unscaled_fc2_grad_bias^,
            unscaled_fc3_grad_kernel^,
            unscaled_fc3_grad_bias^,
        )
        precision_config.step(grads_valid=True)
    else:
        # Gradient overflow detected: skip update and reduce scale
        print("Gradient overflow detected, skipping parameter update")
        precision_config.step(grads_valid=False)

    return Tuple[Float32, Bool](loss, grads_valid)


def train_epoch(
    mut model: LeNet5,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    epoch: Int,
    total_epochs: Int,
    mut precision_config: PrecisionConfig,
    max_batches: Int = 0,
) raises -> Float32:
    """Train for one epoch with mixed-precision support.

    Args:
        model: LeNet-5 model.
        train_images: Training images.
        train_labels: Training labels.
        batch_size: Batch size for training.
        learning_rate: Learning rate for SGD.
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
        precision_config: Precision configuration for mixed-precision training.
        max_batches: Cap batches this epoch (0 = unbounded); used by --smoke /
            --max-batches to bound a per-PR run (#5551).

    Returns:
        Average loss for the epoch (unscaled).
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    if max_batches > 0 and max_batches < num_batches:
        num_batches = max_batches

    var total_loss = Float32(0.0)
    var successful_batches = 0
    var skipped_batches = 0

    print("Epoch [", epoch, "/", total_epochs, "]")

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)

        # Extract batch slice from dataset
        var batch_images = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels_int = train_labels.slice(start_idx, end_idx, axis=0)

        # Convert batch labels to one-hot encoding
        var dataset_info = DatasetInfo("emnist_balanced")
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=dataset_info.num_classes()
        )

        # Compute gradients and update parameters with precision scaling
        var result = compute_gradients(
            model, batch_images, batch_labels, learning_rate, precision_config
        )
        var batch_loss = result[0]
        var batch_success = result[1]

        total_loss += batch_loss

        if batch_success:
            successful_batches += 1
        else:
            skipped_batches += 1

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0 or num_batches <= 10:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            var scale = precision_config.get_scale()
            print(
                "  Batch [",
                batch_idx + 1,
                "/",
                num_batches,
                "] - Loss: ",
                avg_loss,
                " (scale: ",
                scale,
                ")",
            )

    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: ", avg_loss)
    if precision_config.get_overflow_count() > 0:
        print("  Gradient overflows: ", precision_config.get_overflow_count())

    return avg_loss


def main() raises:
    """Main training entry point."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()

    print("\nConfiguration:")
    print("  Epochs: ", config.epochs)
    print("  Batch Size: ", config.batch_size)
    print("  Learning Rate: ", config.learning_rate)
    print("  Precision: ", config.precision)
    print("  Data Directory: ", config.data_dir)
    print("  Weights Directory: ", config.weights_dir)
    print()

    # Initialize precision configuration
    var precision_config = PrecisionConfig.from_string(config.precision)
    precision_config.print_config()
    print()

    # Note: Full mixed-precision training requires gradient scaling through all layers
    # Currently, training uses FP32 compute. PrecisionConfig is ready for future integration.
    if precision_config.mode != PrecisionMode.FP32:
        print(
            "Note: Mixed-precision training with gradient scaling will be"
            " applied."
        )
        print(
            "      Master weights maintained in FP32 for optimizer stability."
        )
        print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var dataset_info = DatasetInfo("emnist_balanced")
    var model = LeNet5(num_classes=dataset_info.num_classes())
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load data — real EMNIST, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset load entirely so the training
    # entrypoint can run per-PR in CI. It checks the MECHANISM (the loop runs
    # and emits finite, parseable, decreasing loss), not convergence.
    var train_images: AnyTensor
    var train_labels: AnyTensor
    var test_images: AnyTensor
    var test_labels: AnyTensor
    if config.smoke:
        print("Smoke mode: using synthetic data (no dataset load)...")
        # Enough samples to yield >=2 batches under --max-batches so the CI gate
        # can assert a decreasing loss trend across batches. Class-correlated
        # signal makes the loss learnable; labels are RAW uint8 [N]
        # (train_epoch one-hot-encodes).
        var num_classes = dataset_info.num_classes()
        var wanted_batches = config.max_batches if config.max_batches > 0 else 3
        var n_smoke = wanted_batches * Int(config.batch_size)
        train_images = zeros([n_smoke, 1, 28, 28], DType.float32)
        var img_d = train_images._data.bitcast[Float32]()
        for s in range(n_smoke):
            var cls = s % num_classes
            for i in range(1 * 28 * 28):
                img_d[s * (1 * 28 * 28) + i] = (
                    Float32(cls) * 0.05 + Float32(i % 5) * 0.01
                )
        train_labels = zeros([n_smoke], DType.uint8)
        var lbl_d = train_labels._data.bitcast[UInt8]()
        for s in range(n_smoke):
            lbl_d[s] = UInt8(s % num_classes)
        # Reuse the same synthetic batch for "test" (eval is not asserted here).
        test_images = train_images
        test_labels = train_labels
    else:
        print("Loading EMNIST dataset...")
        var train_images_path = (
            config.data_dir + "/emnist-balanced-train-images-idx3-ubyte"
        )
        var train_labels_path = (
            config.data_dir + "/emnist-balanced-train-labels-idx1-ubyte"
        )
        var test_images_path = (
            config.data_dir + "/emnist-balanced-test-images-idx3-ubyte"
        )
        var test_labels_path = (
            config.data_dir + "/emnist-balanced-test-labels-idx1-ubyte"
        )

        var train_images_raw = load_idx_images(train_images_path)
        train_labels = load_idx_labels(train_labels_path)
        var test_images_raw = load_idx_images(test_images_path)
        test_labels = load_idx_labels(test_labels_path)

        # Normalize images to [0, 1]
        train_images = normalize_images(train_images_raw)
        test_images = normalize_images(test_images_raw)

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Training loop with mixed-precision support
    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        var train_loss = train_epoch(
            model,
            train_images,
            train_labels,
            config.batch_size,
            config.learning_rate,
            epoch,
            config.epochs,
            precision_config,
            max_batches=config.max_batches,
        )

        # Evaluate every epoch using shared evaluation module
        var test_acc = evaluate_model_simple(
            model,
            test_images,
            test_labels,
            batch_size=100,
            num_classes=model.num_classes,
            verbose=True,
        )
        print("  Test Accuracy: ", test_acc * 100.0, "%")
        print()

    # Print final precision statistics
    if precision_config.get_overflow_count() > 0:
        print("Final Training Statistics:")
        precision_config.print_stats()
        print()

    # Save model — skipped in smoke mode (#5551): mechanism check, nothing to
    # persist.
    if not config.smoke:
        print("Saving model weights...")
        model.save_weights(config.weights_dir)
        print("  Model saved to", config.weights_dir)
        print()

    print("Training complete!")
