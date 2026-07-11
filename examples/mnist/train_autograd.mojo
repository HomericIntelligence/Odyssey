"""Training Script for Simple CNN on MNIST using Tape-Based Autograd.

Autograd variant of `train.mojo` (sub-task 2 of #5454). Implements training with
automatic differentiation using the Variable / GradientTape substrate instead of
the hand-written forward + `*_backward` + `model.update_parameters(...)` chain.

Default optimizer is AdamW (matches examples/grok/lenet_emnist/); pass
`--optimizer sgd` to select SGD. The manual `train.mojo` is intentionally left
in place — the tracker-level cleanup (#5454) removes the manual path once all
ports land.

Usage:
    mojo run examples/mnist/train_autograd.mojo --epochs 1 --batch-size 32 --lr 0.001
    mojo run examples/mnist/train_autograd.mojo --optimizer sgd --lr 0.01

Requirements:
    - MNIST dataset downloaded (run: python scripts/download_mnist.py)
    - Dataset location: datasets/mnist/

References:
    - MNIST Dataset: http://yann.lecun.com/exdb/mnist/
    - Autograd substrate: src/projectodyssey/autograd/README.md
"""

from model import (
    SimpleCNN,
    # Architecture constants for the recorded forward pass.
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
from projectodyssey.autograd import (
    Variable,
    GradientTape,
    AdamW,
    SGD,
    variable_conv2d,
    variable_relu,
    variable_maxpool2d,
    variable_flatten,
    variable_linear,
    variable_cross_entropy,
)
from projectodyssey.autograd.optimizer_base import Optimizer
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.training.evaluation import evaluate_model_simple
from projectodyssey.utils.arg_parser import create_training_parser
from std.collections import List

# Default number of classes for MNIST dataset.
comptime DEFAULT_NUM_CLASSES = 10


struct TrainConfig:
    """Training configuration from command line arguments."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var weight_decay: Float64
    var optimizer: String
    var data_dir: String
    var weights_dir: String
    var max_batches: Int
    var smoke: Bool

    def __init__(
        out self,
        epochs: Int,
        batch_size: Int,
        learning_rate: Float64,
        weight_decay: Float64,
        optimizer: String,
        data_dir: String,
        weights_dir: String,
        max_batches: Int,
        smoke: Bool,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.data_dir = data_dir
        self.weights_dir = weights_dir
        self.max_batches = max_batches
        self.smoke = smoke


def use_adamw(optimizer: String) -> Bool:
    """Whether to use AdamW for the given `--optimizer` string.

    AdamW is the default (DoD of #5558); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd", so we
    cannot rely on the parsed default here — anything other than "sgd" (which
    includes the unset case resolved to "adamw" below) means AdamW.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser.

    Returns:
        TrainConfig with parsed arguments.
    """
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "mnist_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/mnist")

    var args = parser.parse()

    var epochs = args.get_int("epochs", 10)
    var batch_size = args.get_int("batch-size", 32)
    var learning_rate = args.get_float("lr", 0.001)
    var weight_decay = args.get_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates that default
    # into the value map — so `get_string("optimizer", "adamw")` would return
    # "sgd" even when the user passed nothing (the "adamw" fallback is dead).
    # Gate on `was_user_supplied` instead: only an explicit `--optimizer` is
    # honored; an unset flag defaults to AdamW (DoD of #5558).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    var data_dir = args.get_string("data-dir", "datasets/mnist")
    var weights_dir = args.get_string("weights-dir", "mnist_weights_autograd")
    var max_batches = args.get_int("max-batches", 0)
    var smoke = args.get_bool("smoke")

    return TrainConfig(
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        optimizer,
        data_dir,
        weights_dir,
        max_batches,
        smoke,
    )


def train_batch[
    O: Optimizer
](
    mut model: SimpleCNN,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the SimpleCNN forward pass with `variable_*` ops, calls
    `loss.backward(tape)` for automatic gradients, then `optimizer.step(...)`.

    Args:
        model: Simple CNN model (8 trainable tensors).
        input: Batch of images (batch, 1, 28, 28).
        labels: One-hot encoded batch of labels (batch, 10).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    # Wrap the model's 8 parameters as trainable Variables, and the batch
    # tensors as non-trainable inputs.
    var conv1_kernel = Variable(model.conv1_kernel, True, tape)
    var conv1_bias = Variable(model.conv1_bias, True, tape)
    var conv2_kernel = Variable(model.conv2_kernel, True, tape)
    var conv2_bias = Variable(model.conv2_bias, True, tape)
    var fc1_weights = Variable(model.fc1_weights, True, tape)
    var fc1_bias = Variable(model.fc1_bias, True, tape)
    var fc2_weights = Variable(model.fc2_weights, True, tape)
    var fc2_bias = Variable(model.fc2_bias, True, tape)

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

    # FC2 (output logits) — SimpleCNN has a 2-FC tail (no fc3).
    var logits = variable_linear(relu3_out, fc2_weights, fc2_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update ==========
    # Move the 8 trainable Variables into the parameters list (Variable is not
    # implicitly copyable, so transfer ownership with `^`). Order matters: the
    # write-back below uses the same order.
    var parameters: List[Variable] = []
    parameters.append(conv1_kernel^)
    parameters.append(conv1_bias^)
    parameters.append(conv2_kernel^)
    parameters.append(conv2_bias^)
    parameters.append(fc1_weights^)
    parameters.append(fc1_bias^)
    parameters.append(fc2_weights^)
    parameters.append(fc2_bias^)

    optimizer.step(parameters, tape)

    # Copy updated Variable.data back into the model, same order as appended.
    model.conv1_kernel = parameters[0].data
    model.conv1_bias = parameters[1].data
    model.conv2_kernel = parameters[2].data
    model.conv2_bias = parameters[3].data
    model.fc1_weights = parameters[4].data
    model.fc1_bias = parameters[5].data
    model.fc2_weights = parameters[6].data
    model.fc2_bias = parameters[7].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: SimpleCNN,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    epoch: Int,
    total_epochs: Int,
    mut optimizer: O,
    max_batches: Int = 0,
) raises -> Float32:
    """Train for one epoch using autograd.

    Args:
        model: Simple CNN model.
        train_images: Training images (num_samples, 1, 28, 28).
        train_labels: Integer training labels (num_samples,).
        batch_size: Mini-batch size.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        optimizer: The optimizer, constructed once so AdamW moment buffers
            persist across batches.
        max_batches: Cap batches this epoch (0 = unbounded); used by --smoke /
            --max-batches to bound a per-PR run (#5551).

    Returns:
        Average loss for the epoch.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    if max_batches > 0 and max_batches < num_batches:
        num_batches = max_batches

    var total_loss = Float32(0.0)

    for batch_idx in range(num_batches):
        # Fresh tape per batch so recorded operations do not accumulate.
        var tape = GradientTape()
        tape.enable()

        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)
        var actual_batch_size = end_idx - start_idx

        # Extract this batch from the training data.
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

        # Copy data into the batch tensors using dtype-agnostic accessors.
        # NOTE: the manual train.mojo copies via `(_data + i).load()` on the
        # raw UInt8 byte pointer — that moves only 1 byte per element and
        # silently corrupts float32. _get_float64/_set_float64 round-trip
        # through Float64 and preserve the value at the tensor's real precision.
        for i in range(actual_batch_size):
            var sample_idx = start_idx + i
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
            batch_labels_int._set_float64(
                i, train_labels._get_float64(sample_idx)
            )

        # One-hot encode the batch labels.
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=DEFAULT_NUM_CLASSES
        )

        # Autograd forward/backward/update for this batch.
        var batch_loss = train_batch(
            model, batch_images, batch_labels, optimizer, tape
        )
        total_loss += batch_loss

        # Log every 100 batches, or every batch for small (smoke) runs.
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
    print("Simple CNN Training on MNIST Dataset (Autograd)")
    print("=" * 60)

    var config = parse_args()
    var epochs = config.epochs
    var batch_size = config.batch_size
    var learning_rate = config.learning_rate
    var weight_decay = config.weight_decay
    var optimizer_name = config.optimizer
    var data_dir = config.data_dir
    var weights_dir = config.weights_dir
    var max_batches = config.max_batches
    var smoke = config.smoke

    var adamw = use_adamw(optimizer_name)

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Optimizer: ", "AdamW" if adamw else "SGD")
    if adamw:
        print("  Weight Decay: ", weight_decay)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    print()

    # Initialize model
    print("Initializing Simple CNN model...")
    var model = SimpleCNN()
    print("  Model initialized for 10 classes (digits 0-9)")
    print()

    # Load data — real MNIST, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset load entirely so the training
    # entrypoint can run per-PR in CI. It checks the MECHANISM (the loop runs
    # and emits finite, parseable loss), not convergence.
    var train_images: AnyTensor
    var train_labels: AnyTensor
    var test_images: AnyTensor
    var test_labels: AnyTensor
    if smoke:
        print("Smoke mode: using synthetic data (no dataset load)...")
        var wanted_batches = max_batches if max_batches > 0 else 3
        var n_smoke = wanted_batches * Int(batch_size)
        train_images = zeros([n_smoke, 1, 28, 28], DType.float32)
        var img_d = train_images._data.bitcast[Float32]()
        for s in range(n_smoke):
            var cls = s % DEFAULT_NUM_CLASSES
            for i in range(1 * 28 * 28):
                img_d[s * (1 * 28 * 28) + i] = (
                    Float32(cls) * 0.05 + Float32(i % 5) * 0.01
                )
        train_labels = zeros([n_smoke], DType.uint8)
        var lbl_d = train_labels._data.bitcast[UInt8]()
        for s in range(n_smoke):
            lbl_d[s] = UInt8(s % DEFAULT_NUM_CLASSES)
        # Reuse the synthetic batch for "test" (eval is not asserted here).
        test_images = train_images
        test_labels = train_labels
    else:
        print("Loading MNIST dataset...")
        var train_images_path = data_dir + "/train-images-idx3-ubyte"
        var train_labels_path = data_dir + "/train-labels-idx1-ubyte"
        var test_images_path = data_dir + "/t10k-images-idx3-ubyte"
        var test_labels_path = data_dir + "/t10k-labels-idx1-ubyte"

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

    # Training loop. Construct the optimizer ONCE (outside the epoch/batch
    # loops) so AdamW's moment buffers persist across steps.
    print("Starting training...")
    if adamw:
        var optimizer = AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        for epoch in range(1, epochs + 1):
            _ = train_epoch(
                model,
                train_images,
                train_labels,
                batch_size,
                epoch,
                epochs,
                optimizer,
                max_batches=max_batches,
            )
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
    else:
        var optimizer = SGD(learning_rate=learning_rate)
        for epoch in range(1, epochs + 1):
            _ = train_epoch(
                model,
                train_images,
                train_labels,
                batch_size,
                epoch,
                epochs,
                optimizer,
                max_batches=max_batches,
            )
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

    # Save model — skipped in smoke mode (#5551): a smoke run is a mechanism
    # check, so there is nothing to persist.
    if not smoke:
        print("Saving model weights...")
        model.save_weights(weights_dir)
        print("  Model saved to", weights_dir)
        print()

    print("Training complete!")
