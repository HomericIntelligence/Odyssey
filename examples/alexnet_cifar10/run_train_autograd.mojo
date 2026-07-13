"""Training Script for AlexNet on CIFAR-10 using Tape-Based Autograd.

Autograd variant of `run_train.mojo` (sub-task 3 of #5454). Replaces the manual
forward + `*_backward` + `model.update_parameters(...)` chain with the
Variable / GradientTape substrate: the AlexNet forward is recorded via
`variable_*` ops, gradients come from `loss.backward(tape)`, and parameters
update through `optimizer.step(params, tape)`.

Default optimizer is AdamW (matches examples/grok/lenet_emnist/); pass
`--optimizer sgd` to select SGD. The manual `run_train.mojo` is intentionally
left in place — the tracker-level cleanup (#5454) removes the manual path once
all ports land.

NOTE: dropout is OMITTED here. The manual AlexNet forward applies dropout after
fc1 and fc2 (training=True), but the autograd substrate has no `variable_dropout`
op (verified: no OP_DROPOUT / variable_dropout in src/odyssey/autograd/).
The recorded FC tail is therefore fc1 -> relu -> fc2 -> relu -> fc3, matching an
inference-mode / non-regularized forward. This is a mechanism port, not a
convergence-tuned run.

Also omitted vs the manual path: the step-LR schedule and SGD momentum (the
autograd SGD is constructed as SGD(learning_rate=...)); these are training-tuning
details, not required for the mechanism.

Usage:
    mojo run examples/alexnet_cifar10/run_train_autograd.mojo --epochs 1 --batch-size 32 --lr 0.01
    mojo run examples/alexnet_cifar10/run_train_autograd.mojo --optimizer sgd

Requirements:
    - CIFAR-10 dataset downloaded (run: python scripts/download_cifar10.py)
    - Dataset location: datasets/cifar10/
"""

from model import (
    AlexNet,
    CONV1_STRIDE,
    CONV1_PADDING,
    CONV2_STRIDE,
    CONV2_PADDING,
    CONV3_STRIDE,
    CONV3_PADDING,
    CONV4_STRIDE,
    CONV4_PADDING,
    CONV5_STRIDE,
    CONV5_PADDING,
    POOL1_KERNEL_SIZE,
    POOL1_STRIDE,
    POOL1_PADDING,
    POOL2_KERNEL_SIZE,
    POOL2_STRIDE,
    POOL2_PADDING,
    POOL3_KERNEL_SIZE,
    POOL3_STRIDE,
    POOL3_PADDING,
)
from odyssey.data.datasets import CIFAR10Dataset
from odyssey.data.formats import one_hot_encode
from odyssey.data.constants import DatasetInfo
from odyssey.autograd import (
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
from odyssey.autograd.optimizer_base import Optimizer
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.utils.arg_parser import create_training_parser
from std.collections import List


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

    AdamW is the default (DoD of #5559); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd" and
    `parse()` pre-populates it, so we cannot rely on `get_string`'s default —
    the caller gates on `was_user_supplied` before reading the value.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "alexnet_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var epochs = args.resolve_int("epochs", 100)
    var batch_size = args.resolve_int("batch-size", 128)
    var weight_decay = args.resolve_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates it into the
    # value map — so `get_string("optimizer", "adamw")` would return "sgd" even
    # when unset (the fallback is dead). Gate on `was_user_supplied`: only an
    # explicit `--optimizer` is honored; unset defaults to AdamW (DoD of #5559).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    # Optimizer-appropriate default learning rate. AlexNet is deep (5 conv +
    # 3 FC); AdamW's adaptive step diverges on this stack at the SGD-scale lr,
    # so default AdamW to 1e-4 and SGD to 1e-2 when the user did not pass --lr.
    # An explicit --lr always wins.
    var learning_rate: Float64
    if args.was_user_supplied("lr"):
        learning_rate = args.get_float("lr", 0.0001)
    elif optimizer == "sgd":
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    var data_dir = args.resolve_string("data-dir", "datasets/cifar10")
    var weights_dir = args.resolve_string(
        "weights-dir", "alexnet_weights_autograd"
    )
    var max_batches = args.resolve_int("max-batches", 0)
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
    mut model: AlexNet,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the AlexNet forward pass with `variable_*` ops (dropout omitted —
    no autograd dropout op exists), calls `loss.backward(tape)` for automatic
    gradients, then `optimizer.step(...)`.

    Args:
        model: AlexNet model (16 trainable tensors).
        input: Batch of images (batch, 3, 32, 32).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    # Wrap the model's 16 parameters as trainable Variables (field order matches
    # the write-back below), and the batch tensors as non-trainable inputs.
    var conv1_kernel = Variable(model.conv1_kernel, True, tape)
    var conv1_bias = Variable(model.conv1_bias, True, tape)
    var conv2_kernel = Variable(model.conv2_kernel, True, tape)
    var conv2_bias = Variable(model.conv2_bias, True, tape)
    var conv3_kernel = Variable(model.conv3_kernel, True, tape)
    var conv3_bias = Variable(model.conv3_bias, True, tape)
    var conv4_kernel = Variable(model.conv4_kernel, True, tape)
    var conv4_bias = Variable(model.conv4_bias, True, tape)
    var conv5_kernel = Variable(model.conv5_kernel, True, tape)
    var conv5_bias = Variable(model.conv5_bias, True, tape)
    var fc1_weights = Variable(model.fc1_weights, True, tape)
    var fc1_bias = Variable(model.fc1_bias, True, tape)
    var fc2_weights = Variable(model.fc2_weights, True, tape)
    var fc2_bias = Variable(model.fc2_bias, True, tape)
    var fc3_weights = Variable(model.fc3_weights, True, tape)
    var fc3_bias = Variable(model.fc3_bias, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Forward Pass (recorded to tape) ==========
    # AlexNet pools after conv1, conv2, conv5 only (conv3/conv4 have no pool).

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

    var conv3_out = variable_conv2d(
        pool2_out,
        conv3_kernel,
        conv3_bias,
        tape,
        stride=CONV3_STRIDE,
        padding=CONV3_PADDING,
    )
    var relu3_out = variable_relu(conv3_out, tape)

    var conv4_out = variable_conv2d(
        relu3_out,
        conv4_kernel,
        conv4_bias,
        tape,
        stride=CONV4_STRIDE,
        padding=CONV4_PADDING,
    )
    var relu4_out = variable_relu(conv4_out, tape)

    var conv5_out = variable_conv2d(
        relu4_out,
        conv5_kernel,
        conv5_bias,
        tape,
        stride=CONV5_STRIDE,
        padding=CONV5_PADDING,
    )
    var relu5_out = variable_relu(conv5_out, tape)
    var pool3_out = variable_maxpool2d(
        relu5_out,
        tape,
        kernel_size=POOL3_KERNEL_SIZE,
        stride=POOL3_STRIDE,
        padding=POOL3_PADDING,
    )

    # Flatten
    var flattened = variable_flatten(pool3_out, tape)

    # FC1 + ReLU (dropout omitted — see module docstring)
    var fc1_out = variable_linear(flattened, fc1_weights, fc1_bias, tape)
    var relu6_out = variable_relu(fc1_out, tape)

    # FC2 + ReLU (dropout omitted)
    var fc2_out = variable_linear(relu6_out, fc2_weights, fc2_bias, tape)
    var relu7_out = variable_relu(fc2_out, tape)

    # FC3 (output logits)
    var logits = variable_linear(relu7_out, fc3_weights, fc3_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update ==========
    # Move the 16 trainable Variables into the parameters list in field order;
    # the write-back below uses the same order.
    var parameters: List[Variable] = []
    parameters.append(conv1_kernel^)
    parameters.append(conv1_bias^)
    parameters.append(conv2_kernel^)
    parameters.append(conv2_bias^)
    parameters.append(conv3_kernel^)
    parameters.append(conv3_bias^)
    parameters.append(conv4_kernel^)
    parameters.append(conv4_bias^)
    parameters.append(conv5_kernel^)
    parameters.append(conv5_bias^)
    parameters.append(fc1_weights^)
    parameters.append(fc1_bias^)
    parameters.append(fc2_weights^)
    parameters.append(fc2_bias^)
    parameters.append(fc3_weights^)
    parameters.append(fc3_bias^)

    optimizer.step(parameters, tape)

    # Copy updated Variable.data back into the model, same order as appended.
    model.conv1_kernel = parameters[0].data
    model.conv1_bias = parameters[1].data
    model.conv2_kernel = parameters[2].data
    model.conv2_bias = parameters[3].data
    model.conv3_kernel = parameters[4].data
    model.conv3_bias = parameters[5].data
    model.conv4_kernel = parameters[6].data
    model.conv4_bias = parameters[7].data
    model.conv5_kernel = parameters[8].data
    model.conv5_bias = parameters[9].data
    model.fc1_weights = parameters[10].data
    model.fc1_bias = parameters[11].data
    model.fc2_weights = parameters[12].data
    model.fc2_bias = parameters[13].data
    model.fc3_weights = parameters[14].data
    model.fc3_bias = parameters[15].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: AlexNet,
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
        model: AlexNet model.
        train_images: Training images (num_samples, 3, 32, 32).
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

        # Slice the batch (dtype-safe; the proven alexnet batching path).
        var batch_images = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels_int = train_labels.slice(start_idx, end_idx, axis=0)

        # One-hot encode the batch labels.
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=model.num_classes
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
    print("AlexNet Training on CIFAR-10 Dataset (Autograd)")
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
    print("Initializing AlexNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = AlexNet(
        num_classes=dataset_info.num_classes(), dropout_rate=0.5
    )
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load data — real CIFAR-10, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset download so the training entrypoint
    # can run per-PR in CI. It checks the MECHANISM (the loop runs and emits
    # finite, parseable loss), not convergence.
    # Only training data is needed — this port does not run an eval loop, so it
    # loads no test split (avoids an unused-assignment --Werror error).
    var train_images: AnyTensor
    var train_labels: AnyTensor
    if smoke:
        print("Smoke mode: using synthetic data (no dataset download)...")
        var wanted_batches = max_batches if max_batches > 0 else 3
        var n_smoke = wanted_batches * Int(batch_size)
        train_images = zeros([n_smoke, 3, 32, 32], DType.float32)
        var img_d = train_images._data.bitcast[Float32]()
        for s in range(n_smoke):
            var cls = s % 10
            for i in range(3 * 32 * 32):
                img_d[s * (3 * 32 * 32) + i] = (
                    Float32(cls) * 0.05 + Float32(i % 5) * 0.01
                )
        train_labels = zeros([n_smoke], DType.uint8)
        var lbl_d = train_labels._data.bitcast[UInt8]()
        for s in range(n_smoke):
            lbl_d[s] = UInt8(s % 10)
    else:
        print("Loading CIFAR-10 dataset...")
        var dataset = CIFAR10Dataset(data_dir)
        var train_data = dataset.get_train_data()
        train_images = train_data[0]
        train_labels = train_data[1]

    print("  Training samples: ", train_images.shape()[0])
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

    # Save model — skipped in smoke mode (#5551): a smoke run is a mechanism
    # check, so there is nothing to persist.
    if not smoke:
        print("Saving model weights...")
        model.save_weights(weights_dir)
        print("  Model saved to", weights_dir)
        print()

    print("Training complete!")
