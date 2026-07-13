"""Training Script for VGG-16 on CIFAR-10 using Tape-Based Autograd.

Autograd variant of `train.mojo` (sub-task 4 of #5454). Replaces the manual
forward + 32-parameter `*_backward` chain + manual SGD updates with the
Variable / GradientTape substrate: the VGG-16 forward is recorded via
`variable_*` ops, gradients come from `loss.backward(tape)`, and parameters
update through `optimizer.step(params, tape)`.

Default optimizer is AdamW (matches examples/grok/lenet_emnist/); pass
`--optimizer sgd` to select SGD. The manual `train.mojo` is intentionally left
in place — the tracker-level cleanup (#5454) removes the manual path once all
ports land.

NOTE: dropout is OMITTED here. The manual VGG-16 forward applies dropout after
fc1 and fc2 (training=True), but the autograd substrate has no
`variable_dropout` op (verified: no OP_DROPOUT / variable_dropout in
src/odyssey/autograd/). The recorded FC tail is fc1 -> relu -> fc2 ->
relu -> fc3, matching an inference-mode / non-regularized forward. This is a
mechanism port, not a convergence-tuned run.

Also omitted vs the manual path: the step-LR schedule and SGD momentum
(the autograd SGD is constructed as SGD(learning_rate=...)); these are
training-tuning details, not required for the mechanism.

Usage:
    mojo run examples/vgg16_cifar10/train_autograd.mojo --epochs 1 --batch-size 32
    mojo run examples/vgg16_cifar10/train_autograd.mojo --optimizer sgd

Requirements:
    - CIFAR-10 dataset downloaded (run: python scripts/download_cifar10.py)
    - Dataset location: datasets/cifar10/
"""

from model import (
    VGG16,
    CONV_STRIDE,
    CONV_PADDING,
    POOL_KERNEL_SIZE,
    POOL_STRIDE,
    POOL_PADDING,
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

    AdamW is the default (DoD of #5560); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd" and
    `parse()` pre-populates it, so we cannot rely on `get_string`'s default —
    the caller gates on `was_user_supplied` before reading the value.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "vgg16_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var epochs = args.resolve_int("epochs", 100)
    var batch_size = args.resolve_int("batch-size", 128)
    var weight_decay = args.resolve_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates it into the
    # value map — so `get_string("optimizer", "adamw")` would return "sgd" even
    # when unset (the fallback is dead). Gate on `was_user_supplied`: only an
    # explicit `--optimizer` is honored; unset defaults to AdamW (DoD of #5560).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    # Optimizer-appropriate default learning rate. VGG-16 is very deep (13 conv
    # + 3 FC); AdamW's adaptive step diverges on this stack at the SGD-scale lr,
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
        "weights-dir", "vgg16_weights_autograd"
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
    mut model: VGG16,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the VGG-16 forward pass with `variable_*` ops (dropout omitted —
    no autograd dropout op exists), calls `loss.backward(tape)` for automatic
    gradients, then `optimizer.step(...)`.

    Args:
        model: VGG-16 model (32 trainable tensors).
        input: Batch of images (batch, 3, 32, 32).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    # Wrap the model's 32 parameters as trainable Variables (field order matches
    # the write-back below), and the batch tensors as non-trainable inputs.
    var conv1_1_kernel = Variable(model.conv1_1_kernel, True, tape)
    var conv1_1_bias = Variable(model.conv1_1_bias, True, tape)
    var conv1_2_kernel = Variable(model.conv1_2_kernel, True, tape)
    var conv1_2_bias = Variable(model.conv1_2_bias, True, tape)
    var conv2_1_kernel = Variable(model.conv2_1_kernel, True, tape)
    var conv2_1_bias = Variable(model.conv2_1_bias, True, tape)
    var conv2_2_kernel = Variable(model.conv2_2_kernel, True, tape)
    var conv2_2_bias = Variable(model.conv2_2_bias, True, tape)
    var conv3_1_kernel = Variable(model.conv3_1_kernel, True, tape)
    var conv3_1_bias = Variable(model.conv3_1_bias, True, tape)
    var conv3_2_kernel = Variable(model.conv3_2_kernel, True, tape)
    var conv3_2_bias = Variable(model.conv3_2_bias, True, tape)
    var conv3_3_kernel = Variable(model.conv3_3_kernel, True, tape)
    var conv3_3_bias = Variable(model.conv3_3_bias, True, tape)
    var conv4_1_kernel = Variable(model.conv4_1_kernel, True, tape)
    var conv4_1_bias = Variable(model.conv4_1_bias, True, tape)
    var conv4_2_kernel = Variable(model.conv4_2_kernel, True, tape)
    var conv4_2_bias = Variable(model.conv4_2_bias, True, tape)
    var conv4_3_kernel = Variable(model.conv4_3_kernel, True, tape)
    var conv4_3_bias = Variable(model.conv4_3_bias, True, tape)
    var conv5_1_kernel = Variable(model.conv5_1_kernel, True, tape)
    var conv5_1_bias = Variable(model.conv5_1_bias, True, tape)
    var conv5_2_kernel = Variable(model.conv5_2_kernel, True, tape)
    var conv5_2_bias = Variable(model.conv5_2_bias, True, tape)
    var conv5_3_kernel = Variable(model.conv5_3_kernel, True, tape)
    var conv5_3_bias = Variable(model.conv5_3_bias, True, tape)
    var fc1_weights = Variable(model.fc1_weights, True, tape)
    var fc1_bias = Variable(model.fc1_bias, True, tape)
    var fc2_weights = Variable(model.fc2_weights, True, tape)
    var fc2_bias = Variable(model.fc2_bias, True, tape)
    var fc3_weights = Variable(model.fc3_weights, True, tape)
    var fc3_bias = Variable(model.fc3_bias, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Forward Pass (recorded to tape) ==========
    # VGG-16: 5 conv blocks, each ending in a 2x2 maxpool.

    # Block 1: conv1_1 -> relu -> conv1_2 -> relu -> pool
    var c11 = variable_conv2d(
        input_var,
        conv1_1_kernel,
        conv1_1_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r11 = variable_relu(c11, tape)
    var c12 = variable_conv2d(
        r11,
        conv1_2_kernel,
        conv1_2_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r12 = variable_relu(c12, tape)
    var p1 = variable_maxpool2d(
        r12,
        tape,
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )

    # Block 2: conv2_1 -> relu -> conv2_2 -> relu -> pool
    var c21 = variable_conv2d(
        p1,
        conv2_1_kernel,
        conv2_1_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r21 = variable_relu(c21, tape)
    var c22 = variable_conv2d(
        r21,
        conv2_2_kernel,
        conv2_2_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r22 = variable_relu(c22, tape)
    var p2 = variable_maxpool2d(
        r22,
        tape,
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )

    # Block 3: conv3_1 -> relu -> conv3_2 -> relu -> conv3_3 -> relu -> pool
    var c31 = variable_conv2d(
        p2,
        conv3_1_kernel,
        conv3_1_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r31 = variable_relu(c31, tape)
    var c32 = variable_conv2d(
        r31,
        conv3_2_kernel,
        conv3_2_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r32 = variable_relu(c32, tape)
    var c33 = variable_conv2d(
        r32,
        conv3_3_kernel,
        conv3_3_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r33 = variable_relu(c33, tape)
    var p3 = variable_maxpool2d(
        r33,
        tape,
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )

    # Block 4: conv4_1 -> relu -> conv4_2 -> relu -> conv4_3 -> relu -> pool
    var c41 = variable_conv2d(
        p3,
        conv4_1_kernel,
        conv4_1_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r41 = variable_relu(c41, tape)
    var c42 = variable_conv2d(
        r41,
        conv4_2_kernel,
        conv4_2_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r42 = variable_relu(c42, tape)
    var c43 = variable_conv2d(
        r42,
        conv4_3_kernel,
        conv4_3_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r43 = variable_relu(c43, tape)
    var p4 = variable_maxpool2d(
        r43,
        tape,
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )

    # Block 5: conv5_1 -> relu -> conv5_2 -> relu -> conv5_3 -> relu -> pool
    var c51 = variable_conv2d(
        p4,
        conv5_1_kernel,
        conv5_1_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r51 = variable_relu(c51, tape)
    var c52 = variable_conv2d(
        r51,
        conv5_2_kernel,
        conv5_2_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r52 = variable_relu(c52, tape)
    var c53 = variable_conv2d(
        r52,
        conv5_3_kernel,
        conv5_3_bias,
        tape,
        stride=CONV_STRIDE,
        padding=CONV_PADDING,
    )
    var r53 = variable_relu(c53, tape)
    var p5 = variable_maxpool2d(
        r53,
        tape,
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )

    # Flatten
    var flattened = variable_flatten(p5, tape)

    # FC1 + ReLU (dropout omitted)
    var fc1_out = variable_linear(flattened, fc1_weights, fc1_bias, tape)
    var relu_fc1 = variable_relu(fc1_out, tape)

    # FC2 + ReLU (dropout omitted)
    var fc2_out = variable_linear(relu_fc1, fc2_weights, fc2_bias, tape)
    var relu_fc2 = variable_relu(fc2_out, tape)

    # FC3 (output logits)
    var logits = variable_linear(relu_fc2, fc3_weights, fc3_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update ==========
    # Move the 32 trainable Variables into the parameters list in field order;
    # the write-back below uses the same order.
    var parameters: List[Variable] = []
    parameters.append(conv1_1_kernel^)
    parameters.append(conv1_1_bias^)
    parameters.append(conv1_2_kernel^)
    parameters.append(conv1_2_bias^)
    parameters.append(conv2_1_kernel^)
    parameters.append(conv2_1_bias^)
    parameters.append(conv2_2_kernel^)
    parameters.append(conv2_2_bias^)
    parameters.append(conv3_1_kernel^)
    parameters.append(conv3_1_bias^)
    parameters.append(conv3_2_kernel^)
    parameters.append(conv3_2_bias^)
    parameters.append(conv3_3_kernel^)
    parameters.append(conv3_3_bias^)
    parameters.append(conv4_1_kernel^)
    parameters.append(conv4_1_bias^)
    parameters.append(conv4_2_kernel^)
    parameters.append(conv4_2_bias^)
    parameters.append(conv4_3_kernel^)
    parameters.append(conv4_3_bias^)
    parameters.append(conv5_1_kernel^)
    parameters.append(conv5_1_bias^)
    parameters.append(conv5_2_kernel^)
    parameters.append(conv5_2_bias^)
    parameters.append(conv5_3_kernel^)
    parameters.append(conv5_3_bias^)
    parameters.append(fc1_weights^)
    parameters.append(fc1_bias^)
    parameters.append(fc2_weights^)
    parameters.append(fc2_bias^)
    parameters.append(fc3_weights^)
    parameters.append(fc3_bias^)

    optimizer.step(parameters, tape)

    # Copy updated Variable.data back into the model, same order as appended.
    model.conv1_1_kernel = parameters[0].data
    model.conv1_1_bias = parameters[1].data
    model.conv1_2_kernel = parameters[2].data
    model.conv1_2_bias = parameters[3].data
    model.conv2_1_kernel = parameters[4].data
    model.conv2_1_bias = parameters[5].data
    model.conv2_2_kernel = parameters[6].data
    model.conv2_2_bias = parameters[7].data
    model.conv3_1_kernel = parameters[8].data
    model.conv3_1_bias = parameters[9].data
    model.conv3_2_kernel = parameters[10].data
    model.conv3_2_bias = parameters[11].data
    model.conv3_3_kernel = parameters[12].data
    model.conv3_3_bias = parameters[13].data
    model.conv4_1_kernel = parameters[14].data
    model.conv4_1_bias = parameters[15].data
    model.conv4_2_kernel = parameters[16].data
    model.conv4_2_bias = parameters[17].data
    model.conv4_3_kernel = parameters[18].data
    model.conv4_3_bias = parameters[19].data
    model.conv5_1_kernel = parameters[20].data
    model.conv5_1_bias = parameters[21].data
    model.conv5_2_kernel = parameters[22].data
    model.conv5_2_bias = parameters[23].data
    model.conv5_3_kernel = parameters[24].data
    model.conv5_3_bias = parameters[25].data
    model.fc1_weights = parameters[26].data
    model.fc1_bias = parameters[27].data
    model.fc2_weights = parameters[28].data
    model.fc2_bias = parameters[29].data
    model.fc3_weights = parameters[30].data
    model.fc3_bias = parameters[31].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: VGG16,
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
        model: VGG-16 model.
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

        # Slice the batch (dtype-safe; the proven convnet batching path).
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
    print("VGG-16 Training on CIFAR-10 Dataset (Autograd)")
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
    print("Initializing VGG-16 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = VGG16(num_classes=dataset_info.num_classes(), dropout_rate=0.5)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load data — real CIFAR-10, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset download so the training entrypoint
    # can run per-PR in CI. It checks the MECHANISM (the loop runs and emits
    # finite, parseable loss), not convergence. Only training data is loaded —
    # this port runs no eval loop (loading a test split would leave dead
    # assignments that `mojo build --Werror` rejects).
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
        # RAW uint8 class indices [N] — train_epoch one-hot-encodes per batch
        # (unlike the manual train.mojo, which builds one-hot labels because it
        # feeds cross_entropy directly).
        train_labels = zeros([n_smoke], DType.uint8)
        var lbl_d = train_labels._data.bitcast[UInt8]()
        for s in range(n_smoke):
            lbl_d[s] = UInt8(s % 10)
    else:
        print("Loading CIFAR-10 dataset...")
        var cifar_dataset = CIFAR10Dataset(data_dir)
        var train_data = cifar_dataset.get_train_data()
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
