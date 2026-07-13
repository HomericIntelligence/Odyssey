"""Training Script for ResNet-18 on CIFAR-10 using Tape-Based Autograd.

Autograd variant of `train.mojo` (sub-task 5 of #5454). Replaces the manual
forward + 82-parameter `*_backward` chain (`conv2d_backward` /
`batch_norm2d` / `avgpool2d_backward` / manual SGD-momentum updates) with the
Variable / GradientTape substrate: the ResNet-18 forward is recorded via
`variable_*` ops, gradients come from `loss.backward(tape)`, and parameters
update through `optimizer.step(params, tape)`.

Default optimizer is AdamW (matches examples/grok/lenet_emnist/); pass
`--optimizer sgd` to select SGD. The manual `train.mojo` is intentionally left
in place — the tracker-level cleanup (#5454) removes the manual path once all
ports land.

Batch normalization IS ported: `variable_batch_norm` returns
`(output_var, new_running_mean, new_running_var)`. `gamma`/`beta` are trainable
Variables; `running_mean`/`running_var` are plain `AnyTensor` buffers threaded
back into the model's fields after every layer (exactly like the manual
`forward()`), so the running statistics keep advancing. `training=True` is used
here (a training port; no eval loop).

GLOBAL AVERAGE POOLING substitution: the manual forward ends with
`avgpool2d(x, kernel_size=4)` over the (batch, 512, 4, 4) stage-4 output, then a
reshape to (batch, 512) before the FC. The autograd substrate has NO
`variable_avgpool2d`, so GAP is expressed as two `variable_mean` reductions over
the two spatial axes: `variable_mean(x, tape, axis=3)` -> (batch, 512, 4), then
`variable_mean(..., tape, axis=2)` -> (batch, 512). Averaging both spatial axes
is exactly global average pooling, and the result is already the flattened
(batch, 512) shape, so NO `variable_flatten` is needed before the FC. (NOTE:
`variable_mean`'s `axis=-1` default is a FULL reduction to a scalar, not a
last-axis reduction — the explicit positive axes 3 then 2 are required.)

Residual skip connections use `variable_add` (replacing the manual `add`):
identity blocks add `bn2_out + block_input`; projection blocks (s2b1/s3b1/s4b1)
add `bn2_out + proj_bn_out`. Each block ends with `variable_relu` after the add.

Omitted vs the manual path: the step-LR schedule and SGD momentum (the autograd
SGD is constructed as SGD(learning_rate=...)); these are training-tuning
details, not required for the mechanism. ResNet-18 has no dropout, so nothing is
dropped on that front.

Usage:
    mojo run examples/resnet18_cifar10/train_autograd.mojo --epochs 1 --batch-size 32
    mojo run examples/resnet18_cifar10/train_autograd.mojo --optimizer sgd

Requirements:
    - CIFAR-10 dataset downloaded (run: python scripts/download_cifar10.py)
    - Dataset location: datasets/cifar10/
"""

from model import ResNet18
from projectodyssey.data.datasets import CIFAR10Dataset
from projectodyssey.data.formats import one_hot_encode
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.autograd import (
    Variable,
    GradientTape,
    AdamW,
    SGD,
    variable_conv2d,
    variable_relu,
    variable_linear,
    variable_cross_entropy,
    variable_add,
    variable_mean,
    variable_batch_norm,
)
from projectodyssey.autograd.optimizer_base import Optimizer
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.utils.arg_parser import create_training_parser
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

    AdamW is the default (DoD of #5561); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd" and
    `parse()` pre-populates it, so we cannot rely on `get_string`'s default —
    the caller gates on `was_user_supplied` before reading the value.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "resnet18_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var epochs = args.resolve_int("epochs", 100)
    var batch_size = args.resolve_int("batch-size", 128)
    var weight_decay = args.resolve_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates it into the
    # value map — so `get_string("optimizer", "adamw")` would return "sgd" even
    # when unset (the fallback is dead). Gate on `was_user_supplied`: only an
    # explicit `--optimizer` is honored; unset defaults to AdamW (DoD of #5561).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    # Optimizer-appropriate default learning rate. ResNet-18 is deep (17 conv +
    # 1 FC, 82 params); AdamW's adaptive step diverges on this stack at the
    # SGD-scale lr, so default AdamW to 1e-4 and SGD to 1e-2 when the user did
    # not pass --lr. An explicit --lr always wins.
    var learning_rate: Float64
    if args.was_user_supplied("lr"):
        learning_rate = args.get_float("lr", 0.0001)
    elif optimizer == "sgd":
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    var data_dir = args.resolve_string("data-dir", "datasets/cifar10")
    var weights_dir = args.get_string(
        "weights-dir", "resnet18_weights_autograd"
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
    mut model: ResNet18,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the ResNet-18 forward pass with `variable_*` ops (batch norm,
    residual `variable_add` skips, GAP via two `variable_mean` reductions),
    calls `loss.backward(tape)` for automatic gradients, then
    `optimizer.step(...)`.

    Args:
        model: ResNet-18 model (82 trainable tensors + BN running stat buffers).
        input: Batch of images (batch, 3, 32, 32).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    # Wrap the model's 82 trainable parameters as Variables (field order matches
    # the parameters list + write-back below). BN running_mean/var stay plain
    # AnyTensor buffers (non-trainable) and are threaded through/back separately.
    # --- Initial conv + BN ---
    var conv1_kernel = Variable(model.conv1_kernel, True, tape)
    var conv1_bias = Variable(model.conv1_bias, True, tape)
    var bn1_gamma = Variable(model.bn1_gamma, True, tape)
    var bn1_beta = Variable(model.bn1_beta, True, tape)
    # --- Stage 1 ---
    var s1b1_conv1_kernel = Variable(model.s1b1_conv1_kernel, True, tape)
    var s1b1_conv1_bias = Variable(model.s1b1_conv1_bias, True, tape)
    var s1b1_bn1_gamma = Variable(model.s1b1_bn1_gamma, True, tape)
    var s1b1_bn1_beta = Variable(model.s1b1_bn1_beta, True, tape)
    var s1b1_conv2_kernel = Variable(model.s1b1_conv2_kernel, True, tape)
    var s1b1_conv2_bias = Variable(model.s1b1_conv2_bias, True, tape)
    var s1b1_bn2_gamma = Variable(model.s1b1_bn2_gamma, True, tape)
    var s1b1_bn2_beta = Variable(model.s1b1_bn2_beta, True, tape)
    var s1b2_conv1_kernel = Variable(model.s1b2_conv1_kernel, True, tape)
    var s1b2_conv1_bias = Variable(model.s1b2_conv1_bias, True, tape)
    var s1b2_bn1_gamma = Variable(model.s1b2_bn1_gamma, True, tape)
    var s1b2_bn1_beta = Variable(model.s1b2_bn1_beta, True, tape)
    var s1b2_conv2_kernel = Variable(model.s1b2_conv2_kernel, True, tape)
    var s1b2_conv2_bias = Variable(model.s1b2_conv2_bias, True, tape)
    var s1b2_bn2_gamma = Variable(model.s1b2_bn2_gamma, True, tape)
    var s1b2_bn2_beta = Variable(model.s1b2_bn2_beta, True, tape)
    # --- Stage 2 (block1 has projection) ---
    var s2b1_conv1_kernel = Variable(model.s2b1_conv1_kernel, True, tape)
    var s2b1_conv1_bias = Variable(model.s2b1_conv1_bias, True, tape)
    var s2b1_bn1_gamma = Variable(model.s2b1_bn1_gamma, True, tape)
    var s2b1_bn1_beta = Variable(model.s2b1_bn1_beta, True, tape)
    var s2b1_conv2_kernel = Variable(model.s2b1_conv2_kernel, True, tape)
    var s2b1_conv2_bias = Variable(model.s2b1_conv2_bias, True, tape)
    var s2b1_bn2_gamma = Variable(model.s2b1_bn2_gamma, True, tape)
    var s2b1_bn2_beta = Variable(model.s2b1_bn2_beta, True, tape)
    var s2b1_proj_kernel = Variable(model.s2b1_proj_kernel, True, tape)
    var s2b1_proj_bias = Variable(model.s2b1_proj_bias, True, tape)
    var s2b1_proj_bn_gamma = Variable(model.s2b1_proj_bn_gamma, True, tape)
    var s2b1_proj_bn_beta = Variable(model.s2b1_proj_bn_beta, True, tape)
    var s2b2_conv1_kernel = Variable(model.s2b2_conv1_kernel, True, tape)
    var s2b2_conv1_bias = Variable(model.s2b2_conv1_bias, True, tape)
    var s2b2_bn1_gamma = Variable(model.s2b2_bn1_gamma, True, tape)
    var s2b2_bn1_beta = Variable(model.s2b2_bn1_beta, True, tape)
    var s2b2_conv2_kernel = Variable(model.s2b2_conv2_kernel, True, tape)
    var s2b2_conv2_bias = Variable(model.s2b2_conv2_bias, True, tape)
    var s2b2_bn2_gamma = Variable(model.s2b2_bn2_gamma, True, tape)
    var s2b2_bn2_beta = Variable(model.s2b2_bn2_beta, True, tape)
    # --- Stage 3 (block1 has projection) ---
    var s3b1_conv1_kernel = Variable(model.s3b1_conv1_kernel, True, tape)
    var s3b1_conv1_bias = Variable(model.s3b1_conv1_bias, True, tape)
    var s3b1_bn1_gamma = Variable(model.s3b1_bn1_gamma, True, tape)
    var s3b1_bn1_beta = Variable(model.s3b1_bn1_beta, True, tape)
    var s3b1_conv2_kernel = Variable(model.s3b1_conv2_kernel, True, tape)
    var s3b1_conv2_bias = Variable(model.s3b1_conv2_bias, True, tape)
    var s3b1_bn2_gamma = Variable(model.s3b1_bn2_gamma, True, tape)
    var s3b1_bn2_beta = Variable(model.s3b1_bn2_beta, True, tape)
    var s3b1_proj_kernel = Variable(model.s3b1_proj_kernel, True, tape)
    var s3b1_proj_bias = Variable(model.s3b1_proj_bias, True, tape)
    var s3b1_proj_bn_gamma = Variable(model.s3b1_proj_bn_gamma, True, tape)
    var s3b1_proj_bn_beta = Variable(model.s3b1_proj_bn_beta, True, tape)
    var s3b2_conv1_kernel = Variable(model.s3b2_conv1_kernel, True, tape)
    var s3b2_conv1_bias = Variable(model.s3b2_conv1_bias, True, tape)
    var s3b2_bn1_gamma = Variable(model.s3b2_bn1_gamma, True, tape)
    var s3b2_bn1_beta = Variable(model.s3b2_bn1_beta, True, tape)
    var s3b2_conv2_kernel = Variable(model.s3b2_conv2_kernel, True, tape)
    var s3b2_conv2_bias = Variable(model.s3b2_conv2_bias, True, tape)
    var s3b2_bn2_gamma = Variable(model.s3b2_bn2_gamma, True, tape)
    var s3b2_bn2_beta = Variable(model.s3b2_bn2_beta, True, tape)
    # --- Stage 4 (block1 has projection) ---
    var s4b1_conv1_kernel = Variable(model.s4b1_conv1_kernel, True, tape)
    var s4b1_conv1_bias = Variable(model.s4b1_conv1_bias, True, tape)
    var s4b1_bn1_gamma = Variable(model.s4b1_bn1_gamma, True, tape)
    var s4b1_bn1_beta = Variable(model.s4b1_bn1_beta, True, tape)
    var s4b1_conv2_kernel = Variable(model.s4b1_conv2_kernel, True, tape)
    var s4b1_conv2_bias = Variable(model.s4b1_conv2_bias, True, tape)
    var s4b1_bn2_gamma = Variable(model.s4b1_bn2_gamma, True, tape)
    var s4b1_bn2_beta = Variable(model.s4b1_bn2_beta, True, tape)
    var s4b1_proj_kernel = Variable(model.s4b1_proj_kernel, True, tape)
    var s4b1_proj_bias = Variable(model.s4b1_proj_bias, True, tape)
    var s4b1_proj_bn_gamma = Variable(model.s4b1_proj_bn_gamma, True, tape)
    var s4b1_proj_bn_beta = Variable(model.s4b1_proj_bn_beta, True, tape)
    var s4b2_conv1_kernel = Variable(model.s4b2_conv1_kernel, True, tape)
    var s4b2_conv1_bias = Variable(model.s4b2_conv1_bias, True, tape)
    var s4b2_bn1_gamma = Variable(model.s4b2_bn1_gamma, True, tape)
    var s4b2_bn1_beta = Variable(model.s4b2_bn1_beta, True, tape)
    var s4b2_conv2_kernel = Variable(model.s4b2_conv2_kernel, True, tape)
    var s4b2_conv2_bias = Variable(model.s4b2_conv2_bias, True, tape)
    var s4b2_bn2_gamma = Variable(model.s4b2_bn2_gamma, True, tape)
    var s4b2_bn2_beta = Variable(model.s4b2_bn2_beta, True, tape)
    # --- FC head ---
    var fc_weights = Variable(model.fc_weights, True, tape)
    var fc_bias = Variable(model.fc_bias, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Forward Pass (recorded to tape) ==========
    # Initial conv (stride 1, pad 1) + BN + ReLU. BN returns updated running
    # stats which are threaded back into the model immediately (as the manual
    # forward() does), so later batches keep advancing the statistics.
    var conv1_out = variable_conv2d(
        input_var, conv1_kernel, conv1_bias, tape, stride=1, padding=1
    )
    var bn1_res = variable_batch_norm(
        conv1_out,
        bn1_gamma,
        bn1_beta,
        model.bn1_running_mean,
        model.bn1_running_var,
        tape,
        training=True,
    )
    var bn1_out = bn1_res[0].copy()
    model.bn1_running_mean = bn1_res[1]
    model.bn1_running_var = bn1_res[2]
    var relu1_out = variable_relu(bn1_out, tape)

    # ========== Stage 1, Block 1 (identity shortcut) ==========
    var s1b1_c1 = variable_conv2d(
        relu1_out,
        s1b1_conv1_kernel,
        s1b1_conv1_bias,
        tape,
        stride=1,
        padding=1,
    )
    var s1b1_bn1_res = variable_batch_norm(
        s1b1_c1,
        s1b1_bn1_gamma,
        s1b1_bn1_beta,
        model.s1b1_bn1_running_mean,
        model.s1b1_bn1_running_var,
        tape,
        training=True,
    )
    var s1b1_bn1_out = s1b1_bn1_res[0].copy()
    model.s1b1_bn1_running_mean = s1b1_bn1_res[1]
    model.s1b1_bn1_running_var = s1b1_bn1_res[2]
    var s1b1_r1 = variable_relu(s1b1_bn1_out, tape)
    var s1b1_c2 = variable_conv2d(
        s1b1_r1, s1b1_conv2_kernel, s1b1_conv2_bias, tape, stride=1, padding=1
    )
    var s1b1_bn2_res = variable_batch_norm(
        s1b1_c2,
        s1b1_bn2_gamma,
        s1b1_bn2_beta,
        model.s1b1_bn2_running_mean,
        model.s1b1_bn2_running_var,
        tape,
        training=True,
    )
    var s1b1_bn2_out = s1b1_bn2_res[0].copy()
    model.s1b1_bn2_running_mean = s1b1_bn2_res[1]
    model.s1b1_bn2_running_var = s1b1_bn2_res[2]
    var s1b1_skip = variable_add(s1b1_bn2_out, relu1_out, tape)
    var s1b1_out = variable_relu(s1b1_skip, tape)

    # ========== Stage 1, Block 2 (identity shortcut) ==========
    var s1b2_c1 = variable_conv2d(
        s1b1_out, s1b2_conv1_kernel, s1b2_conv1_bias, tape, stride=1, padding=1
    )
    var s1b2_bn1_res = variable_batch_norm(
        s1b2_c1,
        s1b2_bn1_gamma,
        s1b2_bn1_beta,
        model.s1b2_bn1_running_mean,
        model.s1b2_bn1_running_var,
        tape,
        training=True,
    )
    var s1b2_bn1_out = s1b2_bn1_res[0].copy()
    model.s1b2_bn1_running_mean = s1b2_bn1_res[1]
    model.s1b2_bn1_running_var = s1b2_bn1_res[2]
    var s1b2_r1 = variable_relu(s1b2_bn1_out, tape)
    var s1b2_c2 = variable_conv2d(
        s1b2_r1, s1b2_conv2_kernel, s1b2_conv2_bias, tape, stride=1, padding=1
    )
    var s1b2_bn2_res = variable_batch_norm(
        s1b2_c2,
        s1b2_bn2_gamma,
        s1b2_bn2_beta,
        model.s1b2_bn2_running_mean,
        model.s1b2_bn2_running_var,
        tape,
        training=True,
    )
    var s1b2_bn2_out = s1b2_bn2_res[0].copy()
    model.s1b2_bn2_running_mean = s1b2_bn2_res[1]
    model.s1b2_bn2_running_var = s1b2_bn2_res[2]
    var s1b2_skip = variable_add(s1b2_bn2_out, s1b1_out, tape)
    var s1b2_out = variable_relu(s1b2_skip, tape)

    # ========== Stage 2, Block 1 (projection shortcut, stride=2) ==========
    var s2b1_c1 = variable_conv2d(
        s1b2_out, s2b1_conv1_kernel, s2b1_conv1_bias, tape, stride=2, padding=1
    )
    var s2b1_bn1_res = variable_batch_norm(
        s2b1_c1,
        s2b1_bn1_gamma,
        s2b1_bn1_beta,
        model.s2b1_bn1_running_mean,
        model.s2b1_bn1_running_var,
        tape,
        training=True,
    )
    var s2b1_bn1_out = s2b1_bn1_res[0].copy()
    model.s2b1_bn1_running_mean = s2b1_bn1_res[1]
    model.s2b1_bn1_running_var = s2b1_bn1_res[2]
    var s2b1_r1 = variable_relu(s2b1_bn1_out, tape)
    var s2b1_c2 = variable_conv2d(
        s2b1_r1, s2b1_conv2_kernel, s2b1_conv2_bias, tape, stride=1, padding=1
    )
    var s2b1_bn2_res = variable_batch_norm(
        s2b1_c2,
        s2b1_bn2_gamma,
        s2b1_bn2_beta,
        model.s2b1_bn2_running_mean,
        model.s2b1_bn2_running_var,
        tape,
        training=True,
    )
    var s2b1_bn2_out = s2b1_bn2_res[0].copy()
    model.s2b1_bn2_running_mean = s2b1_bn2_res[1]
    model.s2b1_bn2_running_var = s2b1_bn2_res[2]
    # Projection shortcut: 1×1 conv, stride 2, pad 0.
    var s2b1_proj_c = variable_conv2d(
        s1b2_out, s2b1_proj_kernel, s2b1_proj_bias, tape, stride=2, padding=0
    )
    var s2b1_proj_bn_res = variable_batch_norm(
        s2b1_proj_c,
        s2b1_proj_bn_gamma,
        s2b1_proj_bn_beta,
        model.s2b1_proj_bn_running_mean,
        model.s2b1_proj_bn_running_var,
        tape,
        training=True,
    )
    var s2b1_proj_bn_out = s2b1_proj_bn_res[0].copy()
    model.s2b1_proj_bn_running_mean = s2b1_proj_bn_res[1]
    model.s2b1_proj_bn_running_var = s2b1_proj_bn_res[2]
    var s2b1_skip = variable_add(s2b1_bn2_out, s2b1_proj_bn_out, tape)
    var s2b1_out = variable_relu(s2b1_skip, tape)

    # ========== Stage 2, Block 2 (identity shortcut) ==========
    var s2b2_c1 = variable_conv2d(
        s2b1_out, s2b2_conv1_kernel, s2b2_conv1_bias, tape, stride=1, padding=1
    )
    var s2b2_bn1_res = variable_batch_norm(
        s2b2_c1,
        s2b2_bn1_gamma,
        s2b2_bn1_beta,
        model.s2b2_bn1_running_mean,
        model.s2b2_bn1_running_var,
        tape,
        training=True,
    )
    var s2b2_bn1_out = s2b2_bn1_res[0].copy()
    model.s2b2_bn1_running_mean = s2b2_bn1_res[1]
    model.s2b2_bn1_running_var = s2b2_bn1_res[2]
    var s2b2_r1 = variable_relu(s2b2_bn1_out, tape)
    var s2b2_c2 = variable_conv2d(
        s2b2_r1, s2b2_conv2_kernel, s2b2_conv2_bias, tape, stride=1, padding=1
    )
    var s2b2_bn2_res = variable_batch_norm(
        s2b2_c2,
        s2b2_bn2_gamma,
        s2b2_bn2_beta,
        model.s2b2_bn2_running_mean,
        model.s2b2_bn2_running_var,
        tape,
        training=True,
    )
    var s2b2_bn2_out = s2b2_bn2_res[0].copy()
    model.s2b2_bn2_running_mean = s2b2_bn2_res[1]
    model.s2b2_bn2_running_var = s2b2_bn2_res[2]
    var s2b2_skip = variable_add(s2b2_bn2_out, s2b1_out, tape)
    var s2b2_out = variable_relu(s2b2_skip, tape)

    # ========== Stage 3, Block 1 (projection shortcut, stride=2) ==========
    var s3b1_c1 = variable_conv2d(
        s2b2_out, s3b1_conv1_kernel, s3b1_conv1_bias, tape, stride=2, padding=1
    )
    var s3b1_bn1_res = variable_batch_norm(
        s3b1_c1,
        s3b1_bn1_gamma,
        s3b1_bn1_beta,
        model.s3b1_bn1_running_mean,
        model.s3b1_bn1_running_var,
        tape,
        training=True,
    )
    var s3b1_bn1_out = s3b1_bn1_res[0].copy()
    model.s3b1_bn1_running_mean = s3b1_bn1_res[1]
    model.s3b1_bn1_running_var = s3b1_bn1_res[2]
    var s3b1_r1 = variable_relu(s3b1_bn1_out, tape)
    var s3b1_c2 = variable_conv2d(
        s3b1_r1, s3b1_conv2_kernel, s3b1_conv2_bias, tape, stride=1, padding=1
    )
    var s3b1_bn2_res = variable_batch_norm(
        s3b1_c2,
        s3b1_bn2_gamma,
        s3b1_bn2_beta,
        model.s3b1_bn2_running_mean,
        model.s3b1_bn2_running_var,
        tape,
        training=True,
    )
    var s3b1_bn2_out = s3b1_bn2_res[0].copy()
    model.s3b1_bn2_running_mean = s3b1_bn2_res[1]
    model.s3b1_bn2_running_var = s3b1_bn2_res[2]
    var s3b1_proj_c = variable_conv2d(
        s2b2_out, s3b1_proj_kernel, s3b1_proj_bias, tape, stride=2, padding=0
    )
    var s3b1_proj_bn_res = variable_batch_norm(
        s3b1_proj_c,
        s3b1_proj_bn_gamma,
        s3b1_proj_bn_beta,
        model.s3b1_proj_bn_running_mean,
        model.s3b1_proj_bn_running_var,
        tape,
        training=True,
    )
    var s3b1_proj_bn_out = s3b1_proj_bn_res[0].copy()
    model.s3b1_proj_bn_running_mean = s3b1_proj_bn_res[1]
    model.s3b1_proj_bn_running_var = s3b1_proj_bn_res[2]
    var s3b1_skip = variable_add(s3b1_bn2_out, s3b1_proj_bn_out, tape)
    var s3b1_out = variable_relu(s3b1_skip, tape)

    # ========== Stage 3, Block 2 (identity shortcut) ==========
    var s3b2_c1 = variable_conv2d(
        s3b1_out, s3b2_conv1_kernel, s3b2_conv1_bias, tape, stride=1, padding=1
    )
    var s3b2_bn1_res = variable_batch_norm(
        s3b2_c1,
        s3b2_bn1_gamma,
        s3b2_bn1_beta,
        model.s3b2_bn1_running_mean,
        model.s3b2_bn1_running_var,
        tape,
        training=True,
    )
    var s3b2_bn1_out = s3b2_bn1_res[0].copy()
    model.s3b2_bn1_running_mean = s3b2_bn1_res[1]
    model.s3b2_bn1_running_var = s3b2_bn1_res[2]
    var s3b2_r1 = variable_relu(s3b2_bn1_out, tape)
    var s3b2_c2 = variable_conv2d(
        s3b2_r1, s3b2_conv2_kernel, s3b2_conv2_bias, tape, stride=1, padding=1
    )
    var s3b2_bn2_res = variable_batch_norm(
        s3b2_c2,
        s3b2_bn2_gamma,
        s3b2_bn2_beta,
        model.s3b2_bn2_running_mean,
        model.s3b2_bn2_running_var,
        tape,
        training=True,
    )
    var s3b2_bn2_out = s3b2_bn2_res[0].copy()
    model.s3b2_bn2_running_mean = s3b2_bn2_res[1]
    model.s3b2_bn2_running_var = s3b2_bn2_res[2]
    var s3b2_skip = variable_add(s3b2_bn2_out, s3b1_out, tape)
    var s3b2_out = variable_relu(s3b2_skip, tape)

    # ========== Stage 4, Block 1 (projection shortcut, stride=2) ==========
    var s4b1_c1 = variable_conv2d(
        s3b2_out, s4b1_conv1_kernel, s4b1_conv1_bias, tape, stride=2, padding=1
    )
    var s4b1_bn1_res = variable_batch_norm(
        s4b1_c1,
        s4b1_bn1_gamma,
        s4b1_bn1_beta,
        model.s4b1_bn1_running_mean,
        model.s4b1_bn1_running_var,
        tape,
        training=True,
    )
    var s4b1_bn1_out = s4b1_bn1_res[0].copy()
    model.s4b1_bn1_running_mean = s4b1_bn1_res[1]
    model.s4b1_bn1_running_var = s4b1_bn1_res[2]
    var s4b1_r1 = variable_relu(s4b1_bn1_out, tape)
    var s4b1_c2 = variable_conv2d(
        s4b1_r1, s4b1_conv2_kernel, s4b1_conv2_bias, tape, stride=1, padding=1
    )
    var s4b1_bn2_res = variable_batch_norm(
        s4b1_c2,
        s4b1_bn2_gamma,
        s4b1_bn2_beta,
        model.s4b1_bn2_running_mean,
        model.s4b1_bn2_running_var,
        tape,
        training=True,
    )
    var s4b1_bn2_out = s4b1_bn2_res[0].copy()
    model.s4b1_bn2_running_mean = s4b1_bn2_res[1]
    model.s4b1_bn2_running_var = s4b1_bn2_res[2]
    var s4b1_proj_c = variable_conv2d(
        s3b2_out, s4b1_proj_kernel, s4b1_proj_bias, tape, stride=2, padding=0
    )
    var s4b1_proj_bn_res = variable_batch_norm(
        s4b1_proj_c,
        s4b1_proj_bn_gamma,
        s4b1_proj_bn_beta,
        model.s4b1_proj_bn_running_mean,
        model.s4b1_proj_bn_running_var,
        tape,
        training=True,
    )
    var s4b1_proj_bn_out = s4b1_proj_bn_res[0].copy()
    model.s4b1_proj_bn_running_mean = s4b1_proj_bn_res[1]
    model.s4b1_proj_bn_running_var = s4b1_proj_bn_res[2]
    var s4b1_skip = variable_add(s4b1_bn2_out, s4b1_proj_bn_out, tape)
    var s4b1_out = variable_relu(s4b1_skip, tape)

    # ========== Stage 4, Block 2 (identity shortcut) ==========
    var s4b2_c1 = variable_conv2d(
        s4b1_out, s4b2_conv1_kernel, s4b2_conv1_bias, tape, stride=1, padding=1
    )
    var s4b2_bn1_res = variable_batch_norm(
        s4b2_c1,
        s4b2_bn1_gamma,
        s4b2_bn1_beta,
        model.s4b2_bn1_running_mean,
        model.s4b2_bn1_running_var,
        tape,
        training=True,
    )
    var s4b2_bn1_out = s4b2_bn1_res[0].copy()
    model.s4b2_bn1_running_mean = s4b2_bn1_res[1]
    model.s4b2_bn1_running_var = s4b2_bn1_res[2]
    var s4b2_r1 = variable_relu(s4b2_bn1_out, tape)
    var s4b2_c2 = variable_conv2d(
        s4b2_r1, s4b2_conv2_kernel, s4b2_conv2_bias, tape, stride=1, padding=1
    )
    var s4b2_bn2_res = variable_batch_norm(
        s4b2_c2,
        s4b2_bn2_gamma,
        s4b2_bn2_beta,
        model.s4b2_bn2_running_mean,
        model.s4b2_bn2_running_var,
        tape,
        training=True,
    )
    var s4b2_bn2_out = s4b2_bn2_res[0].copy()
    model.s4b2_bn2_running_mean = s4b2_bn2_res[1]
    model.s4b2_bn2_running_var = s4b2_bn2_res[2]
    var s4b2_skip = variable_add(s4b2_bn2_out, s4b1_out, tape)
    var s4b2_out = variable_relu(s4b2_skip, tape)

    # ========== Global Average Pooling via two mean reductions ==========
    # s4b2_out is (batch, 512, 4, 4). The manual forward does
    # avgpool2d(kernel_size=4) -> (batch, 512, 1, 1) -> reshape (batch, 512).
    # There is no variable_avgpool2d, so reduce the two spatial axes with
    # variable_mean: axis=3 -> (batch, 512, 4), then axis=2 -> (batch, 512).
    # Averaging both spatial axes IS global average pooling, and the (batch, 512)
    # result is already flattened, so no variable_flatten is needed.
    var gap_w = variable_mean(s4b2_out, tape, axis=3)
    var gap = variable_mean(gap_w, tape, axis=2)

    # ========== FC layer: (batch, 512) → (batch, num_classes) ==========
    var logits = variable_linear(gap, fc_weights, fc_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update ==========
    # Move the 82 trainable Variables into the parameters list in field order;
    # the write-back below uses the same order.
    var parameters: List[Variable] = []
    parameters.append(conv1_kernel^)
    parameters.append(conv1_bias^)
    parameters.append(bn1_gamma^)
    parameters.append(bn1_beta^)
    parameters.append(s1b1_conv1_kernel^)
    parameters.append(s1b1_conv1_bias^)
    parameters.append(s1b1_bn1_gamma^)
    parameters.append(s1b1_bn1_beta^)
    parameters.append(s1b1_conv2_kernel^)
    parameters.append(s1b1_conv2_bias^)
    parameters.append(s1b1_bn2_gamma^)
    parameters.append(s1b1_bn2_beta^)
    parameters.append(s1b2_conv1_kernel^)
    parameters.append(s1b2_conv1_bias^)
    parameters.append(s1b2_bn1_gamma^)
    parameters.append(s1b2_bn1_beta^)
    parameters.append(s1b2_conv2_kernel^)
    parameters.append(s1b2_conv2_bias^)
    parameters.append(s1b2_bn2_gamma^)
    parameters.append(s1b2_bn2_beta^)
    parameters.append(s2b1_conv1_kernel^)
    parameters.append(s2b1_conv1_bias^)
    parameters.append(s2b1_bn1_gamma^)
    parameters.append(s2b1_bn1_beta^)
    parameters.append(s2b1_conv2_kernel^)
    parameters.append(s2b1_conv2_bias^)
    parameters.append(s2b1_bn2_gamma^)
    parameters.append(s2b1_bn2_beta^)
    parameters.append(s2b1_proj_kernel^)
    parameters.append(s2b1_proj_bias^)
    parameters.append(s2b1_proj_bn_gamma^)
    parameters.append(s2b1_proj_bn_beta^)
    parameters.append(s2b2_conv1_kernel^)
    parameters.append(s2b2_conv1_bias^)
    parameters.append(s2b2_bn1_gamma^)
    parameters.append(s2b2_bn1_beta^)
    parameters.append(s2b2_conv2_kernel^)
    parameters.append(s2b2_conv2_bias^)
    parameters.append(s2b2_bn2_gamma^)
    parameters.append(s2b2_bn2_beta^)
    parameters.append(s3b1_conv1_kernel^)
    parameters.append(s3b1_conv1_bias^)
    parameters.append(s3b1_bn1_gamma^)
    parameters.append(s3b1_bn1_beta^)
    parameters.append(s3b1_conv2_kernel^)
    parameters.append(s3b1_conv2_bias^)
    parameters.append(s3b1_bn2_gamma^)
    parameters.append(s3b1_bn2_beta^)
    parameters.append(s3b1_proj_kernel^)
    parameters.append(s3b1_proj_bias^)
    parameters.append(s3b1_proj_bn_gamma^)
    parameters.append(s3b1_proj_bn_beta^)
    parameters.append(s3b2_conv1_kernel^)
    parameters.append(s3b2_conv1_bias^)
    parameters.append(s3b2_bn1_gamma^)
    parameters.append(s3b2_bn1_beta^)
    parameters.append(s3b2_conv2_kernel^)
    parameters.append(s3b2_conv2_bias^)
    parameters.append(s3b2_bn2_gamma^)
    parameters.append(s3b2_bn2_beta^)
    parameters.append(s4b1_conv1_kernel^)
    parameters.append(s4b1_conv1_bias^)
    parameters.append(s4b1_bn1_gamma^)
    parameters.append(s4b1_bn1_beta^)
    parameters.append(s4b1_conv2_kernel^)
    parameters.append(s4b1_conv2_bias^)
    parameters.append(s4b1_bn2_gamma^)
    parameters.append(s4b1_bn2_beta^)
    parameters.append(s4b1_proj_kernel^)
    parameters.append(s4b1_proj_bias^)
    parameters.append(s4b1_proj_bn_gamma^)
    parameters.append(s4b1_proj_bn_beta^)
    parameters.append(s4b2_conv1_kernel^)
    parameters.append(s4b2_conv1_bias^)
    parameters.append(s4b2_bn1_gamma^)
    parameters.append(s4b2_bn1_beta^)
    parameters.append(s4b2_conv2_kernel^)
    parameters.append(s4b2_conv2_bias^)
    parameters.append(s4b2_bn2_gamma^)
    parameters.append(s4b2_bn2_beta^)
    parameters.append(fc_weights^)
    parameters.append(fc_bias^)

    optimizer.step(parameters, tape)

    # Copy updated Variable.data back into the model, same order as appended.
    model.conv1_kernel = parameters[0].data
    model.conv1_bias = parameters[1].data
    model.bn1_gamma = parameters[2].data
    model.bn1_beta = parameters[3].data
    model.s1b1_conv1_kernel = parameters[4].data
    model.s1b1_conv1_bias = parameters[5].data
    model.s1b1_bn1_gamma = parameters[6].data
    model.s1b1_bn1_beta = parameters[7].data
    model.s1b1_conv2_kernel = parameters[8].data
    model.s1b1_conv2_bias = parameters[9].data
    model.s1b1_bn2_gamma = parameters[10].data
    model.s1b1_bn2_beta = parameters[11].data
    model.s1b2_conv1_kernel = parameters[12].data
    model.s1b2_conv1_bias = parameters[13].data
    model.s1b2_bn1_gamma = parameters[14].data
    model.s1b2_bn1_beta = parameters[15].data
    model.s1b2_conv2_kernel = parameters[16].data
    model.s1b2_conv2_bias = parameters[17].data
    model.s1b2_bn2_gamma = parameters[18].data
    model.s1b2_bn2_beta = parameters[19].data
    model.s2b1_conv1_kernel = parameters[20].data
    model.s2b1_conv1_bias = parameters[21].data
    model.s2b1_bn1_gamma = parameters[22].data
    model.s2b1_bn1_beta = parameters[23].data
    model.s2b1_conv2_kernel = parameters[24].data
    model.s2b1_conv2_bias = parameters[25].data
    model.s2b1_bn2_gamma = parameters[26].data
    model.s2b1_bn2_beta = parameters[27].data
    model.s2b1_proj_kernel = parameters[28].data
    model.s2b1_proj_bias = parameters[29].data
    model.s2b1_proj_bn_gamma = parameters[30].data
    model.s2b1_proj_bn_beta = parameters[31].data
    model.s2b2_conv1_kernel = parameters[32].data
    model.s2b2_conv1_bias = parameters[33].data
    model.s2b2_bn1_gamma = parameters[34].data
    model.s2b2_bn1_beta = parameters[35].data
    model.s2b2_conv2_kernel = parameters[36].data
    model.s2b2_conv2_bias = parameters[37].data
    model.s2b2_bn2_gamma = parameters[38].data
    model.s2b2_bn2_beta = parameters[39].data
    model.s3b1_conv1_kernel = parameters[40].data
    model.s3b1_conv1_bias = parameters[41].data
    model.s3b1_bn1_gamma = parameters[42].data
    model.s3b1_bn1_beta = parameters[43].data
    model.s3b1_conv2_kernel = parameters[44].data
    model.s3b1_conv2_bias = parameters[45].data
    model.s3b1_bn2_gamma = parameters[46].data
    model.s3b1_bn2_beta = parameters[47].data
    model.s3b1_proj_kernel = parameters[48].data
    model.s3b1_proj_bias = parameters[49].data
    model.s3b1_proj_bn_gamma = parameters[50].data
    model.s3b1_proj_bn_beta = parameters[51].data
    model.s3b2_conv1_kernel = parameters[52].data
    model.s3b2_conv1_bias = parameters[53].data
    model.s3b2_bn1_gamma = parameters[54].data
    model.s3b2_bn1_beta = parameters[55].data
    model.s3b2_conv2_kernel = parameters[56].data
    model.s3b2_conv2_bias = parameters[57].data
    model.s3b2_bn2_gamma = parameters[58].data
    model.s3b2_bn2_beta = parameters[59].data
    model.s4b1_conv1_kernel = parameters[60].data
    model.s4b1_conv1_bias = parameters[61].data
    model.s4b1_bn1_gamma = parameters[62].data
    model.s4b1_bn1_beta = parameters[63].data
    model.s4b1_conv2_kernel = parameters[64].data
    model.s4b1_conv2_bias = parameters[65].data
    model.s4b1_bn2_gamma = parameters[66].data
    model.s4b1_bn2_beta = parameters[67].data
    model.s4b1_proj_kernel = parameters[68].data
    model.s4b1_proj_bias = parameters[69].data
    model.s4b1_proj_bn_gamma = parameters[70].data
    model.s4b1_proj_bn_beta = parameters[71].data
    model.s4b2_conv1_kernel = parameters[72].data
    model.s4b2_conv1_bias = parameters[73].data
    model.s4b2_bn1_gamma = parameters[74].data
    model.s4b2_bn1_beta = parameters[75].data
    model.s4b2_conv2_kernel = parameters[76].data
    model.s4b2_conv2_bias = parameters[77].data
    model.s4b2_bn2_gamma = parameters[78].data
    model.s4b2_bn2_beta = parameters[79].data
    model.fc_weights = parameters[80].data
    model.fc_bias = parameters[81].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: ResNet18,
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
        model: ResNet-18 model.
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
    print("ResNet-18 Training on CIFAR-10 Dataset (Autograd)")
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
    print("Initializing ResNet-18 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = ResNet18(num_classes=dataset_info.num_classes())
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
        # RAW uint8 class indices [N] — train_epoch one-hot-encodes per batch.
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
