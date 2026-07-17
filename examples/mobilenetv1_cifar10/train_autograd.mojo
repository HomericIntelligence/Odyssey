"""Training Script for MobileNetV1 on CIFAR-10 using Tape-Based Autograd.

Autograd variant of `train.mojo` (sub-task 6 of #5454). Replaces the manual
forward + `depthwise_conv2d`/`conv2d`/`batch_norm2d`/`global_avgpool2d` +
manual `*_backward` chain with the Variable / GradientTape substrate: the
MobileNetV1 forward is recorded via `variable_*` ops, gradients come from
`loss.backward(tape)`, and parameters update through
`optimizer.step(params, tape)`.

Depthwise-separable wiring: each of the 13 blocks is a depthwise 3x3 conv
(`variable_depthwise_conv2d`, weights `(C, 1, 3, 3)` — one filter per channel)
-> BN -> ReLU, then a pointwise 1x1 conv (`variable_conv2d`, weights
`(out, in, 1, 1)`) -> BN -> ReLU. The stem is a standard `variable_conv2d`
(32 filters, 3x3, stride=2, pad=1) -> BN -> ReLU. A small `ds_block` helper runs
one block through the autograd ops given its 8 trainable Variables plus its 4
running-stat buffers, returning the output Variable and the 4 updated running
stats — mirroring the manual `DepthwiseSeparableBlock.forward()`.

Default optimizer is AdamW (matches examples/resnet18_cifar10/); pass
`--optimizer sgd` to select SGD. The manual `train.mojo` is intentionally left
in place — the tracker-level cleanup (#5454) removes the manual path once all
ports land.

Batch normalization IS ported: `variable_batch_norm` returns
`(output_var, new_running_mean, new_running_var)`. `gamma`/`beta` are trainable
Variables; `running_mean`/`running_var` are plain `AnyTensor` buffers threaded
back into the model's fields after every layer (exactly like the manual
`forward()`), so the running statistics keep advancing. `training=True` is used
here (a training port; no eval loop). The BN output Variable is bound with
`.copy()` because it is not implicitly copyable under `--Werror`.

GLOBAL AVERAGE POOLING substitution: the manual forward ends with
`global_avgpool2d(out)` over the (batch, 1024, H, W) block-13 output, then a
reshape to (batch, 1024) before the FC. The autograd substrate has NO
`variable_global_avgpool`, so GAP is expressed as two `variable_mean` reductions
over the two spatial axes: `variable_mean(x, tape, axis=3)` -> (batch, 1024, H'),
then `variable_mean(..., tape, axis=2)` -> (batch, 1024). Averaging both spatial
axes is exactly global average pooling, and the result is already the flattened
(batch, 1024) shape, so NO `variable_flatten` is needed before the FC. (NOTE:
`variable_mean`'s `axis=-1` default is a FULL reduction to a scalar, not a
last-axis reduction — the explicit positive axes 3 then 2 are required.)

Omitted vs the manual path: MobileNet has no dropout (nothing dropped there),
and the manual path has no LR schedule or SGD momentum to replicate (the
autograd SGD is constructed as SGD(learning_rate=...)); these are training-tuning
details, not required for the mechanism.

Usage:
    mojo run examples/mobilenetv1_cifar10/train_autograd.mojo --epochs 1 --batch-size 32
    mojo run examples/mobilenetv1_cifar10/train_autograd.mojo --optimizer sgd

Requirements:
    - CIFAR-10 dataset downloaded (run: python scripts/download_cifar10.py)
    - Dataset location: datasets/cifar10/
"""

from examples.mobilenetv1_cifar10.model import MobileNetV1
from odyssey.data.datasets import CIFAR10Dataset
from odyssey.data.formats import one_hot_encode
from odyssey.data.constants import DatasetInfo
from odyssey.autograd import (
    Variable,
    GradientTape,
    AdamW,
    SGD,
    variable_conv2d,
    variable_depthwise_conv2d,
    variable_relu,
    variable_linear,
    variable_cross_entropy,
    variable_mean,
    variable_batch_norm,
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

    AdamW is the default (DoD of #5562); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd" and
    `parse()` pre-populates it, so we cannot rely on `get_string`'s default —
    the caller gates on `was_user_supplied` before reading the value.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "mobilenetv1_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var epochs = args.resolve_int("epochs", 100)
    var batch_size = args.resolve_int("batch-size", 128)
    var weight_decay = args.resolve_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates it into the
    # value map — so `get_string("optimizer", "adamw")` would return "sgd" even
    # when unset (the fallback is dead). Gate on `was_user_supplied`: only an
    # explicit `--optimizer` is honored; unset defaults to AdamW (DoD of #5562).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    # Optimizer-appropriate default learning rate. MobileNetV1 is deep (1 stem
    # conv + 13 depthwise-separable blocks + 1 FC); AdamW's adaptive step
    # diverges on this stack at the SGD-scale lr, so default AdamW to 1e-4 and
    # SGD to 1e-2 when the user did not pass --lr. An explicit --lr always wins.
    var learning_rate: Float64
    if args.was_user_supplied("lr"):
        learning_rate = args.get_float("lr", 0.0001)
    elif optimizer == "sgd":
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    var data_dir = args.resolve_string("data-dir", "datasets/cifar10")
    var weights_dir = args.get_string(
        "weights-dir", "mobilenetv1_weights_autograd"
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


def ds_block(
    x: Variable,
    dw_weights: Variable,
    dw_bias: Variable,
    dw_bn_gamma: Variable,
    dw_bn_beta: Variable,
    dw_bn_running_mean: AnyTensor,
    dw_bn_running_var: AnyTensor,
    pw_weights: Variable,
    pw_bias: Variable,
    pw_bn_gamma: Variable,
    pw_bn_beta: Variable,
    pw_bn_running_mean: AnyTensor,
    pw_bn_running_var: AnyTensor,
    stride: Int,
    mut tape: GradientTape,
) raises -> Tuple[Variable, AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Run one depthwise-separable block through the autograd ops.

    Mirrors `DepthwiseSeparableBlock.forward()`:
        depthwise 3x3 conv (stride, pad=1) -> BN -> ReLU
        -> pointwise 1x1 conv (stride=1, pad=0) -> BN -> ReLU.

    The 4 BN running-stat buffers are plain `AnyTensor` (non-trainable); they are
    threaded in and the EMA-updated versions returned so the caller can write
    them back onto the block. The 8 trainable Variables (dw/pw weights, biases
    and BN gamma/beta) are wrapped by the caller in field order.

    Returns:
        (output_var, dw_running_mean, dw_running_var, pw_running_mean,
        pw_running_var).
    """
    # Depthwise conv (per-channel spatial filtering) -> BN -> ReLU.
    var dw_out = variable_depthwise_conv2d(
        x, dw_weights, dw_bias, tape, stride=stride, padding=1
    )
    var dw_bn_res = variable_batch_norm(
        dw_out,
        dw_bn_gamma,
        dw_bn_beta,
        dw_bn_running_mean,
        dw_bn_running_var,
        tape,
        training=True,
    )
    var dw_bn_out = dw_bn_res[0].copy()
    var dw_relu = variable_relu(dw_bn_out, tape)

    # Pointwise conv (1x1 channel mixing) -> BN -> ReLU.
    var pw_out = variable_conv2d(
        dw_relu, pw_weights, pw_bias, tape, stride=1, padding=0
    )
    var pw_bn_res = variable_batch_norm(
        pw_out,
        pw_bn_gamma,
        pw_bn_beta,
        pw_bn_running_mean,
        pw_bn_running_var,
        tape,
        training=True,
    )
    var pw_bn_out = pw_bn_res[0].copy()
    var out = variable_relu(pw_bn_out, tape)

    return (
        out^,
        dw_bn_res[1],
        dw_bn_res[2],
        pw_bn_res[1],
        pw_bn_res[2],
    )


def train_batch[
    O: Optimizer
](
    mut model: MobileNetV1,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the MobileNetV1 forward pass with `variable_*` ops (depthwise +
    pointwise convs, batch norm, GAP via two `variable_mean` reductions), calls
    `loss.backward(tape)` for automatic gradients, then `optimizer.step(...)`.

    Args:
        model: MobileNetV1 model (110 trainable tensors + BN running buffers).
        input: Batch of images (batch, 3, 32, 32).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    # Wrap the trainable parameters as Variables (field order matches the
    # parameters list + write-back below). BN running_mean/var stay plain
    # AnyTensor buffers (non-trainable) and are threaded through/back separately.
    # --- Stem conv + BN ---
    var initial_conv_weights = Variable(model.initial_conv_weights, True, tape)
    var initial_conv_bias = Variable(model.initial_conv_bias, True, tape)
    var initial_bn_gamma = Variable(model.initial_bn_gamma, True, tape)
    var initial_bn_beta = Variable(model.initial_bn_beta, True, tape)
    # --- 13 depthwise-separable blocks (8 trainable Variables each) ---
    var b1_dw_w = Variable(model.ds_block_1.dw_weights, True, tape)
    var b1_dw_b = Variable(model.ds_block_1.dw_bias, True, tape)
    var b1_dw_g = Variable(model.ds_block_1.dw_bn_gamma, True, tape)
    var b1_dw_be = Variable(model.ds_block_1.dw_bn_beta, True, tape)
    var b1_pw_w = Variable(model.ds_block_1.pw_weights, True, tape)
    var b1_pw_b = Variable(model.ds_block_1.pw_bias, True, tape)
    var b1_pw_g = Variable(model.ds_block_1.pw_bn_gamma, True, tape)
    var b1_pw_be = Variable(model.ds_block_1.pw_bn_beta, True, tape)
    var b2_dw_w = Variable(model.ds_block_2.dw_weights, True, tape)
    var b2_dw_b = Variable(model.ds_block_2.dw_bias, True, tape)
    var b2_dw_g = Variable(model.ds_block_2.dw_bn_gamma, True, tape)
    var b2_dw_be = Variable(model.ds_block_2.dw_bn_beta, True, tape)
    var b2_pw_w = Variable(model.ds_block_2.pw_weights, True, tape)
    var b2_pw_b = Variable(model.ds_block_2.pw_bias, True, tape)
    var b2_pw_g = Variable(model.ds_block_2.pw_bn_gamma, True, tape)
    var b2_pw_be = Variable(model.ds_block_2.pw_bn_beta, True, tape)
    var b3_dw_w = Variable(model.ds_block_3.dw_weights, True, tape)
    var b3_dw_b = Variable(model.ds_block_3.dw_bias, True, tape)
    var b3_dw_g = Variable(model.ds_block_3.dw_bn_gamma, True, tape)
    var b3_dw_be = Variable(model.ds_block_3.dw_bn_beta, True, tape)
    var b3_pw_w = Variable(model.ds_block_3.pw_weights, True, tape)
    var b3_pw_b = Variable(model.ds_block_3.pw_bias, True, tape)
    var b3_pw_g = Variable(model.ds_block_3.pw_bn_gamma, True, tape)
    var b3_pw_be = Variable(model.ds_block_3.pw_bn_beta, True, tape)
    var b4_dw_w = Variable(model.ds_block_4.dw_weights, True, tape)
    var b4_dw_b = Variable(model.ds_block_4.dw_bias, True, tape)
    var b4_dw_g = Variable(model.ds_block_4.dw_bn_gamma, True, tape)
    var b4_dw_be = Variable(model.ds_block_4.dw_bn_beta, True, tape)
    var b4_pw_w = Variable(model.ds_block_4.pw_weights, True, tape)
    var b4_pw_b = Variable(model.ds_block_4.pw_bias, True, tape)
    var b4_pw_g = Variable(model.ds_block_4.pw_bn_gamma, True, tape)
    var b4_pw_be = Variable(model.ds_block_4.pw_bn_beta, True, tape)
    var b5_dw_w = Variable(model.ds_block_5.dw_weights, True, tape)
    var b5_dw_b = Variable(model.ds_block_5.dw_bias, True, tape)
    var b5_dw_g = Variable(model.ds_block_5.dw_bn_gamma, True, tape)
    var b5_dw_be = Variable(model.ds_block_5.dw_bn_beta, True, tape)
    var b5_pw_w = Variable(model.ds_block_5.pw_weights, True, tape)
    var b5_pw_b = Variable(model.ds_block_5.pw_bias, True, tape)
    var b5_pw_g = Variable(model.ds_block_5.pw_bn_gamma, True, tape)
    var b5_pw_be = Variable(model.ds_block_5.pw_bn_beta, True, tape)
    var b6_dw_w = Variable(model.ds_block_6.dw_weights, True, tape)
    var b6_dw_b = Variable(model.ds_block_6.dw_bias, True, tape)
    var b6_dw_g = Variable(model.ds_block_6.dw_bn_gamma, True, tape)
    var b6_dw_be = Variable(model.ds_block_6.dw_bn_beta, True, tape)
    var b6_pw_w = Variable(model.ds_block_6.pw_weights, True, tape)
    var b6_pw_b = Variable(model.ds_block_6.pw_bias, True, tape)
    var b6_pw_g = Variable(model.ds_block_6.pw_bn_gamma, True, tape)
    var b6_pw_be = Variable(model.ds_block_6.pw_bn_beta, True, tape)
    var b7_dw_w = Variable(model.ds_block_7.dw_weights, True, tape)
    var b7_dw_b = Variable(model.ds_block_7.dw_bias, True, tape)
    var b7_dw_g = Variable(model.ds_block_7.dw_bn_gamma, True, tape)
    var b7_dw_be = Variable(model.ds_block_7.dw_bn_beta, True, tape)
    var b7_pw_w = Variable(model.ds_block_7.pw_weights, True, tape)
    var b7_pw_b = Variable(model.ds_block_7.pw_bias, True, tape)
    var b7_pw_g = Variable(model.ds_block_7.pw_bn_gamma, True, tape)
    var b7_pw_be = Variable(model.ds_block_7.pw_bn_beta, True, tape)
    var b8_dw_w = Variable(model.ds_block_8.dw_weights, True, tape)
    var b8_dw_b = Variable(model.ds_block_8.dw_bias, True, tape)
    var b8_dw_g = Variable(model.ds_block_8.dw_bn_gamma, True, tape)
    var b8_dw_be = Variable(model.ds_block_8.dw_bn_beta, True, tape)
    var b8_pw_w = Variable(model.ds_block_8.pw_weights, True, tape)
    var b8_pw_b = Variable(model.ds_block_8.pw_bias, True, tape)
    var b8_pw_g = Variable(model.ds_block_8.pw_bn_gamma, True, tape)
    var b8_pw_be = Variable(model.ds_block_8.pw_bn_beta, True, tape)
    var b9_dw_w = Variable(model.ds_block_9.dw_weights, True, tape)
    var b9_dw_b = Variable(model.ds_block_9.dw_bias, True, tape)
    var b9_dw_g = Variable(model.ds_block_9.dw_bn_gamma, True, tape)
    var b9_dw_be = Variable(model.ds_block_9.dw_bn_beta, True, tape)
    var b9_pw_w = Variable(model.ds_block_9.pw_weights, True, tape)
    var b9_pw_b = Variable(model.ds_block_9.pw_bias, True, tape)
    var b9_pw_g = Variable(model.ds_block_9.pw_bn_gamma, True, tape)
    var b9_pw_be = Variable(model.ds_block_9.pw_bn_beta, True, tape)
    var b10_dw_w = Variable(model.ds_block_10.dw_weights, True, tape)
    var b10_dw_b = Variable(model.ds_block_10.dw_bias, True, tape)
    var b10_dw_g = Variable(model.ds_block_10.dw_bn_gamma, True, tape)
    var b10_dw_be = Variable(model.ds_block_10.dw_bn_beta, True, tape)
    var b10_pw_w = Variable(model.ds_block_10.pw_weights, True, tape)
    var b10_pw_b = Variable(model.ds_block_10.pw_bias, True, tape)
    var b10_pw_g = Variable(model.ds_block_10.pw_bn_gamma, True, tape)
    var b10_pw_be = Variable(model.ds_block_10.pw_bn_beta, True, tape)
    var b11_dw_w = Variable(model.ds_block_11.dw_weights, True, tape)
    var b11_dw_b = Variable(model.ds_block_11.dw_bias, True, tape)
    var b11_dw_g = Variable(model.ds_block_11.dw_bn_gamma, True, tape)
    var b11_dw_be = Variable(model.ds_block_11.dw_bn_beta, True, tape)
    var b11_pw_w = Variable(model.ds_block_11.pw_weights, True, tape)
    var b11_pw_b = Variable(model.ds_block_11.pw_bias, True, tape)
    var b11_pw_g = Variable(model.ds_block_11.pw_bn_gamma, True, tape)
    var b11_pw_be = Variable(model.ds_block_11.pw_bn_beta, True, tape)
    var b12_dw_w = Variable(model.ds_block_12.dw_weights, True, tape)
    var b12_dw_b = Variable(model.ds_block_12.dw_bias, True, tape)
    var b12_dw_g = Variable(model.ds_block_12.dw_bn_gamma, True, tape)
    var b12_dw_be = Variable(model.ds_block_12.dw_bn_beta, True, tape)
    var b12_pw_w = Variable(model.ds_block_12.pw_weights, True, tape)
    var b12_pw_b = Variable(model.ds_block_12.pw_bias, True, tape)
    var b12_pw_g = Variable(model.ds_block_12.pw_bn_gamma, True, tape)
    var b12_pw_be = Variable(model.ds_block_12.pw_bn_beta, True, tape)
    var b13_dw_w = Variable(model.ds_block_13.dw_weights, True, tape)
    var b13_dw_b = Variable(model.ds_block_13.dw_bias, True, tape)
    var b13_dw_g = Variable(model.ds_block_13.dw_bn_gamma, True, tape)
    var b13_dw_be = Variable(model.ds_block_13.dw_bn_beta, True, tape)
    var b13_pw_w = Variable(model.ds_block_13.pw_weights, True, tape)
    var b13_pw_b = Variable(model.ds_block_13.pw_bias, True, tape)
    var b13_pw_g = Variable(model.ds_block_13.pw_bn_gamma, True, tape)
    var b13_pw_be = Variable(model.ds_block_13.pw_bn_beta, True, tape)
    # --- FC head ---
    var fc_weights = Variable(model.fc_weights, True, tape)
    var fc_bias = Variable(model.fc_bias, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Forward Pass (recorded to tape) ==========
    # Stem: standard conv (32 filters, 3x3, stride=2, pad=1) + BN + ReLU. BN
    # returns updated running stats which are threaded back into the model
    # immediately (as the manual forward() does), so later batches keep
    # advancing the statistics. Shape after stem: (batch, 32, 16, 16).
    var stem_conv = variable_conv2d(
        input_var,
        initial_conv_weights,
        initial_conv_bias,
        tape,
        stride=2,
        padding=1,
    )
    var stem_bn_res = variable_batch_norm(
        stem_conv,
        initial_bn_gamma,
        initial_bn_beta,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        tape,
        training=True,
    )
    var stem_bn_out = stem_bn_res[0].copy()
    model.initial_bn_running_mean = stem_bn_res[1]
    model.initial_bn_running_var = stem_bn_res[2]
    var stem_out = variable_relu(stem_bn_out, tape)

    # ========== 13 depthwise-separable blocks ==========
    # Block strides: b1 s1, b2 s2, b3 s1, b4 s2, b5 s1, b6 s2, b7-b11 s1,
    # b12 s2, b13 s1. Running stats are threaded back onto each block.
    var r1 = ds_block(
        stem_out,
        b1_dw_w,
        b1_dw_b,
        b1_dw_g,
        b1_dw_be,
        model.ds_block_1.dw_bn_running_mean,
        model.ds_block_1.dw_bn_running_var,
        b1_pw_w,
        b1_pw_b,
        b1_pw_g,
        b1_pw_be,
        model.ds_block_1.pw_bn_running_mean,
        model.ds_block_1.pw_bn_running_var,
        1,
        tape,
    )
    var out1 = r1[0].copy()
    model.ds_block_1.dw_bn_running_mean = r1[1]
    model.ds_block_1.dw_bn_running_var = r1[2]
    model.ds_block_1.pw_bn_running_mean = r1[3]
    model.ds_block_1.pw_bn_running_var = r1[4]

    var r2 = ds_block(
        out1,
        b2_dw_w,
        b2_dw_b,
        b2_dw_g,
        b2_dw_be,
        model.ds_block_2.dw_bn_running_mean,
        model.ds_block_2.dw_bn_running_var,
        b2_pw_w,
        b2_pw_b,
        b2_pw_g,
        b2_pw_be,
        model.ds_block_2.pw_bn_running_mean,
        model.ds_block_2.pw_bn_running_var,
        2,
        tape,
    )
    var out2 = r2[0].copy()
    model.ds_block_2.dw_bn_running_mean = r2[1]
    model.ds_block_2.dw_bn_running_var = r2[2]
    model.ds_block_2.pw_bn_running_mean = r2[3]
    model.ds_block_2.pw_bn_running_var = r2[4]

    var r3 = ds_block(
        out2,
        b3_dw_w,
        b3_dw_b,
        b3_dw_g,
        b3_dw_be,
        model.ds_block_3.dw_bn_running_mean,
        model.ds_block_3.dw_bn_running_var,
        b3_pw_w,
        b3_pw_b,
        b3_pw_g,
        b3_pw_be,
        model.ds_block_3.pw_bn_running_mean,
        model.ds_block_3.pw_bn_running_var,
        1,
        tape,
    )
    var out3 = r3[0].copy()
    model.ds_block_3.dw_bn_running_mean = r3[1]
    model.ds_block_3.dw_bn_running_var = r3[2]
    model.ds_block_3.pw_bn_running_mean = r3[3]
    model.ds_block_3.pw_bn_running_var = r3[4]

    var r4 = ds_block(
        out3,
        b4_dw_w,
        b4_dw_b,
        b4_dw_g,
        b4_dw_be,
        model.ds_block_4.dw_bn_running_mean,
        model.ds_block_4.dw_bn_running_var,
        b4_pw_w,
        b4_pw_b,
        b4_pw_g,
        b4_pw_be,
        model.ds_block_4.pw_bn_running_mean,
        model.ds_block_4.pw_bn_running_var,
        2,
        tape,
    )
    var out4 = r4[0].copy()
    model.ds_block_4.dw_bn_running_mean = r4[1]
    model.ds_block_4.dw_bn_running_var = r4[2]
    model.ds_block_4.pw_bn_running_mean = r4[3]
    model.ds_block_4.pw_bn_running_var = r4[4]

    var r5 = ds_block(
        out4,
        b5_dw_w,
        b5_dw_b,
        b5_dw_g,
        b5_dw_be,
        model.ds_block_5.dw_bn_running_mean,
        model.ds_block_5.dw_bn_running_var,
        b5_pw_w,
        b5_pw_b,
        b5_pw_g,
        b5_pw_be,
        model.ds_block_5.pw_bn_running_mean,
        model.ds_block_5.pw_bn_running_var,
        1,
        tape,
    )
    var out5 = r5[0].copy()
    model.ds_block_5.dw_bn_running_mean = r5[1]
    model.ds_block_5.dw_bn_running_var = r5[2]
    model.ds_block_5.pw_bn_running_mean = r5[3]
    model.ds_block_5.pw_bn_running_var = r5[4]

    var r6 = ds_block(
        out5,
        b6_dw_w,
        b6_dw_b,
        b6_dw_g,
        b6_dw_be,
        model.ds_block_6.dw_bn_running_mean,
        model.ds_block_6.dw_bn_running_var,
        b6_pw_w,
        b6_pw_b,
        b6_pw_g,
        b6_pw_be,
        model.ds_block_6.pw_bn_running_mean,
        model.ds_block_6.pw_bn_running_var,
        2,
        tape,
    )
    var out6 = r6[0].copy()
    model.ds_block_6.dw_bn_running_mean = r6[1]
    model.ds_block_6.dw_bn_running_var = r6[2]
    model.ds_block_6.pw_bn_running_mean = r6[3]
    model.ds_block_6.pw_bn_running_var = r6[4]

    var r7 = ds_block(
        out6,
        b7_dw_w,
        b7_dw_b,
        b7_dw_g,
        b7_dw_be,
        model.ds_block_7.dw_bn_running_mean,
        model.ds_block_7.dw_bn_running_var,
        b7_pw_w,
        b7_pw_b,
        b7_pw_g,
        b7_pw_be,
        model.ds_block_7.pw_bn_running_mean,
        model.ds_block_7.pw_bn_running_var,
        1,
        tape,
    )
    var out7 = r7[0].copy()
    model.ds_block_7.dw_bn_running_mean = r7[1]
    model.ds_block_7.dw_bn_running_var = r7[2]
    model.ds_block_7.pw_bn_running_mean = r7[3]
    model.ds_block_7.pw_bn_running_var = r7[4]

    var r8 = ds_block(
        out7,
        b8_dw_w,
        b8_dw_b,
        b8_dw_g,
        b8_dw_be,
        model.ds_block_8.dw_bn_running_mean,
        model.ds_block_8.dw_bn_running_var,
        b8_pw_w,
        b8_pw_b,
        b8_pw_g,
        b8_pw_be,
        model.ds_block_8.pw_bn_running_mean,
        model.ds_block_8.pw_bn_running_var,
        1,
        tape,
    )
    var out8 = r8[0].copy()
    model.ds_block_8.dw_bn_running_mean = r8[1]
    model.ds_block_8.dw_bn_running_var = r8[2]
    model.ds_block_8.pw_bn_running_mean = r8[3]
    model.ds_block_8.pw_bn_running_var = r8[4]

    var r9 = ds_block(
        out8,
        b9_dw_w,
        b9_dw_b,
        b9_dw_g,
        b9_dw_be,
        model.ds_block_9.dw_bn_running_mean,
        model.ds_block_9.dw_bn_running_var,
        b9_pw_w,
        b9_pw_b,
        b9_pw_g,
        b9_pw_be,
        model.ds_block_9.pw_bn_running_mean,
        model.ds_block_9.pw_bn_running_var,
        1,
        tape,
    )
    var out9 = r9[0].copy()
    model.ds_block_9.dw_bn_running_mean = r9[1]
    model.ds_block_9.dw_bn_running_var = r9[2]
    model.ds_block_9.pw_bn_running_mean = r9[3]
    model.ds_block_9.pw_bn_running_var = r9[4]

    var r10 = ds_block(
        out9,
        b10_dw_w,
        b10_dw_b,
        b10_dw_g,
        b10_dw_be,
        model.ds_block_10.dw_bn_running_mean,
        model.ds_block_10.dw_bn_running_var,
        b10_pw_w,
        b10_pw_b,
        b10_pw_g,
        b10_pw_be,
        model.ds_block_10.pw_bn_running_mean,
        model.ds_block_10.pw_bn_running_var,
        1,
        tape,
    )
    var out10 = r10[0].copy()
    model.ds_block_10.dw_bn_running_mean = r10[1]
    model.ds_block_10.dw_bn_running_var = r10[2]
    model.ds_block_10.pw_bn_running_mean = r10[3]
    model.ds_block_10.pw_bn_running_var = r10[4]

    var r11 = ds_block(
        out10,
        b11_dw_w,
        b11_dw_b,
        b11_dw_g,
        b11_dw_be,
        model.ds_block_11.dw_bn_running_mean,
        model.ds_block_11.dw_bn_running_var,
        b11_pw_w,
        b11_pw_b,
        b11_pw_g,
        b11_pw_be,
        model.ds_block_11.pw_bn_running_mean,
        model.ds_block_11.pw_bn_running_var,
        1,
        tape,
    )
    var out11 = r11[0].copy()
    model.ds_block_11.dw_bn_running_mean = r11[1]
    model.ds_block_11.dw_bn_running_var = r11[2]
    model.ds_block_11.pw_bn_running_mean = r11[3]
    model.ds_block_11.pw_bn_running_var = r11[4]

    var r12 = ds_block(
        out11,
        b12_dw_w,
        b12_dw_b,
        b12_dw_g,
        b12_dw_be,
        model.ds_block_12.dw_bn_running_mean,
        model.ds_block_12.dw_bn_running_var,
        b12_pw_w,
        b12_pw_b,
        b12_pw_g,
        b12_pw_be,
        model.ds_block_12.pw_bn_running_mean,
        model.ds_block_12.pw_bn_running_var,
        2,
        tape,
    )
    var out12 = r12[0].copy()
    model.ds_block_12.dw_bn_running_mean = r12[1]
    model.ds_block_12.dw_bn_running_var = r12[2]
    model.ds_block_12.pw_bn_running_mean = r12[3]
    model.ds_block_12.pw_bn_running_var = r12[4]

    var r13 = ds_block(
        out12,
        b13_dw_w,
        b13_dw_b,
        b13_dw_g,
        b13_dw_be,
        model.ds_block_13.dw_bn_running_mean,
        model.ds_block_13.dw_bn_running_var,
        b13_pw_w,
        b13_pw_b,
        b13_pw_g,
        b13_pw_be,
        model.ds_block_13.pw_bn_running_mean,
        model.ds_block_13.pw_bn_running_var,
        1,
        tape,
    )
    var out13 = r13[0].copy()
    model.ds_block_13.dw_bn_running_mean = r13[1]
    model.ds_block_13.dw_bn_running_var = r13[2]
    model.ds_block_13.pw_bn_running_mean = r13[3]
    model.ds_block_13.pw_bn_running_var = r13[4]

    # ========== Global Average Pooling via two mean reductions ==========
    # out13 is (batch, 1024, H, W). The manual forward does global_avgpool2d ->
    # (batch, 1024, 1, 1) -> reshape (batch, 1024). There is no
    # variable_global_avgpool, so reduce the two spatial axes with variable_mean:
    # axis=3 -> (batch, 1024, H), then axis=2 -> (batch, 1024). Averaging both
    # spatial axes IS global average pooling, and the (batch, 1024) result is
    # already flattened, so no variable_flatten is needed.
    var gap_w = variable_mean(out13, tape, axis=3)
    var gap = variable_mean(gap_w, tape, axis=2)

    # ========== FC layer: (batch, 1024) → (batch, num_classes) ==========
    var logits = variable_linear(gap, fc_weights, fc_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # ========== Parameter Update ==========
    # Move the trainable Variables into the parameters list in field order; the
    # write-back below uses the same order.
    var parameters: List[Variable] = []
    parameters.append(initial_conv_weights^)
    parameters.append(initial_conv_bias^)
    parameters.append(initial_bn_gamma^)
    parameters.append(initial_bn_beta^)
    parameters.append(b1_dw_w^)
    parameters.append(b1_dw_b^)
    parameters.append(b1_dw_g^)
    parameters.append(b1_dw_be^)
    parameters.append(b1_pw_w^)
    parameters.append(b1_pw_b^)
    parameters.append(b1_pw_g^)
    parameters.append(b1_pw_be^)
    parameters.append(b2_dw_w^)
    parameters.append(b2_dw_b^)
    parameters.append(b2_dw_g^)
    parameters.append(b2_dw_be^)
    parameters.append(b2_pw_w^)
    parameters.append(b2_pw_b^)
    parameters.append(b2_pw_g^)
    parameters.append(b2_pw_be^)
    parameters.append(b3_dw_w^)
    parameters.append(b3_dw_b^)
    parameters.append(b3_dw_g^)
    parameters.append(b3_dw_be^)
    parameters.append(b3_pw_w^)
    parameters.append(b3_pw_b^)
    parameters.append(b3_pw_g^)
    parameters.append(b3_pw_be^)
    parameters.append(b4_dw_w^)
    parameters.append(b4_dw_b^)
    parameters.append(b4_dw_g^)
    parameters.append(b4_dw_be^)
    parameters.append(b4_pw_w^)
    parameters.append(b4_pw_b^)
    parameters.append(b4_pw_g^)
    parameters.append(b4_pw_be^)
    parameters.append(b5_dw_w^)
    parameters.append(b5_dw_b^)
    parameters.append(b5_dw_g^)
    parameters.append(b5_dw_be^)
    parameters.append(b5_pw_w^)
    parameters.append(b5_pw_b^)
    parameters.append(b5_pw_g^)
    parameters.append(b5_pw_be^)
    parameters.append(b6_dw_w^)
    parameters.append(b6_dw_b^)
    parameters.append(b6_dw_g^)
    parameters.append(b6_dw_be^)
    parameters.append(b6_pw_w^)
    parameters.append(b6_pw_b^)
    parameters.append(b6_pw_g^)
    parameters.append(b6_pw_be^)
    parameters.append(b7_dw_w^)
    parameters.append(b7_dw_b^)
    parameters.append(b7_dw_g^)
    parameters.append(b7_dw_be^)
    parameters.append(b7_pw_w^)
    parameters.append(b7_pw_b^)
    parameters.append(b7_pw_g^)
    parameters.append(b7_pw_be^)
    parameters.append(b8_dw_w^)
    parameters.append(b8_dw_b^)
    parameters.append(b8_dw_g^)
    parameters.append(b8_dw_be^)
    parameters.append(b8_pw_w^)
    parameters.append(b8_pw_b^)
    parameters.append(b8_pw_g^)
    parameters.append(b8_pw_be^)
    parameters.append(b9_dw_w^)
    parameters.append(b9_dw_b^)
    parameters.append(b9_dw_g^)
    parameters.append(b9_dw_be^)
    parameters.append(b9_pw_w^)
    parameters.append(b9_pw_b^)
    parameters.append(b9_pw_g^)
    parameters.append(b9_pw_be^)
    parameters.append(b10_dw_w^)
    parameters.append(b10_dw_b^)
    parameters.append(b10_dw_g^)
    parameters.append(b10_dw_be^)
    parameters.append(b10_pw_w^)
    parameters.append(b10_pw_b^)
    parameters.append(b10_pw_g^)
    parameters.append(b10_pw_be^)
    parameters.append(b11_dw_w^)
    parameters.append(b11_dw_b^)
    parameters.append(b11_dw_g^)
    parameters.append(b11_dw_be^)
    parameters.append(b11_pw_w^)
    parameters.append(b11_pw_b^)
    parameters.append(b11_pw_g^)
    parameters.append(b11_pw_be^)
    parameters.append(b12_dw_w^)
    parameters.append(b12_dw_b^)
    parameters.append(b12_dw_g^)
    parameters.append(b12_dw_be^)
    parameters.append(b12_pw_w^)
    parameters.append(b12_pw_b^)
    parameters.append(b12_pw_g^)
    parameters.append(b12_pw_be^)
    parameters.append(b13_dw_w^)
    parameters.append(b13_dw_b^)
    parameters.append(b13_dw_g^)
    parameters.append(b13_dw_be^)
    parameters.append(b13_pw_w^)
    parameters.append(b13_pw_b^)
    parameters.append(b13_pw_g^)
    parameters.append(b13_pw_be^)
    parameters.append(fc_weights^)
    parameters.append(fc_bias^)

    optimizer.step(parameters, tape)

    # Copy updated Variable.data back into the model, same order as appended.
    model.initial_conv_weights = parameters[0].data
    model.initial_conv_bias = parameters[1].data
    model.initial_bn_gamma = parameters[2].data
    model.initial_bn_beta = parameters[3].data
    model.ds_block_1.dw_weights = parameters[4].data
    model.ds_block_1.dw_bias = parameters[5].data
    model.ds_block_1.dw_bn_gamma = parameters[6].data
    model.ds_block_1.dw_bn_beta = parameters[7].data
    model.ds_block_1.pw_weights = parameters[8].data
    model.ds_block_1.pw_bias = parameters[9].data
    model.ds_block_1.pw_bn_gamma = parameters[10].data
    model.ds_block_1.pw_bn_beta = parameters[11].data
    model.ds_block_2.dw_weights = parameters[12].data
    model.ds_block_2.dw_bias = parameters[13].data
    model.ds_block_2.dw_bn_gamma = parameters[14].data
    model.ds_block_2.dw_bn_beta = parameters[15].data
    model.ds_block_2.pw_weights = parameters[16].data
    model.ds_block_2.pw_bias = parameters[17].data
    model.ds_block_2.pw_bn_gamma = parameters[18].data
    model.ds_block_2.pw_bn_beta = parameters[19].data
    model.ds_block_3.dw_weights = parameters[20].data
    model.ds_block_3.dw_bias = parameters[21].data
    model.ds_block_3.dw_bn_gamma = parameters[22].data
    model.ds_block_3.dw_bn_beta = parameters[23].data
    model.ds_block_3.pw_weights = parameters[24].data
    model.ds_block_3.pw_bias = parameters[25].data
    model.ds_block_3.pw_bn_gamma = parameters[26].data
    model.ds_block_3.pw_bn_beta = parameters[27].data
    model.ds_block_4.dw_weights = parameters[28].data
    model.ds_block_4.dw_bias = parameters[29].data
    model.ds_block_4.dw_bn_gamma = parameters[30].data
    model.ds_block_4.dw_bn_beta = parameters[31].data
    model.ds_block_4.pw_weights = parameters[32].data
    model.ds_block_4.pw_bias = parameters[33].data
    model.ds_block_4.pw_bn_gamma = parameters[34].data
    model.ds_block_4.pw_bn_beta = parameters[35].data
    model.ds_block_5.dw_weights = parameters[36].data
    model.ds_block_5.dw_bias = parameters[37].data
    model.ds_block_5.dw_bn_gamma = parameters[38].data
    model.ds_block_5.dw_bn_beta = parameters[39].data
    model.ds_block_5.pw_weights = parameters[40].data
    model.ds_block_5.pw_bias = parameters[41].data
    model.ds_block_5.pw_bn_gamma = parameters[42].data
    model.ds_block_5.pw_bn_beta = parameters[43].data
    model.ds_block_6.dw_weights = parameters[44].data
    model.ds_block_6.dw_bias = parameters[45].data
    model.ds_block_6.dw_bn_gamma = parameters[46].data
    model.ds_block_6.dw_bn_beta = parameters[47].data
    model.ds_block_6.pw_weights = parameters[48].data
    model.ds_block_6.pw_bias = parameters[49].data
    model.ds_block_6.pw_bn_gamma = parameters[50].data
    model.ds_block_6.pw_bn_beta = parameters[51].data
    model.ds_block_7.dw_weights = parameters[52].data
    model.ds_block_7.dw_bias = parameters[53].data
    model.ds_block_7.dw_bn_gamma = parameters[54].data
    model.ds_block_7.dw_bn_beta = parameters[55].data
    model.ds_block_7.pw_weights = parameters[56].data
    model.ds_block_7.pw_bias = parameters[57].data
    model.ds_block_7.pw_bn_gamma = parameters[58].data
    model.ds_block_7.pw_bn_beta = parameters[59].data
    model.ds_block_8.dw_weights = parameters[60].data
    model.ds_block_8.dw_bias = parameters[61].data
    model.ds_block_8.dw_bn_gamma = parameters[62].data
    model.ds_block_8.dw_bn_beta = parameters[63].data
    model.ds_block_8.pw_weights = parameters[64].data
    model.ds_block_8.pw_bias = parameters[65].data
    model.ds_block_8.pw_bn_gamma = parameters[66].data
    model.ds_block_8.pw_bn_beta = parameters[67].data
    model.ds_block_9.dw_weights = parameters[68].data
    model.ds_block_9.dw_bias = parameters[69].data
    model.ds_block_9.dw_bn_gamma = parameters[70].data
    model.ds_block_9.dw_bn_beta = parameters[71].data
    model.ds_block_9.pw_weights = parameters[72].data
    model.ds_block_9.pw_bias = parameters[73].data
    model.ds_block_9.pw_bn_gamma = parameters[74].data
    model.ds_block_9.pw_bn_beta = parameters[75].data
    model.ds_block_10.dw_weights = parameters[76].data
    model.ds_block_10.dw_bias = parameters[77].data
    model.ds_block_10.dw_bn_gamma = parameters[78].data
    model.ds_block_10.dw_bn_beta = parameters[79].data
    model.ds_block_10.pw_weights = parameters[80].data
    model.ds_block_10.pw_bias = parameters[81].data
    model.ds_block_10.pw_bn_gamma = parameters[82].data
    model.ds_block_10.pw_bn_beta = parameters[83].data
    model.ds_block_11.dw_weights = parameters[84].data
    model.ds_block_11.dw_bias = parameters[85].data
    model.ds_block_11.dw_bn_gamma = parameters[86].data
    model.ds_block_11.dw_bn_beta = parameters[87].data
    model.ds_block_11.pw_weights = parameters[88].data
    model.ds_block_11.pw_bias = parameters[89].data
    model.ds_block_11.pw_bn_gamma = parameters[90].data
    model.ds_block_11.pw_bn_beta = parameters[91].data
    model.ds_block_12.dw_weights = parameters[92].data
    model.ds_block_12.dw_bias = parameters[93].data
    model.ds_block_12.dw_bn_gamma = parameters[94].data
    model.ds_block_12.dw_bn_beta = parameters[95].data
    model.ds_block_12.pw_weights = parameters[96].data
    model.ds_block_12.pw_bias = parameters[97].data
    model.ds_block_12.pw_bn_gamma = parameters[98].data
    model.ds_block_12.pw_bn_beta = parameters[99].data
    model.ds_block_13.dw_weights = parameters[100].data
    model.ds_block_13.dw_bias = parameters[101].data
    model.ds_block_13.dw_bn_gamma = parameters[102].data
    model.ds_block_13.dw_bn_beta = parameters[103].data
    model.ds_block_13.pw_weights = parameters[104].data
    model.ds_block_13.pw_bias = parameters[105].data
    model.ds_block_13.pw_bn_gamma = parameters[106].data
    model.ds_block_13.pw_bn_beta = parameters[107].data
    model.fc_weights = parameters[108].data
    model.fc_bias = parameters[109].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: MobileNetV1,
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
        model: MobileNetV1 model.
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
            batch_labels_int, num_classes=model.fc_bias.shape()[0]
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
    print("MobileNetV1 Training on CIFAR-10 Dataset (Autograd)")
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
    print("Initializing MobileNetV1 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = MobileNetV1(num_classes=dataset_info.num_classes())
    print("  Model initialized with", model.fc_bias.shape()[0], "classes")
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
