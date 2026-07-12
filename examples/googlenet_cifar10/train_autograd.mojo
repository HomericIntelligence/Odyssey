"""Training Script for GoogLeNet (Inception-v1) on CIFAR-10 using Tape-Based Autograd.

Autograd variant of `train.mojo` (sub-task 7 of #5454). Replaces the manual
forward + `inception_forward_cached` / `inception_backward` /
`concatenate_depthwise_backward` / `conv2d_backward` / `batch_norm2d_backward`
chain with the Variable / GradientTape substrate: the GoogLeNet forward is
recorded via `variable_*` ops, gradients come from `loss.backward(tape)`, and
parameters update through `optimizer.step(params, tape)`.

Default optimizer is AdamW (matches examples/grok/lenet_emnist/); pass
`--optimizer sgd` to select SGD. The manual `train.mojo` is intentionally left
in place — the tracker-level cleanup (#5454) removes the manual path once all
ports land.

INCEPTION DEPTH-CONCAT: each of the 9 Inception modules has 4 parallel branches
(1x1; 1x1->3x3; 1x1->5x5; maxpool->1x1), each ending in conv -> BN -> ReLU. The
manual forward joins them with `concatenate_depthwise(b1, b2, b3, b4)`; the
autograd path uses the new `variable_concat([b1, b2, b3, b4], tape, axis=1)`
op (channel axis for NCHW). `Variable` is NOT implicitly copyable, so the branch
list is built with explicit `.copy()` (which preserves the tape id, so gradients
still route back to each branch's producing ops).

BATCH NORM is ported in ALL four branches of every module (and the stem):
`variable_batch_norm` returns `(output_var, new_running_mean, new_running_var)`.
`gamma`/`beta` are trainable Variables; `running_mean`/`running_var` are plain
`AnyTensor` buffers threaded back into the model's fields after every layer
(exactly like the manual `forward()`), so the running statistics keep advancing.
`training=True` is used here (a training port; no eval loop).

GLOBAL AVERAGE POOLING substitution: the manual forward ends with
`global_avgpool2d(out)` over the (batch, 1024, H, W) inception-5b output, then a
flatten to (batch, 1024) before the FC. The autograd substrate has NO
`variable_global_avgpool`, so GAP is expressed as two `variable_mean` reductions
over the two spatial axes: `variable_mean(x, tape, axis=3)` -> (batch, 1024, H),
then `variable_mean(..., tape, axis=2)` -> (batch, 1024). Averaging both spatial
axes is exactly global average pooling, and the result is already the flattened
(batch, 1024) shape, so NO flatten op is needed before the FC. (NOTE:
`variable_mean`'s `axis=-1` default is a FULL reduction to a scalar, not a
last-axis reduction — the explicit positive axes 3 then 2 are required.)

OMITTED vs the manual path:
    - DROPOUT (p=0.4, applied after GAP in the manual forward): the autograd
      substrate has no `variable_dropout` op (verified: no OP_DROPOUT /
      variable_dropout in src/projectodyssey/autograd/). The recorded head is
      therefore GAP -> linear, matching a non-regularized forward. This is a
      mechanism port, not a convergence-tuned run.
    - The step-LR schedule and SGD momentum (the autograd SGD is constructed as
      SGD(learning_rate=...)); these are training-tuning details, not required
      for the mechanism.

Usage:
    mojo run examples/googlenet_cifar10/train_autograd.mojo --epochs 1 --batch-size 32
    mojo run examples/googlenet_cifar10/train_autograd.mojo --optimizer sgd

Requirements:
    - CIFAR-10 dataset downloaded (run: python scripts/download_cifar10.py)
    - Dataset location: datasets/cifar10/
"""

from model import GoogLeNet, InceptionModule
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
    variable_maxpool2d,
    variable_linear,
    variable_cross_entropy,
    variable_concat,
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

    AdamW is the default (DoD of #5563); only an explicit `--optimizer sgd`
    selects SGD. The parser's built-in default for `optimizer` is "sgd" and
    `parse()` pre-populates it, so we cannot rely on `get_string`'s default —
    the caller gates on `was_user_supplied` before reading the value.
    """
    return optimizer != "sgd"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using the shared training parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "googlenet_weights_autograd")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var epochs = args.get_int("epochs", 100)
    var batch_size = args.get_int("batch-size", 128)
    var weight_decay = args.get_float("weight-decay", 0.01)
    # Resolve the AdamW default in code. The shared parser registers `optimizer`
    # with a built-in default of "sgd", and `parse()` pre-populates it into the
    # value map — so `get_string("optimizer", "adamw")` would return "sgd" even
    # when unset (the fallback is dead). Gate on `was_user_supplied`: only an
    # explicit `--optimizer` is honored; unset defaults to AdamW (DoD of #5563).
    var optimizer: String = "adamw"
    if args.was_user_supplied("optimizer"):
        optimizer = args.get_string("optimizer", "adamw")
    # Optimizer-appropriate default learning rate. GoogLeNet is very deep (stem
    # + 9 Inception modules, each with 6 conv+BN stages); AdamW's adaptive step
    # diverges on this stack at the SGD-scale lr, so default AdamW to 1e-4 and
    # SGD to 1e-2 when the user did not pass --lr. An explicit --lr always wins.
    var learning_rate: Float64
    if args.was_user_supplied("lr"):
        learning_rate = args.get_float("lr", 0.0001)
    elif optimizer == "sgd":
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    var data_dir = args.get_string("data-dir", "datasets/cifar10")
    var weights_dir = args.get_string(
        "weights-dir", "googlenet_weights_autograd"
    )
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


def inception_forward(
    mut module: InceptionModule,
    x: Variable,
    mut tape: GradientTape,
    mut parameters: List[Variable],
) raises -> Variable:
    """Record one Inception module's forward pass on the autograd tape.

    Mirrors `InceptionModule.forward` (model.mojo) using `variable_*` ops:
    four parallel branches (conv -> BN -> ReLU chains), joined by
    `variable_concat` along the channel axis. The module's 24 trainable
    parameters (6 conv weight/bias pairs + 6 BN gamma/beta pairs) are wrapped as
    Variables and APPENDED to the shared `parameters` list IN FIELD ORDER; the
    caller's write-back loop consumes `parameters` in that same order. The BN
    running-stat buffers are threaded back into the module's fields immediately,
    exactly like the manual forward.

    Field/append order (must match the write-back in train_batch):
        b1: conv1x1_1 (w, b), bn1x1_1 (gamma, beta)
        b2: conv1x1_2 (w, b), bn1x1_2 (gamma, beta),
            conv3x3   (w, b), bn3x3   (gamma, beta)
        b3: conv1x1_3 (w, b), bn1x1_3 (gamma, beta),
            conv5x5   (w, b), bn5x5   (gamma, beta)
        b4: conv1x1_4 (w, b), bn1x1_4 (gamma, beta)

    Args:
        module: The Inception module (running stats mutated in place).
        x: Module input Variable (batch, in_channels, H, W).
        tape: Gradient tape for recording.
        parameters: Shared list; this module's 24 Variables are appended.

    Returns:
        The depth-concatenated output Variable
        (batch, out_1x1 + out_3x3 + out_5x5 + pool_proj, H, W).
    """
    # Wrap this module's trainable params as Variables (field order).
    var conv1x1_1_w = Variable(module.conv1x1_1_weights, True, tape)
    var conv1x1_1_b = Variable(module.conv1x1_1_bias, True, tape)
    var bn1x1_1_gamma = Variable(module.bn1x1_1_gamma, True, tape)
    var bn1x1_1_beta = Variable(module.bn1x1_1_beta, True, tape)

    var conv1x1_2_w = Variable(module.conv1x1_2_weights, True, tape)
    var conv1x1_2_b = Variable(module.conv1x1_2_bias, True, tape)
    var bn1x1_2_gamma = Variable(module.bn1x1_2_gamma, True, tape)
    var bn1x1_2_beta = Variable(module.bn1x1_2_beta, True, tape)
    var conv3x3_w = Variable(module.conv3x3_weights, True, tape)
    var conv3x3_b = Variable(module.conv3x3_bias, True, tape)
    var bn3x3_gamma = Variable(module.bn3x3_gamma, True, tape)
    var bn3x3_beta = Variable(module.bn3x3_beta, True, tape)

    var conv1x1_3_w = Variable(module.conv1x1_3_weights, True, tape)
    var conv1x1_3_b = Variable(module.conv1x1_3_bias, True, tape)
    var bn1x1_3_gamma = Variable(module.bn1x1_3_gamma, True, tape)
    var bn1x1_3_beta = Variable(module.bn1x1_3_beta, True, tape)
    var conv5x5_w = Variable(module.conv5x5_weights, True, tape)
    var conv5x5_b = Variable(module.conv5x5_bias, True, tape)
    var bn5x5_gamma = Variable(module.bn5x5_gamma, True, tape)
    var bn5x5_beta = Variable(module.bn5x5_beta, True, tape)

    var conv1x1_4_w = Variable(module.conv1x1_4_weights, True, tape)
    var conv1x1_4_b = Variable(module.conv1x1_4_bias, True, tape)
    var bn1x1_4_gamma = Variable(module.bn1x1_4_gamma, True, tape)
    var bn1x1_4_beta = Variable(module.bn1x1_4_beta, True, tape)

    # ---- Branch 1: 1x1 conv -> BN -> ReLU ----
    var b1_conv = variable_conv2d(
        x, conv1x1_1_w, conv1x1_1_b, tape, stride=1, padding=0
    )
    var b1_bn_res = variable_batch_norm(
        b1_conv,
        bn1x1_1_gamma,
        bn1x1_1_beta,
        module.bn1x1_1_running_mean,
        module.bn1x1_1_running_var,
        tape,
        training=True,
    )
    var b1_bn = b1_bn_res[0].copy()
    module.bn1x1_1_running_mean = b1_bn_res[1]
    module.bn1x1_1_running_var = b1_bn_res[2]
    var b1 = variable_relu(b1_bn, tape)

    # ---- Branch 2: 1x1 reduce -> BN -> ReLU -> 3x3 -> BN -> ReLU ----
    var b2_conv1 = variable_conv2d(
        x, conv1x1_2_w, conv1x1_2_b, tape, stride=1, padding=0
    )
    var b2_bn1_res = variable_batch_norm(
        b2_conv1,
        bn1x1_2_gamma,
        bn1x1_2_beta,
        module.bn1x1_2_running_mean,
        module.bn1x1_2_running_var,
        tape,
        training=True,
    )
    var b2_bn1 = b2_bn1_res[0].copy()
    module.bn1x1_2_running_mean = b2_bn1_res[1]
    module.bn1x1_2_running_var = b2_bn1_res[2]
    var b2_relu1 = variable_relu(b2_bn1, tape)
    var b2_conv2 = variable_conv2d(
        b2_relu1, conv3x3_w, conv3x3_b, tape, stride=1, padding=1
    )
    var b2_bn2_res = variable_batch_norm(
        b2_conv2,
        bn3x3_gamma,
        bn3x3_beta,
        module.bn3x3_running_mean,
        module.bn3x3_running_var,
        tape,
        training=True,
    )
    var b2_bn2 = b2_bn2_res[0].copy()
    module.bn3x3_running_mean = b2_bn2_res[1]
    module.bn3x3_running_var = b2_bn2_res[2]
    var b2 = variable_relu(b2_bn2, tape)

    # ---- Branch 3: 1x1 reduce -> BN -> ReLU -> 5x5 -> BN -> ReLU ----
    var b3_conv1 = variable_conv2d(
        x, conv1x1_3_w, conv1x1_3_b, tape, stride=1, padding=0
    )
    var b3_bn1_res = variable_batch_norm(
        b3_conv1,
        bn1x1_3_gamma,
        bn1x1_3_beta,
        module.bn1x1_3_running_mean,
        module.bn1x1_3_running_var,
        tape,
        training=True,
    )
    var b3_bn1 = b3_bn1_res[0].copy()
    module.bn1x1_3_running_mean = b3_bn1_res[1]
    module.bn1x1_3_running_var = b3_bn1_res[2]
    var b3_relu1 = variable_relu(b3_bn1, tape)
    var b3_conv2 = variable_conv2d(
        b3_relu1, conv5x5_w, conv5x5_b, tape, stride=1, padding=2
    )
    var b3_bn2_res = variable_batch_norm(
        b3_conv2,
        bn5x5_gamma,
        bn5x5_beta,
        module.bn5x5_running_mean,
        module.bn5x5_running_var,
        tape,
        training=True,
    )
    var b3_bn2 = b3_bn2_res[0].copy()
    module.bn5x5_running_mean = b3_bn2_res[1]
    module.bn5x5_running_var = b3_bn2_res[2]
    var b3 = variable_relu(b3_bn2, tape)

    # ---- Branch 4: 3x3 maxpool -> 1x1 projection -> BN -> ReLU ----
    var b4_pool = variable_maxpool2d(
        x, tape, kernel_size=3, stride=1, padding=1
    )
    var b4_conv = variable_conv2d(
        b4_pool, conv1x1_4_w, conv1x1_4_b, tape, stride=1, padding=0
    )
    var b4_bn_res = variable_batch_norm(
        b4_conv,
        bn1x1_4_gamma,
        bn1x1_4_beta,
        module.bn1x1_4_running_mean,
        module.bn1x1_4_running_var,
        tape,
        training=True,
    )
    var b4_bn = b4_bn_res[0].copy()
    module.bn1x1_4_running_mean = b4_bn_res[1]
    module.bn1x1_4_running_var = b4_bn_res[2]
    var b4 = variable_relu(b4_bn, tape)

    # ---- Depth-concat the 4 branches along the channel axis (axis=1) ----
    # Variable is not implicitly copyable; build the input list with .copy()
    # (the copy preserves each branch's tape id, so gradients still route).
    var branches = List[Variable]()
    branches.append(b1.copy())
    branches.append(b2.copy())
    branches.append(b3.copy())
    branches.append(b4.copy())
    var out = variable_concat(branches, tape, axis=1)

    # Append this module's 24 trainable Variables in field order. The caller's
    # write-back loop consumes them in the same order.
    parameters.append(conv1x1_1_w^)
    parameters.append(conv1x1_1_b^)
    parameters.append(bn1x1_1_gamma^)
    parameters.append(bn1x1_1_beta^)
    parameters.append(conv1x1_2_w^)
    parameters.append(conv1x1_2_b^)
    parameters.append(bn1x1_2_gamma^)
    parameters.append(bn1x1_2_beta^)
    parameters.append(conv3x3_w^)
    parameters.append(conv3x3_b^)
    parameters.append(bn3x3_gamma^)
    parameters.append(bn3x3_beta^)
    parameters.append(conv1x1_3_w^)
    parameters.append(conv1x1_3_b^)
    parameters.append(bn1x1_3_gamma^)
    parameters.append(bn1x1_3_beta^)
    parameters.append(conv5x5_w^)
    parameters.append(conv5x5_b^)
    parameters.append(bn5x5_gamma^)
    parameters.append(bn5x5_beta^)
    parameters.append(conv1x1_4_w^)
    parameters.append(conv1x1_4_b^)
    parameters.append(bn1x1_4_gamma^)
    parameters.append(bn1x1_4_beta^)

    return out^


def inception_writeback(
    mut module: InceptionModule, parameters: List[Variable], start: Int
) raises:
    """Write updated Inception Variables back into the module fields.

    `start` is the index in `parameters` where this module's 24 Variables begin
    (same field order as `inception_forward` appended them). Returns nothing;
    the module fields are mutated in place.
    """
    module.conv1x1_1_weights = parameters[start + 0].data
    module.conv1x1_1_bias = parameters[start + 1].data
    module.bn1x1_1_gamma = parameters[start + 2].data
    module.bn1x1_1_beta = parameters[start + 3].data
    module.conv1x1_2_weights = parameters[start + 4].data
    module.conv1x1_2_bias = parameters[start + 5].data
    module.bn1x1_2_gamma = parameters[start + 6].data
    module.bn1x1_2_beta = parameters[start + 7].data
    module.conv3x3_weights = parameters[start + 8].data
    module.conv3x3_bias = parameters[start + 9].data
    module.bn3x3_gamma = parameters[start + 10].data
    module.bn3x3_beta = parameters[start + 11].data
    module.conv1x1_3_weights = parameters[start + 12].data
    module.conv1x1_3_bias = parameters[start + 13].data
    module.bn1x1_3_gamma = parameters[start + 14].data
    module.bn1x1_3_beta = parameters[start + 15].data
    module.conv5x5_weights = parameters[start + 16].data
    module.conv5x5_bias = parameters[start + 17].data
    module.bn5x5_gamma = parameters[start + 18].data
    module.bn5x5_beta = parameters[start + 19].data
    module.conv1x1_4_weights = parameters[start + 20].data
    module.conv1x1_4_bias = parameters[start + 21].data
    module.bn1x1_4_gamma = parameters[start + 22].data
    module.bn1x1_4_beta = parameters[start + 23].data


def train_batch[
    O: Optimizer
](
    mut model: GoogLeNet,
    input: AnyTensor,
    labels: AnyTensor,
    mut optimizer: O,
    mut tape: GradientTape,
) raises -> Float32:
    """Run one batch through the autograd path and update parameters.

    Records the GoogLeNet forward (stem conv+BN, 3 maxpools, 9 Inception modules
    each with 4 branches joined by `variable_concat`, GAP via two `variable_mean`
    reductions, FC head), calls `loss.backward(tape)` for automatic gradients,
    then `optimizer.step(...)`.

    Parameter ordering (append == write-back):
        stem: initial_conv (w, b), initial_bn (gamma, beta) — 4 Variables
        then 9 Inception modules * 24 Variables each — appended by
        `inception_forward` in field order
        FC head: fc_weights, fc_bias — 2 Variables

    Args:
        model: GoogLeNet model (trainable tensors + BN running stat buffers).
        input: Batch of images (batch, 3, 32, 32).
        labels: One-hot encoded batch of labels (batch, num_classes).
        optimizer: An autograd optimizer (AdamW or SGD).
        tape: GradientTape recording this batch's operations.

    Returns:
        Loss value for this batch.
    """
    var parameters: List[Variable] = []

    # ---- Stem params ----
    var initial_conv_w = Variable(model.initial_conv_weights, True, tape)
    var initial_conv_b = Variable(model.initial_conv_bias, True, tape)
    var initial_bn_gamma = Variable(model.initial_bn_gamma, True, tape)
    var initial_bn_beta = Variable(model.initial_bn_beta, True, tape)

    var input_var = Variable(input, False, tape)
    var labels_var = Variable(labels, False, tape)

    # ========== Stem: conv (3x3, s1, p1) -> BN -> ReLU -> maxpool(3, s2, p1) ==
    var stem_conv = variable_conv2d(
        input_var, initial_conv_w, initial_conv_b, tape, stride=1, padding=1
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
    var stem_bn = stem_bn_res[0].copy()
    model.initial_bn_running_mean = stem_bn_res[1]
    model.initial_bn_running_var = stem_bn_res[2]
    var stem_relu = variable_relu(stem_bn, tape)
    var stem_pool = variable_maxpool2d(
        stem_relu, tape, kernel_size=3, stride=2, padding=1
    )
    # (batch, 64, 16, 16)

    # Append stem params (field order).
    parameters.append(initial_conv_w^)
    parameters.append(initial_conv_b^)
    parameters.append(initial_bn_gamma^)
    parameters.append(initial_bn_beta^)

    # ========== Inception 3a, 3b ==========
    var i3a = inception_forward(model.inception_3a, stem_pool, tape, parameters)
    var i3b = inception_forward(model.inception_3b, i3a, tape, parameters)
    # MaxPool 3x3, stride 2 -> (batch, 480, 8, 8)
    var pool3 = variable_maxpool2d(
        i3b, tape, kernel_size=3, stride=2, padding=1
    )

    # ========== Inception 4a-4e ==========
    var i4a = inception_forward(model.inception_4a, pool3, tape, parameters)
    var i4b = inception_forward(model.inception_4b, i4a, tape, parameters)
    var i4c = inception_forward(model.inception_4c, i4b, tape, parameters)
    var i4d = inception_forward(model.inception_4d, i4c, tape, parameters)
    var i4e = inception_forward(model.inception_4e, i4d, tape, parameters)
    # MaxPool 3x3, stride 2 -> (batch, 832, 4, 4)
    var pool4 = variable_maxpool2d(
        i4e, tape, kernel_size=3, stride=2, padding=1
    )

    # ========== Inception 5a, 5b ==========
    var i5a = inception_forward(model.inception_5a, pool4, tape, parameters)
    var i5b = inception_forward(model.inception_5b, i5a, tape, parameters)
    # (batch, 1024, 4, 4)

    # ========== Global Average Pooling via two mean reductions ==========
    # i5b is (batch, 1024, 4, 4). The manual forward does global_avgpool2d ->
    # (batch, 1024, 1, 1) -> flatten (batch, 1024). There is no
    # variable_global_avgpool, so reduce the two spatial axes with
    # variable_mean: axis=3 -> (batch, 1024, 4), then axis=2 -> (batch, 1024).
    # Averaging both spatial axes IS global average pooling, and the
    # (batch, 1024) result is already flattened, so no flatten op is needed.
    var gap_w = variable_mean(i5b, tape, axis=3)
    var gap = variable_mean(gap_w, tape, axis=2)

    # ========== FC head: (batch, 1024) -> (batch, num_classes) ==========
    # Dropout (p=0.4) is OMITTED: no variable_dropout op (see module docstring).
    var fc_weights = Variable(model.fc_weights, True, tape)
    var fc_bias = Variable(model.fc_bias, True, tape)
    var logits = variable_linear(gap, fc_weights, fc_bias, tape)

    # Cross-entropy loss (mean-reduced internally).
    var loss = variable_cross_entropy(logits, labels_var, tape)

    # ========== Backward Pass ==========
    loss.backward(tape)

    # Extract the scalar loss before the Variables are moved below.
    var loss_value = loss.data._data.bitcast[Float32]()[0]

    # Append FC head params (field order, after all inception modules).
    parameters.append(fc_weights^)
    parameters.append(fc_bias^)

    # ========== Parameter Update ==========
    optimizer.step(parameters, tape)

    # ========== Write updated Variable.data back into the model ==========
    # Same order as appended: 4 stem, then 9 * 24 inception, then 2 FC.
    model.initial_conv_weights = parameters[0].data
    model.initial_conv_bias = parameters[1].data
    model.initial_bn_gamma = parameters[2].data
    model.initial_bn_beta = parameters[3].data

    # Each Inception module contributes 24 Variables; write them back at the
    # matching offset (stem = 4 leading params).
    inception_writeback(model.inception_3a, parameters, 4 + 0 * 24)
    inception_writeback(model.inception_3b, parameters, 4 + 1 * 24)
    inception_writeback(model.inception_4a, parameters, 4 + 2 * 24)
    inception_writeback(model.inception_4b, parameters, 4 + 3 * 24)
    inception_writeback(model.inception_4c, parameters, 4 + 4 * 24)
    inception_writeback(model.inception_4d, parameters, 4 + 5 * 24)
    inception_writeback(model.inception_4e, parameters, 4 + 6 * 24)
    inception_writeback(model.inception_5a, parameters, 4 + 7 * 24)
    inception_writeback(model.inception_5b, parameters, 4 + 8 * 24)

    # FC head immediately follows the 9 * 24 = 216 inception params.
    model.fc_weights = parameters[4 + 9 * 24 + 0].data
    model.fc_bias = parameters[4 + 9 * 24 + 1].data

    # Zero gradients for the next batch.
    optimizer.zero_grad(tape)

    return loss_value


def train_epoch[
    O: Optimizer
](
    mut model: GoogLeNet,
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
        model: GoogLeNet model.
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

        # One-hot encode the batch labels. GoogLeNet does not store num_classes
        # as a field; the FC bias length (fc_bias shape [num_classes]) is the
        # authoritative class count.
        var num_classes = model.fc_bias.shape()[0]
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=num_classes
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
    print("GoogLeNet Training on CIFAR-10 Dataset (Autograd)")
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
    print("Initializing GoogLeNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = GoogLeNet(num_classes=dataset_info.num_classes())
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
