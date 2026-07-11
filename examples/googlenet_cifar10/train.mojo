"""Training Script for GoogLeNet on CIFAR-10.

This script trains a GoogLeNet model on CIFAR-10 using manual backpropagation.

Usage:
    # Train with default parameters
    mojo run examples/googlenet_cifar10/train.mojo

    # Train with custom parameters
    mojo run examples/googlenet_cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01

Features:
    - Manual backpropagation through 9 Inception modules
    - SGD optimizer with momentum (0.9)
    - Batch normalization in training mode
    - Learning rate scheduling (step decay)
    - Weight save/load functionality
    - Progress monitoring and validation

Architecture:
    - 22 layers deep (9 Inception modules + initial layers + classifier)
    - Each Inception module has 4 parallel branches
    - ~6.8M parameters total
    - Batch normalization after every convolution
    - Global average pooling + dropout before classifier

Training Details:
    - Loss: Cross-entropy
    - Optimizer: SGD with momentum
    - Learning rate schedule: Step decay (×0.2 every 60 epochs)
    - Batch size: 128 (default)
    - Epochs: 200 (recommended)
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    global_avgpool2d,
    global_avgpool2d_backward,
)
from projectodyssey.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
)
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.dropout import dropout, dropout_backward
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.training.schedulers import step_lr
from projectodyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
)
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.data.datasets import CIFAR10Dataset
from projectodyssey.data import one_hot_encode
from projectodyssey.utils.training_args import parse_training_args_with_defaults
from projectodyssey.training.optimizers import sgd_momentum_update_inplace
from model import (
    GoogLeNet,
    InceptionModule,
    inception_forward_cached,
    inception_backward,
    InceptionGradients,
)


def _append_inception_velocities(
    mut velocities: List[AnyTensor], module: InceptionModule
) raises:
    """Append 24 zero velocity tensors for one Inception module.

    Order matches InceptionModule field order in model.mojo:57-101.
    Running mean/var are NOT trainable, so no velocity for them.
    """
    # Branch 1 (4)
    velocities.append(zeros(module.conv1x1_1_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv1x1_1_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_1_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_1_beta.shape(), DType.float32))
    # Branch 2 (8)
    velocities.append(zeros(module.conv1x1_2_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv1x1_2_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_2_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_2_beta.shape(), DType.float32))
    velocities.append(zeros(module.conv3x3_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv3x3_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn3x3_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn3x3_beta.shape(), DType.float32))
    # Branch 3 (8)
    velocities.append(zeros(module.conv1x1_3_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv1x1_3_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_3_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_3_beta.shape(), DType.float32))
    velocities.append(zeros(module.conv5x5_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv5x5_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn5x5_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn5x5_beta.shape(), DType.float32))
    # Branch 4 (4)
    velocities.append(zeros(module.conv1x1_4_weights.shape(), DType.float32))
    velocities.append(zeros(module.conv1x1_4_bias.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_4_gamma.shape(), DType.float32))
    velocities.append(zeros(module.bn1x1_4_beta.shape(), DType.float32))


def initialize_velocities(model: GoogLeNet) raises -> List[AnyTensor]:
    """Allocate exactly 222 zero velocity tensors for SGD-with-momentum.

    Layout (order must match parameter-update order in the backward slice):
      Initial block: 4      (conv W, B, bn gamma, beta)
      9 Inception  : 216    (24 per module x 9)
      Final FC     : 2      (weights, bias)
      Total        : 222
    """
    var velocities: List[AnyTensor] = []
    # Initial block (4)
    velocities.append(zeros(model.initial_conv_weights.shape(), DType.float32))
    velocities.append(zeros(model.initial_conv_bias.shape(), DType.float32))
    velocities.append(zeros(model.initial_bn_gamma.shape(), DType.float32))
    velocities.append(zeros(model.initial_bn_beta.shape(), DType.float32))
    # 9 Inception modules (216)
    _append_inception_velocities(velocities, model.inception_3a)
    _append_inception_velocities(velocities, model.inception_3b)
    _append_inception_velocities(velocities, model.inception_4a)
    _append_inception_velocities(velocities, model.inception_4b)
    _append_inception_velocities(velocities, model.inception_4c)
    _append_inception_velocities(velocities, model.inception_4d)
    _append_inception_velocities(velocities, model.inception_4e)
    _append_inception_velocities(velocities, model.inception_5a)
    _append_inception_velocities(velocities, model.inception_5b)
    # Final FC (2)
    velocities.append(zeros(model.fc_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc_bias.shape(), DType.float32))
    # Fail-loud runtime guard (executable-convention-guard pattern).
    if len(velocities) != 222:
        raise Error(
            "initialize_velocities: expected exactly 222 velocity tensors"
            " (4 initial + 9 Inception x 24 + 2 FC), got "
            + String(len(velocities))
        )
    return velocities^


def _flatten_gap(gap_out: AnyTensor) raises -> AnyTensor:
    """Flatten global-avgpool output (N, C, 1, 1) -> (N, C) using
    AnyTensor.reshape (src/projectodyssey/tensor/any_tensor.mojo:655).
    """
    var gap_shape = gap_out.shape()
    var flat_shape: List[Int] = [gap_shape[0], gap_shape[1]]
    var flat = gap_out.reshape(flat_shape)
    return flat^


def _unflatten_gap_grad(
    grad_flat: AnyTensor, gap_shape: List[Int]
) raises -> AnyTensor:
    """Reshape (N, C) gradient back into (N, C, 1, 1). Used by the backward
    slice; defined here alongside _flatten_gap so the pair stays consistent.
    """
    return grad_flat.reshape(gap_shape)


def train_epoch(
    mut model: GoogLeNet,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
    epoch: Int,
    total_epochs: Int,
    max_batches: Int = 0,
) raises -> Float32:
    """Train for one epoch. Per-batch work is delegated to compute_gradients.

    `max_batches` caps batches this epoch (0 = unbounded); used by --smoke /
    --max-batches to bound a per-PR run (#5551).
    """
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    if max_batches > 0 and max_batches < num_batches:
        num_batches = max_batches
    var total_loss = Float32(0.0)
    print(
        "Epoch "
        + String(epoch + 1)
        + "/"
        + String(total_epochs)
        + ": lr="
        + String(learning_rate)
    )
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            train_images, train_labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels_raw = batch_pair[1]
        # cross_entropy requires one-hot targets (same shape as logits), but the
        # dataset yields uint8 class indices — encode per batch (matches the
        # ResNet-18 example and cross_entropy's documented contract).
        var batch_labels = one_hot_encode(batch_labels_raw, 10)
        var batch_loss = compute_gradients(
            model,
            batch_images,
            batch_labels,
            learning_rate,
            momentum,
            velocities,
        )
        total_loss += batch_loss
        if (batch_idx + 1) % 100 == 0 or num_batches <= 10:
            var avg = total_loss / Float32(batch_idx + 1)
            print(
                "  Batch "
                + String(batch_idx + 1)
                + "/"
                + String(num_batches)
                + " - Loss: "
                + String(avg)
            )
    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: " + String(avg_loss))
    return avg_loss


def _update_inception(
    mut module: InceptionModule,
    grads: InceptionGradients,
    mut velocities: List[AnyTensor],
    base: Int,
    lr: Float32,
    momentum: Float32,
) raises:
    """Apply SGD-momentum updates for one Inception module's 24 parameters.

    `base` is the velocity index of this module's first parameter. The order
    matches _append_inception_velocities exactly:
      conv1x1_1 W,B,gamma,beta | conv1x1_2 W,B,gamma,beta | conv3x3 W,B,gamma,beta
      | conv1x1_3 W,B,gamma,beta | conv5x5 W,B,gamma,beta | conv1x1_4 W,B,gamma,beta
    """
    sgd_momentum_update_inplace(
        module.conv1x1_1_weights,
        grads.g_conv1x1_1_w,
        velocities[base + 0],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_1_bias,
        grads.g_conv1x1_1_b,
        velocities[base + 1],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_1_gamma,
        grads.g_bn1x1_1_gamma,
        velocities[base + 2],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_1_beta,
        grads.g_bn1x1_1_beta,
        velocities[base + 3],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_2_weights,
        grads.g_conv1x1_2_w,
        velocities[base + 4],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_2_bias,
        grads.g_conv1x1_2_b,
        velocities[base + 5],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_2_gamma,
        grads.g_bn1x1_2_gamma,
        velocities[base + 6],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_2_beta,
        grads.g_bn1x1_2_beta,
        velocities[base + 7],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv3x3_weights,
        grads.g_conv3x3_w,
        velocities[base + 8],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv3x3_bias,
        grads.g_conv3x3_b,
        velocities[base + 9],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn3x3_gamma,
        grads.g_bn3x3_gamma,
        velocities[base + 10],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn3x3_beta,
        grads.g_bn3x3_beta,
        velocities[base + 11],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_3_weights,
        grads.g_conv1x1_3_w,
        velocities[base + 12],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_3_bias,
        grads.g_conv1x1_3_b,
        velocities[base + 13],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_3_gamma,
        grads.g_bn1x1_3_gamma,
        velocities[base + 14],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_3_beta,
        grads.g_bn1x1_3_beta,
        velocities[base + 15],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv5x5_weights,
        grads.g_conv5x5_w,
        velocities[base + 16],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv5x5_bias,
        grads.g_conv5x5_b,
        velocities[base + 17],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn5x5_gamma,
        grads.g_bn5x5_gamma,
        velocities[base + 18],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn5x5_beta,
        grads.g_bn5x5_beta,
        velocities[base + 19],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_4_weights,
        grads.g_conv1x1_4_w,
        velocities[base + 20],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.conv1x1_4_bias,
        grads.g_conv1x1_4_b,
        velocities[base + 21],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_4_gamma,
        grads.g_bn1x1_4_gamma,
        velocities[base + 22],
        Float64(lr),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        module.bn1x1_4_beta,
        grads.g_bn1x1_4_beta,
        velocities[base + 23],
        Float64(lr),
        Float64(momentum),
    )


def compute_gradients(
    mut model: GoogLeNet,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """One training step: forward (with caching) -> loss -> backward -> SGD.

    Full backward pass through all 9 Inception modules (#3184). Uses
    inception_forward_cached / inception_backward for each module, plus the
    initial conv block and the GAP->dropout->FC head. Returns the batch loss.
    """
    # ---- Initial block: conv3x3 -> BN -> ReLU -> MaxPool ----
    var init_conv = conv2d(
        input,
        model.initial_conv_weights,
        model.initial_conv_bias,
        stride=1,
        padding=1,
    )
    var init_bn, _, _ = batch_norm2d(
        init_conv,
        model.initial_bn_gamma,
        model.initial_bn_beta,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        True,
    )
    var init_relu = relu(init_bn)
    var init_pool = maxpool2d(init_relu, kernel_size=3, stride=2, padding=1)

    # ---- 9 Inception modules with maxpool transitions ----
    var f3a = inception_forward_cached(model.inception_3a, init_pool)
    var f3b = inception_forward_cached(model.inception_3b, f3a[0])
    var mid1 = maxpool2d(f3b[0], kernel_size=3, stride=2, padding=1)
    var f4a = inception_forward_cached(model.inception_4a, mid1)
    var f4b = inception_forward_cached(model.inception_4b, f4a[0])
    var f4c = inception_forward_cached(model.inception_4c, f4b[0])
    var f4d = inception_forward_cached(model.inception_4d, f4c[0])
    var f4e = inception_forward_cached(model.inception_4e, f4d[0])
    var mid2 = maxpool2d(f4e[0], kernel_size=3, stride=2, padding=1)
    var f5a = inception_forward_cached(model.inception_5a, mid2)
    var f5b = inception_forward_cached(model.inception_5b, f5a[0])

    # ---- Head: GAP -> flatten -> dropout -> linear -> CE ----
    var gap_out = global_avgpool2d(f5b[0])
    var gap_shape = gap_out.shape()
    var flat_out = _flatten_gap(gap_out)
    var drop_result = dropout(flat_out, Float64(0.4), training=True)
    var drop_out = drop_result[0]
    var drop_mask = drop_result[1]
    var logits = linear(drop_out, model.fc_weights, model.fc_bias)
    # Float32-only contract: the scalar-loss read and grad_ones write below
    # reinterpret memory via bitcast[Float32]. Guard so a non-float32 logits
    # tensor fails loudly instead of silently miscomputing.
    if logits.dtype() != DType.float32:
        raise Error(
            "compute_gradients requires float32 logits, got "
            + String(logits.dtype())
        )
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ================= BACKWARD =================
    # Head backward: CE -> linear -> dropout -> unflatten -> GAP
    var grad_ones = zeros([1], logits.dtype())
    grad_ones._data.bitcast[Float32]()[0] = Float32(1.0)
    var d_logits = cross_entropy_backward(grad_ones, logits, labels)
    var fc = linear_backward(d_logits, drop_out, model.fc_weights)
    var d_drop = dropout_backward(fc.grad_input, drop_mask, Float64(0.4))
    var d_flat = _unflatten_gap_grad(d_drop, gap_shape)
    var d_gap = global_avgpool2d_backward(d_flat, f5b[0])

    # Inception backward in reverse: 5b, 5a, [maxpool], 4e..4a, [maxpool], 3b, 3a
    var g5b = inception_backward(model.inception_5b, d_gap, f5b[1])
    var g5a = inception_backward(model.inception_5a, g5b.grad_input, f5a[1])
    var d_mid2 = maxpool2d_backward(
        g5a.grad_input, f4e[0], kernel_size=3, stride=2, padding=1
    )
    var g4e = inception_backward(model.inception_4e, d_mid2, f4e[1])
    var g4d = inception_backward(model.inception_4d, g4e.grad_input, f4d[1])
    var g4c = inception_backward(model.inception_4c, g4d.grad_input, f4c[1])
    var g4b = inception_backward(model.inception_4b, g4c.grad_input, f4b[1])
    var g4a = inception_backward(model.inception_4a, g4b.grad_input, f4a[1])
    var d_mid1 = maxpool2d_backward(
        g4a.grad_input, f3b[0], kernel_size=3, stride=2, padding=1
    )
    var g3b = inception_backward(model.inception_3b, d_mid1, f3b[1])
    var g3a = inception_backward(model.inception_3a, g3b.grad_input, f3a[1])

    # Initial block backward: maxpool -> relu -> BN -> conv
    var d_init_pool = maxpool2d_backward(
        g3a.grad_input, init_relu, kernel_size=3, stride=2, padding=1
    )
    var d_init_relu = relu_backward(d_init_pool, init_bn)
    var init_bn_in, g_init_bn_gamma, g_init_bn_beta = batch_norm2d_backward(
        d_init_relu,
        init_conv,
        model.initial_bn_gamma,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        training=True,
    )
    var init_c = conv2d_backward(
        init_bn_in, input, model.initial_conv_weights, stride=1, padding=1
    )

    # ================= SGD MOMENTUM UPDATES =================
    # Initial block (velocity indices 0..3)
    sgd_momentum_update_inplace(
        model.initial_conv_weights,
        init_c.grad_weights,
        velocities[0],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_conv_bias,
        init_c.grad_bias,
        velocities[1],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_bn_gamma,
        g_init_bn_gamma,
        velocities[2],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_bn_beta,
        g_init_bn_beta,
        velocities[3],
        Float64(learning_rate),
        Float64(momentum),
    )
    # 9 Inception modules (indices 4 + 24*k)
    _update_inception(
        model.inception_3a, g3a, velocities, 4 + 24 * 0, learning_rate, momentum
    )
    _update_inception(
        model.inception_3b, g3b, velocities, 4 + 24 * 1, learning_rate, momentum
    )
    _update_inception(
        model.inception_4a, g4a, velocities, 4 + 24 * 2, learning_rate, momentum
    )
    _update_inception(
        model.inception_4b, g4b, velocities, 4 + 24 * 3, learning_rate, momentum
    )
    _update_inception(
        model.inception_4c, g4c, velocities, 4 + 24 * 4, learning_rate, momentum
    )
    _update_inception(
        model.inception_4d, g4d, velocities, 4 + 24 * 5, learning_rate, momentum
    )
    _update_inception(
        model.inception_4e, g4e, velocities, 4 + 24 * 6, learning_rate, momentum
    )
    _update_inception(
        model.inception_5a, g5a, velocities, 4 + 24 * 7, learning_rate, momentum
    )
    _update_inception(
        model.inception_5b, g5b, velocities, 4 + 24 * 8, learning_rate, momentum
    )
    # Final FC (indices 220, 221)
    sgd_momentum_update_inplace(
        model.fc_weights,
        fc.grad_weights,
        velocities[220],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.fc_bias,
        fc.grad_bias,
        velocities[221],
        Float64(learning_rate),
        Float64(momentum),
    )

    return loss


def validate(
    mut model: GoogLeNet,
    val_images: AnyTensor,
    val_labels: AnyTensor,
    batch_size: Int,
) raises -> Float32:
    """Validate model on validation set.

    Args:
        model: GoogLeNet model.
        val_images: Validation images (N, 3, 32, 32).
        val_labels: Validation labels (N,).
        batch_size: Mini-batch size.

    Returns:
        Validation accuracy (percentage).
    """
    var num_samples = val_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    var total_correct = 0

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            val_images, val_labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass (inference mode)
        var logits = model.forward(batch_images, training=False)

        # Compute accuracy
        for i in range(current_batch_size):
            var logits_data = logits._data.bitcast[Float32]()
            var pred_class = 0
            var max_logit = logits_data[i * 10]
            for j in range(1, 10):
                if logits_data[i * 10 + j] > max_logit:
                    max_logit = logits_data[i * 10 + j]
                    pred_class = j

            var labels_data = batch_labels._data.bitcast[UInt8]()
            var true_class = Int(labels_data[i])
            if pred_class == true_class:
                total_correct += 1

    var accuracy = Float32(total_correct) / Float32(num_samples) * 100.0
    return accuracy


def main() raises:
    """Main training entry point."""
    print("=" * 60)
    print("GoogLeNet Training on CIFAR-10")
    print("=" * 60)
    print()

    # Parse arguments using standardized TrainingArgs
    var args = parse_training_args_with_defaults(
        default_epochs=200,
        default_batch_size=128,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets/cifar10",
        default_weights_dir="googlenet_weights",
        default_lr_decay_epochs=60,
        default_lr_decay_factor=0.2,
    )

    var epochs = args.epochs
    var batch_size = args.batch_size
    var initial_lr = Float32(args.learning_rate)
    var momentum = Float32(args.momentum)
    var data_dir = args.data_dir
    var weights_dir = args.weights_dir
    var lr_decay_epochs = args.lr_decay_epochs
    var lr_decay_factor = Float32(args.lr_decay_factor)
    var max_batches = args.max_batches
    var smoke = args.smoke

    print("Configuration:")
    print("  Epochs: " + String(epochs))
    print("  Batch size: " + String(batch_size))
    print("  Initial learning rate: " + String(initial_lr))
    print("  Momentum: " + String(momentum))
    print("  Data directory: " + String(data_dir))
    print("  Weights directory: " + String(weights_dir))
    print("  LR Decay Epochs: " + String(lr_decay_epochs))
    print("  LR Decay Factor: " + String(lr_decay_factor))
    print()

    # Load data — real CIFAR-10, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset download entirely so the training
    # entrypoint can run per-PR in CI. It checks the MECHANISM (the loop runs
    # and emits finite, parseable, decreasing loss), not convergence.
    var train_images: AnyTensor
    var train_labels: AnyTensor
    var test_images: AnyTensor
    var test_labels: AnyTensor
    if smoke:
        print("Smoke mode: using synthetic data (no dataset download)...")
        # Enough samples to yield >=2 batches under --max-batches so the CI gate
        # can assert a decreasing loss trend across batches. Class-correlated
        # signal makes the loss learnable; labels are RAW uint8 [N]
        # (train_epoch one-hot-encodes).
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
        # Reuse the same synthetic batch for "test" (eval is not asserted here).
        test_images = train_images
        test_labels = train_labels
    else:
        print("Loading CIFAR-10 training set...")
        var cifar10_dataset = CIFAR10Dataset(data_dir)
        train_images, train_labels = cifar10_dataset.get_train_data()
        # Test set is loaded lazily after training (see below); placeholder here
        # so both branches bind the same names.
        test_images = train_images
        test_labels = train_labels

    var num_train = train_images.shape()[0]
    print("  Training samples: " + String(num_train))
    print("  Image shape: (3, 32, 32)")
    print("  Number of classes: 10")
    print()

    # Initialize model
    print("Initializing GoogLeNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = GoogLeNet(num_classes=dataset_info.num_classes())
    print("  Model architecture: GoogLeNet (Inception-v1)")
    print("  Parameters: ~6.8M")
    print("  Inception modules: 9")
    print()

    # Training loop
    print("Starting training...")
    print()

    var velocities = initialize_velocities(model)
    print(
        "  Momentum velocities: "
        + String(len(velocities))
        + " tensors initialized"
    )
    print()
    for epoch in range(epochs):
        var lr = initial_lr
        if lr_decay_epochs > 0:
            lr = step_lr(
                initial_lr,
                epoch,
                step_size=lr_decay_epochs,
                gamma=lr_decay_factor,
            )
        var train_loss = train_epoch(
            model,
            train_images,
            train_labels,
            batch_size,
            lr,
            momentum,
            velocities,
            epoch,
            epochs,
            max_batches=max_batches,
        )

        print(
            "Epoch "
            + String(epoch + 1)
            + "/"
            + String(epochs)
            + " - Loss: "
            + String(train_loss)
        )

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            var val_acc = validate(
                model, train_images, train_labels, batch_size
            )
            print("  Validation Accuracy: " + String(val_acc) + "%")

        print()

    print("Training complete!")
    print()

    # Evaluate on the held-out TEST set (not the training set) for an honest
    # generalization metric. validate() runs forward inference (training=False)
    # and returns top-1 accuracy.
    print("Evaluating on test set...")
    if not smoke:
        var eval_dataset = CIFAR10Dataset(data_dir)
        test_images, test_labels = eval_dataset.get_test_data()
    var test_acc = validate(model, test_images, test_labels, batch_size)
    print("Test accuracy: " + String(test_acc) + "%")
    print()

    # Save weights
    print("Saving weights to " + String(weights_dir) + "/...")
    try:
        model.save_weights(weights_dir)
        print("  ✓ Weights saved successfully")
    except e:
        print("  ✗ Failed to save weights: " + String(e))

    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print("Total epochs: " + String(epochs))
    var final_lr = initial_lr
    if lr_decay_epochs > 0:
        final_lr = step_lr(
            initial_lr,
            epochs - 1,
            step_size=lr_decay_epochs,
            gamma=lr_decay_factor,
        )
    print("Final learning rate: " + String(final_lr))
    print("Model saved to: " + String(weights_dir) + "/")
    print("=" * 60)
