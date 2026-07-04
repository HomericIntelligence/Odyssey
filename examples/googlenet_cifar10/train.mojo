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
from projectodyssey.core.shape import split_with_indices
from projectodyssey.core.arithmetic import add, add_backward
from projectodyssey.training.schedulers import step_lr
from projectodyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
)
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.data.datasets import CIFAR10Dataset
from projectodyssey.utils.training_args import parse_training_args_with_defaults
from model import GoogLeNet, InceptionModule, concatenate_depthwise


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


def _flatten_gap(gap_out: AnyTensor) raises -> Tuple[AnyTensor, List[Int]]:
    """Flatten global-avgpool output (N, C, 1, 1) -> (N, C) using
    AnyTensor.reshape (src/projectodyssey/tensor/any_tensor.mojo:655).
    Returns the original 4-D shape so _unflatten_gap_grad can undo it.
    """
    var gap_shape = gap_out.shape()
    var flat_shape: List[Int] = [gap_shape[0], gap_shape[1]]
    var flat = gap_out.reshape(flat_shape)
    return (flat^, gap_shape)


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
) raises -> Float32:
    """Train for one epoch. Per-batch work is delegated to compute_gradients."""
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
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
        var batch_labels = batch_pair[1]
        var batch_loss = compute_gradients(
            model,
            batch_images,
            batch_labels,
            learning_rate,
            momentum,
            velocities,
        )
        total_loss += batch_loss
        if (batch_idx + 1) % 100 == 0:
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


def compute_gradients(
    mut model: GoogLeNet,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,  # Threaded in for backward slice (#3184) — unused in forward-only slice
    momentum: Float32,  # Threaded in for backward slice (#3184) — unused in forward-only slice
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """One-batch forward pass with full activation caching.

    Backward + SGD momentum updates are the next slice of #3184.
    The velocities buffer is threaded in NOW so the signature is stable.
    """
    # Suppress unused parameter warnings; these are consumed by backward slice (#3184)
    _ = learning_rate
    _ = momentum
    _ = velocities

    # ---- Initial block: conv3x3 -> BN -> ReLU -> MaxPool ----
    var init_conv_out = conv2d(
        input,
        model.initial_conv_weights,
        model.initial_conv_bias,
        stride=1,
        padding=1,
    )
    var init_bn_out, _, _ = batch_norm2d(
        init_conv_out,
        model.initial_bn_gamma,
        model.initial_bn_beta,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        True,
    )
    var init_relu_out = relu(init_bn_out)
    var init_pool_out = maxpool2d(
        init_relu_out, kernel_size=3, stride=2, padding=1
    )
    # ---- Inception 3a (input: init_pool_out; template for 3b..5b) ----
    var inc3a_b1_conv = conv2d(
        init_pool_out,
        model.inception_3a.conv1x1_1_weights,
        model.inception_3a.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc3a_b1_bn, _, _ = batch_norm2d(
        inc3a_b1_conv,
        model.inception_3a.bn1x1_1_gamma,
        model.inception_3a.bn1x1_1_beta,
        model.inception_3a.bn1x1_1_running_mean,
        model.inception_3a.bn1x1_1_running_var,
        True,
    )
    var inc3a_b1_relu = relu(inc3a_b1_bn)
    var inc3a_b2_conv1 = conv2d(
        init_pool_out,
        model.inception_3a.conv1x1_2_weights,
        model.inception_3a.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc3a_b2_bn1, _, _ = batch_norm2d(
        inc3a_b2_conv1,
        model.inception_3a.bn1x1_2_gamma,
        model.inception_3a.bn1x1_2_beta,
        model.inception_3a.bn1x1_2_running_mean,
        model.inception_3a.bn1x1_2_running_var,
        True,
    )
    var inc3a_b2_relu1 = relu(inc3a_b2_bn1)
    var inc3a_b2_conv2 = conv2d(
        inc3a_b2_relu1,
        model.inception_3a.conv3x3_weights,
        model.inception_3a.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc3a_b2_bn2, _, _ = batch_norm2d(
        inc3a_b2_conv2,
        model.inception_3a.bn3x3_gamma,
        model.inception_3a.bn3x3_beta,
        model.inception_3a.bn3x3_running_mean,
        model.inception_3a.bn3x3_running_var,
        True,
    )
    var inc3a_b2_relu2 = relu(inc3a_b2_bn2)
    var inc3a_b3_conv1 = conv2d(
        init_pool_out,
        model.inception_3a.conv1x1_3_weights,
        model.inception_3a.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc3a_b3_bn1, _, _ = batch_norm2d(
        inc3a_b3_conv1,
        model.inception_3a.bn1x1_3_gamma,
        model.inception_3a.bn1x1_3_beta,
        model.inception_3a.bn1x1_3_running_mean,
        model.inception_3a.bn1x1_3_running_var,
        True,
    )
    var inc3a_b3_relu1 = relu(inc3a_b3_bn1)
    var inc3a_b3_conv2 = conv2d(
        inc3a_b3_relu1,
        model.inception_3a.conv5x5_weights,
        model.inception_3a.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc3a_b3_bn2, _, _ = batch_norm2d(
        inc3a_b3_conv2,
        model.inception_3a.bn5x5_gamma,
        model.inception_3a.bn5x5_beta,
        model.inception_3a.bn5x5_running_mean,
        model.inception_3a.bn5x5_running_var,
        True,
    )
    var inc3a_b3_relu2 = relu(inc3a_b3_bn2)
    var inc3a_b4_pool = maxpool2d(
        init_pool_out, kernel_size=3, stride=1, padding=1
    )
    var inc3a_b4_conv = conv2d(
        inc3a_b4_pool,
        model.inception_3a.conv1x1_4_weights,
        model.inception_3a.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc3a_b4_bn, _, _ = batch_norm2d(
        inc3a_b4_conv,
        model.inception_3a.bn1x1_4_gamma,
        model.inception_3a.bn1x1_4_beta,
        model.inception_3a.bn1x1_4_running_mean,
        model.inception_3a.bn1x1_4_running_var,
        True,
    )
    var inc3a_b4_relu = relu(inc3a_b4_bn)
    var inc3a_out = concatenate_depthwise(
        inc3a_b1_relu, inc3a_b2_relu2, inc3a_b3_relu2, inc3a_b4_relu
    )
    # ---- Inception 3b (input: inc3a_out) ----
    var inc3b_b1_conv = conv2d(
        inc3a_out,
        model.inception_3b.conv1x1_1_weights,
        model.inception_3b.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc3b_b1_bn, _, _ = batch_norm2d(
        inc3b_b1_conv,
        model.inception_3b.bn1x1_1_gamma,
        model.inception_3b.bn1x1_1_beta,
        model.inception_3b.bn1x1_1_running_mean,
        model.inception_3b.bn1x1_1_running_var,
        True,
    )
    var inc3b_b1_relu = relu(inc3b_b1_bn)
    var inc3b_b2_conv1 = conv2d(
        inc3a_out,
        model.inception_3b.conv1x1_2_weights,
        model.inception_3b.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc3b_b2_bn1, _, _ = batch_norm2d(
        inc3b_b2_conv1,
        model.inception_3b.bn1x1_2_gamma,
        model.inception_3b.bn1x1_2_beta,
        model.inception_3b.bn1x1_2_running_mean,
        model.inception_3b.bn1x1_2_running_var,
        True,
    )
    var inc3b_b2_relu1 = relu(inc3b_b2_bn1)
    var inc3b_b2_conv2 = conv2d(
        inc3b_b2_relu1,
        model.inception_3b.conv3x3_weights,
        model.inception_3b.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc3b_b2_bn2, _, _ = batch_norm2d(
        inc3b_b2_conv2,
        model.inception_3b.bn3x3_gamma,
        model.inception_3b.bn3x3_beta,
        model.inception_3b.bn3x3_running_mean,
        model.inception_3b.bn3x3_running_var,
        True,
    )
    var inc3b_b2_relu2 = relu(inc3b_b2_bn2)
    var inc3b_b3_conv1 = conv2d(
        inc3a_out,
        model.inception_3b.conv1x1_3_weights,
        model.inception_3b.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc3b_b3_bn1, _, _ = batch_norm2d(
        inc3b_b3_conv1,
        model.inception_3b.bn1x1_3_gamma,
        model.inception_3b.bn1x1_3_beta,
        model.inception_3b.bn1x1_3_running_mean,
        model.inception_3b.bn1x1_3_running_var,
        True,
    )
    var inc3b_b3_relu1 = relu(inc3b_b3_bn1)
    var inc3b_b3_conv2 = conv2d(
        inc3b_b3_relu1,
        model.inception_3b.conv5x5_weights,
        model.inception_3b.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc3b_b3_bn2, _, _ = batch_norm2d(
        inc3b_b3_conv2,
        model.inception_3b.bn5x5_gamma,
        model.inception_3b.bn5x5_beta,
        model.inception_3b.bn5x5_running_mean,
        model.inception_3b.bn5x5_running_var,
        True,
    )
    var inc3b_b3_relu2 = relu(inc3b_b3_bn2)
    var inc3b_b4_pool = maxpool2d(inc3a_out, kernel_size=3, stride=1, padding=1)
    var inc3b_b4_conv = conv2d(
        inc3b_b4_pool,
        model.inception_3b.conv1x1_4_weights,
        model.inception_3b.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc3b_b4_bn, _, _ = batch_norm2d(
        inc3b_b4_conv,
        model.inception_3b.bn1x1_4_gamma,
        model.inception_3b.bn1x1_4_beta,
        model.inception_3b.bn1x1_4_running_mean,
        model.inception_3b.bn1x1_4_running_var,
        True,
    )
    var inc3b_b4_relu = relu(inc3b_b4_bn)
    var inc3b_out = concatenate_depthwise(
        inc3b_b1_relu, inc3b_b2_relu2, inc3b_b3_relu2, inc3b_b4_relu
    )
    var mid1_pool = maxpool2d(inc3b_out, kernel_size=3, stride=2, padding=1)
    # ---- Inception 4a (input: mid1_pool) ----
    var inc4a_b1_conv = conv2d(
        mid1_pool,
        model.inception_4a.conv1x1_1_weights,
        model.inception_4a.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc4a_b1_bn, _, _ = batch_norm2d(
        inc4a_b1_conv,
        model.inception_4a.bn1x1_1_gamma,
        model.inception_4a.bn1x1_1_beta,
        model.inception_4a.bn1x1_1_running_mean,
        model.inception_4a.bn1x1_1_running_var,
        True,
    )
    var inc4a_b1_relu = relu(inc4a_b1_bn)
    var inc4a_b2_conv1 = conv2d(
        mid1_pool,
        model.inception_4a.conv1x1_2_weights,
        model.inception_4a.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc4a_b2_bn1, _, _ = batch_norm2d(
        inc4a_b2_conv1,
        model.inception_4a.bn1x1_2_gamma,
        model.inception_4a.bn1x1_2_beta,
        model.inception_4a.bn1x1_2_running_mean,
        model.inception_4a.bn1x1_2_running_var,
        True,
    )
    var inc4a_b2_relu1 = relu(inc4a_b2_bn1)
    var inc4a_b2_conv2 = conv2d(
        inc4a_b2_relu1,
        model.inception_4a.conv3x3_weights,
        model.inception_4a.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc4a_b2_bn2, _, _ = batch_norm2d(
        inc4a_b2_conv2,
        model.inception_4a.bn3x3_gamma,
        model.inception_4a.bn3x3_beta,
        model.inception_4a.bn3x3_running_mean,
        model.inception_4a.bn3x3_running_var,
        True,
    )
    var inc4a_b2_relu2 = relu(inc4a_b2_bn2)
    var inc4a_b3_conv1 = conv2d(
        mid1_pool,
        model.inception_4a.conv1x1_3_weights,
        model.inception_4a.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc4a_b3_bn1, _, _ = batch_norm2d(
        inc4a_b3_conv1,
        model.inception_4a.bn1x1_3_gamma,
        model.inception_4a.bn1x1_3_beta,
        model.inception_4a.bn1x1_3_running_mean,
        model.inception_4a.bn1x1_3_running_var,
        True,
    )
    var inc4a_b3_relu1 = relu(inc4a_b3_bn1)
    var inc4a_b3_conv2 = conv2d(
        inc4a_b3_relu1,
        model.inception_4a.conv5x5_weights,
        model.inception_4a.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc4a_b3_bn2, _, _ = batch_norm2d(
        inc4a_b3_conv2,
        model.inception_4a.bn5x5_gamma,
        model.inception_4a.bn5x5_beta,
        model.inception_4a.bn5x5_running_mean,
        model.inception_4a.bn5x5_running_var,
        True,
    )
    var inc4a_b3_relu2 = relu(inc4a_b3_bn2)
    var inc4a_b4_pool = maxpool2d(mid1_pool, kernel_size=3, stride=1, padding=1)
    var inc4a_b4_conv = conv2d(
        inc4a_b4_pool,
        model.inception_4a.conv1x1_4_weights,
        model.inception_4a.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc4a_b4_bn, _, _ = batch_norm2d(
        inc4a_b4_conv,
        model.inception_4a.bn1x1_4_gamma,
        model.inception_4a.bn1x1_4_beta,
        model.inception_4a.bn1x1_4_running_mean,
        model.inception_4a.bn1x1_4_running_var,
        True,
    )
    var inc4a_b4_relu = relu(inc4a_b4_bn)
    var inc4a_out = concatenate_depthwise(
        inc4a_b1_relu, inc4a_b2_relu2, inc4a_b3_relu2, inc4a_b4_relu
    )
    # ---- Inception 4b (input: inc4a_out) ----
    var inc4b_b1_conv = conv2d(
        inc4a_out,
        model.inception_4b.conv1x1_1_weights,
        model.inception_4b.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc4b_b1_bn, _, _ = batch_norm2d(
        inc4b_b1_conv,
        model.inception_4b.bn1x1_1_gamma,
        model.inception_4b.bn1x1_1_beta,
        model.inception_4b.bn1x1_1_running_mean,
        model.inception_4b.bn1x1_1_running_var,
        True,
    )
    var inc4b_b1_relu = relu(inc4b_b1_bn)
    var inc4b_b2_conv1 = conv2d(
        inc4a_out,
        model.inception_4b.conv1x1_2_weights,
        model.inception_4b.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc4b_b2_bn1, _, _ = batch_norm2d(
        inc4b_b2_conv1,
        model.inception_4b.bn1x1_2_gamma,
        model.inception_4b.bn1x1_2_beta,
        model.inception_4b.bn1x1_2_running_mean,
        model.inception_4b.bn1x1_2_running_var,
        True,
    )
    var inc4b_b2_relu1 = relu(inc4b_b2_bn1)
    var inc4b_b2_conv2 = conv2d(
        inc4b_b2_relu1,
        model.inception_4b.conv3x3_weights,
        model.inception_4b.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc4b_b2_bn2, _, _ = batch_norm2d(
        inc4b_b2_conv2,
        model.inception_4b.bn3x3_gamma,
        model.inception_4b.bn3x3_beta,
        model.inception_4b.bn3x3_running_mean,
        model.inception_4b.bn3x3_running_var,
        True,
    )
    var inc4b_b2_relu2 = relu(inc4b_b2_bn2)
    var inc4b_b3_conv1 = conv2d(
        inc4a_out,
        model.inception_4b.conv1x1_3_weights,
        model.inception_4b.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc4b_b3_bn1, _, _ = batch_norm2d(
        inc4b_b3_conv1,
        model.inception_4b.bn1x1_3_gamma,
        model.inception_4b.bn1x1_3_beta,
        model.inception_4b.bn1x1_3_running_mean,
        model.inception_4b.bn1x1_3_running_var,
        True,
    )
    var inc4b_b3_relu1 = relu(inc4b_b3_bn1)
    var inc4b_b3_conv2 = conv2d(
        inc4b_b3_relu1,
        model.inception_4b.conv5x5_weights,
        model.inception_4b.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc4b_b3_bn2, _, _ = batch_norm2d(
        inc4b_b3_conv2,
        model.inception_4b.bn5x5_gamma,
        model.inception_4b.bn5x5_beta,
        model.inception_4b.bn5x5_running_mean,
        model.inception_4b.bn5x5_running_var,
        True,
    )
    var inc4b_b3_relu2 = relu(inc4b_b3_bn2)
    var inc4b_b4_pool = maxpool2d(inc4a_out, kernel_size=3, stride=1, padding=1)
    var inc4b_b4_conv = conv2d(
        inc4b_b4_pool,
        model.inception_4b.conv1x1_4_weights,
        model.inception_4b.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc4b_b4_bn, _, _ = batch_norm2d(
        inc4b_b4_conv,
        model.inception_4b.bn1x1_4_gamma,
        model.inception_4b.bn1x1_4_beta,
        model.inception_4b.bn1x1_4_running_mean,
        model.inception_4b.bn1x1_4_running_var,
        True,
    )
    var inc4b_b4_relu = relu(inc4b_b4_bn)
    var inc4b_out = concatenate_depthwise(
        inc4b_b1_relu, inc4b_b2_relu2, inc4b_b3_relu2, inc4b_b4_relu
    )
    # ---- Inception 4c (input: inc4b_out) ----
    var inc4c_b1_conv = conv2d(
        inc4b_out,
        model.inception_4c.conv1x1_1_weights,
        model.inception_4c.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc4c_b1_bn, _, _ = batch_norm2d(
        inc4c_b1_conv,
        model.inception_4c.bn1x1_1_gamma,
        model.inception_4c.bn1x1_1_beta,
        model.inception_4c.bn1x1_1_running_mean,
        model.inception_4c.bn1x1_1_running_var,
        True,
    )
    var inc4c_b1_relu = relu(inc4c_b1_bn)
    var inc4c_b2_conv1 = conv2d(
        inc4b_out,
        model.inception_4c.conv1x1_2_weights,
        model.inception_4c.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc4c_b2_bn1, _, _ = batch_norm2d(
        inc4c_b2_conv1,
        model.inception_4c.bn1x1_2_gamma,
        model.inception_4c.bn1x1_2_beta,
        model.inception_4c.bn1x1_2_running_mean,
        model.inception_4c.bn1x1_2_running_var,
        True,
    )
    var inc4c_b2_relu1 = relu(inc4c_b2_bn1)
    var inc4c_b2_conv2 = conv2d(
        inc4c_b2_relu1,
        model.inception_4c.conv3x3_weights,
        model.inception_4c.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc4c_b2_bn2, _, _ = batch_norm2d(
        inc4c_b2_conv2,
        model.inception_4c.bn3x3_gamma,
        model.inception_4c.bn3x3_beta,
        model.inception_4c.bn3x3_running_mean,
        model.inception_4c.bn3x3_running_var,
        True,
    )
    var inc4c_b2_relu2 = relu(inc4c_b2_bn2)
    var inc4c_b3_conv1 = conv2d(
        inc4b_out,
        model.inception_4c.conv1x1_3_weights,
        model.inception_4c.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc4c_b3_bn1, _, _ = batch_norm2d(
        inc4c_b3_conv1,
        model.inception_4c.bn1x1_3_gamma,
        model.inception_4c.bn1x1_3_beta,
        model.inception_4c.bn1x1_3_running_mean,
        model.inception_4c.bn1x1_3_running_var,
        True,
    )
    var inc4c_b3_relu1 = relu(inc4c_b3_bn1)
    var inc4c_b3_conv2 = conv2d(
        inc4c_b3_relu1,
        model.inception_4c.conv5x5_weights,
        model.inception_4c.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc4c_b3_bn2, _, _ = batch_norm2d(
        inc4c_b3_conv2,
        model.inception_4c.bn5x5_gamma,
        model.inception_4c.bn5x5_beta,
        model.inception_4c.bn5x5_running_mean,
        model.inception_4c.bn5x5_running_var,
        True,
    )
    var inc4c_b3_relu2 = relu(inc4c_b3_bn2)
    var inc4c_b4_pool = maxpool2d(inc4b_out, kernel_size=3, stride=1, padding=1)
    var inc4c_b4_conv = conv2d(
        inc4c_b4_pool,
        model.inception_4c.conv1x1_4_weights,
        model.inception_4c.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc4c_b4_bn, _, _ = batch_norm2d(
        inc4c_b4_conv,
        model.inception_4c.bn1x1_4_gamma,
        model.inception_4c.bn1x1_4_beta,
        model.inception_4c.bn1x1_4_running_mean,
        model.inception_4c.bn1x1_4_running_var,
        True,
    )
    var inc4c_b4_relu = relu(inc4c_b4_bn)
    var inc4c_out = concatenate_depthwise(
        inc4c_b1_relu, inc4c_b2_relu2, inc4c_b3_relu2, inc4c_b4_relu
    )
    # ---- Inception 4d (input: inc4c_out) ----
    var inc4d_b1_conv = conv2d(
        inc4c_out,
        model.inception_4d.conv1x1_1_weights,
        model.inception_4d.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc4d_b1_bn, _, _ = batch_norm2d(
        inc4d_b1_conv,
        model.inception_4d.bn1x1_1_gamma,
        model.inception_4d.bn1x1_1_beta,
        model.inception_4d.bn1x1_1_running_mean,
        model.inception_4d.bn1x1_1_running_var,
        True,
    )
    var inc4d_b1_relu = relu(inc4d_b1_bn)
    var inc4d_b2_conv1 = conv2d(
        inc4c_out,
        model.inception_4d.conv1x1_2_weights,
        model.inception_4d.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc4d_b2_bn1, _, _ = batch_norm2d(
        inc4d_b2_conv1,
        model.inception_4d.bn1x1_2_gamma,
        model.inception_4d.bn1x1_2_beta,
        model.inception_4d.bn1x1_2_running_mean,
        model.inception_4d.bn1x1_2_running_var,
        True,
    )
    var inc4d_b2_relu1 = relu(inc4d_b2_bn1)
    var inc4d_b2_conv2 = conv2d(
        inc4d_b2_relu1,
        model.inception_4d.conv3x3_weights,
        model.inception_4d.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc4d_b2_bn2, _, _ = batch_norm2d(
        inc4d_b2_conv2,
        model.inception_4d.bn3x3_gamma,
        model.inception_4d.bn3x3_beta,
        model.inception_4d.bn3x3_running_mean,
        model.inception_4d.bn3x3_running_var,
        True,
    )
    var inc4d_b2_relu2 = relu(inc4d_b2_bn2)
    var inc4d_b3_conv1 = conv2d(
        inc4c_out,
        model.inception_4d.conv1x1_3_weights,
        model.inception_4d.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc4d_b3_bn1, _, _ = batch_norm2d(
        inc4d_b3_conv1,
        model.inception_4d.bn1x1_3_gamma,
        model.inception_4d.bn1x1_3_beta,
        model.inception_4d.bn1x1_3_running_mean,
        model.inception_4d.bn1x1_3_running_var,
        True,
    )
    var inc4d_b3_relu1 = relu(inc4d_b3_bn1)
    var inc4d_b3_conv2 = conv2d(
        inc4d_b3_relu1,
        model.inception_4d.conv5x5_weights,
        model.inception_4d.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc4d_b3_bn2, _, _ = batch_norm2d(
        inc4d_b3_conv2,
        model.inception_4d.bn5x5_gamma,
        model.inception_4d.bn5x5_beta,
        model.inception_4d.bn5x5_running_mean,
        model.inception_4d.bn5x5_running_var,
        True,
    )
    var inc4d_b3_relu2 = relu(inc4d_b3_bn2)
    var inc4d_b4_pool = maxpool2d(inc4c_out, kernel_size=3, stride=1, padding=1)
    var inc4d_b4_conv = conv2d(
        inc4d_b4_pool,
        model.inception_4d.conv1x1_4_weights,
        model.inception_4d.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc4d_b4_bn, _, _ = batch_norm2d(
        inc4d_b4_conv,
        model.inception_4d.bn1x1_4_gamma,
        model.inception_4d.bn1x1_4_beta,
        model.inception_4d.bn1x1_4_running_mean,
        model.inception_4d.bn1x1_4_running_var,
        True,
    )
    var inc4d_b4_relu = relu(inc4d_b4_bn)
    var inc4d_out = concatenate_depthwise(
        inc4d_b1_relu, inc4d_b2_relu2, inc4d_b3_relu2, inc4d_b4_relu
    )
    # ---- Inception 4e (input: inc4d_out) ----
    var inc4e_b1_conv = conv2d(
        inc4d_out,
        model.inception_4e.conv1x1_1_weights,
        model.inception_4e.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc4e_b1_bn, _, _ = batch_norm2d(
        inc4e_b1_conv,
        model.inception_4e.bn1x1_1_gamma,
        model.inception_4e.bn1x1_1_beta,
        model.inception_4e.bn1x1_1_running_mean,
        model.inception_4e.bn1x1_1_running_var,
        True,
    )
    var inc4e_b1_relu = relu(inc4e_b1_bn)
    var inc4e_b2_conv1 = conv2d(
        inc4d_out,
        model.inception_4e.conv1x1_2_weights,
        model.inception_4e.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc4e_b2_bn1, _, _ = batch_norm2d(
        inc4e_b2_conv1,
        model.inception_4e.bn1x1_2_gamma,
        model.inception_4e.bn1x1_2_beta,
        model.inception_4e.bn1x1_2_running_mean,
        model.inception_4e.bn1x1_2_running_var,
        True,
    )
    var inc4e_b2_relu1 = relu(inc4e_b2_bn1)
    var inc4e_b2_conv2 = conv2d(
        inc4e_b2_relu1,
        model.inception_4e.conv3x3_weights,
        model.inception_4e.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc4e_b2_bn2, _, _ = batch_norm2d(
        inc4e_b2_conv2,
        model.inception_4e.bn3x3_gamma,
        model.inception_4e.bn3x3_beta,
        model.inception_4e.bn3x3_running_mean,
        model.inception_4e.bn3x3_running_var,
        True,
    )
    var inc4e_b2_relu2 = relu(inc4e_b2_bn2)
    var inc4e_b3_conv1 = conv2d(
        inc4d_out,
        model.inception_4e.conv1x1_3_weights,
        model.inception_4e.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc4e_b3_bn1, _, _ = batch_norm2d(
        inc4e_b3_conv1,
        model.inception_4e.bn1x1_3_gamma,
        model.inception_4e.bn1x1_3_beta,
        model.inception_4e.bn1x1_3_running_mean,
        model.inception_4e.bn1x1_3_running_var,
        True,
    )
    var inc4e_b3_relu1 = relu(inc4e_b3_bn1)
    var inc4e_b3_conv2 = conv2d(
        inc4e_b3_relu1,
        model.inception_4e.conv5x5_weights,
        model.inception_4e.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc4e_b3_bn2, _, _ = batch_norm2d(
        inc4e_b3_conv2,
        model.inception_4e.bn5x5_gamma,
        model.inception_4e.bn5x5_beta,
        model.inception_4e.bn5x5_running_mean,
        model.inception_4e.bn5x5_running_var,
        True,
    )
    var inc4e_b3_relu2 = relu(inc4e_b3_bn2)
    var inc4e_b4_pool = maxpool2d(inc4d_out, kernel_size=3, stride=1, padding=1)
    var inc4e_b4_conv = conv2d(
        inc4e_b4_pool,
        model.inception_4e.conv1x1_4_weights,
        model.inception_4e.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc4e_b4_bn, _, _ = batch_norm2d(
        inc4e_b4_conv,
        model.inception_4e.bn1x1_4_gamma,
        model.inception_4e.bn1x1_4_beta,
        model.inception_4e.bn1x1_4_running_mean,
        model.inception_4e.bn1x1_4_running_var,
        True,
    )
    var inc4e_b4_relu = relu(inc4e_b4_bn)
    var inc4e_out = concatenate_depthwise(
        inc4e_b1_relu, inc4e_b2_relu2, inc4e_b3_relu2, inc4e_b4_relu
    )
    var mid2_pool = maxpool2d(inc4e_out, kernel_size=3, stride=2, padding=1)
    # ---- Inception 5a (input: mid2_pool) ----
    var inc5a_b1_conv = conv2d(
        mid2_pool,
        model.inception_5a.conv1x1_1_weights,
        model.inception_5a.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc5a_b1_bn, _, _ = batch_norm2d(
        inc5a_b1_conv,
        model.inception_5a.bn1x1_1_gamma,
        model.inception_5a.bn1x1_1_beta,
        model.inception_5a.bn1x1_1_running_mean,
        model.inception_5a.bn1x1_1_running_var,
        True,
    )
    var inc5a_b1_relu = relu(inc5a_b1_bn)
    var inc5a_b2_conv1 = conv2d(
        mid2_pool,
        model.inception_5a.conv1x1_2_weights,
        model.inception_5a.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc5a_b2_bn1, _, _ = batch_norm2d(
        inc5a_b2_conv1,
        model.inception_5a.bn1x1_2_gamma,
        model.inception_5a.bn1x1_2_beta,
        model.inception_5a.bn1x1_2_running_mean,
        model.inception_5a.bn1x1_2_running_var,
        True,
    )
    var inc5a_b2_relu1 = relu(inc5a_b2_bn1)
    var inc5a_b2_conv2 = conv2d(
        inc5a_b2_relu1,
        model.inception_5a.conv3x3_weights,
        model.inception_5a.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc5a_b2_bn2, _, _ = batch_norm2d(
        inc5a_b2_conv2,
        model.inception_5a.bn3x3_gamma,
        model.inception_5a.bn3x3_beta,
        model.inception_5a.bn3x3_running_mean,
        model.inception_5a.bn3x3_running_var,
        True,
    )
    var inc5a_b2_relu2 = relu(inc5a_b2_bn2)
    var inc5a_b3_conv1 = conv2d(
        mid2_pool,
        model.inception_5a.conv1x1_3_weights,
        model.inception_5a.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc5a_b3_bn1, _, _ = batch_norm2d(
        inc5a_b3_conv1,
        model.inception_5a.bn1x1_3_gamma,
        model.inception_5a.bn1x1_3_beta,
        model.inception_5a.bn1x1_3_running_mean,
        model.inception_5a.bn1x1_3_running_var,
        True,
    )
    var inc5a_b3_relu1 = relu(inc5a_b3_bn1)
    var inc5a_b3_conv2 = conv2d(
        inc5a_b3_relu1,
        model.inception_5a.conv5x5_weights,
        model.inception_5a.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc5a_b3_bn2, _, _ = batch_norm2d(
        inc5a_b3_conv2,
        model.inception_5a.bn5x5_gamma,
        model.inception_5a.bn5x5_beta,
        model.inception_5a.bn5x5_running_mean,
        model.inception_5a.bn5x5_running_var,
        True,
    )
    var inc5a_b3_relu2 = relu(inc5a_b3_bn2)
    var inc5a_b4_pool = maxpool2d(mid2_pool, kernel_size=3, stride=1, padding=1)
    var inc5a_b4_conv = conv2d(
        inc5a_b4_pool,
        model.inception_5a.conv1x1_4_weights,
        model.inception_5a.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc5a_b4_bn, _, _ = batch_norm2d(
        inc5a_b4_conv,
        model.inception_5a.bn1x1_4_gamma,
        model.inception_5a.bn1x1_4_beta,
        model.inception_5a.bn1x1_4_running_mean,
        model.inception_5a.bn1x1_4_running_var,
        True,
    )
    var inc5a_b4_relu = relu(inc5a_b4_bn)
    var inc5a_out = concatenate_depthwise(
        inc5a_b1_relu, inc5a_b2_relu2, inc5a_b3_relu2, inc5a_b4_relu
    )
    # ---- Inception 5b (input: inc5a_out) ----
    var inc5b_b1_conv = conv2d(
        inc5a_out,
        model.inception_5b.conv1x1_1_weights,
        model.inception_5b.conv1x1_1_bias,
        stride=1,
        padding=0,
    )
    var inc5b_b1_bn, _, _ = batch_norm2d(
        inc5b_b1_conv,
        model.inception_5b.bn1x1_1_gamma,
        model.inception_5b.bn1x1_1_beta,
        model.inception_5b.bn1x1_1_running_mean,
        model.inception_5b.bn1x1_1_running_var,
        True,
    )
    var inc5b_b1_relu = relu(inc5b_b1_bn)
    var inc5b_b2_conv1 = conv2d(
        inc5a_out,
        model.inception_5b.conv1x1_2_weights,
        model.inception_5b.conv1x1_2_bias,
        stride=1,
        padding=0,
    )
    var inc5b_b2_bn1, _, _ = batch_norm2d(
        inc5b_b2_conv1,
        model.inception_5b.bn1x1_2_gamma,
        model.inception_5b.bn1x1_2_beta,
        model.inception_5b.bn1x1_2_running_mean,
        model.inception_5b.bn1x1_2_running_var,
        True,
    )
    var inc5b_b2_relu1 = relu(inc5b_b2_bn1)
    var inc5b_b2_conv2 = conv2d(
        inc5b_b2_relu1,
        model.inception_5b.conv3x3_weights,
        model.inception_5b.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var inc5b_b2_bn2, _, _ = batch_norm2d(
        inc5b_b2_conv2,
        model.inception_5b.bn3x3_gamma,
        model.inception_5b.bn3x3_beta,
        model.inception_5b.bn3x3_running_mean,
        model.inception_5b.bn3x3_running_var,
        True,
    )
    var inc5b_b2_relu2 = relu(inc5b_b2_bn2)
    var inc5b_b3_conv1 = conv2d(
        inc5a_out,
        model.inception_5b.conv1x1_3_weights,
        model.inception_5b.conv1x1_3_bias,
        stride=1,
        padding=0,
    )
    var inc5b_b3_bn1, _, _ = batch_norm2d(
        inc5b_b3_conv1,
        model.inception_5b.bn1x1_3_gamma,
        model.inception_5b.bn1x1_3_beta,
        model.inception_5b.bn1x1_3_running_mean,
        model.inception_5b.bn1x1_3_running_var,
        True,
    )
    var inc5b_b3_relu1 = relu(inc5b_b3_bn1)
    var inc5b_b3_conv2 = conv2d(
        inc5b_b3_relu1,
        model.inception_5b.conv5x5_weights,
        model.inception_5b.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var inc5b_b3_bn2, _, _ = batch_norm2d(
        inc5b_b3_conv2,
        model.inception_5b.bn5x5_gamma,
        model.inception_5b.bn5x5_beta,
        model.inception_5b.bn5x5_running_mean,
        model.inception_5b.bn5x5_running_var,
        True,
    )
    var inc5b_b3_relu2 = relu(inc5b_b3_bn2)
    var inc5b_b4_pool = maxpool2d(inc5a_out, kernel_size=3, stride=1, padding=1)
    var inc5b_b4_conv = conv2d(
        inc5b_b4_pool,
        model.inception_5b.conv1x1_4_weights,
        model.inception_5b.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var inc5b_b4_bn, _, _ = batch_norm2d(
        inc5b_b4_conv,
        model.inception_5b.bn1x1_4_gamma,
        model.inception_5b.bn1x1_4_beta,
        model.inception_5b.bn1x1_4_running_mean,
        model.inception_5b.bn1x1_4_running_var,
        True,
    )
    var inc5b_b4_relu = relu(inc5b_b4_bn)
    var inc5b_out = concatenate_depthwise(
        inc5b_b1_relu, inc5b_b2_relu2, inc5b_b3_relu2, inc5b_b4_relu
    )

    # ---- Global avg pool -> flatten -> dropout(0.4) -> linear -> CE ----
    var gap_out = global_avgpool2d(inc5b_out)  # (N, 1024, 1, 1)
    var flat_result = _flatten_gap(gap_out)  # (N, 1024)
    var flat_out = flat_result[0]
    var gap_shape = flat_result[
        1
    ]  # Captured for backward slice (#3184) to unflatten gradients
    var drop_result = dropout(flat_out, Float64(0.4), training=True)
    var drop_out = drop_result[0]
    var drop_mask = drop_result[
        1
    ]  # Captured for backward slice (#3184) to compute dropout gradients
    var logits = linear(drop_out, model.fc_weights, model.fc_bias)
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ============================================================================
    # BACKWARD PASS (classifier tail -> Inception 5b -> ... -> 3a -> initial block)
    # ============================================================================

    # Seed upstream gradient (1.0) — VGG train.mojo:203-206 pattern.
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output.set(0, Float32(1.0))
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    # Final FC backward -> grad_drop_out, grad_fc_weights, grad_fc_bias.
    var fc_grads = linear_backward(grad_logits, drop_out, model.fc_weights)
    var grad_drop_out = fc_grads.grad_input
    var grad_fc_weights = fc_grads.grad_weights
    var grad_fc_bias = fc_grads.grad_bias

    # Dropout backward.
    var grad_flat_out = dropout_backward(
        grad_drop_out, drop_mask, Float64(0.4)
    )

    # Unflatten (N, C) gradient back to (N, C, 1, 1).
    var grad_gap_out = _unflatten_gap_grad(grad_flat_out, gap_shape)

    # Global average pooling backward -> grad w.r.t. inc5b_out.
    var grad_inc5b_out = global_avgpool2d_backward(grad_gap_out, inc5b_out)

    # ---- Inception 5b backward (input: inc5a_out; splits: [384, 768, 896]) ----
    var inc5b_parts = split_with_indices(
        grad_inc5b_out, [384, 768, 896], axis=1
    )
    var grad_inc5b_b1_relu = inc5b_parts[0]
    var grad_inc5b_b2_relu2 = inc5b_parts[1]
    var grad_inc5b_b3_relu2 = inc5b_parts[2]
    var grad_inc5b_b4_relu = inc5b_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc5a_out).
    var grad_inc5b_b1_bn = relu_backward(grad_inc5b_b1_relu, inc5b_b1_bn)
    var inc5b_b1_bn_bwd = batch_norm2d_backward(
        grad_inc5b_b1_bn, inc5b_b1_conv,
        model.inception_5b.bn1x1_1_gamma,
        model.inception_5b.bn1x1_1_running_mean,
        model.inception_5b.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc5b_b1_conv = inc5b_b1_bn_bwd[0]
    var grad_inc5b_bn1x1_1_gamma = inc5b_b1_bn_bwd[1]
    var grad_inc5b_bn1x1_1_beta = inc5b_b1_bn_bwd[2]
    var inc5b_b1_conv_bwd = conv2d_backward(
        grad_inc5b_b1_conv, inc5a_out,
        model.inception_5b.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc5b_b1_in = inc5b_b1_conv_bwd.grad_input
    var grad_inc5b_conv1x1_1_weights = inc5b_b1_conv_bwd.grad_weights
    var grad_inc5b_conv1x1_1_bias = inc5b_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc5a_out).
    var grad_inc5b_b2_bn2 = relu_backward(grad_inc5b_b2_relu2, inc5b_b2_bn2)
    var inc5b_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc5b_b2_bn2, inc5b_b2_conv2,
        model.inception_5b.bn3x3_gamma,
        model.inception_5b.bn3x3_running_mean,
        model.inception_5b.bn3x3_running_var,
        training=True,
    )
    var grad_inc5b_b2_conv2 = inc5b_b2_bn2_bwd[0]
    var grad_inc5b_bn3x3_gamma = inc5b_b2_bn2_bwd[1]
    var grad_inc5b_bn3x3_beta = inc5b_b2_bn2_bwd[2]
    var inc5b_b2_conv2_bwd = conv2d_backward(
        grad_inc5b_b2_conv2, inc5b_b2_relu1,
        model.inception_5b.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc5b_b2_relu1 = inc5b_b2_conv2_bwd.grad_input
    var grad_inc5b_conv3x3_weights = inc5b_b2_conv2_bwd.grad_weights
    var grad_inc5b_conv3x3_bias = inc5b_b2_conv2_bwd.grad_bias

    var grad_inc5b_b2_bn1 = relu_backward(grad_inc5b_b2_relu1, inc5b_b2_bn1)
    var inc5b_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc5b_b2_bn1, inc5b_b2_conv1,
        model.inception_5b.bn1x1_2_gamma,
        model.inception_5b.bn1x1_2_running_mean,
        model.inception_5b.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc5b_b2_conv1 = inc5b_b2_bn1_bwd[0]
    var grad_inc5b_bn1x1_2_gamma = inc5b_b2_bn1_bwd[1]
    var grad_inc5b_bn1x1_2_beta = inc5b_b2_bn1_bwd[2]
    var inc5b_b2_conv1_bwd = conv2d_backward(
        grad_inc5b_b2_conv1, inc5a_out,
        model.inception_5b.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc5b_b2_in = inc5b_b2_conv1_bwd.grad_input
    var grad_inc5b_conv1x1_2_weights = inc5b_b2_conv1_bwd.grad_weights
    var grad_inc5b_conv1x1_2_bias = inc5b_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc5b_b3_bn2 = relu_backward(grad_inc5b_b3_relu2, inc5b_b3_bn2)
    var inc5b_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc5b_b3_bn2, inc5b_b3_conv2,
        model.inception_5b.bn5x5_gamma,
        model.inception_5b.bn5x5_running_mean,
        model.inception_5b.bn5x5_running_var,
        training=True,
    )
    var grad_inc5b_b3_conv2 = inc5b_b3_bn2_bwd[0]
    var grad_inc5b_bn5x5_gamma = inc5b_b3_bn2_bwd[1]
    var grad_inc5b_bn5x5_beta = inc5b_b3_bn2_bwd[2]
    var inc5b_b3_conv2_bwd = conv2d_backward(
        grad_inc5b_b3_conv2, inc5b_b3_relu1,
        model.inception_5b.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc5b_b3_relu1 = inc5b_b3_conv2_bwd.grad_input
    var grad_inc5b_conv5x5_weights = inc5b_b3_conv2_bwd.grad_weights
    var grad_inc5b_conv5x5_bias = inc5b_b3_conv2_bwd.grad_bias

    var grad_inc5b_b3_bn1 = relu_backward(grad_inc5b_b3_relu1, inc5b_b3_bn1)
    var inc5b_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc5b_b3_bn1, inc5b_b3_conv1,
        model.inception_5b.bn1x1_3_gamma,
        model.inception_5b.bn1x1_3_running_mean,
        model.inception_5b.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc5b_b3_conv1 = inc5b_b3_bn1_bwd[0]
    var grad_inc5b_bn1x1_3_gamma = inc5b_b3_bn1_bwd[1]
    var grad_inc5b_bn1x1_3_beta = inc5b_b3_bn1_bwd[2]
    var inc5b_b3_conv1_bwd = conv2d_backward(
        grad_inc5b_b3_conv1, inc5a_out,
        model.inception_5b.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc5b_b3_in = inc5b_b3_conv1_bwd.grad_input
    var grad_inc5b_conv1x1_3_weights = inc5b_b3_conv1_bwd.grad_weights
    var grad_inc5b_conv1x1_3_bias = inc5b_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc5a_out).
    var grad_inc5b_b4_bn = relu_backward(grad_inc5b_b4_relu, inc5b_b4_bn)
    var inc5b_b4_bn_bwd = batch_norm2d_backward(
        grad_inc5b_b4_bn, inc5b_b4_conv,
        model.inception_5b.bn1x1_4_gamma,
        model.inception_5b.bn1x1_4_running_mean,
        model.inception_5b.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc5b_b4_conv = inc5b_b4_bn_bwd[0]
    var grad_inc5b_bn1x1_4_gamma = inc5b_b4_bn_bwd[1]
    var grad_inc5b_bn1x1_4_beta = inc5b_b4_bn_bwd[2]
    var inc5b_b4_conv_bwd = conv2d_backward(
        grad_inc5b_b4_conv, inc5b_b4_pool,
        model.inception_5b.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc5b_b4_pool = inc5b_b4_conv_bwd.grad_input
    var grad_inc5b_conv1x1_4_weights = inc5b_b4_conv_bwd.grad_weights
    var grad_inc5b_conv1x1_4_bias = inc5b_b4_conv_bwd.grad_bias
    var grad_inc5b_b4_in = maxpool2d_backward(
        grad_inc5b_b4_pool, inc5a_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add (issue-mandated).
    var grad_inc5a_out = add(
        add(add(grad_inc5b_b1_in, grad_inc5b_b2_in), grad_inc5b_b3_in),
        grad_inc5b_b4_in,
    )

    # ---- inc5a backward (input: mid2_pool; splits: [256, 576, 704]) ----
    var inc5a_parts = split_with_indices(
        grad_inc5a_out, [256, 576, 704], axis=1
    )
    var grad_inc5a_b1_relu = inc5a_parts[0]
    var grad_inc5a_b2_relu2 = inc5a_parts[1]
    var grad_inc5a_b3_relu2 = inc5a_parts[2]
    var grad_inc5a_b4_relu = inc5a_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = mid2_pool).
    var grad_inc5a_b1_bn = relu_backward(grad_inc5a_b1_relu, inc5a_b1_bn)
    var inc5a_b1_bn_bwd = batch_norm2d_backward(
        grad_inc5a_b1_bn, inc5a_b1_conv,
        model.inc5a.bn1x1_1_gamma,
        model.inc5a.bn1x1_1_running_mean,
        model.inc5a.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc5a_b1_conv = inc5a_b1_bn_bwd[0]
    var grad_inc5a_bn1x1_1_gamma = inc5a_b1_bn_bwd[1]
    var grad_inc5a_bn1x1_1_beta = inc5a_b1_bn_bwd[2]
    var inc5a_b1_conv_bwd = conv2d_backward(
        grad_inc5a_b1_conv, mid2_pool,
        model.inc5a.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc5a_b1_in = inc5a_b1_conv_bwd.grad_input
    var grad_inc5a_conv1x1_1_weights = inc5a_b1_conv_bwd.grad_weights
    var grad_inc5a_conv1x1_1_bias = inc5a_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = mid2_pool).
    var grad_inc5a_b2_bn2 = relu_backward(grad_inc5a_b2_relu2, inc5a_b2_bn2)
    var inc5a_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc5a_b2_bn2, inc5a_b2_conv2,
        model.inc5a.bn3x3_gamma,
        model.inc5a.bn3x3_running_mean,
        model.inc5a.bn3x3_running_var,
        training=True,
    )
    var grad_inc5a_b2_conv2 = inc5a_b2_bn2_bwd[0]
    var grad_inc5a_bn3x3_gamma = inc5a_b2_bn2_bwd[1]
    var grad_inc5a_bn3x3_beta = inc5a_b2_bn2_bwd[2]
    var inc5a_b2_conv2_bwd = conv2d_backward(
        grad_inc5a_b2_conv2, inc5a_b2_relu1,
        model.inc5a.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc5a_b2_relu1 = inc5a_b2_conv2_bwd.grad_input
    var grad_inc5a_conv3x3_weights = inc5a_b2_conv2_bwd.grad_weights
    var grad_inc5a_conv3x3_bias = inc5a_b2_conv2_bwd.grad_bias

    var grad_inc5a_b2_bn1 = relu_backward(grad_inc5a_b2_relu1, inc5a_b2_bn1)
    var inc5a_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc5a_b2_bn1, inc5a_b2_conv1,
        model.inc5a.bn1x1_2_gamma,
        model.inc5a.bn1x1_2_running_mean,
        model.inc5a.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc5a_b2_conv1 = inc5a_b2_bn1_bwd[0]
    var grad_inc5a_bn1x1_2_gamma = inc5a_b2_bn1_bwd[1]
    var grad_inc5a_bn1x1_2_beta = inc5a_b2_bn1_bwd[2]
    var inc5a_b2_conv1_bwd = conv2d_backward(
        grad_inc5a_b2_conv1, mid2_pool,
        model.inc5a.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc5a_b2_in = inc5a_b2_conv1_bwd.grad_input
    var grad_inc5a_conv1x1_2_weights = inc5a_b2_conv1_bwd.grad_weights
    var grad_inc5a_conv1x1_2_bias = inc5a_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc5a_b3_bn2 = relu_backward(grad_inc5a_b3_relu2, inc5a_b3_bn2)
    var inc5a_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc5a_b3_bn2, inc5a_b3_conv2,
        model.inc5a.bn5x5_gamma,
        model.inc5a.bn5x5_running_mean,
        model.inc5a.bn5x5_running_var,
        training=True,
    )
    var grad_inc5a_b3_conv2 = inc5a_b3_bn2_bwd[0]
    var grad_inc5a_bn5x5_gamma = inc5a_b3_bn2_bwd[1]
    var grad_inc5a_bn5x5_beta = inc5a_b3_bn2_bwd[2]
    var inc5a_b3_conv2_bwd = conv2d_backward(
        grad_inc5a_b3_conv2, inc5a_b3_relu1,
        model.inc5a.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc5a_b3_relu1 = inc5a_b3_conv2_bwd.grad_input
    var grad_inc5a_conv5x5_weights = inc5a_b3_conv2_bwd.grad_weights
    var grad_inc5a_conv5x5_bias = inc5a_b3_conv2_bwd.grad_bias

    var grad_inc5a_b3_bn1 = relu_backward(grad_inc5a_b3_relu1, inc5a_b3_bn1)
    var inc5a_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc5a_b3_bn1, inc5a_b3_conv1,
        model.inc5a.bn1x1_3_gamma,
        model.inc5a.bn1x1_3_running_mean,
        model.inc5a.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc5a_b3_conv1 = inc5a_b3_bn1_bwd[0]
    var grad_inc5a_bn1x1_3_gamma = inc5a_b3_bn1_bwd[1]
    var grad_inc5a_bn1x1_3_beta = inc5a_b3_bn1_bwd[2]
    var inc5a_b3_conv1_bwd = conv2d_backward(
        grad_inc5a_b3_conv1, mid2_pool,
        model.inc5a.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc5a_b3_in = inc5a_b3_conv1_bwd.grad_input
    var grad_inc5a_conv1x1_3_weights = inc5a_b3_conv1_bwd.grad_weights
    var grad_inc5a_conv1x1_3_bias = inc5a_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = mid2_pool).
    var grad_inc5a_b4_bn = relu_backward(grad_inc5a_b4_relu, inc5a_b4_bn)
    var inc5a_b4_bn_bwd = batch_norm2d_backward(
        grad_inc5a_b4_bn, inc5a_b4_conv,
        model.inc5a.bn1x1_4_gamma,
        model.inc5a.bn1x1_4_running_mean,
        model.inc5a.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc5a_b4_conv = inc5a_b4_bn_bwd[0]
    var grad_inc5a_bn1x1_4_gamma = inc5a_b4_bn_bwd[1]
    var grad_inc5a_bn1x1_4_beta = inc5a_b4_bn_bwd[2]
    var inc5a_b4_conv_bwd = conv2d_backward(
        grad_inc5a_b4_conv, inc5a_b4_pool,
        model.inc5a.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc5a_b4_pool = inc5a_b4_conv_bwd.grad_input
    var grad_inc5a_conv1x1_4_weights = inc5a_b4_conv_bwd.grad_weights
    var grad_inc5a_conv1x1_4_bias = inc5a_b4_conv_bwd.grad_bias
    var grad_inc5a_b4_in = maxpool2d_backward(
        grad_inc5a_b4_pool, mid2_pool, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_mid2_pool = add(
        add(add(grad_inc5a_b1_in, grad_inc5a_b2_in), grad_inc5a_b3_in),
        grad_inc5a_b4_in,
    )

    # ---- inc4e backward (input: inc4d_out; splits: [256, 576, 704]) ----
    var inc4e_parts = split_with_indices(
        grad_inc4e_out, [256, 576, 704], axis=1
    )
    var grad_inc4e_b1_relu = inc4e_parts[0]
    var grad_inc4e_b2_relu2 = inc4e_parts[1]
    var grad_inc4e_b3_relu2 = inc4e_parts[2]
    var grad_inc4e_b4_relu = inc4e_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc4d_out).
    var grad_inc4e_b1_bn = relu_backward(grad_inc4e_b1_relu, inc4e_b1_bn)
    var inc4e_b1_bn_bwd = batch_norm2d_backward(
        grad_inc4e_b1_bn, inc4e_b1_conv,
        model.inc4e.bn1x1_1_gamma,
        model.inc4e.bn1x1_1_running_mean,
        model.inc4e.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc4e_b1_conv = inc4e_b1_bn_bwd[0]
    var grad_inc4e_bn1x1_1_gamma = inc4e_b1_bn_bwd[1]
    var grad_inc4e_bn1x1_1_beta = inc4e_b1_bn_bwd[2]
    var inc4e_b1_conv_bwd = conv2d_backward(
        grad_inc4e_b1_conv, inc4d_out,
        model.inc4e.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc4e_b1_in = inc4e_b1_conv_bwd.grad_input
    var grad_inc4e_conv1x1_1_weights = inc4e_b1_conv_bwd.grad_weights
    var grad_inc4e_conv1x1_1_bias = inc4e_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc4d_out).
    var grad_inc4e_b2_bn2 = relu_backward(grad_inc4e_b2_relu2, inc4e_b2_bn2)
    var inc4e_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc4e_b2_bn2, inc4e_b2_conv2,
        model.inc4e.bn3x3_gamma,
        model.inc4e.bn3x3_running_mean,
        model.inc4e.bn3x3_running_var,
        training=True,
    )
    var grad_inc4e_b2_conv2 = inc4e_b2_bn2_bwd[0]
    var grad_inc4e_bn3x3_gamma = inc4e_b2_bn2_bwd[1]
    var grad_inc4e_bn3x3_beta = inc4e_b2_bn2_bwd[2]
    var inc4e_b2_conv2_bwd = conv2d_backward(
        grad_inc4e_b2_conv2, inc4e_b2_relu1,
        model.inc4e.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc4e_b2_relu1 = inc4e_b2_conv2_bwd.grad_input
    var grad_inc4e_conv3x3_weights = inc4e_b2_conv2_bwd.grad_weights
    var grad_inc4e_conv3x3_bias = inc4e_b2_conv2_bwd.grad_bias

    var grad_inc4e_b2_bn1 = relu_backward(grad_inc4e_b2_relu1, inc4e_b2_bn1)
    var inc4e_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc4e_b2_bn1, inc4e_b2_conv1,
        model.inc4e.bn1x1_2_gamma,
        model.inc4e.bn1x1_2_running_mean,
        model.inc4e.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc4e_b2_conv1 = inc4e_b2_bn1_bwd[0]
    var grad_inc4e_bn1x1_2_gamma = inc4e_b2_bn1_bwd[1]
    var grad_inc4e_bn1x1_2_beta = inc4e_b2_bn1_bwd[2]
    var inc4e_b2_conv1_bwd = conv2d_backward(
        grad_inc4e_b2_conv1, inc4d_out,
        model.inc4e.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc4e_b2_in = inc4e_b2_conv1_bwd.grad_input
    var grad_inc4e_conv1x1_2_weights = inc4e_b2_conv1_bwd.grad_weights
    var grad_inc4e_conv1x1_2_bias = inc4e_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc4e_b3_bn2 = relu_backward(grad_inc4e_b3_relu2, inc4e_b3_bn2)
    var inc4e_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc4e_b3_bn2, inc4e_b3_conv2,
        model.inc4e.bn5x5_gamma,
        model.inc4e.bn5x5_running_mean,
        model.inc4e.bn5x5_running_var,
        training=True,
    )
    var grad_inc4e_b3_conv2 = inc4e_b3_bn2_bwd[0]
    var grad_inc4e_bn5x5_gamma = inc4e_b3_bn2_bwd[1]
    var grad_inc4e_bn5x5_beta = inc4e_b3_bn2_bwd[2]
    var inc4e_b3_conv2_bwd = conv2d_backward(
        grad_inc4e_b3_conv2, inc4e_b3_relu1,
        model.inc4e.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc4e_b3_relu1 = inc4e_b3_conv2_bwd.grad_input
    var grad_inc4e_conv5x5_weights = inc4e_b3_conv2_bwd.grad_weights
    var grad_inc4e_conv5x5_bias = inc4e_b3_conv2_bwd.grad_bias

    var grad_inc4e_b3_bn1 = relu_backward(grad_inc4e_b3_relu1, inc4e_b3_bn1)
    var inc4e_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc4e_b3_bn1, inc4e_b3_conv1,
        model.inc4e.bn1x1_3_gamma,
        model.inc4e.bn1x1_3_running_mean,
        model.inc4e.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc4e_b3_conv1 = inc4e_b3_bn1_bwd[0]
    var grad_inc4e_bn1x1_3_gamma = inc4e_b3_bn1_bwd[1]
    var grad_inc4e_bn1x1_3_beta = inc4e_b3_bn1_bwd[2]
    var inc4e_b3_conv1_bwd = conv2d_backward(
        grad_inc4e_b3_conv1, inc4d_out,
        model.inc4e.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc4e_b3_in = inc4e_b3_conv1_bwd.grad_input
    var grad_inc4e_conv1x1_3_weights = inc4e_b3_conv1_bwd.grad_weights
    var grad_inc4e_conv1x1_3_bias = inc4e_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc4d_out).
    var grad_inc4e_b4_bn = relu_backward(grad_inc4e_b4_relu, inc4e_b4_bn)
    var inc4e_b4_bn_bwd = batch_norm2d_backward(
        grad_inc4e_b4_bn, inc4e_b4_conv,
        model.inc4e.bn1x1_4_gamma,
        model.inc4e.bn1x1_4_running_mean,
        model.inc4e.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc4e_b4_conv = inc4e_b4_bn_bwd[0]
    var grad_inc4e_bn1x1_4_gamma = inc4e_b4_bn_bwd[1]
    var grad_inc4e_bn1x1_4_beta = inc4e_b4_bn_bwd[2]
    var inc4e_b4_conv_bwd = conv2d_backward(
        grad_inc4e_b4_conv, inc4e_b4_pool,
        model.inc4e.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc4e_b4_pool = inc4e_b4_conv_bwd.grad_input
    var grad_inc4e_conv1x1_4_weights = inc4e_b4_conv_bwd.grad_weights
    var grad_inc4e_conv1x1_4_bias = inc4e_b4_conv_bwd.grad_bias
    var grad_inc4e_b4_in = maxpool2d_backward(
        grad_inc4e_b4_pool, inc4d_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_inc4e_out = add(
        add(add(grad_inc4e_b1_in, grad_inc4e_b2_in), grad_inc4e_b3_in),
        grad_inc4e_b4_in,
    )

    # ---- inc4d backward (input: inc4c_out; splits: [112, 400, 464]) ----
    var inc4d_parts = split_with_indices(
        grad_inc4d_out, [112, 400, 464], axis=1
    )
    var grad_inc4d_b1_relu = inc4d_parts[0]
    var grad_inc4d_b2_relu2 = inc4d_parts[1]
    var grad_inc4d_b3_relu2 = inc4d_parts[2]
    var grad_inc4d_b4_relu = inc4d_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc4c_out).
    var grad_inc4d_b1_bn = relu_backward(grad_inc4d_b1_relu, inc4d_b1_bn)
    var inc4d_b1_bn_bwd = batch_norm2d_backward(
        grad_inc4d_b1_bn, inc4d_b1_conv,
        model.inc4d.bn1x1_1_gamma,
        model.inc4d.bn1x1_1_running_mean,
        model.inc4d.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc4d_b1_conv = inc4d_b1_bn_bwd[0]
    var grad_inc4d_bn1x1_1_gamma = inc4d_b1_bn_bwd[1]
    var grad_inc4d_bn1x1_1_beta = inc4d_b1_bn_bwd[2]
    var inc4d_b1_conv_bwd = conv2d_backward(
        grad_inc4d_b1_conv, inc4c_out,
        model.inc4d.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc4d_b1_in = inc4d_b1_conv_bwd.grad_input
    var grad_inc4d_conv1x1_1_weights = inc4d_b1_conv_bwd.grad_weights
    var grad_inc4d_conv1x1_1_bias = inc4d_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc4c_out).
    var grad_inc4d_b2_bn2 = relu_backward(grad_inc4d_b2_relu2, inc4d_b2_bn2)
    var inc4d_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc4d_b2_bn2, inc4d_b2_conv2,
        model.inc4d.bn3x3_gamma,
        model.inc4d.bn3x3_running_mean,
        model.inc4d.bn3x3_running_var,
        training=True,
    )
    var grad_inc4d_b2_conv2 = inc4d_b2_bn2_bwd[0]
    var grad_inc4d_bn3x3_gamma = inc4d_b2_bn2_bwd[1]
    var grad_inc4d_bn3x3_beta = inc4d_b2_bn2_bwd[2]
    var inc4d_b2_conv2_bwd = conv2d_backward(
        grad_inc4d_b2_conv2, inc4d_b2_relu1,
        model.inc4d.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc4d_b2_relu1 = inc4d_b2_conv2_bwd.grad_input
    var grad_inc4d_conv3x3_weights = inc4d_b2_conv2_bwd.grad_weights
    var grad_inc4d_conv3x3_bias = inc4d_b2_conv2_bwd.grad_bias

    var grad_inc4d_b2_bn1 = relu_backward(grad_inc4d_b2_relu1, inc4d_b2_bn1)
    var inc4d_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc4d_b2_bn1, inc4d_b2_conv1,
        model.inc4d.bn1x1_2_gamma,
        model.inc4d.bn1x1_2_running_mean,
        model.inc4d.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc4d_b2_conv1 = inc4d_b2_bn1_bwd[0]
    var grad_inc4d_bn1x1_2_gamma = inc4d_b2_bn1_bwd[1]
    var grad_inc4d_bn1x1_2_beta = inc4d_b2_bn1_bwd[2]
    var inc4d_b2_conv1_bwd = conv2d_backward(
        grad_inc4d_b2_conv1, inc4c_out,
        model.inc4d.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc4d_b2_in = inc4d_b2_conv1_bwd.grad_input
    var grad_inc4d_conv1x1_2_weights = inc4d_b2_conv1_bwd.grad_weights
    var grad_inc4d_conv1x1_2_bias = inc4d_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc4d_b3_bn2 = relu_backward(grad_inc4d_b3_relu2, inc4d_b3_bn2)
    var inc4d_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc4d_b3_bn2, inc4d_b3_conv2,
        model.inc4d.bn5x5_gamma,
        model.inc4d.bn5x5_running_mean,
        model.inc4d.bn5x5_running_var,
        training=True,
    )
    var grad_inc4d_b3_conv2 = inc4d_b3_bn2_bwd[0]
    var grad_inc4d_bn5x5_gamma = inc4d_b3_bn2_bwd[1]
    var grad_inc4d_bn5x5_beta = inc4d_b3_bn2_bwd[2]
    var inc4d_b3_conv2_bwd = conv2d_backward(
        grad_inc4d_b3_conv2, inc4d_b3_relu1,
        model.inc4d.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc4d_b3_relu1 = inc4d_b3_conv2_bwd.grad_input
    var grad_inc4d_conv5x5_weights = inc4d_b3_conv2_bwd.grad_weights
    var grad_inc4d_conv5x5_bias = inc4d_b3_conv2_bwd.grad_bias

    var grad_inc4d_b3_bn1 = relu_backward(grad_inc4d_b3_relu1, inc4d_b3_bn1)
    var inc4d_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc4d_b3_bn1, inc4d_b3_conv1,
        model.inc4d.bn1x1_3_gamma,
        model.inc4d.bn1x1_3_running_mean,
        model.inc4d.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc4d_b3_conv1 = inc4d_b3_bn1_bwd[0]
    var grad_inc4d_bn1x1_3_gamma = inc4d_b3_bn1_bwd[1]
    var grad_inc4d_bn1x1_3_beta = inc4d_b3_bn1_bwd[2]
    var inc4d_b3_conv1_bwd = conv2d_backward(
        grad_inc4d_b3_conv1, inc4c_out,
        model.inc4d.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc4d_b3_in = inc4d_b3_conv1_bwd.grad_input
    var grad_inc4d_conv1x1_3_weights = inc4d_b3_conv1_bwd.grad_weights
    var grad_inc4d_conv1x1_3_bias = inc4d_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc4c_out).
    var grad_inc4d_b4_bn = relu_backward(grad_inc4d_b4_relu, inc4d_b4_bn)
    var inc4d_b4_bn_bwd = batch_norm2d_backward(
        grad_inc4d_b4_bn, inc4d_b4_conv,
        model.inc4d.bn1x1_4_gamma,
        model.inc4d.bn1x1_4_running_mean,
        model.inc4d.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc4d_b4_conv = inc4d_b4_bn_bwd[0]
    var grad_inc4d_bn1x1_4_gamma = inc4d_b4_bn_bwd[1]
    var grad_inc4d_bn1x1_4_beta = inc4d_b4_bn_bwd[2]
    var inc4d_b4_conv_bwd = conv2d_backward(
        grad_inc4d_b4_conv, inc4d_b4_pool,
        model.inc4d.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc4d_b4_pool = inc4d_b4_conv_bwd.grad_input
    var grad_inc4d_conv1x1_4_weights = inc4d_b4_conv_bwd.grad_weights
    var grad_inc4d_conv1x1_4_bias = inc4d_b4_conv_bwd.grad_bias
    var grad_inc4d_b4_in = maxpool2d_backward(
        grad_inc4d_b4_pool, inc4c_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_inc4d_out = add(
        add(add(grad_inc4d_b1_in, grad_inc4d_b2_in), grad_inc4d_b3_in),
        grad_inc4d_b4_in,
    )

    # ---- inc4c backward (input: inc4b_out; splits: [128, 384, 448]) ----
    var inc4c_parts = split_with_indices(
        grad_inc4c_out, [128, 384, 448], axis=1
    )
    var grad_inc4c_b1_relu = inc4c_parts[0]
    var grad_inc4c_b2_relu2 = inc4c_parts[1]
    var grad_inc4c_b3_relu2 = inc4c_parts[2]
    var grad_inc4c_b4_relu = inc4c_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc4b_out).
    var grad_inc4c_b1_bn = relu_backward(grad_inc4c_b1_relu, inc4c_b1_bn)
    var inc4c_b1_bn_bwd = batch_norm2d_backward(
        grad_inc4c_b1_bn, inc4c_b1_conv,
        model.inc4c.bn1x1_1_gamma,
        model.inc4c.bn1x1_1_running_mean,
        model.inc4c.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc4c_b1_conv = inc4c_b1_bn_bwd[0]
    var grad_inc4c_bn1x1_1_gamma = inc4c_b1_bn_bwd[1]
    var grad_inc4c_bn1x1_1_beta = inc4c_b1_bn_bwd[2]
    var inc4c_b1_conv_bwd = conv2d_backward(
        grad_inc4c_b1_conv, inc4b_out,
        model.inc4c.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc4c_b1_in = inc4c_b1_conv_bwd.grad_input
    var grad_inc4c_conv1x1_1_weights = inc4c_b1_conv_bwd.grad_weights
    var grad_inc4c_conv1x1_1_bias = inc4c_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc4b_out).
    var grad_inc4c_b2_bn2 = relu_backward(grad_inc4c_b2_relu2, inc4c_b2_bn2)
    var inc4c_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc4c_b2_bn2, inc4c_b2_conv2,
        model.inc4c.bn3x3_gamma,
        model.inc4c.bn3x3_running_mean,
        model.inc4c.bn3x3_running_var,
        training=True,
    )
    var grad_inc4c_b2_conv2 = inc4c_b2_bn2_bwd[0]
    var grad_inc4c_bn3x3_gamma = inc4c_b2_bn2_bwd[1]
    var grad_inc4c_bn3x3_beta = inc4c_b2_bn2_bwd[2]
    var inc4c_b2_conv2_bwd = conv2d_backward(
        grad_inc4c_b2_conv2, inc4c_b2_relu1,
        model.inc4c.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc4c_b2_relu1 = inc4c_b2_conv2_bwd.grad_input
    var grad_inc4c_conv3x3_weights = inc4c_b2_conv2_bwd.grad_weights
    var grad_inc4c_conv3x3_bias = inc4c_b2_conv2_bwd.grad_bias

    var grad_inc4c_b2_bn1 = relu_backward(grad_inc4c_b2_relu1, inc4c_b2_bn1)
    var inc4c_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc4c_b2_bn1, inc4c_b2_conv1,
        model.inc4c.bn1x1_2_gamma,
        model.inc4c.bn1x1_2_running_mean,
        model.inc4c.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc4c_b2_conv1 = inc4c_b2_bn1_bwd[0]
    var grad_inc4c_bn1x1_2_gamma = inc4c_b2_bn1_bwd[1]
    var grad_inc4c_bn1x1_2_beta = inc4c_b2_bn1_bwd[2]
    var inc4c_b2_conv1_bwd = conv2d_backward(
        grad_inc4c_b2_conv1, inc4b_out,
        model.inc4c.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc4c_b2_in = inc4c_b2_conv1_bwd.grad_input
    var grad_inc4c_conv1x1_2_weights = inc4c_b2_conv1_bwd.grad_weights
    var grad_inc4c_conv1x1_2_bias = inc4c_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc4c_b3_bn2 = relu_backward(grad_inc4c_b3_relu2, inc4c_b3_bn2)
    var inc4c_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc4c_b3_bn2, inc4c_b3_conv2,
        model.inc4c.bn5x5_gamma,
        model.inc4c.bn5x5_running_mean,
        model.inc4c.bn5x5_running_var,
        training=True,
    )
    var grad_inc4c_b3_conv2 = inc4c_b3_bn2_bwd[0]
    var grad_inc4c_bn5x5_gamma = inc4c_b3_bn2_bwd[1]
    var grad_inc4c_bn5x5_beta = inc4c_b3_bn2_bwd[2]
    var inc4c_b3_conv2_bwd = conv2d_backward(
        grad_inc4c_b3_conv2, inc4c_b3_relu1,
        model.inc4c.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc4c_b3_relu1 = inc4c_b3_conv2_bwd.grad_input
    var grad_inc4c_conv5x5_weights = inc4c_b3_conv2_bwd.grad_weights
    var grad_inc4c_conv5x5_bias = inc4c_b3_conv2_bwd.grad_bias

    var grad_inc4c_b3_bn1 = relu_backward(grad_inc4c_b3_relu1, inc4c_b3_bn1)
    var inc4c_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc4c_b3_bn1, inc4c_b3_conv1,
        model.inc4c.bn1x1_3_gamma,
        model.inc4c.bn1x1_3_running_mean,
        model.inc4c.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc4c_b3_conv1 = inc4c_b3_bn1_bwd[0]
    var grad_inc4c_bn1x1_3_gamma = inc4c_b3_bn1_bwd[1]
    var grad_inc4c_bn1x1_3_beta = inc4c_b3_bn1_bwd[2]
    var inc4c_b3_conv1_bwd = conv2d_backward(
        grad_inc4c_b3_conv1, inc4b_out,
        model.inc4c.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc4c_b3_in = inc4c_b3_conv1_bwd.grad_input
    var grad_inc4c_conv1x1_3_weights = inc4c_b3_conv1_bwd.grad_weights
    var grad_inc4c_conv1x1_3_bias = inc4c_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc4b_out).
    var grad_inc4c_b4_bn = relu_backward(grad_inc4c_b4_relu, inc4c_b4_bn)
    var inc4c_b4_bn_bwd = batch_norm2d_backward(
        grad_inc4c_b4_bn, inc4c_b4_conv,
        model.inc4c.bn1x1_4_gamma,
        model.inc4c.bn1x1_4_running_mean,
        model.inc4c.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc4c_b4_conv = inc4c_b4_bn_bwd[0]
    var grad_inc4c_bn1x1_4_gamma = inc4c_b4_bn_bwd[1]
    var grad_inc4c_bn1x1_4_beta = inc4c_b4_bn_bwd[2]
    var inc4c_b4_conv_bwd = conv2d_backward(
        grad_inc4c_b4_conv, inc4c_b4_pool,
        model.inc4c.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc4c_b4_pool = inc4c_b4_conv_bwd.grad_input
    var grad_inc4c_conv1x1_4_weights = inc4c_b4_conv_bwd.grad_weights
    var grad_inc4c_conv1x1_4_bias = inc4c_b4_conv_bwd.grad_bias
    var grad_inc4c_b4_in = maxpool2d_backward(
        grad_inc4c_b4_pool, inc4b_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_inc4c_out = add(
        add(add(grad_inc4c_b1_in, grad_inc4c_b2_in), grad_inc4c_b3_in),
        grad_inc4c_b4_in,
    )

    # ---- inc4b backward (input: inc4a_out; splits: [160, 384, 448]) ----
    var inc4b_parts = split_with_indices(
        grad_inc4b_out, [160, 384, 448], axis=1
    )
    var grad_inc4b_b1_relu = inc4b_parts[0]
    var grad_inc4b_b2_relu2 = inc4b_parts[1]
    var grad_inc4b_b3_relu2 = inc4b_parts[2]
    var grad_inc4b_b4_relu = inc4b_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc4a_out).
    var grad_inc4b_b1_bn = relu_backward(grad_inc4b_b1_relu, inc4b_b1_bn)
    var inc4b_b1_bn_bwd = batch_norm2d_backward(
        grad_inc4b_b1_bn, inc4b_b1_conv,
        model.inc4b.bn1x1_1_gamma,
        model.inc4b.bn1x1_1_running_mean,
        model.inc4b.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc4b_b1_conv = inc4b_b1_bn_bwd[0]
    var grad_inc4b_bn1x1_1_gamma = inc4b_b1_bn_bwd[1]
    var grad_inc4b_bn1x1_1_beta = inc4b_b1_bn_bwd[2]
    var inc4b_b1_conv_bwd = conv2d_backward(
        grad_inc4b_b1_conv, inc4a_out,
        model.inc4b.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc4b_b1_in = inc4b_b1_conv_bwd.grad_input
    var grad_inc4b_conv1x1_1_weights = inc4b_b1_conv_bwd.grad_weights
    var grad_inc4b_conv1x1_1_bias = inc4b_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc4a_out).
    var grad_inc4b_b2_bn2 = relu_backward(grad_inc4b_b2_relu2, inc4b_b2_bn2)
    var inc4b_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc4b_b2_bn2, inc4b_b2_conv2,
        model.inc4b.bn3x3_gamma,
        model.inc4b.bn3x3_running_mean,
        model.inc4b.bn3x3_running_var,
        training=True,
    )
    var grad_inc4b_b2_conv2 = inc4b_b2_bn2_bwd[0]
    var grad_inc4b_bn3x3_gamma = inc4b_b2_bn2_bwd[1]
    var grad_inc4b_bn3x3_beta = inc4b_b2_bn2_bwd[2]
    var inc4b_b2_conv2_bwd = conv2d_backward(
        grad_inc4b_b2_conv2, inc4b_b2_relu1,
        model.inc4b.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc4b_b2_relu1 = inc4b_b2_conv2_bwd.grad_input
    var grad_inc4b_conv3x3_weights = inc4b_b2_conv2_bwd.grad_weights
    var grad_inc4b_conv3x3_bias = inc4b_b2_conv2_bwd.grad_bias

    var grad_inc4b_b2_bn1 = relu_backward(grad_inc4b_b2_relu1, inc4b_b2_bn1)
    var inc4b_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc4b_b2_bn1, inc4b_b2_conv1,
        model.inc4b.bn1x1_2_gamma,
        model.inc4b.bn1x1_2_running_mean,
        model.inc4b.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc4b_b2_conv1 = inc4b_b2_bn1_bwd[0]
    var grad_inc4b_bn1x1_2_gamma = inc4b_b2_bn1_bwd[1]
    var grad_inc4b_bn1x1_2_beta = inc4b_b2_bn1_bwd[2]
    var inc4b_b2_conv1_bwd = conv2d_backward(
        grad_inc4b_b2_conv1, inc4a_out,
        model.inc4b.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc4b_b2_in = inc4b_b2_conv1_bwd.grad_input
    var grad_inc4b_conv1x1_2_weights = inc4b_b2_conv1_bwd.grad_weights
    var grad_inc4b_conv1x1_2_bias = inc4b_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc4b_b3_bn2 = relu_backward(grad_inc4b_b3_relu2, inc4b_b3_bn2)
    var inc4b_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc4b_b3_bn2, inc4b_b3_conv2,
        model.inc4b.bn5x5_gamma,
        model.inc4b.bn5x5_running_mean,
        model.inc4b.bn5x5_running_var,
        training=True,
    )
    var grad_inc4b_b3_conv2 = inc4b_b3_bn2_bwd[0]
    var grad_inc4b_bn5x5_gamma = inc4b_b3_bn2_bwd[1]
    var grad_inc4b_bn5x5_beta = inc4b_b3_bn2_bwd[2]
    var inc4b_b3_conv2_bwd = conv2d_backward(
        grad_inc4b_b3_conv2, inc4b_b3_relu1,
        model.inc4b.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc4b_b3_relu1 = inc4b_b3_conv2_bwd.grad_input
    var grad_inc4b_conv5x5_weights = inc4b_b3_conv2_bwd.grad_weights
    var grad_inc4b_conv5x5_bias = inc4b_b3_conv2_bwd.grad_bias

    var grad_inc4b_b3_bn1 = relu_backward(grad_inc4b_b3_relu1, inc4b_b3_bn1)
    var inc4b_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc4b_b3_bn1, inc4b_b3_conv1,
        model.inc4b.bn1x1_3_gamma,
        model.inc4b.bn1x1_3_running_mean,
        model.inc4b.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc4b_b3_conv1 = inc4b_b3_bn1_bwd[0]
    var grad_inc4b_bn1x1_3_gamma = inc4b_b3_bn1_bwd[1]
    var grad_inc4b_bn1x1_3_beta = inc4b_b3_bn1_bwd[2]
    var inc4b_b3_conv1_bwd = conv2d_backward(
        grad_inc4b_b3_conv1, inc4a_out,
        model.inc4b.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc4b_b3_in = inc4b_b3_conv1_bwd.grad_input
    var grad_inc4b_conv1x1_3_weights = inc4b_b3_conv1_bwd.grad_weights
    var grad_inc4b_conv1x1_3_bias = inc4b_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc4a_out).
    var grad_inc4b_b4_bn = relu_backward(grad_inc4b_b4_relu, inc4b_b4_bn)
    var inc4b_b4_bn_bwd = batch_norm2d_backward(
        grad_inc4b_b4_bn, inc4b_b4_conv,
        model.inc4b.bn1x1_4_gamma,
        model.inc4b.bn1x1_4_running_mean,
        model.inc4b.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc4b_b4_conv = inc4b_b4_bn_bwd[0]
    var grad_inc4b_bn1x1_4_gamma = inc4b_b4_bn_bwd[1]
    var grad_inc4b_bn1x1_4_beta = inc4b_b4_bn_bwd[2]
    var inc4b_b4_conv_bwd = conv2d_backward(
        grad_inc4b_b4_conv, inc4b_b4_pool,
        model.inc4b.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc4b_b4_pool = inc4b_b4_conv_bwd.grad_input
    var grad_inc4b_conv1x1_4_weights = inc4b_b4_conv_bwd.grad_weights
    var grad_inc4b_conv1x1_4_bias = inc4b_b4_conv_bwd.grad_bias
    var grad_inc4b_b4_in = maxpool2d_backward(
        grad_inc4b_b4_pool, inc4a_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_inc4b_out = add(
        add(add(grad_inc4b_b1_in, grad_inc4b_b2_in), grad_inc4b_b3_in),
        grad_inc4b_b4_in,
    )

    # ---- inc4a backward (input: mid1_pool; splits: [192, 400, 448]) ----
    var inc4a_parts = split_with_indices(
        grad_inc4a_out, [192, 400, 448], axis=1
    )
    var grad_inc4a_b1_relu = inc4a_parts[0]
    var grad_inc4a_b2_relu2 = inc4a_parts[1]
    var grad_inc4a_b3_relu2 = inc4a_parts[2]
    var grad_inc4a_b4_relu = inc4a_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = mid1_pool).
    var grad_inc4a_b1_bn = relu_backward(grad_inc4a_b1_relu, inc4a_b1_bn)
    var inc4a_b1_bn_bwd = batch_norm2d_backward(
        grad_inc4a_b1_bn, inc4a_b1_conv,
        model.inc4a.bn1x1_1_gamma,
        model.inc4a.bn1x1_1_running_mean,
        model.inc4a.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc4a_b1_conv = inc4a_b1_bn_bwd[0]
    var grad_inc4a_bn1x1_1_gamma = inc4a_b1_bn_bwd[1]
    var grad_inc4a_bn1x1_1_beta = inc4a_b1_bn_bwd[2]
    var inc4a_b1_conv_bwd = conv2d_backward(
        grad_inc4a_b1_conv, mid1_pool,
        model.inc4a.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc4a_b1_in = inc4a_b1_conv_bwd.grad_input
    var grad_inc4a_conv1x1_1_weights = inc4a_b1_conv_bwd.grad_weights
    var grad_inc4a_conv1x1_1_bias = inc4a_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = mid1_pool).
    var grad_inc4a_b2_bn2 = relu_backward(grad_inc4a_b2_relu2, inc4a_b2_bn2)
    var inc4a_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc4a_b2_bn2, inc4a_b2_conv2,
        model.inc4a.bn3x3_gamma,
        model.inc4a.bn3x3_running_mean,
        model.inc4a.bn3x3_running_var,
        training=True,
    )
    var grad_inc4a_b2_conv2 = inc4a_b2_bn2_bwd[0]
    var grad_inc4a_bn3x3_gamma = inc4a_b2_bn2_bwd[1]
    var grad_inc4a_bn3x3_beta = inc4a_b2_bn2_bwd[2]
    var inc4a_b2_conv2_bwd = conv2d_backward(
        grad_inc4a_b2_conv2, inc4a_b2_relu1,
        model.inc4a.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc4a_b2_relu1 = inc4a_b2_conv2_bwd.grad_input
    var grad_inc4a_conv3x3_weights = inc4a_b2_conv2_bwd.grad_weights
    var grad_inc4a_conv3x3_bias = inc4a_b2_conv2_bwd.grad_bias

    var grad_inc4a_b2_bn1 = relu_backward(grad_inc4a_b2_relu1, inc4a_b2_bn1)
    var inc4a_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc4a_b2_bn1, inc4a_b2_conv1,
        model.inc4a.bn1x1_2_gamma,
        model.inc4a.bn1x1_2_running_mean,
        model.inc4a.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc4a_b2_conv1 = inc4a_b2_bn1_bwd[0]
    var grad_inc4a_bn1x1_2_gamma = inc4a_b2_bn1_bwd[1]
    var grad_inc4a_bn1x1_2_beta = inc4a_b2_bn1_bwd[2]
    var inc4a_b2_conv1_bwd = conv2d_backward(
        grad_inc4a_b2_conv1, mid1_pool,
        model.inc4a.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc4a_b2_in = inc4a_b2_conv1_bwd.grad_input
    var grad_inc4a_conv1x1_2_weights = inc4a_b2_conv1_bwd.grad_weights
    var grad_inc4a_conv1x1_2_bias = inc4a_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc4a_b3_bn2 = relu_backward(grad_inc4a_b3_relu2, inc4a_b3_bn2)
    var inc4a_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc4a_b3_bn2, inc4a_b3_conv2,
        model.inc4a.bn5x5_gamma,
        model.inc4a.bn5x5_running_mean,
        model.inc4a.bn5x5_running_var,
        training=True,
    )
    var grad_inc4a_b3_conv2 = inc4a_b3_bn2_bwd[0]
    var grad_inc4a_bn5x5_gamma = inc4a_b3_bn2_bwd[1]
    var grad_inc4a_bn5x5_beta = inc4a_b3_bn2_bwd[2]
    var inc4a_b3_conv2_bwd = conv2d_backward(
        grad_inc4a_b3_conv2, inc4a_b3_relu1,
        model.inc4a.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc4a_b3_relu1 = inc4a_b3_conv2_bwd.grad_input
    var grad_inc4a_conv5x5_weights = inc4a_b3_conv2_bwd.grad_weights
    var grad_inc4a_conv5x5_bias = inc4a_b3_conv2_bwd.grad_bias

    var grad_inc4a_b3_bn1 = relu_backward(grad_inc4a_b3_relu1, inc4a_b3_bn1)
    var inc4a_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc4a_b3_bn1, inc4a_b3_conv1,
        model.inc4a.bn1x1_3_gamma,
        model.inc4a.bn1x1_3_running_mean,
        model.inc4a.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc4a_b3_conv1 = inc4a_b3_bn1_bwd[0]
    var grad_inc4a_bn1x1_3_gamma = inc4a_b3_bn1_bwd[1]
    var grad_inc4a_bn1x1_3_beta = inc4a_b3_bn1_bwd[2]
    var inc4a_b3_conv1_bwd = conv2d_backward(
        grad_inc4a_b3_conv1, mid1_pool,
        model.inc4a.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc4a_b3_in = inc4a_b3_conv1_bwd.grad_input
    var grad_inc4a_conv1x1_3_weights = inc4a_b3_conv1_bwd.grad_weights
    var grad_inc4a_conv1x1_3_bias = inc4a_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = mid1_pool).
    var grad_inc4a_b4_bn = relu_backward(grad_inc4a_b4_relu, inc4a_b4_bn)
    var inc4a_b4_bn_bwd = batch_norm2d_backward(
        grad_inc4a_b4_bn, inc4a_b4_conv,
        model.inc4a.bn1x1_4_gamma,
        model.inc4a.bn1x1_4_running_mean,
        model.inc4a.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc4a_b4_conv = inc4a_b4_bn_bwd[0]
    var grad_inc4a_bn1x1_4_gamma = inc4a_b4_bn_bwd[1]
    var grad_inc4a_bn1x1_4_beta = inc4a_b4_bn_bwd[2]
    var inc4a_b4_conv_bwd = conv2d_backward(
        grad_inc4a_b4_conv, inc4a_b4_pool,
        model.inc4a.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc4a_b4_pool = inc4a_b4_conv_bwd.grad_input
    var grad_inc4a_conv1x1_4_weights = inc4a_b4_conv_bwd.grad_weights
    var grad_inc4a_conv1x1_4_bias = inc4a_b4_conv_bwd.grad_bias
    var grad_inc4a_b4_in = maxpool2d_backward(
        grad_inc4a_b4_pool, mid1_pool, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_mid1_pool = add(
        add(add(grad_inc4a_b1_in, grad_inc4a_b2_in), grad_inc4a_b3_in),
        grad_inc4a_b4_in,
    )

    # ---- inc3b backward (input: inc3a_out; splits: [128, 320, 416]) ----
    var inc3b_parts = split_with_indices(
        grad_inc3b_out, [128, 320, 416], axis=1
    )
    var grad_inc3b_b1_relu = inc3b_parts[0]
    var grad_inc3b_b2_relu2 = inc3b_parts[1]
    var grad_inc3b_b3_relu2 = inc3b_parts[2]
    var grad_inc3b_b4_relu = inc3b_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = inc3a_out).
    var grad_inc3b_b1_bn = relu_backward(grad_inc3b_b1_relu, inc3b_b1_bn)
    var inc3b_b1_bn_bwd = batch_norm2d_backward(
        grad_inc3b_b1_bn, inc3b_b1_conv,
        model.inc3b.bn1x1_1_gamma,
        model.inc3b.bn1x1_1_running_mean,
        model.inc3b.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc3b_b1_conv = inc3b_b1_bn_bwd[0]
    var grad_inc3b_bn1x1_1_gamma = inc3b_b1_bn_bwd[1]
    var grad_inc3b_bn1x1_1_beta = inc3b_b1_bn_bwd[2]
    var inc3b_b1_conv_bwd = conv2d_backward(
        grad_inc3b_b1_conv, inc3a_out,
        model.inc3b.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc3b_b1_in = inc3b_b1_conv_bwd.grad_input
    var grad_inc3b_conv1x1_1_weights = inc3b_b1_conv_bwd.grad_weights
    var grad_inc3b_conv1x1_1_bias = inc3b_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = inc3a_out).
    var grad_inc3b_b2_bn2 = relu_backward(grad_inc3b_b2_relu2, inc3b_b2_bn2)
    var inc3b_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc3b_b2_bn2, inc3b_b2_conv2,
        model.inc3b.bn3x3_gamma,
        model.inc3b.bn3x3_running_mean,
        model.inc3b.bn3x3_running_var,
        training=True,
    )
    var grad_inc3b_b2_conv2 = inc3b_b2_bn2_bwd[0]
    var grad_inc3b_bn3x3_gamma = inc3b_b2_bn2_bwd[1]
    var grad_inc3b_bn3x3_beta = inc3b_b2_bn2_bwd[2]
    var inc3b_b2_conv2_bwd = conv2d_backward(
        grad_inc3b_b2_conv2, inc3b_b2_relu1,
        model.inc3b.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc3b_b2_relu1 = inc3b_b2_conv2_bwd.grad_input
    var grad_inc3b_conv3x3_weights = inc3b_b2_conv2_bwd.grad_weights
    var grad_inc3b_conv3x3_bias = inc3b_b2_conv2_bwd.grad_bias

    var grad_inc3b_b2_bn1 = relu_backward(grad_inc3b_b2_relu1, inc3b_b2_bn1)
    var inc3b_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc3b_b2_bn1, inc3b_b2_conv1,
        model.inc3b.bn1x1_2_gamma,
        model.inc3b.bn1x1_2_running_mean,
        model.inc3b.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc3b_b2_conv1 = inc3b_b2_bn1_bwd[0]
    var grad_inc3b_bn1x1_2_gamma = inc3b_b2_bn1_bwd[1]
    var grad_inc3b_bn1x1_2_beta = inc3b_b2_bn1_bwd[2]
    var inc3b_b2_conv1_bwd = conv2d_backward(
        grad_inc3b_b2_conv1, inc3a_out,
        model.inc3b.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc3b_b2_in = inc3b_b2_conv1_bwd.grad_input
    var grad_inc3b_conv1x1_2_weights = inc3b_b2_conv1_bwd.grad_weights
    var grad_inc3b_conv1x1_2_bias = inc3b_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc3b_b3_bn2 = relu_backward(grad_inc3b_b3_relu2, inc3b_b3_bn2)
    var inc3b_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc3b_b3_bn2, inc3b_b3_conv2,
        model.inc3b.bn5x5_gamma,
        model.inc3b.bn5x5_running_mean,
        model.inc3b.bn5x5_running_var,
        training=True,
    )
    var grad_inc3b_b3_conv2 = inc3b_b3_bn2_bwd[0]
    var grad_inc3b_bn5x5_gamma = inc3b_b3_bn2_bwd[1]
    var grad_inc3b_bn5x5_beta = inc3b_b3_bn2_bwd[2]
    var inc3b_b3_conv2_bwd = conv2d_backward(
        grad_inc3b_b3_conv2, inc3b_b3_relu1,
        model.inc3b.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc3b_b3_relu1 = inc3b_b3_conv2_bwd.grad_input
    var grad_inc3b_conv5x5_weights = inc3b_b3_conv2_bwd.grad_weights
    var grad_inc3b_conv5x5_bias = inc3b_b3_conv2_bwd.grad_bias

    var grad_inc3b_b3_bn1 = relu_backward(grad_inc3b_b3_relu1, inc3b_b3_bn1)
    var inc3b_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc3b_b3_bn1, inc3b_b3_conv1,
        model.inc3b.bn1x1_3_gamma,
        model.inc3b.bn1x1_3_running_mean,
        model.inc3b.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc3b_b3_conv1 = inc3b_b3_bn1_bwd[0]
    var grad_inc3b_bn1x1_3_gamma = inc3b_b3_bn1_bwd[1]
    var grad_inc3b_bn1x1_3_beta = inc3b_b3_bn1_bwd[2]
    var inc3b_b3_conv1_bwd = conv2d_backward(
        grad_inc3b_b3_conv1, inc3a_out,
        model.inc3b.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc3b_b3_in = inc3b_b3_conv1_bwd.grad_input
    var grad_inc3b_conv1x1_3_weights = inc3b_b3_conv1_bwd.grad_weights
    var grad_inc3b_conv1x1_3_bias = inc3b_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = inc3a_out).
    var grad_inc3b_b4_bn = relu_backward(grad_inc3b_b4_relu, inc3b_b4_bn)
    var inc3b_b4_bn_bwd = batch_norm2d_backward(
        grad_inc3b_b4_bn, inc3b_b4_conv,
        model.inc3b.bn1x1_4_gamma,
        model.inc3b.bn1x1_4_running_mean,
        model.inc3b.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc3b_b4_conv = inc3b_b4_bn_bwd[0]
    var grad_inc3b_bn1x1_4_gamma = inc3b_b4_bn_bwd[1]
    var grad_inc3b_bn1x1_4_beta = inc3b_b4_bn_bwd[2]
    var inc3b_b4_conv_bwd = conv2d_backward(
        grad_inc3b_b4_conv, inc3b_b4_pool,
        model.inc3b.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc3b_b4_pool = inc3b_b4_conv_bwd.grad_input
    var grad_inc3b_conv1x1_4_weights = inc3b_b4_conv_bwd.grad_weights
    var grad_inc3b_conv1x1_4_bias = inc3b_b4_conv_bwd.grad_bias
    var grad_inc3b_b4_in = maxpool2d_backward(
        grad_inc3b_b4_pool, inc3a_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_inc3b_out = add(
        add(add(grad_inc3b_b1_in, grad_inc3b_b2_in), grad_inc3b_b3_in),
        grad_inc3b_b4_in,
    )

    # ---- inc3a backward (input: init_pool_out; splits: [64, 192, 224]) ----
    var inc3a_parts = split_with_indices(
        grad_inc3a_out, [64, 192, 224], axis=1
    )
    var grad_inc3a_b1_relu = inc3a_parts[0]
    var grad_inc3a_b2_relu2 = inc3a_parts[1]
    var grad_inc3a_b3_relu2 = inc3a_parts[2]
    var grad_inc3a_b4_relu = inc3a_parts[3]

    # Branch 1: ReLU -> BN -> Conv1x1 (input = init_pool_out).
    var grad_inc3a_b1_bn = relu_backward(grad_inc3a_b1_relu, inc3a_b1_bn)
    var inc3a_b1_bn_bwd = batch_norm2d_backward(
        grad_inc3a_b1_bn, inc3a_b1_conv,
        model.inc3a.bn1x1_1_gamma,
        model.inc3a.bn1x1_1_running_mean,
        model.inc3a.bn1x1_1_running_var,
        training=True,
    )
    var grad_inc3a_b1_conv = inc3a_b1_bn_bwd[0]
    var grad_inc3a_bn1x1_1_gamma = inc3a_b1_bn_bwd[1]
    var grad_inc3a_bn1x1_1_beta = inc3a_b1_bn_bwd[2]
    var inc3a_b1_conv_bwd = conv2d_backward(
        grad_inc3a_b1_conv, init_pool_out,
        model.inc3a.conv1x1_1_weights, stride=1, padding=0,
    )
    var grad_inc3a_b1_in = inc3a_b1_conv_bwd.grad_input
    var grad_inc3a_conv1x1_1_weights = inc3a_b1_conv_bwd.grad_weights
    var grad_inc3a_conv1x1_1_bias = inc3a_b1_conv_bwd.grad_bias

    # Branch 2: ReLU2 -> BN2 -> Conv3x3 -> ReLU1 -> BN1 -> Conv1x1_2 (input = init_pool_out).
    var grad_inc3a_b2_bn2 = relu_backward(grad_inc3a_b2_relu2, inc3a_b2_bn2)
    var inc3a_b2_bn2_bwd = batch_norm2d_backward(
        grad_inc3a_b2_bn2, inc3a_b2_conv2,
        model.inc3a.bn3x3_gamma,
        model.inc3a.bn3x3_running_mean,
        model.inc3a.bn3x3_running_var,
        training=True,
    )
    var grad_inc3a_b2_conv2 = inc3a_b2_bn2_bwd[0]
    var grad_inc3a_bn3x3_gamma = inc3a_b2_bn2_bwd[1]
    var grad_inc3a_bn3x3_beta = inc3a_b2_bn2_bwd[2]
    var inc3a_b2_conv2_bwd = conv2d_backward(
        grad_inc3a_b2_conv2, inc3a_b2_relu1,
        model.inc3a.conv3x3_weights, stride=1, padding=1,
    )
    var grad_inc3a_b2_relu1 = inc3a_b2_conv2_bwd.grad_input
    var grad_inc3a_conv3x3_weights = inc3a_b2_conv2_bwd.grad_weights
    var grad_inc3a_conv3x3_bias = inc3a_b2_conv2_bwd.grad_bias

    var grad_inc3a_b2_bn1 = relu_backward(grad_inc3a_b2_relu1, inc3a_b2_bn1)
    var inc3a_b2_bn1_bwd = batch_norm2d_backward(
        grad_inc3a_b2_bn1, inc3a_b2_conv1,
        model.inc3a.bn1x1_2_gamma,
        model.inc3a.bn1x1_2_running_mean,
        model.inc3a.bn1x1_2_running_var,
        training=True,
    )
    var grad_inc3a_b2_conv1 = inc3a_b2_bn1_bwd[0]
    var grad_inc3a_bn1x1_2_gamma = inc3a_b2_bn1_bwd[1]
    var grad_inc3a_bn1x1_2_beta = inc3a_b2_bn1_bwd[2]
    var inc3a_b2_conv1_bwd = conv2d_backward(
        grad_inc3a_b2_conv1, init_pool_out,
        model.inc3a.conv1x1_2_weights, stride=1, padding=0,
    )
    var grad_inc3a_b2_in = inc3a_b2_conv1_bwd.grad_input
    var grad_inc3a_conv1x1_2_weights = inc3a_b2_conv1_bwd.grad_weights
    var grad_inc3a_conv1x1_2_bias = inc3a_b2_conv1_bwd.grad_bias

    # Branch 3: ReLU2 -> BN5x5 -> Conv5x5 -> ReLU1 -> BN1x1_3 -> Conv1x1_3.
    var grad_inc3a_b3_bn2 = relu_backward(grad_inc3a_b3_relu2, inc3a_b3_bn2)
    var inc3a_b3_bn2_bwd = batch_norm2d_backward(
        grad_inc3a_b3_bn2, inc3a_b3_conv2,
        model.inc3a.bn5x5_gamma,
        model.inc3a.bn5x5_running_mean,
        model.inc3a.bn5x5_running_var,
        training=True,
    )
    var grad_inc3a_b3_conv2 = inc3a_b3_bn2_bwd[0]
    var grad_inc3a_bn5x5_gamma = inc3a_b3_bn2_bwd[1]
    var grad_inc3a_bn5x5_beta = inc3a_b3_bn2_bwd[2]
    var inc3a_b3_conv2_bwd = conv2d_backward(
        grad_inc3a_b3_conv2, inc3a_b3_relu1,
        model.inc3a.conv5x5_weights, stride=1, padding=2,
    )
    var grad_inc3a_b3_relu1 = inc3a_b3_conv2_bwd.grad_input
    var grad_inc3a_conv5x5_weights = inc3a_b3_conv2_bwd.grad_weights
    var grad_inc3a_conv5x5_bias = inc3a_b3_conv2_bwd.grad_bias

    var grad_inc3a_b3_bn1 = relu_backward(grad_inc3a_b3_relu1, inc3a_b3_bn1)
    var inc3a_b3_bn1_bwd = batch_norm2d_backward(
        grad_inc3a_b3_bn1, inc3a_b3_conv1,
        model.inc3a.bn1x1_3_gamma,
        model.inc3a.bn1x1_3_running_mean,
        model.inc3a.bn1x1_3_running_var,
        training=True,
    )
    var grad_inc3a_b3_conv1 = inc3a_b3_bn1_bwd[0]
    var grad_inc3a_bn1x1_3_gamma = inc3a_b3_bn1_bwd[1]
    var grad_inc3a_bn1x1_3_beta = inc3a_b3_bn1_bwd[2]
    var inc3a_b3_conv1_bwd = conv2d_backward(
        grad_inc3a_b3_conv1, init_pool_out,
        model.inc3a.conv1x1_3_weights, stride=1, padding=0,
    )
    var grad_inc3a_b3_in = inc3a_b3_conv1_bwd.grad_input
    var grad_inc3a_conv1x1_3_weights = inc3a_b3_conv1_bwd.grad_weights
    var grad_inc3a_conv1x1_3_bias = inc3a_b3_conv1_bwd.grad_bias

    # Branch 4: ReLU -> BN -> Conv1x1_4 -> MaxPool (input = init_pool_out).
    var grad_inc3a_b4_bn = relu_backward(grad_inc3a_b4_relu, inc3a_b4_bn)
    var inc3a_b4_bn_bwd = batch_norm2d_backward(
        grad_inc3a_b4_bn, inc3a_b4_conv,
        model.inc3a.bn1x1_4_gamma,
        model.inc3a.bn1x1_4_running_mean,
        model.inc3a.bn1x1_4_running_var,
        training=True,
    )
    var grad_inc3a_b4_conv = inc3a_b4_bn_bwd[0]
    var grad_inc3a_bn1x1_4_gamma = inc3a_b4_bn_bwd[1]
    var grad_inc3a_bn1x1_4_beta = inc3a_b4_bn_bwd[2]
    var inc3a_b4_conv_bwd = conv2d_backward(
        grad_inc3a_b4_conv, inc3a_b4_pool,
        model.inc3a.conv1x1_4_weights, stride=1, padding=0,
    )
    var grad_inc3a_b4_pool = inc3a_b4_conv_bwd.grad_input
    var grad_inc3a_conv1x1_4_weights = inc3a_b4_conv_bwd.grad_weights
    var grad_inc3a_conv1x1_4_bias = inc3a_b4_conv_bwd.grad_bias
    var grad_inc3a_b4_in = maxpool2d_backward(
        grad_inc3a_b4_pool, init_pool_out, 3, stride=1, padding=1
    )

    # Combine four branch input-gradients via core.arithmetic.add.
    var grad_init_pool_out = add(
        add(add(grad_inc3a_b1_in, grad_inc3a_b2_in), grad_inc3a_b3_in),
        grad_inc3a_b4_in,
    )


    # After all 5a..4a blocks reduce, undo mid2_pool = maxpool2d(inc4e_out, k=3, s=2, p=1).
    var grad_inc4e_out = maxpool2d_backward(
        grad_mid2_pool, inc4e_out, 3, stride=2, padding=1
    )

    # After 4a produces grad_mid1_pool, undo mid1_pool = maxpool2d(inc3b_out, k=3, s=2, p=1).
    var grad_inc3b_out = maxpool2d_backward(
        grad_mid1_pool, inc3b_out, 3, stride=2, padding=1
    )

    # MaxPool backward: undo init_pool_out = maxpool2d(init_relu_out, k=3, s=2, p=1).
    var grad_init_relu_out = maxpool2d_backward(
        grad_init_pool_out, init_relu_out, 3, stride=2, padding=1
    )
    # ReLU backward.
    var grad_init_bn_out = relu_backward(grad_init_relu_out, init_bn_out)
    # BatchNorm backward.
    var init_bn_bwd = batch_norm2d_backward(
        grad_init_bn_out, init_conv_out,
        model.initial_bn_gamma,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        training=True,
    )
    var grad_init_conv_out = init_bn_bwd[0]
    var grad_initial_bn_gamma = init_bn_bwd[1]
    var grad_initial_bn_beta = init_bn_bwd[2]
    # Conv backward (input gradient discarded — no upstream).
    var init_conv_bwd = conv2d_backward(
        grad_init_conv_out, input,
        model.initial_conv_weights, stride=1, padding=1,
    )
    var _ = init_conv_bwd.grad_input
    var grad_initial_conv_weights = init_conv_bwd.grad_weights
    var grad_initial_conv_bias = init_conv_bwd.grad_bias

    # ============================================================================
    # PARAMETER UPDATE (SGD with Momentum) — exactly 222 calls in initialize_velocities order.
    # ============================================================================
    from projectodyssey.training.optimizers import sgd_momentum_update_inplace

    # Initial block (velocities[0..3]).
    sgd_momentum_update_inplace(model.initial_conv_weights, grad_initial_conv_weights, velocities[0], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.initial_conv_bias, grad_initial_conv_bias, velocities[1], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.initial_bn_gamma, grad_initial_bn_gamma, velocities[2], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.initial_bn_beta, grad_initial_bn_beta, velocities[3], Float64(learning_rate), Float64(momentum))

    # ---- Inception 3a (velocities[4..27]) ----
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_1_weights, grad_inc3a_conv1x1_1_weights, velocities[4], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_1_bias, grad_inc3a_conv1x1_1_bias, velocities[5], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_1_gamma, grad_inc3a_bn1x1_1_gamma, velocities[6], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_1_beta, grad_inc3a_bn1x1_1_beta, velocities[7], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_2_weights, grad_inc3a_conv1x1_2_weights, velocities[8], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_2_bias, grad_inc3a_conv1x1_2_bias, velocities[9], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_2_gamma, grad_inc3a_bn1x1_2_gamma, velocities[10], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_2_beta, grad_inc3a_bn1x1_2_beta, velocities[11], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv3x3_weights, grad_inc3a_conv3x3_weights, velocities[12], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv3x3_bias, grad_inc3a_conv3x3_bias, velocities[13], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn3x3_gamma, grad_inc3a_bn3x3_gamma, velocities[14], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn3x3_beta, grad_inc3a_bn3x3_beta, velocities[15], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_3_weights, grad_inc3a_conv1x1_3_weights, velocities[16], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_3_bias, grad_inc3a_conv1x1_3_bias, velocities[17], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_3_gamma, grad_inc3a_bn1x1_3_gamma, velocities[18], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_3_beta, grad_inc3a_bn1x1_3_beta, velocities[19], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv5x5_weights, grad_inc3a_conv5x5_weights, velocities[20], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv5x5_bias, grad_inc3a_conv5x5_bias, velocities[21], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn5x5_gamma, grad_inc3a_bn5x5_gamma, velocities[22], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn5x5_beta, grad_inc3a_bn5x5_beta, velocities[23], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_4_weights, grad_inc3a_conv1x1_4_weights, velocities[24], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.conv1x1_4_bias, grad_inc3a_conv1x1_4_bias, velocities[25], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_4_gamma, grad_inc3a_bn1x1_4_gamma, velocities[26], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3a.bn1x1_4_beta, grad_inc3a_bn1x1_4_beta, velocities[27], Float64(learning_rate), Float64(momentum))

    # ---- Inception 3b (velocities[28..51]) ----
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_1_weights, grad_inc3b_conv1x1_1_weights, velocities[28], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_1_bias, grad_inc3b_conv1x1_1_bias, velocities[29], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_1_gamma, grad_inc3b_bn1x1_1_gamma, velocities[30], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_1_beta, grad_inc3b_bn1x1_1_beta, velocities[31], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_2_weights, grad_inc3b_conv1x1_2_weights, velocities[32], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_2_bias, grad_inc3b_conv1x1_2_bias, velocities[33], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_2_gamma, grad_inc3b_bn1x1_2_gamma, velocities[34], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_2_beta, grad_inc3b_bn1x1_2_beta, velocities[35], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv3x3_weights, grad_inc3b_conv3x3_weights, velocities[36], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv3x3_bias, grad_inc3b_conv3x3_bias, velocities[37], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn3x3_gamma, grad_inc3b_bn3x3_gamma, velocities[38], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn3x3_beta, grad_inc3b_bn3x3_beta, velocities[39], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_3_weights, grad_inc3b_conv1x1_3_weights, velocities[40], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_3_bias, grad_inc3b_conv1x1_3_bias, velocities[41], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_3_gamma, grad_inc3b_bn1x1_3_gamma, velocities[42], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_3_beta, grad_inc3b_bn1x1_3_beta, velocities[43], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv5x5_weights, grad_inc3b_conv5x5_weights, velocities[44], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv5x5_bias, grad_inc3b_conv5x5_bias, velocities[45], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn5x5_gamma, grad_inc3b_bn5x5_gamma, velocities[46], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn5x5_beta, grad_inc3b_bn5x5_beta, velocities[47], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_4_weights, grad_inc3b_conv1x1_4_weights, velocities[48], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.conv1x1_4_bias, grad_inc3b_conv1x1_4_bias, velocities[49], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_4_gamma, grad_inc3b_bn1x1_4_gamma, velocities[50], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_3b.bn1x1_4_beta, grad_inc3b_bn1x1_4_beta, velocities[51], Float64(learning_rate), Float64(momentum))

    # ---- Inception 4a (velocities[52..75]) ----
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_1_weights, grad_inc4a_conv1x1_1_weights, velocities[52], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_1_bias, grad_inc4a_conv1x1_1_bias, velocities[53], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_1_gamma, grad_inc4a_bn1x1_1_gamma, velocities[54], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_1_beta, grad_inc4a_bn1x1_1_beta, velocities[55], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_2_weights, grad_inc4a_conv1x1_2_weights, velocities[56], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_2_bias, grad_inc4a_conv1x1_2_bias, velocities[57], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_2_gamma, grad_inc4a_bn1x1_2_gamma, velocities[58], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_2_beta, grad_inc4a_bn1x1_2_beta, velocities[59], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv3x3_weights, grad_inc4a_conv3x3_weights, velocities[60], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv3x3_bias, grad_inc4a_conv3x3_bias, velocities[61], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn3x3_gamma, grad_inc4a_bn3x3_gamma, velocities[62], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn3x3_beta, grad_inc4a_bn3x3_beta, velocities[63], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_3_weights, grad_inc4a_conv1x1_3_weights, velocities[64], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_3_bias, grad_inc4a_conv1x1_3_bias, velocities[65], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_3_gamma, grad_inc4a_bn1x1_3_gamma, velocities[66], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_3_beta, grad_inc4a_bn1x1_3_beta, velocities[67], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv5x5_weights, grad_inc4a_conv5x5_weights, velocities[68], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv5x5_bias, grad_inc4a_conv5x5_bias, velocities[69], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn5x5_gamma, grad_inc4a_bn5x5_gamma, velocities[70], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn5x5_beta, grad_inc4a_bn5x5_beta, velocities[71], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_4_weights, grad_inc4a_conv1x1_4_weights, velocities[72], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.conv1x1_4_bias, grad_inc4a_conv1x1_4_bias, velocities[73], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_4_gamma, grad_inc4a_bn1x1_4_gamma, velocities[74], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4a.bn1x1_4_beta, grad_inc4a_bn1x1_4_beta, velocities[75], Float64(learning_rate), Float64(momentum))

    # ---- Inception 4b (velocities[76..99]) ----
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_1_weights, grad_inc4b_conv1x1_1_weights, velocities[76], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_1_bias, grad_inc4b_conv1x1_1_bias, velocities[77], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_1_gamma, grad_inc4b_bn1x1_1_gamma, velocities[78], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_1_beta, grad_inc4b_bn1x1_1_beta, velocities[79], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_2_weights, grad_inc4b_conv1x1_2_weights, velocities[80], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_2_bias, grad_inc4b_conv1x1_2_bias, velocities[81], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_2_gamma, grad_inc4b_bn1x1_2_gamma, velocities[82], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_2_beta, grad_inc4b_bn1x1_2_beta, velocities[83], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv3x3_weights, grad_inc4b_conv3x3_weights, velocities[84], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv3x3_bias, grad_inc4b_conv3x3_bias, velocities[85], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn3x3_gamma, grad_inc4b_bn3x3_gamma, velocities[86], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn3x3_beta, grad_inc4b_bn3x3_beta, velocities[87], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_3_weights, grad_inc4b_conv1x1_3_weights, velocities[88], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_3_bias, grad_inc4b_conv1x1_3_bias, velocities[89], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_3_gamma, grad_inc4b_bn1x1_3_gamma, velocities[90], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_3_beta, grad_inc4b_bn1x1_3_beta, velocities[91], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv5x5_weights, grad_inc4b_conv5x5_weights, velocities[92], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv5x5_bias, grad_inc4b_conv5x5_bias, velocities[93], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn5x5_gamma, grad_inc4b_bn5x5_gamma, velocities[94], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn5x5_beta, grad_inc4b_bn5x5_beta, velocities[95], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_4_weights, grad_inc4b_conv1x1_4_weights, velocities[96], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.conv1x1_4_bias, grad_inc4b_conv1x1_4_bias, velocities[97], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_4_gamma, grad_inc4b_bn1x1_4_gamma, velocities[98], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4b.bn1x1_4_beta, grad_inc4b_bn1x1_4_beta, velocities[99], Float64(learning_rate), Float64(momentum))

    # ---- Inception 4c (velocities[100..123]) ----
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_1_weights, grad_inc4c_conv1x1_1_weights, velocities[100], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_1_bias, grad_inc4c_conv1x1_1_bias, velocities[101], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_1_gamma, grad_inc4c_bn1x1_1_gamma, velocities[102], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_1_beta, grad_inc4c_bn1x1_1_beta, velocities[103], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_2_weights, grad_inc4c_conv1x1_2_weights, velocities[104], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_2_bias, grad_inc4c_conv1x1_2_bias, velocities[105], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_2_gamma, grad_inc4c_bn1x1_2_gamma, velocities[106], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_2_beta, grad_inc4c_bn1x1_2_beta, velocities[107], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv3x3_weights, grad_inc4c_conv3x3_weights, velocities[108], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv3x3_bias, grad_inc4c_conv3x3_bias, velocities[109], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn3x3_gamma, grad_inc4c_bn3x3_gamma, velocities[110], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn3x3_beta, grad_inc4c_bn3x3_beta, velocities[111], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_3_weights, grad_inc4c_conv1x1_3_weights, velocities[112], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_3_bias, grad_inc4c_conv1x1_3_bias, velocities[113], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_3_gamma, grad_inc4c_bn1x1_3_gamma, velocities[114], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_3_beta, grad_inc4c_bn1x1_3_beta, velocities[115], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv5x5_weights, grad_inc4c_conv5x5_weights, velocities[116], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv5x5_bias, grad_inc4c_conv5x5_bias, velocities[117], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn5x5_gamma, grad_inc4c_bn5x5_gamma, velocities[118], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn5x5_beta, grad_inc4c_bn5x5_beta, velocities[119], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_4_weights, grad_inc4c_conv1x1_4_weights, velocities[120], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.conv1x1_4_bias, grad_inc4c_conv1x1_4_bias, velocities[121], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_4_gamma, grad_inc4c_bn1x1_4_gamma, velocities[122], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4c.bn1x1_4_beta, grad_inc4c_bn1x1_4_beta, velocities[123], Float64(learning_rate), Float64(momentum))

    # ---- Inception 4d (velocities[124..147]) ----
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_1_weights, grad_inc4d_conv1x1_1_weights, velocities[124], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_1_bias, grad_inc4d_conv1x1_1_bias, velocities[125], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_1_gamma, grad_inc4d_bn1x1_1_gamma, velocities[126], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_1_beta, grad_inc4d_bn1x1_1_beta, velocities[127], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_2_weights, grad_inc4d_conv1x1_2_weights, velocities[128], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_2_bias, grad_inc4d_conv1x1_2_bias, velocities[129], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_2_gamma, grad_inc4d_bn1x1_2_gamma, velocities[130], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_2_beta, grad_inc4d_bn1x1_2_beta, velocities[131], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv3x3_weights, grad_inc4d_conv3x3_weights, velocities[132], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv3x3_bias, grad_inc4d_conv3x3_bias, velocities[133], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn3x3_gamma, grad_inc4d_bn3x3_gamma, velocities[134], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn3x3_beta, grad_inc4d_bn3x3_beta, velocities[135], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_3_weights, grad_inc4d_conv1x1_3_weights, velocities[136], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_3_bias, grad_inc4d_conv1x1_3_bias, velocities[137], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_3_gamma, grad_inc4d_bn1x1_3_gamma, velocities[138], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_3_beta, grad_inc4d_bn1x1_3_beta, velocities[139], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv5x5_weights, grad_inc4d_conv5x5_weights, velocities[140], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv5x5_bias, grad_inc4d_conv5x5_bias, velocities[141], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn5x5_gamma, grad_inc4d_bn5x5_gamma, velocities[142], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn5x5_beta, grad_inc4d_bn5x5_beta, velocities[143], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_4_weights, grad_inc4d_conv1x1_4_weights, velocities[144], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.conv1x1_4_bias, grad_inc4d_conv1x1_4_bias, velocities[145], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_4_gamma, grad_inc4d_bn1x1_4_gamma, velocities[146], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4d.bn1x1_4_beta, grad_inc4d_bn1x1_4_beta, velocities[147], Float64(learning_rate), Float64(momentum))

    # ---- Inception 4e (velocities[148..171]) ----
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_1_weights, grad_inc4e_conv1x1_1_weights, velocities[148], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_1_bias, grad_inc4e_conv1x1_1_bias, velocities[149], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_1_gamma, grad_inc4e_bn1x1_1_gamma, velocities[150], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_1_beta, grad_inc4e_bn1x1_1_beta, velocities[151], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_2_weights, grad_inc4e_conv1x1_2_weights, velocities[152], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_2_bias, grad_inc4e_conv1x1_2_bias, velocities[153], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_2_gamma, grad_inc4e_bn1x1_2_gamma, velocities[154], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_2_beta, grad_inc4e_bn1x1_2_beta, velocities[155], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv3x3_weights, grad_inc4e_conv3x3_weights, velocities[156], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv3x3_bias, grad_inc4e_conv3x3_bias, velocities[157], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn3x3_gamma, grad_inc4e_bn3x3_gamma, velocities[158], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn3x3_beta, grad_inc4e_bn3x3_beta, velocities[159], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_3_weights, grad_inc4e_conv1x1_3_weights, velocities[160], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_3_bias, grad_inc4e_conv1x1_3_bias, velocities[161], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_3_gamma, grad_inc4e_bn1x1_3_gamma, velocities[162], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_3_beta, grad_inc4e_bn1x1_3_beta, velocities[163], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv5x5_weights, grad_inc4e_conv5x5_weights, velocities[164], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv5x5_bias, grad_inc4e_conv5x5_bias, velocities[165], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn5x5_gamma, grad_inc4e_bn5x5_gamma, velocities[166], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn5x5_beta, grad_inc4e_bn5x5_beta, velocities[167], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_4_weights, grad_inc4e_conv1x1_4_weights, velocities[168], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.conv1x1_4_bias, grad_inc4e_conv1x1_4_bias, velocities[169], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_4_gamma, grad_inc4e_bn1x1_4_gamma, velocities[170], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_4e.bn1x1_4_beta, grad_inc4e_bn1x1_4_beta, velocities[171], Float64(learning_rate), Float64(momentum))

    # ---- Inception 5a (velocities[172..195]) ----
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_1_weights, grad_inc5a_conv1x1_1_weights, velocities[172], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_1_bias, grad_inc5a_conv1x1_1_bias, velocities[173], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_1_gamma, grad_inc5a_bn1x1_1_gamma, velocities[174], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_1_beta, grad_inc5a_bn1x1_1_beta, velocities[175], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_2_weights, grad_inc5a_conv1x1_2_weights, velocities[176], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_2_bias, grad_inc5a_conv1x1_2_bias, velocities[177], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_2_gamma, grad_inc5a_bn1x1_2_gamma, velocities[178], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_2_beta, grad_inc5a_bn1x1_2_beta, velocities[179], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv3x3_weights, grad_inc5a_conv3x3_weights, velocities[180], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv3x3_bias, grad_inc5a_conv3x3_bias, velocities[181], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn3x3_gamma, grad_inc5a_bn3x3_gamma, velocities[182], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn3x3_beta, grad_inc5a_bn3x3_beta, velocities[183], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_3_weights, grad_inc5a_conv1x1_3_weights, velocities[184], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_3_bias, grad_inc5a_conv1x1_3_bias, velocities[185], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_3_gamma, grad_inc5a_bn1x1_3_gamma, velocities[186], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_3_beta, grad_inc5a_bn1x1_3_beta, velocities[187], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv5x5_weights, grad_inc5a_conv5x5_weights, velocities[188], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv5x5_bias, grad_inc5a_conv5x5_bias, velocities[189], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn5x5_gamma, grad_inc5a_bn5x5_gamma, velocities[190], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn5x5_beta, grad_inc5a_bn5x5_beta, velocities[191], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_4_weights, grad_inc5a_conv1x1_4_weights, velocities[192], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.conv1x1_4_bias, grad_inc5a_conv1x1_4_bias, velocities[193], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_4_gamma, grad_inc5a_bn1x1_4_gamma, velocities[194], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5a.bn1x1_4_beta, grad_inc5a_bn1x1_4_beta, velocities[195], Float64(learning_rate), Float64(momentum))

    # ---- Inception 5b (velocities[196..219]) ----
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_1_weights, grad_inc5b_conv1x1_1_weights, velocities[196], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_1_bias, grad_inc5b_conv1x1_1_bias, velocities[197], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_1_gamma, grad_inc5b_bn1x1_1_gamma, velocities[198], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_1_beta, grad_inc5b_bn1x1_1_beta, velocities[199], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_2_weights, grad_inc5b_conv1x1_2_weights, velocities[200], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_2_bias, grad_inc5b_conv1x1_2_bias, velocities[201], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_2_gamma, grad_inc5b_bn1x1_2_gamma, velocities[202], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_2_beta, grad_inc5b_bn1x1_2_beta, velocities[203], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv3x3_weights, grad_inc5b_conv3x3_weights, velocities[204], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv3x3_bias, grad_inc5b_conv3x3_bias, velocities[205], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn3x3_gamma, grad_inc5b_bn3x3_gamma, velocities[206], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn3x3_beta, grad_inc5b_bn3x3_beta, velocities[207], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_3_weights, grad_inc5b_conv1x1_3_weights, velocities[208], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_3_bias, grad_inc5b_conv1x1_3_bias, velocities[209], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_3_gamma, grad_inc5b_bn1x1_3_gamma, velocities[210], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_3_beta, grad_inc5b_bn1x1_3_beta, velocities[211], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv5x5_weights, grad_inc5b_conv5x5_weights, velocities[212], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv5x5_bias, grad_inc5b_conv5x5_bias, velocities[213], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn5x5_gamma, grad_inc5b_bn5x5_gamma, velocities[214], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn5x5_beta, grad_inc5b_bn5x5_beta, velocities[215], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_4_weights, grad_inc5b_conv1x1_4_weights, velocities[216], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.conv1x1_4_bias, grad_inc5b_conv1x1_4_bias, velocities[217], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_4_gamma, grad_inc5b_bn1x1_4_gamma, velocities[218], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.inception_5b.bn1x1_4_beta, grad_inc5b_bn1x1_4_beta, velocities[219], Float64(learning_rate), Float64(momentum))

    # Final FC (velocities[220..221]).
    sgd_momentum_update_inplace(model.fc_weights, grad_fc_weights, velocities[220], Float64(learning_rate), Float64(momentum))
    sgd_momentum_update_inplace(model.fc_bias, grad_fc_bias, velocities[221], Float64(learning_rate), Float64(momentum))

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

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 training set...")
    var cifar10_dataset = CIFAR10Dataset(data_dir)
    var (train_images, train_labels) = cifar10_dataset.get_train_data()

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
