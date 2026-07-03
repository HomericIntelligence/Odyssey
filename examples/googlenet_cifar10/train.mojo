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
from projectodyssey.core.normalization import batch_norm2d, batch_norm2d_backward
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
        "Epoch " + String(epoch + 1) + "/" + String(total_epochs)
        + ": lr=" + String(learning_rate)
    )
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            train_images, train_labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var batch_loss = compute_gradients(
            model, batch_images, batch_labels,
            learning_rate, momentum, velocities,
        )
        total_loss += batch_loss
        if (batch_idx + 1) % 100 == 0:
            var avg = total_loss / Float32(batch_idx + 1)
            print(
                "  Batch " + String(batch_idx + 1) + "/"
                + String(num_batches) + " - Loss: " + String(avg)
            )
    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: " + String(avg_loss))
    return avg_loss


def compute_gradients(
    mut model: GoogLeNet,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """One-batch forward pass with full activation caching.

    Backward + SGD momentum updates are the next slice of #3184.
    The velocities buffer is threaded in NOW so the signature is stable.
    """
    # ---- Initial block: conv3x3 -> BN -> ReLU -> MaxPool ----
    var init_conv_out = conv2d(
        input,
        model.initial_conv_weights, model.initial_conv_bias,
        stride=1, padding=1,
    )
    var init_bn_out, _, _ = batch_norm2d(
        init_conv_out,
        model.initial_bn_gamma, model.initial_bn_beta,
        model.initial_bn_running_mean, model.initial_bn_running_var,
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
        stride=1, padding=0,
    )
    var inc3a_b1_bn, _, _ = batch_norm2d(
        inc3a_b1_conv,
        model.inception_3a.bn1x1_1_gamma, model.inception_3a.bn1x1_1_beta,
        model.inception_3a.bn1x1_1_running_mean,
        model.inception_3a.bn1x1_1_running_var, True,
    )
    var inc3a_b1_relu = relu(inc3a_b1_bn)
    var inc3a_b2_conv1 = conv2d(
        init_pool_out,
        model.inception_3a.conv1x1_2_weights,
        model.inception_3a.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc3a_b2_bn1, _, _ = batch_norm2d(
        inc3a_b2_conv1,
        model.inception_3a.bn1x1_2_gamma, model.inception_3a.bn1x1_2_beta,
        model.inception_3a.bn1x1_2_running_mean,
        model.inception_3a.bn1x1_2_running_var, True,
    )
    var inc3a_b2_relu1 = relu(inc3a_b2_bn1)
    var inc3a_b2_conv2 = conv2d(
        inc3a_b2_relu1,
        model.inception_3a.conv3x3_weights,
        model.inception_3a.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc3a_b2_bn2, _, _ = batch_norm2d(
        inc3a_b2_conv2,
        model.inception_3a.bn3x3_gamma, model.inception_3a.bn3x3_beta,
        model.inception_3a.bn3x3_running_mean,
        model.inception_3a.bn3x3_running_var, True,
    )
    var inc3a_b2_relu2 = relu(inc3a_b2_bn2)
    var inc3a_b3_conv1 = conv2d(
        init_pool_out,
        model.inception_3a.conv1x1_3_weights,
        model.inception_3a.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc3a_b3_bn1, _, _ = batch_norm2d(
        inc3a_b3_conv1,
        model.inception_3a.bn1x1_3_gamma, model.inception_3a.bn1x1_3_beta,
        model.inception_3a.bn1x1_3_running_mean,
        model.inception_3a.bn1x1_3_running_var, True,
    )
    var inc3a_b3_relu1 = relu(inc3a_b3_bn1)
    var inc3a_b3_conv2 = conv2d(
        inc3a_b3_relu1,
        model.inception_3a.conv5x5_weights,
        model.inception_3a.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc3a_b3_bn2, _, _ = batch_norm2d(
        inc3a_b3_conv2,
        model.inception_3a.bn5x5_gamma, model.inception_3a.bn5x5_beta,
        model.inception_3a.bn5x5_running_mean,
        model.inception_3a.bn5x5_running_var, True,
    )
    var inc3a_b3_relu2 = relu(inc3a_b3_bn2)
    var inc3a_b4_pool = maxpool2d(
        init_pool_out, kernel_size=3, stride=1, padding=1
    )
    var inc3a_b4_conv = conv2d(
        inc3a_b4_pool,
        model.inception_3a.conv1x1_4_weights,
        model.inception_3a.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc3a_b4_bn, _, _ = batch_norm2d(
        inc3a_b4_conv,
        model.inception_3a.bn1x1_4_gamma, model.inception_3a.bn1x1_4_beta,
        model.inception_3a.bn1x1_4_running_mean,
        model.inception_3a.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc3b_b1_bn, _, _ = batch_norm2d(
        inc3b_b1_conv,
        model.inception_3b.bn1x1_1_gamma, model.inception_3b.bn1x1_1_beta,
        model.inception_3b.bn1x1_1_running_mean,
        model.inception_3b.bn1x1_1_running_var, True,
    )
    var inc3b_b1_relu = relu(inc3b_b1_bn)
    var inc3b_b2_conv1 = conv2d(
        inc3a_out,
        model.inception_3b.conv1x1_2_weights,
        model.inception_3b.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc3b_b2_bn1, _, _ = batch_norm2d(
        inc3b_b2_conv1,
        model.inception_3b.bn1x1_2_gamma, model.inception_3b.bn1x1_2_beta,
        model.inception_3b.bn1x1_2_running_mean,
        model.inception_3b.bn1x1_2_running_var, True,
    )
    var inc3b_b2_relu1 = relu(inc3b_b2_bn1)
    var inc3b_b2_conv2 = conv2d(
        inc3b_b2_relu1,
        model.inception_3b.conv3x3_weights,
        model.inception_3b.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc3b_b2_bn2, _, _ = batch_norm2d(
        inc3b_b2_conv2,
        model.inception_3b.bn3x3_gamma, model.inception_3b.bn3x3_beta,
        model.inception_3b.bn3x3_running_mean,
        model.inception_3b.bn3x3_running_var, True,
    )
    var inc3b_b2_relu2 = relu(inc3b_b2_bn2)
    var inc3b_b3_conv1 = conv2d(
        inc3a_out,
        model.inception_3b.conv1x1_3_weights,
        model.inception_3b.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc3b_b3_bn1, _, _ = batch_norm2d(
        inc3b_b3_conv1,
        model.inception_3b.bn1x1_3_gamma, model.inception_3b.bn1x1_3_beta,
        model.inception_3b.bn1x1_3_running_mean,
        model.inception_3b.bn1x1_3_running_var, True,
    )
    var inc3b_b3_relu1 = relu(inc3b_b3_bn1)
    var inc3b_b3_conv2 = conv2d(
        inc3b_b3_relu1,
        model.inception_3b.conv5x5_weights,
        model.inception_3b.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc3b_b3_bn2, _, _ = batch_norm2d(
        inc3b_b3_conv2,
        model.inception_3b.bn5x5_gamma, model.inception_3b.bn5x5_beta,
        model.inception_3b.bn5x5_running_mean,
        model.inception_3b.bn5x5_running_var, True,
    )
    var inc3b_b3_relu2 = relu(inc3b_b3_bn2)
    var inc3b_b4_pool = maxpool2d(
        inc3a_out, kernel_size=3, stride=1, padding=1
    )
    var inc3b_b4_conv = conv2d(
        inc3b_b4_pool,
        model.inception_3b.conv1x1_4_weights,
        model.inception_3b.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc3b_b4_bn, _, _ = batch_norm2d(
        inc3b_b4_conv,
        model.inception_3b.bn1x1_4_gamma, model.inception_3b.bn1x1_4_beta,
        model.inception_3b.bn1x1_4_running_mean,
        model.inception_3b.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc4a_b1_bn, _, _ = batch_norm2d(
        inc4a_b1_conv,
        model.inception_4a.bn1x1_1_gamma, model.inception_4a.bn1x1_1_beta,
        model.inception_4a.bn1x1_1_running_mean,
        model.inception_4a.bn1x1_1_running_var, True,
    )
    var inc4a_b1_relu = relu(inc4a_b1_bn)
    var inc4a_b2_conv1 = conv2d(
        mid1_pool,
        model.inception_4a.conv1x1_2_weights,
        model.inception_4a.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc4a_b2_bn1, _, _ = batch_norm2d(
        inc4a_b2_conv1,
        model.inception_4a.bn1x1_2_gamma, model.inception_4a.bn1x1_2_beta,
        model.inception_4a.bn1x1_2_running_mean,
        model.inception_4a.bn1x1_2_running_var, True,
    )
    var inc4a_b2_relu1 = relu(inc4a_b2_bn1)
    var inc4a_b2_conv2 = conv2d(
        inc4a_b2_relu1,
        model.inception_4a.conv3x3_weights,
        model.inception_4a.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc4a_b2_bn2, _, _ = batch_norm2d(
        inc4a_b2_conv2,
        model.inception_4a.bn3x3_gamma, model.inception_4a.bn3x3_beta,
        model.inception_4a.bn3x3_running_mean,
        model.inception_4a.bn3x3_running_var, True,
    )
    var inc4a_b2_relu2 = relu(inc4a_b2_bn2)
    var inc4a_b3_conv1 = conv2d(
        mid1_pool,
        model.inception_4a.conv1x1_3_weights,
        model.inception_4a.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc4a_b3_bn1, _, _ = batch_norm2d(
        inc4a_b3_conv1,
        model.inception_4a.bn1x1_3_gamma, model.inception_4a.bn1x1_3_beta,
        model.inception_4a.bn1x1_3_running_mean,
        model.inception_4a.bn1x1_3_running_var, True,
    )
    var inc4a_b3_relu1 = relu(inc4a_b3_bn1)
    var inc4a_b3_conv2 = conv2d(
        inc4a_b3_relu1,
        model.inception_4a.conv5x5_weights,
        model.inception_4a.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc4a_b3_bn2, _, _ = batch_norm2d(
        inc4a_b3_conv2,
        model.inception_4a.bn5x5_gamma, model.inception_4a.bn5x5_beta,
        model.inception_4a.bn5x5_running_mean,
        model.inception_4a.bn5x5_running_var, True,
    )
    var inc4a_b3_relu2 = relu(inc4a_b3_bn2)
    var inc4a_b4_pool = maxpool2d(
        mid1_pool, kernel_size=3, stride=1, padding=1
    )
    var inc4a_b4_conv = conv2d(
        inc4a_b4_pool,
        model.inception_4a.conv1x1_4_weights,
        model.inception_4a.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc4a_b4_bn, _, _ = batch_norm2d(
        inc4a_b4_conv,
        model.inception_4a.bn1x1_4_gamma, model.inception_4a.bn1x1_4_beta,
        model.inception_4a.bn1x1_4_running_mean,
        model.inception_4a.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc4b_b1_bn, _, _ = batch_norm2d(
        inc4b_b1_conv,
        model.inception_4b.bn1x1_1_gamma, model.inception_4b.bn1x1_1_beta,
        model.inception_4b.bn1x1_1_running_mean,
        model.inception_4b.bn1x1_1_running_var, True,
    )
    var inc4b_b1_relu = relu(inc4b_b1_bn)
    var inc4b_b2_conv1 = conv2d(
        inc4a_out,
        model.inception_4b.conv1x1_2_weights,
        model.inception_4b.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc4b_b2_bn1, _, _ = batch_norm2d(
        inc4b_b2_conv1,
        model.inception_4b.bn1x1_2_gamma, model.inception_4b.bn1x1_2_beta,
        model.inception_4b.bn1x1_2_running_mean,
        model.inception_4b.bn1x1_2_running_var, True,
    )
    var inc4b_b2_relu1 = relu(inc4b_b2_bn1)
    var inc4b_b2_conv2 = conv2d(
        inc4b_b2_relu1,
        model.inception_4b.conv3x3_weights,
        model.inception_4b.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc4b_b2_bn2, _, _ = batch_norm2d(
        inc4b_b2_conv2,
        model.inception_4b.bn3x3_gamma, model.inception_4b.bn3x3_beta,
        model.inception_4b.bn3x3_running_mean,
        model.inception_4b.bn3x3_running_var, True,
    )
    var inc4b_b2_relu2 = relu(inc4b_b2_bn2)
    var inc4b_b3_conv1 = conv2d(
        inc4a_out,
        model.inception_4b.conv1x1_3_weights,
        model.inception_4b.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc4b_b3_bn1, _, _ = batch_norm2d(
        inc4b_b3_conv1,
        model.inception_4b.bn1x1_3_gamma, model.inception_4b.bn1x1_3_beta,
        model.inception_4b.bn1x1_3_running_mean,
        model.inception_4b.bn1x1_3_running_var, True,
    )
    var inc4b_b3_relu1 = relu(inc4b_b3_bn1)
    var inc4b_b3_conv2 = conv2d(
        inc4b_b3_relu1,
        model.inception_4b.conv5x5_weights,
        model.inception_4b.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc4b_b3_bn2, _, _ = batch_norm2d(
        inc4b_b3_conv2,
        model.inception_4b.bn5x5_gamma, model.inception_4b.bn5x5_beta,
        model.inception_4b.bn5x5_running_mean,
        model.inception_4b.bn5x5_running_var, True,
    )
    var inc4b_b3_relu2 = relu(inc4b_b3_bn2)
    var inc4b_b4_pool = maxpool2d(
        inc4a_out, kernel_size=3, stride=1, padding=1
    )
    var inc4b_b4_conv = conv2d(
        inc4b_b4_pool,
        model.inception_4b.conv1x1_4_weights,
        model.inception_4b.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc4b_b4_bn, _, _ = batch_norm2d(
        inc4b_b4_conv,
        model.inception_4b.bn1x1_4_gamma, model.inception_4b.bn1x1_4_beta,
        model.inception_4b.bn1x1_4_running_mean,
        model.inception_4b.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc4c_b1_bn, _, _ = batch_norm2d(
        inc4c_b1_conv,
        model.inception_4c.bn1x1_1_gamma, model.inception_4c.bn1x1_1_beta,
        model.inception_4c.bn1x1_1_running_mean,
        model.inception_4c.bn1x1_1_running_var, True,
    )
    var inc4c_b1_relu = relu(inc4c_b1_bn)
    var inc4c_b2_conv1 = conv2d(
        inc4b_out,
        model.inception_4c.conv1x1_2_weights,
        model.inception_4c.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc4c_b2_bn1, _, _ = batch_norm2d(
        inc4c_b2_conv1,
        model.inception_4c.bn1x1_2_gamma, model.inception_4c.bn1x1_2_beta,
        model.inception_4c.bn1x1_2_running_mean,
        model.inception_4c.bn1x1_2_running_var, True,
    )
    var inc4c_b2_relu1 = relu(inc4c_b2_bn1)
    var inc4c_b2_conv2 = conv2d(
        inc4c_b2_relu1,
        model.inception_4c.conv3x3_weights,
        model.inception_4c.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc4c_b2_bn2, _, _ = batch_norm2d(
        inc4c_b2_conv2,
        model.inception_4c.bn3x3_gamma, model.inception_4c.bn3x3_beta,
        model.inception_4c.bn3x3_running_mean,
        model.inception_4c.bn3x3_running_var, True,
    )
    var inc4c_b2_relu2 = relu(inc4c_b2_bn2)
    var inc4c_b3_conv1 = conv2d(
        inc4b_out,
        model.inception_4c.conv1x1_3_weights,
        model.inception_4c.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc4c_b3_bn1, _, _ = batch_norm2d(
        inc4c_b3_conv1,
        model.inception_4c.bn1x1_3_gamma, model.inception_4c.bn1x1_3_beta,
        model.inception_4c.bn1x1_3_running_mean,
        model.inception_4c.bn1x1_3_running_var, True,
    )
    var inc4c_b3_relu1 = relu(inc4c_b3_bn1)
    var inc4c_b3_conv2 = conv2d(
        inc4c_b3_relu1,
        model.inception_4c.conv5x5_weights,
        model.inception_4c.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc4c_b3_bn2, _, _ = batch_norm2d(
        inc4c_b3_conv2,
        model.inception_4c.bn5x5_gamma, model.inception_4c.bn5x5_beta,
        model.inception_4c.bn5x5_running_mean,
        model.inception_4c.bn5x5_running_var, True,
    )
    var inc4c_b3_relu2 = relu(inc4c_b3_bn2)
    var inc4c_b4_pool = maxpool2d(
        inc4b_out, kernel_size=3, stride=1, padding=1
    )
    var inc4c_b4_conv = conv2d(
        inc4c_b4_pool,
        model.inception_4c.conv1x1_4_weights,
        model.inception_4c.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc4c_b4_bn, _, _ = batch_norm2d(
        inc4c_b4_conv,
        model.inception_4c.bn1x1_4_gamma, model.inception_4c.bn1x1_4_beta,
        model.inception_4c.bn1x1_4_running_mean,
        model.inception_4c.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc4d_b1_bn, _, _ = batch_norm2d(
        inc4d_b1_conv,
        model.inception_4d.bn1x1_1_gamma, model.inception_4d.bn1x1_1_beta,
        model.inception_4d.bn1x1_1_running_mean,
        model.inception_4d.bn1x1_1_running_var, True,
    )
    var inc4d_b1_relu = relu(inc4d_b1_bn)
    var inc4d_b2_conv1 = conv2d(
        inc4c_out,
        model.inception_4d.conv1x1_2_weights,
        model.inception_4d.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc4d_b2_bn1, _, _ = batch_norm2d(
        inc4d_b2_conv1,
        model.inception_4d.bn1x1_2_gamma, model.inception_4d.bn1x1_2_beta,
        model.inception_4d.bn1x1_2_running_mean,
        model.inception_4d.bn1x1_2_running_var, True,
    )
    var inc4d_b2_relu1 = relu(inc4d_b2_bn1)
    var inc4d_b2_conv2 = conv2d(
        inc4d_b2_relu1,
        model.inception_4d.conv3x3_weights,
        model.inception_4d.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc4d_b2_bn2, _, _ = batch_norm2d(
        inc4d_b2_conv2,
        model.inception_4d.bn3x3_gamma, model.inception_4d.bn3x3_beta,
        model.inception_4d.bn3x3_running_mean,
        model.inception_4d.bn3x3_running_var, True,
    )
    var inc4d_b2_relu2 = relu(inc4d_b2_bn2)
    var inc4d_b3_conv1 = conv2d(
        inc4c_out,
        model.inception_4d.conv1x1_3_weights,
        model.inception_4d.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc4d_b3_bn1, _, _ = batch_norm2d(
        inc4d_b3_conv1,
        model.inception_4d.bn1x1_3_gamma, model.inception_4d.bn1x1_3_beta,
        model.inception_4d.bn1x1_3_running_mean,
        model.inception_4d.bn1x1_3_running_var, True,
    )
    var inc4d_b3_relu1 = relu(inc4d_b3_bn1)
    var inc4d_b3_conv2 = conv2d(
        inc4d_b3_relu1,
        model.inception_4d.conv5x5_weights,
        model.inception_4d.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc4d_b3_bn2, _, _ = batch_norm2d(
        inc4d_b3_conv2,
        model.inception_4d.bn5x5_gamma, model.inception_4d.bn5x5_beta,
        model.inception_4d.bn5x5_running_mean,
        model.inception_4d.bn5x5_running_var, True,
    )
    var inc4d_b3_relu2 = relu(inc4d_b3_bn2)
    var inc4d_b4_pool = maxpool2d(
        inc4c_out, kernel_size=3, stride=1, padding=1
    )
    var inc4d_b4_conv = conv2d(
        inc4d_b4_pool,
        model.inception_4d.conv1x1_4_weights,
        model.inception_4d.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc4d_b4_bn, _, _ = batch_norm2d(
        inc4d_b4_conv,
        model.inception_4d.bn1x1_4_gamma, model.inception_4d.bn1x1_4_beta,
        model.inception_4d.bn1x1_4_running_mean,
        model.inception_4d.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc4e_b1_bn, _, _ = batch_norm2d(
        inc4e_b1_conv,
        model.inception_4e.bn1x1_1_gamma, model.inception_4e.bn1x1_1_beta,
        model.inception_4e.bn1x1_1_running_mean,
        model.inception_4e.bn1x1_1_running_var, True,
    )
    var inc4e_b1_relu = relu(inc4e_b1_bn)
    var inc4e_b2_conv1 = conv2d(
        inc4d_out,
        model.inception_4e.conv1x1_2_weights,
        model.inception_4e.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc4e_b2_bn1, _, _ = batch_norm2d(
        inc4e_b2_conv1,
        model.inception_4e.bn1x1_2_gamma, model.inception_4e.bn1x1_2_beta,
        model.inception_4e.bn1x1_2_running_mean,
        model.inception_4e.bn1x1_2_running_var, True,
    )
    var inc4e_b2_relu1 = relu(inc4e_b2_bn1)
    var inc4e_b2_conv2 = conv2d(
        inc4e_b2_relu1,
        model.inception_4e.conv3x3_weights,
        model.inception_4e.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc4e_b2_bn2, _, _ = batch_norm2d(
        inc4e_b2_conv2,
        model.inception_4e.bn3x3_gamma, model.inception_4e.bn3x3_beta,
        model.inception_4e.bn3x3_running_mean,
        model.inception_4e.bn3x3_running_var, True,
    )
    var inc4e_b2_relu2 = relu(inc4e_b2_bn2)
    var inc4e_b3_conv1 = conv2d(
        inc4d_out,
        model.inception_4e.conv1x1_3_weights,
        model.inception_4e.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc4e_b3_bn1, _, _ = batch_norm2d(
        inc4e_b3_conv1,
        model.inception_4e.bn1x1_3_gamma, model.inception_4e.bn1x1_3_beta,
        model.inception_4e.bn1x1_3_running_mean,
        model.inception_4e.bn1x1_3_running_var, True,
    )
    var inc4e_b3_relu1 = relu(inc4e_b3_bn1)
    var inc4e_b3_conv2 = conv2d(
        inc4e_b3_relu1,
        model.inception_4e.conv5x5_weights,
        model.inception_4e.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc4e_b3_bn2, _, _ = batch_norm2d(
        inc4e_b3_conv2,
        model.inception_4e.bn5x5_gamma, model.inception_4e.bn5x5_beta,
        model.inception_4e.bn5x5_running_mean,
        model.inception_4e.bn5x5_running_var, True,
    )
    var inc4e_b3_relu2 = relu(inc4e_b3_bn2)
    var inc4e_b4_pool = maxpool2d(
        inc4d_out, kernel_size=3, stride=1, padding=1
    )
    var inc4e_b4_conv = conv2d(
        inc4e_b4_pool,
        model.inception_4e.conv1x1_4_weights,
        model.inception_4e.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc4e_b4_bn, _, _ = batch_norm2d(
        inc4e_b4_conv,
        model.inception_4e.bn1x1_4_gamma, model.inception_4e.bn1x1_4_beta,
        model.inception_4e.bn1x1_4_running_mean,
        model.inception_4e.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc5a_b1_bn, _, _ = batch_norm2d(
        inc5a_b1_conv,
        model.inception_5a.bn1x1_1_gamma, model.inception_5a.bn1x1_1_beta,
        model.inception_5a.bn1x1_1_running_mean,
        model.inception_5a.bn1x1_1_running_var, True,
    )
    var inc5a_b1_relu = relu(inc5a_b1_bn)
    var inc5a_b2_conv1 = conv2d(
        mid2_pool,
        model.inception_5a.conv1x1_2_weights,
        model.inception_5a.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc5a_b2_bn1, _, _ = batch_norm2d(
        inc5a_b2_conv1,
        model.inception_5a.bn1x1_2_gamma, model.inception_5a.bn1x1_2_beta,
        model.inception_5a.bn1x1_2_running_mean,
        model.inception_5a.bn1x1_2_running_var, True,
    )
    var inc5a_b2_relu1 = relu(inc5a_b2_bn1)
    var inc5a_b2_conv2 = conv2d(
        inc5a_b2_relu1,
        model.inception_5a.conv3x3_weights,
        model.inception_5a.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc5a_b2_bn2, _, _ = batch_norm2d(
        inc5a_b2_conv2,
        model.inception_5a.bn3x3_gamma, model.inception_5a.bn3x3_beta,
        model.inception_5a.bn3x3_running_mean,
        model.inception_5a.bn3x3_running_var, True,
    )
    var inc5a_b2_relu2 = relu(inc5a_b2_bn2)
    var inc5a_b3_conv1 = conv2d(
        mid2_pool,
        model.inception_5a.conv1x1_3_weights,
        model.inception_5a.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc5a_b3_bn1, _, _ = batch_norm2d(
        inc5a_b3_conv1,
        model.inception_5a.bn1x1_3_gamma, model.inception_5a.bn1x1_3_beta,
        model.inception_5a.bn1x1_3_running_mean,
        model.inception_5a.bn1x1_3_running_var, True,
    )
    var inc5a_b3_relu1 = relu(inc5a_b3_bn1)
    var inc5a_b3_conv2 = conv2d(
        inc5a_b3_relu1,
        model.inception_5a.conv5x5_weights,
        model.inception_5a.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc5a_b3_bn2, _, _ = batch_norm2d(
        inc5a_b3_conv2,
        model.inception_5a.bn5x5_gamma, model.inception_5a.bn5x5_beta,
        model.inception_5a.bn5x5_running_mean,
        model.inception_5a.bn5x5_running_var, True,
    )
    var inc5a_b3_relu2 = relu(inc5a_b3_bn2)
    var inc5a_b4_pool = maxpool2d(
        mid2_pool, kernel_size=3, stride=1, padding=1
    )
    var inc5a_b4_conv = conv2d(
        inc5a_b4_pool,
        model.inception_5a.conv1x1_4_weights,
        model.inception_5a.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc5a_b4_bn, _, _ = batch_norm2d(
        inc5a_b4_conv,
        model.inception_5a.bn1x1_4_gamma, model.inception_5a.bn1x1_4_beta,
        model.inception_5a.bn1x1_4_running_mean,
        model.inception_5a.bn1x1_4_running_var, True,
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
        stride=1, padding=0,
    )
    var inc5b_b1_bn, _, _ = batch_norm2d(
        inc5b_b1_conv,
        model.inception_5b.bn1x1_1_gamma, model.inception_5b.bn1x1_1_beta,
        model.inception_5b.bn1x1_1_running_mean,
        model.inception_5b.bn1x1_1_running_var, True,
    )
    var inc5b_b1_relu = relu(inc5b_b1_bn)
    var inc5b_b2_conv1 = conv2d(
        inc5a_out,
        model.inception_5b.conv1x1_2_weights,
        model.inception_5b.conv1x1_2_bias,
        stride=1, padding=0,
    )
    var inc5b_b2_bn1, _, _ = batch_norm2d(
        inc5b_b2_conv1,
        model.inception_5b.bn1x1_2_gamma, model.inception_5b.bn1x1_2_beta,
        model.inception_5b.bn1x1_2_running_mean,
        model.inception_5b.bn1x1_2_running_var, True,
    )
    var inc5b_b2_relu1 = relu(inc5b_b2_bn1)
    var inc5b_b2_conv2 = conv2d(
        inc5b_b2_relu1,
        model.inception_5b.conv3x3_weights,
        model.inception_5b.conv3x3_bias,
        stride=1, padding=1,
    )
    var inc5b_b2_bn2, _, _ = batch_norm2d(
        inc5b_b2_conv2,
        model.inception_5b.bn3x3_gamma, model.inception_5b.bn3x3_beta,
        model.inception_5b.bn3x3_running_mean,
        model.inception_5b.bn3x3_running_var, True,
    )
    var inc5b_b2_relu2 = relu(inc5b_b2_bn2)
    var inc5b_b3_conv1 = conv2d(
        inc5a_out,
        model.inception_5b.conv1x1_3_weights,
        model.inception_5b.conv1x1_3_bias,
        stride=1, padding=0,
    )
    var inc5b_b3_bn1, _, _ = batch_norm2d(
        inc5b_b3_conv1,
        model.inception_5b.bn1x1_3_gamma, model.inception_5b.bn1x1_3_beta,
        model.inception_5b.bn1x1_3_running_mean,
        model.inception_5b.bn1x1_3_running_var, True,
    )
    var inc5b_b3_relu1 = relu(inc5b_b3_bn1)
    var inc5b_b3_conv2 = conv2d(
        inc5b_b3_relu1,
        model.inception_5b.conv5x5_weights,
        model.inception_5b.conv5x5_bias,
        stride=1, padding=2,
    )
    var inc5b_b3_bn2, _, _ = batch_norm2d(
        inc5b_b3_conv2,
        model.inception_5b.bn5x5_gamma, model.inception_5b.bn5x5_beta,
        model.inception_5b.bn5x5_running_mean,
        model.inception_5b.bn5x5_running_var, True,
    )
    var inc5b_b3_relu2 = relu(inc5b_b3_bn2)
    var inc5b_b4_pool = maxpool2d(
        inc5a_out, kernel_size=3, stride=1, padding=1
    )
    var inc5b_b4_conv = conv2d(
        inc5b_b4_pool,
        model.inception_5b.conv1x1_4_weights,
        model.inception_5b.conv1x1_4_bias,
        stride=1, padding=0,
    )
    var inc5b_b4_bn, _, _ = batch_norm2d(
        inc5b_b4_conv,
        model.inception_5b.bn1x1_4_gamma, model.inception_5b.bn1x1_4_beta,
        model.inception_5b.bn1x1_4_running_mean,
        model.inception_5b.bn1x1_4_running_var, True,
    )
    var inc5b_b4_relu = relu(inc5b_b4_bn)
    var inc5b_out = concatenate_depthwise(
        inc5b_b1_relu, inc5b_b2_relu2, inc5b_b3_relu2, inc5b_b4_relu
    )

    # ---- Global avg pool -> flatten -> dropout(0.4) -> linear -> CE ----
    var gap_out = global_avgpool2d(inc5b_out)              # (N, 1024, 1, 1)
    var flat_result = _flatten_gap(gap_out)                # (N, 1024)
    var flat_out = flat_result[0]
    var gap_shape = flat_result[1]
    var drop_result = dropout(flat_out, Float64(0.4), training=True)
    var drop_out = drop_result[0]
    var drop_mask = drop_result[1]
    var logits = linear(drop_out, model.fc_weights, model.fc_bias)
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # BACKWARD + SGD are the next slice of #3184.
    # Cached activations available to that slice:
    #   init_{conv_out, bn_out, relu_out, pool_out}
    #   inc{Nx}_{b1_{conv,bn,relu},
    #            b2_{conv1,bn1,relu1,conv2,bn2,relu2},
    #            b3_{conv1,bn1,relu1,conv2,bn2,relu2},
    #            b4_{pool,conv,bn,relu}, out}
    #     for Nx in {3a,3b,4a,4b,4c,4d,4e,5a,5b}
    #   mid1_pool, mid2_pool, gap_out, flat_out, gap_shape,
    #   drop_out, drop_mask, logits
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
        "  Momentum velocities: " + String(len(velocities))
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
            model, train_images, train_labels,
            batch_size, lr, momentum, velocities, epoch, epochs,
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
