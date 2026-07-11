"""Training Script for MobileNetV1 on CIFAR-10.

This script trains a MobileNetV1 model on CIFAR-10 using manual backpropagation.

Usage:
    mojo run examples/mobilenetv1_cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01

Features:
    - Manual backpropagation through 13 depthwise separable blocks
    - SGD optimizer with momentum (0.9)
    - Batch normalization in training mode
    - Learning rate scheduling (step decay)
    - Depthwise separable convolutions for efficiency

Architecture:
    - 28 layers deep (initial + 13 depthwise separable blocks + classifier)
    - Each block: Depthwise (3×3) → BN → ReLU → Pointwise (1×1) → BN → ReLU
    - ~4.2M parameters total
    - 60M operations per inference (vs VGG's 15B!)

Training Details:
    - Loss: Cross-entropy
    - Optimizer: SGD with momentum
    - Learning rate schedule: Step decay (×0.2 every 60 epochs)
    - Batch size: 128 (default)
    - Epochs: 200 (recommended)
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core.conv import (
    conv2d,
    conv2d_backward,
    depthwise_conv2d,
    depthwise_conv2d_backward,
)
from projectodyssey.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
)
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.pooling import (
    global_avgpool2d,
    global_avgpool2d_backward,
)
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.training.optimizers.sgd import sgd_momentum_update_inplace
from projectodyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
)
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.data.datasets import CIFAR10Dataset
from projectodyssey.data import one_hot_encode
from projectodyssey.training.schedulers import step_lr
from projectodyssey.utils.training_args import parse_training_args_with_defaults
from model import MobileNetV1


def compute_gradients(
    mut model: MobileNetV1,
    input: AnyTensor,
    labels_onehot: AnyTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """Forward + backward + SGD-momentum update for one batch.

    BN running_mean/running_var from every batch_norm2d call are written back
    to the model (#5543), so inference via model.forward(training=False) uses
    statistics accumulated during training rather than init values.
    """
    # ===== Forward pass (caches: <name>_out per stage; block inputs bN_in) =====

    # Initial conv (stride=2, padding=1): (B,3,32,32) -> (B,32,16,16)
    var init_conv_out = conv2d(
        input,
        model.initial_conv_weights,
        model.initial_conv_bias,
        stride=2,
        padding=1,
    )
    var init_bn_tuple = batch_norm2d(
        init_conv_out,
        model.initial_bn_gamma,
        model.initial_bn_beta,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        True,
    )
    model.initial_bn_running_mean = init_bn_tuple[1]
    model.initial_bn_running_var = init_bn_tuple[2]
    var init_bn_out = init_bn_tuple[0]
    var init_relu_out = relu(init_bn_out)

    # ---- Block 1 (in=32, out=64, stride=1) ----
    var b1_in = init_relu_out
    var b1_dw_out = depthwise_conv2d(
        b1_in,
        model.ds_block_1.dw_weights,
        model.ds_block_1.dw_bias,
        stride=1,
        padding=1,
    )
    var b1_dw_bn_tuple = batch_norm2d(
        b1_dw_out,
        model.ds_block_1.dw_bn_gamma,
        model.ds_block_1.dw_bn_beta,
        model.ds_block_1.dw_bn_running_mean,
        model.ds_block_1.dw_bn_running_var,
        True,
    )
    model.ds_block_1.dw_bn_running_mean = b1_dw_bn_tuple[1]
    model.ds_block_1.dw_bn_running_var = b1_dw_bn_tuple[2]
    var b1_dw_bn_out = b1_dw_bn_tuple[0]
    var b1_dw_relu_out = relu(b1_dw_bn_out)
    var b1_pw_out = conv2d(
        b1_dw_relu_out,
        model.ds_block_1.pw_weights,
        model.ds_block_1.pw_bias,
        stride=1,
        padding=0,
    )
    var b1_pw_bn_tuple = batch_norm2d(
        b1_pw_out,
        model.ds_block_1.pw_bn_gamma,
        model.ds_block_1.pw_bn_beta,
        model.ds_block_1.pw_bn_running_mean,
        model.ds_block_1.pw_bn_running_var,
        True,
    )
    model.ds_block_1.pw_bn_running_mean = b1_pw_bn_tuple[1]
    model.ds_block_1.pw_bn_running_var = b1_pw_bn_tuple[2]
    var b1_pw_bn_out = b1_pw_bn_tuple[0]
    var b1_out = relu(b1_pw_bn_out)

    # ---- Block 2 (in=64, out=128, stride=2) ----
    var b2_in = b1_out
    var b2_dw_out = depthwise_conv2d(
        b2_in,
        model.ds_block_2.dw_weights,
        model.ds_block_2.dw_bias,
        stride=2,
        padding=1,
    )
    var b2_dw_bn_tuple = batch_norm2d(
        b2_dw_out,
        model.ds_block_2.dw_bn_gamma,
        model.ds_block_2.dw_bn_beta,
        model.ds_block_2.dw_bn_running_mean,
        model.ds_block_2.dw_bn_running_var,
        True,
    )
    model.ds_block_2.dw_bn_running_mean = b2_dw_bn_tuple[1]
    model.ds_block_2.dw_bn_running_var = b2_dw_bn_tuple[2]
    var b2_dw_bn_out = b2_dw_bn_tuple[0]
    var b2_dw_relu_out = relu(b2_dw_bn_out)
    var b2_pw_out = conv2d(
        b2_dw_relu_out,
        model.ds_block_2.pw_weights,
        model.ds_block_2.pw_bias,
        stride=1,
        padding=0,
    )
    var b2_pw_bn_tuple = batch_norm2d(
        b2_pw_out,
        model.ds_block_2.pw_bn_gamma,
        model.ds_block_2.pw_bn_beta,
        model.ds_block_2.pw_bn_running_mean,
        model.ds_block_2.pw_bn_running_var,
        True,
    )
    model.ds_block_2.pw_bn_running_mean = b2_pw_bn_tuple[1]
    model.ds_block_2.pw_bn_running_var = b2_pw_bn_tuple[2]
    var b2_pw_bn_out = b2_pw_bn_tuple[0]
    var b2_out = relu(b2_pw_bn_out)

    # ---- Block 3 (in=128, out=128, stride=1) ----
    var b3_in = b2_out
    var b3_dw_out = depthwise_conv2d(
        b3_in,
        model.ds_block_3.dw_weights,
        model.ds_block_3.dw_bias,
        stride=1,
        padding=1,
    )
    var b3_dw_bn_tuple = batch_norm2d(
        b3_dw_out,
        model.ds_block_3.dw_bn_gamma,
        model.ds_block_3.dw_bn_beta,
        model.ds_block_3.dw_bn_running_mean,
        model.ds_block_3.dw_bn_running_var,
        True,
    )
    model.ds_block_3.dw_bn_running_mean = b3_dw_bn_tuple[1]
    model.ds_block_3.dw_bn_running_var = b3_dw_bn_tuple[2]
    var b3_dw_bn_out = b3_dw_bn_tuple[0]
    var b3_dw_relu_out = relu(b3_dw_bn_out)
    var b3_pw_out = conv2d(
        b3_dw_relu_out,
        model.ds_block_3.pw_weights,
        model.ds_block_3.pw_bias,
        stride=1,
        padding=0,
    )
    var b3_pw_bn_tuple = batch_norm2d(
        b3_pw_out,
        model.ds_block_3.pw_bn_gamma,
        model.ds_block_3.pw_bn_beta,
        model.ds_block_3.pw_bn_running_mean,
        model.ds_block_3.pw_bn_running_var,
        True,
    )
    model.ds_block_3.pw_bn_running_mean = b3_pw_bn_tuple[1]
    model.ds_block_3.pw_bn_running_var = b3_pw_bn_tuple[2]
    var b3_pw_bn_out = b3_pw_bn_tuple[0]
    var b3_out = relu(b3_pw_bn_out)

    # ---- Block 4 (in=128, out=256, stride=2) ----
    var b4_in = b3_out
    var b4_dw_out = depthwise_conv2d(
        b4_in,
        model.ds_block_4.dw_weights,
        model.ds_block_4.dw_bias,
        stride=2,
        padding=1,
    )
    var b4_dw_bn_tuple = batch_norm2d(
        b4_dw_out,
        model.ds_block_4.dw_bn_gamma,
        model.ds_block_4.dw_bn_beta,
        model.ds_block_4.dw_bn_running_mean,
        model.ds_block_4.dw_bn_running_var,
        True,
    )
    model.ds_block_4.dw_bn_running_mean = b4_dw_bn_tuple[1]
    model.ds_block_4.dw_bn_running_var = b4_dw_bn_tuple[2]
    var b4_dw_bn_out = b4_dw_bn_tuple[0]
    var b4_dw_relu_out = relu(b4_dw_bn_out)
    var b4_pw_out = conv2d(
        b4_dw_relu_out,
        model.ds_block_4.pw_weights,
        model.ds_block_4.pw_bias,
        stride=1,
        padding=0,
    )
    var b4_pw_bn_tuple = batch_norm2d(
        b4_pw_out,
        model.ds_block_4.pw_bn_gamma,
        model.ds_block_4.pw_bn_beta,
        model.ds_block_4.pw_bn_running_mean,
        model.ds_block_4.pw_bn_running_var,
        True,
    )
    model.ds_block_4.pw_bn_running_mean = b4_pw_bn_tuple[1]
    model.ds_block_4.pw_bn_running_var = b4_pw_bn_tuple[2]
    var b4_pw_bn_out = b4_pw_bn_tuple[0]
    var b4_out = relu(b4_pw_bn_out)

    # ---- Block 5 (in=256, out=256, stride=1) ----
    var b5_in = b4_out
    var b5_dw_out = depthwise_conv2d(
        b5_in,
        model.ds_block_5.dw_weights,
        model.ds_block_5.dw_bias,
        stride=1,
        padding=1,
    )
    var b5_dw_bn_tuple = batch_norm2d(
        b5_dw_out,
        model.ds_block_5.dw_bn_gamma,
        model.ds_block_5.dw_bn_beta,
        model.ds_block_5.dw_bn_running_mean,
        model.ds_block_5.dw_bn_running_var,
        True,
    )
    model.ds_block_5.dw_bn_running_mean = b5_dw_bn_tuple[1]
    model.ds_block_5.dw_bn_running_var = b5_dw_bn_tuple[2]
    var b5_dw_bn_out = b5_dw_bn_tuple[0]
    var b5_dw_relu_out = relu(b5_dw_bn_out)
    var b5_pw_out = conv2d(
        b5_dw_relu_out,
        model.ds_block_5.pw_weights,
        model.ds_block_5.pw_bias,
        stride=1,
        padding=0,
    )
    var b5_pw_bn_tuple = batch_norm2d(
        b5_pw_out,
        model.ds_block_5.pw_bn_gamma,
        model.ds_block_5.pw_bn_beta,
        model.ds_block_5.pw_bn_running_mean,
        model.ds_block_5.pw_bn_running_var,
        True,
    )
    model.ds_block_5.pw_bn_running_mean = b5_pw_bn_tuple[1]
    model.ds_block_5.pw_bn_running_var = b5_pw_bn_tuple[2]
    var b5_pw_bn_out = b5_pw_bn_tuple[0]
    var b5_out = relu(b5_pw_bn_out)

    # ---- Block 6 (in=256, out=512, stride=2) ----
    var b6_in = b5_out
    var b6_dw_out = depthwise_conv2d(
        b6_in,
        model.ds_block_6.dw_weights,
        model.ds_block_6.dw_bias,
        stride=2,
        padding=1,
    )
    var b6_dw_bn_tuple = batch_norm2d(
        b6_dw_out,
        model.ds_block_6.dw_bn_gamma,
        model.ds_block_6.dw_bn_beta,
        model.ds_block_6.dw_bn_running_mean,
        model.ds_block_6.dw_bn_running_var,
        True,
    )
    model.ds_block_6.dw_bn_running_mean = b6_dw_bn_tuple[1]
    model.ds_block_6.dw_bn_running_var = b6_dw_bn_tuple[2]
    var b6_dw_bn_out = b6_dw_bn_tuple[0]
    var b6_dw_relu_out = relu(b6_dw_bn_out)
    var b6_pw_out = conv2d(
        b6_dw_relu_out,
        model.ds_block_6.pw_weights,
        model.ds_block_6.pw_bias,
        stride=1,
        padding=0,
    )
    var b6_pw_bn_tuple = batch_norm2d(
        b6_pw_out,
        model.ds_block_6.pw_bn_gamma,
        model.ds_block_6.pw_bn_beta,
        model.ds_block_6.pw_bn_running_mean,
        model.ds_block_6.pw_bn_running_var,
        True,
    )
    model.ds_block_6.pw_bn_running_mean = b6_pw_bn_tuple[1]
    model.ds_block_6.pw_bn_running_var = b6_pw_bn_tuple[2]
    var b6_pw_bn_out = b6_pw_bn_tuple[0]
    var b6_out = relu(b6_pw_bn_out)

    # ---- Block 7 (in=512, out=512, stride=1) ----
    var b7_in = b6_out
    var b7_dw_out = depthwise_conv2d(
        b7_in,
        model.ds_block_7.dw_weights,
        model.ds_block_7.dw_bias,
        stride=1,
        padding=1,
    )
    var b7_dw_bn_tuple = batch_norm2d(
        b7_dw_out,
        model.ds_block_7.dw_bn_gamma,
        model.ds_block_7.dw_bn_beta,
        model.ds_block_7.dw_bn_running_mean,
        model.ds_block_7.dw_bn_running_var,
        True,
    )
    model.ds_block_7.dw_bn_running_mean = b7_dw_bn_tuple[1]
    model.ds_block_7.dw_bn_running_var = b7_dw_bn_tuple[2]
    var b7_dw_bn_out = b7_dw_bn_tuple[0]
    var b7_dw_relu_out = relu(b7_dw_bn_out)
    var b7_pw_out = conv2d(
        b7_dw_relu_out,
        model.ds_block_7.pw_weights,
        model.ds_block_7.pw_bias,
        stride=1,
        padding=0,
    )
    var b7_pw_bn_tuple = batch_norm2d(
        b7_pw_out,
        model.ds_block_7.pw_bn_gamma,
        model.ds_block_7.pw_bn_beta,
        model.ds_block_7.pw_bn_running_mean,
        model.ds_block_7.pw_bn_running_var,
        True,
    )
    model.ds_block_7.pw_bn_running_mean = b7_pw_bn_tuple[1]
    model.ds_block_7.pw_bn_running_var = b7_pw_bn_tuple[2]
    var b7_pw_bn_out = b7_pw_bn_tuple[0]
    var b7_out = relu(b7_pw_bn_out)

    # ---- Block 8 (in=512, out=512, stride=1) ----
    var b8_in = b7_out
    var b8_dw_out = depthwise_conv2d(
        b8_in,
        model.ds_block_8.dw_weights,
        model.ds_block_8.dw_bias,
        stride=1,
        padding=1,
    )
    var b8_dw_bn_tuple = batch_norm2d(
        b8_dw_out,
        model.ds_block_8.dw_bn_gamma,
        model.ds_block_8.dw_bn_beta,
        model.ds_block_8.dw_bn_running_mean,
        model.ds_block_8.dw_bn_running_var,
        True,
    )
    model.ds_block_8.dw_bn_running_mean = b8_dw_bn_tuple[1]
    model.ds_block_8.dw_bn_running_var = b8_dw_bn_tuple[2]
    var b8_dw_bn_out = b8_dw_bn_tuple[0]
    var b8_dw_relu_out = relu(b8_dw_bn_out)
    var b8_pw_out = conv2d(
        b8_dw_relu_out,
        model.ds_block_8.pw_weights,
        model.ds_block_8.pw_bias,
        stride=1,
        padding=0,
    )
    var b8_pw_bn_tuple = batch_norm2d(
        b8_pw_out,
        model.ds_block_8.pw_bn_gamma,
        model.ds_block_8.pw_bn_beta,
        model.ds_block_8.pw_bn_running_mean,
        model.ds_block_8.pw_bn_running_var,
        True,
    )
    model.ds_block_8.pw_bn_running_mean = b8_pw_bn_tuple[1]
    model.ds_block_8.pw_bn_running_var = b8_pw_bn_tuple[2]
    var b8_pw_bn_out = b8_pw_bn_tuple[0]
    var b8_out = relu(b8_pw_bn_out)

    # ---- Block 9 (in=512, out=512, stride=1) ----
    var b9_in = b8_out
    var b9_dw_out = depthwise_conv2d(
        b9_in,
        model.ds_block_9.dw_weights,
        model.ds_block_9.dw_bias,
        stride=1,
        padding=1,
    )
    var b9_dw_bn_tuple = batch_norm2d(
        b9_dw_out,
        model.ds_block_9.dw_bn_gamma,
        model.ds_block_9.dw_bn_beta,
        model.ds_block_9.dw_bn_running_mean,
        model.ds_block_9.dw_bn_running_var,
        True,
    )
    model.ds_block_9.dw_bn_running_mean = b9_dw_bn_tuple[1]
    model.ds_block_9.dw_bn_running_var = b9_dw_bn_tuple[2]
    var b9_dw_bn_out = b9_dw_bn_tuple[0]
    var b9_dw_relu_out = relu(b9_dw_bn_out)
    var b9_pw_out = conv2d(
        b9_dw_relu_out,
        model.ds_block_9.pw_weights,
        model.ds_block_9.pw_bias,
        stride=1,
        padding=0,
    )
    var b9_pw_bn_tuple = batch_norm2d(
        b9_pw_out,
        model.ds_block_9.pw_bn_gamma,
        model.ds_block_9.pw_bn_beta,
        model.ds_block_9.pw_bn_running_mean,
        model.ds_block_9.pw_bn_running_var,
        True,
    )
    model.ds_block_9.pw_bn_running_mean = b9_pw_bn_tuple[1]
    model.ds_block_9.pw_bn_running_var = b9_pw_bn_tuple[2]
    var b9_pw_bn_out = b9_pw_bn_tuple[0]
    var b9_out = relu(b9_pw_bn_out)

    # ---- Block 10 (in=512, out=512, stride=1) ----
    var b10_in = b9_out
    var b10_dw_out = depthwise_conv2d(
        b10_in,
        model.ds_block_10.dw_weights,
        model.ds_block_10.dw_bias,
        stride=1,
        padding=1,
    )
    var b10_dw_bn_tuple = batch_norm2d(
        b10_dw_out,
        model.ds_block_10.dw_bn_gamma,
        model.ds_block_10.dw_bn_beta,
        model.ds_block_10.dw_bn_running_mean,
        model.ds_block_10.dw_bn_running_var,
        True,
    )
    model.ds_block_10.dw_bn_running_mean = b10_dw_bn_tuple[1]
    model.ds_block_10.dw_bn_running_var = b10_dw_bn_tuple[2]
    var b10_dw_bn_out = b10_dw_bn_tuple[0]
    var b10_dw_relu_out = relu(b10_dw_bn_out)
    var b10_pw_out = conv2d(
        b10_dw_relu_out,
        model.ds_block_10.pw_weights,
        model.ds_block_10.pw_bias,
        stride=1,
        padding=0,
    )
    var b10_pw_bn_tuple = batch_norm2d(
        b10_pw_out,
        model.ds_block_10.pw_bn_gamma,
        model.ds_block_10.pw_bn_beta,
        model.ds_block_10.pw_bn_running_mean,
        model.ds_block_10.pw_bn_running_var,
        True,
    )
    model.ds_block_10.pw_bn_running_mean = b10_pw_bn_tuple[1]
    model.ds_block_10.pw_bn_running_var = b10_pw_bn_tuple[2]
    var b10_pw_bn_out = b10_pw_bn_tuple[0]
    var b10_out = relu(b10_pw_bn_out)

    # ---- Block 11 (in=512, out=512, stride=1) ----
    var b11_in = b10_out
    var b11_dw_out = depthwise_conv2d(
        b11_in,
        model.ds_block_11.dw_weights,
        model.ds_block_11.dw_bias,
        stride=1,
        padding=1,
    )
    var b11_dw_bn_tuple = batch_norm2d(
        b11_dw_out,
        model.ds_block_11.dw_bn_gamma,
        model.ds_block_11.dw_bn_beta,
        model.ds_block_11.dw_bn_running_mean,
        model.ds_block_11.dw_bn_running_var,
        True,
    )
    model.ds_block_11.dw_bn_running_mean = b11_dw_bn_tuple[1]
    model.ds_block_11.dw_bn_running_var = b11_dw_bn_tuple[2]
    var b11_dw_bn_out = b11_dw_bn_tuple[0]
    var b11_dw_relu_out = relu(b11_dw_bn_out)
    var b11_pw_out = conv2d(
        b11_dw_relu_out,
        model.ds_block_11.pw_weights,
        model.ds_block_11.pw_bias,
        stride=1,
        padding=0,
    )
    var b11_pw_bn_tuple = batch_norm2d(
        b11_pw_out,
        model.ds_block_11.pw_bn_gamma,
        model.ds_block_11.pw_bn_beta,
        model.ds_block_11.pw_bn_running_mean,
        model.ds_block_11.pw_bn_running_var,
        True,
    )
    model.ds_block_11.pw_bn_running_mean = b11_pw_bn_tuple[1]
    model.ds_block_11.pw_bn_running_var = b11_pw_bn_tuple[2]
    var b11_pw_bn_out = b11_pw_bn_tuple[0]
    var b11_out = relu(b11_pw_bn_out)

    # ---- Block 12 (in=512, out=1024, stride=2) ----
    var b12_in = b11_out
    var b12_dw_out = depthwise_conv2d(
        b12_in,
        model.ds_block_12.dw_weights,
        model.ds_block_12.dw_bias,
        stride=2,
        padding=1,
    )
    var b12_dw_bn_tuple = batch_norm2d(
        b12_dw_out,
        model.ds_block_12.dw_bn_gamma,
        model.ds_block_12.dw_bn_beta,
        model.ds_block_12.dw_bn_running_mean,
        model.ds_block_12.dw_bn_running_var,
        True,
    )
    model.ds_block_12.dw_bn_running_mean = b12_dw_bn_tuple[1]
    model.ds_block_12.dw_bn_running_var = b12_dw_bn_tuple[2]
    var b12_dw_bn_out = b12_dw_bn_tuple[0]
    var b12_dw_relu_out = relu(b12_dw_bn_out)
    var b12_pw_out = conv2d(
        b12_dw_relu_out,
        model.ds_block_12.pw_weights,
        model.ds_block_12.pw_bias,
        stride=1,
        padding=0,
    )
    var b12_pw_bn_tuple = batch_norm2d(
        b12_pw_out,
        model.ds_block_12.pw_bn_gamma,
        model.ds_block_12.pw_bn_beta,
        model.ds_block_12.pw_bn_running_mean,
        model.ds_block_12.pw_bn_running_var,
        True,
    )
    model.ds_block_12.pw_bn_running_mean = b12_pw_bn_tuple[1]
    model.ds_block_12.pw_bn_running_var = b12_pw_bn_tuple[2]
    var b12_pw_bn_out = b12_pw_bn_tuple[0]
    var b12_out = relu(b12_pw_bn_out)

    # ---- Block 13 (in=1024, out=1024, stride=1) ----
    var b13_in = b12_out
    var b13_dw_out = depthwise_conv2d(
        b13_in,
        model.ds_block_13.dw_weights,
        model.ds_block_13.dw_bias,
        stride=1,
        padding=1,
    )
    var b13_dw_bn_tuple = batch_norm2d(
        b13_dw_out,
        model.ds_block_13.dw_bn_gamma,
        model.ds_block_13.dw_bn_beta,
        model.ds_block_13.dw_bn_running_mean,
        model.ds_block_13.dw_bn_running_var,
        True,
    )
    model.ds_block_13.dw_bn_running_mean = b13_dw_bn_tuple[1]
    model.ds_block_13.dw_bn_running_var = b13_dw_bn_tuple[2]
    var b13_dw_bn_out = b13_dw_bn_tuple[0]
    var b13_dw_relu_out = relu(b13_dw_bn_out)
    var b13_pw_out = conv2d(
        b13_dw_relu_out,
        model.ds_block_13.pw_weights,
        model.ds_block_13.pw_bias,
        stride=1,
        padding=0,
    )
    var b13_pw_bn_tuple = batch_norm2d(
        b13_pw_out,
        model.ds_block_13.pw_bn_gamma,
        model.ds_block_13.pw_bn_beta,
        model.ds_block_13.pw_bn_running_mean,
        model.ds_block_13.pw_bn_running_var,
        True,
    )
    model.ds_block_13.pw_bn_running_mean = b13_pw_bn_tuple[1]
    model.ds_block_13.pw_bn_running_var = b13_pw_bn_tuple[2]
    var b13_pw_bn_out = b13_pw_bn_tuple[0]
    var b13_out = relu(b13_pw_bn_out)

    # ---- Global avg pool + flatten + FC ----
    var gap_out = global_avgpool2d(b13_out)
    var flat_shape: List[Int] = [gap_out.shape()[0], 1024]
    var flat = gap_out.reshape(flat_shape)
    var logits = linear(flat, model.fc_weights, model.fc_bias)
    var loss_t = cross_entropy(logits, labels_onehot)
    var loss = loss_t.load[DType.float32](0)

    # ===== Backward pass (reverse; mirrors VGG16 template style) =====

    var grad_seed_shape: List[Int] = [1]
    var grad_seed = zeros(grad_seed_shape, logits.dtype())
    grad_seed.set(0, Float32(1.0))
    var grad_logits = cross_entropy_backward(grad_seed, logits, labels_onehot)

    var fc_grads = linear_backward(grad_logits, flat, model.fc_weights)
    var grad_flat = fc_grads.grad_input
    var grad_fc_w = fc_grads.grad_weights
    var grad_fc_b = fc_grads.grad_bias

    var grad_gap_out = grad_flat.reshape(gap_out.shape())
    var grad_b13_out = global_avgpool2d_backward(grad_gap_out, b13_out)

    # ---- Block 13 backward (in=1024, out=1024, stride=1) ----
    var grad_b13_pw_bn_out = relu_backward(grad_b13_out, b13_pw_bn_out)
    var b13_pw_bn_bwd = batch_norm2d_backward(
        grad_b13_pw_bn_out,
        b13_pw_out,
        model.ds_block_13.pw_bn_gamma,
        model.ds_block_13.pw_bn_running_mean,
        model.ds_block_13.pw_bn_running_var,
        True,
    )
    var grad_b13_pw_out = b13_pw_bn_bwd[0]
    var grad_b13_pw_gamma = b13_pw_bn_bwd[1]
    var grad_b13_pw_beta = b13_pw_bn_bwd[2]
    var b13_pw_grads = conv2d_backward(
        grad_b13_pw_out,
        b13_dw_relu_out,
        model.ds_block_13.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b13_dw_relu_out = b13_pw_grads.grad_input
    var grad_b13_pw_w = b13_pw_grads.grad_weights
    var grad_b13_pw_b = b13_pw_grads.grad_bias

    var grad_b13_dw_bn_out = relu_backward(grad_b13_dw_relu_out, b13_dw_bn_out)
    var b13_dw_bn_bwd = batch_norm2d_backward(
        grad_b13_dw_bn_out,
        b13_dw_out,
        model.ds_block_13.dw_bn_gamma,
        model.ds_block_13.dw_bn_running_mean,
        model.ds_block_13.dw_bn_running_var,
        True,
    )
    var grad_b13_dw_out = b13_dw_bn_bwd[0]
    var grad_b13_dw_gamma = b13_dw_bn_bwd[1]
    var grad_b13_dw_beta = b13_dw_bn_bwd[2]
    var b13_dw_grads = depthwise_conv2d_backward(
        grad_b13_dw_out,
        b13_in,
        model.ds_block_13.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b13_in = b13_dw_grads.grad_input
    var grad_b13_dw_w = b13_dw_grads.grad_weights
    var grad_b13_dw_b = b13_dw_grads.grad_bias

    # ---- Block 12 backward (in=512, out=1024, stride=2) ----
    var grad_b12_out = grad_b13_in
    var grad_b12_pw_bn_out = relu_backward(grad_b12_out, b12_pw_bn_out)
    var b12_pw_bn_bwd = batch_norm2d_backward(
        grad_b12_pw_bn_out,
        b12_pw_out,
        model.ds_block_12.pw_bn_gamma,
        model.ds_block_12.pw_bn_running_mean,
        model.ds_block_12.pw_bn_running_var,
        True,
    )
    var grad_b12_pw_out = b12_pw_bn_bwd[0]
    var grad_b12_pw_gamma = b12_pw_bn_bwd[1]
    var grad_b12_pw_beta = b12_pw_bn_bwd[2]
    var b12_pw_grads = conv2d_backward(
        grad_b12_pw_out,
        b12_dw_relu_out,
        model.ds_block_12.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b12_dw_relu_out = b12_pw_grads.grad_input
    var grad_b12_pw_w = b12_pw_grads.grad_weights
    var grad_b12_pw_b = b12_pw_grads.grad_bias

    var grad_b12_dw_bn_out = relu_backward(grad_b12_dw_relu_out, b12_dw_bn_out)
    var b12_dw_bn_bwd = batch_norm2d_backward(
        grad_b12_dw_bn_out,
        b12_dw_out,
        model.ds_block_12.dw_bn_gamma,
        model.ds_block_12.dw_bn_running_mean,
        model.ds_block_12.dw_bn_running_var,
        True,
    )
    var grad_b12_dw_out = b12_dw_bn_bwd[0]
    var grad_b12_dw_gamma = b12_dw_bn_bwd[1]
    var grad_b12_dw_beta = b12_dw_bn_bwd[2]
    var b12_dw_grads = depthwise_conv2d_backward(
        grad_b12_dw_out,
        b12_in,
        model.ds_block_12.dw_weights,
        stride=2,
        padding=1,
    )
    var grad_b12_in = b12_dw_grads.grad_input
    var grad_b12_dw_w = b12_dw_grads.grad_weights
    var grad_b12_dw_b = b12_dw_grads.grad_bias

    # ---- Block 11 backward (in=512, out=512, stride=1) ----
    var grad_b11_out = grad_b12_in
    var grad_b11_pw_bn_out = relu_backward(grad_b11_out, b11_pw_bn_out)
    var b11_pw_bn_bwd = batch_norm2d_backward(
        grad_b11_pw_bn_out,
        b11_pw_out,
        model.ds_block_11.pw_bn_gamma,
        model.ds_block_11.pw_bn_running_mean,
        model.ds_block_11.pw_bn_running_var,
        True,
    )
    var grad_b11_pw_out = b11_pw_bn_bwd[0]
    var grad_b11_pw_gamma = b11_pw_bn_bwd[1]
    var grad_b11_pw_beta = b11_pw_bn_bwd[2]
    var b11_pw_grads = conv2d_backward(
        grad_b11_pw_out,
        b11_dw_relu_out,
        model.ds_block_11.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b11_dw_relu_out = b11_pw_grads.grad_input
    var grad_b11_pw_w = b11_pw_grads.grad_weights
    var grad_b11_pw_b = b11_pw_grads.grad_bias

    var grad_b11_dw_bn_out = relu_backward(grad_b11_dw_relu_out, b11_dw_bn_out)
    var b11_dw_bn_bwd = batch_norm2d_backward(
        grad_b11_dw_bn_out,
        b11_dw_out,
        model.ds_block_11.dw_bn_gamma,
        model.ds_block_11.dw_bn_running_mean,
        model.ds_block_11.dw_bn_running_var,
        True,
    )
    var grad_b11_dw_out = b11_dw_bn_bwd[0]
    var grad_b11_dw_gamma = b11_dw_bn_bwd[1]
    var grad_b11_dw_beta = b11_dw_bn_bwd[2]
    var b11_dw_grads = depthwise_conv2d_backward(
        grad_b11_dw_out,
        b11_in,
        model.ds_block_11.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b11_in = b11_dw_grads.grad_input
    var grad_b11_dw_w = b11_dw_grads.grad_weights
    var grad_b11_dw_b = b11_dw_grads.grad_bias

    # ---- Block 10 backward (in=512, out=512, stride=1) ----
    var grad_b10_out = grad_b11_in
    var grad_b10_pw_bn_out = relu_backward(grad_b10_out, b10_pw_bn_out)
    var b10_pw_bn_bwd = batch_norm2d_backward(
        grad_b10_pw_bn_out,
        b10_pw_out,
        model.ds_block_10.pw_bn_gamma,
        model.ds_block_10.pw_bn_running_mean,
        model.ds_block_10.pw_bn_running_var,
        True,
    )
    var grad_b10_pw_out = b10_pw_bn_bwd[0]
    var grad_b10_pw_gamma = b10_pw_bn_bwd[1]
    var grad_b10_pw_beta = b10_pw_bn_bwd[2]
    var b10_pw_grads = conv2d_backward(
        grad_b10_pw_out,
        b10_dw_relu_out,
        model.ds_block_10.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b10_dw_relu_out = b10_pw_grads.grad_input
    var grad_b10_pw_w = b10_pw_grads.grad_weights
    var grad_b10_pw_b = b10_pw_grads.grad_bias

    var grad_b10_dw_bn_out = relu_backward(grad_b10_dw_relu_out, b10_dw_bn_out)
    var b10_dw_bn_bwd = batch_norm2d_backward(
        grad_b10_dw_bn_out,
        b10_dw_out,
        model.ds_block_10.dw_bn_gamma,
        model.ds_block_10.dw_bn_running_mean,
        model.ds_block_10.dw_bn_running_var,
        True,
    )
    var grad_b10_dw_out = b10_dw_bn_bwd[0]
    var grad_b10_dw_gamma = b10_dw_bn_bwd[1]
    var grad_b10_dw_beta = b10_dw_bn_bwd[2]
    var b10_dw_grads = depthwise_conv2d_backward(
        grad_b10_dw_out,
        b10_in,
        model.ds_block_10.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b10_in = b10_dw_grads.grad_input
    var grad_b10_dw_w = b10_dw_grads.grad_weights
    var grad_b10_dw_b = b10_dw_grads.grad_bias

    # ---- Block 9 backward (in=512, out=512, stride=1) ----
    var grad_b9_out = grad_b10_in
    var grad_b9_pw_bn_out = relu_backward(grad_b9_out, b9_pw_bn_out)
    var b9_pw_bn_bwd = batch_norm2d_backward(
        grad_b9_pw_bn_out,
        b9_pw_out,
        model.ds_block_9.pw_bn_gamma,
        model.ds_block_9.pw_bn_running_mean,
        model.ds_block_9.pw_bn_running_var,
        True,
    )
    var grad_b9_pw_out = b9_pw_bn_bwd[0]
    var grad_b9_pw_gamma = b9_pw_bn_bwd[1]
    var grad_b9_pw_beta = b9_pw_bn_bwd[2]
    var b9_pw_grads = conv2d_backward(
        grad_b9_pw_out,
        b9_dw_relu_out,
        model.ds_block_9.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b9_dw_relu_out = b9_pw_grads.grad_input
    var grad_b9_pw_w = b9_pw_grads.grad_weights
    var grad_b9_pw_b = b9_pw_grads.grad_bias

    var grad_b9_dw_bn_out = relu_backward(grad_b9_dw_relu_out, b9_dw_bn_out)
    var b9_dw_bn_bwd = batch_norm2d_backward(
        grad_b9_dw_bn_out,
        b9_dw_out,
        model.ds_block_9.dw_bn_gamma,
        model.ds_block_9.dw_bn_running_mean,
        model.ds_block_9.dw_bn_running_var,
        True,
    )
    var grad_b9_dw_out = b9_dw_bn_bwd[0]
    var grad_b9_dw_gamma = b9_dw_bn_bwd[1]
    var grad_b9_dw_beta = b9_dw_bn_bwd[2]
    var b9_dw_grads = depthwise_conv2d_backward(
        grad_b9_dw_out,
        b9_in,
        model.ds_block_9.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b9_in = b9_dw_grads.grad_input
    var grad_b9_dw_w = b9_dw_grads.grad_weights
    var grad_b9_dw_b = b9_dw_grads.grad_bias

    # ---- Block 8 backward (in=512, out=512, stride=1) ----
    var grad_b8_out = grad_b9_in
    var grad_b8_pw_bn_out = relu_backward(grad_b8_out, b8_pw_bn_out)
    var b8_pw_bn_bwd = batch_norm2d_backward(
        grad_b8_pw_bn_out,
        b8_pw_out,
        model.ds_block_8.pw_bn_gamma,
        model.ds_block_8.pw_bn_running_mean,
        model.ds_block_8.pw_bn_running_var,
        True,
    )
    var grad_b8_pw_out = b8_pw_bn_bwd[0]
    var grad_b8_pw_gamma = b8_pw_bn_bwd[1]
    var grad_b8_pw_beta = b8_pw_bn_bwd[2]
    var b8_pw_grads = conv2d_backward(
        grad_b8_pw_out,
        b8_dw_relu_out,
        model.ds_block_8.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b8_dw_relu_out = b8_pw_grads.grad_input
    var grad_b8_pw_w = b8_pw_grads.grad_weights
    var grad_b8_pw_b = b8_pw_grads.grad_bias

    var grad_b8_dw_bn_out = relu_backward(grad_b8_dw_relu_out, b8_dw_bn_out)
    var b8_dw_bn_bwd = batch_norm2d_backward(
        grad_b8_dw_bn_out,
        b8_dw_out,
        model.ds_block_8.dw_bn_gamma,
        model.ds_block_8.dw_bn_running_mean,
        model.ds_block_8.dw_bn_running_var,
        True,
    )
    var grad_b8_dw_out = b8_dw_bn_bwd[0]
    var grad_b8_dw_gamma = b8_dw_bn_bwd[1]
    var grad_b8_dw_beta = b8_dw_bn_bwd[2]
    var b8_dw_grads = depthwise_conv2d_backward(
        grad_b8_dw_out,
        b8_in,
        model.ds_block_8.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b8_in = b8_dw_grads.grad_input
    var grad_b8_dw_w = b8_dw_grads.grad_weights
    var grad_b8_dw_b = b8_dw_grads.grad_bias

    # ---- Block 7 backward (in=512, out=512, stride=1) ----
    var grad_b7_out = grad_b8_in
    var grad_b7_pw_bn_out = relu_backward(grad_b7_out, b7_pw_bn_out)
    var b7_pw_bn_bwd = batch_norm2d_backward(
        grad_b7_pw_bn_out,
        b7_pw_out,
        model.ds_block_7.pw_bn_gamma,
        model.ds_block_7.pw_bn_running_mean,
        model.ds_block_7.pw_bn_running_var,
        True,
    )
    var grad_b7_pw_out = b7_pw_bn_bwd[0]
    var grad_b7_pw_gamma = b7_pw_bn_bwd[1]
    var grad_b7_pw_beta = b7_pw_bn_bwd[2]
    var b7_pw_grads = conv2d_backward(
        grad_b7_pw_out,
        b7_dw_relu_out,
        model.ds_block_7.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b7_dw_relu_out = b7_pw_grads.grad_input
    var grad_b7_pw_w = b7_pw_grads.grad_weights
    var grad_b7_pw_b = b7_pw_grads.grad_bias

    var grad_b7_dw_bn_out = relu_backward(grad_b7_dw_relu_out, b7_dw_bn_out)
    var b7_dw_bn_bwd = batch_norm2d_backward(
        grad_b7_dw_bn_out,
        b7_dw_out,
        model.ds_block_7.dw_bn_gamma,
        model.ds_block_7.dw_bn_running_mean,
        model.ds_block_7.dw_bn_running_var,
        True,
    )
    var grad_b7_dw_out = b7_dw_bn_bwd[0]
    var grad_b7_dw_gamma = b7_dw_bn_bwd[1]
    var grad_b7_dw_beta = b7_dw_bn_bwd[2]
    var b7_dw_grads = depthwise_conv2d_backward(
        grad_b7_dw_out,
        b7_in,
        model.ds_block_7.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b7_in = b7_dw_grads.grad_input
    var grad_b7_dw_w = b7_dw_grads.grad_weights
    var grad_b7_dw_b = b7_dw_grads.grad_bias

    # ---- Block 6 backward (in=256, out=512, stride=2) ----
    var grad_b6_out = grad_b7_in
    var grad_b6_pw_bn_out = relu_backward(grad_b6_out, b6_pw_bn_out)
    var b6_pw_bn_bwd = batch_norm2d_backward(
        grad_b6_pw_bn_out,
        b6_pw_out,
        model.ds_block_6.pw_bn_gamma,
        model.ds_block_6.pw_bn_running_mean,
        model.ds_block_6.pw_bn_running_var,
        True,
    )
    var grad_b6_pw_out = b6_pw_bn_bwd[0]
    var grad_b6_pw_gamma = b6_pw_bn_bwd[1]
    var grad_b6_pw_beta = b6_pw_bn_bwd[2]
    var b6_pw_grads = conv2d_backward(
        grad_b6_pw_out,
        b6_dw_relu_out,
        model.ds_block_6.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b6_dw_relu_out = b6_pw_grads.grad_input
    var grad_b6_pw_w = b6_pw_grads.grad_weights
    var grad_b6_pw_b = b6_pw_grads.grad_bias

    var grad_b6_dw_bn_out = relu_backward(grad_b6_dw_relu_out, b6_dw_bn_out)
    var b6_dw_bn_bwd = batch_norm2d_backward(
        grad_b6_dw_bn_out,
        b6_dw_out,
        model.ds_block_6.dw_bn_gamma,
        model.ds_block_6.dw_bn_running_mean,
        model.ds_block_6.dw_bn_running_var,
        True,
    )
    var grad_b6_dw_out = b6_dw_bn_bwd[0]
    var grad_b6_dw_gamma = b6_dw_bn_bwd[1]
    var grad_b6_dw_beta = b6_dw_bn_bwd[2]
    var b6_dw_grads = depthwise_conv2d_backward(
        grad_b6_dw_out,
        b6_in,
        model.ds_block_6.dw_weights,
        stride=2,
        padding=1,
    )
    var grad_b6_in = b6_dw_grads.grad_input
    var grad_b6_dw_w = b6_dw_grads.grad_weights
    var grad_b6_dw_b = b6_dw_grads.grad_bias

    # ---- Block 5 backward (in=256, out=256, stride=1) ----
    var grad_b5_out = grad_b6_in
    var grad_b5_pw_bn_out = relu_backward(grad_b5_out, b5_pw_bn_out)
    var b5_pw_bn_bwd = batch_norm2d_backward(
        grad_b5_pw_bn_out,
        b5_pw_out,
        model.ds_block_5.pw_bn_gamma,
        model.ds_block_5.pw_bn_running_mean,
        model.ds_block_5.pw_bn_running_var,
        True,
    )
    var grad_b5_pw_out = b5_pw_bn_bwd[0]
    var grad_b5_pw_gamma = b5_pw_bn_bwd[1]
    var grad_b5_pw_beta = b5_pw_bn_bwd[2]
    var b5_pw_grads = conv2d_backward(
        grad_b5_pw_out,
        b5_dw_relu_out,
        model.ds_block_5.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b5_dw_relu_out = b5_pw_grads.grad_input
    var grad_b5_pw_w = b5_pw_grads.grad_weights
    var grad_b5_pw_b = b5_pw_grads.grad_bias

    var grad_b5_dw_bn_out = relu_backward(grad_b5_dw_relu_out, b5_dw_bn_out)
    var b5_dw_bn_bwd = batch_norm2d_backward(
        grad_b5_dw_bn_out,
        b5_dw_out,
        model.ds_block_5.dw_bn_gamma,
        model.ds_block_5.dw_bn_running_mean,
        model.ds_block_5.dw_bn_running_var,
        True,
    )
    var grad_b5_dw_out = b5_dw_bn_bwd[0]
    var grad_b5_dw_gamma = b5_dw_bn_bwd[1]
    var grad_b5_dw_beta = b5_dw_bn_bwd[2]
    var b5_dw_grads = depthwise_conv2d_backward(
        grad_b5_dw_out,
        b5_in,
        model.ds_block_5.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b5_in = b5_dw_grads.grad_input
    var grad_b5_dw_w = b5_dw_grads.grad_weights
    var grad_b5_dw_b = b5_dw_grads.grad_bias

    # ---- Block 4 backward (in=128, out=256, stride=2) ----
    var grad_b4_out = grad_b5_in
    var grad_b4_pw_bn_out = relu_backward(grad_b4_out, b4_pw_bn_out)
    var b4_pw_bn_bwd = batch_norm2d_backward(
        grad_b4_pw_bn_out,
        b4_pw_out,
        model.ds_block_4.pw_bn_gamma,
        model.ds_block_4.pw_bn_running_mean,
        model.ds_block_4.pw_bn_running_var,
        True,
    )
    var grad_b4_pw_out = b4_pw_bn_bwd[0]
    var grad_b4_pw_gamma = b4_pw_bn_bwd[1]
    var grad_b4_pw_beta = b4_pw_bn_bwd[2]
    var b4_pw_grads = conv2d_backward(
        grad_b4_pw_out,
        b4_dw_relu_out,
        model.ds_block_4.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b4_dw_relu_out = b4_pw_grads.grad_input
    var grad_b4_pw_w = b4_pw_grads.grad_weights
    var grad_b4_pw_b = b4_pw_grads.grad_bias

    var grad_b4_dw_bn_out = relu_backward(grad_b4_dw_relu_out, b4_dw_bn_out)
    var b4_dw_bn_bwd = batch_norm2d_backward(
        grad_b4_dw_bn_out,
        b4_dw_out,
        model.ds_block_4.dw_bn_gamma,
        model.ds_block_4.dw_bn_running_mean,
        model.ds_block_4.dw_bn_running_var,
        True,
    )
    var grad_b4_dw_out = b4_dw_bn_bwd[0]
    var grad_b4_dw_gamma = b4_dw_bn_bwd[1]
    var grad_b4_dw_beta = b4_dw_bn_bwd[2]
    var b4_dw_grads = depthwise_conv2d_backward(
        grad_b4_dw_out,
        b4_in,
        model.ds_block_4.dw_weights,
        stride=2,
        padding=1,
    )
    var grad_b4_in = b4_dw_grads.grad_input
    var grad_b4_dw_w = b4_dw_grads.grad_weights
    var grad_b4_dw_b = b4_dw_grads.grad_bias

    # ---- Block 3 backward (in=128, out=128, stride=1) ----
    var grad_b3_out = grad_b4_in
    var grad_b3_pw_bn_out = relu_backward(grad_b3_out, b3_pw_bn_out)
    var b3_pw_bn_bwd = batch_norm2d_backward(
        grad_b3_pw_bn_out,
        b3_pw_out,
        model.ds_block_3.pw_bn_gamma,
        model.ds_block_3.pw_bn_running_mean,
        model.ds_block_3.pw_bn_running_var,
        True,
    )
    var grad_b3_pw_out = b3_pw_bn_bwd[0]
    var grad_b3_pw_gamma = b3_pw_bn_bwd[1]
    var grad_b3_pw_beta = b3_pw_bn_bwd[2]
    var b3_pw_grads = conv2d_backward(
        grad_b3_pw_out,
        b3_dw_relu_out,
        model.ds_block_3.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b3_dw_relu_out = b3_pw_grads.grad_input
    var grad_b3_pw_w = b3_pw_grads.grad_weights
    var grad_b3_pw_b = b3_pw_grads.grad_bias

    var grad_b3_dw_bn_out = relu_backward(grad_b3_dw_relu_out, b3_dw_bn_out)
    var b3_dw_bn_bwd = batch_norm2d_backward(
        grad_b3_dw_bn_out,
        b3_dw_out,
        model.ds_block_3.dw_bn_gamma,
        model.ds_block_3.dw_bn_running_mean,
        model.ds_block_3.dw_bn_running_var,
        True,
    )
    var grad_b3_dw_out = b3_dw_bn_bwd[0]
    var grad_b3_dw_gamma = b3_dw_bn_bwd[1]
    var grad_b3_dw_beta = b3_dw_bn_bwd[2]
    var b3_dw_grads = depthwise_conv2d_backward(
        grad_b3_dw_out,
        b3_in,
        model.ds_block_3.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b3_in = b3_dw_grads.grad_input
    var grad_b3_dw_w = b3_dw_grads.grad_weights
    var grad_b3_dw_b = b3_dw_grads.grad_bias

    # ---- Block 2 backward (in=64, out=128, stride=2) ----
    var grad_b2_out = grad_b3_in
    var grad_b2_pw_bn_out = relu_backward(grad_b2_out, b2_pw_bn_out)
    var b2_pw_bn_bwd = batch_norm2d_backward(
        grad_b2_pw_bn_out,
        b2_pw_out,
        model.ds_block_2.pw_bn_gamma,
        model.ds_block_2.pw_bn_running_mean,
        model.ds_block_2.pw_bn_running_var,
        True,
    )
    var grad_b2_pw_out = b2_pw_bn_bwd[0]
    var grad_b2_pw_gamma = b2_pw_bn_bwd[1]
    var grad_b2_pw_beta = b2_pw_bn_bwd[2]
    var b2_pw_grads = conv2d_backward(
        grad_b2_pw_out,
        b2_dw_relu_out,
        model.ds_block_2.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b2_dw_relu_out = b2_pw_grads.grad_input
    var grad_b2_pw_w = b2_pw_grads.grad_weights
    var grad_b2_pw_b = b2_pw_grads.grad_bias

    var grad_b2_dw_bn_out = relu_backward(grad_b2_dw_relu_out, b2_dw_bn_out)
    var b2_dw_bn_bwd = batch_norm2d_backward(
        grad_b2_dw_bn_out,
        b2_dw_out,
        model.ds_block_2.dw_bn_gamma,
        model.ds_block_2.dw_bn_running_mean,
        model.ds_block_2.dw_bn_running_var,
        True,
    )
    var grad_b2_dw_out = b2_dw_bn_bwd[0]
    var grad_b2_dw_gamma = b2_dw_bn_bwd[1]
    var grad_b2_dw_beta = b2_dw_bn_bwd[2]
    var b2_dw_grads = depthwise_conv2d_backward(
        grad_b2_dw_out,
        b2_in,
        model.ds_block_2.dw_weights,
        stride=2,
        padding=1,
    )
    var grad_b2_in = b2_dw_grads.grad_input
    var grad_b2_dw_w = b2_dw_grads.grad_weights
    var grad_b2_dw_b = b2_dw_grads.grad_bias

    # ---- Block 1 backward (in=32, out=64, stride=1) ----
    var grad_b1_out = grad_b2_in
    var grad_b1_pw_bn_out = relu_backward(grad_b1_out, b1_pw_bn_out)
    var b1_pw_bn_bwd = batch_norm2d_backward(
        grad_b1_pw_bn_out,
        b1_pw_out,
        model.ds_block_1.pw_bn_gamma,
        model.ds_block_1.pw_bn_running_mean,
        model.ds_block_1.pw_bn_running_var,
        True,
    )
    var grad_b1_pw_out = b1_pw_bn_bwd[0]
    var grad_b1_pw_gamma = b1_pw_bn_bwd[1]
    var grad_b1_pw_beta = b1_pw_bn_bwd[2]
    var b1_pw_grads = conv2d_backward(
        grad_b1_pw_out,
        b1_dw_relu_out,
        model.ds_block_1.pw_weights,
        stride=1,
        padding=0,
    )
    var grad_b1_dw_relu_out = b1_pw_grads.grad_input
    var grad_b1_pw_w = b1_pw_grads.grad_weights
    var grad_b1_pw_b = b1_pw_grads.grad_bias

    var grad_b1_dw_bn_out = relu_backward(grad_b1_dw_relu_out, b1_dw_bn_out)
    var b1_dw_bn_bwd = batch_norm2d_backward(
        grad_b1_dw_bn_out,
        b1_dw_out,
        model.ds_block_1.dw_bn_gamma,
        model.ds_block_1.dw_bn_running_mean,
        model.ds_block_1.dw_bn_running_var,
        True,
    )
    var grad_b1_dw_out = b1_dw_bn_bwd[0]
    var grad_b1_dw_gamma = b1_dw_bn_bwd[1]
    var grad_b1_dw_beta = b1_dw_bn_bwd[2]
    var b1_dw_grads = depthwise_conv2d_backward(
        grad_b1_dw_out,
        b1_in,
        model.ds_block_1.dw_weights,
        stride=1,
        padding=1,
    )
    var grad_b1_in = b1_dw_grads.grad_input
    var grad_b1_dw_w = b1_dw_grads.grad_weights
    var grad_b1_dw_b = b1_dw_grads.grad_bias

    # ---- Initial conv/BN/ReLU backward ----
    # grad_b1_in is defined in the block-1 backward stanza above and feeds the
    # initial-relu output-grad (init_relu_out feeds b1_in in the forward).
    var grad_init_bn_out = relu_backward(grad_b1_in, init_bn_out)
    var init_bn_bwd = batch_norm2d_backward(
        grad_init_bn_out,
        init_conv_out,
        model.initial_bn_gamma,
        model.initial_bn_running_mean,
        model.initial_bn_running_var,
        True,
    )
    var grad_init_conv_out = init_bn_bwd[0]
    var grad_init_bn_gamma = init_bn_bwd[1]
    var grad_init_bn_beta = init_bn_bwd[2]
    var init_conv_grads = conv2d_backward(
        grad_init_conv_out,
        input,
        model.initial_conv_weights,
        stride=2,
        padding=1,
    )
    var grad_init_conv_w = init_conv_grads.grad_weights
    var grad_init_conv_b = init_conv_grads.grad_bias

    # ===== SGD-momentum updates, 110 total, canonical order matching initialize_velocities =====
    # Layout: [0..3] initial(4), [4..107] blocks 1..13 × 8 each, [108..109] fc(2)
    sgd_momentum_update_inplace(
        model.initial_conv_weights,
        grad_init_conv_w,
        velocities[0],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_conv_bias,
        grad_init_conv_b,
        velocities[1],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_bn_gamma,
        grad_init_bn_gamma,
        velocities[2],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.initial_bn_beta,
        grad_init_bn_beta,
        velocities[3],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 1 (base=4):
    sgd_momentum_update_inplace(
        model.ds_block_1.dw_weights,
        grad_b1_dw_w,
        velocities[4],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.dw_bias,
        grad_b1_dw_b,
        velocities[5],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.dw_bn_gamma,
        grad_b1_dw_gamma,
        velocities[6],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.dw_bn_beta,
        grad_b1_dw_beta,
        velocities[7],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.pw_weights,
        grad_b1_pw_w,
        velocities[8],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.pw_bias,
        grad_b1_pw_b,
        velocities[9],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.pw_bn_gamma,
        grad_b1_pw_gamma,
        velocities[10],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_1.pw_bn_beta,
        grad_b1_pw_beta,
        velocities[11],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 2 (base=12):
    sgd_momentum_update_inplace(
        model.ds_block_2.dw_weights,
        grad_b2_dw_w,
        velocities[12],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.dw_bias,
        grad_b2_dw_b,
        velocities[13],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.dw_bn_gamma,
        grad_b2_dw_gamma,
        velocities[14],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.dw_bn_beta,
        grad_b2_dw_beta,
        velocities[15],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.pw_weights,
        grad_b2_pw_w,
        velocities[16],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.pw_bias,
        grad_b2_pw_b,
        velocities[17],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.pw_bn_gamma,
        grad_b2_pw_gamma,
        velocities[18],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_2.pw_bn_beta,
        grad_b2_pw_beta,
        velocities[19],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 3 (base=20):
    sgd_momentum_update_inplace(
        model.ds_block_3.dw_weights,
        grad_b3_dw_w,
        velocities[20],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.dw_bias,
        grad_b3_dw_b,
        velocities[21],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.dw_bn_gamma,
        grad_b3_dw_gamma,
        velocities[22],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.dw_bn_beta,
        grad_b3_dw_beta,
        velocities[23],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.pw_weights,
        grad_b3_pw_w,
        velocities[24],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.pw_bias,
        grad_b3_pw_b,
        velocities[25],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.pw_bn_gamma,
        grad_b3_pw_gamma,
        velocities[26],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_3.pw_bn_beta,
        grad_b3_pw_beta,
        velocities[27],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 4 (base=28):
    sgd_momentum_update_inplace(
        model.ds_block_4.dw_weights,
        grad_b4_dw_w,
        velocities[28],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.dw_bias,
        grad_b4_dw_b,
        velocities[29],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.dw_bn_gamma,
        grad_b4_dw_gamma,
        velocities[30],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.dw_bn_beta,
        grad_b4_dw_beta,
        velocities[31],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.pw_weights,
        grad_b4_pw_w,
        velocities[32],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.pw_bias,
        grad_b4_pw_b,
        velocities[33],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.pw_bn_gamma,
        grad_b4_pw_gamma,
        velocities[34],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_4.pw_bn_beta,
        grad_b4_pw_beta,
        velocities[35],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 5 (base=36):
    sgd_momentum_update_inplace(
        model.ds_block_5.dw_weights,
        grad_b5_dw_w,
        velocities[36],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.dw_bias,
        grad_b5_dw_b,
        velocities[37],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.dw_bn_gamma,
        grad_b5_dw_gamma,
        velocities[38],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.dw_bn_beta,
        grad_b5_dw_beta,
        velocities[39],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.pw_weights,
        grad_b5_pw_w,
        velocities[40],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.pw_bias,
        grad_b5_pw_b,
        velocities[41],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.pw_bn_gamma,
        grad_b5_pw_gamma,
        velocities[42],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_5.pw_bn_beta,
        grad_b5_pw_beta,
        velocities[43],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 6 (base=44):
    sgd_momentum_update_inplace(
        model.ds_block_6.dw_weights,
        grad_b6_dw_w,
        velocities[44],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.dw_bias,
        grad_b6_dw_b,
        velocities[45],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.dw_bn_gamma,
        grad_b6_dw_gamma,
        velocities[46],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.dw_bn_beta,
        grad_b6_dw_beta,
        velocities[47],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.pw_weights,
        grad_b6_pw_w,
        velocities[48],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.pw_bias,
        grad_b6_pw_b,
        velocities[49],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.pw_bn_gamma,
        grad_b6_pw_gamma,
        velocities[50],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_6.pw_bn_beta,
        grad_b6_pw_beta,
        velocities[51],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 7 (base=52):
    sgd_momentum_update_inplace(
        model.ds_block_7.dw_weights,
        grad_b7_dw_w,
        velocities[52],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.dw_bias,
        grad_b7_dw_b,
        velocities[53],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.dw_bn_gamma,
        grad_b7_dw_gamma,
        velocities[54],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.dw_bn_beta,
        grad_b7_dw_beta,
        velocities[55],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.pw_weights,
        grad_b7_pw_w,
        velocities[56],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.pw_bias,
        grad_b7_pw_b,
        velocities[57],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.pw_bn_gamma,
        grad_b7_pw_gamma,
        velocities[58],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_7.pw_bn_beta,
        grad_b7_pw_beta,
        velocities[59],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 8 (base=60):
    sgd_momentum_update_inplace(
        model.ds_block_8.dw_weights,
        grad_b8_dw_w,
        velocities[60],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.dw_bias,
        grad_b8_dw_b,
        velocities[61],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.dw_bn_gamma,
        grad_b8_dw_gamma,
        velocities[62],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.dw_bn_beta,
        grad_b8_dw_beta,
        velocities[63],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.pw_weights,
        grad_b8_pw_w,
        velocities[64],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.pw_bias,
        grad_b8_pw_b,
        velocities[65],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.pw_bn_gamma,
        grad_b8_pw_gamma,
        velocities[66],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_8.pw_bn_beta,
        grad_b8_pw_beta,
        velocities[67],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 9 (base=68):
    sgd_momentum_update_inplace(
        model.ds_block_9.dw_weights,
        grad_b9_dw_w,
        velocities[68],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.dw_bias,
        grad_b9_dw_b,
        velocities[69],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.dw_bn_gamma,
        grad_b9_dw_gamma,
        velocities[70],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.dw_bn_beta,
        grad_b9_dw_beta,
        velocities[71],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.pw_weights,
        grad_b9_pw_w,
        velocities[72],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.pw_bias,
        grad_b9_pw_b,
        velocities[73],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.pw_bn_gamma,
        grad_b9_pw_gamma,
        velocities[74],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_9.pw_bn_beta,
        grad_b9_pw_beta,
        velocities[75],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 10 (base=76):
    sgd_momentum_update_inplace(
        model.ds_block_10.dw_weights,
        grad_b10_dw_w,
        velocities[76],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.dw_bias,
        grad_b10_dw_b,
        velocities[77],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.dw_bn_gamma,
        grad_b10_dw_gamma,
        velocities[78],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.dw_bn_beta,
        grad_b10_dw_beta,
        velocities[79],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.pw_weights,
        grad_b10_pw_w,
        velocities[80],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.pw_bias,
        grad_b10_pw_b,
        velocities[81],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.pw_bn_gamma,
        grad_b10_pw_gamma,
        velocities[82],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_10.pw_bn_beta,
        grad_b10_pw_beta,
        velocities[83],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 11 (base=84):
    sgd_momentum_update_inplace(
        model.ds_block_11.dw_weights,
        grad_b11_dw_w,
        velocities[84],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.dw_bias,
        grad_b11_dw_b,
        velocities[85],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.dw_bn_gamma,
        grad_b11_dw_gamma,
        velocities[86],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.dw_bn_beta,
        grad_b11_dw_beta,
        velocities[87],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.pw_weights,
        grad_b11_pw_w,
        velocities[88],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.pw_bias,
        grad_b11_pw_b,
        velocities[89],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.pw_bn_gamma,
        grad_b11_pw_gamma,
        velocities[90],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_11.pw_bn_beta,
        grad_b11_pw_beta,
        velocities[91],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 12 (base=92):
    sgd_momentum_update_inplace(
        model.ds_block_12.dw_weights,
        grad_b12_dw_w,
        velocities[92],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.dw_bias,
        grad_b12_dw_b,
        velocities[93],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.dw_bn_gamma,
        grad_b12_dw_gamma,
        velocities[94],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.dw_bn_beta,
        grad_b12_dw_beta,
        velocities[95],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.pw_weights,
        grad_b12_pw_w,
        velocities[96],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.pw_bias,
        grad_b12_pw_b,
        velocities[97],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.pw_bn_gamma,
        grad_b12_pw_gamma,
        velocities[98],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_12.pw_bn_beta,
        grad_b12_pw_beta,
        velocities[99],
        Float64(learning_rate),
        Float64(momentum),
    )

    # Block 13 (base=100):
    sgd_momentum_update_inplace(
        model.ds_block_13.dw_weights,
        grad_b13_dw_w,
        velocities[100],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.dw_bias,
        grad_b13_dw_b,
        velocities[101],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.dw_bn_gamma,
        grad_b13_dw_gamma,
        velocities[102],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.dw_bn_beta,
        grad_b13_dw_beta,
        velocities[103],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.pw_weights,
        grad_b13_pw_w,
        velocities[104],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.pw_bias,
        grad_b13_pw_b,
        velocities[105],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.pw_bn_gamma,
        grad_b13_pw_gamma,
        velocities[106],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.ds_block_13.pw_bn_beta,
        grad_b13_pw_beta,
        velocities[107],
        Float64(learning_rate),
        Float64(momentum),
    )

    # FC (base = 108):
    sgd_momentum_update_inplace(
        model.fc_weights,
        grad_fc_w,
        velocities[108],
        Float64(learning_rate),
        Float64(momentum),
    )
    sgd_momentum_update_inplace(
        model.fc_bias,
        grad_fc_b,
        velocities[109],
        Float64(learning_rate),
        Float64(momentum),
    )

    return loss


def initialize_velocities(model: MobileNetV1) raises -> List[AnyTensor]:
    """Zero velocity buffers for the 110 trainable parameters, canonical order.

    Layout: initial(4) | 13 blocks × 8 each | fc(2) = 110.
    """
    var v: List[AnyTensor] = []
    # Initial (indices 0..3)
    v.append(zeros(model.initial_conv_weights.shape(), DType.float32))
    v.append(zeros(model.initial_conv_bias.shape(), DType.float32))
    v.append(zeros(model.initial_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.initial_bn_beta.shape(), DType.float32))
    # Block 1 (indices 4..11)
    v.append(zeros(model.ds_block_1.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_1.pw_bn_beta.shape(), DType.float32))
    # Block 2 (indices 12..19)
    v.append(zeros(model.ds_block_2.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_2.pw_bn_beta.shape(), DType.float32))
    # Block 3 (indices 20..27)
    v.append(zeros(model.ds_block_3.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_3.pw_bn_beta.shape(), DType.float32))
    # Block 4 (indices 28..35)
    v.append(zeros(model.ds_block_4.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_4.pw_bn_beta.shape(), DType.float32))
    # Block 5 (indices 36..43)
    v.append(zeros(model.ds_block_5.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_5.pw_bn_beta.shape(), DType.float32))
    # Block 6 (indices 44..51)
    v.append(zeros(model.ds_block_6.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_6.pw_bn_beta.shape(), DType.float32))
    # Block 7 (indices 52..59)
    v.append(zeros(model.ds_block_7.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_7.pw_bn_beta.shape(), DType.float32))
    # Block 8 (indices 60..67)
    v.append(zeros(model.ds_block_8.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_8.pw_bn_beta.shape(), DType.float32))
    # Block 9 (indices 68..75)
    v.append(zeros(model.ds_block_9.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_9.pw_bn_beta.shape(), DType.float32))
    # Block 10 (indices 76..83)
    v.append(zeros(model.ds_block_10.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_10.pw_bn_beta.shape(), DType.float32))
    # Block 11 (indices 84..91)
    v.append(zeros(model.ds_block_11.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_11.pw_bn_beta.shape(), DType.float32))
    # Block 12 (indices 92..99)
    v.append(zeros(model.ds_block_12.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_12.pw_bn_beta.shape(), DType.float32))
    # Block 13 (indices 100..107)
    v.append(zeros(model.ds_block_13.dw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.dw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.dw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.dw_bn_beta.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.pw_weights.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.pw_bias.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.pw_bn_gamma.shape(), DType.float32))
    v.append(zeros(model.ds_block_13.pw_bn_beta.shape(), DType.float32))
    # FC (indices 108..109)
    v.append(zeros(model.fc_weights.shape(), DType.float32))
    v.append(zeros(model.fc_bias.shape(), DType.float32))
    return v^


def train_epoch(
    mut model: MobileNetV1,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    epoch: Int,
    mut velocities: List[AnyTensor],
    max_batches: Int = 0,
) raises -> Float32:
    """Train for one epoch using compute_gradients per batch.

    Labels are one-hot-encoded per batch to match cross_entropy's (B, num_classes)
    contract (src/projectodyssey/core/loss.mojo:309 asserts logits.shape() == targets.shape()).

    `max_batches` caps batches this epoch (0 = unbounded); used by --smoke /
    --max-batches to bound a per-PR run (#5551).

    NOTE: BN running mean/var are discarded this pass — see #5525 follow-up.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    if max_batches > 0 and max_batches < num_batches:
        num_batches = max_batches
    var total_loss = Float32(0.0)

    print("Epoch " + String(epoch + 1))
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            train_images, train_labels, start_idx, batch_size
        )
        var batch_labels_onehot = one_hot_encode(batch_pair[1], 10)
        var batch_loss = compute_gradients(
            model,
            batch_pair[0],
            batch_labels_onehot,
            learning_rate,
            momentum,
            velocities,
        )
        total_loss += batch_loss
        if (batch_idx + 1) % 100 == 0 or num_batches <= 10:
            print(
                "  Batch "
                + String(batch_idx + 1)
                + "/"
                + String(num_batches)
                + " - Loss: "
                + String(total_loss / Float32(batch_idx + 1))
            )
    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: " + String(avg_loss))
    return avg_loss


def validate(
    mut model: MobileNetV1,
    val_images: AnyTensor,
    val_labels: AnyTensor,
    batch_size: Int,
) raises -> Float32:
    """Validate model on validation set."""
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

        var logits = model.forward(batch_images, training=False)

        for i in range(current_batch_size):
            var logits_data = logits.data_ptr[DType.float32]()
            var pred_class = 0
            var max_logit = logits_data[i * 10]
            for j in range(1, 10):
                if logits_data[i * 10 + j] > max_logit:
                    max_logit = logits_data[i * 10 + j]
                    pred_class = j

            var true_class = Int(batch_labels[i])
            if pred_class == true_class:
                total_correct += 1

    var accuracy = Float32(total_correct) / Float32(num_samples) * 100.0
    return accuracy


def main() raises:
    """Main training entry point."""
    print("=" * 60)
    print("MobileNetV1 Training on CIFAR-10")
    print("=" * 60)
    print()

    # Parse arguments using standardized TrainingArgs
    var args = parse_training_args_with_defaults(
        default_epochs=200,
        default_batch_size=128,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets/cifar10",
        default_weights_dir="weights",
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
    print()

    # Load data — real CIFAR-10, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset download entirely so the training
    # entrypoint can run per-PR in CI. It checks the MECHANISM (the loop runs
    # and emits finite, parseable, decreasing loss), not convergence.
    var train_images: AnyTensor
    var train_labels: AnyTensor
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
    else:
        print("Loading CIFAR-10 training set...")
        var dataset = CIFAR10Dataset(data_dir)
        var train_data = dataset.get_train_data()
        train_images = train_data[0]
        train_labels = train_data[1]
    print("  Training samples: " + String(train_images.shape()[0]))
    print()

    print("Initializing MobileNetV1 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = MobileNetV1(num_classes=dataset_info.num_classes())
    print("  Model architecture: MobileNetV1")
    print("  Parameters: ~4.2M")
    print("  Depthwise separable blocks: 13")
    print()

    print("Initializing momentum velocities...")
    var velocities = initialize_velocities(model)
    print("  Velocities initialized for 110 parameters")
    print()

    print("Starting training...")
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
            epoch,
            velocities,
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

        if (epoch + 1) % 10 == 0:
            var val_acc = validate(
                model, train_images, train_labels, batch_size
            )
            print("  Validation Accuracy: " + String(val_acc) + "%")

        print()

    print("Training complete!")
    print()

    # Save weights — skipped in smoke mode (#5551): mechanism check.
    if not smoke:
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
