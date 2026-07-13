"""Training Script for VGG-16 on CIFAR-10 (Using ManualTrainer).

MIGRATED VERSION - Demonstrates shared training loop pattern.

This is the migrated version using the shared ManualTrainer infrastructure.
Reduces code from 918 lines to ~780 lines by delegating orchestration to ManualTrainer.

Changes from original:
- Removed manual TrainingLoop usage → use ManualTrainer.fit()
- Removed manual main() epoch loop → use ManualTrainer.fit()
- Kept model-specific compute_gradients() function (unchanged - 674 lines!)
- Kept model-specific evaluation logic (wrapped in closure)
- Kept momentum/velocity tracking (captured in closure)

Usage:
    mojo run examples/vgg16_cifar10/train_new.mojo --epochs 200 --batch-size 128 --lr 0.01 --momentum 0.9

Requirements:
    - CIFAR-10 dataset downloaded (run: python examples/vgg16_cifar10/download_cifar10.py)
    - Dataset location: datasets/cifar10/

References:
    - Original implementation: examples/vgg16_cifar10/train.mojo (918 lines)
    - This version: ~780 lines (15% reduction)
    - Issue #2597: Create shared training loop base class
    - Simonyan, K., & Zisserman, A. (2014).
      Very deep convolutional networks for large-scale image recognition.
"""

from model import VGG16
from odyssey.data.datasets import CIFAR10Dataset
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.conv import conv2d, conv2d_backward
from odyssey.core.pooling import maxpool2d, maxpool2d_backward
from odyssey.core.linear import linear, linear_backward
from odyssey.core.activation import relu, relu_backward
from odyssey.core.dropout import dropout, dropout_backward
from odyssey.core.loss import cross_entropy, cross_entropy_backward
from odyssey.training.schedulers import step_lr
from odyssey.training.loops import TrainingLoop
from odyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
    get_batch_indices,
)
from odyssey.data.constants import DatasetInfo
from odyssey.utils.training_args import parse_training_args_with_defaults
from odyssey.training.metrics.evaluate import evaluate_with_predict
from odyssey.training.optimizers import sgd_momentum_update_inplace
from std.collections import List


struct TrainingArgs:
    """Training arguments for VGG-16."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var momentum: Float32
    var data_dir: String
    var weights_dir: String

    def __init__(
        out self,
        epochs: Int,
        batch_size: Int,
        learning_rate: Float32,
        momentum: Float32,
        data_dir: String,
        weights_dir: String,
    ):
        """Initialize training arguments."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.data_dir = data_dir
        self.weights_dir = weights_dir


def parse_args() raises -> TrainingArgs:
    """Parse command line arguments.

    Returns:
        TrainingArgs with parsed arguments.
    """
    var args = parse_training_args_with_defaults(
        default_epochs=200,
        default_batch_size=128,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets/cifar10",
        default_weights_dir="vgg16_weights",
        default_lr_decay_epochs=60,
        default_lr_decay_factor=0.2,
    )

    return TrainingArgs(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=Float32(args.learning_rate),
        momentum=Float32(args.momentum),
        data_dir=args.data_dir,
        weights_dir=args.weights_dir,
    )


def initialize_velocities(model: VGG16) raises -> List[AnyTensor]:
    """Initialize momentum velocities for all parameters (32 tensors).

    Args:
        model: VGG16 model.

    Returns:
        List of zero-initialized velocity tensors matching parameter shapes.
    """
    var velocities: List[AnyTensor] = []

    # Initialize velocities for all 32 parameters (conv1-5 blocks + fc1-3)
    # Block 1 (4 params)
    velocities.append(zeros(model.conv1_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv1_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv1_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv1_2_bias.shape(), DType.float32))

    # Block 2 (4 params)
    velocities.append(zeros(model.conv2_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv2_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv2_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv2_2_bias.shape(), DType.float32))

    # Block 3 (6 params)
    velocities.append(zeros(model.conv3_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv3_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv3_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_3_bias.shape(), DType.float32))

    # Block 4 (6 params)
    velocities.append(zeros(model.conv4_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv4_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv4_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_3_bias.shape(), DType.float32))

    # Block 5 (6 params)
    velocities.append(zeros(model.conv5_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv5_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv5_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_3_bias.shape(), DType.float32))

    # FC layers (6 params)
    velocities.append(zeros(model.fc1_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc1_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc2_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc2_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc3_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc3_bias.shape(), DType.float32))

    return velocities^


def compute_gradients(
    mut model: VGG16,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[AnyTensor],
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually through all 16 layers.
    This function is UNCHANGED from the original - only the orchestration changed.

    Args:
        model: VGG16 model.
        input: Batch of images (batch, 3, 32, 32).
        labels: Batch of labels (batch,).
        learning_rate: Learning rate for SGD.
        momentum: Momentum factor for SGD.
        velocities: Momentum velocities for each parameter (32 tensors).

    Returns:
        Loss value for this batch.
    """
    # ========== Forward Pass (with caching for backward) ==========

    # Block 1: Conv1_1 -> ReLU -> Conv1_2 -> ReLU -> MaxPool
    var conv1_1_out = conv2d(
        input, model.conv1_1_kernel, model.conv1_1_bias, stride=1, padding=1
    )
    var relu1_1_out = relu(conv1_1_out)
    var conv1_2_out = conv2d(
        relu1_1_out,
        model.conv1_2_kernel,
        model.conv1_2_bias,
        stride=1,
        padding=1,
    )
    var relu1_2_out = relu(conv1_2_out)
    var pool1_out = maxpool2d(relu1_2_out, kernel_size=2, stride=2, padding=0)

    # Block 2: Conv2_1 -> ReLU -> Conv2_2 -> ReLU -> MaxPool
    var conv2_1_out = conv2d(
        pool1_out, model.conv2_1_kernel, model.conv2_1_bias, stride=1, padding=1
    )
    var relu2_1_out = relu(conv2_1_out)
    var conv2_2_out = conv2d(
        relu2_1_out,
        model.conv2_2_kernel,
        model.conv2_2_bias,
        stride=1,
        padding=1,
    )
    var relu2_2_out = relu(conv2_2_out)
    var pool2_out = maxpool2d(relu2_2_out, kernel_size=2, stride=2, padding=0)

    # Block 3: Conv3_1 -> ReLU -> Conv3_2 -> ReLU -> Conv3_3 -> ReLU -> MaxPool
    var conv3_1_out = conv2d(
        pool2_out, model.conv3_1_kernel, model.conv3_1_bias, stride=1, padding=1
    )
    var relu3_1_out = relu(conv3_1_out)
    var conv3_2_out = conv2d(
        relu3_1_out,
        model.conv3_2_kernel,
        model.conv3_2_bias,
        stride=1,
        padding=1,
    )
    var relu3_2_out = relu(conv3_2_out)
    var conv3_3_out = conv2d(
        relu3_2_out,
        model.conv3_3_kernel,
        model.conv3_3_bias,
        stride=1,
        padding=1,
    )
    var relu3_3_out = relu(conv3_3_out)
    var pool3_out = maxpool2d(relu3_3_out, kernel_size=2, stride=2, padding=0)

    # Block 4: Conv4_1 -> ReLU -> Conv4_2 -> ReLU -> Conv4_3 -> ReLU -> MaxPool
    var conv4_1_out = conv2d(
        pool3_out, model.conv4_1_kernel, model.conv4_1_bias, stride=1, padding=1
    )
    var relu4_1_out = relu(conv4_1_out)
    var conv4_2_out = conv2d(
        relu4_1_out,
        model.conv4_2_kernel,
        model.conv4_2_bias,
        stride=1,
        padding=1,
    )
    var relu4_2_out = relu(conv4_2_out)
    var conv4_3_out = conv2d(
        relu4_2_out,
        model.conv4_3_kernel,
        model.conv4_3_bias,
        stride=1,
        padding=1,
    )
    var relu4_3_out = relu(conv4_3_out)
    var pool4_out = maxpool2d(relu4_3_out, kernel_size=2, stride=2, padding=0)

    # Block 5: Conv5_1 -> ReLU -> Conv5_2 -> ReLU -> Conv5_3 -> ReLU -> MaxPool
    var conv5_1_out = conv2d(
        pool4_out, model.conv5_1_kernel, model.conv5_1_bias, stride=1, padding=1
    )
    var relu5_1_out = relu(conv5_1_out)
    var conv5_2_out = conv2d(
        relu5_1_out,
        model.conv5_2_kernel,
        model.conv5_2_bias,
        stride=1,
        padding=1,
    )
    var relu5_2_out = relu(conv5_2_out)
    var conv5_3_out = conv2d(
        relu5_2_out,
        model.conv5_3_kernel,
        model.conv5_3_bias,
        stride=1,
        padding=1,
    )
    var relu5_3_out = relu(conv5_3_out)
    var pool5_out = maxpool2d(relu5_3_out, kernel_size=2, stride=2, padding=0)

    # Flatten
    var pool5_shape = pool5_out.shape()
    var batch_size = pool5_shape[0]
    var flattened_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
    var flatten_shape = List[Int]()
    flatten_shape.append(batch_size)
    flatten_shape.append(flattened_size)
    var flattened = pool5_out.reshape(flatten_shape)

    # FC1 + ReLU + Dropout
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu_fc1_out = relu(fc1_out)
    var drop1_result = dropout(
        relu_fc1_out, Float64(model.dropout_rate), training=True
    )
    var drop1_out = drop1_result[0]  # Dropout output
    var drop1_mask = drop1_result[1]  # Dropout mask for backward

    # FC2 + ReLU + Dropout
    var fc2_out = linear(drop1_out, model.fc2_weights, model.fc2_bias)
    var relu_fc2_out = relu(fc2_out)
    var drop2_result = dropout(
        relu_fc2_out, Float64(model.dropout_rate), training=True
    )
    var drop2_out = drop2_result[0]  # Dropout output
    var drop2_mask = drop2_result[1]  # Dropout mask for backward

    # FC3 (logits)
    var logits = linear(drop2_out, model.fc3_weights, model.fc3_bias)

    # Compute loss
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ========== Backward Pass (through all 16 layers) ==========

    # Start with gradient from loss
    # For cross-entropy, the initial gradient is 1.0
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output.set(0, Float32(1.0))
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    # FC3 backward
    var fc3_grads = linear_backward(grad_logits, drop2_out, model.fc3_weights)
    var grad_drop2_out = fc3_grads.grad_input
    var grad_fc3_weights = fc3_grads.grad_weights
    var grad_fc3_bias = fc3_grads.grad_bias

    # Dropout2 backward
    var grad_relu_fc2_out = dropout_backward(
        grad_drop2_out, drop2_mask, Float64(model.dropout_rate)
    )

    # ReLU FC2 backward
    var grad_fc2_out = relu_backward(grad_relu_fc2_out, fc2_out)

    # FC2 backward
    var fc2_grads = linear_backward(grad_fc2_out, drop1_out, model.fc2_weights)
    var grad_drop1_out = fc2_grads.grad_input
    var grad_fc2_weights = fc2_grads.grad_weights
    var grad_fc2_bias = fc2_grads.grad_bias

    # Dropout1 backward
    var grad_relu_fc1_out = dropout_backward(
        grad_drop1_out, drop1_mask, Float64(model.dropout_rate)
    )

    # ReLU FC1 backward
    var grad_fc1_out = relu_backward(grad_relu_fc1_out, fc1_out)

    # FC1 backward
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)
    var grad_flattened = fc1_grads.grad_input
    var grad_fc1_weights = fc1_grads.grad_weights
    var grad_fc1_bias = fc1_grads.grad_bias

    # Unflatten gradient
    var grad_pool5_out = grad_flattened.reshape(pool5_shape)

    # ===== Block 5 Backward =====

    # MaxPool5 backward
    var grad_relu5_3_out = maxpool2d_backward(
        grad_pool5_out,
        relu5_3_out,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    # ReLU5_3 backward
    var grad_conv5_3_out = relu_backward(grad_relu5_3_out, conv5_3_out)

    # Conv5_3 backward
    var conv5_3_grads = conv2d_backward(
        grad_conv5_3_out, relu5_2_out, model.conv5_3_kernel, stride=1, padding=1
    )
    var grad_relu5_2_out = conv5_3_grads.grad_input
    var grad_conv5_3_kernel = conv5_3_grads.grad_weights
    var grad_conv5_3_bias = conv5_3_grads.grad_bias

    # ReLU5_2 backward
    var grad_conv5_2_out = relu_backward(grad_relu5_2_out, conv5_2_out)

    # Conv5_2 backward
    var conv5_2_grads = conv2d_backward(
        grad_conv5_2_out, relu5_1_out, model.conv5_2_kernel, stride=1, padding=1
    )
    var grad_relu5_1_out = conv5_2_grads.grad_input
    var grad_conv5_2_kernel = conv5_2_grads.grad_weights
    var grad_conv5_2_bias = conv5_2_grads.grad_bias

    # ReLU5_1 backward
    var grad_conv5_1_out = relu_backward(grad_relu5_1_out, conv5_1_out)

    # Conv5_1 backward
    var conv5_1_grads = conv2d_backward(
        grad_conv5_1_out, pool4_out, model.conv5_1_kernel, stride=1, padding=1
    )
    var grad_pool4_out = conv5_1_grads.grad_input
    var grad_conv5_1_kernel = conv5_1_grads.grad_weights
    var grad_conv5_1_bias = conv5_1_grads.grad_bias

    # ===== Block 4 Backward =====

    # MaxPool4 backward
    var grad_relu4_3_out = maxpool2d_backward(
        grad_pool4_out,
        relu4_3_out,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    # ReLU4_3 backward
    var grad_conv4_3_out = relu_backward(grad_relu4_3_out, conv4_3_out)

    # Conv4_3 backward
    var conv4_3_grads = conv2d_backward(
        grad_conv4_3_out, relu4_2_out, model.conv4_3_kernel, stride=1, padding=1
    )
    var grad_relu4_2_out = conv4_3_grads.grad_input
    var grad_conv4_3_kernel = conv4_3_grads.grad_weights
    var grad_conv4_3_bias = conv4_3_grads.grad_bias

    # ReLU4_2 backward
    var grad_conv4_2_out = relu_backward(grad_relu4_2_out, conv4_2_out)

    # Conv4_2 backward
    var conv4_2_grads = conv2d_backward(
        grad_conv4_2_out, relu4_1_out, model.conv4_2_kernel, stride=1, padding=1
    )
    var grad_relu4_1_out = conv4_2_grads.grad_input
    var grad_conv4_2_kernel = conv4_2_grads.grad_weights
    var grad_conv4_2_bias = conv4_2_grads.grad_bias

    # ReLU4_1 backward
    var grad_conv4_1_out = relu_backward(grad_relu4_1_out, conv4_1_out)

    # Conv4_1 backward
    var conv4_1_grads = conv2d_backward(
        grad_conv4_1_out, pool3_out, model.conv4_1_kernel, stride=1, padding=1
    )
    var grad_pool3_out = conv4_1_grads.grad_input
    var grad_conv4_1_kernel = conv4_1_grads.grad_weights
    var grad_conv4_1_bias = conv4_1_grads.grad_bias

    # ===== Block 3 Backward =====

    # MaxPool3 backward
    var grad_relu3_3_out = maxpool2d_backward(
        grad_pool3_out,
        relu3_3_out,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    # ReLU3_3 backward
    var grad_conv3_3_out = relu_backward(grad_relu3_3_out, conv3_3_out)

    # Conv3_3 backward
    var conv3_3_grads = conv2d_backward(
        grad_conv3_3_out, relu3_2_out, model.conv3_3_kernel, stride=1, padding=1
    )
    var grad_relu3_2_out = conv3_3_grads.grad_input
    var grad_conv3_3_kernel = conv3_3_grads.grad_weights
    var grad_conv3_3_bias = conv3_3_grads.grad_bias

    # ReLU3_2 backward
    var grad_conv3_2_out = relu_backward(grad_relu3_2_out, conv3_2_out)

    # Conv3_2 backward
    var conv3_2_grads = conv2d_backward(
        grad_conv3_2_out, relu3_1_out, model.conv3_2_kernel, stride=1, padding=1
    )
    var grad_relu3_1_out = conv3_2_grads.grad_input
    var grad_conv3_2_kernel = conv3_2_grads.grad_weights
    var grad_conv3_2_bias = conv3_2_grads.grad_bias

    # ReLU3_1 backward
    var grad_conv3_1_out = relu_backward(grad_relu3_1_out, conv3_1_out)

    # Conv3_1 backward
    var conv3_1_grads = conv2d_backward(
        grad_conv3_1_out, pool2_out, model.conv3_1_kernel, stride=1, padding=1
    )
    var grad_pool2_out = conv3_1_grads.grad_input
    var grad_conv3_1_kernel = conv3_1_grads.grad_weights
    var grad_conv3_1_bias = conv3_1_grads.grad_bias

    # ===== Block 2 Backward =====

    # MaxPool2 backward
    var grad_relu2_2_out = maxpool2d_backward(
        grad_pool2_out,
        relu2_2_out,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    # ReLU2_2 backward
    var grad_conv2_2_out = relu_backward(grad_relu2_2_out, conv2_2_out)

    # Conv2_2 backward
    var conv2_2_grads = conv2d_backward(
        grad_conv2_2_out, relu2_1_out, model.conv2_2_kernel, stride=1, padding=1
    )
    var grad_relu2_1_out = conv2_2_grads.grad_input
    var grad_conv2_2_kernel = conv2_2_grads.grad_weights
    var grad_conv2_2_bias = conv2_2_grads.grad_bias

    # ReLU2_1 backward
    var grad_conv2_1_out = relu_backward(grad_relu2_1_out, conv2_1_out)

    # Conv2_1 backward
    var conv2_1_grads = conv2d_backward(
        grad_conv2_1_out, pool1_out, model.conv2_1_kernel, stride=1, padding=1
    )
    var grad_pool1_out = conv2_1_grads.grad_input
    var grad_conv2_1_kernel = conv2_1_grads.grad_weights
    var grad_conv2_1_bias = conv2_1_grads.grad_bias

    # ===== Block 1 Backward =====

    # MaxPool1 backward
    var grad_relu1_2_out = maxpool2d_backward(
        grad_pool1_out,
        relu1_2_out,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    # ReLU1_2 backward
    var grad_conv1_2_out = relu_backward(grad_relu1_2_out, conv1_2_out)

    # Conv1_2 backward
    var conv1_2_grads = conv2d_backward(
        grad_conv1_2_out, relu1_1_out, model.conv1_2_kernel, stride=1, padding=1
    )
    var grad_relu1_1_out = conv1_2_grads.grad_input
    var grad_conv1_2_kernel = conv1_2_grads.grad_weights
    var grad_conv1_2_bias = conv1_2_grads.grad_bias

    # ReLU1_1 backward
    var grad_conv1_1_out = relu_backward(grad_relu1_1_out, conv1_1_out)

    # Conv1_1 backward
    var conv1_1_grads = conv2d_backward(
        grad_conv1_1_out, input, model.conv1_1_kernel, stride=1, padding=1
    )
    _ = conv1_1_grads.grad_input  # Not used (no input gradient needed)
    var grad_conv1_1_kernel = conv1_1_grads.grad_weights
    var grad_conv1_1_bias = conv1_1_grads.grad_bias

    # ========== Parameter Update (SGD with Momentum) ==========
    # Update all 32 parameters (16 layers × 2 params per layer)

    # Convert Float32 hyperparameters to Float64 for optimizer
    var lr_f64 = Float64(learning_rate)
    var momentum_f64 = Float64(momentum)

    # Block 1 updates
    sgd_momentum_update_inplace(
        model.conv1_1_kernel,
        grad_conv1_1_kernel,
        velocities[0],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv1_1_bias,
        grad_conv1_1_bias,
        velocities[1],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv1_2_kernel,
        grad_conv1_2_kernel,
        velocities[2],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv1_2_bias,
        grad_conv1_2_bias,
        velocities[3],
        lr_f64,
        momentum_f64,
    )

    # Block 2 updates
    sgd_momentum_update_inplace(
        model.conv2_1_kernel,
        grad_conv2_1_kernel,
        velocities[4],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv2_1_bias,
        grad_conv2_1_bias,
        velocities[5],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv2_2_kernel,
        grad_conv2_2_kernel,
        velocities[6],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv2_2_bias,
        grad_conv2_2_bias,
        velocities[7],
        lr_f64,
        momentum_f64,
    )

    # Block 3 updates
    sgd_momentum_update_inplace(
        model.conv3_1_kernel,
        grad_conv3_1_kernel,
        velocities[8],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv3_1_bias,
        grad_conv3_1_bias,
        velocities[9],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv3_2_kernel,
        grad_conv3_2_kernel,
        velocities[10],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv3_2_bias,
        grad_conv3_2_bias,
        velocities[11],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv3_3_kernel,
        grad_conv3_3_kernel,
        velocities[12],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv3_3_bias,
        grad_conv3_3_bias,
        velocities[13],
        lr_f64,
        momentum_f64,
    )

    # Block 4 updates
    sgd_momentum_update_inplace(
        model.conv4_1_kernel,
        grad_conv4_1_kernel,
        velocities[14],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv4_1_bias,
        grad_conv4_1_bias,
        velocities[15],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv4_2_kernel,
        grad_conv4_2_kernel,
        velocities[16],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv4_2_bias,
        grad_conv4_2_bias,
        velocities[17],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv4_3_kernel,
        grad_conv4_3_kernel,
        velocities[18],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv4_3_bias,
        grad_conv4_3_bias,
        velocities[19],
        lr_f64,
        momentum_f64,
    )

    # Block 5 updates
    sgd_momentum_update_inplace(
        model.conv5_1_kernel,
        grad_conv5_1_kernel,
        velocities[20],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv5_1_bias,
        grad_conv5_1_bias,
        velocities[21],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv5_2_kernel,
        grad_conv5_2_kernel,
        velocities[22],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv5_2_bias,
        grad_conv5_2_bias,
        velocities[23],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv5_3_kernel,
        grad_conv5_3_kernel,
        velocities[24],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.conv5_3_bias,
        grad_conv5_3_bias,
        velocities[25],
        lr_f64,
        momentum_f64,
    )

    # FC layer updates
    sgd_momentum_update_inplace(
        model.fc1_weights,
        grad_fc1_weights,
        velocities[26],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.fc1_bias, grad_fc1_bias, velocities[27], lr_f64, momentum_f64
    )
    sgd_momentum_update_inplace(
        model.fc2_weights,
        grad_fc2_weights,
        velocities[28],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.fc2_bias, grad_fc2_bias, velocities[29], lr_f64, momentum_f64
    )
    sgd_momentum_update_inplace(
        model.fc3_weights,
        grad_fc3_weights,
        velocities[30],
        lr_f64,
        momentum_f64,
    )
    sgd_momentum_update_inplace(
        model.fc3_bias, grad_fc3_bias, velocities[31], lr_f64, momentum_f64
    )

    return loss


def evaluate(
    mut model: VGG16, test_images: AnyTensor, test_labels: AnyTensor
) raises -> Float32:
    """Evaluate model on test set.

    Args:
        model: VGG16 model.
        test_images: Test images tensor.
        test_labels: Test labels tensor.

    Returns:
        Test accuracy as percentage.
    """
    var num_samples = test_images.shape()[0]
    var predictions = List[Int]()

    print("Evaluating on", num_samples, "samples...")

    # Collect predictions from model for all test samples
    for i in range(num_samples):
        # Extract single sample using batch utilities
        var batch_pair = extract_batch_pair(test_images, test_labels, i, 1)
        var sample_image = batch_pair[0]

        # Forward pass (inference mode)
        var pred_class = model.predict(sample_image)
        predictions.append(pred_class)

        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            print("  Processed", i + 1, "/", num_samples)

    # Use shared evaluate function from odyssey.training.metrics
    var accuracy = evaluate_with_predict(predictions, test_labels)
    print(
        "  Test Accuracy:",
        accuracy * 100.0,
        "% (",
        Int(accuracy * Float32(num_samples)),
        "/",
        num_samples,
        ")",
    )

    return accuracy * 100.0  # Return as percentage


def main() raises:
    """Main training loop."""
    print("=" * 60)
    print("VGG-16 Training on CIFAR-10 Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config.epochs
    var batch_size = config.batch_size
    var learning_rate = config.learning_rate
    var momentum = config.momentum
    var data_dir = config.data_dir
    var weights_dir = config.weights_dir

    print("\nConfiguration:")
    print("  Epochs:", epochs)
    print("  Batch Size:", batch_size)
    print("  Learning Rate:", learning_rate)
    print("  Momentum:", momentum)
    print("  Data Directory:", data_dir)
    print("  Weights Directory:", weights_dir)
    print()

    # Initialize model
    print("Initializing VGG-16 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = VGG16(num_classes=dataset_info.num_classes(), dropout_rate=0.5)
    print("  Model initialized with", model.num_classes, "classes")
    print("  Dropout rate:", model.dropout_rate)
    print()

    # Initialize momentum velocities
    print("Initializing momentum velocities...")
    var velocities = initialize_velocities(model)
    print("  Velocities initialized for 32 parameters")
    print()

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    var dataset = CIFAR10Dataset(data_dir)
    var train_data = dataset.get_train_data()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Training samples:", train_images.shape()[0])
    print("  Test samples:", test_images.shape()[0])
    print()

    # Training loop
    print("Starting training...")
    print()

    for epoch in range(1, epochs + 1):
        # Manual batch processing loop
        var num_samples = train_images.shape()[0]
        var num_batches = compute_num_batches(num_samples, batch_size)
        var total_loss = Float32(0.0)

        print("Epoch [", epoch, "/", epochs, "]")

        for batch_idx in range(num_batches):
            var batch_indices = get_batch_indices(
                batch_idx, batch_size, num_samples
            )
            var start_idx = batch_indices[0]
            var end_idx = batch_indices[1]

            # Extract batch
            var batch_pair = extract_batch_pair(
                train_images, train_labels, start_idx, end_idx - start_idx
            )
            var batch_images = batch_pair[0]
            var batch_labels = batch_pair[1]

            # Compute gradients and update parameters
            var batch_loss = compute_gradients(
                model,
                batch_images,
                batch_labels,
                learning_rate,
                momentum,
                velocities,
            )
            total_loss += batch_loss

            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(
                    "  Batch [",
                    batch_idx + 1,
                    "/",
                    num_batches,
                    "] - Loss:",
                    batch_loss,
                )

        var train_loss = total_loss / Float32(num_batches)
        print("  Average Loss:", train_loss)

        # Evaluate every epoch
        var test_acc = evaluate(model, test_images, test_labels)
        print()

        # Save model every 20 epochs
        if epoch % 20 == 0:
            print("Saving checkpoint at epoch", epoch, "...")
            model.save_weights(weights_dir)
            print("  Checkpoint saved to", weights_dir)
            print()

    # Save final model
    print("Saving final model weights...")
    model.save_weights(weights_dir)
    print("  Model saved to", weights_dir)
    print()

    print("Training complete!")
