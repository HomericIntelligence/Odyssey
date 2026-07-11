"""Training Script for ResNet-18 on CIFAR-10.

This script demonstrates manual backpropagation through a deep residual network
with skip connections and batch normalization.

Key Implementation:
    - Full forward pass with activation caching
    - Manual backward pass through all 18 layers
    - Batch normalization backward (batch_norm2d_backward)
    - Skip connection gradient splitting (add_backward)
    - SGD with momentum optimization

Training Strategy:
    - SGD with momentum (0.9)
    - Learning rate decay (step: 0.2x every 60 epochs)
    - Mini-batch training (batch_size=128)
    - Cross-entropy loss

Shared Modules Used:
    - projectodyssey.core: Tensor operations (conv2d, relu, batch_norm2d, etc.)
    - projectodyssey.core.loss: cross_entropy loss functions
    - projectodyssey.data: Data loading and batch extraction
    - projectodyssey.data.datasets: CIFAR-10 dataset loading
    - projectodyssey.training.optimizers: SGD with momentum
    - projectodyssey.training.metrics: Evaluation utilities
    - projectodyssey.utils.arg_parser: Command-line argument parsing

Usage:
    mojo run examples/resnet18_cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import avgpool2d, avgpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
)
from projectodyssey.core.arithmetic import add, add_backward
from projectodyssey.core.gradient_types import (
    IdentityBlockGradients,
    ProjectionBlockGradients,
)
from projectodyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
)
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.data.datasets import CIFAR10Dataset
from projectodyssey.data import one_hot_encode
from projectodyssey.training.optimizers import sgd_momentum_update_inplace
from projectodyssey.training.metrics import evaluate_logits_batch
from projectodyssey.utils.training_args import parse_training_args_with_defaults
from model import (
    ResNet18,
    ResNet18Velocities,
    initialize_velocities,
    ResNet18ForwardCache,
    IdentityCache,
    ProjectionCache,
)


def evaluate_test_set(
    mut model: ResNet18,
    test_images: AnyTensor,
    test_labels_raw: AnyTensor,
    batch_size: Int = 100,
) raises -> Tuple[Float32, Float32]:
    """Evaluate model on test set and return loss and accuracy.

    Args:
        model: ResNet-18 model.
        test_images: Test images (N, 3, 32, 32).
        test_labels_raw: Test labels (N,) with class indices.
        batch_size: Batch size for evaluation.

    Returns:
        Tuple of (test_loss, top1_accuracy).
    """
    var num_samples = test_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    var total_loss = Float32(0.0)
    var total_correct = 0
    var total_samples_processed = 0

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            test_images, test_labels_raw, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels_raw = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # One-hot encode labels
        var batch_labels = one_hot_encode(batch_labels_raw, 10)

        # Forward pass (inference mode)
        var logits = model.forward(batch_images, training=False)

        # Compute loss (weighted by batch size)
        var batch_loss = cross_entropy(logits, batch_labels)
        total_loss = total_loss + Float32(
            batch_loss.load[DType.float32](0)
        ) * Float32(current_batch_size)

        # Compute batch accuracy
        var batch_acc_fraction = evaluate_logits_batch(logits, batch_labels_raw)
        var batch_correct = Int(
            batch_acc_fraction * Float32(current_batch_size)
        )
        total_correct += batch_correct
        total_samples_processed += current_batch_size

    var avg_loss = total_loss / Float32(total_samples_processed)
    var accuracy = Float32(total_correct) / Float32(num_samples) * 100.0

    return (avg_loss, accuracy)


def backward_identity_block(
    dY: AnyTensor,
    cache: IdentityCache,
    conv1_kernel: AnyTensor,
    conv1_bias: AnyTensor,
    bn1_gamma: AnyTensor,
    bn1_running_mean: AnyTensor,
    bn1_running_var: AnyTensor,
    conv2_kernel: AnyTensor,
    conv2_bias: AnyTensor,
    bn2_gamma: AnyTensor,
    bn2_running_mean: AnyTensor,
    bn2_running_var: AnyTensor,
) raises -> IdentityBlockGradients:
    """Backward pass through identity (non-projection) residual block.

    Reverse order: y_out = relu(bn2_out + block_input)
        → relu_backward
        → add_backward (split main/skip paths)
        → BN2 backward
        → conv2 backward
        → ReLU1 backward
        → BN1 backward
        → conv1 backward
        → add main + skip gradients at block_input
    """
    # Reverse: y_out = relu(bn2_out + block_input)
    var dPreRelu = relu_backward(dY, cache.skip_sum)
    # add_backward for shape-matching inputs returns (grad_output, grad_output) unchanged.
    # Both are the two path-gradients reconverging on block_input (standard residual backward).
    var pair = add_backward(dPreRelu, cache.conv2_pre_bn, cache.block_input)
    var dMain = pair.grad_a
    var dSkip = pair.grad_b

    # BN2 backward — training-mode impl (normalization.mojo:371) re-derives batch stats from x,
    # so passing model's running_mean/var is inert. Included for signature compliance.
    var bn2_input, bn2_gamma_grad, bn2_beta_grad = batch_norm2d_backward(
        dMain,
        cache.conv2_pre_bn,
        bn2_gamma,
        bn2_running_mean,
        bn2_running_var,
        training=True,
    )
    var c2 = conv2d_backward(
        bn2_input, cache.relu1_out, conv2_kernel, stride=1, padding=1
    )
    var dReLU1 = relu_backward(c2.grad_input, cache.bn1_pre_relu)
    var bn1_input, bn1_gamma_grad, bn1_beta_grad = batch_norm2d_backward(
        dReLU1,
        cache.conv1_pre_bn,
        bn1_gamma,
        bn1_running_mean,
        bn1_running_var,
        training=True,
    )
    var c1 = conv2d_backward(
        bn1_input, cache.block_input, conv1_kernel, stride=1, padding=1
    )
    # Residual sum at block_input: main-path gradient + identity-skip gradient.
    var grad_input = add(c1.grad_input, dSkip)

    return IdentityBlockGradients(
        grad_input^,
        c1.grad_weights,
        c1.grad_bias,
        bn1_gamma_grad,
        bn1_beta_grad,
        c2.grad_weights,
        c2.grad_bias,
        bn2_gamma_grad,
        bn2_beta_grad,
    )


def backward_projection_block(
    dY: AnyTensor,
    cache: ProjectionCache,
    conv1_kernel: AnyTensor,
    conv1_bias: AnyTensor,
    bn1_gamma: AnyTensor,
    bn1_running_mean: AnyTensor,
    bn1_running_var: AnyTensor,
    conv2_kernel: AnyTensor,
    conv2_bias: AnyTensor,
    bn2_gamma: AnyTensor,
    bn2_running_mean: AnyTensor,
    bn2_running_var: AnyTensor,
    proj_kernel: AnyTensor,
    proj_bias: AnyTensor,
    proj_bn_gamma: AnyTensor,
    proj_bn_running_mean: AnyTensor,
    proj_bn_running_var: AnyTensor,
    main_stride: Int,
) raises -> ProjectionBlockGradients:
    """Backward pass through projection (stride-2, channel-change) residual block.

    Handles both main path (stride=main_stride) and projection skip path (1x1 conv
    with stride=main_stride). Main block uses stride=main_stride for first conv.
    """
    var dPreRelu = relu_backward(dY, cache.skip_sum)
    # Split into main and projection path gradients
    var pair = add_backward(dPreRelu, cache.conv2_pre_bn, cache.proj_bn_out)
    var dMain = pair.grad_a
    var dProj = pair.grad_b

    # Main path: BN2 → conv2 → ReLU1 → BN1 → conv1
    var bn2_input, bn2_gamma_grad, bn2_beta_grad = batch_norm2d_backward(
        dMain,
        cache.conv2_pre_bn,
        bn2_gamma,
        bn2_running_mean,
        bn2_running_var,
        training=True,
    )
    var c2 = conv2d_backward(
        bn2_input, cache.relu1_out, conv2_kernel, stride=1, padding=1
    )
    var dReLU1 = relu_backward(c2.grad_input, cache.bn1_pre_relu)
    var bn1_input, bn1_gamma_grad, bn1_beta_grad = batch_norm2d_backward(
        dReLU1,
        cache.conv1_pre_bn,
        bn1_gamma,
        bn1_running_mean,
        bn1_running_var,
        training=True,
    )
    var c1 = conv2d_backward(
        bn1_input,
        cache.block_input,
        conv1_kernel,
        stride=main_stride,
        padding=1,
    )

    # Projection path: proj_bn → proj_conv
    var pbn_input, pbn_gamma_grad, pbn_beta_grad = batch_norm2d_backward(
        dProj,
        cache.proj_conv_pre_bn,
        proj_bn_gamma,
        proj_bn_running_mean,
        proj_bn_running_var,
        training=True,
    )
    var pc = conv2d_backward(
        pbn_input, cache.block_input, proj_kernel, stride=main_stride, padding=0
    )
    var grad_input = add(c1.grad_input, pc.grad_input)

    return ProjectionBlockGradients(
        grad_input^,
        c1.grad_weights,
        c1.grad_bias,
        bn1_gamma_grad,
        bn1_beta_grad,
        c2.grad_weights,
        c2.grad_bias,
        bn2_gamma_grad,
        bn2_beta_grad,
        pc.grad_weights,
        pc.grad_bias,
        pbn_gamma_grad,
        pbn_beta_grad,
    )


def train_step(
    mut model: ResNet18,
    images: AnyTensor,
    labels: AnyTensor,
    mut vel: ResNet18Velocities,
    lr: Float64 = 0.01,
    momentum: Float64 = 0.9,
) raises -> Float32:
    """Full forward, backward, and SGD-momentum update for one batch.

    Args:
        model: ResNet-18 model.
        images: Batch images (batch_size, 3, 32, 32).
        labels: Batch labels (batch_size,).
        vel: SGD momentum velocities (82 fields, one per trainable param).
        lr: Learning rate.
        momentum: Momentum factor.

    Returns:
        Loss value for this batch.
    """
    # Forward with caching (does NOT mutate model.bn*_running_*)
    var fwd = model.forward_with_cache(images, training=True)
    var loss = cross_entropy(fwd.logits, labels)

    # Backward: loss → logits
    var grad_out = zeros([1], fwd.logits.dtype())
    grad_out.set(0, Float32(1.0))
    var dLogits = cross_entropy_backward(grad_out, fwd.logits, labels)

    # Backward: FC layer
    var fc = linear_backward(dLogits, fwd.flattened, model.fc_weights)

    # Backward: GAP (reshape (batch,512) → (batch,512,1,1), then avgpool_backward)
    var batch = images.shape()[0]
    var dGap4D = fc.grad_input.reshape([batch, 512, 1, 1])
    var d_s4b2_out = avgpool2d_backward(
        dGap4D, fwd.s4b2_cache.block_out, kernel_size=4, stride=1, padding=0
    )

    # Stage 4 (reverse: b2 identity, then b1 projection)
    var g_s4b2 = backward_identity_block(
        d_s4b2_out,
        fwd.s4b2_cache,
        model.s4b2_conv1_kernel,
        model.s4b2_conv1_bias,
        model.s4b2_bn1_gamma,
        model.s4b2_bn1_running_mean,
        model.s4b2_bn1_running_var,
        model.s4b2_conv2_kernel,
        model.s4b2_conv2_bias,
        model.s4b2_bn2_gamma,
        model.s4b2_bn2_running_mean,
        model.s4b2_bn2_running_var,
    )
    var g_s4b1 = backward_projection_block(
        g_s4b2.grad_input,
        fwd.s4b1_cache,
        model.s4b1_conv1_kernel,
        model.s4b1_conv1_bias,
        model.s4b1_bn1_gamma,
        model.s4b1_bn1_running_mean,
        model.s4b1_bn1_running_var,
        model.s4b1_conv2_kernel,
        model.s4b1_conv2_bias,
        model.s4b1_bn2_gamma,
        model.s4b1_bn2_running_mean,
        model.s4b1_bn2_running_var,
        model.s4b1_proj_kernel,
        model.s4b1_proj_bias,
        model.s4b1_proj_bn_gamma,
        model.s4b1_proj_bn_running_mean,
        model.s4b1_proj_bn_running_var,
        main_stride=2,
    )

    # Stage 3 (b2 identity, b1 projection stride=2)
    var g_s3b2 = backward_identity_block(
        g_s4b1.grad_input,
        fwd.s3b2_cache,
        model.s3b2_conv1_kernel,
        model.s3b2_conv1_bias,
        model.s3b2_bn1_gamma,
        model.s3b2_bn1_running_mean,
        model.s3b2_bn1_running_var,
        model.s3b2_conv2_kernel,
        model.s3b2_conv2_bias,
        model.s3b2_bn2_gamma,
        model.s3b2_bn2_running_mean,
        model.s3b2_bn2_running_var,
    )
    var g_s3b1 = backward_projection_block(
        g_s3b2.grad_input,
        fwd.s3b1_cache,
        model.s3b1_conv1_kernel,
        model.s3b1_conv1_bias,
        model.s3b1_bn1_gamma,
        model.s3b1_bn1_running_mean,
        model.s3b1_bn1_running_var,
        model.s3b1_conv2_kernel,
        model.s3b1_conv2_bias,
        model.s3b1_bn2_gamma,
        model.s3b1_bn2_running_mean,
        model.s3b1_bn2_running_var,
        model.s3b1_proj_kernel,
        model.s3b1_proj_bias,
        model.s3b1_proj_bn_gamma,
        model.s3b1_proj_bn_running_mean,
        model.s3b1_proj_bn_running_var,
        main_stride=2,
    )

    # Stage 2 (b2 identity, b1 projection stride=2)
    var g_s2b2 = backward_identity_block(
        g_s3b1.grad_input,
        fwd.s2b2_cache,
        model.s2b2_conv1_kernel,
        model.s2b2_conv1_bias,
        model.s2b2_bn1_gamma,
        model.s2b2_bn1_running_mean,
        model.s2b2_bn1_running_var,
        model.s2b2_conv2_kernel,
        model.s2b2_conv2_bias,
        model.s2b2_bn2_gamma,
        model.s2b2_bn2_running_mean,
        model.s2b2_bn2_running_var,
    )
    var g_s2b1 = backward_projection_block(
        g_s2b2.grad_input,
        fwd.s2b1_cache,
        model.s2b1_conv1_kernel,
        model.s2b1_conv1_bias,
        model.s2b1_bn1_gamma,
        model.s2b1_bn1_running_mean,
        model.s2b1_bn1_running_var,
        model.s2b1_conv2_kernel,
        model.s2b1_conv2_bias,
        model.s2b1_bn2_gamma,
        model.s2b1_bn2_running_mean,
        model.s2b1_bn2_running_var,
        model.s2b1_proj_kernel,
        model.s2b1_proj_bias,
        model.s2b1_proj_bn_gamma,
        model.s2b1_proj_bn_running_mean,
        model.s2b1_proj_bn_running_var,
        main_stride=2,
    )

    # Stage 1 (both identity, no projection, stride=1)
    var g_s1b2 = backward_identity_block(
        g_s2b1.grad_input,
        fwd.s1b2_cache,
        model.s1b2_conv1_kernel,
        model.s1b2_conv1_bias,
        model.s1b2_bn1_gamma,
        model.s1b2_bn1_running_mean,
        model.s1b2_bn1_running_var,
        model.s1b2_conv2_kernel,
        model.s1b2_conv2_bias,
        model.s1b2_bn2_gamma,
        model.s1b2_bn2_running_mean,
        model.s1b2_bn2_running_var,
    )
    var g_s1b1 = backward_identity_block(
        g_s1b2.grad_input,
        fwd.s1b1_cache,
        model.s1b1_conv1_kernel,
        model.s1b1_conv1_bias,
        model.s1b1_bn1_gamma,
        model.s1b1_bn1_running_mean,
        model.s1b1_bn1_running_var,
        model.s1b1_conv2_kernel,
        model.s1b1_conv2_bias,
        model.s1b1_bn2_gamma,
        model.s1b1_bn2_running_mean,
        model.s1b1_bn2_running_var,
    )

    # Initial conv + BN + ReLU
    var dInitBn = relu_backward(g_s1b1.grad_input, fwd.bn1_pre_relu)
    var init_bn = batch_norm2d_backward(
        dInitBn,
        fwd.conv1_pre_bn,
        model.bn1_gamma,
        model.bn1_running_mean,
        model.bn1_running_var,
        training=True,
    )
    var init_c = conv2d_backward(
        init_bn[0], images, model.conv1_kernel, stride=1, padding=1
    )

    # SGD momentum updates (82 parameters total)
    sgd_momentum_update_inplace(
        model.conv1_kernel, init_c.grad_weights, vel.conv1_kernel, lr, momentum
    )
    sgd_momentum_update_inplace(
        model.conv1_bias, init_c.grad_bias, vel.conv1_bias, lr, momentum
    )
    sgd_momentum_update_inplace(
        model.bn1_gamma, init_bn[1], vel.bn1_gamma, lr, momentum
    )
    sgd_momentum_update_inplace(
        model.bn1_beta, init_bn[2], vel.bn1_beta, lr, momentum
    )

    # Stage 1 identity blocks (8 params × 2 = 16)
    sgd_momentum_update_inplace(
        model.s1b1_conv1_kernel,
        g_s1b1.grad_conv1_kernel,
        vel.s1b1_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_conv1_bias,
        g_s1b1.grad_conv1_bias,
        vel.s1b1_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_bn1_gamma,
        g_s1b1.grad_bn1_gamma,
        vel.s1b1_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_bn1_beta,
        g_s1b1.grad_bn1_beta,
        vel.s1b1_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_conv2_kernel,
        g_s1b1.grad_conv2_kernel,
        vel.s1b1_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_conv2_bias,
        g_s1b1.grad_conv2_bias,
        vel.s1b1_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_bn2_gamma,
        g_s1b1.grad_bn2_gamma,
        vel.s1b1_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b1_bn2_beta,
        g_s1b1.grad_bn2_beta,
        vel.s1b1_bn2_beta,
        lr,
        momentum,
    )

    sgd_momentum_update_inplace(
        model.s1b2_conv1_kernel,
        g_s1b2.grad_conv1_kernel,
        vel.s1b2_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_conv1_bias,
        g_s1b2.grad_conv1_bias,
        vel.s1b2_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_bn1_gamma,
        g_s1b2.grad_bn1_gamma,
        vel.s1b2_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_bn1_beta,
        g_s1b2.grad_bn1_beta,
        vel.s1b2_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_conv2_kernel,
        g_s1b2.grad_conv2_kernel,
        vel.s1b2_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_conv2_bias,
        g_s1b2.grad_conv2_bias,
        vel.s1b2_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_bn2_gamma,
        g_s1b2.grad_bn2_gamma,
        vel.s1b2_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s1b2_bn2_beta,
        g_s1b2.grad_bn2_beta,
        vel.s1b2_bn2_beta,
        lr,
        momentum,
    )

    # Stage 2 projection block (12 params)
    sgd_momentum_update_inplace(
        model.s2b1_conv1_kernel,
        g_s2b1.grad_conv1_kernel,
        vel.s2b1_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_conv1_bias,
        g_s2b1.grad_conv1_bias,
        vel.s2b1_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_bn1_gamma,
        g_s2b1.grad_bn1_gamma,
        vel.s2b1_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_bn1_beta,
        g_s2b1.grad_bn1_beta,
        vel.s2b1_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_conv2_kernel,
        g_s2b1.grad_conv2_kernel,
        vel.s2b1_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_conv2_bias,
        g_s2b1.grad_conv2_bias,
        vel.s2b1_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_bn2_gamma,
        g_s2b1.grad_bn2_gamma,
        vel.s2b1_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_bn2_beta,
        g_s2b1.grad_bn2_beta,
        vel.s2b1_bn2_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_proj_kernel,
        g_s2b1.grad_proj_kernel,
        vel.s2b1_proj_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_proj_bias,
        g_s2b1.grad_proj_bias,
        vel.s2b1_proj_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_proj_bn_gamma,
        g_s2b1.grad_proj_bn_gamma,
        vel.s2b1_proj_bn_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b1_proj_bn_beta,
        g_s2b1.grad_proj_bn_beta,
        vel.s2b1_proj_bn_beta,
        lr,
        momentum,
    )

    # Stage 2 identity block (8 params)
    sgd_momentum_update_inplace(
        model.s2b2_conv1_kernel,
        g_s2b2.grad_conv1_kernel,
        vel.s2b2_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_conv1_bias,
        g_s2b2.grad_conv1_bias,
        vel.s2b2_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_bn1_gamma,
        g_s2b2.grad_bn1_gamma,
        vel.s2b2_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_bn1_beta,
        g_s2b2.grad_bn1_beta,
        vel.s2b2_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_conv2_kernel,
        g_s2b2.grad_conv2_kernel,
        vel.s2b2_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_conv2_bias,
        g_s2b2.grad_conv2_bias,
        vel.s2b2_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_bn2_gamma,
        g_s2b2.grad_bn2_gamma,
        vel.s2b2_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s2b2_bn2_beta,
        g_s2b2.grad_bn2_beta,
        vel.s2b2_bn2_beta,
        lr,
        momentum,
    )

    # Stage 3 projection block (12 params)
    sgd_momentum_update_inplace(
        model.s3b1_conv1_kernel,
        g_s3b1.grad_conv1_kernel,
        vel.s3b1_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_conv1_bias,
        g_s3b1.grad_conv1_bias,
        vel.s3b1_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_bn1_gamma,
        g_s3b1.grad_bn1_gamma,
        vel.s3b1_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_bn1_beta,
        g_s3b1.grad_bn1_beta,
        vel.s3b1_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_conv2_kernel,
        g_s3b1.grad_conv2_kernel,
        vel.s3b1_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_conv2_bias,
        g_s3b1.grad_conv2_bias,
        vel.s3b1_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_bn2_gamma,
        g_s3b1.grad_bn2_gamma,
        vel.s3b1_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_bn2_beta,
        g_s3b1.grad_bn2_beta,
        vel.s3b1_bn2_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_proj_kernel,
        g_s3b1.grad_proj_kernel,
        vel.s3b1_proj_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_proj_bias,
        g_s3b1.grad_proj_bias,
        vel.s3b1_proj_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_proj_bn_gamma,
        g_s3b1.grad_proj_bn_gamma,
        vel.s3b1_proj_bn_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b1_proj_bn_beta,
        g_s3b1.grad_proj_bn_beta,
        vel.s3b1_proj_bn_beta,
        lr,
        momentum,
    )

    # Stage 3 identity block (8 params)
    sgd_momentum_update_inplace(
        model.s3b2_conv1_kernel,
        g_s3b2.grad_conv1_kernel,
        vel.s3b2_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_conv1_bias,
        g_s3b2.grad_conv1_bias,
        vel.s3b2_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_bn1_gamma,
        g_s3b2.grad_bn1_gamma,
        vel.s3b2_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_bn1_beta,
        g_s3b2.grad_bn1_beta,
        vel.s3b2_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_conv2_kernel,
        g_s3b2.grad_conv2_kernel,
        vel.s3b2_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_conv2_bias,
        g_s3b2.grad_conv2_bias,
        vel.s3b2_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_bn2_gamma,
        g_s3b2.grad_bn2_gamma,
        vel.s3b2_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s3b2_bn2_beta,
        g_s3b2.grad_bn2_beta,
        vel.s3b2_bn2_beta,
        lr,
        momentum,
    )

    # Stage 4 projection block (12 params)
    sgd_momentum_update_inplace(
        model.s4b1_conv1_kernel,
        g_s4b1.grad_conv1_kernel,
        vel.s4b1_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_conv1_bias,
        g_s4b1.grad_conv1_bias,
        vel.s4b1_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_bn1_gamma,
        g_s4b1.grad_bn1_gamma,
        vel.s4b1_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_bn1_beta,
        g_s4b1.grad_bn1_beta,
        vel.s4b1_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_conv2_kernel,
        g_s4b1.grad_conv2_kernel,
        vel.s4b1_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_conv2_bias,
        g_s4b1.grad_conv2_bias,
        vel.s4b1_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_bn2_gamma,
        g_s4b1.grad_bn2_gamma,
        vel.s4b1_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_bn2_beta,
        g_s4b1.grad_bn2_beta,
        vel.s4b1_bn2_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_proj_kernel,
        g_s4b1.grad_proj_kernel,
        vel.s4b1_proj_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_proj_bias,
        g_s4b1.grad_proj_bias,
        vel.s4b1_proj_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_proj_bn_gamma,
        g_s4b1.grad_proj_bn_gamma,
        vel.s4b1_proj_bn_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b1_proj_bn_beta,
        g_s4b1.grad_proj_bn_beta,
        vel.s4b1_proj_bn_beta,
        lr,
        momentum,
    )

    # Stage 4 identity block (8 params)
    sgd_momentum_update_inplace(
        model.s4b2_conv1_kernel,
        g_s4b2.grad_conv1_kernel,
        vel.s4b2_conv1_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_conv1_bias,
        g_s4b2.grad_conv1_bias,
        vel.s4b2_conv1_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_bn1_gamma,
        g_s4b2.grad_bn1_gamma,
        vel.s4b2_bn1_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_bn1_beta,
        g_s4b2.grad_bn1_beta,
        vel.s4b2_bn1_beta,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_conv2_kernel,
        g_s4b2.grad_conv2_kernel,
        vel.s4b2_conv2_kernel,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_conv2_bias,
        g_s4b2.grad_conv2_bias,
        vel.s4b2_conv2_bias,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_bn2_gamma,
        g_s4b2.grad_bn2_gamma,
        vel.s4b2_bn2_gamma,
        lr,
        momentum,
    )
    sgd_momentum_update_inplace(
        model.s4b2_bn2_beta,
        g_s4b2.grad_bn2_beta,
        vel.s4b2_bn2_beta,
        lr,
        momentum,
    )

    # FC (2 params)
    sgd_momentum_update_inplace(
        model.fc_weights, fc.grad_weights, vel.fc_weights, lr, momentum
    )
    sgd_momentum_update_inplace(
        model.fc_bias, fc.grad_bias, vel.fc_bias, lr, momentum
    )

    return Float32(loss.load[DType.float32](0))


def train_epoch(
    mut model: ResNet18,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: ResNet18Velocities,
    mut loss_history: List[Float32],
    epoch: Int,
    total_epochs: Int,
    max_batches: Int = 0,
) raises -> Float32:
    """Train for one epoch with full backprop and SGD momentum.

    Args:
        model: ResNet-18 model.
        train_images: Training images (N, 3, 32, 32).
        train_labels: Training labels (N,).
        batch_size: Mini-batch size.
        learning_rate: Learning rate for SGD.
        momentum: Momentum factor.
        velocities: SGD momentum velocities (82 fields, one per trainable parameter).
        loss_history: Mutable list to record per-batch loss for convergence testing.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        max_batches: Cap batches this epoch (0 = unbounded); used by --smoke /
            --max-batches to bound a per-PR run (#5551).

    Returns:
        Average training loss for the epoch.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    if max_batches > 0 and max_batches < num_batches:
        num_batches = max_batches
    var total_loss = Float32(0.0)

    print("Epoch " + String(epoch) + "/" + String(total_epochs))

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            train_images, train_labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels_raw = batch_pair[1]

        # One-hot encode labels for cross_entropy loss
        var batch_labels = one_hot_encode(batch_labels_raw, 10)

        # Forward, backward, and SGD update for this batch
        var batch_loss = train_step(
            model,
            batch_images,
            batch_labels,
            velocities,
            lr=Float64(learning_rate),
            momentum=Float64(momentum),
        )
        loss_history.append(batch_loss)
        total_loss = total_loss + batch_loss

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0 or num_batches <= 10:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            print(
                "  Batch "
                + String(batch_idx + 1)
                + "/"
                + String(num_batches)
                + ", Loss: "
                + String(avg_loss)
            )

    var avg_loss = total_loss / Float32(num_batches)
    return avg_loss


def main() raises:
    """Main training loop for ResNet-18 on CIFAR-10."""
    print("=" * 60)
    print("ResNet-18 Training on CIFAR-10")
    print("=" * 60)
    print()

    # Parse arguments using standardized TrainingArgs
    var args = parse_training_args_with_defaults(
        default_epochs=200,
        default_batch_size=128,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets/cifar10",
        default_weights_dir="resnet18_weights",
        default_lr_decay_epochs=60,
        default_lr_decay_factor=0.2,
    )

    var epochs = args.epochs
    var batch_size = args.batch_size
    var initial_lr = Float32(args.learning_rate)
    var momentum = Float32(args.momentum)
    var data_dir = args.data_dir
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
    print(
        "  LR decay: "
        + String(lr_decay_factor)
        + "x every "
        + String(lr_decay_epochs)
        + " epochs"
    )
    print()

    # Load data — real CIFAR-10, or a tiny in-process synthetic batch in smoke
    # mode (#5551): smoke skips the dataset download entirely so the training
    # entrypoint can run per-PR in CI. It checks the MECHANISM (the loop runs
    # and emits finite, parseable, decreasing loss), not convergence.
    var train_images: AnyTensor
    var train_labels_raw: AnyTensor
    var test_images: AnyTensor
    var test_labels_raw: AnyTensor
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
        train_labels_raw = zeros([n_smoke], DType.uint8)
        var lbl_d = train_labels_raw._data.bitcast[UInt8]()
        for s in range(n_smoke):
            lbl_d[s] = UInt8(s % 10)
        # Reuse the same synthetic batch for "test" (eval is not asserted here).
        test_images = train_images
        test_labels_raw = train_labels_raw
    else:
        print("Loading CIFAR-10 dataset...")
        var dataset = CIFAR10Dataset(data_dir)
        train_images, train_labels_raw = dataset.get_train_data()
        test_images, test_labels_raw = dataset.get_test_data()

    print("  Training samples: " + String(train_images.shape()[0]))
    print("  Test samples: " + String(test_images.shape()[0]))
    print()

    # Initialize model
    print("Initializing ResNet-18 model...")
    var dataset_info = DatasetInfo("cifar10")
    var num_classes = dataset_info.num_classes()
    var model = ResNet18(num_classes=num_classes)
    print("  Total trainable parameters: 82")
    print("  Model size: ~11M parameters (actual tensor elements)")
    print()

    # Initialize momentum velocities (82 parameters, one per trainable param)
    print("Initializing momentum velocities...")
    var velocities = initialize_velocities(model)
    print()

    # Run training loop (1 epoch on real CIFAR-10 data)
    var model_mut: ResNet18 = model^
    var vel_mut: ResNet18Velocities = velocities^
    var loss_history: List[Float32] = []
    var epoch_loss = train_epoch(
        model_mut,
        train_images,
        train_labels_raw,
        batch_size=Int(batch_size),
        learning_rate=initial_lr,
        momentum=momentum,
        velocities=vel_mut,
        loss_history=loss_history,
        epoch=1,
        total_epochs=1,
        max_batches=max_batches,
    )
    print()
    print("Training complete. Epoch loss: " + String(epoch_loss))
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    var eval_result = evaluate_test_set(
        model_mut, test_images, test_labels_raw, batch_size=100
    )
    var test_loss = eval_result[0]
    var test_accuracy = eval_result[1]
    print("Test loss: " + String(test_loss))
    print("Test accuracy: " + String(test_accuracy) + "%")
    print()
