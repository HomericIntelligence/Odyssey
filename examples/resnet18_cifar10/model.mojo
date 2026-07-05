"""ResNet-18 Model for CIFAR-10 Classification.

ResNet-18 architecture adapted for CIFAR-10 dataset (32x32 RGB images).

Architecture:
    ResNet-18 uses residual blocks with skip connections to enable training
    very deep networks. The key innovation is identity mapping that allows
    gradients to flow directly through the network.

    Input (32×32×3) →

    Initial Block:
        Conv2D(64, 3×3, stride=1, pad=1) → BatchNorm → ReLU

    Stage 1 (64 channels, no downsampling):
        ResBlock(64→64) → ResBlock(64→64)

    Stage 2 (128 channels, downsample):
        ResBlock(64→128, stride=2) → ResBlock(128→128)

    Stage 3 (256 channels, downsample):
        ResBlock(128→256, stride=2) → ResBlock(256→256)

    Stage 4 (512 channels, downsample):
        ResBlock(256→512, stride=2) → ResBlock(512→512)

    Global Average Pool (4×4 → 1×1) →
    Flatten (512) →
    Linear(512 → 10)

    Each ResBlock:
        x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU
        If stride=2 or channels change:
            x → Conv1×1(stride) → BN → (+) → ReLU  (projection shortcut)

Key Innovation:
    - Skip connections (residual learning)
    - Identity mapping enables gradient flow
    - Batch normalization after each convolution
    - Solves vanishing gradient problem in deep networks

Shared Modules Used:
    - projectodyssey.core: Core tensor operations (AnyTensor, zeros, ones)
    - projectodyssey.core.conv: Convolution operations (conv2d, conv2d_backward)
    - projectodyssey.core.pooling: Pooling operations (avgpool2d, avgpool2d_backward)
    - projectodyssey.core.linear: Linear/fully-connected layers (linear, linear_backward)
    - projectodyssey.core.activation: Activation functions (relu, relu_backward)
    - projectodyssey.core.normalization: Batch normalization (batch_norm2d)
    - projectodyssey.core.initializers: Weight initialization (he_uniform)
    - projectodyssey.core.arithmetic: Element-wise operations (add for skip connections)
    - projectodyssey.training.optimizers: Optimization algorithms (sgd_momentum_update_inplace)
    - projectodyssey.utils.serialization: Model persistence (save_tensor, load_tensor)

References:
    - He, K., Zhang, X., Ren, S., & Sun, J. (2015).
      Deep residual learning for image recognition.
      arXiv preprint arXiv:1512.03385.
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import avgpool2d, avgpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.normalization import batch_norm2d
from projectodyssey.core.initializers import he_uniform
from projectodyssey.core.arithmetic import add  # For skip connections
from projectodyssey.training.optimizers import sgd_momentum_update_inplace
from projectodyssey.training.model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
)
from projectodyssey.utils.serialization import save_tensor, load_tensor
from std.collections import List


@fieldwise_init
struct IdentityCache(Movable):
    """Per-block activation cache for identity-shortcut residual blocks.

    Fields hold intermediate activations the backward pass needs. bn1/bn2
    running-mean/var snapshots capture pre-update stats — safe under AnyTensor
    refcounting (see docstring at any_tensor.mojo:96).
    """

    var block_input: AnyTensor
    var conv1_pre_bn: AnyTensor
    var bn1_pre_relu: AnyTensor
    var relu1_out: AnyTensor
    var conv2_pre_bn: AnyTensor
    var bn2_pre_relu: AnyTensor
    var skip_sum: AnyTensor
    var block_out: AnyTensor
    var bn1_running_mean_snapshot: AnyTensor
    var bn1_running_var_snapshot: AnyTensor
    var bn2_running_mean_snapshot: AnyTensor
    var bn2_running_var_snapshot: AnyTensor


@fieldwise_init
struct ProjectionCache(Movable):
    """Per-block cache for projection-shortcut residual blocks (stride-2 stages).

    Adds the 1×1 projection branch tensors on top of IdentityCache's fields.
    """

    var block_input: AnyTensor
    var conv1_pre_bn: AnyTensor
    var bn1_pre_relu: AnyTensor
    var relu1_out: AnyTensor
    var conv2_pre_bn: AnyTensor
    var bn2_pre_relu: AnyTensor
    var proj_conv_pre_bn: AnyTensor
    var proj_bn_out: AnyTensor
    var skip_sum: AnyTensor
    var block_out: AnyTensor
    var bn1_running_mean_snapshot: AnyTensor
    var bn1_running_var_snapshot: AnyTensor
    var bn2_running_mean_snapshot: AnyTensor
    var bn2_running_var_snapshot: AnyTensor
    var proj_bn_running_mean_snapshot: AnyTensor
    var proj_bn_running_var_snapshot: AnyTensor


@fieldwise_init
struct ResNet18ForwardCache(Movable):
    """Complete activation cache from forward_with_cache — one named field per
    intermediate value backward will need. Replaces the previously-planned
    16-element Tuple to avoid POLA violations and Mojo Tuple size uncertainty.
    """

    var logits: AnyTensor
    # Initial stem
    var conv1_pre_bn: AnyTensor
    var bn1_pre_relu: AnyTensor
    var relu1_out: AnyTensor
    var initial_bn_running_mean_snapshot: AnyTensor
    var initial_bn_running_var_snapshot: AnyTensor
    # 8 residual blocks (identity for s1b1/s1b2/s2b2/s3b2/s4b2, projection for s2b1/s3b1/s4b1)
    var s1b1_cache: IdentityCache
    var s1b2_cache: IdentityCache
    var s2b1_cache: ProjectionCache
    var s2b2_cache: IdentityCache
    var s3b1_cache: ProjectionCache
    var s3b2_cache: IdentityCache
    var s4b1_cache: ProjectionCache
    var s4b2_cache: IdentityCache
    # GAP + flatten (pre-FC)
    var gap: AnyTensor
    var flattened: AnyTensor


@fieldwise_init
struct ResNet18Velocities(Movable):
    """SGD-momentum velocity buffers, one field per trainable ResNet-18 parameter.

    Total: 82 trainable parameter fields (matches the ResNet18 struct docstring's
    "82 parameters" count; bn1_running_mean/var are non-trainable BatchNorm
    statistics and are excluded from this total).
    """

    var conv1_kernel: AnyTensor
    var conv1_bias: AnyTensor
    var bn1_gamma: AnyTensor
    var bn1_beta: AnyTensor
    # Stage 1 — no projection
    var s1b1_conv1_kernel: AnyTensor
    var s1b1_conv1_bias: AnyTensor
    var s1b1_bn1_gamma: AnyTensor
    var s1b1_bn1_beta: AnyTensor
    var s1b1_conv2_kernel: AnyTensor
    var s1b1_conv2_bias: AnyTensor
    var s1b1_bn2_gamma: AnyTensor
    var s1b1_bn2_beta: AnyTensor
    var s1b2_conv1_kernel: AnyTensor
    var s1b2_conv1_bias: AnyTensor
    var s1b2_bn1_gamma: AnyTensor
    var s1b2_bn1_beta: AnyTensor
    var s1b2_conv2_kernel: AnyTensor
    var s1b2_conv2_bias: AnyTensor
    var s1b2_bn2_gamma: AnyTensor
    var s1b2_bn2_beta: AnyTensor
    # Stage 2 (block1 has projection)
    var s2b1_conv1_kernel: AnyTensor
    var s2b1_conv1_bias: AnyTensor
    var s2b1_bn1_gamma: AnyTensor
    var s2b1_bn1_beta: AnyTensor
    var s2b1_conv2_kernel: AnyTensor
    var s2b1_conv2_bias: AnyTensor
    var s2b1_bn2_gamma: AnyTensor
    var s2b1_bn2_beta: AnyTensor
    var s2b1_proj_kernel: AnyTensor
    var s2b1_proj_bias: AnyTensor
    var s2b1_proj_bn_gamma: AnyTensor
    var s2b1_proj_bn_beta: AnyTensor
    var s2b2_conv1_kernel: AnyTensor
    var s2b2_conv1_bias: AnyTensor
    var s2b2_bn1_gamma: AnyTensor
    var s2b2_bn1_beta: AnyTensor
    var s2b2_conv2_kernel: AnyTensor
    var s2b2_conv2_bias: AnyTensor
    var s2b2_bn2_gamma: AnyTensor
    var s2b2_bn2_beta: AnyTensor
    # Stage 3 (block1 has projection)
    var s3b1_conv1_kernel: AnyTensor
    var s3b1_conv1_bias: AnyTensor
    var s3b1_bn1_gamma: AnyTensor
    var s3b1_bn1_beta: AnyTensor
    var s3b1_conv2_kernel: AnyTensor
    var s3b1_conv2_bias: AnyTensor
    var s3b1_bn2_gamma: AnyTensor
    var s3b1_bn2_beta: AnyTensor
    var s3b1_proj_kernel: AnyTensor
    var s3b1_proj_bias: AnyTensor
    var s3b1_proj_bn_gamma: AnyTensor
    var s3b1_proj_bn_beta: AnyTensor
    var s3b2_conv1_kernel: AnyTensor
    var s3b2_conv1_bias: AnyTensor
    var s3b2_bn1_gamma: AnyTensor
    var s3b2_bn1_beta: AnyTensor
    var s3b2_conv2_kernel: AnyTensor
    var s3b2_conv2_bias: AnyTensor
    var s3b2_bn2_gamma: AnyTensor
    var s3b2_bn2_beta: AnyTensor
    # Stage 4 (block1 has projection)
    var s4b1_conv1_kernel: AnyTensor
    var s4b1_conv1_bias: AnyTensor
    var s4b1_bn1_gamma: AnyTensor
    var s4b1_bn1_beta: AnyTensor
    var s4b1_conv2_kernel: AnyTensor
    var s4b1_conv2_bias: AnyTensor
    var s4b1_bn2_gamma: AnyTensor
    var s4b1_bn2_beta: AnyTensor
    var s4b1_proj_kernel: AnyTensor
    var s4b1_proj_bias: AnyTensor
    var s4b1_proj_bn_gamma: AnyTensor
    var s4b1_proj_bn_beta: AnyTensor
    var s4b2_conv1_kernel: AnyTensor
    var s4b2_conv1_bias: AnyTensor
    var s4b2_bn1_gamma: AnyTensor
    var s4b2_bn1_beta: AnyTensor
    var s4b2_conv2_kernel: AnyTensor
    var s4b2_conv2_bias: AnyTensor
    var s4b2_bn2_gamma: AnyTensor
    var s4b2_bn2_beta: AnyTensor
    # FC head
    var fc_weights: AnyTensor
    var fc_bias: AnyTensor


def initialize_velocities(model: ResNet18) raises -> ResNet18Velocities:
    """Zero-fill velocity buffers matching every ResNet18 trainable parameter shape.

    Uses keyword-argument construction, which @fieldwise_init supports (verified
    against src/projectodyssey/training/precision_config.mojo:fp32() factory).
    """
    return ResNet18Velocities(
        conv1_kernel=zeros(model.conv1_kernel.shape(), DType.float32),
        conv1_bias=zeros(model.conv1_bias.shape(), DType.float32),
        bn1_gamma=zeros(model.bn1_gamma.shape(), DType.float32),
        bn1_beta=zeros(model.bn1_beta.shape(), DType.float32),
        s1b1_conv1_kernel=zeros(model.s1b1_conv1_kernel.shape(), DType.float32),
        s1b1_conv1_bias=zeros(model.s1b1_conv1_bias.shape(), DType.float32),
        s1b1_bn1_gamma=zeros(model.s1b1_bn1_gamma.shape(), DType.float32),
        s1b1_bn1_beta=zeros(model.s1b1_bn1_beta.shape(), DType.float32),
        s1b1_conv2_kernel=zeros(model.s1b1_conv2_kernel.shape(), DType.float32),
        s1b1_conv2_bias=zeros(model.s1b1_conv2_bias.shape(), DType.float32),
        s1b1_bn2_gamma=zeros(model.s1b1_bn2_gamma.shape(), DType.float32),
        s1b1_bn2_beta=zeros(model.s1b1_bn2_beta.shape(), DType.float32),
        s1b2_conv1_kernel=zeros(model.s1b2_conv1_kernel.shape(), DType.float32),
        s1b2_conv1_bias=zeros(model.s1b2_conv1_bias.shape(), DType.float32),
        s1b2_bn1_gamma=zeros(model.s1b2_bn1_gamma.shape(), DType.float32),
        s1b2_bn1_beta=zeros(model.s1b2_bn1_beta.shape(), DType.float32),
        s1b2_conv2_kernel=zeros(model.s1b2_conv2_kernel.shape(), DType.float32),
        s1b2_conv2_bias=zeros(model.s1b2_conv2_bias.shape(), DType.float32),
        s1b2_bn2_gamma=zeros(model.s1b2_bn2_gamma.shape(), DType.float32),
        s1b2_bn2_beta=zeros(model.s1b2_bn2_beta.shape(), DType.float32),
        s2b1_conv1_kernel=zeros(model.s2b1_conv1_kernel.shape(), DType.float32),
        s2b1_conv1_bias=zeros(model.s2b1_conv1_bias.shape(), DType.float32),
        s2b1_bn1_gamma=zeros(model.s2b1_bn1_gamma.shape(), DType.float32),
        s2b1_bn1_beta=zeros(model.s2b1_bn1_beta.shape(), DType.float32),
        s2b1_conv2_kernel=zeros(model.s2b1_conv2_kernel.shape(), DType.float32),
        s2b1_conv2_bias=zeros(model.s2b1_conv2_bias.shape(), DType.float32),
        s2b1_bn2_gamma=zeros(model.s2b1_bn2_gamma.shape(), DType.float32),
        s2b1_bn2_beta=zeros(model.s2b1_bn2_beta.shape(), DType.float32),
        s2b1_proj_kernel=zeros(model.s2b1_proj_kernel.shape(), DType.float32),
        s2b1_proj_bias=zeros(model.s2b1_proj_bias.shape(), DType.float32),
        s2b1_proj_bn_gamma=zeros(
            model.s2b1_proj_bn_gamma.shape(), DType.float32
        ),
        s2b1_proj_bn_beta=zeros(model.s2b1_proj_bn_beta.shape(), DType.float32),
        s2b2_conv1_kernel=zeros(model.s2b2_conv1_kernel.shape(), DType.float32),
        s2b2_conv1_bias=zeros(model.s2b2_conv1_bias.shape(), DType.float32),
        s2b2_bn1_gamma=zeros(model.s2b2_bn1_gamma.shape(), DType.float32),
        s2b2_bn1_beta=zeros(model.s2b2_bn1_beta.shape(), DType.float32),
        s2b2_conv2_kernel=zeros(model.s2b2_conv2_kernel.shape(), DType.float32),
        s2b2_conv2_bias=zeros(model.s2b2_conv2_bias.shape(), DType.float32),
        s2b2_bn2_gamma=zeros(model.s2b2_bn2_gamma.shape(), DType.float32),
        s2b2_bn2_beta=zeros(model.s2b2_bn2_beta.shape(), DType.float32),
        s3b1_conv1_kernel=zeros(model.s3b1_conv1_kernel.shape(), DType.float32),
        s3b1_conv1_bias=zeros(model.s3b1_conv1_bias.shape(), DType.float32),
        s3b1_bn1_gamma=zeros(model.s3b1_bn1_gamma.shape(), DType.float32),
        s3b1_bn1_beta=zeros(model.s3b1_bn1_beta.shape(), DType.float32),
        s3b1_conv2_kernel=zeros(model.s3b1_conv2_kernel.shape(), DType.float32),
        s3b1_conv2_bias=zeros(model.s3b1_conv2_bias.shape(), DType.float32),
        s3b1_bn2_gamma=zeros(model.s3b1_bn2_gamma.shape(), DType.float32),
        s3b1_bn2_beta=zeros(model.s3b1_bn2_beta.shape(), DType.float32),
        s3b1_proj_kernel=zeros(model.s3b1_proj_kernel.shape(), DType.float32),
        s3b1_proj_bias=zeros(model.s3b1_proj_bias.shape(), DType.float32),
        s3b1_proj_bn_gamma=zeros(
            model.s3b1_proj_bn_gamma.shape(), DType.float32
        ),
        s3b1_proj_bn_beta=zeros(model.s3b1_proj_bn_beta.shape(), DType.float32),
        s3b2_conv1_kernel=zeros(model.s3b2_conv1_kernel.shape(), DType.float32),
        s3b2_conv1_bias=zeros(model.s3b2_conv1_bias.shape(), DType.float32),
        s3b2_bn1_gamma=zeros(model.s3b2_bn1_gamma.shape(), DType.float32),
        s3b2_bn1_beta=zeros(model.s3b2_bn1_beta.shape(), DType.float32),
        s3b2_conv2_kernel=zeros(model.s3b2_conv2_kernel.shape(), DType.float32),
        s3b2_conv2_bias=zeros(model.s3b2_conv2_bias.shape(), DType.float32),
        s3b2_bn2_gamma=zeros(model.s3b2_bn2_gamma.shape(), DType.float32),
        s3b2_bn2_beta=zeros(model.s3b2_bn2_beta.shape(), DType.float32),
        s4b1_conv1_kernel=zeros(model.s4b1_conv1_kernel.shape(), DType.float32),
        s4b1_conv1_bias=zeros(model.s4b1_conv1_bias.shape(), DType.float32),
        s4b1_bn1_gamma=zeros(model.s4b1_bn1_gamma.shape(), DType.float32),
        s4b1_bn1_beta=zeros(model.s4b1_bn1_beta.shape(), DType.float32),
        s4b1_conv2_kernel=zeros(model.s4b1_conv2_kernel.shape(), DType.float32),
        s4b1_conv2_bias=zeros(model.s4b1_conv2_bias.shape(), DType.float32),
        s4b1_bn2_gamma=zeros(model.s4b1_bn2_gamma.shape(), DType.float32),
        s4b1_bn2_beta=zeros(model.s4b1_bn2_beta.shape(), DType.float32),
        s4b1_proj_kernel=zeros(model.s4b1_proj_kernel.shape(), DType.float32),
        s4b1_proj_bias=zeros(model.s4b1_proj_bias.shape(), DType.float32),
        s4b1_proj_bn_gamma=zeros(
            model.s4b1_proj_bn_gamma.shape(), DType.float32
        ),
        s4b1_proj_bn_beta=zeros(model.s4b1_proj_bn_beta.shape(), DType.float32),
        s4b2_conv1_kernel=zeros(model.s4b2_conv1_kernel.shape(), DType.float32),
        s4b2_conv1_bias=zeros(model.s4b2_conv1_bias.shape(), DType.float32),
        s4b2_bn1_gamma=zeros(model.s4b2_bn1_gamma.shape(), DType.float32),
        s4b2_bn1_beta=zeros(model.s4b2_bn1_beta.shape(), DType.float32),
        s4b2_conv2_kernel=zeros(model.s4b2_conv2_kernel.shape(), DType.float32),
        s4b2_conv2_bias=zeros(model.s4b2_conv2_bias.shape(), DType.float32),
        s4b2_bn2_gamma=zeros(model.s4b2_bn2_gamma.shape(), DType.float32),
        s4b2_bn2_beta=zeros(model.s4b2_bn2_beta.shape(), DType.float32),
        fc_weights=zeros(model.fc_weights.shape(), DType.float32),
        fc_bias=zeros(model.fc_bias.shape(), DType.float32),
    )


struct ResNet18(Movable):
    """ResNet-18 model for CIFAR-10 classification.

    Adapted for 32×32 input (vs 224×224 in original paper):
    - Use 3×3 initial conv instead of 7×7
    - Remove initial max pooling
    - Keep 4 residual stages with 2 blocks each

    Total layers: 1 (initial) + 8 (residual blocks) × 2 + 1 (FC) = 18 layers

    Attributes:
        num_classes: Number of output classes (10 for CIFAR-10)

        # Initial convolution (4 trainable params; BN running stats are buffers)
        conv1_kernel, conv1_bias: (64, 3, 3, 3)
        bn1_gamma, bn1_beta, bn1_running_mean, bn1_running_var: (64,)

        # Stage 1: 2 blocks, 64 channels (16 params, no projection)
        # Stage 2: 2 blocks, 128 channels (20 params, block1 has projection)
        # Stage 3: 2 blocks, 256 channels (20 params, block1 has projection)
        # Stage 4: 2 blocks, 512 channels (20 params, block1 has projection)

        # Fully connected (2 params)
        fc_weights, fc_bias: (num_classes, 512)

        # Note: bn1_running_mean/bn1_running_var are non-trainable BatchNorm
        # statistics, so the initial block contributes 4 trainable params (not 6).
        Total trainable params: 4 + 16 + 20 + 20 + 20 + 2 = 82 parameters.
    """

    var num_classes: Int

    # Initial conv + BN (6 params)
    var conv1_kernel: AnyTensor
    var conv1_bias: AnyTensor
    var bn1_gamma: AnyTensor
    var bn1_beta: AnyTensor
    var bn1_running_mean: AnyTensor
    var bn1_running_var: AnyTensor

    # ========== Stage 1 (64 channels): 2 blocks, no projection ==========
    # Block 1 (8 params)
    var s1b1_conv1_kernel: AnyTensor
    var s1b1_conv1_bias: AnyTensor
    var s1b1_bn1_gamma: AnyTensor
    var s1b1_bn1_beta: AnyTensor
    var s1b1_bn1_running_mean: AnyTensor
    var s1b1_bn1_running_var: AnyTensor
    var s1b1_conv2_kernel: AnyTensor
    var s1b1_conv2_bias: AnyTensor
    var s1b1_bn2_gamma: AnyTensor
    var s1b1_bn2_beta: AnyTensor
    var s1b1_bn2_running_mean: AnyTensor
    var s1b1_bn2_running_var: AnyTensor

    # Block 2 (8 params)
    var s1b2_conv1_kernel: AnyTensor
    var s1b2_conv1_bias: AnyTensor
    var s1b2_bn1_gamma: AnyTensor
    var s1b2_bn1_beta: AnyTensor
    var s1b2_bn1_running_mean: AnyTensor
    var s1b2_bn1_running_var: AnyTensor
    var s1b2_conv2_kernel: AnyTensor
    var s1b2_conv2_bias: AnyTensor
    var s1b2_bn2_gamma: AnyTensor
    var s1b2_bn2_beta: AnyTensor
    var s1b2_bn2_running_mean: AnyTensor
    var s1b2_bn2_running_var: AnyTensor

    # ========== Stage 2 (128 channels): 2 blocks, block1 has projection ==========
    # Block 1 (12 params: 8 main + 4 projection)
    var s2b1_conv1_kernel: AnyTensor
    var s2b1_conv1_bias: AnyTensor
    var s2b1_bn1_gamma: AnyTensor
    var s2b1_bn1_beta: AnyTensor
    var s2b1_bn1_running_mean: AnyTensor
    var s2b1_bn1_running_var: AnyTensor
    var s2b1_conv2_kernel: AnyTensor
    var s2b1_conv2_bias: AnyTensor
    var s2b1_bn2_gamma: AnyTensor
    var s2b1_bn2_beta: AnyTensor
    var s2b1_bn2_running_mean: AnyTensor
    var s2b1_bn2_running_var: AnyTensor
    # Projection shortcut (4 params)
    var s2b1_proj_kernel: AnyTensor
    var s2b1_proj_bias: AnyTensor
    var s2b1_proj_bn_gamma: AnyTensor
    var s2b1_proj_bn_beta: AnyTensor
    var s2b1_proj_bn_running_mean: AnyTensor
    var s2b1_proj_bn_running_var: AnyTensor

    # Block 2 (8 params)
    var s2b2_conv1_kernel: AnyTensor
    var s2b2_conv1_bias: AnyTensor
    var s2b2_bn1_gamma: AnyTensor
    var s2b2_bn1_beta: AnyTensor
    var s2b2_bn1_running_mean: AnyTensor
    var s2b2_bn1_running_var: AnyTensor
    var s2b2_conv2_kernel: AnyTensor
    var s2b2_conv2_bias: AnyTensor
    var s2b2_bn2_gamma: AnyTensor
    var s2b2_bn2_beta: AnyTensor
    var s2b2_bn2_running_mean: AnyTensor
    var s2b2_bn2_running_var: AnyTensor

    # ========== Stage 3 (256 channels): 2 blocks, block1 has projection ==========
    # Block 1 (12 params: 8 main + 4 projection)
    var s3b1_conv1_kernel: AnyTensor
    var s3b1_conv1_bias: AnyTensor
    var s3b1_bn1_gamma: AnyTensor
    var s3b1_bn1_beta: AnyTensor
    var s3b1_bn1_running_mean: AnyTensor
    var s3b1_bn1_running_var: AnyTensor
    var s3b1_conv2_kernel: AnyTensor
    var s3b1_conv2_bias: AnyTensor
    var s3b1_bn2_gamma: AnyTensor
    var s3b1_bn2_beta: AnyTensor
    var s3b1_bn2_running_mean: AnyTensor
    var s3b1_bn2_running_var: AnyTensor
    # Projection shortcut (4 params)
    var s3b1_proj_kernel: AnyTensor
    var s3b1_proj_bias: AnyTensor
    var s3b1_proj_bn_gamma: AnyTensor
    var s3b1_proj_bn_beta: AnyTensor
    var s3b1_proj_bn_running_mean: AnyTensor
    var s3b1_proj_bn_running_var: AnyTensor

    # Block 2 (8 params)
    var s3b2_conv1_kernel: AnyTensor
    var s3b2_conv1_bias: AnyTensor
    var s3b2_bn1_gamma: AnyTensor
    var s3b2_bn1_beta: AnyTensor
    var s3b2_bn1_running_mean: AnyTensor
    var s3b2_bn1_running_var: AnyTensor
    var s3b2_conv2_kernel: AnyTensor
    var s3b2_conv2_bias: AnyTensor
    var s3b2_bn2_gamma: AnyTensor
    var s3b2_bn2_beta: AnyTensor
    var s3b2_bn2_running_mean: AnyTensor
    var s3b2_bn2_running_var: AnyTensor

    # ========== Stage 4 (512 channels): 2 blocks, block1 has projection ==========
    # Block 1 (12 params: 8 main + 4 projection)
    var s4b1_conv1_kernel: AnyTensor
    var s4b1_conv1_bias: AnyTensor
    var s4b1_bn1_gamma: AnyTensor
    var s4b1_bn1_beta: AnyTensor
    var s4b1_bn1_running_mean: AnyTensor
    var s4b1_bn1_running_var: AnyTensor
    var s4b1_conv2_kernel: AnyTensor
    var s4b1_conv2_bias: AnyTensor
    var s4b1_bn2_gamma: AnyTensor
    var s4b1_bn2_beta: AnyTensor
    var s4b1_bn2_running_mean: AnyTensor
    var s4b1_bn2_running_var: AnyTensor
    # Projection shortcut (4 params)
    var s4b1_proj_kernel: AnyTensor
    var s4b1_proj_bias: AnyTensor
    var s4b1_proj_bn_gamma: AnyTensor
    var s4b1_proj_bn_beta: AnyTensor
    var s4b1_proj_bn_running_mean: AnyTensor
    var s4b1_proj_bn_running_var: AnyTensor

    # Block 2 (8 params)
    var s4b2_conv1_kernel: AnyTensor
    var s4b2_conv1_bias: AnyTensor
    var s4b2_bn1_gamma: AnyTensor
    var s4b2_bn1_beta: AnyTensor
    var s4b2_bn1_running_mean: AnyTensor
    var s4b2_bn1_running_var: AnyTensor
    var s4b2_conv2_kernel: AnyTensor
    var s4b2_conv2_bias: AnyTensor
    var s4b2_bn2_gamma: AnyTensor
    var s4b2_bn2_beta: AnyTensor
    var s4b2_bn2_running_mean: AnyTensor
    var s4b2_bn2_running_var: AnyTensor

    # FC layer (2 params)
    var fc_weights: AnyTensor
    var fc_bias: AnyTensor

    def __init__(out self, num_classes: Int = 10) raises:
        """Initialize ResNet-18 model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10).
        """
        self.num_classes = num_classes

        # ========== Initial conv: 3 → 64 channels, 3×3 kernel ==========
        var conv1_shape = List[Int]()
        conv1_shape.append(64)  # out_channels
        conv1_shape.append(3)  # in_channels (RGB)
        conv1_shape.append(3)  # kernel_height
        conv1_shape.append(3)  # kernel_width
        self.conv1_kernel = he_uniform(conv1_shape, DType.float32)

        var conv1_bias_shape = List[Int]()
        conv1_bias_shape.append(64)
        self.conv1_bias = zeros(conv1_bias_shape, DType.float32)

        var bn1_shape = List[Int]()
        bn1_shape.append(64)
        self.bn1_gamma = ones(bn1_shape, DType.float32)
        self.bn1_beta = zeros(bn1_shape, DType.float32)
        self.bn1_running_mean = zeros(bn1_shape, DType.float32)
        self.bn1_running_var = ones(bn1_shape, DType.float32)

        # ========== Stage 1: 64 → 64 (no projection) ==========
        # Block 1
        var s1_shape = List[Int]()
        s1_shape.append(64)
        s1_shape.append(64)
        s1_shape.append(3)
        s1_shape.append(3)

        var s1_bias_shape = List[Int]()
        s1_bias_shape.append(64)

        var s1_bn_shape = List[Int]()
        s1_bn_shape.append(64)

        self.s1b1_conv1_kernel = he_uniform(s1_shape, DType.float32)
        self.s1b1_conv1_bias = zeros(s1_bias_shape, DType.float32)
        self.s1b1_bn1_gamma = ones(s1_bn_shape, DType.float32)
        self.s1b1_bn1_beta = zeros(s1_bn_shape, DType.float32)
        self.s1b1_bn1_running_mean = zeros(s1_bn_shape, DType.float32)
        self.s1b1_bn1_running_var = ones(s1_bn_shape, DType.float32)

        self.s1b1_conv2_kernel = he_uniform(s1_shape, DType.float32)
        self.s1b1_conv2_bias = zeros(s1_bias_shape, DType.float32)
        self.s1b1_bn2_gamma = ones(s1_bn_shape, DType.float32)
        self.s1b1_bn2_beta = zeros(s1_bn_shape, DType.float32)
        self.s1b1_bn2_running_mean = zeros(s1_bn_shape, DType.float32)
        self.s1b1_bn2_running_var = ones(s1_bn_shape, DType.float32)

        # Block 2
        self.s1b2_conv1_kernel = he_uniform(s1_shape, DType.float32)
        self.s1b2_conv1_bias = zeros(s1_bias_shape, DType.float32)
        self.s1b2_bn1_gamma = ones(s1_bn_shape, DType.float32)
        self.s1b2_bn1_beta = zeros(s1_bn_shape, DType.float32)
        self.s1b2_bn1_running_mean = zeros(s1_bn_shape, DType.float32)
        self.s1b2_bn1_running_var = ones(s1_bn_shape, DType.float32)

        self.s1b2_conv2_kernel = he_uniform(s1_shape, DType.float32)
        self.s1b2_conv2_bias = zeros(s1_bias_shape, DType.float32)
        self.s1b2_bn2_gamma = ones(s1_bn_shape, DType.float32)
        self.s1b2_bn2_beta = zeros(s1_bn_shape, DType.float32)
        self.s1b2_bn2_running_mean = zeros(s1_bn_shape, DType.float32)
        self.s1b2_bn2_running_var = ones(s1_bn_shape, DType.float32)

        # ========== Stage 2: 64 → 128 → 128 (block1 has projection) ==========
        # Block 1 (stride=2 for first conv)
        var s2b1_conv1_shape = List[Int]()
        s2b1_conv1_shape.append(128)
        s2b1_conv1_shape.append(64)
        s2b1_conv1_shape.append(3)
        s2b1_conv1_shape.append(3)

        var s2b1_conv2_shape = List[Int]()
        s2b1_conv2_shape.append(128)
        s2b1_conv2_shape.append(128)
        s2b1_conv2_shape.append(3)
        s2b1_conv2_shape.append(3)

        var s2_bias_shape = List[Int]()
        s2_bias_shape.append(128)

        var s2_bn_shape = List[Int]()
        s2_bn_shape.append(128)

        self.s2b1_conv1_kernel = he_uniform(s2b1_conv1_shape, DType.float32)
        self.s2b1_conv1_bias = zeros(s2_bias_shape, DType.float32)
        self.s2b1_bn1_gamma = ones(s2_bn_shape, DType.float32)
        self.s2b1_bn1_beta = zeros(s2_bn_shape, DType.float32)
        self.s2b1_bn1_running_mean = zeros(s2_bn_shape, DType.float32)
        self.s2b1_bn1_running_var = ones(s2_bn_shape, DType.float32)

        self.s2b1_conv2_kernel = he_uniform(s2b1_conv2_shape, DType.float32)
        self.s2b1_conv2_bias = zeros(s2_bias_shape, DType.float32)
        self.s2b1_bn2_gamma = ones(s2_bn_shape, DType.float32)
        self.s2b1_bn2_beta = zeros(s2_bn_shape, DType.float32)
        self.s2b1_bn2_running_mean = zeros(s2_bn_shape, DType.float32)
        self.s2b1_bn2_running_var = ones(s2_bn_shape, DType.float32)

        # Projection shortcut: 1×1 conv, 64→128, stride=2
        var s2b1_proj_shape = List[Int]()
        s2b1_proj_shape.append(128)
        s2b1_proj_shape.append(64)
        s2b1_proj_shape.append(1)
        s2b1_proj_shape.append(1)

        self.s2b1_proj_kernel = he_uniform(s2b1_proj_shape, DType.float32)
        self.s2b1_proj_bias = zeros(s2_bias_shape, DType.float32)
        self.s2b1_proj_bn_gamma = ones(s2_bn_shape, DType.float32)
        self.s2b1_proj_bn_beta = zeros(s2_bn_shape, DType.float32)
        self.s2b1_proj_bn_running_mean = zeros(s2_bn_shape, DType.float32)
        self.s2b1_proj_bn_running_var = ones(s2_bn_shape, DType.float32)

        # Block 2
        self.s2b2_conv1_kernel = he_uniform(s2b1_conv2_shape, DType.float32)
        self.s2b2_conv1_bias = zeros(s2_bias_shape, DType.float32)
        self.s2b2_bn1_gamma = ones(s2_bn_shape, DType.float32)
        self.s2b2_bn1_beta = zeros(s2_bn_shape, DType.float32)
        self.s2b2_bn1_running_mean = zeros(s2_bn_shape, DType.float32)
        self.s2b2_bn1_running_var = ones(s2_bn_shape, DType.float32)

        self.s2b2_conv2_kernel = he_uniform(s2b1_conv2_shape, DType.float32)
        self.s2b2_conv2_bias = zeros(s2_bias_shape, DType.float32)
        self.s2b2_bn2_gamma = ones(s2_bn_shape, DType.float32)
        self.s2b2_bn2_beta = zeros(s2_bn_shape, DType.float32)
        self.s2b2_bn2_running_mean = zeros(s2_bn_shape, DType.float32)
        self.s2b2_bn2_running_var = ones(s2_bn_shape, DType.float32)

        # ========== Stage 3: 128 → 256 → 256 (block1 has projection) ==========
        # Block 1 (stride=2 for first conv)
        var s3b1_conv1_shape = List[Int]()
        s3b1_conv1_shape.append(256)
        s3b1_conv1_shape.append(128)
        s3b1_conv1_shape.append(3)
        s3b1_conv1_shape.append(3)

        var s3b1_conv2_shape = List[Int]()
        s3b1_conv2_shape.append(256)
        s3b1_conv2_shape.append(256)
        s3b1_conv2_shape.append(3)
        s3b1_conv2_shape.append(3)

        var s3_bias_shape = List[Int]()
        s3_bias_shape.append(256)

        var s3_bn_shape = List[Int]()
        s3_bn_shape.append(256)

        self.s3b1_conv1_kernel = he_uniform(s3b1_conv1_shape, DType.float32)
        self.s3b1_conv1_bias = zeros(s3_bias_shape, DType.float32)
        self.s3b1_bn1_gamma = ones(s3_bn_shape, DType.float32)
        self.s3b1_bn1_beta = zeros(s3_bn_shape, DType.float32)
        self.s3b1_bn1_running_mean = zeros(s3_bn_shape, DType.float32)
        self.s3b1_bn1_running_var = ones(s3_bn_shape, DType.float32)

        self.s3b1_conv2_kernel = he_uniform(s3b1_conv2_shape, DType.float32)
        self.s3b1_conv2_bias = zeros(s3_bias_shape, DType.float32)
        self.s3b1_bn2_gamma = ones(s3_bn_shape, DType.float32)
        self.s3b1_bn2_beta = zeros(s3_bn_shape, DType.float32)
        self.s3b1_bn2_running_mean = zeros(s3_bn_shape, DType.float32)
        self.s3b1_bn2_running_var = ones(s3_bn_shape, DType.float32)

        # Projection shortcut: 1×1 conv, 128→256, stride=2
        var s3b1_proj_shape = List[Int]()
        s3b1_proj_shape.append(256)
        s3b1_proj_shape.append(128)
        s3b1_proj_shape.append(1)
        s3b1_proj_shape.append(1)

        self.s3b1_proj_kernel = he_uniform(s3b1_proj_shape, DType.float32)
        self.s3b1_proj_bias = zeros(s3_bias_shape, DType.float32)
        self.s3b1_proj_bn_gamma = ones(s3_bn_shape, DType.float32)
        self.s3b1_proj_bn_beta = zeros(s3_bn_shape, DType.float32)
        self.s3b1_proj_bn_running_mean = zeros(s3_bn_shape, DType.float32)
        self.s3b1_proj_bn_running_var = ones(s3_bn_shape, DType.float32)

        # Block 2
        self.s3b2_conv1_kernel = he_uniform(s3b1_conv2_shape, DType.float32)
        self.s3b2_conv1_bias = zeros(s3_bias_shape, DType.float32)
        self.s3b2_bn1_gamma = ones(s3_bn_shape, DType.float32)
        self.s3b2_bn1_beta = zeros(s3_bn_shape, DType.float32)
        self.s3b2_bn1_running_mean = zeros(s3_bn_shape, DType.float32)
        self.s3b2_bn1_running_var = ones(s3_bn_shape, DType.float32)

        self.s3b2_conv2_kernel = he_uniform(s3b1_conv2_shape, DType.float32)
        self.s3b2_conv2_bias = zeros(s3_bias_shape, DType.float32)
        self.s3b2_bn2_gamma = ones(s3_bn_shape, DType.float32)
        self.s3b2_bn2_beta = zeros(s3_bn_shape, DType.float32)
        self.s3b2_bn2_running_mean = zeros(s3_bn_shape, DType.float32)
        self.s3b2_bn2_running_var = ones(s3_bn_shape, DType.float32)

        # ========== Stage 4: 256 → 512 → 512 (block1 has projection) ==========
        # Block 1 (stride=2 for first conv)
        var s4b1_conv1_shape = List[Int]()
        s4b1_conv1_shape.append(512)
        s4b1_conv1_shape.append(256)
        s4b1_conv1_shape.append(3)
        s4b1_conv1_shape.append(3)

        var s4b1_conv2_shape = List[Int]()
        s4b1_conv2_shape.append(512)
        s4b1_conv2_shape.append(512)
        s4b1_conv2_shape.append(3)
        s4b1_conv2_shape.append(3)

        var s4_bias_shape = List[Int]()
        s4_bias_shape.append(512)

        var s4_bn_shape = List[Int]()
        s4_bn_shape.append(512)

        self.s4b1_conv1_kernel = he_uniform(s4b1_conv1_shape, DType.float32)
        self.s4b1_conv1_bias = zeros(s4_bias_shape, DType.float32)
        self.s4b1_bn1_gamma = ones(s4_bn_shape, DType.float32)
        self.s4b1_bn1_beta = zeros(s4_bn_shape, DType.float32)
        self.s4b1_bn1_running_mean = zeros(s4_bn_shape, DType.float32)
        self.s4b1_bn1_running_var = ones(s4_bn_shape, DType.float32)

        self.s4b1_conv2_kernel = he_uniform(s4b1_conv2_shape, DType.float32)
        self.s4b1_conv2_bias = zeros(s4_bias_shape, DType.float32)
        self.s4b1_bn2_gamma = ones(s4_bn_shape, DType.float32)
        self.s4b1_bn2_beta = zeros(s4_bn_shape, DType.float32)
        self.s4b1_bn2_running_mean = zeros(s4_bn_shape, DType.float32)
        self.s4b1_bn2_running_var = ones(s4_bn_shape, DType.float32)

        # Projection shortcut: 1×1 conv, 256→512, stride=2
        var s4b1_proj_shape = List[Int]()
        s4b1_proj_shape.append(512)
        s4b1_proj_shape.append(256)
        s4b1_proj_shape.append(1)
        s4b1_proj_shape.append(1)

        self.s4b1_proj_kernel = he_uniform(s4b1_proj_shape, DType.float32)
        self.s4b1_proj_bias = zeros(s4_bias_shape, DType.float32)
        self.s4b1_proj_bn_gamma = ones(s4_bn_shape, DType.float32)
        self.s4b1_proj_bn_beta = zeros(s4_bn_shape, DType.float32)
        self.s4b1_proj_bn_running_mean = zeros(s4_bn_shape, DType.float32)
        self.s4b1_proj_bn_running_var = ones(s4_bn_shape, DType.float32)

        # Block 2
        self.s4b2_conv1_kernel = he_uniform(s4b1_conv2_shape, DType.float32)
        self.s4b2_conv1_bias = zeros(s4_bias_shape, DType.float32)
        self.s4b2_bn1_gamma = ones(s4_bn_shape, DType.float32)
        self.s4b2_bn1_beta = zeros(s4_bn_shape, DType.float32)
        self.s4b2_bn1_running_mean = zeros(s4_bn_shape, DType.float32)
        self.s4b2_bn1_running_var = ones(s4_bn_shape, DType.float32)

        self.s4b2_conv2_kernel = he_uniform(s4b1_conv2_shape, DType.float32)
        self.s4b2_conv2_bias = zeros(s4_bias_shape, DType.float32)
        self.s4b2_bn2_gamma = ones(s4_bn_shape, DType.float32)
        self.s4b2_bn2_beta = zeros(s4_bn_shape, DType.float32)
        self.s4b2_bn2_running_mean = zeros(s4_bn_shape, DType.float32)
        self.s4b2_bn2_running_var = ones(s4_bn_shape, DType.float32)

        # ========== FC layer: 512 → num_classes ==========
        var fc_shape = List[Int]()
        fc_shape.append(num_classes)
        fc_shape.append(512)
        self.fc_weights = he_uniform(fc_shape, DType.float32)

        var fc_bias_shape = List[Int]()
        fc_bias_shape.append(num_classes)
        self.fc_bias = zeros(fc_bias_shape, DType.float32)

    def forward(
        mut self, input: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass through ResNet-18.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32).
            training: Whether in training mode (updates BN running stats).

        Returns:
            Output logits of shape (batch, num_classes)

        Forward pass flow:
            Input (batch, 3, 32, 32) →
            Conv1+BN+ReLU → (batch, 64, 32, 32) →
            Stage1 → (batch, 64, 32, 32) →
            Stage2 → (batch, 128, 16, 16) →
            Stage3 → (batch, 256, 8, 8) →
            Stage4 → (batch, 512, 4, 4) →
            GAP → (batch, 512, 1, 1) →
            Flatten → (batch, 512) →
            FC → (batch, num_classes).
        """
        # ========== Initial conv + BN + ReLU ==========
        var conv1 = conv2d(
            input, self.conv1_kernel, self.conv1_bias, stride=1, padding=1
        )
        var bn1_result = batch_norm2d(
            conv1,
            self.bn1_gamma,
            self.bn1_beta,
            self.bn1_running_mean,
            self.bn1_running_var,
            training=training,
        )
        var bn1 = bn1_result[0]
        self.bn1_running_mean = bn1_result[1]
        self.bn1_running_var = bn1_result[2]
        var relu1 = relu(bn1)

        # ========== Stage 1, Block 1 (identity shortcut) ==========
        var s1b1_conv1 = conv2d(
            relu1,
            self.s1b1_conv1_kernel,
            self.s1b1_conv1_bias,
            stride=1,
            padding=1,
        )
        var s1b1_bn1_result = batch_norm2d(
            s1b1_conv1,
            self.s1b1_bn1_gamma,
            self.s1b1_bn1_beta,
            self.s1b1_bn1_running_mean,
            self.s1b1_bn1_running_var,
            training=training,
        )
        var s1b1_bn1 = s1b1_bn1_result[0]
        self.s1b1_bn1_running_mean = s1b1_bn1_result[1]
        self.s1b1_bn1_running_var = s1b1_bn1_result[2]
        var s1b1_relu1 = relu(s1b1_bn1)

        var s1b1_conv2 = conv2d(
            s1b1_relu1,
            self.s1b1_conv2_kernel,
            self.s1b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s1b1_bn2_result = batch_norm2d(
            s1b1_conv2,
            self.s1b1_bn2_gamma,
            self.s1b1_bn2_beta,
            self.s1b1_bn2_running_mean,
            self.s1b1_bn2_running_var,
            training=training,
        )
        var s1b1_bn2 = s1b1_bn2_result[0]
        self.s1b1_bn2_running_mean = s1b1_bn2_result[1]
        self.s1b1_bn2_running_var = s1b1_bn2_result[2]

        var s1b1_skip = add(s1b1_bn2, relu1)
        var s1b1_out = relu(s1b1_skip)

        # ========== Stage 1, Block 2 (identity shortcut) ==========
        var s1b2_conv1 = conv2d(
            s1b1_out,
            self.s1b2_conv1_kernel,
            self.s1b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s1b2_bn1_result = batch_norm2d(
            s1b2_conv1,
            self.s1b2_bn1_gamma,
            self.s1b2_bn1_beta,
            self.s1b2_bn1_running_mean,
            self.s1b2_bn1_running_var,
            training=training,
        )
        var s1b2_bn1 = s1b2_bn1_result[0]
        self.s1b2_bn1_running_mean = s1b2_bn1_result[1]
        self.s1b2_bn1_running_var = s1b2_bn1_result[2]
        var s1b2_relu1 = relu(s1b2_bn1)

        var s1b2_conv2 = conv2d(
            s1b2_relu1,
            self.s1b2_conv2_kernel,
            self.s1b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s1b2_bn2_result = batch_norm2d(
            s1b2_conv2,
            self.s1b2_bn2_gamma,
            self.s1b2_bn2_beta,
            self.s1b2_bn2_running_mean,
            self.s1b2_bn2_running_var,
            training=training,
        )
        var s1b2_bn2 = s1b2_bn2_result[0]
        self.s1b2_bn2_running_mean = s1b2_bn2_result[1]
        self.s1b2_bn2_running_var = s1b2_bn2_result[2]

        var s1b2_skip = add(s1b2_bn2, s1b1_out)
        var s1b2_out = relu(s1b2_skip)

        # ========== Stage 2, Block 1 (projection shortcut, stride=2) ==========
        var s2b1_conv1 = conv2d(
            s1b2_out,
            self.s2b1_conv1_kernel,
            self.s2b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s2b1_bn1_result = batch_norm2d(
            s2b1_conv1,
            self.s2b1_bn1_gamma,
            self.s2b1_bn1_beta,
            self.s2b1_bn1_running_mean,
            self.s2b1_bn1_running_var,
            training=training,
        )
        var s2b1_bn1 = s2b1_bn1_result[0]
        self.s2b1_bn1_running_mean = s2b1_bn1_result[1]
        self.s2b1_bn1_running_var = s2b1_bn1_result[2]
        var s2b1_relu1 = relu(s2b1_bn1)

        var s2b1_conv2 = conv2d(
            s2b1_relu1,
            self.s2b1_conv2_kernel,
            self.s2b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s2b1_bn2_result = batch_norm2d(
            s2b1_conv2,
            self.s2b1_bn2_gamma,
            self.s2b1_bn2_beta,
            self.s2b1_bn2_running_mean,
            self.s2b1_bn2_running_var,
            training=training,
        )
        var s2b1_bn2 = s2b1_bn2_result[0]
        self.s2b1_bn2_running_mean = s2b1_bn2_result[1]
        self.s2b1_bn2_running_var = s2b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s2b1_proj_conv = conv2d(
            s1b2_out,
            self.s2b1_proj_kernel,
            self.s2b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s2b1_proj_bn_result = batch_norm2d(
            s2b1_proj_conv,
            self.s2b1_proj_bn_gamma,
            self.s2b1_proj_bn_beta,
            self.s2b1_proj_bn_running_mean,
            self.s2b1_proj_bn_running_var,
            training=training,
        )
        var s2b1_proj_bn = s2b1_proj_bn_result[0]
        self.s2b1_proj_bn_running_mean = s2b1_proj_bn_result[1]
        self.s2b1_proj_bn_running_var = s2b1_proj_bn_result[2]

        var s2b1_skip = add(s2b1_bn2, s2b1_proj_bn)
        var s2b1_out = relu(s2b1_skip)

        # ========== Stage 2, Block 2 (identity shortcut) ==========
        var s2b2_conv1 = conv2d(
            s2b1_out,
            self.s2b2_conv1_kernel,
            self.s2b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s2b2_bn1_result = batch_norm2d(
            s2b2_conv1,
            self.s2b2_bn1_gamma,
            self.s2b2_bn1_beta,
            self.s2b2_bn1_running_mean,
            self.s2b2_bn1_running_var,
            training=training,
        )
        var s2b2_bn1 = s2b2_bn1_result[0]
        self.s2b2_bn1_running_mean = s2b2_bn1_result[1]
        self.s2b2_bn1_running_var = s2b2_bn1_result[2]
        var s2b2_relu1 = relu(s2b2_bn1)

        var s2b2_conv2 = conv2d(
            s2b2_relu1,
            self.s2b2_conv2_kernel,
            self.s2b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s2b2_bn2_result = batch_norm2d(
            s2b2_conv2,
            self.s2b2_bn2_gamma,
            self.s2b2_bn2_beta,
            self.s2b2_bn2_running_mean,
            self.s2b2_bn2_running_var,
            training=training,
        )
        var s2b2_bn2 = s2b2_bn2_result[0]
        self.s2b2_bn2_running_mean = s2b2_bn2_result[1]
        self.s2b2_bn2_running_var = s2b2_bn2_result[2]

        var s2b2_skip = add(s2b2_bn2, s2b1_out)
        var s2b2_out = relu(s2b2_skip)

        # ========== Stage 3, Block 1 (projection shortcut, stride=2) ==========
        var s3b1_conv1 = conv2d(
            s2b2_out,
            self.s3b1_conv1_kernel,
            self.s3b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s3b1_bn1_result = batch_norm2d(
            s3b1_conv1,
            self.s3b1_bn1_gamma,
            self.s3b1_bn1_beta,
            self.s3b1_bn1_running_mean,
            self.s3b1_bn1_running_var,
            training=training,
        )
        var s3b1_bn1 = s3b1_bn1_result[0]
        self.s3b1_bn1_running_mean = s3b1_bn1_result[1]
        self.s3b1_bn1_running_var = s3b1_bn1_result[2]
        var s3b1_relu1 = relu(s3b1_bn1)

        var s3b1_conv2 = conv2d(
            s3b1_relu1,
            self.s3b1_conv2_kernel,
            self.s3b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s3b1_bn2_result = batch_norm2d(
            s3b1_conv2,
            self.s3b1_bn2_gamma,
            self.s3b1_bn2_beta,
            self.s3b1_bn2_running_mean,
            self.s3b1_bn2_running_var,
            training=training,
        )
        var s3b1_bn2 = s3b1_bn2_result[0]
        self.s3b1_bn2_running_mean = s3b1_bn2_result[1]
        self.s3b1_bn2_running_var = s3b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s3b1_proj_conv = conv2d(
            s2b2_out,
            self.s3b1_proj_kernel,
            self.s3b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s3b1_proj_bn_result = batch_norm2d(
            s3b1_proj_conv,
            self.s3b1_proj_bn_gamma,
            self.s3b1_proj_bn_beta,
            self.s3b1_proj_bn_running_mean,
            self.s3b1_proj_bn_running_var,
            training=training,
        )
        var s3b1_proj_bn = s3b1_proj_bn_result[0]
        self.s3b1_proj_bn_running_mean = s3b1_proj_bn_result[1]
        self.s3b1_proj_bn_running_var = s3b1_proj_bn_result[2]

        var s3b1_skip = add(s3b1_bn2, s3b1_proj_bn)
        var s3b1_out = relu(s3b1_skip)

        # ========== Stage 3, Block 2 (identity shortcut) ==========
        var s3b2_conv1 = conv2d(
            s3b1_out,
            self.s3b2_conv1_kernel,
            self.s3b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s3b2_bn1_result = batch_norm2d(
            s3b2_conv1,
            self.s3b2_bn1_gamma,
            self.s3b2_bn1_beta,
            self.s3b2_bn1_running_mean,
            self.s3b2_bn1_running_var,
            training=training,
        )
        var s3b2_bn1 = s3b2_bn1_result[0]
        self.s3b2_bn1_running_mean = s3b2_bn1_result[1]
        self.s3b2_bn1_running_var = s3b2_bn1_result[2]
        var s3b2_relu1 = relu(s3b2_bn1)

        var s3b2_conv2 = conv2d(
            s3b2_relu1,
            self.s3b2_conv2_kernel,
            self.s3b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s3b2_bn2_result = batch_norm2d(
            s3b2_conv2,
            self.s3b2_bn2_gamma,
            self.s3b2_bn2_beta,
            self.s3b2_bn2_running_mean,
            self.s3b2_bn2_running_var,
            training=training,
        )
        var s3b2_bn2 = s3b2_bn2_result[0]
        self.s3b2_bn2_running_mean = s3b2_bn2_result[1]
        self.s3b2_bn2_running_var = s3b2_bn2_result[2]

        var s3b2_skip = add(s3b2_bn2, s3b1_out)
        var s3b2_out = relu(s3b2_skip)

        # ========== Stage 4, Block 1 (projection shortcut, stride=2) ==========
        var s4b1_conv1 = conv2d(
            s3b2_out,
            self.s4b1_conv1_kernel,
            self.s4b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s4b1_bn1_result = batch_norm2d(
            s4b1_conv1,
            self.s4b1_bn1_gamma,
            self.s4b1_bn1_beta,
            self.s4b1_bn1_running_mean,
            self.s4b1_bn1_running_var,
            training=training,
        )
        var s4b1_bn1 = s4b1_bn1_result[0]
        self.s4b1_bn1_running_mean = s4b1_bn1_result[1]
        self.s4b1_bn1_running_var = s4b1_bn1_result[2]
        var s4b1_relu1 = relu(s4b1_bn1)

        var s4b1_conv2 = conv2d(
            s4b1_relu1,
            self.s4b1_conv2_kernel,
            self.s4b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s4b1_bn2_result = batch_norm2d(
            s4b1_conv2,
            self.s4b1_bn2_gamma,
            self.s4b1_bn2_beta,
            self.s4b1_bn2_running_mean,
            self.s4b1_bn2_running_var,
            training=training,
        )
        var s4b1_bn2 = s4b1_bn2_result[0]
        self.s4b1_bn2_running_mean = s4b1_bn2_result[1]
        self.s4b1_bn2_running_var = s4b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s4b1_proj_conv = conv2d(
            s3b2_out,
            self.s4b1_proj_kernel,
            self.s4b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s4b1_proj_bn_result = batch_norm2d(
            s4b1_proj_conv,
            self.s4b1_proj_bn_gamma,
            self.s4b1_proj_bn_beta,
            self.s4b1_proj_bn_running_mean,
            self.s4b1_proj_bn_running_var,
            training=training,
        )
        var s4b1_proj_bn = s4b1_proj_bn_result[0]
        self.s4b1_proj_bn_running_mean = s4b1_proj_bn_result[1]
        self.s4b1_proj_bn_running_var = s4b1_proj_bn_result[2]

        var s4b1_skip = add(s4b1_bn2, s4b1_proj_bn)
        var s4b1_out = relu(s4b1_skip)

        # ========== Stage 4, Block 2 (identity shortcut) ==========
        var s4b2_conv1 = conv2d(
            s4b1_out,
            self.s4b2_conv1_kernel,
            self.s4b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s4b2_bn1_result = batch_norm2d(
            s4b2_conv1,
            self.s4b2_bn1_gamma,
            self.s4b2_bn1_beta,
            self.s4b2_bn1_running_mean,
            self.s4b2_bn1_running_var,
            training=training,
        )
        var s4b2_bn1 = s4b2_bn1_result[0]
        self.s4b2_bn1_running_mean = s4b2_bn1_result[1]
        self.s4b2_bn1_running_var = s4b2_bn1_result[2]
        var s4b2_relu1 = relu(s4b2_bn1)

        var s4b2_conv2 = conv2d(
            s4b2_relu1,
            self.s4b2_conv2_kernel,
            self.s4b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s4b2_bn2_result = batch_norm2d(
            s4b2_conv2,
            self.s4b2_bn2_gamma,
            self.s4b2_bn2_beta,
            self.s4b2_bn2_running_mean,
            self.s4b2_bn2_running_var,
            training=training,
        )
        var s4b2_bn2 = s4b2_bn2_result[0]
        self.s4b2_bn2_running_mean = s4b2_bn2_result[1]
        self.s4b2_bn2_running_var = s4b2_bn2_result[2]

        var s4b2_skip = add(s4b2_bn2, s4b1_out)
        var s4b2_out = relu(s4b2_skip)

        # ========== Global Average Pooling: (batch, 512, 4, 4) → (batch, 512, 1, 1) ==========
        var gap = avgpool2d(s4b2_out, kernel_size=4, stride=1, padding=0)

        # ========== Flatten: (batch, 512, 1, 1) → (batch, 512) ==========
        var gap_shape = gap.shape()
        var batch_size = gap_shape[0]
        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(512)
        var flattened = gap.reshape(flatten_shape)

        # ========== FC layer: (batch, 512) → (batch, num_classes) ==========
        var logits = linear(flattened, self.fc_weights, self.fc_bias)

        return logits

    def forward_with_cache(
        mut self, input: AnyTensor, training: Bool = True
    ) raises -> ResNet18ForwardCache:
        """Same computation as forward(), but returns all intermediate activations.

        BN running_mean/var snapshots are captured BEFORE batch_norm2d mutates
        the model's fields. Safe under AnyTensor refcounting (any_tensor.mojo:96):
        `var snap = self.field` bumps refcount; reassigning `self.field` preserves
        the old buffer via `snap`'s reference.
        """
        # ========== Initial conv + BN + ReLU (snapshot BEFORE BN) ====
        var conv1 = conv2d(
            input, self.conv1_kernel, self.conv1_bias, stride=1, padding=1
        )
        var bn1_rm_snap = self.bn1_running_mean
        var bn1_rv_snap = self.bn1_running_var
        var bn1_result = batch_norm2d(
            conv1,
            self.bn1_gamma,
            self.bn1_beta,
            self.bn1_running_mean,
            self.bn1_running_var,
            training=training,
        )
        var bn1 = bn1_result[0]
        self.bn1_running_mean = bn1_result[1]
        self.bn1_running_var = bn1_result[2]
        var relu1 = relu(bn1)

        # ========== Stage 1, Block 1 (identity shortcut) ==========
        var s1b1_conv1 = conv2d(
            relu1,
            self.s1b1_conv1_kernel,
            self.s1b1_conv1_bias,
            stride=1,
            padding=1,
        )
        var s1b1_bn1_rm_snap = self.s1b1_bn1_running_mean
        var s1b1_bn1_rv_snap = self.s1b1_bn1_running_var
        var s1b1_bn1_result = batch_norm2d(
            s1b1_conv1,
            self.s1b1_bn1_gamma,
            self.s1b1_bn1_beta,
            self.s1b1_bn1_running_mean,
            self.s1b1_bn1_running_var,
            training=training,
        )
        var s1b1_bn1 = s1b1_bn1_result[0]
        self.s1b1_bn1_running_mean = s1b1_bn1_result[1]
        self.s1b1_bn1_running_var = s1b1_bn1_result[2]
        var s1b1_relu1 = relu(s1b1_bn1)

        var s1b1_conv2 = conv2d(
            s1b1_relu1,
            self.s1b1_conv2_kernel,
            self.s1b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s1b1_bn2_rm_snap = self.s1b1_bn2_running_mean
        var s1b1_bn2_rv_snap = self.s1b1_bn2_running_var
        var s1b1_bn2_result = batch_norm2d(
            s1b1_conv2,
            self.s1b1_bn2_gamma,
            self.s1b1_bn2_beta,
            self.s1b1_bn2_running_mean,
            self.s1b1_bn2_running_var,
            training=training,
        )
        var s1b1_bn2 = s1b1_bn2_result[0]
        self.s1b1_bn2_running_mean = s1b1_bn2_result[1]
        self.s1b1_bn2_running_var = s1b1_bn2_result[2]

        var s1b1_skip = add(s1b1_bn2, relu1)
        var s1b1_out = relu(s1b1_skip)
        var s1b1_cache = IdentityCache(
            block_input=relu1,
            conv1_pre_bn=s1b1_conv1,
            bn1_pre_relu=s1b1_bn1,
            relu1_out=s1b1_relu1,
            conv2_pre_bn=s1b1_conv2,
            bn2_pre_relu=s1b1_bn2,
            skip_sum=s1b1_skip,
            block_out=s1b1_out,
            bn1_running_mean_snapshot=s1b1_bn1_rm_snap,
            bn1_running_var_snapshot=s1b1_bn1_rv_snap,
            bn2_running_mean_snapshot=s1b1_bn2_rm_snap,
            bn2_running_var_snapshot=s1b1_bn2_rv_snap,
        )

        # ========== Stage 1, Block 2 (identity shortcut) ==========
        var s1b2_conv1 = conv2d(
            s1b1_out,
            self.s1b2_conv1_kernel,
            self.s1b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s1b2_bn1_rm_snap = self.s1b2_bn1_running_mean
        var s1b2_bn1_rv_snap = self.s1b2_bn1_running_var
        var s1b2_bn1_result = batch_norm2d(
            s1b2_conv1,
            self.s1b2_bn1_gamma,
            self.s1b2_bn1_beta,
            self.s1b2_bn1_running_mean,
            self.s1b2_bn1_running_var,
            training=training,
        )
        var s1b2_bn1 = s1b2_bn1_result[0]
        self.s1b2_bn1_running_mean = s1b2_bn1_result[1]
        self.s1b2_bn1_running_var = s1b2_bn1_result[2]
        var s1b2_relu1 = relu(s1b2_bn1)

        var s1b2_conv2 = conv2d(
            s1b2_relu1,
            self.s1b2_conv2_kernel,
            self.s1b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s1b2_bn2_rm_snap = self.s1b2_bn2_running_mean
        var s1b2_bn2_rv_snap = self.s1b2_bn2_running_var
        var s1b2_bn2_result = batch_norm2d(
            s1b2_conv2,
            self.s1b2_bn2_gamma,
            self.s1b2_bn2_beta,
            self.s1b2_bn2_running_mean,
            self.s1b2_bn2_running_var,
            training=training,
        )
        var s1b2_bn2 = s1b2_bn2_result[0]
        self.s1b2_bn2_running_mean = s1b2_bn2_result[1]
        self.s1b2_bn2_running_var = s1b2_bn2_result[2]

        var s1b2_skip = add(s1b2_bn2, s1b1_out)
        var s1b2_out = relu(s1b2_skip)
        var s1b2_cache = IdentityCache(
            block_input=s1b1_out,
            conv1_pre_bn=s1b2_conv1,
            bn1_pre_relu=s1b2_bn1,
            relu1_out=s1b2_relu1,
            conv2_pre_bn=s1b2_conv2,
            bn2_pre_relu=s1b2_bn2,
            skip_sum=s1b2_skip,
            block_out=s1b2_out,
            bn1_running_mean_snapshot=s1b2_bn1_rm_snap,
            bn1_running_var_snapshot=s1b2_bn1_rv_snap,
            bn2_running_mean_snapshot=s1b2_bn2_rm_snap,
            bn2_running_var_snapshot=s1b2_bn2_rv_snap,
        )

        # ========== Stage 2, Block 1 (projection shortcut, stride=2) ==========
        var s2b1_conv1 = conv2d(
            s1b2_out,
            self.s2b1_conv1_kernel,
            self.s2b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s2b1_bn1_rm_snap = self.s2b1_bn1_running_mean
        var s2b1_bn1_rv_snap = self.s2b1_bn1_running_var
        var s2b1_bn1_result = batch_norm2d(
            s2b1_conv1,
            self.s2b1_bn1_gamma,
            self.s2b1_bn1_beta,
            self.s2b1_bn1_running_mean,
            self.s2b1_bn1_running_var,
            training=training,
        )
        var s2b1_bn1 = s2b1_bn1_result[0]
        self.s2b1_bn1_running_mean = s2b1_bn1_result[1]
        self.s2b1_bn1_running_var = s2b1_bn1_result[2]
        var s2b1_relu1 = relu(s2b1_bn1)

        var s2b1_conv2 = conv2d(
            s2b1_relu1,
            self.s2b1_conv2_kernel,
            self.s2b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s2b1_bn2_rm_snap = self.s2b1_bn2_running_mean
        var s2b1_bn2_rv_snap = self.s2b1_bn2_running_var
        var s2b1_bn2_result = batch_norm2d(
            s2b1_conv2,
            self.s2b1_bn2_gamma,
            self.s2b1_bn2_beta,
            self.s2b1_bn2_running_mean,
            self.s2b1_bn2_running_var,
            training=training,
        )
        var s2b1_bn2 = s2b1_bn2_result[0]
        self.s2b1_bn2_running_mean = s2b1_bn2_result[1]
        self.s2b1_bn2_running_var = s2b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s2b1_proj_conv = conv2d(
            s1b2_out,
            self.s2b1_proj_kernel,
            self.s2b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s2b1_proj_bn_rm_snap = self.s2b1_proj_bn_running_mean
        var s2b1_proj_bn_rv_snap = self.s2b1_proj_bn_running_var
        var s2b1_proj_bn_result = batch_norm2d(
            s2b1_proj_conv,
            self.s2b1_proj_bn_gamma,
            self.s2b1_proj_bn_beta,
            self.s2b1_proj_bn_running_mean,
            self.s2b1_proj_bn_running_var,
            training=training,
        )
        var s2b1_proj_bn = s2b1_proj_bn_result[0]
        self.s2b1_proj_bn_running_mean = s2b1_proj_bn_result[1]
        self.s2b1_proj_bn_running_var = s2b1_proj_bn_result[2]

        var s2b1_skip = add(s2b1_bn2, s2b1_proj_bn)
        var s2b1_out = relu(s2b1_skip)
        var s2b1_cache = ProjectionCache(
            block_input=s1b2_out,
            conv1_pre_bn=s2b1_conv1,
            bn1_pre_relu=s2b1_bn1,
            relu1_out=s2b1_relu1,
            conv2_pre_bn=s2b1_conv2,
            bn2_pre_relu=s2b1_bn2,
            proj_conv_pre_bn=s2b1_proj_conv,
            proj_bn_out=s2b1_proj_bn,
            skip_sum=s2b1_skip,
            block_out=s2b1_out,
            bn1_running_mean_snapshot=s2b1_bn1_rm_snap,
            bn1_running_var_snapshot=s2b1_bn1_rv_snap,
            bn2_running_mean_snapshot=s2b1_bn2_rm_snap,
            bn2_running_var_snapshot=s2b1_bn2_rv_snap,
            proj_bn_running_mean_snapshot=s2b1_proj_bn_rm_snap,
            proj_bn_running_var_snapshot=s2b1_proj_bn_rv_snap,
        )

        # ========== Stage 2, Block 2 (identity shortcut) ==========
        var s2b2_conv1 = conv2d(
            s2b1_out,
            self.s2b2_conv1_kernel,
            self.s2b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s2b2_bn1_rm_snap = self.s2b2_bn1_running_mean
        var s2b2_bn1_rv_snap = self.s2b2_bn1_running_var
        var s2b2_bn1_result = batch_norm2d(
            s2b2_conv1,
            self.s2b2_bn1_gamma,
            self.s2b2_bn1_beta,
            self.s2b2_bn1_running_mean,
            self.s2b2_bn1_running_var,
            training=training,
        )
        var s2b2_bn1 = s2b2_bn1_result[0]
        self.s2b2_bn1_running_mean = s2b2_bn1_result[1]
        self.s2b2_bn1_running_var = s2b2_bn1_result[2]
        var s2b2_relu1 = relu(s2b2_bn1)

        var s2b2_conv2 = conv2d(
            s2b2_relu1,
            self.s2b2_conv2_kernel,
            self.s2b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s2b2_bn2_rm_snap = self.s2b2_bn2_running_mean
        var s2b2_bn2_rv_snap = self.s2b2_bn2_running_var
        var s2b2_bn2_result = batch_norm2d(
            s2b2_conv2,
            self.s2b2_bn2_gamma,
            self.s2b2_bn2_beta,
            self.s2b2_bn2_running_mean,
            self.s2b2_bn2_running_var,
            training=training,
        )
        var s2b2_bn2 = s2b2_bn2_result[0]
        self.s2b2_bn2_running_mean = s2b2_bn2_result[1]
        self.s2b2_bn2_running_var = s2b2_bn2_result[2]

        var s2b2_skip = add(s2b2_bn2, s2b1_out)
        var s2b2_out = relu(s2b2_skip)
        var s2b2_cache = IdentityCache(
            block_input=s2b1_out,
            conv1_pre_bn=s2b2_conv1,
            bn1_pre_relu=s2b2_bn1,
            relu1_out=s2b2_relu1,
            conv2_pre_bn=s2b2_conv2,
            bn2_pre_relu=s2b2_bn2,
            skip_sum=s2b2_skip,
            block_out=s2b2_out,
            bn1_running_mean_snapshot=s2b2_bn1_rm_snap,
            bn1_running_var_snapshot=s2b2_bn1_rv_snap,
            bn2_running_mean_snapshot=s2b2_bn2_rm_snap,
            bn2_running_var_snapshot=s2b2_bn2_rv_snap,
        )

        # ========== Stage 3, Block 1 (projection shortcut, stride=2) ==========
        var s3b1_conv1 = conv2d(
            s2b2_out,
            self.s3b1_conv1_kernel,
            self.s3b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s3b1_bn1_rm_snap = self.s3b1_bn1_running_mean
        var s3b1_bn1_rv_snap = self.s3b1_bn1_running_var
        var s3b1_bn1_result = batch_norm2d(
            s3b1_conv1,
            self.s3b1_bn1_gamma,
            self.s3b1_bn1_beta,
            self.s3b1_bn1_running_mean,
            self.s3b1_bn1_running_var,
            training=training,
        )
        var s3b1_bn1 = s3b1_bn1_result[0]
        self.s3b1_bn1_running_mean = s3b1_bn1_result[1]
        self.s3b1_bn1_running_var = s3b1_bn1_result[2]
        var s3b1_relu1 = relu(s3b1_bn1)

        var s3b1_conv2 = conv2d(
            s3b1_relu1,
            self.s3b1_conv2_kernel,
            self.s3b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s3b1_bn2_rm_snap = self.s3b1_bn2_running_mean
        var s3b1_bn2_rv_snap = self.s3b1_bn2_running_var
        var s3b1_bn2_result = batch_norm2d(
            s3b1_conv2,
            self.s3b1_bn2_gamma,
            self.s3b1_bn2_beta,
            self.s3b1_bn2_running_mean,
            self.s3b1_bn2_running_var,
            training=training,
        )
        var s3b1_bn2 = s3b1_bn2_result[0]
        self.s3b1_bn2_running_mean = s3b1_bn2_result[1]
        self.s3b1_bn2_running_var = s3b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s3b1_proj_conv = conv2d(
            s2b2_out,
            self.s3b1_proj_kernel,
            self.s3b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s3b1_proj_bn_rm_snap = self.s3b1_proj_bn_running_mean
        var s3b1_proj_bn_rv_snap = self.s3b1_proj_bn_running_var
        var s3b1_proj_bn_result = batch_norm2d(
            s3b1_proj_conv,
            self.s3b1_proj_bn_gamma,
            self.s3b1_proj_bn_beta,
            self.s3b1_proj_bn_running_mean,
            self.s3b1_proj_bn_running_var,
            training=training,
        )
        var s3b1_proj_bn = s3b1_proj_bn_result[0]
        self.s3b1_proj_bn_running_mean = s3b1_proj_bn_result[1]
        self.s3b1_proj_bn_running_var = s3b1_proj_bn_result[2]

        var s3b1_skip = add(s3b1_bn2, s3b1_proj_bn)
        var s3b1_out = relu(s3b1_skip)
        var s3b1_cache = ProjectionCache(
            block_input=s2b2_out,
            conv1_pre_bn=s3b1_conv1,
            bn1_pre_relu=s3b1_bn1,
            relu1_out=s3b1_relu1,
            conv2_pre_bn=s3b1_conv2,
            bn2_pre_relu=s3b1_bn2,
            proj_conv_pre_bn=s3b1_proj_conv,
            proj_bn_out=s3b1_proj_bn,
            skip_sum=s3b1_skip,
            block_out=s3b1_out,
            bn1_running_mean_snapshot=s3b1_bn1_rm_snap,
            bn1_running_var_snapshot=s3b1_bn1_rv_snap,
            bn2_running_mean_snapshot=s3b1_bn2_rm_snap,
            bn2_running_var_snapshot=s3b1_bn2_rv_snap,
            proj_bn_running_mean_snapshot=s3b1_proj_bn_rm_snap,
            proj_bn_running_var_snapshot=s3b1_proj_bn_rv_snap,
        )

        # ========== Stage 3, Block 2 (identity shortcut) ==========
        var s3b2_conv1 = conv2d(
            s3b1_out,
            self.s3b2_conv1_kernel,
            self.s3b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s3b2_bn1_rm_snap = self.s3b2_bn1_running_mean
        var s3b2_bn1_rv_snap = self.s3b2_bn1_running_var
        var s3b2_bn1_result = batch_norm2d(
            s3b2_conv1,
            self.s3b2_bn1_gamma,
            self.s3b2_bn1_beta,
            self.s3b2_bn1_running_mean,
            self.s3b2_bn1_running_var,
            training=training,
        )
        var s3b2_bn1 = s3b2_bn1_result[0]
        self.s3b2_bn1_running_mean = s3b2_bn1_result[1]
        self.s3b2_bn1_running_var = s3b2_bn1_result[2]
        var s3b2_relu1 = relu(s3b2_bn1)

        var s3b2_conv2 = conv2d(
            s3b2_relu1,
            self.s3b2_conv2_kernel,
            self.s3b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s3b2_bn2_rm_snap = self.s3b2_bn2_running_mean
        var s3b2_bn2_rv_snap = self.s3b2_bn2_running_var
        var s3b2_bn2_result = batch_norm2d(
            s3b2_conv2,
            self.s3b2_bn2_gamma,
            self.s3b2_bn2_beta,
            self.s3b2_bn2_running_mean,
            self.s3b2_bn2_running_var,
            training=training,
        )
        var s3b2_bn2 = s3b2_bn2_result[0]
        self.s3b2_bn2_running_mean = s3b2_bn2_result[1]
        self.s3b2_bn2_running_var = s3b2_bn2_result[2]

        var s3b2_skip = add(s3b2_bn2, s3b1_out)
        var s3b2_out = relu(s3b2_skip)
        var s3b2_cache = IdentityCache(
            block_input=s3b1_out,
            conv1_pre_bn=s3b2_conv1,
            bn1_pre_relu=s3b2_bn1,
            relu1_out=s3b2_relu1,
            conv2_pre_bn=s3b2_conv2,
            bn2_pre_relu=s3b2_bn2,
            skip_sum=s3b2_skip,
            block_out=s3b2_out,
            bn1_running_mean_snapshot=s3b2_bn1_rm_snap,
            bn1_running_var_snapshot=s3b2_bn1_rv_snap,
            bn2_running_mean_snapshot=s3b2_bn2_rm_snap,
            bn2_running_var_snapshot=s3b2_bn2_rv_snap,
        )

        # ========== Stage 4, Block 1 (projection shortcut, stride=2) ==========
        var s4b1_conv1 = conv2d(
            s3b2_out,
            self.s4b1_conv1_kernel,
            self.s4b1_conv1_bias,
            stride=2,
            padding=1,
        )
        var s4b1_bn1_rm_snap = self.s4b1_bn1_running_mean
        var s4b1_bn1_rv_snap = self.s4b1_bn1_running_var
        var s4b1_bn1_result = batch_norm2d(
            s4b1_conv1,
            self.s4b1_bn1_gamma,
            self.s4b1_bn1_beta,
            self.s4b1_bn1_running_mean,
            self.s4b1_bn1_running_var,
            training=training,
        )
        var s4b1_bn1 = s4b1_bn1_result[0]
        self.s4b1_bn1_running_mean = s4b1_bn1_result[1]
        self.s4b1_bn1_running_var = s4b1_bn1_result[2]
        var s4b1_relu1 = relu(s4b1_bn1)

        var s4b1_conv2 = conv2d(
            s4b1_relu1,
            self.s4b1_conv2_kernel,
            self.s4b1_conv2_bias,
            stride=1,
            padding=1,
        )
        var s4b1_bn2_rm_snap = self.s4b1_bn2_running_mean
        var s4b1_bn2_rv_snap = self.s4b1_bn2_running_var
        var s4b1_bn2_result = batch_norm2d(
            s4b1_conv2,
            self.s4b1_bn2_gamma,
            self.s4b1_bn2_beta,
            self.s4b1_bn2_running_mean,
            self.s4b1_bn2_running_var,
            training=training,
        )
        var s4b1_bn2 = s4b1_bn2_result[0]
        self.s4b1_bn2_running_mean = s4b1_bn2_result[1]
        self.s4b1_bn2_running_var = s4b1_bn2_result[2]

        # Projection shortcut: 1×1 conv, stride=2
        var s4b1_proj_conv = conv2d(
            s3b2_out,
            self.s4b1_proj_kernel,
            self.s4b1_proj_bias,
            stride=2,
            padding=0,
        )
        var s4b1_proj_bn_rm_snap = self.s4b1_proj_bn_running_mean
        var s4b1_proj_bn_rv_snap = self.s4b1_proj_bn_running_var
        var s4b1_proj_bn_result = batch_norm2d(
            s4b1_proj_conv,
            self.s4b1_proj_bn_gamma,
            self.s4b1_proj_bn_beta,
            self.s4b1_proj_bn_running_mean,
            self.s4b1_proj_bn_running_var,
            training=training,
        )
        var s4b1_proj_bn = s4b1_proj_bn_result[0]
        self.s4b1_proj_bn_running_mean = s4b1_proj_bn_result[1]
        self.s4b1_proj_bn_running_var = s4b1_proj_bn_result[2]

        var s4b1_skip = add(s4b1_bn2, s4b1_proj_bn)
        var s4b1_out = relu(s4b1_skip)
        var s4b1_cache = ProjectionCache(
            block_input=s3b2_out,
            conv1_pre_bn=s4b1_conv1,
            bn1_pre_relu=s4b1_bn1,
            relu1_out=s4b1_relu1,
            conv2_pre_bn=s4b1_conv2,
            bn2_pre_relu=s4b1_bn2,
            proj_conv_pre_bn=s4b1_proj_conv,
            proj_bn_out=s4b1_proj_bn,
            skip_sum=s4b1_skip,
            block_out=s4b1_out,
            bn1_running_mean_snapshot=s4b1_bn1_rm_snap,
            bn1_running_var_snapshot=s4b1_bn1_rv_snap,
            bn2_running_mean_snapshot=s4b1_bn2_rm_snap,
            bn2_running_var_snapshot=s4b1_bn2_rv_snap,
            proj_bn_running_mean_snapshot=s4b1_proj_bn_rm_snap,
            proj_bn_running_var_snapshot=s4b1_proj_bn_rv_snap,
        )

        # ========== Stage 4, Block 2 (identity shortcut) ==========
        var s4b2_conv1 = conv2d(
            s4b1_out,
            self.s4b2_conv1_kernel,
            self.s4b2_conv1_bias,
            stride=1,
            padding=1,
        )
        var s4b2_bn1_rm_snap = self.s4b2_bn1_running_mean
        var s4b2_bn1_rv_snap = self.s4b2_bn1_running_var
        var s4b2_bn1_result = batch_norm2d(
            s4b2_conv1,
            self.s4b2_bn1_gamma,
            self.s4b2_bn1_beta,
            self.s4b2_bn1_running_mean,
            self.s4b2_bn1_running_var,
            training=training,
        )
        var s4b2_bn1 = s4b2_bn1_result[0]
        self.s4b2_bn1_running_mean = s4b2_bn1_result[1]
        self.s4b2_bn1_running_var = s4b2_bn1_result[2]
        var s4b2_relu1 = relu(s4b2_bn1)

        var s4b2_conv2 = conv2d(
            s4b2_relu1,
            self.s4b2_conv2_kernel,
            self.s4b2_conv2_bias,
            stride=1,
            padding=1,
        )
        var s4b2_bn2_rm_snap = self.s4b2_bn2_running_mean
        var s4b2_bn2_rv_snap = self.s4b2_bn2_running_var
        var s4b2_bn2_result = batch_norm2d(
            s4b2_conv2,
            self.s4b2_bn2_gamma,
            self.s4b2_bn2_beta,
            self.s4b2_bn2_running_mean,
            self.s4b2_bn2_running_var,
            training=training,
        )
        var s4b2_bn2 = s4b2_bn2_result[0]
        self.s4b2_bn2_running_mean = s4b2_bn2_result[1]
        self.s4b2_bn2_running_var = s4b2_bn2_result[2]

        var s4b2_skip = add(s4b2_bn2, s4b1_out)
        var s4b2_out = relu(s4b2_skip)
        var s4b2_cache = IdentityCache(
            block_input=s4b1_out,
            conv1_pre_bn=s4b2_conv1,
            bn1_pre_relu=s4b2_bn1,
            relu1_out=s4b2_relu1,
            conv2_pre_bn=s4b2_conv2,
            bn2_pre_relu=s4b2_bn2,
            skip_sum=s4b2_skip,
            block_out=s4b2_out,
            bn1_running_mean_snapshot=s4b2_bn1_rm_snap,
            bn1_running_var_snapshot=s4b2_bn1_rv_snap,
            bn2_running_mean_snapshot=s4b2_bn2_rm_snap,
            bn2_running_var_snapshot=s4b2_bn2_rv_snap,
        )

        # ========== Global Average Pooling: (batch, 512, 4, 4) → (batch, 512, 1, 1) ==========
        var gap = avgpool2d(s4b2_out, kernel_size=4, stride=1, padding=0)

        # ========== Flatten: (batch, 512, 1, 1) → (batch, 512) ==========
        var gap_shape = gap.shape()
        var batch_size = gap_shape[0]
        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(512)
        var flattened = gap.reshape(flatten_shape)

        # ========== FC layer: (batch, 512) → (batch, num_classes) ==========
        var logits = linear(flattened, self.fc_weights, self.fc_bias)

        return ResNet18ForwardCache(
            logits=logits^,
            conv1_pre_bn=conv1^,
            bn1_pre_relu=bn1^,
            relu1_out=relu1^,
            initial_bn_running_mean_snapshot=bn1_rm_snap^,
            initial_bn_running_var_snapshot=bn1_rv_snap^,
            s1b1_cache=s1b1_cache^,
            s1b2_cache=s1b2_cache^,
            s2b1_cache=s2b1_cache^,
            s2b2_cache=s2b2_cache^,
            s3b1_cache=s3b1_cache^,
            s3b2_cache=s3b2_cache^,
            s4b1_cache=s4b1_cache^,
            s4b2_cache=s4b2_cache^,
            gap=gap^,
            flattened=flattened^,
        )

    def predict(mut self, input: AnyTensor) raises -> Int:
        """Predict class for a single input.

        Args:
            input: Input tensor of shape (1, 3, 32, 32).

        Returns:
            Predicted class index (0 to num_classes-1).
        """
        var logits = self.forward(input, training=False)

        # Find argmax
        var logits_shape = logits.shape()
        var max_idx = 0
        var max_val = logits[0]

        for i in range(1, logits_shape[1]):
            if logits[i] > max_val:
                max_val = logits[i]
                max_idx = i

        return max_idx

    def save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files.

        Note:
            Running stats (running_mean, running_var) are not saved as they
            are recomputed during training from scratch.
        """
        # Initial conv + BN
        save_tensor(
            self.conv1_kernel,
            weights_dir + "/conv1_kernel.weights",
            "conv1_kernel",
        )
        save_tensor(
            self.conv1_bias, weights_dir + "/conv1_bias.weights", "conv1_bias"
        )
        save_tensor(
            self.bn1_gamma, weights_dir + "/bn1_gamma.weights", "bn1_gamma"
        )
        save_tensor(
            self.bn1_beta, weights_dir + "/bn1_beta.weights", "bn1_beta"
        )

        # Stage 1
        save_tensor(
            self.s1b1_conv1_kernel,
            weights_dir + "/s1b1_conv1_kernel.weights",
            "s1b1_conv1_kernel",
        )
        save_tensor(
            self.s1b1_conv1_bias,
            weights_dir + "/s1b1_conv1_bias.weights",
            "s1b1_conv1_bias",
        )
        save_tensor(
            self.s1b1_bn1_gamma,
            weights_dir + "/s1b1_bn1_gamma.weights",
            "s1b1_bn1_gamma",
        )
        save_tensor(
            self.s1b1_bn1_beta,
            weights_dir + "/s1b1_bn1_beta.weights",
            "s1b1_bn1_beta",
        )
        save_tensor(
            self.s1b1_conv2_kernel,
            weights_dir + "/s1b1_conv2_kernel.weights",
            "s1b1_conv2_kernel",
        )
        save_tensor(
            self.s1b1_conv2_bias,
            weights_dir + "/s1b1_conv2_bias.weights",
            "s1b1_conv2_bias",
        )
        save_tensor(
            self.s1b1_bn2_gamma,
            weights_dir + "/s1b1_bn2_gamma.weights",
            "s1b1_bn2_gamma",
        )
        save_tensor(
            self.s1b1_bn2_beta,
            weights_dir + "/s1b1_bn2_beta.weights",
            "s1b1_bn2_beta",
        )

        save_tensor(
            self.s1b2_conv1_kernel,
            weights_dir + "/s1b2_conv1_kernel.weights",
            "s1b2_conv1_kernel",
        )
        save_tensor(
            self.s1b2_conv1_bias,
            weights_dir + "/s1b2_conv1_bias.weights",
            "s1b2_conv1_bias",
        )
        save_tensor(
            self.s1b2_bn1_gamma,
            weights_dir + "/s1b2_bn1_gamma.weights",
            "s1b2_bn1_gamma",
        )
        save_tensor(
            self.s1b2_bn1_beta,
            weights_dir + "/s1b2_bn1_beta.weights",
            "s1b2_bn1_beta",
        )
        save_tensor(
            self.s1b2_conv2_kernel,
            weights_dir + "/s1b2_conv2_kernel.weights",
            "s1b2_conv2_kernel",
        )
        save_tensor(
            self.s1b2_conv2_bias,
            weights_dir + "/s1b2_conv2_bias.weights",
            "s1b2_conv2_bias",
        )
        save_tensor(
            self.s1b2_bn2_gamma,
            weights_dir + "/s1b2_bn2_gamma.weights",
            "s1b2_bn2_gamma",
        )
        save_tensor(
            self.s1b2_bn2_beta,
            weights_dir + "/s1b2_bn2_beta.weights",
            "s1b2_bn2_beta",
        )

        # Stage 2
        save_tensor(
            self.s2b1_conv1_kernel,
            weights_dir + "/s2b1_conv1_kernel.weights",
            "s2b1_conv1_kernel",
        )
        save_tensor(
            self.s2b1_conv1_bias,
            weights_dir + "/s2b1_conv1_bias.weights",
            "s2b1_conv1_bias",
        )
        save_tensor(
            self.s2b1_bn1_gamma,
            weights_dir + "/s2b1_bn1_gamma.weights",
            "s2b1_bn1_gamma",
        )
        save_tensor(
            self.s2b1_bn1_beta,
            weights_dir + "/s2b1_bn1_beta.weights",
            "s2b1_bn1_beta",
        )
        save_tensor(
            self.s2b1_conv2_kernel,
            weights_dir + "/s2b1_conv2_kernel.weights",
            "s2b1_conv2_kernel",
        )
        save_tensor(
            self.s2b1_conv2_bias,
            weights_dir + "/s2b1_conv2_bias.weights",
            "s2b1_conv2_bias",
        )
        save_tensor(
            self.s2b1_bn2_gamma,
            weights_dir + "/s2b1_bn2_gamma.weights",
            "s2b1_bn2_gamma",
        )
        save_tensor(
            self.s2b1_bn2_beta,
            weights_dir + "/s2b1_bn2_beta.weights",
            "s2b1_bn2_beta",
        )
        save_tensor(
            self.s2b1_proj_kernel,
            weights_dir + "/s2b1_proj_kernel.weights",
            "s2b1_proj_kernel",
        )
        save_tensor(
            self.s2b1_proj_bias,
            weights_dir + "/s2b1_proj_bias.weights",
            "s2b1_proj_bias",
        )
        save_tensor(
            self.s2b1_proj_bn_gamma,
            weights_dir + "/s2b1_proj_bn_gamma.weights",
            "s2b1_proj_bn_gamma",
        )
        save_tensor(
            self.s2b1_proj_bn_beta,
            weights_dir + "/s2b1_proj_bn_beta.weights",
            "s2b1_proj_bn_beta",
        )

        save_tensor(
            self.s2b2_conv1_kernel,
            weights_dir + "/s2b2_conv1_kernel.weights",
            "s2b2_conv1_kernel",
        )
        save_tensor(
            self.s2b2_conv1_bias,
            weights_dir + "/s2b2_conv1_bias.weights",
            "s2b2_conv1_bias",
        )
        save_tensor(
            self.s2b2_bn1_gamma,
            weights_dir + "/s2b2_bn1_gamma.weights",
            "s2b2_bn1_gamma",
        )
        save_tensor(
            self.s2b2_bn1_beta,
            weights_dir + "/s2b2_bn1_beta.weights",
            "s2b2_bn1_beta",
        )
        save_tensor(
            self.s2b2_conv2_kernel,
            weights_dir + "/s2b2_conv2_kernel.weights",
            "s2b2_conv2_kernel",
        )
        save_tensor(
            self.s2b2_conv2_bias,
            weights_dir + "/s2b2_conv2_bias.weights",
            "s2b2_conv2_bias",
        )
        save_tensor(
            self.s2b2_bn2_gamma,
            weights_dir + "/s2b2_bn2_gamma.weights",
            "s2b2_bn2_gamma",
        )
        save_tensor(
            self.s2b2_bn2_beta,
            weights_dir + "/s2b2_bn2_beta.weights",
            "s2b2_bn2_beta",
        )

        # Stage 3
        save_tensor(
            self.s3b1_conv1_kernel,
            weights_dir + "/s3b1_conv1_kernel.weights",
            "s3b1_conv1_kernel",
        )
        save_tensor(
            self.s3b1_conv1_bias,
            weights_dir + "/s3b1_conv1_bias.weights",
            "s3b1_conv1_bias",
        )
        save_tensor(
            self.s3b1_bn1_gamma,
            weights_dir + "/s3b1_bn1_gamma.weights",
            "s3b1_bn1_gamma",
        )
        save_tensor(
            self.s3b1_bn1_beta,
            weights_dir + "/s3b1_bn1_beta.weights",
            "s3b1_bn1_beta",
        )
        save_tensor(
            self.s3b1_conv2_kernel,
            weights_dir + "/s3b1_conv2_kernel.weights",
            "s3b1_conv2_kernel",
        )
        save_tensor(
            self.s3b1_conv2_bias,
            weights_dir + "/s3b1_conv2_bias.weights",
            "s3b1_conv2_bias",
        )
        save_tensor(
            self.s3b1_bn2_gamma,
            weights_dir + "/s3b1_bn2_gamma.weights",
            "s3b1_bn2_gamma",
        )
        save_tensor(
            self.s3b1_bn2_beta,
            weights_dir + "/s3b1_bn2_beta.weights",
            "s3b1_bn2_beta",
        )
        save_tensor(
            self.s3b1_proj_kernel,
            weights_dir + "/s3b1_proj_kernel.weights",
            "s3b1_proj_kernel",
        )
        save_tensor(
            self.s3b1_proj_bias,
            weights_dir + "/s3b1_proj_bias.weights",
            "s3b1_proj_bias",
        )
        save_tensor(
            self.s3b1_proj_bn_gamma,
            weights_dir + "/s3b1_proj_bn_gamma.weights",
            "s3b1_proj_bn_gamma",
        )
        save_tensor(
            self.s3b1_proj_bn_beta,
            weights_dir + "/s3b1_proj_bn_beta.weights",
            "s3b1_proj_bn_beta",
        )

        save_tensor(
            self.s3b2_conv1_kernel,
            weights_dir + "/s3b2_conv1_kernel.weights",
            "s3b2_conv1_kernel",
        )
        save_tensor(
            self.s3b2_conv1_bias,
            weights_dir + "/s3b2_conv1_bias.weights",
            "s3b2_conv1_bias",
        )
        save_tensor(
            self.s3b2_bn1_gamma,
            weights_dir + "/s3b2_bn1_gamma.weights",
            "s3b2_bn1_gamma",
        )
        save_tensor(
            self.s3b2_bn1_beta,
            weights_dir + "/s3b2_bn1_beta.weights",
            "s3b2_bn1_beta",
        )
        save_tensor(
            self.s3b2_conv2_kernel,
            weights_dir + "/s3b2_conv2_kernel.weights",
            "s3b2_conv2_kernel",
        )
        save_tensor(
            self.s3b2_conv2_bias,
            weights_dir + "/s3b2_conv2_bias.weights",
            "s3b2_conv2_bias",
        )
        save_tensor(
            self.s3b2_bn2_gamma,
            weights_dir + "/s3b2_bn2_gamma.weights",
            "s3b2_bn2_gamma",
        )
        save_tensor(
            self.s3b2_bn2_beta,
            weights_dir + "/s3b2_bn2_beta.weights",
            "s3b2_bn2_beta",
        )

        # Stage 4
        save_tensor(
            self.s4b1_conv1_kernel,
            weights_dir + "/s4b1_conv1_kernel.weights",
            "s4b1_conv1_kernel",
        )
        save_tensor(
            self.s4b1_conv1_bias,
            weights_dir + "/s4b1_conv1_bias.weights",
            "s4b1_conv1_bias",
        )
        save_tensor(
            self.s4b1_bn1_gamma,
            weights_dir + "/s4b1_bn1_gamma.weights",
            "s4b1_bn1_gamma",
        )
        save_tensor(
            self.s4b1_bn1_beta,
            weights_dir + "/s4b1_bn1_beta.weights",
            "s4b1_bn1_beta",
        )
        save_tensor(
            self.s4b1_conv2_kernel,
            weights_dir + "/s4b1_conv2_kernel.weights",
            "s4b1_conv2_kernel",
        )
        save_tensor(
            self.s4b1_conv2_bias,
            weights_dir + "/s4b1_conv2_bias.weights",
            "s4b1_conv2_bias",
        )
        save_tensor(
            self.s4b1_bn2_gamma,
            weights_dir + "/s4b1_bn2_gamma.weights",
            "s4b1_bn2_gamma",
        )
        save_tensor(
            self.s4b1_bn2_beta,
            weights_dir + "/s4b1_bn2_beta.weights",
            "s4b1_bn2_beta",
        )
        save_tensor(
            self.s4b1_proj_kernel,
            weights_dir + "/s4b1_proj_kernel.weights",
            "s4b1_proj_kernel",
        )
        save_tensor(
            self.s4b1_proj_bias,
            weights_dir + "/s4b1_proj_bias.weights",
            "s4b1_proj_bias",
        )
        save_tensor(
            self.s4b1_proj_bn_gamma,
            weights_dir + "/s4b1_proj_bn_gamma.weights",
            "s4b1_proj_bn_gamma",
        )
        save_tensor(
            self.s4b1_proj_bn_beta,
            weights_dir + "/s4b1_proj_bn_beta.weights",
            "s4b1_proj_bn_beta",
        )

        save_tensor(
            self.s4b2_conv1_kernel,
            weights_dir + "/s4b2_conv1_kernel.weights",
            "s4b2_conv1_kernel",
        )
        save_tensor(
            self.s4b2_conv1_bias,
            weights_dir + "/s4b2_conv1_bias.weights",
            "s4b2_conv1_bias",
        )
        save_tensor(
            self.s4b2_bn1_gamma,
            weights_dir + "/s4b2_bn1_gamma.weights",
            "s4b2_bn1_gamma",
        )
        save_tensor(
            self.s4b2_bn1_beta,
            weights_dir + "/s4b2_bn1_beta.weights",
            "s4b2_bn1_beta",
        )
        save_tensor(
            self.s4b2_conv2_kernel,
            weights_dir + "/s4b2_conv2_kernel.weights",
            "s4b2_conv2_kernel",
        )
        save_tensor(
            self.s4b2_conv2_bias,
            weights_dir + "/s4b2_conv2_bias.weights",
            "s4b2_conv2_bias",
        )
        save_tensor(
            self.s4b2_bn2_gamma,
            weights_dir + "/s4b2_bn2_gamma.weights",
            "s4b2_bn2_gamma",
        )
        save_tensor(
            self.s4b2_bn2_beta,
            weights_dir + "/s4b2_bn2_beta.weights",
            "s4b2_bn2_beta",
        )

        # FC layer
        save_tensor(
            self.fc_weights, weights_dir + "/fc_weights.weights", "fc_weights"
        )
        save_tensor(self.fc_bias, weights_dir + "/fc_bias.weights", "fc_bias")

    def load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files.

        Note:
            Running stats (running_mean, running_var) are initialized to defaults
            and will be updated during first forward pass.
        """
        # Initial conv + BN
        self.conv1_kernel = load_tensor(weights_dir + "/conv1_kernel.weights")
        self.conv1_bias = load_tensor(weights_dir + "/conv1_bias.weights")
        self.bn1_gamma = load_tensor(weights_dir + "/bn1_gamma.weights")
        self.bn1_beta = load_tensor(weights_dir + "/bn1_beta.weights")

        # Stage 1
        self.s1b1_conv1_kernel = load_tensor(
            weights_dir + "/s1b1_conv1_kernel.weights"
        )
        self.s1b1_conv1_bias = load_tensor(
            weights_dir + "/s1b1_conv1_bias.weights"
        )
        self.s1b1_bn1_gamma = load_tensor(
            weights_dir + "/s1b1_bn1_gamma.weights"
        )
        self.s1b1_bn1_beta = load_tensor(weights_dir + "/s1b1_bn1_beta.weights")
        self.s1b1_conv2_kernel = load_tensor(
            weights_dir + "/s1b1_conv2_kernel.weights"
        )
        self.s1b1_conv2_bias = load_tensor(
            weights_dir + "/s1b1_conv2_bias.weights"
        )
        self.s1b1_bn2_gamma = load_tensor(
            weights_dir + "/s1b1_bn2_gamma.weights"
        )
        self.s1b1_bn2_beta = load_tensor(weights_dir + "/s1b1_bn2_beta.weights")

        self.s1b2_conv1_kernel = load_tensor(
            weights_dir + "/s1b2_conv1_kernel.weights"
        )
        self.s1b2_conv1_bias = load_tensor(
            weights_dir + "/s1b2_conv1_bias.weights"
        )
        self.s1b2_bn1_gamma = load_tensor(
            weights_dir + "/s1b2_bn1_gamma.weights"
        )
        self.s1b2_bn1_beta = load_tensor(weights_dir + "/s1b2_bn1_beta.weights")
        self.s1b2_conv2_kernel = load_tensor(
            weights_dir + "/s1b2_conv2_kernel.weights"
        )
        self.s1b2_conv2_bias = load_tensor(
            weights_dir + "/s1b2_conv2_bias.weights"
        )
        self.s1b2_bn2_gamma = load_tensor(
            weights_dir + "/s1b2_bn2_gamma.weights"
        )
        self.s1b2_bn2_beta = load_tensor(weights_dir + "/s1b2_bn2_beta.weights")

        # Stage 2
        self.s2b1_conv1_kernel = load_tensor(
            weights_dir + "/s2b1_conv1_kernel.weights"
        )
        self.s2b1_conv1_bias = load_tensor(
            weights_dir + "/s2b1_conv1_bias.weights"
        )
        self.s2b1_bn1_gamma = load_tensor(
            weights_dir + "/s2b1_bn1_gamma.weights"
        )
        self.s2b1_bn1_beta = load_tensor(weights_dir + "/s2b1_bn1_beta.weights")
        self.s2b1_conv2_kernel = load_tensor(
            weights_dir + "/s2b1_conv2_kernel.weights"
        )
        self.s2b1_conv2_bias = load_tensor(
            weights_dir + "/s2b1_conv2_bias.weights"
        )
        self.s2b1_bn2_gamma = load_tensor(
            weights_dir + "/s2b1_bn2_gamma.weights"
        )
        self.s2b1_bn2_beta = load_tensor(weights_dir + "/s2b1_bn2_beta.weights")
        self.s2b1_proj_kernel = load_tensor(
            weights_dir + "/s2b1_proj_kernel.weights"
        )
        self.s2b1_proj_bias = load_tensor(
            weights_dir + "/s2b1_proj_bias.weights"
        )
        self.s2b1_proj_bn_gamma = load_tensor(
            weights_dir + "/s2b1_proj_bn_gamma.weights"
        )
        self.s2b1_proj_bn_beta = load_tensor(
            weights_dir + "/s2b1_proj_bn_beta.weights"
        )

        self.s2b2_conv1_kernel = load_tensor(
            weights_dir + "/s2b2_conv1_kernel.weights"
        )
        self.s2b2_conv1_bias = load_tensor(
            weights_dir + "/s2b2_conv1_bias.weights"
        )
        self.s2b2_bn1_gamma = load_tensor(
            weights_dir + "/s2b2_bn1_gamma.weights"
        )
        self.s2b2_bn1_beta = load_tensor(weights_dir + "/s2b2_bn1_beta.weights")
        self.s2b2_conv2_kernel = load_tensor(
            weights_dir + "/s2b2_conv2_kernel.weights"
        )
        self.s2b2_conv2_bias = load_tensor(
            weights_dir + "/s2b2_conv2_bias.weights"
        )
        self.s2b2_bn2_gamma = load_tensor(
            weights_dir + "/s2b2_bn2_gamma.weights"
        )
        self.s2b2_bn2_beta = load_tensor(weights_dir + "/s2b2_bn2_beta.weights")

        # Stage 3
        self.s3b1_conv1_kernel = load_tensor(
            weights_dir + "/s3b1_conv1_kernel.weights"
        )
        self.s3b1_conv1_bias = load_tensor(
            weights_dir + "/s3b1_conv1_bias.weights"
        )
        self.s3b1_bn1_gamma = load_tensor(
            weights_dir + "/s3b1_bn1_gamma.weights"
        )
        self.s3b1_bn1_beta = load_tensor(weights_dir + "/s3b1_bn1_beta.weights")
        self.s3b1_conv2_kernel = load_tensor(
            weights_dir + "/s3b1_conv2_kernel.weights"
        )
        self.s3b1_conv2_bias = load_tensor(
            weights_dir + "/s3b1_conv2_bias.weights"
        )
        self.s3b1_bn2_gamma = load_tensor(
            weights_dir + "/s3b1_bn2_gamma.weights"
        )
        self.s3b1_bn2_beta = load_tensor(weights_dir + "/s3b1_bn2_beta.weights")
        self.s3b1_proj_kernel = load_tensor(
            weights_dir + "/s3b1_proj_kernel.weights"
        )
        self.s3b1_proj_bias = load_tensor(
            weights_dir + "/s3b1_proj_bias.weights"
        )
        self.s3b1_proj_bn_gamma = load_tensor(
            weights_dir + "/s3b1_proj_bn_gamma.weights"
        )
        self.s3b1_proj_bn_beta = load_tensor(
            weights_dir + "/s3b1_proj_bn_beta.weights"
        )

        self.s3b2_conv1_kernel = load_tensor(
            weights_dir + "/s3b2_conv1_kernel.weights"
        )
        self.s3b2_conv1_bias = load_tensor(
            weights_dir + "/s3b2_conv1_bias.weights"
        )
        self.s3b2_bn1_gamma = load_tensor(
            weights_dir + "/s3b2_bn1_gamma.weights"
        )
        self.s3b2_bn1_beta = load_tensor(weights_dir + "/s3b2_bn1_beta.weights")
        self.s3b2_conv2_kernel = load_tensor(
            weights_dir + "/s3b2_conv2_kernel.weights"
        )
        self.s3b2_conv2_bias = load_tensor(
            weights_dir + "/s3b2_conv2_bias.weights"
        )
        self.s3b2_bn2_gamma = load_tensor(
            weights_dir + "/s3b2_bn2_gamma.weights"
        )
        self.s3b2_bn2_beta = load_tensor(weights_dir + "/s3b2_bn2_beta.weights")

        # Stage 4
        self.s4b1_conv1_kernel = load_tensor(
            weights_dir + "/s4b1_conv1_kernel.weights"
        )
        self.s4b1_conv1_bias = load_tensor(
            weights_dir + "/s4b1_conv1_bias.weights"
        )
        self.s4b1_bn1_gamma = load_tensor(
            weights_dir + "/s4b1_bn1_gamma.weights"
        )
        self.s4b1_bn1_beta = load_tensor(weights_dir + "/s4b1_bn1_beta.weights")
        self.s4b1_conv2_kernel = load_tensor(
            weights_dir + "/s4b1_conv2_kernel.weights"
        )
        self.s4b1_conv2_bias = load_tensor(
            weights_dir + "/s4b1_conv2_bias.weights"
        )
        self.s4b1_bn2_gamma = load_tensor(
            weights_dir + "/s4b1_bn2_gamma.weights"
        )
        self.s4b1_bn2_beta = load_tensor(weights_dir + "/s4b1_bn2_beta.weights")
        self.s4b1_proj_kernel = load_tensor(
            weights_dir + "/s4b1_proj_kernel.weights"
        )
        self.s4b1_proj_bias = load_tensor(
            weights_dir + "/s4b1_proj_bias.weights"
        )
        self.s4b1_proj_bn_gamma = load_tensor(
            weights_dir + "/s4b1_proj_bn_gamma.weights"
        )
        self.s4b1_proj_bn_beta = load_tensor(
            weights_dir + "/s4b1_proj_bn_beta.weights"
        )

        self.s4b2_conv1_kernel = load_tensor(
            weights_dir + "/s4b2_conv1_kernel.weights"
        )
        self.s4b2_conv1_bias = load_tensor(
            weights_dir + "/s4b2_conv1_bias.weights"
        )
        self.s4b2_bn1_gamma = load_tensor(
            weights_dir + "/s4b2_bn1_gamma.weights"
        )
        self.s4b2_bn1_beta = load_tensor(weights_dir + "/s4b2_bn1_beta.weights")
        self.s4b2_conv2_kernel = load_tensor(
            weights_dir + "/s4b2_conv2_kernel.weights"
        )
        self.s4b2_conv2_bias = load_tensor(
            weights_dir + "/s4b2_conv2_bias.weights"
        )
        self.s4b2_bn2_gamma = load_tensor(
            weights_dir + "/s4b2_bn2_gamma.weights"
        )
        self.s4b2_bn2_beta = load_tensor(weights_dir + "/s4b2_bn2_beta.weights")

        # FC layer
        self.fc_weights = load_tensor(weights_dir + "/fc_weights.weights")
        self.fc_bias = load_tensor(weights_dir + "/fc_bias.weights")
