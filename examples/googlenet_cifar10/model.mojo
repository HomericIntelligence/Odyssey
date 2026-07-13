"""GoogLeNet (Inception-v1) Model for CIFAR-10.

This module implements the GoogLeNet/Inception-v1 architecture adapted for CIFAR-10.

Architecture:
    - 22 layers deep (9 Inception modules + initial layers + classifier)
    - Input: 32×32×3 RGB images
    - Output: 10 classes
    - ~6.8M parameters

Key Innovation:
    - Inception modules: Multi-scale parallel convolutions
    - 1×1 convolutions for dimensionality reduction
    - Global average pooling instead of large FC layers
    - Much more efficient than VGG-16 (fewer parameters)

References:
    Szegedy et al. (2015) - Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core import (
    conv2d,
    conv2d_backward,
    maxpool2d,
    maxpool2d_backward,
    global_avgpool2d,
    batch_norm2d,
    batch_norm2d_backward,
    relu,
    relu_backward,
    kaiming_normal,
    xavier_normal,
    constant,
)
from projectodyssey.core.arithmetic import add
from projectodyssey.core.linear import linear
from projectodyssey.core.dropout import dropout
from projectodyssey.utils.serialization import save_tensor, load_tensor


struct InceptionModule:
    """Inception module with 4 parallel branches.

    Architecture:
        Branch 1: 1×1 conv
        Branch 2: 1×1 conv (reduce) → 3×3 conv
        Branch 3: 1×1 conv (reduce) → 5×5 conv
        Branch 4: 3×3 max pool → 1×1 conv (project)

    All branches are concatenated depth-wise.

    The module contains the following parameters for each branch:
        Branch 1: conv1x1_1 (weights, bias), bn1x1_1 (gamma, beta, running_mean, running_var).
        Branch 2: conv1x1_2 (reduce), bn1x1_2, conv3x3, bn3x3.
        Branch 3: conv1x1_3 (reduce), bn1x1_3, conv5x5, bn5x5.
        Branch 4: conv1x1_4 (project after pool), bn1x1_4.
    """

    # Branch 1: 1×1 convolution
    var conv1x1_1_weights: AnyTensor
    var conv1x1_1_bias: AnyTensor
    var bn1x1_1_gamma: AnyTensor
    var bn1x1_1_beta: AnyTensor
    var bn1x1_1_running_mean: AnyTensor
    var bn1x1_1_running_var: AnyTensor

    # Branch 2: 1×1 reduce → 3×3
    var conv1x1_2_weights: AnyTensor
    var conv1x1_2_bias: AnyTensor
    var bn1x1_2_gamma: AnyTensor
    var bn1x1_2_beta: AnyTensor
    var bn1x1_2_running_mean: AnyTensor
    var bn1x1_2_running_var: AnyTensor

    var conv3x3_weights: AnyTensor
    var conv3x3_bias: AnyTensor
    var bn3x3_gamma: AnyTensor
    var bn3x3_beta: AnyTensor
    var bn3x3_running_mean: AnyTensor
    var bn3x3_running_var: AnyTensor

    # Branch 3: 1×1 reduce → 5×5
    var conv1x1_3_weights: AnyTensor
    var conv1x1_3_bias: AnyTensor
    var bn1x1_3_gamma: AnyTensor
    var bn1x1_3_beta: AnyTensor
    var bn1x1_3_running_mean: AnyTensor
    var bn1x1_3_running_var: AnyTensor

    var conv5x5_weights: AnyTensor
    var conv5x5_bias: AnyTensor
    var bn5x5_gamma: AnyTensor
    var bn5x5_beta: AnyTensor
    var bn5x5_running_mean: AnyTensor
    var bn5x5_running_var: AnyTensor

    # Branch 4: pool → 1×1 projection
    var conv1x1_4_weights: AnyTensor
    var conv1x1_4_bias: AnyTensor
    var bn1x1_4_gamma: AnyTensor
    var bn1x1_4_beta: AnyTensor
    var bn1x1_4_running_mean: AnyTensor
    var bn1x1_4_running_var: AnyTensor

    def __init__(
        out self,
        in_channels: Int,
        out_1x1: Int,
        reduce_3x3: Int,
        out_3x3: Int,
        reduce_5x5: Int,
        out_5x5: Int,
        pool_proj: Int,
    ) raises:
        """Initialize Inception module with specified channel configurations.

        Args:
            in_channels: Number of input channels.
            out_1x1: Output channels for 1×1 branch.
            reduce_3x3: Reduction channels before 3×3 conv.
            out_3x3: Output channels for 3×3 branch.
            reduce_5x5: Reduction channels before 5×5 conv.
            out_5x5: Output channels for 5×5 branch.
            pool_proj: Projection channels after pooling.
        """
        # Branch 1: 1×1 conv
        var conv1x1_1_weights_shape: List[Int] = [out_1x1, in_channels, 1, 1]
        self.conv1x1_1_weights = kaiming_normal(
            fan_in=in_channels, fan_out=out_1x1, shape=conv1x1_1_weights_shape
        )
        var conv1x1_1_bias_shape: List[Int] = [out_1x1]
        self.conv1x1_1_bias = zeros(conv1x1_1_bias_shape, DType.float32)
        self.bn1x1_1_gamma = constant(conv1x1_1_bias_shape, 1.0)
        self.bn1x1_1_beta = zeros(conv1x1_1_bias_shape, DType.float32)
        self.bn1x1_1_running_mean = zeros(conv1x1_1_bias_shape, DType.float32)
        self.bn1x1_1_running_var = constant(conv1x1_1_bias_shape, 1.0)

        # Branch 2: 1×1 reduce
        var conv1x1_2_weights_shape: List[Int] = [reduce_3x3, in_channels, 1, 1]
        self.conv1x1_2_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=reduce_3x3,
            shape=conv1x1_2_weights_shape,
        )
        var conv1x1_2_bias_shape: List[Int] = [reduce_3x3]
        self.conv1x1_2_bias = zeros(conv1x1_2_bias_shape, DType.float32)
        self.bn1x1_2_gamma = constant(conv1x1_2_bias_shape, 1.0)
        self.bn1x1_2_beta = zeros(conv1x1_2_bias_shape, DType.float32)
        self.bn1x1_2_running_mean = zeros(conv1x1_2_bias_shape, DType.float32)
        self.bn1x1_2_running_var = constant(conv1x1_2_bias_shape, 1.0)

        # Branch 2: 3×3 conv
        var conv3x3_weights_shape: List[Int] = [out_3x3, reduce_3x3, 3, 3]
        self.conv3x3_weights = kaiming_normal(
            fan_in=reduce_3x3 * 9, fan_out=out_3x3, shape=conv3x3_weights_shape
        )
        var conv3x3_bias_shape: List[Int] = [out_3x3]
        self.conv3x3_bias = zeros(conv3x3_bias_shape, DType.float32)
        self.bn3x3_gamma = constant(conv3x3_bias_shape, 1.0)
        self.bn3x3_beta = zeros(conv3x3_bias_shape, DType.float32)
        self.bn3x3_running_mean = zeros(conv3x3_bias_shape, DType.float32)
        self.bn3x3_running_var = constant(conv3x3_bias_shape, 1.0)

        # Branch 3: 1×1 reduce
        var conv1x1_3_weights_shape: List[Int] = [reduce_5x5, in_channels, 1, 1]
        self.conv1x1_3_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=reduce_5x5,
            shape=conv1x1_3_weights_shape,
        )
        var conv1x1_3_bias_shape: List[Int] = [reduce_5x5]
        self.conv1x1_3_bias = zeros(conv1x1_3_bias_shape, DType.float32)
        self.bn1x1_3_gamma = constant(conv1x1_3_bias_shape, 1.0)
        self.bn1x1_3_beta = zeros(conv1x1_3_bias_shape, DType.float32)
        self.bn1x1_3_running_mean = zeros(conv1x1_3_bias_shape, DType.float32)
        self.bn1x1_3_running_var = constant(conv1x1_3_bias_shape, 1.0)

        # Branch 3: 5×5 conv
        var conv5x5_weights_shape: List[Int] = [out_5x5, reduce_5x5, 5, 5]
        self.conv5x5_weights = kaiming_normal(
            fan_in=reduce_5x5 * 25, fan_out=out_5x5, shape=conv5x5_weights_shape
        )
        var conv5x5_bias_shape: List[Int] = [out_5x5]
        self.conv5x5_bias = zeros(conv5x5_bias_shape, DType.float32)
        self.bn5x5_gamma = constant(conv5x5_bias_shape, 1.0)
        self.bn5x5_beta = zeros(conv5x5_bias_shape, DType.float32)
        self.bn5x5_running_mean = zeros(conv5x5_bias_shape, DType.float32)
        self.bn5x5_running_var = constant(conv5x5_bias_shape, 1.0)

        # Branch 4: 1×1 projection after pooling
        var conv1x1_4_weights_shape: List[Int] = [pool_proj, in_channels, 1, 1]
        self.conv1x1_4_weights = kaiming_normal(
            fan_in=in_channels, fan_out=pool_proj, shape=conv1x1_4_weights_shape
        )
        var conv1x1_4_bias_shape: List[Int] = [pool_proj]
        self.conv1x1_4_bias = zeros(conv1x1_4_bias_shape, DType.float32)
        self.bn1x1_4_gamma = constant(conv1x1_4_bias_shape, 1.0)
        self.bn1x1_4_beta = zeros(conv1x1_4_bias_shape, DType.float32)
        self.bn1x1_4_running_mean = zeros(conv1x1_4_bias_shape, DType.float32)
        self.bn1x1_4_running_var = constant(conv1x1_4_bias_shape, 1.0)

    def forward(mut self, x: AnyTensor, training: Bool) raises -> AnyTensor:
        """Forward pass through Inception module.

        Args:
            x: Input tensor (batch, in_channels, H, W).
            training: Training mode flag (affects batch norm).

        Returns:
            Output tensor (batch, out_channels, H, W)
            where out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj.
        """
        # Branch 1: 1×1 conv
        var b1 = conv2d(
            x, self.conv1x1_1_weights, self.conv1x1_1_bias, stride=1, padding=0
        )
        var b1_bn_result = batch_norm2d(
            b1,
            self.bn1x1_1_gamma,
            self.bn1x1_1_beta,
            self.bn1x1_1_running_mean,
            self.bn1x1_1_running_var,
            training,
        )
        b1 = b1_bn_result[0]
        self.bn1x1_1_running_mean = b1_bn_result[1]
        self.bn1x1_1_running_var = b1_bn_result[2]
        b1 = relu(b1)

        # Branch 2: 1×1 reduce → 3×3 conv
        var b2 = conv2d(
            x, self.conv1x1_2_weights, self.conv1x1_2_bias, stride=1, padding=0
        )
        var b2_bn1_result = batch_norm2d(
            b2,
            self.bn1x1_2_gamma,
            self.bn1x1_2_beta,
            self.bn1x1_2_running_mean,
            self.bn1x1_2_running_var,
            training,
        )
        b2 = b2_bn1_result[0]
        self.bn1x1_2_running_mean = b2_bn1_result[1]
        self.bn1x1_2_running_var = b2_bn1_result[2]
        b2 = relu(b2)
        b2 = conv2d(
            b2, self.conv3x3_weights, self.conv3x3_bias, stride=1, padding=1
        )
        var b2_bn2_result = batch_norm2d(
            b2,
            self.bn3x3_gamma,
            self.bn3x3_beta,
            self.bn3x3_running_mean,
            self.bn3x3_running_var,
            training,
        )
        b2 = b2_bn2_result[0]
        self.bn3x3_running_mean = b2_bn2_result[1]
        self.bn3x3_running_var = b2_bn2_result[2]
        b2 = relu(b2)

        # Branch 3: 1×1 reduce → 5×5 conv
        var b3 = conv2d(
            x, self.conv1x1_3_weights, self.conv1x1_3_bias, stride=1, padding=0
        )
        var b3_bn1_result = batch_norm2d(
            b3,
            self.bn1x1_3_gamma,
            self.bn1x1_3_beta,
            self.bn1x1_3_running_mean,
            self.bn1x1_3_running_var,
            training,
        )
        b3 = b3_bn1_result[0]
        self.bn1x1_3_running_mean = b3_bn1_result[1]
        self.bn1x1_3_running_var = b3_bn1_result[2]
        b3 = relu(b3)
        b3 = conv2d(
            b3, self.conv5x5_weights, self.conv5x5_bias, stride=1, padding=2
        )
        var b3_bn2_result = batch_norm2d(
            b3,
            self.bn5x5_gamma,
            self.bn5x5_beta,
            self.bn5x5_running_mean,
            self.bn5x5_running_var,
            training,
        )
        b3 = b3_bn2_result[0]
        self.bn5x5_running_mean = b3_bn2_result[1]
        self.bn5x5_running_var = b3_bn2_result[2]
        b3 = relu(b3)

        # Branch 4: 3×3 max pool → 1×1 projection
        var b4 = maxpool2d(x, kernel_size=3, stride=1, padding=1)
        b4 = conv2d(
            b4, self.conv1x1_4_weights, self.conv1x1_4_bias, stride=1, padding=0
        )
        var b4_bn_result = batch_norm2d(
            b4,
            self.bn1x1_4_gamma,
            self.bn1x1_4_beta,
            self.bn1x1_4_running_mean,
            self.bn1x1_4_running_var,
            training,
        )
        b4 = b4_bn_result[0]
        self.bn1x1_4_running_mean = b4_bn_result[1]
        self.bn1x1_4_running_var = b4_bn_result[2]
        b4 = relu(b4)

        # Concatenate all branches depth-wise
        return concatenate_depthwise(b1, b2, b3, b4)

    def save_weights(self, weights_dir: String, prefix: String) raises:
        """Save all Inception module weights to directory.

        Args:
            weights_dir: Directory to save weight files.
            prefix: Prefix for weight file names (e.g., "inception_3a").

        Note:
            Running stats (running_mean, running_var) are not saved as they
            are recomputed during training.
        """
        # Branch 1: 1×1 conv
        save_tensor(
            self.conv1x1_1_weights,
            weights_dir + "/" + prefix + "_b1_conv_weights.bin",
            prefix + "_b1_conv_weights",
        )
        save_tensor(
            self.conv1x1_1_bias,
            weights_dir + "/" + prefix + "_b1_conv_bias.bin",
            prefix + "_b1_conv_bias",
        )
        save_tensor(
            self.bn1x1_1_gamma,
            weights_dir + "/" + prefix + "_b1_bn_gamma.bin",
            prefix + "_b1_bn_gamma",
        )
        save_tensor(
            self.bn1x1_1_beta,
            weights_dir + "/" + prefix + "_b1_bn_beta.bin",
            prefix + "_b1_bn_beta",
        )

        # Branch 2: 1×1 reduce + 3×3
        save_tensor(
            self.conv1x1_2_weights,
            weights_dir + "/" + prefix + "_b2_reduce_weights.bin",
            prefix + "_b2_reduce_weights",
        )
        save_tensor(
            self.conv1x1_2_bias,
            weights_dir + "/" + prefix + "_b2_reduce_bias.bin",
            prefix + "_b2_reduce_bias",
        )
        save_tensor(
            self.bn1x1_2_gamma,
            weights_dir + "/" + prefix + "_b2_reduce_bn_gamma.bin",
            prefix + "_b2_reduce_bn_gamma",
        )
        save_tensor(
            self.bn1x1_2_beta,
            weights_dir + "/" + prefix + "_b2_reduce_bn_beta.bin",
            prefix + "_b2_reduce_bn_beta",
        )
        save_tensor(
            self.conv3x3_weights,
            weights_dir + "/" + prefix + "_b2_conv3x3_weights.bin",
            prefix + "_b2_conv3x3_weights",
        )
        save_tensor(
            self.conv3x3_bias,
            weights_dir + "/" + prefix + "_b2_conv3x3_bias.bin",
            prefix + "_b2_conv3x3_bias",
        )
        save_tensor(
            self.bn3x3_gamma,
            weights_dir + "/" + prefix + "_b2_bn3x3_gamma.bin",
            prefix + "_b2_bn3x3_gamma",
        )
        save_tensor(
            self.bn3x3_beta,
            weights_dir + "/" + prefix + "_b2_bn3x3_beta.bin",
            prefix + "_b2_bn3x3_beta",
        )

        # Branch 3: 1×1 reduce + 5×5
        save_tensor(
            self.conv1x1_3_weights,
            weights_dir + "/" + prefix + "_b3_reduce_weights.bin",
            prefix + "_b3_reduce_weights",
        )
        save_tensor(
            self.conv1x1_3_bias,
            weights_dir + "/" + prefix + "_b3_reduce_bias.bin",
            prefix + "_b3_reduce_bias",
        )
        save_tensor(
            self.bn1x1_3_gamma,
            weights_dir + "/" + prefix + "_b3_reduce_bn_gamma.bin",
            prefix + "_b3_reduce_bn_gamma",
        )
        save_tensor(
            self.bn1x1_3_beta,
            weights_dir + "/" + prefix + "_b3_reduce_bn_beta.bin",
            prefix + "_b3_reduce_bn_beta",
        )
        save_tensor(
            self.conv5x5_weights,
            weights_dir + "/" + prefix + "_b3_conv5x5_weights.bin",
            prefix + "_b3_conv5x5_weights",
        )
        save_tensor(
            self.conv5x5_bias,
            weights_dir + "/" + prefix + "_b3_conv5x5_bias.bin",
            prefix + "_b3_conv5x5_bias",
        )
        save_tensor(
            self.bn5x5_gamma,
            weights_dir + "/" + prefix + "_b3_bn5x5_gamma.bin",
            prefix + "_b3_bn5x5_gamma",
        )
        save_tensor(
            self.bn5x5_beta,
            weights_dir + "/" + prefix + "_b3_bn5x5_beta.bin",
            prefix + "_b3_bn5x5_beta",
        )

        # Branch 4: pool + 1×1 projection
        save_tensor(
            self.conv1x1_4_weights,
            weights_dir + "/" + prefix + "_b4_proj_weights.bin",
            prefix + "_b4_proj_weights",
        )
        save_tensor(
            self.conv1x1_4_bias,
            weights_dir + "/" + prefix + "_b4_proj_bias.bin",
            prefix + "_b4_proj_bias",
        )
        save_tensor(
            self.bn1x1_4_gamma,
            weights_dir + "/" + prefix + "_b4_bn_gamma.bin",
            prefix + "_b4_bn_gamma",
        )
        save_tensor(
            self.bn1x1_4_beta,
            weights_dir + "/" + prefix + "_b4_bn_beta.bin",
            prefix + "_b4_bn_beta",
        )

    def load_weights(mut self, weights_dir: String, prefix: String) raises:
        """Load all Inception module weights from directory.

        Args:
            weights_dir: Directory containing saved weight files.
            prefix: Prefix for weight file names (e.g., "inception_3a").
        """
        # Branch 1: 1×1 conv
        self.conv1x1_1_weights = load_tensor(
            weights_dir + "/" + prefix + "_b1_conv_weights.bin"
        )
        self.conv1x1_1_bias = load_tensor(
            weights_dir + "/" + prefix + "_b1_conv_bias.bin"
        )
        self.bn1x1_1_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b1_bn_gamma.bin"
        )
        self.bn1x1_1_beta = load_tensor(
            weights_dir + "/" + prefix + "_b1_bn_beta.bin"
        )

        # Branch 2: 1×1 reduce + 3×3
        self.conv1x1_2_weights = load_tensor(
            weights_dir + "/" + prefix + "_b2_reduce_weights.bin"
        )
        self.conv1x1_2_bias = load_tensor(
            weights_dir + "/" + prefix + "_b2_reduce_bias.bin"
        )
        self.bn1x1_2_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b2_reduce_bn_gamma.bin"
        )
        self.bn1x1_2_beta = load_tensor(
            weights_dir + "/" + prefix + "_b2_reduce_bn_beta.bin"
        )
        self.conv3x3_weights = load_tensor(
            weights_dir + "/" + prefix + "_b2_conv3x3_weights.bin"
        )
        self.conv3x3_bias = load_tensor(
            weights_dir + "/" + prefix + "_b2_conv3x3_bias.bin"
        )
        self.bn3x3_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b2_bn3x3_gamma.bin"
        )
        self.bn3x3_beta = load_tensor(
            weights_dir + "/" + prefix + "_b2_bn3x3_beta.bin"
        )

        # Branch 3: 1×1 reduce + 5×5
        self.conv1x1_3_weights = load_tensor(
            weights_dir + "/" + prefix + "_b3_reduce_weights.bin"
        )
        self.conv1x1_3_bias = load_tensor(
            weights_dir + "/" + prefix + "_b3_reduce_bias.bin"
        )
        self.bn1x1_3_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b3_reduce_bn_gamma.bin"
        )
        self.bn1x1_3_beta = load_tensor(
            weights_dir + "/" + prefix + "_b3_reduce_bn_beta.bin"
        )
        self.conv5x5_weights = load_tensor(
            weights_dir + "/" + prefix + "_b3_conv5x5_weights.bin"
        )
        self.conv5x5_bias = load_tensor(
            weights_dir + "/" + prefix + "_b3_conv5x5_bias.bin"
        )
        self.bn5x5_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b3_bn5x5_gamma.bin"
        )
        self.bn5x5_beta = load_tensor(
            weights_dir + "/" + prefix + "_b3_bn5x5_beta.bin"
        )

        # Branch 4: pool + 1×1 projection
        self.conv1x1_4_weights = load_tensor(
            weights_dir + "/" + prefix + "_b4_proj_weights.bin"
        )
        self.conv1x1_4_bias = load_tensor(
            weights_dir + "/" + prefix + "_b4_proj_bias.bin"
        )
        self.bn1x1_4_gamma = load_tensor(
            weights_dir + "/" + prefix + "_b4_bn_gamma.bin"
        )
        self.bn1x1_4_beta = load_tensor(
            weights_dir + "/" + prefix + "_b4_bn_beta.bin"
        )


def concatenate_depthwise(
    t1: AnyTensor, t2: AnyTensor, t3: AnyTensor, t4: AnyTensor
) raises -> AnyTensor:
    """Concatenate 4 tensors along the channel dimension (axis=1).

    Args:
        t1: Tensor 1 (batch, C1, H, W).
        t2: Tensor 2 (batch, C2, H, W).
        t3: Tensor 3 (batch, C3, H, W).
        t4: Tensor 4 (batch, C4, H, W).

    Returns:
        Concatenated tensor (batch, C1+C2+C3+C4, H, W).
    """
    var batch_size = t1.shape()[0]
    var c1 = t1.shape()[1]
    var c2 = t2.shape()[1]
    var c3 = t3.shape()[1]
    var c4 = t4.shape()[1]
    var height = t1.shape()[2]
    var width = t1.shape()[3]

    var total_channels = c1 + c2 + c3 + c4
    var result_shape: List[Int] = [batch_size, total_channels, height, width]
    var result = zeros(result_shape, t1.dtype())

    # Copy data from each tensor
    var result_data = result._data.bitcast[Float32]()
    var t1_data = t1._data.bitcast[Float32]()
    var t2_data = t2._data.bitcast[Float32]()
    var t3_data = t3._data.bitcast[Float32]()
    var t4_data = t4._data.bitcast[Float32]()

    var hw = height * width

    for b in range(batch_size):
        # Copy t1 channels
        for c in range(c1):
            for i in range(hw):
                var src_idx = ((b * c1 + c) * hw) + i
                var dst_idx = ((b * total_channels + c) * hw) + i
                result_data[dst_idx] = t1_data[src_idx]

        # Copy t2 channels (offset by c1)
        for c in range(c2):
            for i in range(hw):
                var src_idx = ((b * c2 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c)) * hw) + i
                result_data[dst_idx] = t2_data[src_idx]

        # Copy t3 channels (offset by c1+c2)
        for c in range(c3):
            for i in range(hw):
                var src_idx = ((b * c3 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c2 + c)) * hw) + i
                result_data[dst_idx] = t3_data[src_idx]

        # Copy t4 channels (offset by c1+c2+c3)
        for c in range(c4):
            for i in range(hw):
                var src_idx = ((b * c4 + c) * hw) + i
                var dst_idx = (
                    (b * total_channels + (c1 + c2 + c3 + c)) * hw
                ) + i
                result_data[dst_idx] = t4_data[src_idx]

    return result


def concatenate_depthwise_backward(
    grad_output: AnyTensor, c1: Int, c2: Int, c3: Int, c4: Int
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Backward pass for concatenate_depthwise (channel-dim split).

    concatenate_depthwise stacks 4 tensors along axis=1, so the gradient
    w.r.t. each input is simply the corresponding channel slice of
    grad_output. This is the exact inverse of the forward copy loops above.

    Args:
        grad_output: Gradient w.r.t. the concatenated output
            (batch, c1+c2+c3+c4, H, W).
        c1: Channel count of branch 1.
        c2: Channel count of branch 2.
        c3: Channel count of branch 3.
        c4: Channel count of branch 4.

    Returns:
        Tuple of gradients (g1, g2, g3, g4) with shapes
        (batch, c1, H, W), (batch, c2, H, W), (batch, c3, H, W),
        (batch, c4, H, W).
    """
    # Float32-only contract: the copy loops below reinterpret memory via
    # bitcast[Float32]. A non-float32 grad_output would be silently
    # type-confused, so fail loudly instead.
    if grad_output.dtype() != DType.float32:
        raise Error(
            "concatenate_depthwise_backward requires float32 grad_output, got "
            + String(grad_output.dtype())
        )

    var batch_size = grad_output.shape()[0]
    var total_channels = grad_output.shape()[1]
    var height = grad_output.shape()[2]
    var width = grad_output.shape()[3]
    var hw = height * width

    var g1 = zeros([batch_size, c1, height, width], grad_output.dtype())
    var g2 = zeros([batch_size, c2, height, width], grad_output.dtype())
    var g3 = zeros([batch_size, c3, height, width], grad_output.dtype())
    var g4 = zeros([batch_size, c4, height, width], grad_output.dtype())

    var go_data = grad_output._data.bitcast[Float32]()
    var g1_data = g1._data.bitcast[Float32]()
    var g2_data = g2._data.bitcast[Float32]()
    var g3_data = g3._data.bitcast[Float32]()
    var g4_data = g4._data.bitcast[Float32]()

    for b in range(batch_size):
        # Branch 1 channels: [0, c1)
        for c in range(c1):
            for i in range(hw):
                var src_idx = ((b * total_channels + c) * hw) + i
                var dst_idx = ((b * c1 + c) * hw) + i
                g1_data[dst_idx] = go_data[src_idx]

        # Branch 2 channels: [c1, c1+c2)
        for c in range(c2):
            for i in range(hw):
                var src_idx = ((b * total_channels + (c1 + c)) * hw) + i
                var dst_idx = ((b * c2 + c) * hw) + i
                g2_data[dst_idx] = go_data[src_idx]

        # Branch 3 channels: [c1+c2, c1+c2+c3)
        for c in range(c3):
            for i in range(hw):
                var src_idx = ((b * total_channels + (c1 + c2 + c)) * hw) + i
                var dst_idx = ((b * c3 + c) * hw) + i
                g3_data[dst_idx] = go_data[src_idx]

        # Branch 4 channels: [c1+c2+c3, total)
        for c in range(c4):
            for i in range(hw):
                var src_idx = (
                    (b * total_channels + (c1 + c2 + c3 + c)) * hw
                ) + i
                var dst_idx = ((b * c4 + c) * hw) + i
                g4_data[dst_idx] = go_data[src_idx]

    return (g1^, g2^, g3^, g4^)


@fieldwise_init
struct InceptionCache(Movable):
    """Cached forward activations for one Inception module's backward pass.

    Stores the pre-BN (conv output) and pre-ReLU (BN output) activations of
    every conv/BN stage in all four branches, plus the module input and the
    branch channel counts. These are exactly the tensors the backward
    primitives (conv2d_backward, batch_norm2d_backward, relu_backward,
    maxpool2d_backward) consume.
    """

    var block_input: AnyTensor  # x into the module

    # Branch 1: conv1x1_1 -> bn -> relu
    var b1_conv_out: AnyTensor  # pre-BN (conv output)
    var b1_bn_out: AnyTensor  # pre-ReLU (BN output)

    # Branch 2: conv1x1_2 -> bn -> relu -> conv3x3 -> bn -> relu
    var b2_conv1_out: AnyTensor
    var b2_bn1_out: AnyTensor
    var b2_relu1_out: AnyTensor  # input to conv3x3
    var b2_conv2_out: AnyTensor
    var b2_bn2_out: AnyTensor

    # Branch 3: conv1x1_3 -> bn -> relu -> conv5x5 -> bn -> relu
    var b3_conv1_out: AnyTensor
    var b3_bn1_out: AnyTensor
    var b3_relu1_out: AnyTensor  # input to conv5x5
    var b3_conv2_out: AnyTensor
    var b3_bn2_out: AnyTensor

    # Branch 4: maxpool -> conv1x1_4 -> bn -> relu
    var b4_pool_out: AnyTensor  # input to conv1x1_4
    var b4_conv_out: AnyTensor
    var b4_bn_out: AnyTensor

    # Branch channel counts (for the concat split)
    var c1: Int
    var c2: Int
    var c3: Int
    var c4: Int


@fieldwise_init
struct InceptionGradients(Movable):
    """Gradient bundle for one Inception module.

    grad_input is the gradient w.r.t. the module input x (summed over the four
    branches). The remaining fields are gradients for every trainable weight,
    in the same order the velocities buffer stores them.
    """

    var grad_input: AnyTensor

    # Branch 1
    var g_conv1x1_1_w: AnyTensor
    var g_conv1x1_1_b: AnyTensor
    var g_bn1x1_1_gamma: AnyTensor
    var g_bn1x1_1_beta: AnyTensor

    # Branch 2
    var g_conv1x1_2_w: AnyTensor
    var g_conv1x1_2_b: AnyTensor
    var g_bn1x1_2_gamma: AnyTensor
    var g_bn1x1_2_beta: AnyTensor
    var g_conv3x3_w: AnyTensor
    var g_conv3x3_b: AnyTensor
    var g_bn3x3_gamma: AnyTensor
    var g_bn3x3_beta: AnyTensor

    # Branch 3
    var g_conv1x1_3_w: AnyTensor
    var g_conv1x1_3_b: AnyTensor
    var g_bn1x1_3_gamma: AnyTensor
    var g_bn1x1_3_beta: AnyTensor
    var g_conv5x5_w: AnyTensor
    var g_conv5x5_b: AnyTensor
    var g_bn5x5_gamma: AnyTensor
    var g_bn5x5_beta: AnyTensor

    # Branch 4
    var g_conv1x1_4_w: AnyTensor
    var g_conv1x1_4_b: AnyTensor
    var g_bn1x1_4_gamma: AnyTensor
    var g_bn1x1_4_beta: AnyTensor


def inception_forward_cached(
    mut module: InceptionModule, x: AnyTensor
) raises -> Tuple[AnyTensor, InceptionCache]:
    """Forward pass through one Inception module, caching every activation.

    Runs BatchNorm in training mode and writes the EMA-updated running_mean /
    running_var back onto ``module`` at every BN site, so that post-training
    inference (``training=False``) uses the accumulated statistics rather than
    the init values (mean=0, var=1). Taking ``module`` mutably is what makes the
    write-back possible (fixes #5575).

    Returns:
        (output, cache) where output is the depth-wise concatenation of the
        four branches and cache holds the intermediate activations.
    """
    # Branch 1: conv1x1_1 -> BN -> ReLU
    var b1_conv = conv2d(
        x, module.conv1x1_1_weights, module.conv1x1_1_bias, stride=1, padding=0
    )
    var b1_bn_result = batch_norm2d(
        b1_conv,
        module.bn1x1_1_gamma,
        module.bn1x1_1_beta,
        module.bn1x1_1_running_mean,
        module.bn1x1_1_running_var,
        True,
    )
    var b1_bn = b1_bn_result[0]
    module.bn1x1_1_running_mean = b1_bn_result[1]
    module.bn1x1_1_running_var = b1_bn_result[2]
    var b1 = relu(b1_bn)

    # Branch 2: conv1x1_2 -> BN -> ReLU -> conv3x3 -> BN -> ReLU
    var b2_conv1 = conv2d(
        x, module.conv1x1_2_weights, module.conv1x1_2_bias, stride=1, padding=0
    )
    var b2_bn1_result = batch_norm2d(
        b2_conv1,
        module.bn1x1_2_gamma,
        module.bn1x1_2_beta,
        module.bn1x1_2_running_mean,
        module.bn1x1_2_running_var,
        True,
    )
    var b2_bn1 = b2_bn1_result[0]
    module.bn1x1_2_running_mean = b2_bn1_result[1]
    module.bn1x1_2_running_var = b2_bn1_result[2]
    var b2_relu1 = relu(b2_bn1)
    var b2_conv2 = conv2d(
        b2_relu1,
        module.conv3x3_weights,
        module.conv3x3_bias,
        stride=1,
        padding=1,
    )
    var b2_bn2_result = batch_norm2d(
        b2_conv2,
        module.bn3x3_gamma,
        module.bn3x3_beta,
        module.bn3x3_running_mean,
        module.bn3x3_running_var,
        True,
    )
    var b2_bn2 = b2_bn2_result[0]
    module.bn3x3_running_mean = b2_bn2_result[1]
    module.bn3x3_running_var = b2_bn2_result[2]
    var b2 = relu(b2_bn2)

    # Branch 3: conv1x1_3 -> BN -> ReLU -> conv5x5 -> BN -> ReLU
    var b3_conv1 = conv2d(
        x, module.conv1x1_3_weights, module.conv1x1_3_bias, stride=1, padding=0
    )
    var b3_bn1_result = batch_norm2d(
        b3_conv1,
        module.bn1x1_3_gamma,
        module.bn1x1_3_beta,
        module.bn1x1_3_running_mean,
        module.bn1x1_3_running_var,
        True,
    )
    var b3_bn1 = b3_bn1_result[0]
    module.bn1x1_3_running_mean = b3_bn1_result[1]
    module.bn1x1_3_running_var = b3_bn1_result[2]
    var b3_relu1 = relu(b3_bn1)
    var b3_conv2 = conv2d(
        b3_relu1,
        module.conv5x5_weights,
        module.conv5x5_bias,
        stride=1,
        padding=2,
    )
    var b3_bn2_result = batch_norm2d(
        b3_conv2,
        module.bn5x5_gamma,
        module.bn5x5_beta,
        module.bn5x5_running_mean,
        module.bn5x5_running_var,
        True,
    )
    var b3_bn2 = b3_bn2_result[0]
    module.bn5x5_running_mean = b3_bn2_result[1]
    module.bn5x5_running_var = b3_bn2_result[2]
    var b3 = relu(b3_bn2)

    # Branch 4: maxpool -> conv1x1_4 -> BN -> ReLU
    var b4_pool = maxpool2d(x, kernel_size=3, stride=1, padding=1)
    var b4_conv = conv2d(
        b4_pool,
        module.conv1x1_4_weights,
        module.conv1x1_4_bias,
        stride=1,
        padding=0,
    )
    var b4_bn_result = batch_norm2d(
        b4_conv,
        module.bn1x1_4_gamma,
        module.bn1x1_4_beta,
        module.bn1x1_4_running_mean,
        module.bn1x1_4_running_var,
        True,
    )
    var b4_bn = b4_bn_result[0]
    module.bn1x1_4_running_mean = b4_bn_result[1]
    module.bn1x1_4_running_var = b4_bn_result[2]
    var b4 = relu(b4_bn)

    var c1 = b1.shape()[1]
    var c2 = b2.shape()[1]
    var c3 = b3.shape()[1]
    var c4 = b4.shape()[1]
    var out = concatenate_depthwise(b1, b2, b3, b4)

    var cache = InceptionCache(
        x,
        b1_conv^,
        b1_bn^,
        b2_conv1^,
        b2_bn1^,
        b2_relu1^,
        b2_conv2^,
        b2_bn2^,
        b3_conv1^,
        b3_bn1^,
        b3_relu1^,
        b3_conv2^,
        b3_bn2^,
        b4_pool^,
        b4_conv^,
        b4_bn^,
        c1,
        c2,
        c3,
        c4,
    )
    return (out^, cache^)


def inception_backward(
    module: InceptionModule, grad_output: AnyTensor, cache: InceptionCache
) raises -> InceptionGradients:
    """Backward pass through one Inception module.

    Splits grad_output into 4 branch gradients (channel-dim), runs each branch
    backward (relu -> BN -> conv chains), and sums the four input gradients at x.
    """
    var splits = concatenate_depthwise_backward(
        grad_output, cache.c1, cache.c2, cache.c3, cache.c4
    )
    var d_b1 = splits[0]
    var d_b2 = splits[1]
    var d_b3 = splits[2]
    var d_b4 = splits[3]

    # ---- Branch 1: relu <- BN <- conv1x1_1 ----
    var d_b1_bn = relu_backward(d_b1, cache.b1_bn_out)
    var b1_bn_in, g_bn1_gamma, g_bn1_beta = batch_norm2d_backward(
        d_b1_bn,
        cache.b1_conv_out,
        module.bn1x1_1_gamma,
        module.bn1x1_1_running_mean,
        module.bn1x1_1_running_var,
        training=True,
    )
    var b1c = conv2d_backward(
        b1_bn_in,
        cache.block_input,
        module.conv1x1_1_weights,
        stride=1,
        padding=0,
    )

    # ---- Branch 2: relu <- BN <- conv3x3 <- relu <- BN <- conv1x1_2 ----
    var d_b2_bn2 = relu_backward(d_b2, cache.b2_bn2_out)
    var b2_bn2_in, g_bn3x3_gamma, g_bn3x3_beta = batch_norm2d_backward(
        d_b2_bn2,
        cache.b2_conv2_out,
        module.bn3x3_gamma,
        module.bn3x3_running_mean,
        module.bn3x3_running_var,
        training=True,
    )
    var b2c2 = conv2d_backward(
        b2_bn2_in,
        cache.b2_relu1_out,
        module.conv3x3_weights,
        stride=1,
        padding=1,
    )
    var d_b2_bn1 = relu_backward(b2c2.grad_input, cache.b2_bn1_out)
    var b2_bn1_in, g_bn1x1_2_gamma, g_bn1x1_2_beta = batch_norm2d_backward(
        d_b2_bn1,
        cache.b2_conv1_out,
        module.bn1x1_2_gamma,
        module.bn1x1_2_running_mean,
        module.bn1x1_2_running_var,
        training=True,
    )
    var b2c1 = conv2d_backward(
        b2_bn1_in,
        cache.block_input,
        module.conv1x1_2_weights,
        stride=1,
        padding=0,
    )

    # ---- Branch 3: relu <- BN <- conv5x5 <- relu <- BN <- conv1x1_3 ----
    var d_b3_bn2 = relu_backward(d_b3, cache.b3_bn2_out)
    var b3_bn2_in, g_bn5x5_gamma, g_bn5x5_beta = batch_norm2d_backward(
        d_b3_bn2,
        cache.b3_conv2_out,
        module.bn5x5_gamma,
        module.bn5x5_running_mean,
        module.bn5x5_running_var,
        training=True,
    )
    var b3c2 = conv2d_backward(
        b3_bn2_in,
        cache.b3_relu1_out,
        module.conv5x5_weights,
        stride=1,
        padding=2,
    )
    var d_b3_bn1 = relu_backward(b3c2.grad_input, cache.b3_bn1_out)
    var b3_bn1_in, g_bn1x1_3_gamma, g_bn1x1_3_beta = batch_norm2d_backward(
        d_b3_bn1,
        cache.b3_conv1_out,
        module.bn1x1_3_gamma,
        module.bn1x1_3_running_mean,
        module.bn1x1_3_running_var,
        training=True,
    )
    var b3c1 = conv2d_backward(
        b3_bn1_in,
        cache.block_input,
        module.conv1x1_3_weights,
        stride=1,
        padding=0,
    )

    # ---- Branch 4: relu <- BN <- conv1x1_4 <- maxpool ----
    var d_b4_bn = relu_backward(d_b4, cache.b4_bn_out)
    var b4_bn_in, g_bn1x1_4_gamma, g_bn1x1_4_beta = batch_norm2d_backward(
        d_b4_bn,
        cache.b4_conv_out,
        module.bn1x1_4_gamma,
        module.bn1x1_4_running_mean,
        module.bn1x1_4_running_var,
        training=True,
    )
    var b4c = conv2d_backward(
        b4_bn_in,
        cache.b4_pool_out,
        module.conv1x1_4_weights,
        stride=1,
        padding=0,
    )
    var d_b4_pool = maxpool2d_backward(
        b4c.grad_input, cache.block_input, kernel_size=3, stride=1, padding=1
    )

    # ---- Sum the four branch input-gradients at x ----
    var grad_input = add(b1c.grad_input, b2c1.grad_input)
    grad_input = add(grad_input, b3c1.grad_input)
    grad_input = add(grad_input, d_b4_pool)

    return InceptionGradients(
        grad_input^,
        b1c.grad_weights,
        b1c.grad_bias,
        g_bn1_gamma,
        g_bn1_beta,
        b2c1.grad_weights,
        b2c1.grad_bias,
        g_bn1x1_2_gamma,
        g_bn1x1_2_beta,
        b2c2.grad_weights,
        b2c2.grad_bias,
        g_bn3x3_gamma,
        g_bn3x3_beta,
        b3c1.grad_weights,
        b3c1.grad_bias,
        g_bn1x1_3_gamma,
        g_bn1x1_3_beta,
        b3c2.grad_weights,
        b3c2.grad_bias,
        g_bn5x5_gamma,
        g_bn5x5_beta,
        b4c.grad_weights,
        b4c.grad_bias,
        g_bn1x1_4_gamma,
        g_bn1x1_4_beta,
    )


struct GoogLeNet:
    """GoogLeNet (Inception-v1) for CIFAR-10.

    Architecture:
        - Input: 32×32×3
        - Initial conv block
        - 9 Inception modules (3a, 3b, 4a-e, 5a, 5b)
        - Global average pooling
        - Dropout + FC layer
        - Output: 10 classes

    Total parameters: ~6.8M.
    """

    # Initial convolution block
    var initial_conv_weights: AnyTensor
    var initial_conv_bias: AnyTensor
    var initial_bn_gamma: AnyTensor
    var initial_bn_beta: AnyTensor
    var initial_bn_running_mean: AnyTensor
    var initial_bn_running_var: AnyTensor

    # Inception modules
    var inception_3a: InceptionModule
    var inception_3b: InceptionModule
    var inception_4a: InceptionModule
    var inception_4b: InceptionModule
    var inception_4c: InceptionModule
    var inception_4d: InceptionModule
    var inception_4e: InceptionModule
    var inception_5a: InceptionModule
    var inception_5b: InceptionModule

    # Final classifier
    var fc_weights: AnyTensor
    var fc_bias: AnyTensor

    def __init__(out self, num_classes: Int = 10) raises:
        """Initialize GoogLeNet model.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10).
        """
        # Initial convolution: 3×3, 64 filters
        var initial_conv_weights_shape: List[Int] = [64, 3, 3, 3]
        self.initial_conv_weights = kaiming_normal(
            fan_in=3 * 9,
            fan_out=64,
            shape=initial_conv_weights_shape,
        )
        var initial_bias_shape: List[Int] = [64]
        self.initial_conv_bias = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_gamma = constant(initial_bias_shape, 1.0)
        self.initial_bn_beta = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_running_mean = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_running_var = constant(initial_bias_shape, 1.0)

        # Inception 3a: input 64, output 256 (64 + 128 + 32 + 32)
        self.inception_3a = InceptionModule(
            in_channels=64,
            out_1x1=64,
            reduce_3x3=96,
            out_3x3=128,
            reduce_5x5=16,
            out_5x5=32,
            pool_proj=32,
        )

        # Inception 3b: input 256, output 480 (128 + 192 + 96 + 64)
        self.inception_3b = InceptionModule(
            in_channels=256,
            out_1x1=128,
            reduce_3x3=128,
            out_3x3=192,
            reduce_5x5=32,
            out_5x5=96,
            pool_proj=64,
        )

        # Inception 4a: input 480, output 512 (192 + 208 + 48 + 64)
        self.inception_4a = InceptionModule(
            in_channels=480,
            out_1x1=192,
            reduce_3x3=96,
            out_3x3=208,
            reduce_5x5=16,
            out_5x5=48,
            pool_proj=64,
        )

        # Inception 4b: input 512, output 512 (160 + 224 + 64 + 64)
        self.inception_4b = InceptionModule(
            in_channels=512,
            out_1x1=160,
            reduce_3x3=112,
            out_3x3=224,
            reduce_5x5=24,
            out_5x5=64,
            pool_proj=64,
        )

        # Inception 4c: input 512, output 512 (128 + 256 + 64 + 64)
        self.inception_4c = InceptionModule(
            in_channels=512,
            out_1x1=128,
            reduce_3x3=128,
            out_3x3=256,
            reduce_5x5=24,
            out_5x5=64,
            pool_proj=64,
        )

        # Inception 4d: input 512, output 528 (112 + 288 + 64 + 64)
        self.inception_4d = InceptionModule(
            in_channels=512,
            out_1x1=112,
            reduce_3x3=144,
            out_3x3=288,
            reduce_5x5=32,
            out_5x5=64,
            pool_proj=64,
        )

        # Inception 4e: input 528, output 832 (256 + 320 + 128 + 128)
        self.inception_4e = InceptionModule(
            in_channels=528,
            out_1x1=256,
            reduce_3x3=160,
            out_3x3=320,
            reduce_5x5=32,
            out_5x5=128,
            pool_proj=128,
        )

        # Inception 5a: input 832, output 832 (256 + 320 + 128 + 128)
        self.inception_5a = InceptionModule(
            in_channels=832,
            out_1x1=256,
            reduce_3x3=160,
            out_3x3=320,
            reduce_5x5=32,
            out_5x5=128,
            pool_proj=128,
        )

        # Inception 5b: input 832, output 1024 (384 + 384 + 128 + 128)
        self.inception_5b = InceptionModule(
            in_channels=832,
            out_1x1=384,
            reduce_3x3=192,
            out_3x3=384,
            reduce_5x5=48,
            out_5x5=128,
            pool_proj=128,
        )

        # Final FC layer: 1024 → num_classes
        var fc_weights_shape: List[Int] = [num_classes, 1024]
        self.fc_weights = xavier_normal(
            fan_in=1024,
            fan_out=num_classes,
            shape=fc_weights_shape,
        )
        var fc_bias_shape: List[Int] = [num_classes]
        self.fc_bias = zeros(fc_bias_shape, DType.float32)

    def forward(
        mut self, x: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass through GoogLeNet.

        Args:
            x: Input tensor (batch, 3, 32, 32).
            training: Training mode flag (affects batch norm and dropout).

        Returns:
            Logits tensor (batch, num_classes).
        """
        # Initial convolution block
        var out = conv2d(
            x,
            self.initial_conv_weights,
            self.initial_conv_bias,
            stride=1,
            padding=1,
        )
        var initial_bn_result = batch_norm2d(
            out,
            self.initial_bn_gamma,
            self.initial_bn_beta,
            self.initial_bn_running_mean,
            self.initial_bn_running_var,
            training,
        )
        out = initial_bn_result[0]
        self.initial_bn_running_mean = initial_bn_result[1]
        self.initial_bn_running_var = initial_bn_result[2]
        out = relu(out)
        # Shape: (batch, 64, 32, 32)

        # MaxPool 3×3, stride=2
        out = maxpool2d(out, kernel_size=3, stride=2, padding=1)
        # Shape: (batch, 64, 16, 16)

        # Inception 3a
        out = self.inception_3a.forward(out, training)
        # Shape: (batch, 256, 16, 16)

        # Inception 3b
        out = self.inception_3b.forward(out, training)
        # Shape: (batch, 480, 16, 16)

        # MaxPool 3×3, stride=2
        out = maxpool2d(out, kernel_size=3, stride=2, padding=1)
        # Shape: (batch, 480, 8, 8)

        # Inception 4a
        out = self.inception_4a.forward(out, training)
        # Shape: (batch, 512, 8, 8)

        # Inception 4b
        out = self.inception_4b.forward(out, training)
        # Shape: (batch, 512, 8, 8)

        # Inception 4c
        out = self.inception_4c.forward(out, training)
        # Shape: (batch, 512, 8, 8)

        # Inception 4d
        out = self.inception_4d.forward(out, training)
        # Shape: (batch, 528, 8, 8)

        # Inception 4e
        out = self.inception_4e.forward(out, training)
        # Shape: (batch, 832, 8, 8)

        # MaxPool 3×3, stride=2
        out = maxpool2d(out, kernel_size=3, stride=2, padding=1)
        # Shape: (batch, 832, 4, 4)

        # Inception 5a
        out = self.inception_5a.forward(out, training)
        # Shape: (batch, 832, 4, 4)

        # Inception 5b
        out = self.inception_5b.forward(out, training)
        # Shape: (batch, 1024, 4, 4)

        # Global average pooling
        out = global_avgpool2d(out)
        # Shape: (batch, 1024, 1, 1)

        # Flatten
        var batch_size = out.shape()[0]
        var channels = out.shape()[1]
        var flattened_shape: List[Int] = [batch_size, channels]
        var flattened = zeros(flattened_shape, out.dtype())
        var flattened_data = flattened._data.bitcast[Float32]()
        var out_data = out._data.bitcast[Float32]()

        for b in range(batch_size):
            for c in range(channels):
                flattened_data[b * channels + c] = out_data[
                    ((b * channels + c) * 1) + 0
                ]

        # Dropout (p=0.4)
        if training:
            flattened, _ = dropout(flattened, p=0.4, training=True)

        # Final FC layer
        var logits = linear(flattened, self.fc_weights, self.fc_bias)
        # Shape: (batch, num_classes)

        return logits

    def load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing saved weight files.
        """
        # Initial convolution block
        self.initial_conv_weights = load_tensor(
            weights_dir + "/initial_conv_weights.bin"
        )
        self.initial_conv_bias = load_tensor(
            weights_dir + "/initial_conv_bias.bin"
        )
        self.initial_bn_gamma = load_tensor(
            weights_dir + "/initial_bn_gamma.bin"
        )
        self.initial_bn_beta = load_tensor(weights_dir + "/initial_bn_beta.bin")

        # Load all Inception module weights
        self.inception_3a.load_weights(weights_dir, "inception_3a")
        self.inception_3b.load_weights(weights_dir, "inception_3b")
        self.inception_4a.load_weights(weights_dir, "inception_4a")
        self.inception_4b.load_weights(weights_dir, "inception_4b")
        self.inception_4c.load_weights(weights_dir, "inception_4c")
        self.inception_4d.load_weights(weights_dir, "inception_4d")
        self.inception_4e.load_weights(weights_dir, "inception_4e")
        self.inception_5a.load_weights(weights_dir, "inception_5a")
        self.inception_5b.load_weights(weights_dir, "inception_5b")

        # Final FC layer
        self.fc_weights = load_tensor(weights_dir + "/fc_weights.bin")
        self.fc_bias = load_tensor(weights_dir + "/fc_bias.bin")

    def save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files.

        Note:
            Running stats (running_mean, running_var) are not saved as they
            are recomputed during training from scratch.
        """
        # Initial convolution block
        save_tensor(
            self.initial_conv_weights,
            weights_dir + "/initial_conv_weights.bin",
            "initial_conv_weights",
        )
        save_tensor(
            self.initial_conv_bias,
            weights_dir + "/initial_conv_bias.bin",
            "initial_conv_bias",
        )
        save_tensor(
            self.initial_bn_gamma,
            weights_dir + "/initial_bn_gamma.bin",
            "initial_bn_gamma",
        )
        save_tensor(
            self.initial_bn_beta,
            weights_dir + "/initial_bn_beta.bin",
            "initial_bn_beta",
        )

        # Save all Inception module weights
        self.inception_3a.save_weights(weights_dir, "inception_3a")
        self.inception_3b.save_weights(weights_dir, "inception_3b")
        self.inception_4a.save_weights(weights_dir, "inception_4a")
        self.inception_4b.save_weights(weights_dir, "inception_4b")
        self.inception_4c.save_weights(weights_dir, "inception_4c")
        self.inception_4d.save_weights(weights_dir, "inception_4d")
        self.inception_4e.save_weights(weights_dir, "inception_4e")
        self.inception_5a.save_weights(weights_dir, "inception_5a")
        self.inception_5b.save_weights(weights_dir, "inception_5b")

        # Final FC layer
        save_tensor(
            self.fc_weights, weights_dir + "/fc_weights.bin", "fc_weights"
        )
        save_tensor(self.fc_bias, weights_dir + "/fc_bias.bin", "fc_bias")


def main():
    """Main function for build verification only."""
    print("GoogLeNet model module - import this module, do not run directly")
