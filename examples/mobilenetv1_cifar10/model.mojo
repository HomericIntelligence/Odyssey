"""MobileNetV1 Model for CIFAR-10.

This module implements the MobileNetV1 architecture adapted for CIFAR-10.

Architecture:
    - 28 layers deep (13 depthwise separable blocks + initial + classifier)
    - Input: 32×32×3 RGB images
    - Output: 10 classes
    - ~4.2M parameters

Key Innovation:
    - Depthwise separable convolutions: 8-9× fewer operations
    - Depthwise: Spatial filtering per channel (no cross-channel mixing)
    - Pointwise: Channel mixing with 1×1 convolutions
    - Much more efficient than standard convolutions

References:
    Howard et al. (2017) - MobileNets: Efficient Convolutional Neural Networks
    https://arxiv.org/abs/1704.04861
"""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros
from projectodyssey.core import (
    conv2d,
    batch_norm2d,
    relu,
    global_avgpool2d,
    kaiming_normal,
    xavier_normal,
    constant,
)
from projectodyssey.core.linear import linear
from projectodyssey.core.shape import conv2d_output_shape


def depthwise_conv2d(
    x: AnyTensor,
    weights: AnyTensor,
    bias: AnyTensor,
    stride: Int = 1,
    padding: Int = 1,
) raises -> AnyTensor:
    """Depthwise convolution: Apply one filter per input channel.

    Unlike standard convolution that mixes all input channels, depthwise
    convolution applies a separate filter to each input channel independently.

    Args:
        x: Input tensor (batch, channels, height, width).
        weights: Depthwise filters (channels, 1, kernel_h, kernel_w).
        bias: Bias per channel (channels,).
        stride: Convolution stride.
        padding: Padding size.

    Returns:
        Output tensor (batch, channels, out_h, out_w)

    Note:
        This is a naive implementation. For production, this should be
        implemented in the shared library with SIMD optimizations.
    """
    var batch_size = x.shape()[0]
    var channels = x.shape()[1]
    var height = x.shape()[2]
    var width = x.shape()[3]
    var kernel_h = weights.shape()[2]
    var kernel_w = weights.shape()[3]

    # Calculate output dimensions using shared shape function
    var out_h, out_w = conv2d_output_shape(
        height, width, kernel_h, kernel_w, stride, padding
    )

    # Create output tensor
    var output_shape: List[Int] = [batch_size, channels, out_h, out_w]
    var output = zeros(output_shape, x.dtype())

    # Process each channel independently
    var x_data = x._data.bitcast[Float32]()
    var weights_data = weights._data.bitcast[Float32]()
    var bias_data = bias._data.bitcast[Float32]()
    var output_data = output._data.bitcast[Float32]()

    # For each batch and channel, apply the depthwise filter
    for b in range(batch_size):
        for c in range(channels):
            # Extract single channel
            var channel_input_shape: List[Int] = [1, 1, height, width]
            var channel_input = zeros(channel_input_shape, x.dtype())
            var channel_input_data = channel_input._data.bitcast[Float32]()

            # Copy channel data
            for h in range(height):
                for w in range(width):
                    var src_idx = ((b * channels + c) * height + h) * width + w
                    var dst_idx = h * width + w
                    channel_input_data[dst_idx] = x_data[src_idx]

            # Extract single filter for this channel
            var channel_filter_shape: List[Int] = [1, 1, kernel_h, kernel_w]
            var channel_filter = zeros(channel_filter_shape, weights.dtype())
            var channel_filter_data = channel_filter._data.bitcast[Float32]()

            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    var filter_idx = (
                        (c * 1 + 0) * kernel_h + kh
                    ) * kernel_w + kw
                    var dst_idx = kh * kernel_w + kw
                    channel_filter_data[dst_idx] = weights_data[filter_idx]

            # Create single-element bias
            var channel_bias_shape: List[Int] = [1]
            var channel_bias = zeros(channel_bias_shape, bias.dtype())
            var channel_bias_data = channel_bias._data.bitcast[Float32]()
            channel_bias_data[0] = bias_data[c]

            # Apply 2D convolution for this channel
            var channel_output = conv2d(
                channel_input, channel_filter, channel_bias, stride, padding
            )

            # Copy result back to output
            var channel_output_data = channel_output._data.bitcast[Float32]()
            for h in range(out_h):
                for w in range(out_w):
                    var src_idx = h * out_w + w
                    var dst_idx = ((b * channels + c) * out_h + h) * out_w + w
                    output_data[dst_idx] = channel_output_data[src_idx]

    return output


struct DepthwiseSeparableBlock:
    """Depthwise separable convolution block.

    Architecture:
        1. Depthwise conv (3×3, per-channel filtering)
        2. Batch norm + ReLU
        3. Pointwise conv (1×1, channel mixing)
        4. Batch norm + ReLU

    Note:
        Depthwise: weights (in_channels, 1, 3, 3), bias (in_channels,), BN params.
        Pointwise: weights (out_channels, in_channels, 1, 1), bias (out_channels,), BN params.
    """

    # Depthwise convolution (3×3, per-channel)
    var dw_weights: AnyTensor  # (in_channels, 1, 3, 3)
    var dw_bias: AnyTensor  # (in_channels,)
    var dw_bn_gamma: AnyTensor
    var dw_bn_beta: AnyTensor
    var dw_bn_running_mean: AnyTensor
    var dw_bn_running_var: AnyTensor

    # Pointwise convolution (1×1, channel mixing)
    var pw_weights: AnyTensor  # (out_channels, in_channels, 1, 1)
    var pw_bias: AnyTensor  # (out_channels,)
    var pw_bn_gamma: AnyTensor
    var pw_bn_beta: AnyTensor
    var pw_bn_running_mean: AnyTensor
    var pw_bn_running_var: AnyTensor

    def __init__(
        out self, in_channels: Int, out_channels: Int, stride: Int = 1
    ) raises:
        """Initialize depthwise separable block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for depthwise convolution (1 or 2).
        """
        # Depthwise convolution weights (one 3×3 filter per channel)
        var dw_weights_shape: List[Int] = [in_channels, 1, 3, 3]
        self.dw_weights = kaiming_normal(
            fan_in=9, fan_out=in_channels, shape=dw_weights_shape
        )  # 3×3 kernel
        var dw_bias_shape: List[Int] = [in_channels]
        self.dw_bias = zeros(dw_bias_shape, DType.float32)
        self.dw_bn_gamma = constant(dw_bias_shape, 1.0)
        self.dw_bn_beta = zeros(dw_bias_shape, DType.float32)
        self.dw_bn_running_mean = zeros(dw_bias_shape, DType.float32)
        self.dw_bn_running_var = constant(dw_bias_shape, 1.0)

        # Pointwise convolution weights (1×1, channel mixing)
        var pw_weights_shape: List[Int] = [out_channels, in_channels, 1, 1]
        self.pw_weights = xavier_normal(
            fan_in=in_channels,
            fan_out=out_channels,
            shape=pw_weights_shape,
        )
        var pw_bias_shape: List[Int] = [out_channels]
        self.pw_bias = zeros(pw_bias_shape, DType.float32)
        self.pw_bn_gamma = constant(pw_bias_shape, 1.0)
        self.pw_bn_beta = zeros(pw_bias_shape, DType.float32)
        self.pw_bn_running_mean = zeros(pw_bias_shape, DType.float32)
        self.pw_bn_running_var = constant(pw_bias_shape, 1.0)

    def forward(
        mut self, x: AnyTensor, stride: Int, training: Bool
    ) raises -> AnyTensor:
        """Forward pass through depthwise separable block.

        Args:
            x: Input tensor (batch, in_channels, H, W).
            stride: Stride for depthwise convolution.
            training: Training mode flag (affects batch norm).

        Returns:
            Output tensor (batch, out_channels, H/stride, W/stride).
        """
        # Depthwise convolution (spatial filtering, per-channel)
        var out = depthwise_conv2d(
            x, self.dw_weights, self.dw_bias, stride=stride, padding=1
        )
        out, _, _ = batch_norm2d(
            out,
            self.dw_bn_gamma,
            self.dw_bn_beta,
            self.dw_bn_running_mean,
            self.dw_bn_running_var,
            training,
        )
        out = relu(out)

        # Pointwise convolution (channel mixing, 1×1)
        out = conv2d(out, self.pw_weights, self.pw_bias, stride=1, padding=0)
        out, _, _ = batch_norm2d(
            out,
            self.pw_bn_gamma,
            self.pw_bn_beta,
            self.pw_bn_running_mean,
            self.pw_bn_running_var,
            training,
        )
        out = relu(out)

        return out


struct MobileNetV1:
    """MobileNetV1 for CIFAR-10.

    Architecture:
        - Input: 32×32×3
        - Initial standard conv (32 filters, stride=2)
        - 13 depthwise separable blocks
        - Global average pooling
        - FC layer
        - Output: 10 classes

    Total parameters: ~4.2M.
    """

    # Initial standard convolution
    var initial_conv_weights: AnyTensor
    var initial_conv_bias: AnyTensor
    var initial_bn_gamma: AnyTensor
    var initial_bn_beta: AnyTensor
    var initial_bn_running_mean: AnyTensor
    var initial_bn_running_var: AnyTensor

    # Depthwise separable blocks (13 total)
    var ds_block_1: DepthwiseSeparableBlock  # 32 → 64, stride=1
    var ds_block_2: DepthwiseSeparableBlock  # 64 → 128, stride=2
    var ds_block_3: DepthwiseSeparableBlock  # 128 → 128, stride=1
    var ds_block_4: DepthwiseSeparableBlock  # 128 → 256, stride=2
    var ds_block_5: DepthwiseSeparableBlock  # 256 → 256, stride=1
    var ds_block_6: DepthwiseSeparableBlock  # 256 → 512, stride=2
    var ds_block_7: DepthwiseSeparableBlock  # 512 → 512, stride=1
    var ds_block_8: DepthwiseSeparableBlock  # 512 → 512, stride=1
    var ds_block_9: DepthwiseSeparableBlock  # 512 → 512, stride=1
    var ds_block_10: DepthwiseSeparableBlock  # 512 → 512, stride=1
    var ds_block_11: DepthwiseSeparableBlock  # 512 → 512, stride=1
    var ds_block_12: DepthwiseSeparableBlock  # 512 → 1024, stride=2
    var ds_block_13: DepthwiseSeparableBlock  # 1024 → 1024, stride=1

    # Final classifier
    var fc_weights: AnyTensor
    var fc_bias: AnyTensor

    def __init__(out self, num_classes: Int = 10) raises:
        """Initialize MobileNetV1 model.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10).
        """
        # Initial standard convolution: 3×3, 32 filters, stride=2
        var initial_conv_weights_shape: List[Int] = [32, 3, 3, 3]
        self.initial_conv_weights = kaiming_normal(
            fan_in=3 * 9,
            fan_out=32,
            shape=initial_conv_weights_shape,
        )
        var initial_bias_shape: List[Int] = [32]
        self.initial_conv_bias = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_gamma = constant(initial_bias_shape, 1.0)
        self.initial_bn_beta = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_running_mean = zeros(initial_bias_shape, DType.float32)
        self.initial_bn_running_var = constant(initial_bias_shape, 1.0)

        # Depthwise separable blocks
        # Channel progression: 32 → 64 → 128 → 256 → 512 → 1024
        self.ds_block_1 = DepthwiseSeparableBlock(32, 64, stride=1)
        self.ds_block_2 = DepthwiseSeparableBlock(64, 128, stride=2)
        self.ds_block_3 = DepthwiseSeparableBlock(128, 128, stride=1)
        self.ds_block_4 = DepthwiseSeparableBlock(128, 256, stride=2)
        self.ds_block_5 = DepthwiseSeparableBlock(256, 256, stride=1)
        self.ds_block_6 = DepthwiseSeparableBlock(256, 512, stride=2)
        self.ds_block_7 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.ds_block_8 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.ds_block_9 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.ds_block_10 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.ds_block_11 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.ds_block_12 = DepthwiseSeparableBlock(512, 1024, stride=2)
        self.ds_block_13 = DepthwiseSeparableBlock(1024, 1024, stride=1)

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
        """Forward pass through MobileNetV1.

        Args:
            x: Input tensor (batch, 3, 32, 32).
            training: Training mode flag (affects batch norm).

        Returns:
            Logits tensor (batch, num_classes).
        """
        # Initial standard convolution
        var out = conv2d(
            x,
            self.initial_conv_weights,
            self.initial_conv_bias,
            stride=2,
            padding=1,
        )
        out, _, _ = batch_norm2d(
            out,
            self.initial_bn_gamma,
            self.initial_bn_beta,
            self.initial_bn_running_mean,
            self.initial_bn_running_var,
            training,
        )
        out = relu(out)
        # Shape: (batch, 32, 16, 16)

        # Depthwise separable blocks
        out = self.ds_block_1.forward(out, stride=1, training=training)
        # Shape: (batch, 64, 16, 16)

        out = self.ds_block_2.forward(out, stride=2, training=training)
        # Shape: (batch, 128, 8, 8)

        out = self.ds_block_3.forward(out, stride=1, training=training)
        # Shape: (batch, 128, 8, 8)

        out = self.ds_block_4.forward(out, stride=2, training=training)
        # Shape: (batch, 256, 4, 4)

        out = self.ds_block_5.forward(out, stride=1, training=training)
        # Shape: (batch, 256, 4, 4)

        out = self.ds_block_6.forward(out, stride=2, training=training)
        # Shape: (batch, 512, 2, 2)

        out = self.ds_block_7.forward(out, stride=1, training=training)
        out = self.ds_block_8.forward(out, stride=1, training=training)
        out = self.ds_block_9.forward(out, stride=1, training=training)
        out = self.ds_block_10.forward(out, stride=1, training=training)
        out = self.ds_block_11.forward(out, stride=1, training=training)
        # Shape: (batch, 512, 2, 2)

        out = self.ds_block_12.forward(out, stride=2, training=training)
        # Shape: (batch, 1024, 1, 1)

        out = self.ds_block_13.forward(out, stride=1, training=training)
        # Shape: (batch, 1024, 1, 1)

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

        # Final FC layer
        var logits = linear(flattened, self.fc_weights, self.fc_bias)
        # Shape: (batch, num_classes)

        return logits

    def load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing saved weight files.

        Raises:
            Error: If weight files are missing or have incompatible shapes

        Note:
            Weights are loaded from individual .weights files in the directory.
            Expected files:
            - initial_conv_weights.weights
            - initial_bn_gamma.weights, initial_bn_beta.weights
            - ds_block_N_{dw,pw}_{weights,bias,bn_*}.weights for each block
            - fc_weights.weights, fc_bias.weights.
        """
        from projectodyssey.training.model_utils import (
            load_model_weights,
            get_model_parameter_names,
        )

        # Get standard parameter names for MobileNetV1
        var param_names = get_model_parameter_names("mobilenetv1")

        # Create empty list for loaded parameters
        var loaded_params: List[AnyTensor] = []

        # Load using shared utility
        load_model_weights(loaded_params, weights_dir, param_names)

        # Validate we loaded the correct number of parameters
        if len(loaded_params) < 100:
            raise Error(
                "Invalid checkpoint: expected ~156 parameters for MobileNetV1,"
                " got "
                + String(len(loaded_params))
            )

        # Assign loaded parameters to model fields (matching order in get_model_parameter_names)
        var idx = 0

        # Initial convolution
        self.initial_conv_weights = loaded_params[idx]
        idx += 1
        self.initial_conv_bias = loaded_params[idx]
        idx += 1
        self.initial_bn_gamma = loaded_params[idx]
        idx += 1
        self.initial_bn_beta = loaded_params[idx]
        idx += 1
        self.initial_bn_running_mean = loaded_params[idx]
        idx += 1
        self.initial_bn_running_var = loaded_params[idx]
        idx += 1

        # Depthwise separable blocks (13 blocks × 12 params per block)
        # Each block has: dw_weights, dw_bias, dw_bn_gamma, dw_bn_beta, dw_bn_running_mean, dw_bn_running_var
        #                pw_weights, pw_bias, pw_bn_gamma, pw_bn_beta, pw_bn_running_mean, pw_bn_running_var

        # Block 1
        self.ds_block_1.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_1.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_1.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_1.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_1.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_1.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_1.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 2
        self.ds_block_2.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_2.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_2.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_2.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_2.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_2.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_2.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 3
        self.ds_block_3.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_3.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_3.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_3.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_3.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_3.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_3.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 4
        self.ds_block_4.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_4.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_4.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_4.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_4.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_4.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_4.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 5
        self.ds_block_5.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_5.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_5.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_5.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_5.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_5.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_5.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 6
        self.ds_block_6.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_6.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_6.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_6.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_6.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_6.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_6.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 7
        self.ds_block_7.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_7.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_7.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_7.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_7.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_7.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_7.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 8
        self.ds_block_8.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_8.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_8.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_8.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_8.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_8.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_8.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 9
        self.ds_block_9.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_9.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_9.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_9.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_9.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_9.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_9.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 10
        self.ds_block_10.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_10.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_10.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_10.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_10.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_10.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_10.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 11
        self.ds_block_11.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_11.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_11.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_11.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_11.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_11.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_11.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 12
        self.ds_block_12.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_12.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_12.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_12.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_12.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_12.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_12.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Block 13
        self.ds_block_13.dw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_13.dw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_13.dw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_13.dw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_13.dw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_13.dw_bn_running_var = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_weights = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_bias = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_bn_gamma = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_bn_beta = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_bn_running_mean = loaded_params[idx]
        idx += 1
        self.ds_block_13.pw_bn_running_var = loaded_params[idx]
        idx += 1

        # Final fully connected layer
        self.fc_weights = loaded_params[idx]
        idx += 1
        self.fc_bias = loaded_params[idx]
        idx += 1

    def save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files.

        Raises:
            Error: If directory creation or file write fails

        Note:
            Creates directory if it doesn't exist. Each parameter is saved as:
            - <param_name>.weights

            Total parameters saved: ~156 (6 initial conv, 13 blocks × 12 params, 2 fc).
        """
        from projectodyssey.training.model_utils import (
            save_model_weights,
            get_model_parameter_names,
        )

        # Collect all parameters in order
        var parameters: List[AnyTensor] = []

        # Initial convolution parameters
        parameters.append(self.initial_conv_weights)
        parameters.append(self.initial_conv_bias)
        parameters.append(self.initial_bn_gamma)
        parameters.append(self.initial_bn_beta)
        parameters.append(self.initial_bn_running_mean)
        parameters.append(self.initial_bn_running_var)

        # Depthwise separable blocks
        # Block 1
        parameters.append(self.ds_block_1.dw_weights)
        parameters.append(self.ds_block_1.dw_bias)
        parameters.append(self.ds_block_1.dw_bn_gamma)
        parameters.append(self.ds_block_1.dw_bn_beta)
        parameters.append(self.ds_block_1.dw_bn_running_mean)
        parameters.append(self.ds_block_1.dw_bn_running_var)
        parameters.append(self.ds_block_1.pw_weights)
        parameters.append(self.ds_block_1.pw_bias)
        parameters.append(self.ds_block_1.pw_bn_gamma)
        parameters.append(self.ds_block_1.pw_bn_beta)
        parameters.append(self.ds_block_1.pw_bn_running_mean)
        parameters.append(self.ds_block_1.pw_bn_running_var)

        # Block 2
        parameters.append(self.ds_block_2.dw_weights)
        parameters.append(self.ds_block_2.dw_bias)
        parameters.append(self.ds_block_2.dw_bn_gamma)
        parameters.append(self.ds_block_2.dw_bn_beta)
        parameters.append(self.ds_block_2.dw_bn_running_mean)
        parameters.append(self.ds_block_2.dw_bn_running_var)
        parameters.append(self.ds_block_2.pw_weights)
        parameters.append(self.ds_block_2.pw_bias)
        parameters.append(self.ds_block_2.pw_bn_gamma)
        parameters.append(self.ds_block_2.pw_bn_beta)
        parameters.append(self.ds_block_2.pw_bn_running_mean)
        parameters.append(self.ds_block_2.pw_bn_running_var)

        # Block 3
        parameters.append(self.ds_block_3.dw_weights)
        parameters.append(self.ds_block_3.dw_bias)
        parameters.append(self.ds_block_3.dw_bn_gamma)
        parameters.append(self.ds_block_3.dw_bn_beta)
        parameters.append(self.ds_block_3.dw_bn_running_mean)
        parameters.append(self.ds_block_3.dw_bn_running_var)
        parameters.append(self.ds_block_3.pw_weights)
        parameters.append(self.ds_block_3.pw_bias)
        parameters.append(self.ds_block_3.pw_bn_gamma)
        parameters.append(self.ds_block_3.pw_bn_beta)
        parameters.append(self.ds_block_3.pw_bn_running_mean)
        parameters.append(self.ds_block_3.pw_bn_running_var)

        # Block 4
        parameters.append(self.ds_block_4.dw_weights)
        parameters.append(self.ds_block_4.dw_bias)
        parameters.append(self.ds_block_4.dw_bn_gamma)
        parameters.append(self.ds_block_4.dw_bn_beta)
        parameters.append(self.ds_block_4.dw_bn_running_mean)
        parameters.append(self.ds_block_4.dw_bn_running_var)
        parameters.append(self.ds_block_4.pw_weights)
        parameters.append(self.ds_block_4.pw_bias)
        parameters.append(self.ds_block_4.pw_bn_gamma)
        parameters.append(self.ds_block_4.pw_bn_beta)
        parameters.append(self.ds_block_4.pw_bn_running_mean)
        parameters.append(self.ds_block_4.pw_bn_running_var)

        # Block 5
        parameters.append(self.ds_block_5.dw_weights)
        parameters.append(self.ds_block_5.dw_bias)
        parameters.append(self.ds_block_5.dw_bn_gamma)
        parameters.append(self.ds_block_5.dw_bn_beta)
        parameters.append(self.ds_block_5.dw_bn_running_mean)
        parameters.append(self.ds_block_5.dw_bn_running_var)
        parameters.append(self.ds_block_5.pw_weights)
        parameters.append(self.ds_block_5.pw_bias)
        parameters.append(self.ds_block_5.pw_bn_gamma)
        parameters.append(self.ds_block_5.pw_bn_beta)
        parameters.append(self.ds_block_5.pw_bn_running_mean)
        parameters.append(self.ds_block_5.pw_bn_running_var)

        # Block 6
        parameters.append(self.ds_block_6.dw_weights)
        parameters.append(self.ds_block_6.dw_bias)
        parameters.append(self.ds_block_6.dw_bn_gamma)
        parameters.append(self.ds_block_6.dw_bn_beta)
        parameters.append(self.ds_block_6.dw_bn_running_mean)
        parameters.append(self.ds_block_6.dw_bn_running_var)
        parameters.append(self.ds_block_6.pw_weights)
        parameters.append(self.ds_block_6.pw_bias)
        parameters.append(self.ds_block_6.pw_bn_gamma)
        parameters.append(self.ds_block_6.pw_bn_beta)
        parameters.append(self.ds_block_6.pw_bn_running_mean)
        parameters.append(self.ds_block_6.pw_bn_running_var)

        # Block 7
        parameters.append(self.ds_block_7.dw_weights)
        parameters.append(self.ds_block_7.dw_bias)
        parameters.append(self.ds_block_7.dw_bn_gamma)
        parameters.append(self.ds_block_7.dw_bn_beta)
        parameters.append(self.ds_block_7.dw_bn_running_mean)
        parameters.append(self.ds_block_7.dw_bn_running_var)
        parameters.append(self.ds_block_7.pw_weights)
        parameters.append(self.ds_block_7.pw_bias)
        parameters.append(self.ds_block_7.pw_bn_gamma)
        parameters.append(self.ds_block_7.pw_bn_beta)
        parameters.append(self.ds_block_7.pw_bn_running_mean)
        parameters.append(self.ds_block_7.pw_bn_running_var)

        # Block 8
        parameters.append(self.ds_block_8.dw_weights)
        parameters.append(self.ds_block_8.dw_bias)
        parameters.append(self.ds_block_8.dw_bn_gamma)
        parameters.append(self.ds_block_8.dw_bn_beta)
        parameters.append(self.ds_block_8.dw_bn_running_mean)
        parameters.append(self.ds_block_8.dw_bn_running_var)
        parameters.append(self.ds_block_8.pw_weights)
        parameters.append(self.ds_block_8.pw_bias)
        parameters.append(self.ds_block_8.pw_bn_gamma)
        parameters.append(self.ds_block_8.pw_bn_beta)
        parameters.append(self.ds_block_8.pw_bn_running_mean)
        parameters.append(self.ds_block_8.pw_bn_running_var)

        # Block 9
        parameters.append(self.ds_block_9.dw_weights)
        parameters.append(self.ds_block_9.dw_bias)
        parameters.append(self.ds_block_9.dw_bn_gamma)
        parameters.append(self.ds_block_9.dw_bn_beta)
        parameters.append(self.ds_block_9.dw_bn_running_mean)
        parameters.append(self.ds_block_9.dw_bn_running_var)
        parameters.append(self.ds_block_9.pw_weights)
        parameters.append(self.ds_block_9.pw_bias)
        parameters.append(self.ds_block_9.pw_bn_gamma)
        parameters.append(self.ds_block_9.pw_bn_beta)
        parameters.append(self.ds_block_9.pw_bn_running_mean)
        parameters.append(self.ds_block_9.pw_bn_running_var)

        # Block 10
        parameters.append(self.ds_block_10.dw_weights)
        parameters.append(self.ds_block_10.dw_bias)
        parameters.append(self.ds_block_10.dw_bn_gamma)
        parameters.append(self.ds_block_10.dw_bn_beta)
        parameters.append(self.ds_block_10.dw_bn_running_mean)
        parameters.append(self.ds_block_10.dw_bn_running_var)
        parameters.append(self.ds_block_10.pw_weights)
        parameters.append(self.ds_block_10.pw_bias)
        parameters.append(self.ds_block_10.pw_bn_gamma)
        parameters.append(self.ds_block_10.pw_bn_beta)
        parameters.append(self.ds_block_10.pw_bn_running_mean)
        parameters.append(self.ds_block_10.pw_bn_running_var)

        # Block 11
        parameters.append(self.ds_block_11.dw_weights)
        parameters.append(self.ds_block_11.dw_bias)
        parameters.append(self.ds_block_11.dw_bn_gamma)
        parameters.append(self.ds_block_11.dw_bn_beta)
        parameters.append(self.ds_block_11.dw_bn_running_mean)
        parameters.append(self.ds_block_11.dw_bn_running_var)
        parameters.append(self.ds_block_11.pw_weights)
        parameters.append(self.ds_block_11.pw_bias)
        parameters.append(self.ds_block_11.pw_bn_gamma)
        parameters.append(self.ds_block_11.pw_bn_beta)
        parameters.append(self.ds_block_11.pw_bn_running_mean)
        parameters.append(self.ds_block_11.pw_bn_running_var)

        # Block 12
        parameters.append(self.ds_block_12.dw_weights)
        parameters.append(self.ds_block_12.dw_bias)
        parameters.append(self.ds_block_12.dw_bn_gamma)
        parameters.append(self.ds_block_12.dw_bn_beta)
        parameters.append(self.ds_block_12.dw_bn_running_mean)
        parameters.append(self.ds_block_12.dw_bn_running_var)
        parameters.append(self.ds_block_12.pw_weights)
        parameters.append(self.ds_block_12.pw_bias)
        parameters.append(self.ds_block_12.pw_bn_gamma)
        parameters.append(self.ds_block_12.pw_bn_beta)
        parameters.append(self.ds_block_12.pw_bn_running_mean)
        parameters.append(self.ds_block_12.pw_bn_running_var)

        # Block 13
        parameters.append(self.ds_block_13.dw_weights)
        parameters.append(self.ds_block_13.dw_bias)
        parameters.append(self.ds_block_13.dw_bn_gamma)
        parameters.append(self.ds_block_13.dw_bn_beta)
        parameters.append(self.ds_block_13.dw_bn_running_mean)
        parameters.append(self.ds_block_13.dw_bn_running_var)
        parameters.append(self.ds_block_13.pw_weights)
        parameters.append(self.ds_block_13.pw_bias)
        parameters.append(self.ds_block_13.pw_bn_gamma)
        parameters.append(self.ds_block_13.pw_bn_beta)
        parameters.append(self.ds_block_13.pw_bn_running_mean)
        parameters.append(self.ds_block_13.pw_bn_running_var)

        # Final fully connected layer
        parameters.append(self.fc_weights)
        parameters.append(self.fc_bias)

        # Get standard parameter names
        var param_names = get_model_parameter_names("mobilenetv1")

        # Save using shared utility
        save_model_weights(parameters, weights_dir, param_names)
