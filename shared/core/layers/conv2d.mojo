"""Conv2D (2D convolutional) layer with parameter management.

This module provides a Conv2dLayer wrapper class that manages weights and biases
for 2D convolution operations. The layer wraps the pure functional conv2d function
and maintains learnable parameters.

Key components:
- Conv2dLayer: 2D convolutional layer with learnable weights and bias
  Implements: y = conv2d(x, weight, bias, stride, padding)
"""

from ..any_tensor import AnyTensor, zeros, randn, zeros_like
from ..initializers import kaiming_uniform
from ..conv import conv2d, conv2d_backward
from shared.tensor.tensor import Tensor


struct Conv2dLayer[dtype: DType = DType.float32](Copyable, Movable):
    """2D Convolutional layer: y = conv2d(x, weight, bias, stride, padding).

    A 2D convolutional neural network layer that applies learnable filters
    to spatially structured inputs (images).

    Parameters:
        dtype: Data type for weight and bias (default: float32).

    Attributes:
        weight: Filter weights of shape (out_ch, in_ch, kernel_h, kernel_w).
        bias: Bias vector of shape (out_channels,).
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_h: Kernel height.
        kernel_w: Kernel width.
        stride: Stride for convolution.
        padding: Zero-padding added to input.
    """

    var weight: AnyTensor
    """Filter weights of shape (out_ch, in_ch, kernel_h, kernel_w)."""
    var bias: AnyTensor
    """Bias vector of shape (out_channels,)."""
    var in_channels: Int
    """Number of input channels."""
    var out_channels: Int
    """Number of output channels (filters)."""
    var kernel_h: Int
    """Kernel height."""
    var kernel_w: Int
    """Kernel width."""
    var stride: Int
    """Stride for convolution."""
    var padding: Int
    """Zero-padding added to input."""

    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_h: Int,
        kernel_w: Int,
        stride: Int = 1,
        padding: Int = 0,
    ) raises:
        """Initialize Conv2D layer with He/Kaiming weights and zero bias.

        Uses He initialization for weights. Bias is initialized to zero.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_h: Height of convolutional kernel.
            kernel_w: Width of convolutional kernel.
            stride: Stride for convolution (default: 1).
            padding: Zero-padding added to input (default: 0).

        Raises:
            Error if tensor creation fails.

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3, stride=1, padding=1)
            ```
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding

        # Initialize weights with Kaiming/He initialization
        var weight_shape = List[Int]()
        weight_shape.append(out_channels)
        weight_shape.append(in_channels)
        weight_shape.append(kernel_h)
        weight_shape.append(kernel_w)

        var fan_in = in_channels * kernel_h * kernel_w
        var fan_out = out_channels * kernel_h * kernel_w
        self.weight = kaiming_uniform(
            fan_in, fan_out, weight_shape, "fan_in", Self.dtype
        )

        # Initialize bias to zeros
        var bias_shape = List[Int]()
        bias_shape.append(out_channels)
        self.bias = zeros(bias_shape, Self.dtype)

    fn forward(self, input: Tensor[Self.dtype]) raises -> Tensor[Self.dtype]:
        """Forward pass: y = conv2d(x, weight, bias, stride, padding).

        Applies the learned convolutional filters to the input.

        Args:
            input: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, out_h, out_w).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3, stride=1, padding=1)
            var input_t = Tensor[DType.float32]([1, 3, 32, 32])
            var output = layer.forward(input_t)
            ```
        """
        return conv2d(
            input.as_any(), self.weight, self.bias, self.stride, self.padding
        ).as_tensor[Self.dtype]()

    fn backward(
        self, grad_output: Tensor[Self.dtype], input: Tensor[Self.dtype]
    ) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
        """Backward pass: compute gradients w.r.t. input, weight, and bias.

        Args:
            grad_output: Gradient w.r.t. output, shape
                (batch, out_channels, out_H, out_W).
            input: Input from forward pass, shape
                (batch, in_channels, in_H, in_W).

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3)
            var input_t = Tensor[DType.float32]([2, 3, 32, 32])
            var output = layer.forward(input_t)
            var grad = Tensor[DType.float32](output.shape())
            var (gi, gw, gb) = layer.backward(grad, input_t)
            ```
        """
        var result = conv2d_backward(
            grad_output.as_any(),
            input.as_any(),
            self.weight,
            self.stride,
            self.padding,
        )
        return (result.grad_input, result.grad_weights, result.grad_bias)

    fn parameters(self) raises -> List[AnyTensor]:
        """Get list of trainable parameters.

        Returns weight and bias directly. No byte-level copy needed since
        fields are already AnyTensor with the correct dtype.

        Returns:
            List containing [weight, bias] tensors that need gradients

        Raises:
            Error if tensor copying fails

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3)
            var params = layer.parameters()
            # params[0] is weight, params[1] is bias
            ```
        """
        var params = List[AnyTensor]()
        params.append(self.weight)
        params.append(self.bias)
        return params^
