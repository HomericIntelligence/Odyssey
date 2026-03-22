"""Conv2D (2D convolutional) layer with parameter management.

This module provides a Conv2dLayer wrapper class that manages weights and biases
for 2D convolution operations. The layer wraps the pure functional conv2d function
and maintains learnable parameters.

Key components:
- Conv2dLayer: 2D convolutional layer with learnable weights and bias
  Implements: y = conv2d(x, weight, bias, stride, padding)
"""

from ..extensor import ExTensor, zeros, randn, zeros_like
from ..initializers import kaiming_uniform
from ..conv import conv2d, conv2d_backward
from shared.tensor.tensor import Tensor


struct Conv2dLayer[dtype: DType = DType.float32](Copyable, Movable):
    """2D Convolutional layer parameterized on dtype.

    y = conv2d(x, weight, bias, stride, padding).

    A 2D convolutional neural network layer that applies learnable filters
    to spatially structured inputs (images).

    Parameters:
        dtype: The data type for weight and bias tensors (default: float32).

    Attributes:
        weight: Filter weights of shape (out_channels, in_channels, kernel_h, kernel_w).
        bias: Bias vector of shape (out_channels,).
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_h: Kernel height.
        kernel_w: Kernel width.
        stride: Stride for convolution.
        padding: Zero-padding added to input.
    """

    var weight: Tensor[dtype]
    """Filter weights of shape (out_channels, in_channels, kernel_h, kernel_w)."""
    var bias: Tensor[dtype]
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

        Uses He initialization for weights (scaled by sqrt(2 / (in_channels * kH * kW))).
        Bias is initialized to zero.

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
            # 3x3 convolution: 3 input channels -> 16 output channels
            var layer = Conv2dLayer(3, 16, 3, 3, stride=1, padding=1)

            # FP16 convolution
            var layer16 = Conv2dLayer[DType.float16](3, 16, 3, 3)
            ```
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding

        # Initialize weights with Kaiming/He initialization
        # Shape: (out_channels, in_channels, kernel_h, kernel_w)
        var weight_shape = List[Int]()
        weight_shape.append(out_channels)
        weight_shape.append(in_channels)
        weight_shape.append(kernel_h)
        weight_shape.append(kernel_w)

        # Fan-in for conv2d: in_channels * kernel_h * kernel_w
        var fan_in = in_channels * kernel_h * kernel_w
        # Fan-out for conv2d: out_channels * kernel_h * kernel_w
        var fan_out = out_channels * kernel_h * kernel_w
        self.weight = kaiming_uniform(
            fan_in, fan_out, weight_shape, "fan_in", dtype
        ).as_tensor[dtype]()

        # Initialize bias to zeros
        # Shape: (out_channels,)
        var bias_shape = List[Int]()
        bias_shape.append(out_channels)
        self.bias = zeros(bias_shape, dtype).as_tensor[dtype]()

    fn forward(self, input: Tensor[dtype]) raises -> Tensor[dtype]:
        """Forward pass: y = conv2d(x, weight, bias, stride, padding).

        Applies the learned convolutional filters to the input.

        Args:
            input: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).

        Raises:
            Error if tensor operations fail.

        Note:
            The output spatial dimensions are computed as:
            - `out_height = (height + 2*padding - kernel_h) // stride + 1`
            - `out_width = (width + 2*padding - kernel_w) // stride + 1`

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3, stride=1, padding=1)
            var input = randn([1, 3, 32, 32], DType.float32)  # 32x32 RGB image
            var output = layer.forward(input)  # Shape: [1, 16, 32, 32]
            ```
        """
        return conv2d(
            input.as_any(),
            self.weight.as_any(),
            self.bias.as_any(),
            self.stride,
            self.padding,
        ).as_tensor[dtype]()

    fn backward(
        self, grad_output: Tensor[dtype], input: Tensor[dtype]
    ) raises -> Tuple[Tensor[dtype], Tensor[dtype], Tensor[dtype]]:
        """Backward pass: compute gradients w.r.t. input, weight, and bias.

        Computes gradients needed for training via backpropagation.

        Args:
            grad_output: Gradient w.r.t. output, shape (batch, out_channels, out_H, out_W).
            input: Input from forward pass, shape (batch, in_channels, in_H, in_W).

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias):
            - grad_input: Gradient w.r.t. input, shape (batch, in_channels, in_H, in_W).
            - grad_weight: Gradient w.r.t. weight, shape (out_channels, in_channels, kH, kW).
            - grad_bias: Gradient w.r.t. bias, shape (out_channels,).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3)
            var input = randn([2, 3, 32, 32], DType.float32)
            var output = layer.forward(input)

            # Compute gradients
            var grad_output = randn(output.shape(), DType.float32)
            var (grad_input, grad_weight, grad_bias) = layer.backward(grad_output, input)
            ```
        """
        var result = conv2d_backward(
            grad_output.as_any(),
            input.as_any(),
            self.weight.as_any(),
            self.stride,
            self.padding,
        )
        return (
            result.grad_input.as_tensor[dtype](),
            result.grad_weights.as_tensor[dtype](),
            result.grad_bias.as_tensor[dtype](),
        )

    fn parameters(self) raises -> List[Tensor[dtype]]:
        """Get list of trainable parameters.

        Returns:
            List containing [weight, bias] tensors that need gradients.

        Raises:
            Error if tensor copying fails.

        Example:
            ```mojo
            var layer = Conv2dLayer(3, 16, 3, 3)
            var params = layer.parameters()
            # params[0] is weight, params[1] is bias
            ```
        """
        var params = List[Tensor[dtype]]()

        # Create copies via ExTensor round-trip
        var weight_any = zeros_like(self.weight.as_any())
        var bias_any = zeros_like(self.bias.as_any())

        var weight_size = self.weight.numel()
        var bias_size = self.bias.numel()

        for i in range(weight_size):
            weight_any._data[i] = self.weight.as_any()._data[i]

        for i in range(bias_size):
            bias_any._data[i] = self.bias.as_any()._data[i]

        params.append(weight_any.as_tensor[dtype]())
        params.append(bias_any.as_tensor[dtype]())
        return params^
