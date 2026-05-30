"""Linear (fully connected) layer for neural networks.

This module provides a fully connected (dense) layer that transforms inputs
from in_features dimensions to out_features dimensions using learnable weights
and biases.

Key components:
- Linear: Fully connected layer with learnable weights and bias.
  Implements: y = xW + b (with broadcasting support for batched inputs).
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import randn, zeros, zeros_like
from projectodyssey.core.module import Module
from projectodyssey.tensor.tensor import Tensor


struct Linear[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Linear layer: y = xW + b.

    A fully connected neural network layer that transforms inputs
    from in_features to out_features dimensions with proper matrix
    multiplication and bias broadcasting.

    Parameters:
        dtype: Data type for weights and bias (default: float32).

    Attributes:
        weight: Weight matrix of shape (in_features, out_features).
        bias: Bias vector of shape (out_features,).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    var weight: AnyTensor
    var bias: AnyTensor
    var in_features: Int
    var out_features: Int

    def __init__(out self, in_features: Int, out_features: Int) raises:
        """Initialize linear layer with random weights and zero bias.

        Uses Xavier-style initialization for weights. Bias is initialized to zero.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.

        Raises:
            Error if tensor creation fails.

        Example:
            ```mojo
            var layer = Linear(10, 5)  # 10 inputs, 5 outputs
            ```
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with randn (standard normal distribution)
        self.weight = randn([in_features, out_features], Self.dtype)

        # Initialize bias to zeros
        self.bias = zeros([out_features], Self.dtype)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: y = xW + b.

        Computes the linear transformation: output = input @ weight + bias.
        Supports batched inputs through matrix multiplication broadcasting.

        Args:
            input: Input tensor of shape (batch_size, in_features) or
                (in_features,).

        Returns:
            Output tensor of shape (batch_size, out_features) or
                (out_features,).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = Linear(10, 5)
            var input = ones([4, 10], DType.float32)  # batch of 4 samples
            var output = layer.forward(input)  # Shape: [4, 5]
            ```
        """
        # Compute: output = input @ weight + bias
        # Matrix multiplication: input @ weight
        from projectodyssey.core.matrix import matmul

        var matmul_result = matmul(input, self.weight)

        # Add bias with broadcasting support
        var output = matmul_result + self.bias

        return output

    def parameters(self) raises -> List[AnyTensor]:
        """Get list of trainable parameters.

        Returns:
            List containing [weight, bias] tensors

        Raises:
            Error if tensor copying fails

        Example:
            ```mojo
            var layer = Linear(10, 5)
            var params = layer.parameters()
            # params[0] is weight, params[1] is bias
            ```
        """
        var params = List[AnyTensor]()
        params.append(self.weight)
        params.append(self.bias)
        return params^

    def train(mut self):
        """Switch to training mode (no-op for Linear layer)."""
        pass

    def eval(mut self):
        """Switch to inference mode (no-op for Linear layer)."""
        pass
