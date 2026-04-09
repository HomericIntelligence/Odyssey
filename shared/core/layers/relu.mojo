"""ReLU (Rectified Linear Unit) activation layer.

This module provides a ReLU layer wrapper that applies the ReLU activation
function: max(0, x). The layer includes both forward and backward passes
for use in neural network training.

Key components:
- ReLULayer: ReLU activation layer
  Implements: y = max(0, x) forward, dL/dx = grad * (x > 0) backward
"""

from shared.tensor.any_tensor import AnyTensor, zeros_like
from ..activation import relu, relu_backward
from ..module import Module


struct ReLULayer(Copyable, Module, Movable):
    """ReLU activation layer: y = max(0, x).

    A simple activation layer that applies the Rectified Linear Unit (ReLU)
    function element-wise. ReLU is the most common activation in deep learning,
    promoting sparse activation patterns by zeroing negative values.

    Attributes:
        No learnable parameters.

    Example:
        ```mojo
        var layer = ReLULayer()
        var input = randn([4, 10], DType.float32)
        var output = layer.forward(input)

        # Backward pass
        var grad_output = randn(output.shape(), DType.float32)
        var grad_input = layer.backward(grad_output, input)
        ```
    """

    def __init__(out self):
        """Initialize ReLU layer.

        ReLU has no learnable parameters or state

        Example:
            ```mojo
            var layer = ReLULayer()
            ```
        """
        pass

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: y = max(0, x).

        Applies ReLU activation element-wise to the input tensor.

        Args:
            input: Input tensor of any shape.

        Returns:
            Output tensor with ReLU applied, same shape as input.

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = ReLULayer()
            var input = AnyTensor.from_list([-2, -1, 0, 1, 2], DType.float32)
            var output = layer.forward(input)  # [0, 0, 0, 1, 2]
            ```
        """
        return relu(input)

    def backward(
        mut self, grad_output: AnyTensor, input: AnyTensor
    ) raises -> AnyTensor:
        """Backward pass: compute gradient w.r.t. input.

        Computes the gradient of ReLU with respect to input.
        Gradient is passed through where input > 0, zeroed elsewhere.

        Args:
            grad_output: Gradient w.r.t. output from upstream, same shape
                as input.
            input: Input tensor from forward pass.

        Returns:
            Gradient w.r.t. input, same shape as input:
            - grad_input[i] = grad_output[i] if input[i] > 0.
            - grad_input[i] = 0 if input[i] <= 0.

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = ReLULayer()
            var input = AnyTensor.from_list([-2, -1, 0, 1, 2], DType.float32)
            var output = layer.forward(input)
            var grad_output = AnyTensor.from_list(
                [0.1, 0.2, 0.3, 0.4, 0.5], DType.float32
            )
            var grad_input = layer.backward(grad_output, input)
            # grad_input = [0, 0, 0, 0.4, 0.5]
            ```
        """
        return relu_backward(grad_output, input)

    def parameters(self) raises -> List[AnyTensor]:
        """Get list of trainable parameters.

        Returns:
            Empty list since ReLU has no learnable parameters

        Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var layer = ReLULayer()
            var params = layer.parameters()
            # params is empty
            ```
        """
        var params: List[AnyTensor] = []
        return params^

    def train(mut self):
        """Switch to training mode (no-op for ReLULayer)."""
        pass

    def eval(mut self):
        """Switch to inference mode (no-op for ReLULayer)."""
        pass
