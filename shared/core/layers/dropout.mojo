"""Dropout layer for regularization during training.

This module provides a Dropout layer that randomly zeroes input elements
during training to prevent overfitting. During inference (training=False),
dropout is disabled and the input is scaled appropriately.

Key components:
- DropoutLayer: Dropout regularization layer
  Implements: y = mask * x / (1 - dropout_rate) during training
               y = x during inference
"""

from ..extensor import ExTensor, zeros_like, full
from shared.tensor.tensor import Tensor
from random import random_float64


struct DropoutLayer[dtype: DType = DType.float32](Copyable, Movable):
    """Dropout layer for regularization, parameterized on dtype.

    Dropout randomly sets input elements to zero during training to prevent
    co-adaptation and reduce overfitting. The remaining elements are scaled
    by 1/(1-p) where p is the dropout rate, ensuring expected value is preserved.

    During inference (training=False), the layer is disabled and passes input
    through unchanged.

    Parameters:
        dtype: The data type for mask and I/O tensors (default: float32).

    Attributes:
        dropout_rate: Probability of dropping each element (default: 0.5).
        training: Whether layer is in training mode (default: False).

    Example:
        ```mojo
        var layer = DropoutLayer(0.5)
        layer.set_training(True)  # Enable dropout for training

        var input = randn([4, 10], DType.float32)
        var output = layer.forward(input)

        # Backward pass
        var grad_output = randn(output.shape(), DType.float32)
        var grad_input = layer.backward(grad_output, layer.last_mask)
        ```
    """

    var dropout_rate: Float32
    var training: Bool
    var last_mask: ExTensor  # Store mask for backward pass (kept as ExTensor)

    fn __init__(out self, dropout_rate: Float32 = 0.5) raises:
        """Initialize dropout layer.

        Args:
            dropout_rate: Probability of dropping each element. Must be in [0, 1)
                         (default: 0.5).

        Raises:
            Error if dropout_rate is not in valid range.

        Example:
            ```mojo
            var layer = DropoutLayer(0.5)  # 50% dropout
            var layer2 = DropoutLayer(0.1)  # 10% dropout
            ```
        """
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise Error(
                "dropout_rate must be in [0, 1), got: " + String(dropout_rate)
            )

        self.dropout_rate = dropout_rate
        self.training = False

        # Initialize with a dummy mask (will be replaced in forward pass)
        self.last_mask = zeros_like(ExTensor([1], DType.float32))

    fn set_training(mut self, training: Bool):
        """Set training mode.

        Args:
            training: True to enable dropout during forward pass,
                     False to disable dropout (inference mode).

        Example:
            ```mojo
            var layer = DropoutLayer(0.5)
            layer.set_training(True)   # Enable dropout for training
            var output = layer.forward(input)
            layer.set_training(False)  # Disable for inference
            var output = layer.forward(input)
            ```
        """
        self.training = training

    fn forward(mut self, input: Tensor[dtype]) raises -> Tensor[dtype]:
        """Forward pass: apply dropout during training, pass through during inference.

        During training (training=True):
        1. Generate random mask where each element is in [0, 1].
        2. Keep elements where mask > dropout_rate, zero others.
        3. Scale output by 1/(1-dropout_rate) to maintain expected value.
        4. Store mask for backward pass.

        During inference (training=False):
        - Return input unchanged (no dropout applied).

        Args:
            input: Input tensor of any shape.

        Returns:
            Output tensor with dropout applied (if training) or unchanged (if inference).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = DropoutLayer(0.5)
            var input = ones([4, 10], DType.float32)

            # During training
            layer.set_training(True)
            var output = layer.forward(input)  # ~50% zeros, others scaled by 2.0

            # During inference
            layer.set_training(False)
            var output = layer.forward(input)  # Unchanged
            ```
        """
        if not self.training:
            # During inference, return input unchanged
            return input

        # Convert to ExTensor for internal computation
        var input_any = input.as_any()

        # Generate random mask: elements > dropout_rate are kept, others dropped
        var mask = ExTensor(input_any._shape, DType.float32)

        if input_any._dtype == DType.float32:
            for i in range(input_any._numel):
                # Generate random value in [0, 1)
                var rand_val = Float32(random_float64())
                # Keep element if rand_val > dropout_rate, else drop it
                mask[i] = Float32(1.0) if (
                    rand_val > Float32(self.dropout_rate)
                ) else Float32(0.0)
        elif input_any._dtype == DType.float64:
            for i in range(input_any._numel):
                var rand_val = random_float64()
                mask.set(i, 1.0 if (
                    rand_val > Float64(self.dropout_rate)
                ) else 0.0)
        elif input_any._dtype == DType.float16:
            for i in range(input_any._numel):
                var rand_val = Float32(random_float64())
                mask[i] = Float32(1.0) if (
                    rand_val > Float32(self.dropout_rate)
                ) else Float32(0.0)
        else:
            raise Error("dropout: only float16/32/64 dtypes supported")

        # Store mask for backward pass
        self.last_mask = mask

        # Apply mask and scale: output = mask * input / (1 - dropout_rate)
        var scale = Float32(1.0) / (Float32(1.0) - self.dropout_rate)
        var result = ExTensor(input_any._shape, input_any._dtype)

        if input_any._dtype == DType.float32:
            for i in range(input_any._numel):
                var input_val = input_any._data.bitcast[Float32]()[i]
                var mask_val = mask._data.bitcast[Float32]()[i]
                result[i] = Float32(mask_val * input_val * scale)
        elif input_any._dtype == DType.float64:
            for i in range(input_any._numel):
                var input_val = input_any._data.bitcast[Float64]()[i]
                var mask_val = Float64(mask._data.bitcast[Float32]()[i])
                result.set(i, mask_val * input_val * Float64(scale))
        elif input_any._dtype == DType.float16:
            for i in range(input_any._numel):
                var input_val = input_any._data.bitcast[Float16]()[i]
                var mask_val = Float16(mask._data.bitcast[Float32]()[i])
                result.set(i, Float16(mask_val * input_val * Float16(scale)))
        else:
            raise Error("dropout: only float16/32/64 dtypes supported")

        return result.as_tensor[dtype]()

    fn backward(
        self, grad_output: Tensor[dtype], mask: ExTensor
    ) raises -> Tensor[dtype]:
        """Backward pass: apply same mask as forward pass.

        During training, propagates gradient through kept elements only,
        using the same scale factor as forward pass.

        Args:
            grad_output: Gradient w.r.t. output from upstream.
            mask: The dropout mask from forward pass (1 for kept, 0 for dropped).

        Returns:
            Gradient w.r.t. input with dropout mask applied:
            grad_input = mask * grad_output / (1 - dropout_rate).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var layer = DropoutLayer(0.5)
            layer.set_training(True)
            var input = ExTensor(...)
            var output = layer.forward(input)
            var grad_output = ExTensor(...)
            var grad_input = layer.backward(grad_output, layer.last_mask)
            ```
        """
        var grad_any = grad_output.as_any()
        var scale = Float32(1.0) / (Float32(1.0) - self.dropout_rate)
        var result = ExTensor(grad_any._shape, grad_any._dtype)

        if grad_any._dtype == DType.float32:
            for i in range(grad_any._numel):
                var grad_val = grad_any._data.bitcast[Float32]()[i]
                var mask_val = mask._data.bitcast[Float32]()[i]
                result[i] = Float32(mask_val * grad_val * scale)
        elif grad_any._dtype == DType.float64:
            for i in range(grad_any._numel):
                var grad_val = grad_any._data.bitcast[Float64]()[i]
                var mask_val = Float64(mask._data.bitcast[Float32]()[i])
                result.set(i, mask_val * grad_val * Float64(scale))
        elif grad_any._dtype == DType.float16:
            for i in range(grad_any._numel):
                var grad_val = grad_any._data.bitcast[Float16]()[i]
                var mask_val = Float16(mask._data.bitcast[Float32]()[i])
                result.set(i, Float16(mask_val * grad_val * Float16(scale)))
        else:
            raise Error("dropout backward: only float16/32/64 dtypes supported")

        return result.as_tensor[dtype]()

    fn parameters(self) raises -> List[Tensor[dtype]]:
        """Get list of trainable parameters.

        Returns:
            Empty list since Dropout has no learnable parameters.

        Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var layer = DropoutLayer(0.5)
            var params = layer.parameters()
            # params is empty
            ```
        """
        var params = List[Tensor[dtype]]()
        return params^
