"""Functional dropout regularization.

This module provides pure functional implementations of dropout for regularization
In pure functional design, dropout returns both the output and the mask needed for backward.

Dropout randomly zeros some elements of the input tensor with probability p during training
This helps prevent overfitting by randomly "dropping out" neurons.
"""

from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    zeros_like,
    ones_like,
    full_like,
)
from shared.core.arithmetic import multiply, divide
from std import random


def dropout(
    x: AnyTensor, p: Float64, training: Bool, seed: Int = 0
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Functional dropout with mask return.

        Randomly zeros elements with probability p during training
        In pure functional design, returns both output and mask for backward pass

    Args:
            x: Input tensor of any shape.
            p: Probability of dropping an element (0.0 to 1.0).
            training: If True, apply dropout. If False, return input unchanged.
            seed: Random seed for reproducibility (default: 0 uses random seed).

    Returns:
            Tuple of (output, mask):
                - output: Dropped-out tensor (scaled by 1/(1-p) during training)
                - mask: Binary mask showing which elements were kept (1.0) or dropped (0.0)

    Raises:
        Error: If operation fails.

        Example:
            ```mojo
            from shared.core import AnyTensor, dropout, dropout_backward

            # Training mode
            var (output, mask) = dropout(x, p=0.5, training=True, seed=42)

            # Later in backward pass
            var grad_x = dropout_backward(grad_output, mask, p=0.5)

            # Inference mode (no dropout)
            var (output, _) = dropout(x, p=0.5, training=False)
            # output == x in inference mode
            ```

    Note:
            - During training: output = x * mask / (1 - p) for scaling.
            - During inference: output = x (no dropout).
            - Mask is needed for backward pass (must be saved by caller).
            - Pure functional: caller manages mask state.
    """
    if p < 0.0 or p >= 1.0:
        raise Error("Dropout probability must be in [0, 1)")

    # Inference mode: no dropout
    if not training:
        var ones_mask = ones_like(x)
        return (x, ones_mask)

    # Training mode: apply dropout
    var size = x.numel()
    var mask = zeros_like(x)

    # Generate random mask
    if seed > 0:
        random.seed(seed)

    if x.dtype() == DType.float32:
        for i in range(size):
            var rand_val = Float32(random.random_float64())
            mask[i] = Float32(1.0) if rand_val > Float32(p) else Float32(0.0)
    elif x.dtype() == DType.float64:
        for i in range(size):
            var rand_val = random.random_float64()
            mask.set(i, 1.0 if rand_val > p else 0.0)
    elif x.dtype() == DType.float16:
        for i in range(size):
            var rand_val = Float32(random.random_float64())
            mask[i] = Float32(1.0) if rand_val > Float32(p) else Float32(0.0)
    else:
        raise Error("dropout: only float16/32/64 dtypes supported")

    # Apply mask and scale: output = x * mask / (1 - p)
    var masked = multiply(x, mask)
    var scale = 1.0 / (1.0 - p)
    var scale_tensor = full_like(x, scale)
    var output = multiply(masked, scale_tensor)

    return (output, mask)


def dropout2d(
    x: AnyTensor, p: Float64, training: Bool, seed: Int = 0
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Functional 2D dropout (spatial dropout) for CNNs.

        Randomly zeros entire channels with probability p during training
        This is more effective for convolutional layers than regular dropout

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            p: Probability of dropping a channel (0.0 to 1.0).
            training: If True, apply dropout. If False, return input unchanged.
            seed: Random seed for reproducibility (default: 0 uses random seed).

    Returns:
            Tuple of (output, mask):
                - output: Dropped-out tensor (entire channels zeroed)
                - mask: Binary mask at channel level (batch, channels, 1, 1)

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import dropout2d, dropout2d_backward

            # Training mode - drops entire feature maps
            var (output, mask) = dropout2d(x, p=0.2, training=True, seed=42)

            # Backward pass
            var grad_x = dropout2d_backward(grad_output, mask, p=0.2)
            ```

    Note:
            - Drops entire channels (all spatial positions in a channel).
            - More effective than standard dropout for CNNs.
            - Mask shape is (batch, channels, 1, 1) for broadcasting.
    """
    if p < 0.0 or p >= 1.0:
        raise Error("Dropout probability must be in [0, 1)")

    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "dropout2d requires 4D input (batch, channels, height, width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    # Inference mode: no dropout
    if not training:
        var ones_mask = ones_like(x)
        return (x, ones_mask)

    # Training mode: create channel-level mask
    var mask_shape = List[Int]()
    mask_shape.append(batch)
    mask_shape.append(channels)
    mask_shape.append(1)
    mask_shape.append(1)
    var channel_mask = zeros(mask_shape, x.dtype())

    # Generate random mask at channel level
    if seed > 0:
        random.seed(seed)

    if x.dtype() == DType.float32:
        for b in range(batch):
            for c in range(channels):
                var rand_val = Float32(random.random_float64())
                var idx = b * channels + c
                channel_mask[idx] = Float32(1.0) if rand_val > Float32(
                    p
                ) else Float32(0.0)
    elif x.dtype() == DType.float64:
        for b in range(batch):
            for c in range(channels):
                var rand_val = random.random_float64()
                var idx = b * channels + c
                channel_mask.set(idx, 1.0 if rand_val > p else 0.0)
    elif x.dtype() == DType.float16:
        for b in range(batch):
            for c in range(channels):
                var rand_val = Float32(random.random_float64())
                var idx = b * channels + c
                channel_mask[idx] = Float32(1.0) if rand_val > Float32(
                    p
                ) else Float32(0.0)
    else:
        raise Error("dropout2d: only float16/32/64 dtypes supported")

    # Broadcast mask to full tensor shape and apply
    var full_mask = zeros_like(x)

    if x.dtype() == DType.float32:
        for b in range(batch):
            for c in range(channels):
                var mask_val = channel_mask._data.bitcast[Float32]()[
                    b * channels + c
                ]
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        full_mask[idx] = Float32(mask_val)
    elif x.dtype() == DType.float64:
        for b in range(batch):
            for c in range(channels):
                var mask_val = channel_mask._data.bitcast[Float64]()[
                    b * channels + c
                ]
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        full_mask.set(idx, mask_val)
    elif x.dtype() == DType.float16:
        for b in range(batch):
            for c in range(channels):
                var mask_val = channel_mask._data.bitcast[Float16]()[
                    b * channels + c
                ]
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        full_mask.set(idx, Float16(mask_val))

    # Apply mask and scale
    var masked = multiply(x, full_mask)
    var scale = 1.0 / (1.0 - p)
    var scale_tensor = full_like(x, scale)
    var output = multiply(masked, scale_tensor)

    return (output, full_mask)


def dropout_backward(
    grad_output: AnyTensor, mask: AnyTensor, p: Float64
) raises -> AnyTensor:
    """Backward pass for dropout.

        Routes gradients only through positions that were not dropped

    Args:
            grad_output: Gradient from upstream.
            mask: Binary mask from forward pass (1.0 = kept, 0.0 = dropped).
            p: Dropout probability (must match forward pass).

    Returns:
            Gradient with respect to input

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            # Forward pass
            var (output, mask) = dropout(x, p=0.5, training=True)

            # ... compute loss and grad_output ...

            # Backward pass
            var grad_x = dropout_backward(grad_output, mask, p=0.5)
            ```

    Note:
            - Gradient flows only through non-dropped elements.
            - Scaled by 1/(1-p) to match forward pass scaling.
    """
    # Apply mask and scale: grad_input = grad_output * mask / (1 - p)
    var masked_grad = multiply(grad_output, mask)
    var scale = 1.0 / (1.0 - p)
    var scale_tensor = full_like(grad_output, scale)
    return multiply(masked_grad, scale_tensor)


def dropout2d_backward(
    grad_output: AnyTensor, mask: AnyTensor, p: Float64
) raises -> AnyTensor:
    """Backward pass for 2D dropout (spatial dropout).

    Args:
            grad_output: Gradient from upstream.
            mask: Binary mask from forward pass (full spatial mask).
            p: Dropout probability (must match forward pass).

    Returns:
            Gradient with respect to input

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            # Forward pass
            var (output, mask) = dropout2d(x, p=0.2, training=True)

            # ... compute loss and grad_output ...

            # Backward pass
            var grad_x = dropout2d_backward(grad_output, mask, p=0.2)
            ```
    """
    # Same as regular dropout backward - mask is already broadcast
    return dropout_backward(grad_output, mask, p)
