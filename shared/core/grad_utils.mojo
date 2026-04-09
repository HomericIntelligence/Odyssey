"""Gradient clipping utilities for preventing exploding gradients.

This module provides functions to clip gradients during training, which helps
prevent numerical instability and divergence in deep neural networks.

Implemented techniques:
- clip_grad_value_: Clip each gradient element to [-max_value, max_value]
- clip_grad_norm_: Clip gradient norm if it exceeds max_norm (per-parameter)
- clip_grad_global_norm_: Clip based on global norm across all parameters

Usage Example:
        ```
    from shared.autograd import clip_grad_value_, clip_grad_norm_, clip_grad_global_norm_

    # Clip individual gradient elements
    clip_grad_value_(grad, max_value=1.0)

    # Clip a single gradient by its L2 norm
    var norm = clip_grad_norm_(grad, max_norm=1.0)

    # Clip multiple gradients by their global L2 norm
    var gradients : List[AnyTensor] = [grad1, grad2, grad3]
    var global_norm = clip_grad_global_norm_(gradients, max_norm=1.0)
        ```

References:
    - On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2013)
      https://arxiv.org/abs/1211.1541
"""

from std.math import sqrt
from shared.tensor.any_tensor import AnyTensor


def clip_grad_value_(mut grad: AnyTensor, max_value: Float64) raises:
    """Clip each gradient element to [-max_value, max_value].

        This is the simplest form of gradient clipping. Each element is
        independently clipped to stay within the specified range

    Args:
            grad: The gradient tensor to clip (modified in-place).
            max_value: Maximum absolute value allowed. Elements outside
                       [-max_value, max_value] are clipped.

    Raises:
            Error: If max_value is negative

    Examples:
    ```
            var grad = ones([3, 4], DType.float32)
            clip_grad_value_(grad, max_value=1.0)
            # All elements in grad now in [-1.0, 1.0]

            var grad2 = full([2, 3], 5.0, DType.float32)
            clip_grad_value_(grad2, max_value=1.0)
    ```
    """
    if max_value < 0.0:
        raise Error("max_value must be non-negative, got: " + String(max_value))

    # Iterate over all elements and clip each one
    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        # Clip to range [-max_value, max_value]
        if val > max_value:
            grad._set_float64(i, max_value)
        elif val < -max_value:
            grad._set_float64(i, -max_value)


def clip_grad_norm_(mut grad: AnyTensor, max_norm: Float64) raises -> Float64:
    """Clip gradient if its L2 norm exceeds max_norm.

        Computes the L2 norm of the gradient: norm = sqrt(sum(grad^2))
        If norm > max_norm, scales the gradient by (max_norm / norm)
        This preserves the direction of the gradient while limiting its magnitude

    Args:
            grad: The gradient tensor to clip (modified in-place if norm exceeds max_norm).
            max_norm: Maximum allowed L2 norm. If gradient norm exceeds this,
                      the gradient is scaled down proportionally.

    Returns:
            The original L2 norm of the gradient (before clipping)

    Raises:
            Error: If max_norm is negative

    Examples:
    ```
            var grad = full([100,], 1.0, DType.float32)
            var norm = clip_grad_norm_(grad, max_norm=1.0)
            # norm is approximately sqrt(100) = 10
            # grad is scaled by 1.0/10 = 0.1, all elements become 0.1

            var grad2 = full([10,], 0.05, DType.float32)
            var norm2 = clip_grad_norm_(grad2, max_norm=1.0)
            # norm2 is approximately 0.158
            # Since 0.158 < 1.0, grad2 is unchanged.
    ```

    Note:
            The norm is computed as the L2 (Euclidean) norm: `sqrt(sum(x_i^2))`
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    # Compute L2 norm: sqrt(sum(grad^2))
    var norm_squared = 0.0
    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        norm_squared += val * val

    var norm = sqrt(norm_squared)

    # Clip if necessary
    if norm > max_norm and norm > 0.0:
        # Scale factor: max_norm / norm
        var scale_factor = max_norm / norm
        for i in range(grad.numel()):
            var val = grad._get_float64(i)
            grad._set_float64(i, val * scale_factor)

    return norm


def clip_grad_global_norm_(
    mut grads: List[AnyTensor], max_norm: Float64
) raises -> Float64:
    """Clip gradients based on their global L2 norm across all parameters.

    This is the recommended approach for recurrent neural networks (RNNs) and other
    architectures prone to exploding gradients. It computes a single norm across all
    gradient tensors and clips all gradients uniformly if needed.

    Global norm: sqrt(sum over all parameters of sum(grad_i^2))

    Args:
        grads: List of gradient tensors (modified in-place if global norm exceeds max_norm).
        max_norm: Maximum allowed global L2 norm. If exceeded, all gradients
            are scaled down proportionally.

    Returns:
        The original global L2 norm (before clipping).

    Raises:
        Error: If max_norm is negative or grads list is empty.

    Example:
        ```mojo
        var grad1 = full([10,], 1.0, DType.float32)
        var grad2 = full([20,], 1.0, DType.float32)
        var grads: List[AnyTensor] = [grad1, grad2]

        var global_norm = clip_grad_global_norm_(grads, max_norm=1.0)
        # global_norm is sqrt(10 + 20) = sqrt(30) ≈ 5.48
        # Both gradients scaled by 1.0/5.48 ≈ 0.182
        ```

    Note:
        This function modifies all tensors in the grads list in-place.
        If you need to preserve the original gradients, create copies first.

        Reference:
            On the difficulty of training Recurrent Neural Networks
            (Pascanu et al., 2013)
            `https://arxiv.org/abs/1211.1541`
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    if len(grads) == 0:
        raise Error("grads list cannot be empty")

    # Compute global L2 norm: sqrt(sum over all parameters of sum(grad^2))
    var total_norm_squared = 0.0
    for grad_idx in range(len(grads)):
        var grad = grads[grad_idx]
        for elem_idx in range(grad.numel()):
            var val = grad._get_float64(elem_idx)
            total_norm_squared += val * val

    var global_norm = sqrt(total_norm_squared)

    # Clip if necessary
    if global_norm > max_norm and global_norm > 0.0:
        # Scale factor: max_norm / global_norm
        var scale_factor = max_norm / global_norm
        for grad_idx in range(len(grads)):
            var grad = grads[grad_idx]
            for elem_idx in range(grad.numel()):
                var val = grad._get_float64(elem_idx)
                grad._set_float64(elem_idx, val * scale_factor)

    return global_norm
