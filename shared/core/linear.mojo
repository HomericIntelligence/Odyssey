"""Functional linear (fully connected) transformation.

This module provides a pure functional implementation of linear transformations,
following the pattern y = xW^T + b. The caller manages all state (weights, bias).
"""

from shared.core.extensor import ExTensor
from shared.core.matrix import matmul, transpose
from shared.core.arithmetic import add
from shared.core.reduction import sum
from shared.core.gradient_types import GradientPair, GradientTriple


fn linear(x: ExTensor, weights: ExTensor, bias: ExTensor) raises -> ExTensor:
    """Functional linear transformation: y = xW^T + b.

        Pure function - caller manages weights and bias. No internal state

    Args:
            x: Input tensor of shape (batch_size, in_features).
            weights: Weight matrix of shape (out_features, in_features).
            bias: Bias vector of shape (out_features,).

    Returns:
            Output tensor of shape (batch_size, out_features).

        Example:
            ```mojo
            from shared.core import ExTensor, linear, zeros, xavier_uniform

            # Caller manages state
            var w = xavier_uniform(10, 784, DType.float32)
            var b = zeros(10, DType.float32)

            # Pure function call
            var output = linear(input, w, b)
            ```

    Raises:
            Error if shapes are incompatible for matrix multiplication.
    """
    # Compute xW^T
    var out = matmul(x, transpose(weights))

    # Add bias
    return add(out, bias)


fn linear_no_bias(x: ExTensor, weights: ExTensor) raises -> ExTensor:
    """Functional linear transformation without bias: y = xW^T.

        Pure function for linear transformation with no bias term.

    Args:
            x: Input tensor of shape (batch_size, in_features).
            weights: Weight matrix of shape (out_features, in_features).

    Returns:
            Output tensor of shape (batch_size, out_features).

    Raises:
            Error if shapes are incompatible for matrix multiplication.
    """
    return matmul(x, transpose(weights))


fn linear_backward(
    grad_output: ExTensor, x: ExTensor, weights: ExTensor
) raises -> GradientTriple:
    """Backward pass for linear transformation.

        Computes gradients with respect to input, weights, and bias.

        Math:
            Given: y = xW^T + b
            grad_input = grad_output @ W
            grad_kernel = grad_output^T @ x
            grad_bias = sum(grad_output, axis=0)

    Args:
            grad_output: Gradient of loss w.r.t. output, shape (batch_size, out_features).
            x: Input tensor from forward pass, shape (batch_size, in_features).
            weights: Weight matrix from forward pass, shape (out_features, in_features).

    Returns:
            GradientTriple containing:
                - grad_input: Gradient w.r.t. input, shape (batch_size, in_features).
                - grad_kernel: Gradient w.r.t. weights, shape (out_features, in_features).
                - grad_bias: Gradient w.r.t. bias, shape (out_features,).

        Example:
            ```
            from shared.core import ExTensor, linear, linear_backward

            # Forward pass
            var output = linear(x, weights, bias)
            # ... compute loss and grad_output ...

            # Backward pass
            var result = linear_backward(grad_output, x, weights)
            var grad_x = result.grad_input
            var grad_w = result.grad_weights
            var grad_b = result.grad_bias
            ```

    Raises:
            Error if tensor shapes are incompatible.
    """
    # grad_input = grad_output @ W
    # weights is (out_features, in_features), so we use it directly
    var grad_input = matmul(grad_output, weights)

    # grad_kernel = grad_output^T @ x
    # grad_output: (batch, out_features) -> transpose -> Tuple[out_features, batch]
    # x: (batch, in_features)
    # result: (out_features, in_features)
    var grad_kernel = matmul(transpose(grad_output), x)

    # grad_bias = sum(grad_output, axis=0)
    # Sum over batch dimension to get (out_features,)
    var grad_bias = sum(grad_output, axis=0)

    return GradientTriple(grad_input^, grad_kernel^, grad_bias^)


fn linear_no_bias_backward(
    grad_output: ExTensor, x: ExTensor, weights: ExTensor
) raises -> GradientPair:
    """Backward pass for linear transformation without bias.

        Computes gradients with respect to input and weights only.

    Args:
            grad_output: Gradient of loss w.r.t. output, shape (batch_size, out_features).
            x: Input tensor from forward pass, shape (batch_size, in_features).
            weights: Weight matrix from forward pass, shape (out_features, in_features).

    Returns:
            GradientPair containing:
                - grad_input: Gradient w.r.t. input, shape (batch_size, in_features).
                - grad_kernel: Gradient w.r.t. weights, shape (out_features, in_features).

    Raises:
            Error if tensor shapes are incompatible.
    """
    # grad_input = grad_output @ W
    var grad_input = matmul(grad_output, weights)

    # grad_kernel = grad_output^T @ x
    var grad_kernel = matmul(transpose(grad_output), x)

    return GradientPair(grad_input^, grad_kernel^)
