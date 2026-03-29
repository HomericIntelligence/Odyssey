"""Functional autograd helpers - practical gradient computation.

This module provides helper functions for common gradient patterns,
avoiding the complexity of full computation graph autograd while still
reducing boilerplate code.

Design Philosophy:
- YAGNI: Implement helpers for patterns we actually use
- KISS: Simple function calls, not complex graph management
- Practical: Works today with current Mojo constraints

Each helper function computes gradients for a specific loss + reduction pattern
This covers 90% of real use cases without the complexity of full autograd.

Example:
   ```mojo
    from shared.autograd.functional import mse_loss_and_grad, multiply_scalar

    # Compute loss and gradient in one call
    var result = mse_loss_and_grad(predictions, targets)

    # Apply gradient with scalar learning rate
    parameters = subtract(parameters, multiply_scalar(result.grad, 0.01))
    ```

Common Patterns Supported:
- MSE loss + mean reduction
- Binary cross-entropy loss + mean reduction
- Cross-entropy loss (includes softmax)
- Scalar operations (multiply_scalar, add_scalar, etc.)
- Parameter update helpers
"""

from shared.tensor.any_tensor import AnyTensor, ones
from shared.core.arithmetic import add, multiply, subtract, divide
from shared.core.reduction import sum, mean, sum_backward, mean_backward
from shared.core.loss import (
    mean_squared_error,
    mean_squared_error_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
    cross_entropy,
    cross_entropy_backward,
)


# ============================================================================
# Scalar Operations - Efficient tensor-scalar arithmetic
# ============================================================================


fn multiply_scalar(tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Multiply tensor by a scalar value.

        More efficient than creating a full tensor filled with the scalar value.

    Args:
            tensor: Input tensor.
            scalar: Scalar value to multiply by.

    Returns:
            New tensor with result.

    Raises:
            Error: If operation fails.
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        result._set_float64(i, val * scalar)
    return result


fn add_scalar(tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Add a scalar value to all elements of a tensor.

    Args:
            tensor: Input tensor.
            scalar: Scalar value to add.

    Returns:
            New tensor with result.

    Raises:
            Error: If operation fails.
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        result._set_float64(i, val + scalar)
    return result


fn subtract_scalar(tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Subtract a scalar value from all elements of a tensor.

    Args:
            tensor: Input tensor.
            scalar: Scalar value to subtract.

    Returns:
            New tensor with result.

    Raises:
            Error: If operation fails.
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        result._set_float64(i, val - scalar)
    return result


fn divide_scalar(tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Divide all elements of a tensor by a scalar value.

    Args:
            tensor: Input tensor.
            scalar: Scalar value to divide by.

    Returns:
            New tensor with result.

    Raises:
            Error: If scalar is zero.
    """
    if scalar == 0.0:
        raise Error("Cannot divide by zero")

    var result = AnyTensor(tensor.shape(), tensor.dtype())
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        result._set_float64(i, val / scalar)
    return result


# ============================================================================
# Parameter Update Helpers
# ============================================================================


fn apply_gradient(
    parameter: AnyTensor, gradient: AnyTensor, learning_rate: Float64
) raises -> AnyTensor:
    """Apply a gradient to a parameter with given learning rate.

        Performs: parameter = parameter - learning_rate * gradient.

    Args:
            parameter: Parameter tensor to update.
            gradient: Gradient tensor (same shape as parameter).
            learning_rate: Learning rate (step size).

    Returns:
            Updated parameter tensor.

    Raises:
            Error: If shapes don't match or operation fails.
    """
    if gradient.shape() != parameter.shape():
        raise Error("Gradient shape must match parameter shape")

    # Compute: parameter - lr * gradient
    var update = multiply_scalar(gradient, learning_rate)
    return subtract(parameter, update)


fn apply_gradients(
    mut parameters: List[AnyTensor],
    gradients: List[AnyTensor],
    learning_rate: Float64,
) raises:
    """Apply gradients to multiple parameters in-place.

        Performs: parameters[i] = parameters[i] - learning_rate * gradients[i].

    Args:
            parameters: Parameter tensors to update (modified in-place).
            gradients: Gradient tensors (same shapes as parameters).
            learning_rate: Learning rate (step size).

    Raises:
            Error: If parameter count doesn't match gradient count or shape mismatch occurs.
    """
    if len(parameters) != len(gradients):
        raise Error("Parameter count must match gradient count")

    for i in range(len(parameters)):
        parameters[i] = apply_gradient(
            parameters[i], gradients[i], learning_rate
        )


# ============================================================================
# Loss and Gradient Helpers
# ============================================================================


struct LossAndGrad:
    """Container for loss value and gradient.

    Returned by loss_and_grad helper functions.

    Attributes:
        loss: Scalar loss value.
        grad: Gradient tensor (same shape as input).
    """

    var loss: AnyTensor
    var grad: AnyTensor

    fn __init__(out self, var loss: AnyTensor, var grad: AnyTensor):
        """Initialize loss and gradient pair.

        Args:
            loss: Scalar loss tensor (ownership transferred).
            grad: Gradient tensor (ownership transferred).
        """
        self.loss = loss^
        self.grad = grad^


fn mse_loss_and_grad(
    predictions: AnyTensor, targets: AnyTensor
) raises -> LossAndGrad:
    """Compute MSE loss and gradient in one pass.

        Computes:
            loss = mean(mean_squared_error(predictions, targets))
            grad = ∂loss/∂predictions

        This is the most common loss pattern for regression.

    Args:
            predictions: Model predictions, any shape.
            targets: Ground truth targets, same shape as predictions.

    Returns:
            LossAndGrad containing scalar loss and gradient tensor.

    Raises:
            Error: If operation fails.
    """
    # Forward pass
    var squared_errors = mean_squared_error(predictions, targets)
    var loss = mean(squared_errors, axis=-1, keepdims=False)

    # Backward pass
    var grad_loss = ones(loss.shape(), loss.dtype())
    var grad_squared_errors = mean_backward(grad_loss, squared_errors, axis=-1)
    var grad_predictions = mean_squared_error_backward(
        grad_squared_errors, predictions, targets
    )

    return LossAndGrad(loss, grad_predictions)


fn bce_loss_and_grad(
    predictions: AnyTensor, targets: AnyTensor, epsilon: Float64 = 1e-7
) raises -> LossAndGrad:
    """Compute binary cross-entropy loss and gradient.

        Computes:
            loss = mean(binary_cross_entropy(predictions, targets))
            grad = ∂loss/∂predictions

        Used for binary classification (predictions from sigmoid).

    Args:
            predictions: Predicted probabilities in [0, 1], shape (batch_size,) or (batch_size, 1).
            targets: Binary labels (0 or 1), same shape as predictions.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            LossAndGrad containing scalar loss and gradient tensor.

    Raises:
            Error: If operation fails.
    """
    # Forward pass
    var bce_per_sample = binary_cross_entropy(predictions, targets, epsilon)
    var loss = mean(bce_per_sample, axis=-1, keepdims=False)

    # Backward pass
    var grad_loss = ones(loss.shape(), loss.dtype())
    var grad_bce = mean_backward(grad_loss, bce_per_sample, axis=-1)
    var grad_predictions = binary_cross_entropy_backward(
        grad_bce, predictions, targets, epsilon
    )

    return LossAndGrad(loss, grad_predictions)


fn ce_loss_and_grad(
    logits: AnyTensor, targets: AnyTensor, epsilon: Float64 = 1e-7
) raises -> LossAndGrad:
    """Compute cross-entropy loss and gradient.

        Computes:
            loss = cross_entropy(logits, targets)  # Already includes mean
            grad = ∂loss/∂logits

        Used for multi-class classification. Includes softmax internally.

    Args:
            logits: Raw model outputs (before softmax), shape (batch_size, num_classes).
            targets: One-hot encoded labels, same shape as logits.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            LossAndGrad containing scalar loss and gradient tensor.

    Raises:
            Error: If operation fails.

    Note:
            The cross_entropy function already computes mean reduction internally.
    """
    # Forward pass (cross_entropy already includes mean reduction)
    var loss = cross_entropy(logits, targets, axis=-1, epsilon=epsilon)

    # Backward pass
    var grad_loss = ones(loss.shape(), loss.dtype())
    var grad_logits = cross_entropy_backward(
        grad_loss, logits, targets, epsilon
    )

    return LossAndGrad(loss, grad_logits)


# Helper function for manual gradient computation patterns
fn compute_gradient(
    predictions: AnyTensor, targets: AnyTensor, loss_type: String = "mse"
) raises -> AnyTensor:
    """Compute gradient for common loss functions.

        Convenience function that dispatches to the appropriate loss_and_grad
        helper based on loss_type string.

    Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            loss_type: One of "mse", "bce", "ce" (default: "mse").

    Returns:
            Gradient tensor.

    Raises:
            Error: If unknown loss type or operation fails.

    Note:
            For more control, use the specific loss_and_grad functions directly.
    """
    if loss_type == "mse":
        var result = mse_loss_and_grad(predictions, targets)
        return result.grad
    elif loss_type == "bce":
        var result = bce_loss_and_grad(predictions, targets)
        return result.grad
    elif loss_type == "ce":
        var result = ce_loss_and_grad(predictions, targets)
        return result.grad
    else:
        raise Error(
            "Unknown loss type: " + loss_type + ". Use 'mse', 'bce', or 'ce'."
        )


# ============================================================================
# Module Validation Entry Point
# ============================================================================
