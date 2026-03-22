"""Utility functions for loss functions.

This module provides common utility operations used across various loss functions,
including:
- Prediction clipping for numerical stability
- Epsilon tensor creation
- Gradient computation utilities
- Shape validation helpers

All functions are pure functional - they process inputs to produce outputs without
maintaining internal state.
"""

from .any_tensor import AnyTensor, ones_like, zeros_like, full_like
from .arithmetic import add, subtract, multiply, divide
from .elementwise import log, clip, abs
from .comparison import less, greater
from .dtype_cast import cast_tensor


fn clip_predictions(
    predictions: AnyTensor, epsilon: Float64 = 1e-7
) raises -> AnyTensor:
    """Clip predictions to prevent log(0) and numerical instability.

        Formula:
            clipped = clip(predictions, epsilon, 1.0 - epsilon).

        This is used in loss functions like BCE and Focal Loss to prevent taking
        log of zero or one, which would produce NaN or Inf values.

    Args:
            predictions: Input tensor with values typically in [0, 1] range.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            Clipped tensor with values in [epsilon, 1.0 - epsilon].

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var predictions = sigmoid(logits)  # Some values may be 0 or 1
            var clipped = clip_predictions(predictions)
            var log_pred = log(clipped)  # Safe to take log now
            ```
    """
    return clip(predictions, epsilon, 1.0 - epsilon)


fn create_epsilon_tensor(
    template: AnyTensor, epsilon: Float64 = 1e-7
) raises -> AnyTensor:
    """Create an epsilon tensor with same shape as template.

    Args:
            template: Template tensor determining output shape.
            epsilon: Epsilon value to fill (default: 1e-7).

    Returns:
            Tensor filled with epsilon value, same shape as template.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var pred = predictions
            var eps = create_epsilon_tensor(pred)
            var denominator = add(pred, eps)  # Add epsilon to prevent division by zero
            ```
    """
    return full_like(template, epsilon)


fn validate_tensor_shapes(
    tensor1: AnyTensor, tensor2: AnyTensor, operation: String
) raises:
    """Validate that two tensors have compatible shapes.

    Args:
            tensor1: First tensor to validate.
            tensor2: Second tensor to validate.
            operation: Name of operation for error message.

    Raises:
            Error: If shapes don't match.

        Example:
            ```mojo
            validate_tensor_shapes(predictions, targets, "cross_entropy")
            ```
    """
    if tensor1.shape() != tensor2.shape():
        raise Error(operation + ": Input tensors must have the same shape")


fn validate_tensor_dtypes(
    tensor1: AnyTensor, tensor2: AnyTensor, operation: String
) raises:
    """Validate that two tensors have compatible dtypes.

    Args:
            tensor1: First tensor to validate.
            tensor2: Second tensor to validate.
            operation: Name of operation for error message.

    Raises:
            Error: If dtypes don't match.

        Example:
            ```mojo
            validate_tensor_dtypes(predictions, targets, "cross_entropy")
            ```
    """
    if tensor1.dtype() != tensor2.dtype():
        raise Error(operation + ": Input tensors must have the same dtype")


fn compute_one_minus_tensor(tensor: AnyTensor) raises -> AnyTensor:
    """Compute 1.0 - tensor efficiently.

        This is a common operation in loss functions (e.g., 1 - predictions).

    Args:
            tensor: Input tensor.

    Returns:
            Tensor with values: 1.0 - tensor[i] for each element.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var one_minus_pred = compute_one_minus_tensor(predictions)
            ```
    """
    var one = ones_like(tensor)
    return subtract(one, tensor)


fn compute_sign_tensor(tensor: AnyTensor) raises -> AnyTensor:
    """Compute sign of tensor: +1 if x > 0, -1 if x < 0, 0 if x == 0.

        This is used in loss functions like smooth L1 for gradient computation.

    Args:
            tensor: Input tensor.

    Returns:
            Tensor with sign values (-1, 0, or 1).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             For hinge loss gradient
            var diff = subtract(predictions, targets)
            var sign = compute_sign_tensor(diff)
            ```
    """
    var zero = zeros_like(tensor)
    var one = ones_like(tensor)
    var neg_one = full_like(tensor, -1.0)

    # sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    var is_positive_bool = greater(tensor, zero)
    var is_negative_bool = less(tensor, zero)

    # Cast bool masks to same dtype as tensor for arithmetic operations
    var is_positive = cast_tensor(is_positive_bool, tensor.dtype())
    var is_negative = cast_tensor(is_negative_bool, tensor.dtype())

    return add(multiply(is_positive, one), multiply(is_negative, neg_one))


fn blend_tensors(
    tensor1: AnyTensor, tensor2: AnyTensor, mask: AnyTensor
) raises -> AnyTensor:
    """Blend two tensors based on a binary mask.

        Formula:
            result = tensor1 * mask + tensor2 * (1 - mask).

        This is used when selecting between different computations based on conditions
        (e.g., quadratic vs linear term in smooth L1 loss).

    Args:
            tensor1: Values to use where mask is 1.
            tensor2: Values to use where mask is 0.
            mask: Binary mask with values 0 or 1 (should be float for multiplication).

    Returns:
            Blended tensor with shape of inputs.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             In smooth L1 loss
            var quadratic = ...  # For |x| < beta
            var linear = ...      # For |x| >= beta
            var mask = less(abs_diff, beta)  # Bool: true where |x| < beta
            var result = blend_tensors(quadratic, linear, mask)
            ```
    """
    var one = ones_like(tensor1)
    var mask_inv = subtract(one, mask)

    var term1 = multiply(tensor1, mask)
    var term2 = multiply(tensor2, mask_inv)

    return add(term1, term2)


fn compute_max_stable(tensor: AnyTensor) raises -> AnyTensor:
    """Find maximum value in tensor for numerical stability (log-sum-exp trick).

        This is used in cross-entropy to find the max logit for numerical stability.

    Args:
            tensor: Input tensor.

    Returns:
            Maximum value in tensor.

    Raises:
        Error: If operation fails.

    Note:
            This is a placeholder for max_reduce import.
    """
    # For now, we import max_reduce from reduction module when needed
    # This function is documented here for reference
    return tensor


fn compute_difference(tensor1: AnyTensor, tensor2: AnyTensor) raises -> AnyTensor:
    """Compute tensor1 - tensor2 with error checking.

    Args:
            tensor1: First tensor (minuend).
            tensor2: Second tensor (subtrahend).

    Returns:
            Difference tensor.

    Raises:
            Error: If shapes don't match.
    """
    if tensor1.shape() != tensor2.shape():
        raise Error("compute_difference: Tensors must have the same shape")

    return subtract(tensor1, tensor2)


fn compute_product(tensor1: AnyTensor, tensor2: AnyTensor) raises -> AnyTensor:
    """Compute element-wise product of two tensors with error checking.

    Args:
            tensor1: First tensor.
            tensor2: Second tensor.

    Returns:
            Product tensor.

    Raises:
            Error: If shapes don't match.
    """
    if tensor1.shape() != tensor2.shape():
        raise Error("compute_product: Tensors must have the same shape")

    return multiply(tensor1, tensor2)


fn compute_ratio(
    tensor1: AnyTensor, tensor2: AnyTensor, epsilon: Float64 = 1e-7
) raises -> AnyTensor:
    """Compute element-wise ratio tensor1 / tensor2 with numerical stability.

        Adds epsilon to denominator to prevent division by zero.

    Args:
            tensor1: Numerator tensor.
            tensor2: Denominator tensor.
            epsilon: Small value to add to denominator for stability (default: 1e-7).

    Returns:
            Ratio tensor.

    Raises:
            Error: If shapes don't match.

        Example:
            ```mojo
             Safe division in loss gradients
            var numerator = ...
            var denominator = ...
            var ratio = compute_ratio(numerator, denominator)
            ```
    """
    if tensor1.shape() != tensor2.shape():
        raise Error("compute_ratio: Tensors must have the same shape")

    # Add epsilon to denominator for numerical stability
    var epsilon_tensor = full_like(tensor2, epsilon)
    var stable_denominator = add(tensor2, epsilon_tensor)

    return divide(tensor1, stable_denominator)


fn negate_tensor(tensor: AnyTensor) raises -> AnyTensor:
    """Negate all elements of a tensor (multiply by -1).

    Args:
            tensor: Input tensor.

    Returns:
            Negated tensor with opposite signs.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             In BCE loss: negate the sum of terms
            var negated = negate_tensor(sum_terms)
            ```
    """
    var zero = zeros_like(tensor)
    return subtract(zero, tensor)
