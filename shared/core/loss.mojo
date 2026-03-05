"""Loss functions for training neural networks.

This module provides common loss functions used in supervised learning,
along with their backward passes for gradient computation.

Implemented losses:
- Binary Cross-Entropy (BCE): For binary classification
- Mean Squared Error (MSE): For regression
- Cross-Entropy: For multi-class classification (with softmax)
- Smooth L1 Loss (Huber Loss): Robust regression loss
- Hinge Loss: For Support Vector Machines (SVMs)
- Focal Loss: For addressing class imbalance
- KL Divergence: For distribution matching

All loss functions include:
- Numerical stability (epsilon handling, clipping)
- Proper gradient computation
- Support for batched inputs
"""

from shared.core.extensor import ExTensor, ones_like, zeros_like, full_like
from shared.core.arithmetic import add, subtract, multiply, divide, power
from shared.core.elementwise import log, clip, exp, abs
from shared.core.reduction import mean, sum, max_reduce
from shared.core.activation import softmax
from shared.core.comparison import less, greater
from shared.core.dtype_dispatch import dispatch_binary, dispatch_scalar
from shared.core.dtype_cast import cast_tensor


fn binary_cross_entropy(
    predictions: ExTensor, targets: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Binary cross-entropy loss for binary classification.

        Formula:
            BCE = -[y*log(p) + (1-y)*log(1-p)]

        where:
            p = predictions (should be in [0, 1] range, typically from sigmoid)
            y = targets (ground truth labels, 0 or 1)

    Args:
            predictions: Predicted probabilities, shape (batch_size,) or (batch_size, 1).
            targets: Ground truth binary labels (0 or 1), same shape as predictions.
            epsilon: Small constant for numerical stability (prevents log(0)).

    Returns:
            Loss tensor of same shape as inputs (element-wise loss).
            Use mean() or sum() to get scalar loss for backpropagation.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var predictions = sigmoid(logits)  # (batch_size,)
            var targets = ... # (batch_size,) with values 0 or 1
            var loss_per_sample = binary_cross_entropy(predictions, targets)
            var loss = mean(loss_per_sample)  # Scalar loss for backprop
            ```

        Numerical Stability:
            - Clips predictions to [epsilon, 1-epsilon] to prevent log(0).
            - Uses epsilon=1e-7 by default.
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Clip predictions to prevent log(0) and log(1)
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)

    # Compute log(p) and log(1-p)
    var log_pred = log(clipped)
    var one = ones_like(clipped)

    var one_minus_pred = subtract(one, clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    var term1 = multiply(targets, log_pred)
    var one_minus_targets = subtract(one, targets)
    var term2 = multiply(one_minus_targets, log_one_minus_pred)
    var sum_terms = add(term1, term2)

    # Negate: BCE = -(term1 + term2)
    var zero = zeros_like(sum_terms)

    return subtract(zero, sum_terms)


fn binary_cross_entropy_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    epsilon: Float64 = 1e-7,
) raises -> ExTensor:
    """Backward pass for binary cross-entropy loss.

        Computes gradient of BCE loss with respect to predictions.

        Formula:
            ∂BCE/∂p = -y/p + (1-y)/(1-p)

        Which simplifies to:
            ∂BCE/∂p = (-y(1-p) + (1-y)p) / (p(1-p))
            ∂BCE/∂p = (p - y) / (p(1-p))

        For numerical stability, we add epsilon to the denominator:
            ∂BCE/∂p = (p - y) / (p(1-p) + epsilon)

    Args:
            grad_output: Gradient from upstream (e.g., from mean_backward).
            predictions: Original predictions passed to forward pass.
            targets: Original targets passed to forward pass.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            Gradient with respect to predictions, same shape as predictions.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var bce_loss = binary_cross_entropy(predictions, targets)
            var loss = mean(bce_loss)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_bce = mean_backward(grad_loss, bce_loss)
            var grad_pred = binary_cross_entropy_backward(grad_bce, predictions, targets)
            ```
    """
    # Gradient formula: (p - y) / (p(1-p) + epsilon)
    var one = ones_like(predictions)

    # Numerator: (predictions - targets)
    var numerator = subtract(predictions, targets)

    # Denominator: p(1-p) + epsilon
    # First compute: (1 - predictions)
    var one_minus_pred = subtract(one, predictions)

    # Then compute: predictions * (1 - predictions)
    var pred_times_one_minus_pred = multiply(predictions, one_minus_pred)

    # Create epsilon tensor
    var epsilon_tensor = full_like(predictions, epsilon)

    # Compute: p(1-p) + epsilon
    var denominator = add(pred_times_one_minus_pred, epsilon_tensor)

    # Compute gradient: (p - y) / (p(1-p) + epsilon)
    var grad = divide(numerator, denominator)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn mean_squared_error(
    predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    """Mean squared error loss for regression.

        Formula:
            MSE = (predictions - targets)^2

        Returns element-wise squared error. Use mean() to get scalar loss.

    Args:
            predictions: Predicted values, any shape.
            targets: Ground truth values, same shape as predictions.

    Returns:
            Squared error tensor, same shape as inputs.
            Use mean() or sum() to get scalar loss for backpropagation.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var predictions = model(x)  # (batch_size, output_dim)
            var targets = y_true        # (batch_size, output_dim)
            var loss_per_sample = mean_squared_error(predictions, targets)
            var loss = mean(loss_per_sample)  # Scalar loss
            ```
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # MSE = (predictions - targets)^2
    var diff = subtract(predictions, targets)
    return multiply(diff, diff)


fn mean_squared_error_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    """Backward pass for mean squared error loss.

        Computes gradient of MSE loss with respect to predictions.

        Formula:
            ∂MSE/∂predictions = 2 * (predictions - targets)

    Args:
            grad_output: Gradient from upstream (e.g., from mean_backward).
            predictions: Original predictions passed to forward pass.
            targets: Original targets passed to forward pass.

    Returns:
            Gradient with respect to predictions, same shape as predictions.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var squared_error = mean_squared_error(predictions, targets)
            var loss = mean(squared_error)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_squared_error = mean_backward(grad_loss, squared_error)
            var grad_pred = mean_squared_error_backward(grad_squared_error, predictions, targets)
            ```
    """
    # Gradient: 2 * (predictions - targets)
    var diff = subtract(predictions, targets)

    # Create tensor with value 2.0
    var two = full_like(diff, 2.0)

    var grad = multiply(two, diff)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn cross_entropy(
    logits: ExTensor, targets: ExTensor, axis: Int = -1, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Cross-entropy loss for multi-class classification.

        Formula:
            CE = -sum(targets * log(softmax(logits)))

        This implementation uses the log-sum-exp trick for numerical stability.

    Args:
            logits: Raw model outputs (before softmax), shape (batch_size, num_classes).
            targets: One-hot encoded ground truth, same shape as logits.
            axis: Axis along which to compute softmax (default: -1, last axis).
            epsilon: Small constant for numerical stability in log operations (default: 1e-7).

    Returns:
            Loss tensor, shape depends on reduction.
            Use mean() to get scalar loss for backpropagation.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var logits = model(x)           # (batch_size, num_classes)
            var targets_onehot = ...        # (batch_size, num_classes) one-hot
            var loss_per_sample = cross_entropy(logits, targets_onehot)
            var loss = mean(loss_per_sample)  # Scalar loss
            ```

    Note:
            For efficiency, this does NOT compute softmax explicitly.
            Instead, it uses: CE = -sum(targets * (logits - log_sum_exp(logits))).

        Numerical Stability:
            - Uses log-sum-exp trick to prevent overflow/underflow.
            - Adds epsilon to log argument to prevent log(0).
    """
    if logits.dtype() != targets.dtype():
        raise Error("Logits and targets must have the same dtype")

    if logits.shape() != targets.shape():
        raise Error("Logits and targets must have the same shape")

    # Implement cross-entropy with log-sum-exp trick for numerical stability
    # CE = -sum(targets * (logits - log_sum_exp(logits)))
    # where log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))

    # Assume last axis is the class dimension
    var shape = logits.shape()
    var class_axis = len(shape) - 1 if axis == -1 else axis

    # Step 1: Find max along class axis for numerical stability
    var max_logits = max_reduce(logits, axis=class_axis, keepdims=True)

    # Step 2: Subtract max from logits: x_stable = x - max(x)
    var logits_stable = subtract(logits, max_logits)

    # Step 3: Compute exp(x_stable)
    var exp_logits = exp(logits_stable)

    # Step 4: Sum exp values along class axis
    var sum_exp = sum(exp_logits, axis=class_axis, keepdims=True)

    # Step 5: Compute log_sum_exp = max + log(sum_exp + epsilon)
    # Add epsilon for numerical stability to prevent log(0)
    var epsilon_tensor = full_like(sum_exp, epsilon)
    var sum_exp_stable = add(sum_exp, epsilon_tensor)
    var log_sum_exp = add(max_logits, log(sum_exp_stable))

    # Step 6: Compute log probabilities: log(p) = logits - log_sum_exp
    var log_probs = subtract(logits, log_sum_exp)

    # Step 7: Compute cross-entropy: CE = -sum(targets * log_probs)
    var ce_per_sample = multiply(targets, log_probs)
    var ce_sum = sum(
        ce_per_sample, axis=class_axis, keepdims=False
    )  # Sum over classes

    # Negate: create -1.0 scalar tensor
    var neg_one = full_like(ce_sum, -1.0)
    var ce = multiply(ce_sum, neg_one)

    # Return mean over batch (first dimension)
    return mean(ce, axis=0, keepdims=False)


fn cross_entropy_backward(
    grad_output: ExTensor,
    logits: ExTensor,
    targets: ExTensor,
    epsilon: Float64 = 1e-7,
) raises -> ExTensor:
    """Backward pass for cross-entropy loss.

        For cross-entropy with softmax, the gradient simplifies to:
            `∂CE/∂logits = softmax(logits) - targets`

        This beautiful result comes from the chain rule and the properties
        of the softmax function.

    Args:
        grad_output: Gradient from upstream (scalar for mean reduction).
        logits: Original logits passed to forward pass, shape (batch, num_classes).
        targets: Original one-hot targets, shape (batch, num_classes).
        epsilon: Value to check cross entropy for(optional).

    Returns:
        Gradient with respect to logits, shape (batch, num_classes).

    Raises:
            Error: If operation fails.

    Example:
        ```mojo
        from shared.core import cross_entropy, cross_entropy_backward

        # Forward pass
        var loss = cross_entropy(logits, targets)
        # Backward pass (grad_output is usually 1.0 for scalar loss)
        var grad_logits = cross_entropy_backward(grad_output, logits, targets)
        ```

    Note:
            The gradient is already averaged over the batch if the forward pass
            used mean reduction.
    """
    # Compute softmax probabilities
    var axis = len(logits.shape()) - 1  # Last axis is classes
    var probs = softmax(logits, axis=axis)

    # Gradient: softmax(logits) - targets
    var grad = subtract(probs, targets)

    # Scale by upstream gradient and average over batch
    # Forward pass averages via mean(ce, axis=0), so we divide by batch_size here
    var batch_size = Float32(logits.shape()[0])
    var scale_val = 1.0 / batch_size

    # Create scale tensor with same shape as grad
    var scale_tensor = full_like(grad, Float64(scale_val))

    var grad_scaled = multiply(grad, scale_tensor)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_scaled, grad_output)


fn smooth_l1_loss(
    predictions: ExTensor, targets: ExTensor, beta: Float32 = 1.0
) raises -> ExTensor:
    """Smooth L1 loss (Huber loss) for robust regression.

        Formula:
            If |x| < beta:
                L = 0.5 * (x)^2 / beta
            Else:
                L = |x| - 0.5 * beta

            where x = predictions - targets

        This loss is less sensitive to outliers than MSE, making it more robust
        for regression tasks with noisy data.

    Args:
            predictions: Predicted values, any shape.
            targets: Ground truth values, same shape as predictions.
            beta: Threshold parameter that controls the transition between L2 and L1.
                    Smaller beta makes the function more similar to L1.
                    Default: 1.0.

    Returns:
            Loss tensor, same shape as inputs. Use mean() to get scalar loss.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var predictions = model(x)  # (batch_size, output_dim)
            var targets = y_true        # (batch_size, output_dim)
            var loss_per_sample = smooth_l1_loss(predictions, targets, beta=1.0)
            var loss = mean(loss_per_sample)  # Scalar loss for backprop
            ```

        Numerical Stability:
            - Uses absolute value for robust handling of differences.
            - Beta parameter prevents division by zero in gradient.
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Compute differences: x = predictions - targets
    var diff = subtract(predictions, targets)
    var abs_diff = abs(diff)

    # Create tensors for beta and 0.5 * beta
    var beta_tensor = full_like(diff, Float64(beta))
    var half = full_like(diff, 0.5)
    var half_beta = multiply(beta_tensor, half)

    # Quadratic part: 0.5 * x^2 / beta (for |x| < beta)
    var diff_squared = multiply(diff, diff)
    var quadratic = divide(multiply(half, diff_squared), beta_tensor)

    # Linear part: |x| - 0.5 * beta (for |x| >= beta)
    var linear = subtract(abs_diff, half_beta)

    # Create mask for where |x| < beta
    var is_quadratic_bool = less(abs_diff, beta_tensor)
    # Cast bool mask to same dtype as diff for arithmetic operations
    var is_quadratic = cast_tensor(is_quadratic_bool, diff.dtype())

    # Blend quadratic and linear based on mask: if |x| < beta use quadratic else linear
    # Result = quadratic * mask + linear * (1 - mask)
    var one = ones_like(diff)
    var mask_inv = subtract(one, is_quadratic)
    var result = add(
        multiply(quadratic, is_quadratic), multiply(linear, mask_inv)
    )

    return result


fn smooth_l1_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    beta: Float32 = 1.0,
) raises -> ExTensor:
    """Backward pass for Smooth L1 loss (Huber loss).

        Computes gradient of Smooth L1 loss with respect to predictions.

        Formula:
            If |x| < beta:
                ∂L/∂pred = x / beta
            Else:
                ∂L/∂pred = sign(x)

            where x = predictions - targets, sign(x) = 1 if x > 0 else -1

    Args:
            grad_output: Gradient from upstream (e.g., from mean_backward).
            predictions: Original predictions passed to forward pass.
            targets: Original targets passed to forward pass.
            beta: Threshold parameter (must match forward pass).

    Returns:
            Gradient with respect to predictions, same shape as predictions.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var smoothl1_loss = smooth_l1_loss(predictions, targets, beta=1.0)
            var loss = mean(smoothl1_loss)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_smoothl1 = mean_backward(grad_loss, smoothl1_loss)
            var grad_pred = smooth_l1_loss_backward(grad_smoothl1, predictions, targets, beta=1.0)
            ```
    """
    if grad_output.dtype() != predictions.dtype():
        raise Error(
            "smooth_l1_loss_backward: grad_output and predictions must have"
            " same dtype"
        )
    if grad_output.shape() != predictions.shape():
        raise Error(
            "smooth_l1_loss_backward: grad_output and predictions must have"
            " same shape"
        )

    # Compute differences: x = predictions - targets
    var diff = subtract(predictions, targets)
    var abs_diff = abs(diff)

    # Create tensors for beta
    var beta_tensor = full_like(diff, Float64(beta))
    var one = ones_like(diff)
    var zero = zeros_like(diff)
    var neg_one = full_like(diff, -1.0)

    # Quadratic gradient: x / beta (for |x| < beta)
    var quadratic_grad = divide(diff, beta_tensor)

    # Linear gradient: sign(x) (for |x| >= beta)
    # sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    var is_positive_bool = greater(diff, zero)
    var is_negative_bool = less(diff, zero)
    # Cast bool masks to same dtype as diff for arithmetic operations
    var is_positive = cast_tensor(is_positive_bool, diff.dtype())
    var is_negative = cast_tensor(is_negative_bool, diff.dtype())
    var sign_diff = add(
        multiply(is_positive, one), multiply(is_negative, neg_one)
    )

    # Create mask for where |x| < beta
    var is_quadratic_bool = less(abs_diff, beta_tensor)
    var is_quadratic = cast_tensor(is_quadratic_bool, diff.dtype())
    var mask_inv = subtract(one, is_quadratic)

    # Blend gradients: quadratic_grad if |x| < beta else sign_diff
    var blended_grad = add(
        multiply(quadratic_grad, is_quadratic), multiply(sign_diff, mask_inv)
    )

    # Multiply by upstream gradient
    return multiply(grad_output, blended_grad)


fn hinge_loss(predictions: ExTensor, targets: ExTensor) raises -> ExTensor:
    """Hinge loss for Support Vector Machines (SVMs).

        Formula:
            L = max(0, 1 - y * pred)

        where:
            y = targets (must be -1 or 1)
            pred = predictions (model output)

        The hinge loss penalizes predictions that are not confident enough.
        A prediction is correct when y * pred >= 1 (margin of 1).

    Args:
            predictions: Model predictions (real-valued scores).
            targets: Ground truth labels, must be -1 or 1, same shape as predictions.

    Returns:
            Loss tensor, same shape as inputs. Use mean() to get scalar loss.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var predictions = model(x)  # (batch_size,) or (batch_size, 1)
            var targets = y_true        # (batch_size,) with values -1 or 1
            var loss_per_sample = hinge_loss(predictions, targets)
            var loss = mean(loss_per_sample)  # Scalar loss for backprop
            ```

    Note:
            Hinge loss is typically used with hard labels (-1 or 1) rather than
            probabilities. For multi-class SVM, use with one-vs-rest strategy.

        Numerical Stability:
            - Uses max(0, ...) to prevent negative losses.
            - Avoids numerical issues with extreme values.
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Compute y * pred
    var y_pred = multiply(targets, predictions)

    # Compute margin = 1 - y * pred
    var one = ones_like(y_pred)
    var margin = subtract(one, y_pred)

    # Return max(0, margin)
    var zero = zeros_like(margin)
    var is_positive_bool = greater(margin, zero)
    # Cast bool mask to same dtype as margin for arithmetic operations
    var is_positive = cast_tensor(is_positive_bool, margin.dtype())

    # max(0, margin) = margin * (margin > 0)
    return multiply(margin, is_positive)


fn hinge_loss_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    """Backward pass for hinge loss.

        Computes gradient of hinge loss with respect to predictions.

        Formula:
            If y * pred < 1 (margin violated):
                ∂L/∂pred = -y
            Else (margin satisfied):
                ∂L/∂pred = 0

        where y = targets, pred = predictions

    Args:
            grad_output: Gradient from upstream (e.g., from mean_backward).
            predictions: Original predictions passed to forward pass.
            targets: Original targets passed to forward pass (-1 or 1).

    Returns:
            Gradient with respect to predictions, same shape as predictions.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var hinge = hinge_loss(predictions, targets)
            var loss = mean(hinge)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_hinge = mean_backward(grad_loss, hinge)
            var grad_pred = hinge_loss_backward(grad_hinge, predictions, targets)
            ```
    """
    if grad_output.dtype() != predictions.dtype():
        raise Error(
            "hinge_loss_backward: grad_output and predictions must have same"
            " dtype"
        )
    if grad_output.shape() != predictions.shape():
        raise Error(
            "hinge_loss_backward: grad_output and predictions must have same"
            " shape"
        )

    # Compute y * pred
    var y_pred = multiply(targets, predictions)

    # Create tensors
    var one = ones_like(y_pred)
    var zero = zeros_like(y_pred)
    var neg_one = full_like(targets, -1.0)

    # Check if margin is violated: y * pred < 1
    var margin_violated_bool = less(y_pred, one)
    # Cast bool mask to same dtype as targets for arithmetic operations
    var margin_violated = cast_tensor(margin_violated_bool, targets.dtype())

    # Gradient: if y * pred < 1 then -y else 0
    var neg_targets = multiply(targets, neg_one)
    var hinge_grad = multiply(neg_targets, margin_violated)

    # Multiply by upstream gradient
    return multiply(grad_output, hinge_grad)


fn focal_loss(
    predictions: ExTensor,
    targets: ExTensor,
    alpha: Float32 = 0.25,
    gamma: Float32 = 2.0,
) raises -> ExTensor:
    """Focal loss for addressing class imbalance in classification.

        Formula:
            FL = -alpha * (1 - p)^gamma * target * log(p) - (1 - alpha) * p^gamma * (1 - target) * log(1 - p)

        where:
            p = predictions (probabilities, should be in [0, 1] range)
            target = ground truth labels (0 or 1)
            alpha = weighting factor (default: 0.25)
            gamma = focusing parameter (default: 2.0)

        The focal loss applies a modulating term (1 - p)^gamma to the cross entropy loss.
        This down-weights easy examples and focuses training on hard examples.
        It is particularly useful for addressing class imbalance.

    Args:
            predictions: Predicted probabilities, shape (batch_size,) or (batch_size, 1).
            targets: Ground truth binary labels (0 or 1), same shape as predictions.
            alpha: Weighting factor for class 1 (default: 0.25).
            gamma: Focusing parameter (default: 2.0).

    Returns:
            Loss tensor of same shape as inputs (element-wise loss).
            Use mean() to get scalar loss for backpropagation.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var predictions = sigmoid(logits)  # (batch_size,)
            var targets = ...  # (batch_size,) with values 0 or 1
            var loss_per_sample = focal_loss(predictions, targets)
            var loss = mean(loss_per_sample)  # Scalar loss
            ```

        Numerical Stability:
            - Clips predictions to [epsilon, 1-epsilon] to prevent log(0).
            - Uses epsilon=1e-7 by default.
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    var epsilon = 1e-7

    # Clip predictions to prevent log(0) and log(1)
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)

    # Compute log(p) and log(1-p)
    var log_pred = log(clipped)
    var one = ones_like(clipped)
    var one_minus_pred = subtract(one, clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # Compute alpha and (1-alpha) tensors
    var alpha_tensor = full_like(clipped, Float64(alpha))
    var one_minus_alpha = subtract(one, alpha_tensor)

    # Compute (1 - p)^gamma for positive class
    var gamma_tensor = full_like(clipped, Float64(gamma))
    var one_minus_p_pow = power(one_minus_pred, gamma_tensor)

    # Compute p^gamma for negative class
    var p_pow = power(clipped, gamma_tensor)

    # Focal loss: -alpha * (1-p)^gamma * target * log(p) - (1-alpha) * p^gamma * (1-target) * log(1-p)
    var one_minus_targets = subtract(one, targets)

    # First term: -alpha * (1-p)^gamma * target * log(p)
    var term1 = multiply(alpha_tensor, one_minus_p_pow)
    term1 = multiply(term1, targets)
    term1 = multiply(term1, log_pred)

    # Second term: -(1-alpha) * p^gamma * (1-target) * log(1-p)
    var term2 = multiply(one_minus_alpha, p_pow)
    term2 = multiply(term2, one_minus_targets)
    term2 = multiply(term2, log_one_minus_pred)

    # Combine: FL = -(term1 + term2)
    var sum_terms = add(term1, term2)
    var zero = zeros_like(sum_terms)

    return subtract(zero, sum_terms)


fn focal_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    alpha: Float32 = 0.25,
    gamma: Float32 = 2.0,
) raises -> ExTensor:
    """Backward pass for focal loss.

        Computes gradient of focal loss with respect to predictions.

        The gradient formula for focal loss is:
            dFL/dp = alpha * gamma * (1-p)^(gamma-1) * target * log(p)
                     - alpha * (1-p)^gamma * target / p
                     - (1-alpha) * gamma * p^(gamma-1) * (1-target) * log(1-p)
                     + (1-alpha) * p^gamma * (1-target) / (1-p)

    Args:
            grad_output: Gradient from upstream (e.g., from mean_backward).
            predictions: Original predictions passed to forward pass.
            targets: Original targets passed to forward pass.
            alpha: Weighting factor (default: 0.25).
            gamma: Focusing parameter (default: 2.0).

    Returns:
            Gradient with respect to predictions, same shape as predictions.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var focal = focal_loss(predictions, targets, alpha, gamma)
            var loss = mean(focal)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_focal = mean_backward(grad_loss, focal)
            var grad_pred = focal_loss_backward(grad_focal, predictions, targets, alpha, gamma)
            ```
    """
    var epsilon = 1e-7

    # Clip predictions to prevent division by zero
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)
    var one = ones_like(clipped)
    var one_minus_pred = subtract(one, clipped)

    # Compute tensor versions of alpha and gamma
    var alpha_tensor = full_like(clipped, Float64(alpha))
    var one_minus_alpha = subtract(one, alpha_tensor)
    var gamma_tensor = full_like(clipped, Float64(gamma))
    var gamma_minus_one = subtract(gamma_tensor, ones_like(gamma_tensor))

    # Compute powers for gradient
    # (1-p)^(gamma-1)
    var one_minus_p_pow_gm1 = power(one_minus_pred, gamma_minus_one)
    # p^(gamma-1)
    var p_pow_gm1 = power(clipped, gamma_minus_one)
    # (1-p)^gamma
    var one_minus_p_pow_g = power(one_minus_pred, gamma_tensor)
    # p^gamma
    var p_pow_g = power(clipped, gamma_tensor)

    # Compute log terms
    var log_pred = log(clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # Gradient computation:
    # dFL/dp = alpha * [gamma * (1-p)^(gamma-1) * target * log(p)] - alpha * [(1-p)^gamma * target / p]
    #          - (1-alpha) * [gamma * p^(gamma-1) * (1-target) * log(1-p)] + (1-alpha) * [p^gamma * (1-target) / (1-p)]

    var one_minus_targets = subtract(one, targets)

    # First component: alpha * gamma * (1-p)^(gamma-1) * target * log(p)
    var term1_a = multiply(alpha_tensor, gamma_tensor)
    term1_a = multiply(term1_a, one_minus_p_pow_gm1)
    term1_a = multiply(term1_a, targets)
    term1_a = multiply(term1_a, log_pred)

    # Second component: -alpha * (1-p)^gamma * target / p
    var term1_b = multiply(alpha_tensor, one_minus_p_pow_g)
    term1_b = multiply(term1_b, targets)
    term1_b = divide(term1_b, clipped)
    # Negate term1_b
    var neg_one = full_like(term1_b, -1.0)
    term1_b = multiply(term1_b, neg_one)

    # Combine first part: term1_a + (-term1_b)
    var term1 = add(term1_a, term1_b)

    # Third component: (1-alpha) * gamma * p^(gamma-1) * (1-target) * log(1-p)
    var term2_a = multiply(one_minus_alpha, gamma_tensor)
    term2_a = multiply(term2_a, p_pow_gm1)
    term2_a = multiply(term2_a, one_minus_targets)
    term2_a = multiply(term2_a, log_one_minus_pred)
    # Negate term2_a
    var neg_one_2 = full_like(term2_a, -1.0)
    term2_a = multiply(term2_a, neg_one_2)

    # Fourth component: (1-alpha) * p^gamma * (1-target) / (1-p)
    var term2_b = multiply(one_minus_alpha, p_pow_g)
    term2_b = multiply(term2_b, one_minus_targets)
    term2_b = divide(term2_b, one_minus_pred)

    # Combine second part: (-term2_a) + term2_b
    var term2 = add(term2_a, term2_b)

    # Final gradient
    var grad = add(term1, term2)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn kl_divergence(
    p: ExTensor, q: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Kullback-Leibler divergence loss for distribution matching.

        Formula:
            KL(p||q) = p * log(p / q) = p * (log(p) - log(q))

        where:
            p = reference distribution (target)
            q = approximating distribution (predicted)

        KL divergence measures how much one probability distribution differs from another.
        It is always non-negative and is zero only when p == q almost everywhere.

    Args:
            p: Reference distribution (target), should sum to 1 along class axis.
            q: Approximating distribution (predicted), should sum to 1 along class axis.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            Element-wise KL divergence contribution, same shape as inputs.
            Use sum() or mean() to get scalar loss for backpropagation.

    Raises:
            Error: Shapes don't match or dtypes are incompatible.

        Example:
            ```mojo
            var p_dist = softmax(targets_logits, axis=1)  # (batch_size, num_classes)
            var q_dist = softmax(predictions_logits, axis=1)  # (batch_size, num_classes)
            var kl_per_element = kl_divergence(p_dist, q_dist)  # (batch_size, num_classes)
            var loss_per_sample = sum(kl_per_element, axis=1)  # (batch_size,)
            var loss = mean(loss_per_sample)  # Scalar loss
            ```

    Note:
            This implementation assumes inputs are already probabilities (sum to 1).
            For raw logits, apply softmax first.

        Numerical Stability:
            - Clips both p and q to [epsilon, 1] to prevent log(0).
            - Handles zero probabilities gracefully.
    """
    if p.dtype() != q.dtype():
        raise Error("p and q must have the same dtype")

    if p.shape() != q.shape():
        raise Error("p and q must have the same shape")

    # Clip both distributions to prevent log(0)
    var clipped_p = clip(p, epsilon, 1.0)
    var clipped_q = clip(q, epsilon, 1.0)

    # Compute log(p) and log(q)
    var log_p = log(clipped_p)
    var log_q = log(clipped_q)

    # Compute log(p/q) = log(p) - log(q)
    var log_ratio = subtract(log_p, log_q)

    # Compute p * log(p/q) - element-wise KL contribution
    # Return per-element values (user can sum or mean as needed)
    return multiply(p, log_ratio)


fn kl_divergence_backward(
    grad_output: ExTensor, p: ExTensor, q: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Backward pass for KL divergence loss.

        Computes gradient of KL divergence with respect to q (the predicted distribution).

        Formula:
            dKL/dq = -p / q

        The gradient with respect to p is:
            dKL/dp = log(p) - log(q) + 1 (not used in typical backprop since targets are fixed)

    Args:
            grad_output: Gradient from upstream, same shape as forward output (same as inputs).
            p: Reference distribution passed to forward pass.
            q: Approximating distribution passed to forward pass.
            epsilon: Small constant for numerical stability (default: 1e-7).

    Returns:
            Gradient with respect to q, same shape as q.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
             Forward
            var kl_per_element = kl_divergence(p_dist, q_dist)
            var loss_per_sample = sum(kl_per_element, axis=1)
            var loss = mean(loss_per_sample)

            # Backward
            var grad_loss = ones_like(loss)
            var grad_per_sample = mean_backward(grad_loss, loss_per_sample)
            var grad_per_element = sum_backward(grad_per_sample, kl_per_element)
            var grad_q = kl_divergence_backward(grad_per_element, p_dist, q_dist)
            ```
    """
    # Clip q to prevent division by zero
    var clipped_q = clip(q, epsilon, 1.0)

    # Gradient: -p / q
    var grad = divide(p, clipped_q)
    var neg_one = full_like(grad, -1.0)
    grad = multiply(grad, neg_one)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)
