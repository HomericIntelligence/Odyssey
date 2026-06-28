"""Lion optimizer (Symbolic Discovery of Optimization Algorithms).

This module provides the Lion optimizer for updating model parameters
during training using signed momentum with efficient memory usage.

Lion is a memory-efficient optimizer that uses the sign of momentum updates
instead of raw momentum values. It achieves lower memory footprint than AdamW
(approximately half) and often provides better generalization.

Standard Lion update rule:
    m = beta1 * m + (1 - beta1) * gradients
    update = sign(m)
    params = params - learning_rate * update - weight_decay * learning_rate * params

Key characteristics:
    - Memory: ~50% of AdamW (only momentum buffer, no second moment)
    - Learning rate: Typically 3-10x lower than AdamW
    - Sensitivity: Requires careful learning rate tuning

Reference:
    Chen, X., Chen, C., Ramaswamy, A., & Darrell, T. (2023).
    Symbolic Discovery of Optimization Algorithms.
    arXiv preprint arXiv:2302.06675
    https://github.com/google/automl/tree/master/lion
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full_like
from projectodyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)
from projectodyssey.core.elementwise import sign


def lion_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.99,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single Lion optimization step - pure functional.

    Returns new parameters and new momentum buffer.
    Caller manages all state.

    WARNING: Lion requires careful learning rate tuning. Typical learning rates
    are 3-10x lower than AdamW. If using AdamW learning rate, reduce by 10x
    as a starting point.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: Momentum buffer (exponential moving average of gradients).
        learning_rate: Step size for parameter updates. Should be 3-10x lower
            than equivalent AdamW learning rate.
        beta1: Exponential decay rate for momentum (default: 0.9).
        beta2: Exponential decay rate for momentum accumulation (default: 0.99).
            Note: beta2 controls momentum smoothing; higher values (closer to 1)
            lead to heavier momentum averaging.
        weight_decay: L2 weight decay factor (default: 0.0).

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error("lion_step: params and gradients must have the same shape")
    if params.shape() != momentum.shape():
        raise Error("lion_step: params and momentum must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error("lion_step: params and gradients must have the same dtype")

    # Update momentum: m = beta2 * m + (1 - beta2) * gradients
    var beta2_tensor = full_like(momentum, beta2)
    var one_minus_beta2 = full_like(momentum, 1.0 - beta2)
    var new_momentum = multiply_simd(beta2_tensor, momentum)
    var grad_contrib = multiply_simd(one_minus_beta2, gradients)
    new_momentum = add_simd(new_momentum, grad_contrib)

    # Compute update: update = sign(beta1 * old_momentum + (1 - beta1) * gradients)
    var beta1_tensor = full_like(momentum, beta1)
    var one_minus_beta1 = full_like(momentum, 1.0 - beta1)
    var momentum_contrib = multiply_simd(beta1_tensor, momentum)
    var grad_contrib2 = multiply_simd(one_minus_beta1, gradients)
    var update_before_sign = add_simd(momentum_contrib, grad_contrib2)
    var update = sign(update_before_sign)

    # Apply update with weight decay: params = params - lr * update - lr * weight_decay * params
    var lr_tensor = full_like(update, learning_rate)
    var lr_update = multiply_simd(lr_tensor, update)
    var new_params = subtract_simd(params, lr_update)

    # Apply weight decay if specified
    if weight_decay != 0.0:
        var wd_coeff_tensor = full_like(params, weight_decay * learning_rate)
        var wd_term = multiply_simd(wd_coeff_tensor, params)
        new_params = subtract_simd(new_params, wd_term)

    return (new_params, new_momentum)


def lion_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified Lion step with default hyperparameters (beta1=0.9, beta2=0.99).

    Convenience wrapper around lion_step for the common case.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: Momentum buffer (exponential moving average of gradients).
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return lion_step(
        params,
        gradients,
        momentum,
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )
