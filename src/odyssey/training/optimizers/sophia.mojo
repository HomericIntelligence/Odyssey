"""Sophia optimizer (Second-order Clipped Stochastic Optimization).

Sophia is a light-weight second-order optimizer that uses a cheap, infrequent
diagonal Hessian estimate to precondition an EMA of gradients, then CLIPS the
per-coordinate update. The clipping bounds the influence of coordinates with a
small or mis-estimated Hessian, giving second-order-like convergence at close to
first-order cost.

This module implements the per-step update as a pure function; the caller
supplies the diagonal Hessian estimate `hessian` (from either the Hutchinson
estimator, Sophia-H, or the Gauss-Newton-Bartlett estimator, Sophia-G). The
Hessian is expensive to compute, so in practice it is refreshed only every
`update_period` steps; between refreshes the caller passes the same stale
Hessian (its EMA `hessian_moment` simply re-averages the same value), which is
exactly what the reference implementations do.

Sophia update rule (per step):
    momentum       = beta1 * momentum + (1 - beta1) * gradients
    hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian
    update         = clip(momentum / max(hessian_moment, eps), -rho, +rho)
    params         = params - learning_rate * update

with optional decoupled (AdamW-style) weight decay applied after the step.

Key characteristics:
    - Memory: first-moment `momentum` + diagonal-Hessian EMA `hessian_moment`
      (same footprint as Adam's two buffers).
    - The clip threshold `rho` (paper default 0.01-0.05) is the fraction of
      coordinates left un-clipped; it is the single most important hyperparameter.
    - The denominator is clamped below by `eps` (not `+ eps`), so a zero/negative
      Hessian estimate degrades gracefully to a large-but-clipped update.

Reference:
    Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. (2023).
    Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
    Pre-training. arXiv preprint arXiv:2305.14342.
    https://github.com/Liuhong99/Sophia
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import clip


def sophia_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    hessian_moment: AnyTensor,
    # `hessian` and `beta2` are consumed by the companion
    # `sophia_update_hessian_moment` (which owns the Hessian-EMA refresh), not by
    # the step body itself; they are accepted here for signature symmetry so the
    # two functions share one call shape.
    hessian: AnyTensor,
    learning_rate: Float64,
    beta1: Float64 = 0.96,
    beta2: Float64 = 0.99,
    rho: Float64 = 0.04,
    epsilon: Float64 = 1e-12,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single Sophia optimization step - pure functional.

    Returns new parameters and the new first-moment (momentum) buffer. The
    second buffer, `hessian_moment`, is updated IN THE CALLER using
    `sophia_update_hessian_moment` on the steps where a fresh `hessian` estimate
    is available (typically every `update_period` steps). This split keeps the
    signature honest: `sophia_step` reads the current `hessian_moment` but does
    not decide the Hessian refresh schedule, which the caller owns.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m (EMA of gradients).
        hessian_moment: EMA of the diagonal Hessian estimate (current value).
        hessian: The diagonal Hessian estimate for this step. On steps where no
            fresh estimate is computed, pass the previous `hessian_moment` (or a
            zero tensor) — the update simply uses the standing `hessian_moment`.
        learning_rate: Step size for parameter updates.
        beta1: EMA decay for the first moment (default: 0.96).
        beta2: EMA decay for the Hessian moment (default: 0.99).
        rho: Symmetric clip bound on the per-coordinate update (default: 0.04).
        epsilon: Lower clamp bound on `hessian_moment` in the denominator,
            i.e. denom = max(hessian_moment, epsilon) (default: 1e-12).
        weight_decay: Decoupled (AdamW-style) weight decay factor (default: 0.0).

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error(
            "sophia_step: params and gradients must have the same shape"
        )
    if params.shape() != momentum.shape():
        raise Error("sophia_step: params and momentum must have the same shape")
    if params.shape() != hessian_moment.shape():
        raise Error(
            "sophia_step: params and hessian_moment must have the same shape"
        )
    if params.dtype() != gradients.dtype():
        raise Error(
            "sophia_step: params and gradients must have the same dtype"
        )

    # First moment: m = beta1 * m + (1 - beta1) * gradients
    var beta1_tensor = full_like(momentum, beta1)
    var one_minus_beta1 = full_like(momentum, 1.0 - beta1)
    var new_momentum = multiply_simd(beta1_tensor, momentum)
    var grad_contrib = multiply_simd(one_minus_beta1, gradients)
    new_momentum = add_simd(new_momentum, grad_contrib)

    # Preconditioned, clipped update:
    #   update = clip(m / max(hessian_moment, eps), -rho, +rho)
    var denom = clip(hessian_moment, epsilon, 1.0e30)
    var raw_update = divide_simd(new_momentum, denom)
    var update = clip(raw_update, -rho, rho)

    # Parameter update: params = params - learning_rate * update
    var lr_tensor = full_like(update, learning_rate)
    var lr_update = multiply_simd(lr_tensor, update)
    var new_params = subtract_simd(params, lr_update)

    # Decoupled weight decay (AdamW-style), applied after the gradient step.
    if weight_decay != 0.0:
        var wd_coeff_tensor = full_like(params, weight_decay * learning_rate)
        var wd_term = multiply_simd(wd_coeff_tensor, params)
        new_params = subtract_simd(new_params, wd_term)

    return (new_params, new_momentum)


def sophia_update_hessian_moment(
    hessian_moment: AnyTensor,
    hessian: AnyTensor,
    beta2: Float64 = 0.99,
) raises -> AnyTensor:
    """Update the diagonal-Hessian EMA with a fresh estimate.

    Call this only on the steps where a fresh Hessian estimate is available
    (every `update_period` steps in the reference implementation):

        hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian

    Args:
        hessian_moment: Current Hessian EMA buffer.
        hessian: Fresh diagonal Hessian estimate.
        beta2: EMA decay for the Hessian moment (default: 0.99).

    Returns:
        The updated hessian_moment buffer.

    Raises:
        Error: If shapes don't match.
    """
    if hessian_moment.shape() != hessian.shape():
        raise Error(
            "sophia_update_hessian_moment: hessian_moment and hessian must have"
            " the same shape"
        )
    var beta2_tensor = full_like(hessian_moment, beta2)
    var one_minus_beta2 = full_like(hessian_moment, 1.0 - beta2)
    var new_hm = multiply_simd(beta2_tensor, hessian_moment)
    var h_contrib = multiply_simd(one_minus_beta2, hessian)
    return add_simd(new_hm, h_contrib)


def sophia_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    hessian_moment: AnyTensor,
    hessian: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified Sophia step with default hyperparameters (Liu et al. 2023).

    Convenience wrapper around `sophia_step` with the paper defaults
    (betas (0.96, 0.99), rho 0.04, eps 1e-12, no weight decay). Caller manages
    the momentum and hessian_moment buffers (see `sophia_update_hessian_moment`).

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m. Initialize to zeros_like(params).
        hessian_moment: Diagonal-Hessian EMA buffer.
        hessian: Diagonal Hessian estimate for this step.
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return sophia_step(
        params, gradients, momentum, hessian_moment, hessian, learning_rate
    )
