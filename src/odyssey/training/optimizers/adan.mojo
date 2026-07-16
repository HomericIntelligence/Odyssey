"""Adan optimizer (Adaptive Nesterov Momentum).

Adan reformulates Nesterov momentum for adaptive optimizers: it tracks an EMA of
the gradient (`exp_avg`), an EMA of the gradient DIFFERENCE `g_t - g_{t-1}`
(`exp_avg_diff`), and an EMA of the squared "look-ahead" gradient
`g_t + beta2*(g_t - g_{t-1})` (`exp_avg_sq`). The parameter step combines the
gradient-EMA and the difference-EMA, each bias-corrected and preconditioned by
the square-root of the second-moment EMA. Adan converges faster and is more
robust to the learning rate than Adam/AdamW across CV, NLP, and RL.

Adan update rule (per step t, 1-indexed):
    grad_diff = grad - prev_grad
    exp_avg      = beta1 * exp_avg      + (1 - beta1) * grad
    exp_avg_diff = beta2 * exp_avg_diff + (1 - beta2) * grad_diff
    u            = grad + beta2 * grad_diff
    exp_avg_sq   = beta3 * exp_avg_sq   + (1 - beta3) * u^2
    bc1 = 1 - beta1^t ;  bc2 = 1 - beta2^t ;  bc3 = 1 - beta3^t
    denom  = sqrt(exp_avg_sq) / sqrt(bc3) + eps
    params = params
             - (lr / bc1) * (exp_avg / denom)
             - (lr * beta2 / bc2) * (exp_avg_diff / denom)
    prev_grad = grad   (stored for the next step)

Key characteristics:
    - Three betas (default 0.98, 0.92, 0.99) and three buffers (exp_avg,
      exp_avg_diff, exp_avg_sq) — one more buffer than Adam.
    - The caller tracks the integer step `t` (for bias correction) and the
      previous gradient `prev_grad`. On step 1, pass prev_grad = grad so that
      grad_diff is zero (the reference initializes `previous_grad` to the first
      gradient, i.e. a zero initial difference).

Reference:
    Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2022). Adan: Adaptive
    Nesterov Momentum Algorithm for Faster Optimizing Deep Models.
    arXiv preprint arXiv:2208.06677.
    https://github.com/sail-sg/Adan
"""

from std.math import log, exp, sqrt as scalar_sqrt

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import sqrt


def _bias_correction(beta: Float64, step: Int) -> Float64:
    """Adam-style bias-correction factor 1 - beta^step (scalar)."""
    var beta_pow = exp(Float64(step) * log(beta))
    return 1.0 - beta_pow


def adan_step(
    params: AnyTensor,
    gradients: AnyTensor,
    exp_avg: AnyTensor,
    exp_avg_diff: AnyTensor,
    exp_avg_sq: AnyTensor,
    prev_grad: AnyTensor,
    step: Int,
    learning_rate: Float64,
    beta1: Float64 = 0.98,
    beta2: Float64 = 0.92,
    beta3: Float64 = 0.99,
    epsilon: Float64 = 1e-8,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single Adan optimization step - pure functional.

    Returns (new_params, new_exp_avg, new_exp_avg_diff, new_exp_avg_sq,
    new_prev_grad). Caller manages all state, including the integer step `t`
    (for bias correction) and `prev_grad` (the previous step's gradient).

    On the FIRST step (t == 1), pass `prev_grad = gradients` so the gradient
    difference is zero (matching the reference, which initializes the previous
    gradient to the first gradient). On subsequent steps, pass the `new_prev_grad`
    returned by the previous call.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        exp_avg: EMA of the gradient (first moment).
        exp_avg_diff: EMA of the gradient difference g_t - g_{t-1}.
        exp_avg_sq: EMA of the squared look-ahead gradient.
        prev_grad: Gradient from the previous step (== gradients on step 1).
        step: 1-indexed step counter (used for bias correction).
        learning_rate: Step size for parameter updates.
        beta1: EMA decay for the gradient (default: 0.98).
        beta2: EMA decay for the gradient difference (default: 0.92).
        beta3: EMA decay for the squared look-ahead gradient (default: 0.99).
        epsilon: Numerical-stability term added to the denominator (default: 1e-8).
        weight_decay: Decoupled (AdamW-style) weight decay factor (default: 0.0).

    Returns:
        Tuple of (new_params, new_exp_avg, new_exp_avg_diff, new_exp_avg_sq,
        new_prev_grad).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error("adan_step: params and gradients must have the same shape")
    if params.shape() != exp_avg.shape():
        raise Error("adan_step: params and exp_avg must have the same shape")
    if params.shape() != exp_avg_diff.shape():
        raise Error(
            "adan_step: params and exp_avg_diff must have the same shape"
        )
    if params.shape() != exp_avg_sq.shape():
        raise Error("adan_step: params and exp_avg_sq must have the same shape")
    if params.shape() != prev_grad.shape():
        raise Error("adan_step: params and prev_grad must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error("adan_step: params and gradients must have the same dtype")

    # grad_diff = grad - prev_grad
    var grad_diff = subtract_simd(gradients, prev_grad)

    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    var beta1_t = full_like(exp_avg, beta1)
    var one_minus_beta1 = full_like(exp_avg, 1.0 - beta1)
    var new_exp_avg = add_simd(
        multiply_simd(beta1_t, exp_avg),
        multiply_simd(one_minus_beta1, gradients),
    )

    # exp_avg_diff = beta2 * exp_avg_diff + (1 - beta2) * grad_diff
    var beta2_t = full_like(exp_avg_diff, beta2)
    var one_minus_beta2 = full_like(exp_avg_diff, 1.0 - beta2)
    var new_exp_avg_diff = add_simd(
        multiply_simd(beta2_t, exp_avg_diff),
        multiply_simd(one_minus_beta2, grad_diff),
    )

    # u = grad + beta2 * grad_diff ; exp_avg_sq = beta3*eas + (1-beta3)*u^2
    var beta2_u = full_like(grad_diff, beta2)
    var u = add_simd(gradients, multiply_simd(beta2_u, grad_diff))
    var beta3_t = full_like(exp_avg_sq, beta3)
    var one_minus_beta3 = full_like(exp_avg_sq, 1.0 - beta3)
    var new_exp_avg_sq = add_simd(
        multiply_simd(beta3_t, exp_avg_sq),
        multiply_simd(one_minus_beta3, multiply_simd(u, u)),
    )

    # Bias corrections and denom = sqrt(exp_avg_sq)/sqrt(bc3) + eps
    var bc1 = _bias_correction(beta1, step)
    var bc2 = _bias_correction(beta2, step)
    var bc3 = _bias_correction(beta3, step)
    var inv_sqrt_bc3 = full_like(new_exp_avg_sq, 1.0 / scalar_sqrt(bc3))
    var denom_core = multiply_simd(sqrt(new_exp_avg_sq), inv_sqrt_bc3)
    var eps_t = full_like(denom_core, epsilon)
    var denom = add_simd(denom_core, eps_t)

    # params -= (lr/bc1)*(exp_avg/denom) + (lr*beta2/bc2)*(exp_avg_diff/denom)
    var step_avg = multiply_simd(
        full_like(new_exp_avg, learning_rate / bc1),
        divide_simd(new_exp_avg, denom),
    )
    var step_diff = multiply_simd(
        full_like(new_exp_avg_diff, learning_rate * beta2 / bc2),
        divide_simd(new_exp_avg_diff, denom),
    )
    var new_params = subtract_simd(params, step_avg)
    new_params = subtract_simd(new_params, step_diff)

    # Decoupled weight decay (AdamW-style), applied after the gradient step.
    if weight_decay != 0.0:
        var wd_coeff = full_like(params, weight_decay * learning_rate)
        new_params = subtract_simd(new_params, multiply_simd(wd_coeff, params))

    # prev_grad for the next step is a copy of the current gradient.
    var new_prev_grad = add_simd(gradients, full_like(gradients, 0.0))

    return (
        new_params,
        new_exp_avg,
        new_exp_avg_diff,
        new_exp_avg_sq,
        new_prev_grad,
    )


def adan_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    exp_avg: AnyTensor,
    exp_avg_diff: AnyTensor,
    exp_avg_sq: AnyTensor,
    prev_grad: AnyTensor,
    step: Int,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Simplified Adan step with default hyperparameters.

    Convenience wrapper around `adan_step` with the paper's default betas
    (0.98, 0.92, 0.99), epsilon = 1e-8, and no weight decay. The caller still
    manages all state, including the integer step `t` and `prev_grad` (pass
    `prev_grad = gradients` on step 1 so the initial gradient difference is
    zero).

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        exp_avg: EMA of the gradient (first moment).
        exp_avg_diff: EMA of the gradient difference g_t - g_{t-1}.
        exp_avg_sq: EMA of the squared look-ahead gradient.
        prev_grad: Gradient from the previous step (== gradients on step 1).
        step: 1-indexed step counter (used for bias correction).
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_exp_avg, new_exp_avg_diff, new_exp_avg_sq,
        new_prev_grad).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return adan_step(
        params,
        gradients,
        exp_avg,
        exp_avg_diff,
        exp_avg_sq,
        prev_grad,
        step,
        learning_rate=learning_rate,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
    )
