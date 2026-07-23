"""ADOPT optimizer (Adaptive gradient method with the Optimal convergence rate).

This module provides the ADOPT optimizer for updating model parameters during
training. ADOPT is a modified Adam that achieves the optimal O(1/sqrt(T))
convergence rate for smooth non-convex objectives with any beta2 in [0, 1),
removing Adam's dependence on a problem-specific beta2 bound.

ADOPT makes two changes to Adam:
    1. It removes the current gradient from the second-moment normalization by
       normalizing with the SECOND MOMENT (v is updated AFTER the normalization
       uses it), decorrelating the numerator and denominator.
    2. It applies element-wise clipping to the normalized gradient, which
       stabilizes early-iteration updates when the second-moment estimate is
       still poorly conditioned.

ADOPT update rule (per step t, 1-indexed):
    normalized = clip(gradients / max(sqrt(v), eps), -clip_value, +clip_value)
    m = beta1 * m + (1 - beta1) * normalized
    params = params - learning_rate * m
    v = beta2 * v + (1 - beta2) * gradients^2   (second moment updated LAST)

The second-moment buffer `v` is updated with the current squared gradient only
AFTER it has been used (via its previous value) to normalize the gradient. This
is the key difference from Adam, where the same-step second moment is used.

Weight decay, when nonzero, is decoupled (AdamW-style):
    params = params - learning_rate * weight_decay * params

Key characteristics:
    - Memory: same as Adam (first moment `m` + second moment `v`).
    - Convergence: optimal rate for any beta2 in [0, 1); robust to beta2 choice.
    - Clipping: the clip_value schedule in the paper grows with t; this
      pure-functional step takes a fixed clip_value per call, so the caller may
      pass a step-dependent value (e.g. clip_value = t**0.25) to reproduce the
      paper's schedule, or a large constant to disable clipping.

Reference:
    Taniguchi, S., Harada, K., Minegishi, G., Oshima, Y., Jeong, S. C.,
    Nagahara, G., Iiyama, T., Suzuki, M., Iwasawa, Y., & Matsuo, Y. (2024).
    ADOPT: Modified Adam Can Converge with Any beta2 with the Optimal Rate.
    Advances in Neural Information Processing Systems (NeurIPS) 2024.
    arXiv preprint arXiv:2411.02853
    https://github.com/iShohei220/adopt
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import sqrt, clip


def adopt_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    second_moment: AnyTensor,
    learning_rate: Float64,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.9999,
    epsilon: Float64 = 1e-6,
    clip_value: Float64 = 1.0e30,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single ADOPT optimization step - pure functional.

    Returns new parameters, new first-moment (momentum) buffer, and new
    second-moment buffer. Caller manages all state.

    IMPORTANT: the second-moment buffer passed in must be the value from the
    PREVIOUS step (or a small positive initialization such as gradients^2 on
    step 1 to avoid a zero denominator). ADOPT normalizes the current gradient
    by this previous second moment, then updates the second moment with the
    current squared gradient. On the very first step, initialize `second_moment`
    to the first squared gradient (v_0 = g_0^2) as recommended by the paper so
    that the normalized gradient is well-scaled; passing all-zeros makes the
    step-1 denominator equal to `epsilon` for every coordinate.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m (EMA of normalized gradients).
        second_moment: Second-moment buffer v (EMA of squared gradients) from
            the previous step.
        learning_rate: Step size for parameter updates.
        beta1: Exponential decay rate for the first moment (default: 0.9).
        beta2: Exponential decay rate for the second moment (default: 0.9999).
            ADOPT converges at the optimal rate for ANY beta2 in [0, 1).
        epsilon: Numerical-stability term used as the lower clamp bound on
            sqrt(v) in the denominator, i.e. denom = max(sqrt(v), epsilon)
            (default: 1e-6).
        clip_value: Symmetric element-wise clip bound applied to the normalized
            gradient; values are clamped to [-clip_value, +clip_value]. Pass a
            step-dependent value (e.g. t**0.25) to reproduce the paper's growing
            clip schedule, or leave at the large default to disable clipping.
        weight_decay: Decoupled (AdamW-style) weight decay factor (default: 0.0).

    Returns:
        Tuple of (new_params, new_momentum, new_second_moment).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error("adopt_step: params and gradients must have the same shape")
    if params.shape() != momentum.shape():
        raise Error("adopt_step: params and momentum must have the same shape")
    if params.shape() != second_moment.shape():
        raise Error(
            "adopt_step: params and second_moment must have the same shape"
        )
    if params.dtype() != gradients.dtype():
        raise Error("adopt_step: params and gradients must have the same dtype")

    # Denominator uses the PREVIOUS second moment, clamped below by epsilon:
    #   denom = max(sqrt(v), epsilon)
    # This matches the reference ADOPT (`exp_avg_sq.sqrt().clamp_(min=eps)`),
    # NOT the Adam-style `sqrt(v) + eps`. `clip(x, epsilon, +inf)` is the
    # element-wise clamp-min since Odyssey has no dedicated maximum() op.
    var sqrt_v = sqrt(second_moment)
    var denom = clip(sqrt_v, epsilon, 1.0e30)

    # Normalize the gradient by the previous second moment, then clip.
    var normalized = divide_simd(gradients, denom)
    normalized = clip(normalized, -clip_value, clip_value)

    # First moment: m = beta1 * m + (1 - beta1) * normalized
    var beta1_tensor = full_like(momentum, beta1)
    var one_minus_beta1 = full_like(momentum, 1.0 - beta1)
    var new_momentum = multiply_simd(beta1_tensor, momentum)
    var norm_contrib = multiply_simd(one_minus_beta1, normalized)
    new_momentum = add_simd(new_momentum, norm_contrib)

    # Parameter update: params = params - learning_rate * m
    var lr_tensor = full_like(new_momentum, learning_rate)
    var lr_step = multiply_simd(lr_tensor, new_momentum)
    var new_params = subtract_simd(params, lr_step)

    # Decoupled weight decay (AdamW-style), applied after the gradient step.
    if weight_decay != 0.0:
        var wd_coeff_tensor = full_like(params, weight_decay * learning_rate)
        var wd_term = multiply_simd(wd_coeff_tensor, params)
        new_params = subtract_simd(new_params, wd_term)

    # Second moment updated LAST: v = beta2 * v + (1 - beta2) * gradients^2
    var beta2_tensor = full_like(second_moment, beta2)
    var one_minus_beta2 = full_like(second_moment, 1.0 - beta2)
    var grad_sq = multiply_simd(gradients, gradients)
    var new_second_moment = multiply_simd(beta2_tensor, second_moment)
    var v_contrib = multiply_simd(one_minus_beta2, grad_sq)
    new_second_moment = add_simd(new_second_moment, v_contrib)

    return (new_params, new_momentum, new_second_moment)


def adopt_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    second_moment: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Simplified ADOPT step with default hyperparameters (Taniguchi et al. 2024).

    Convenience wrapper around `adopt_step` for the common case: default betas
    (0.9, 0.9999), epsilon 1e-6, clipping disabled, and no weight decay. Caller
    still manages the momentum and second-moment buffers.

    On the first step, initialize `second_moment` to the first squared gradient
    (v_0 = g_0^2) as recommended by the paper (see `adopt_step`).

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m. Initialize to zeros_like(params).
        second_moment: Second-moment buffer v from the previous step.
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_momentum, new_second_moment).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return adopt_step(params, gradients, momentum, second_moment, learning_rate)


def init_adopt_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the adopt optimizer.

    Returns a `List[List[AnyTensor]]` with outer length == `len(params_list)` (one entry per parameter) and inner length == 2 (one entry per state buffer the optimizer threads across calls).

    Each state buffer starts at zero with the parameter's shape. The dtype matches the parameter, or float64 if `force_f64=True`.

    Args:
        params_list: Model parameters.
        force_f64: Up-cast all state buffers to float64 regardless of param dtype.

    Returns:
        A list of state buffer lists in the same order as `params_list`.

    Raises:
        Error: If shape contracts cannot be honored.
    """
    from odyssey.tensor.tensor_creation import zeros

    var all_states: List[List[AnyTensor]] = []
    for i in range(len(params_list)):
        var p = params_list[i]
        var d = p.dtype() if not force_f64 else DType.float64
        var per: List[AnyTensor] = []
        for _ in range(2):
            per.append(zeros(p.shape(), d))
        all_states.append(per^)
    return all_states^
