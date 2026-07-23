"""FTRL-Proximal optimizer (Follow-The-Regularized-Leader).

This module provides the FTRL-Proximal optimizer for updating model parameters
during training. FTRL-Proximal (McMahan et al. 2013) is the online-learning
optimizer behind Google's click-through-rate models: it accumulates a
per-coordinate linearized gradient sum and, at each step, solves a closed-form
regularized objective that yields SPARSE weights via L1 (lasso) shrinkage.

FTRL-Proximal update rule (per coordinate i, per step; 1-indexed):
    sigma_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha       # per-coord learning-rate delta
    z_i    += g_i - sigma_i * w_i                           # linearized weight-sum
    n_i    += g_i^2                                         # sum of squared gradients
    w_i     = 0                                   if |z_i| <= lambda1
              -(z_i - sign(z_i) * lambda1)
              / ((beta + sqrt(n_i)) / alpha + lambda2)      otherwise

where `z` (the linearized gradient/weight sum) and `n` (the sum of squared
gradients) are the two per-parameter state buffers the caller threads across
steps, and:
    - alpha    is the per-coordinate learning-rate scale,
    - beta     smooths the adaptive per-coordinate rate (paper default 1.0),
    - lambda1  is the L1 (sparsity) regularization strength,
    - lambda2  is the L2 regularization strength.

The `w_i = 0 if |z_i| <= lambda1` branch is what makes FTRL produce exact zeros
(sparsity), unlike Adam/SGD. This implementation expresses the whole update
element-wise with vectorized ops — the soft-threshold magnitude
`max(|z| - lambda1, 0)` is exactly zero on the `|z| <= lambda1` branch, so the
piecewise rule collapses to a single closed form with no per-element control
flow:
    w = -sign(z) * clip(|z| - lambda1, 0, +inf)
        / ((beta + sqrt(n)) / alpha + lambda2)

Note the standard `learning_rate` argument multiplies the final weight so the
optimizer fits the shared `<name>_step(..., learning_rate, ...)` signature;
setting it to 1.0 recovers textbook FTRL (whose step size lives in `alpha`).

Key characteristics:
    - Memory: two buffers per parameter (`z` and `n`), same footprint as Adam.
    - Sparsity: L1 term drives coordinates to EXACT zero (feature selection).
    - Online: designed for one-pass streaming; robust to non-stationary data.

Reference:
    McMahan, H. B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J.,
    Nie, L., Phillips, T., Davydov, E., Golovin, D., Chikkerur, S., Liu, D.,
    Wattenberg, M., Hrafnkelsson, A. M., Boulos, T., & Kubica, J. (2013).
    Ad Click Prediction: a View from the Trenches. Proceedings of the 19th ACM
    SIGKDD International Conference on Knowledge Discovery and Data Mining
    (KDD '13), Algorithm 1 (Per-Coordinate FTRL-Proximal with L1 and L2).
    https://research.google.com/pubs/archive/41159.pdf
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import sqrt, abs, sign, clip, negate


def ftrl_step(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    n: AnyTensor,
    learning_rate: Float64,
    alpha: Float64 = 0.1,
    beta: Float64 = 1.0,
    lambda1: Float64 = 0.0,
    lambda2: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single FTRL-Proximal optimization step - pure functional.

    Returns new parameters and the two updated state buffers `(z, n)`. The
    caller manages all state; initialize both `z` and `n` to zeros on step 1.

    Unlike SGD/Adam, FTRL RECOMPUTES the weight from `z` and `n` each step
    rather than incrementing it, so the incoming `params` value is not read for
    the update magnitude — it is only used to validate shape/dtype. Pass the
    current params for that validation; the returned params are the freshly
    solved FTRL weights.

    Args:
        params: Model parameters (used for shape/dtype validation; FTRL solves
            the new weights from `z`/`n`, it does not read the old magnitude).
        gradients: Gradients of loss with respect to params.
        z: Linearized gradient/weight-sum buffer (per-coordinate). Zeros on
            step 1.
        n: Sum-of-squared-gradients buffer (per-coordinate). Zeros on step 1.
        learning_rate: Global step-size multiplier applied to the solved weight
            (1.0 = textbook FTRL, whose per-coordinate rate lives in `alpha`).
        alpha: Per-coordinate learning-rate scale (default: 0.1).
        beta: Smoothing constant for the adaptive per-coordinate rate
            (default: 1.0, the paper's default).
        lambda1: L1 (lasso) regularization strength — drives weights to exact
            zero (default: 0.0).
        lambda2: L2 regularization strength (default: 0.0).

    Returns:
        Tuple of (new_params, new_z, new_n).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error("ftrl_step: params and gradients must have the same shape")
    if params.shape() != z.shape():
        raise Error("ftrl_step: params and z must have the same shape")
    if params.shape() != n.shape():
        raise Error("ftrl_step: params and n must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error("ftrl_step: params and gradients must have the same dtype")

    # --- Update the state buffers ---------------------------------------------
    # sigma = (sqrt(n + g^2) - sqrt(n)) / alpha
    var grad_sq = multiply_simd(gradients, gradients)
    var n_new = add_simd(n, grad_sq)
    var sqrt_n = sqrt(n)
    var sqrt_n_new = sqrt(n_new)
    var alpha_tensor = full_like(n, alpha)
    var sigma = divide_simd(subtract_simd(sqrt_n_new, sqrt_n), alpha_tensor)

    # z += g - sigma * w   (w = the OLD weights = incoming params)
    var sigma_w = multiply_simd(sigma, params)
    var z_new = add_simd(z, subtract_simd(gradients, sigma_w))

    # --- Solve the new weights from (z, n) ------------------------------------
    # numerator (soft-threshold): -sign(z) * max(|z| - lambda1, 0)
    #   max(|z| - lambda1, 0) is exactly 0 on the |z| <= lambda1 branch, so the
    #   piecewise `w = 0 if |z| <= lambda1` rule needs no control flow.
    var abs_z = abs(z_new)
    var lambda1_tensor = full_like(z_new, lambda1)
    var shrunk = clip(subtract_simd(abs_z, lambda1_tensor), 0.0, 1.0e30)
    var numerator = negate(multiply_simd(sign(z_new), shrunk))

    # denominator: (beta + sqrt(n_new)) / alpha + lambda2   (always > 0)
    var beta_tensor = full_like(n_new, beta)
    var denom = divide_simd(add_simd(beta_tensor, sqrt_n_new), alpha_tensor)
    denom = add_simd(denom, full_like(n_new, lambda2))

    var new_params = divide_simd(numerator, denom)

    # Global learning-rate multiplier (1.0 = textbook FTRL).
    if learning_rate != 1.0:
        var lr_tensor = full_like(new_params, learning_rate)
        new_params = multiply_simd(lr_tensor, new_params)

    return (new_params, z_new, n_new)


def ftrl_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    n: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Simplified FTRL-Proximal step with default hyperparameters.

    Convenience wrapper around `ftrl_step` for the common case: alpha 0.1,
    beta 1.0, and NO regularization (lambda1 = lambda2 = 0). With no L1 term the
    output is dense (no sparsity); pass `ftrl_step` with a nonzero `lambda1` to
    enable feature-selecting sparsity. Caller still manages the `z` and `n`
    buffers (both zeros on step 1).

    Args:
        params: Model parameters (used for shape/dtype validation).
        gradients: Gradients of loss with respect to params.
        z: Linearized gradient/weight-sum buffer. Zeros on step 1.
        n: Sum-of-squared-gradients buffer. Zeros on step 1.
        learning_rate: Global step-size multiplier on the solved weight.

    Returns:
        Tuple of (new_params, new_z, new_n).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return ftrl_step(
        params,
        gradients,
        z,
        n,
        learning_rate,
        alpha=0.1,
        beta=1.0,
        lambda1=0.0,
        lambda2=0.0,
    )


def init_ftrl_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the ftrl optimizer.

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
