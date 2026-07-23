"""Sophia clipped preconditioned update step (arXiv:2305.14342).

SCOPE: this module implements the Sophia-style CLIPPED PRECONDITIONED UPDATE
STEP only, operating on a CALLER-SUPPLIED diagonal-Hessian estimate. The
estimators that produce that estimate in the paper — Sophia-H (Hutchinson) and
Sophia-G (Gauss-Newton-Bartlett) — both require Hessian-vector products /
higher-order autodiff, which Odyssey's autograd does not provide yet (tracked
in `src/odyssey/autograd/TODO.md`, Phase 3 "Higher-order gradients" / "Hessian
computation"). Until then,
callers must obtain the diagonal Hessian estimate elsewhere and pass it in.

Sophia (Liu et al. 2023) preconditions an EMA of gradients with a cheap,
infrequently refreshed diagonal-Hessian EMA, then CLIPS the per-coordinate
update. The clipping bounds the influence of coordinates with a small or
mis-estimated Hessian, giving second-order-like convergence at close to
first-order cost.

Update rule implemented here (per step):
    momentum       = beta1 * momentum + (1 - beta1) * gradients
    update         = clip(momentum / max(gamma * hessian_moment, eps), -rho, +rho)
    params         = params - learning_rate * update

and, on the steps where the caller has a fresh Hessian estimate (every
`update_period` steps in the reference implementations):
    hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian
via `sophia_update_hessian_moment`.

On the gamma / rho parametrization: the paper's Algorithm 3 writes the update
as clip(m_t / max(gamma * h_t, eps), 1), and the official implementation
(github.com/Liuhong99/Sophia, sophia.py) computes
`ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)` — i.e.
gamma = rho * bs (bs = batch size), so gamma has NO batch-size-free universal
default. This module therefore exposes gamma explicitly with default 1.0,
which is exactly the SophiaH parametrization used by pytorch_optimizer
(kozistr/pytorch_optimizer): the +-rho clip carries the threshold and callers
fold any gamma = rho * bs scale into the Hessian estimate they supply. To
reproduce the paper's Algorithm 3 form verbatim, pass gamma explicitly and set
rho = 1.

Key characteristics:
    - Memory: first-moment `momentum` + diagonal-Hessian EMA `hessian_moment`
      (same footprint as Adam's two buffers).
    - The clip threshold `rho` (paper default 0.01-0.05) bounds every
      per-coordinate update; it is the single most important hyperparameter.
    - The denominator is clamped below by `eps` (not `+ eps`), so a
      zero/negative Hessian estimate degrades gracefully to a
      large-but-clipped update.

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
    learning_rate: Float64,
    beta1: Float64 = 0.96,
    gamma: Float64 = 1.0,
    rho: Float64 = 0.04,
    epsilon: Float64 = 1e-12,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single Sophia clipped-preconditioned update step.

    Pure functional. Returns new parameters and the new first-moment
    (momentum) buffer. The preconditioner `hessian_moment` is consumed as-is;
    it is refreshed IN THE CALLER via `sophia_update_hessian_moment` on the
    steps where a fresh caller-supplied diagonal-Hessian estimate is available
    (typically every `update_period` steps). This split keeps the signature
    honest: every parameter of this function participates in the update, and
    the Hessian refresh schedule — like the Hessian estimator itself (Sophia-H
    / Sophia-G, not implemented in Odyssey; see module docstring) — is owned
    by the caller.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m (EMA of gradients).
        hessian_moment: Diagonal-Hessian EMA used as the preconditioner,
            maintained by the caller via `sophia_update_hessian_moment`.
        learning_rate: Step size for parameter updates.
        beta1: EMA decay for the first moment (default: 0.96).
        gamma: Scale on the preconditioner in the denominator,
            denom = max(gamma * hessian_moment, epsilon). The official Liu et
            al. code uses gamma = rho * bs (batch-size coupled, no universal
            default); the default 1.0 is the SophiaH/pytorch_optimizer
            parametrization where callers fold that scale into the Hessian
            estimate they supply. See module docstring.
        rho: Symmetric clip bound on the per-coordinate update (default: 0.04,
            the official SophiaG default).
        epsilon: Lower clamp bound on the scaled denominator,
            i.e. denom = max(gamma * hessian_moment, epsilon) (default: 1e-12).
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
    #   update = clip(m / max(gamma * hessian_moment, eps), -rho, +rho)
    var gamma_tensor = full_like(hessian_moment, gamma)
    var scaled_hm = multiply_simd(gamma_tensor, hessian_moment)
    var denom = clip(scaled_hm, epsilon, 1.0e30)
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
    """Update the diagonal-Hessian EMA with a fresh caller-supplied estimate.

    Call this only on the steps where a fresh diagonal-Hessian estimate is
    available (every `update_period` steps in the reference implementations):

        hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian

    Note that Odyssey does not implement the Sophia-H (Hutchinson) or Sophia-G
    (Gauss-Newton-Bartlett) estimators that produce `hessian` in the paper —
    they need higher-order autodiff, tracked in `src/odyssey/autograd/TODO.md`
    (Phase 3, "Higher-order gradients"). The estimate is the caller's to supply.

    Args:
        hessian_moment: Current Hessian EMA buffer.
        hessian: Fresh caller-supplied diagonal Hessian estimate.
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
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified Sophia update step with default hyperparameters.

    Convenience wrapper around `sophia_step` with the defaults documented
    there (beta1 0.96, gamma 1.0, rho 0.04, eps 1e-12, no weight decay).
    Caller manages the momentum and hessian_moment buffers (see
    `sophia_update_hessian_moment`) and supplies its own diagonal-Hessian
    estimates — the Sophia-H/G estimators are not part of this module.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        momentum: First-moment buffer m. Initialize to zeros_like(params).
        hessian_moment: Diagonal-Hessian EMA buffer (the preconditioner).
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return sophia_step(
        params, gradients, momentum, hessian_moment, learning_rate
    )


def init_sophia_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the sophia optimizer.

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
