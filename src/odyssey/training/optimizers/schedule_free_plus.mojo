"""ScheduleFree+ optimizer (large-batch-stable schedule-free).

ScheduleFree+ is the large-batch-stability variant of Schedule-Free. The base
Schedule-Free method removes explicit learning-rate schedules via online iterate
averaging over the fast sequence `z` and the running average `x`, evaluating the
gradient at an interpolated query point `y`. At LLM / large-batch scale the base
method can diverge and its uniform averaging under-performs on long runs.
ScheduleFree+ adds three stability mechanisms — this module transcribes them
EXACTLY as written in the tracking issue (mvillmow/Random#78), whose rule wins
over any base-paper ambiguity:

    (1) Inner momentum — a momentum buffer `m` is accumulated INSIDE the base-seq
        update, and the fast sequence `z` steps on the momentum buffer (not the
        raw gradient). Momentum helps large-batch training where the raw-gradient
        step diverges.

            m_{t+1} = mu * m_t + (1 - mu) * g_t
            z_{t+1} = z_t - lr_t * m_{t+1}

    (2) Polyak step-size — the per-step step size `lr_t` is a scaled Polyak step
        formed from the (caller-supplied) globally-reduced objective `f(y_t)`,
        the correlation between the gradient and `(z_t - x_t)`, and an L1 EMA of
        the gradient magnitude:

            gnorm_t = rho * gnorm_{t-1} + (1 - rho) * mean(|g_t|)   (L1-EMA norm)
            polyak  = max(0, f(y_t) + beta_sf * <g_t, z_t - x_t>) / (gnorm_t + eps)
            lr_t    = learning_rate * polyak

    (3) Increasing outer momentum — the outer momentum `beta_out` (the
        interpolation / averaging weight) anneals from `beta_sf` up to `beta_max`
        across the run, which helps long training runs:

            frac      = min(1, (t - 1) / max(1, horizon - 1))
            beta_out  = beta_sf + (beta_max - beta_sf) * frac
            x_{t+1}   = beta_out * x_t + (1 - beta_out) * z_{t+1}
            y_{t+1}   = (1 - beta_out) * z_{t+1} + beta_out * x_{t+1}  (next query)

`z`, `x`, and `m` are the caller-managed persisted state; `gnorm` (a scalar L1
EMA) is threaded through the return; `y` is recomputed each step. The model does
its forward/backward pass at `y` (the "train" buffer), while `x` is the
"eval"/checkpoint buffer. On the FIRST step, initialize `z = x = params`,
`m = 0`, and `gnorm = 0.0` (so `y_1 = params`). The Polyak numerator requires the
globally-reduced objective `f(y_t)` at the current query point — the caller
supplies it (per the issue's "Polyak step needs the globally-reduced objective
f(y_t)").

Key characteristics:
    - Three persisted buffers (`z`, `x`, `m`) plus one scalar (`gnorm`) — one
      buffer more than base Schedule-Free (the added inner-momentum buffer).
    - Anytime: `x` is a valid checkpoint at every step; no horizon/schedule for
      the LR itself (the `horizon` argument only paces the outer-momentum anneal
      and is NOT a training-length dependency for the averaged iterate).
    - The integer step `t` is caller-managed (drives the outer-momentum anneal).
    - dtypes: implemented against the float32/float64 elementwise SIMD API
      (`add_simd`/`subtract_simd`/`multiply_simd` on matching-dtype tensors);
      the scalar reductions (`<g, z-x>`, `mean(|g|)`) read every element as
      Float64 and so work for any real params dtype. All buffers must share the
      params dtype; the step raises on a params/gradients dtype OR shape mismatch
      (see the guards below). No mixed dtypes: float32 params with float16
      gradients raises.

References:
    Base Schedule-Free method:
        Defazio, A., Yang, X. A., Mehta, H., Mishchenko, K., Khaled, A., &
        Cutkosky, A. (2024). The Road Less Scheduled. NeurIPS 2024.
        arXiv:2405.15682. https://github.com/facebookresearch/schedule_free
    ScheduleFree+ (large-batch-stable variant):
        Defazio, A. (2026). ScheduleFree+: Scaling Learning-Rate-Free &
        Schedule-Free Learning to Large Language Models. arXiv:2605.19095.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)


def _dot_float64(a: AnyTensor, b: AnyTensor) -> Float64:
    """Scalar dot product <a, b> read elementwise as Float64 (dtype-agnostic).
    """
    var acc = 0.0
    for i in range(a.numel()):
        acc += a._get_float64(i) * b._get_float64(i)
    return acc


def _mean_abs_float64(a: AnyTensor) -> Float64:
    """Mean absolute value of `a` read elementwise as Float64 (L1-EMA input)."""
    var acc = 0.0
    var n = a.numel()
    if n == 0:
        return 0.0
    for i in range(n):
        var v = a._get_float64(i)
        if v < 0:
            v = -v
        acc += v
    return acc / Float64(n)


def schedule_free_plus_step(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    m: AnyTensor,
    gnorm: Float64,
    objective: Float64,
    step: Int,
    learning_rate: Float64,
    mu: Float64 = 0.9,
    beta_sf: Float64 = 0.9,
    beta_max: Float64 = 0.98,
    rho: Float64 = 0.9,
    epsilon: Float64 = 1e-8,
    horizon: Int = 1000,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, Float64, AnyTensor, Float64]:
    """Perform a single ScheduleFree+ optimization step - pure functional.

    Returns `(new_z, new_x, new_m, new_gnorm, y_next, lr_t)`: the advanced fast
    sequence `z_{t+1}`, the advanced running average `x_{t+1}` (the
    eval/checkpoint buffer), the advanced inner-momentum buffer `m_{t+1}`, the
    advanced L1-EMA gradient-norm scalar `gnorm_{t+1}`, the NEXT gradient query
    point `y_{t+1} = (1 - beta_out) * z_{t+1} + beta_out * x_{t+1}` (the train
    buffer the caller should evaluate the next gradient/objective at), and the
    Polyak step size `lr_t` actually used this step (returned for logging).

    IMPORTANT: `gradients` and `objective` must be the gradient and the
    globally-reduced objective value evaluated at the CURRENT query point
    `y_t = (1 - beta_out) * z_t + beta_out * x_t`, NOT at `z_t` or `x_t`. The
    caller is responsible for having formed `y_t` (returned as `y_next` by the
    previous call) and evaluated the gradient and objective there. On the FIRST
    step, initialize `z = x = params`, `m = 0`, and `gnorm = 0.0` so that
    `y_1 = params`.

    Args:
        params: Model parameters (used for shape/dtype guards; the live state is
            `z`/`x`/`m`). Initialize `z = x = params`, `m = 0` on step 1.
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast-sequence buffer `z_t` (steps on the inner-momentum buffer). Init
            to `params`.
        x: Running-average buffer `x_t` (eval/checkpoint iterate). Init to
            `params`.
        m: Inner-momentum buffer `m_t`. Init to zeros.
        gnorm: L1-EMA of the gradient magnitude `gnorm_{t-1}` (scalar). Init to
            0.0 on step 1.
        objective: The globally-reduced objective value `f(y_t)` at the current
            query point (Polyak numerator). Caller supplies it.
        step: 1-indexed step counter `t`; drives the outer-momentum anneal.
        learning_rate: Base step size the Polyak factor scales.
        mu: Inner-momentum coefficient (default 0.9).
        beta_sf: Base outer momentum AND the Polyak correlation weight (default
            0.9; the outer-momentum anneal starts here).
        beta_max: Outer momentum annealed to across the run (default 0.98).
        rho: L1-EMA decay for the gradient-magnitude normaliser (default 0.9).
        epsilon: Numerical-stability term added to the L1-EMA denominator
            (default 1e-8).
        horizon: Anneal length (in steps) for the increasing outer momentum;
            `beta_out` reaches `beta_max` at `step == horizon` (default 1000).
            This paces the outer-momentum anneal ONLY — it is not a schedule for
            the LR or the averaged iterate.

    Returns:
        Tuple of (new_z, new_x, new_m, new_gnorm, y_next, lr_t).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error(
            "schedule_free_plus_step: params and gradients must have the same"
            " shape"
        )
    if params.shape() != z.shape():
        raise Error(
            "schedule_free_plus_step: params and z must have the same shape"
        )
    if params.shape() != x.shape():
        raise Error(
            "schedule_free_plus_step: params and x must have the same shape"
        )
    if params.shape() != m.shape():
        raise Error(
            "schedule_free_plus_step: params and m must have the same shape"
        )
    if params.dtype() != gradients.dtype():
        raise Error(
            "schedule_free_plus_step: params and gradients must have the same"
            " dtype"
        )

    # (1) Inner momentum: m_{t+1} = mu * m_t + (1 - mu) * g_t
    var mu_t = full_like(m, mu)
    var one_minus_mu = full_like(m, 1.0 - mu)
    var new_m = add_simd(
        multiply_simd(mu_t, m),
        multiply_simd(one_minus_mu, gradients),
    )

    # (2) Polyak step-size.
    #   gnorm_t = rho * gnorm_{t-1} + (1 - rho) * mean(|g_t|)   (L1-EMA norm)
    var new_gnorm = rho * gnorm + (1.0 - rho) * _mean_abs_float64(gradients)
    #   polyak  = max(0, f(y_t) + beta_sf * <g_t, z_t - x_t>) / (gnorm_t + eps)
    var corr = _dot_float64(gradients, subtract_simd(z, x))
    var polyak_num = objective + beta_sf * corr
    if polyak_num < 0.0:
        polyak_num = 0.0
    var polyak = polyak_num / (new_gnorm + epsilon)
    var lr_t = learning_rate * polyak

    # Fast sequence steps on the momentum buffer: z_{t+1} = z_t - lr_t * m_{t+1}
    var lr_tensor = full_like(z, lr_t)
    var new_z = subtract_simd(z, multiply_simd(lr_tensor, new_m))

    # (3) Increasing outer momentum: anneal beta_out from beta_sf to beta_max.
    var frac = Float64(step - 1) / Float64(horizon - 1 if horizon > 1 else 1)
    if frac > 1.0:
        frac = 1.0
    var beta_out = beta_sf + (beta_max - beta_sf) * frac

    #   x_{t+1} = beta_out * x_t + (1 - beta_out) * z_{t+1}
    var beta_out_t = full_like(x, beta_out)
    var one_minus_beta_out = full_like(x, 1.0 - beta_out)
    var new_x = add_simd(
        multiply_simd(beta_out_t, x),
        multiply_simd(one_minus_beta_out, new_z),
    )

    #   y_{t+1} = (1 - beta_out) * z_{t+1} + beta_out * x_{t+1}
    var one_minus_beta_out_z = full_like(new_z, 1.0 - beta_out)
    var beta_out_x = full_like(new_x, beta_out)
    var y_next = add_simd(
        multiply_simd(one_minus_beta_out_z, new_z),
        multiply_simd(beta_out_x, new_x),
    )

    return (new_z, new_x, new_m, new_gnorm, y_next, lr_t)


def schedule_free_plus_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    m: AnyTensor,
    gnorm: Float64,
    objective: Float64,
    step: Int,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, Float64, AnyTensor, Float64]:
    """Simplified ScheduleFree+ step with default hyperparameters.

    Convenience wrapper around `schedule_free_plus_step` with the default inner
    momentum `mu = 0.9`, `beta_sf = 0.9`, `beta_max = 0.98`, `rho = 0.9`,
    `epsilon = 1e-8`, and `horizon = 1000`. The caller still manages the
    `z`/`x`/`m` buffers, the scalar `gnorm`, the objective `f(y_t)`, and the
    integer step `t`. On step 1, initialize `z = x = params`, `m = 0`,
    `gnorm = 0.0` (see `schedule_free_plus_step`).

    Args:
        params: Model parameters (shape/dtype guards; live state is `z`/`x`/`m`).
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast-sequence buffer `z_t`. Init to `params`.
        x: Running-average buffer `x_t`. Init to `params`.
        m: Inner-momentum buffer `m_t`. Init to zeros.
        gnorm: L1-EMA gradient-norm scalar `gnorm_{t-1}`. Init to 0.0.
        objective: Globally-reduced objective `f(y_t)` at the query point.
        step: 1-indexed step counter `t`.
        learning_rate: Base step size scaled by the Polyak factor.

    Returns:
        Tuple of (new_z, new_x, new_m, new_gnorm, y_next, lr_t).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return schedule_free_plus_step(
        params,
        gradients,
        z,
        x,
        m,
        gnorm,
        objective,
        step,
        learning_rate=learning_rate,
        mu=0.9,
        beta_sf=0.9,
        beta_max=0.98,
        rho=0.9,
        epsilon=1e-8,
        horizon=1000,
    )
