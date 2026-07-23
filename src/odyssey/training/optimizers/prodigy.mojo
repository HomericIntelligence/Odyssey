"""Prodigy optimizer (parameter-free step-size estimation, D-Adaptation family).

Prodigy eliminates the learning-rate hyperparameter by estimating the distance
to the solution, d_t ~= ||x_0 - x*||, online and using it to scale an Adam-style
update. It is an "expeditiously adaptive" refinement of D-Adaptation: it weights
the distance accumulators by an extra factor of d_t, which makes the estimate
grow to the right scale far faster than D-Adaptation's linear schedule.

This module implements the Adam variant (Algorithm 4 of the paper). The step is
PURE FUNCTIONAL in the sibling idiom of this package: all state travels in and
out. The per-coordinate buffers are the first moment `m`, second moment `v`, and
the distance-numerator accumulator `s`; the SCALAR state is the distance
numerator `r` and the current distance estimate `d`. The caller also supplies
`x0`, the INITIAL parameter vector (captured once before the first step), which
the distance accumulators reference for the whole run.

Prodigy update rule (per step, using the CURRENT d in the accumulators and the
parameter update — the paper's k=0.. recurrence):

    m'  = beta1 * m + (1 - beta1) * d * g
    v'  = beta2 * v + (1 - beta2) * d^2 * g^2
    r'  = sqrt(beta2) * r + (1 - sqrt(beta2)) * gamma * d^2 * <g, x0 - x>
    s'  = sqrt(beta2) * s + (1 - sqrt(beta2)) * gamma * d^2 * g
    d_hat = r' / ||s'||_1                        (0 when ||s'||_1 == 0)
    d'  = max(d, d_hat)                          (monotone non-decreasing)
    x'  = x - gamma * d * m' / (sqrt(v') + d * eps)

where `<.,.>` is the Euclidean inner product (a scalar), `||.||_1` is the L1
norm (a scalar), `r`/`d` are scalars, and `m`/`v`/`s`/`x`/`x0`/`g` are
per-coordinate tensors. `gamma` is the base step-size SCHEDULE, default 1.0:
Prodigy has "no learning rate to tune", so `gamma` normally stays fixed at 1.0
and the d-estimate supplies the step scale. A `growth_rate` cap limits how fast
d may grow per step (`d' <= growth_rate * d`), acting as a natural warmup; the
default (`1e30`) effectively disables the cap (matching the paper's base
algorithm).

Key characteristics:
    - No learning rate. `d` is estimated online and is monotone non-decreasing.
    - The distance estimate stays at `d0` until the numerator `r` becomes
      positive (on step 1, x == x0 so <g, x0 - x> = 0, r stays 0, d stays d0);
      it then grows as the parameters move away from x0.
    - Memory: two Adam buffers (`m`, `v`) plus one distance-accumulator buffer
      (`s`), plus the initial-parameter reference `x0` and two scalars (`r`,
      `d`) — comparable to Adam plus one extra buffer.

Dtype support:
    Tested and numerically verified on FLOAT32 and FLOAT64 (the parity test runs
    in float64). Like the sibling first-order optimizers in this package, the
    step requires `params` and `gradients` to share a dtype and shape and will
    `raise` otherwise (guarded and pinned by a raises-test). All state buffers
    and `x0` must match the params shape. Mixed-precision (e.g. float16 params
    with float32 state) is NOT supported and is not exercised.

Reference:
    Mishchenko, K., & Defazio, A. (2023). Prodigy: An Expeditiously Adaptive
    Parameter-Free Learner. arXiv preprint arXiv:2306.06101.
    (Algorithm 4, the Adam variant.)
    https://arxiv.org/abs/2306.06101
    https://github.com/konstmish/prodigy
"""

from std.math import sqrt as scalar_sqrt

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import sqrt, abs as elementwise_abs
from odyssey.core.reduction import sum as reduce_sum


def _scalar_sum(tensor: AnyTensor) raises -> Float64:
    """Reduce a tensor to the scalar sum of all its elements (as Float64).

    Reads the reduced scalar using the tensor's OWN dtype: `sum` preserves the
    input dtype, so a float32 tensor must be read with `load[DType.float32]`.
    Reading a float32-backed scalar with `load[DType.float64]` reinterprets the
    lanes and yields garbage (the same class of bug seen in mixed-dtype helper
    code), which would silently freeze the d-estimate on float32 runs.
    """
    var reduced = reduce_sum(tensor, axis=-1)
    if tensor.dtype() == DType.float64:
        return Float64(reduced.load[DType.float64](0))
    else:
        return Float64(reduced.load[DType.float32](0))


def prodigy_step(
    params: AnyTensor,
    gradients: AnyTensor,
    m: AnyTensor,
    v: AnyTensor,
    s: AnyTensor,
    x0: AnyTensor,
    r: Float64,
    d: Float64,
    gamma: Float64 = 1.0,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.999,
    epsilon: Float64 = 1e-8,
    growth_rate: Float64 = 1.0e30,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor, Float64, Float64]:
    """Perform a single Prodigy optimization step - pure functional.

    Returns (new_params, new_m, new_v, new_s, new_r, new_d). The caller manages
    all state, including the two SCALARS `r` (distance numerator) and `d` (the
    distance estimate), and supplies `x0`, the INITIAL parameter vector captured
    once before the first step (the distance accumulators reference it for the
    whole run — pass the SAME `x0` on every call).

    On the FIRST step, initialize `m`, `v`, `s` to zeros_like(params), `r = 0.0`,
    `d = d0` (a small positive value such as 1e-6), and `x0 = params`. On step 1,
    x == x0 so the inner product <g, x0 - x> is zero: `r` and `d_hat` stay 0 and
    `d` stays at `d0`. From step 2 on, as the parameters move away from `x0`, the
    numerator grows and `d` increases (monotone non-decreasing).

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        m: First-moment buffer (EMA of d-weighted gradients).
        v: Second-moment buffer (EMA of d^2-weighted squared gradients).
        s: Distance-numerator accumulator (EMA of d^2-weighted gradients).
        x0: Initial parameter vector (constant for the whole run; = params on
            step 1). Referenced by the distance accumulators via <g, x0 - x>.
        r: Scalar distance numerator from the previous step (0.0 on step 1).
        d: Scalar distance estimate from the previous step (d0 on step 1).
        gamma: Base step-size schedule (default 1.0). Prodigy has no learning
            rate to tune; leave at 1.0 unless applying an external schedule.
        beta1: EMA decay for the first moment (default: 0.9).
        beta2: EMA decay for the second moment; its square root also decays the
            distance accumulators r and s (default: 0.999).
        epsilon: Numerical-stability term; enters the denominator as `d * eps`
            (scaled by the current distance estimate, per the paper) (default:
            1e-8).
        growth_rate: Upper cap on per-step growth of the distance estimate:
            `d' <= growth_rate * d`. The default (1e30) disables the cap; a
            finite value (e.g. 1.01) provides a natural warmup by limiting how
            fast d expands.

    Returns:
        Tuple of (new_params, new_m, new_v, new_s, new_r, new_d), where new_r
        and new_d are Float64 scalars.

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error(
            "prodigy_step: params and gradients must have the same shape"
        )
    if params.shape() != m.shape():
        raise Error("prodigy_step: params and m must have the same shape")
    if params.shape() != v.shape():
        raise Error("prodigy_step: params and v must have the same shape")
    if params.shape() != s.shape():
        raise Error("prodigy_step: params and s must have the same shape")
    if params.shape() != x0.shape():
        raise Error("prodigy_step: params and x0 must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error(
            "prodigy_step: params and gradients must have the same dtype"
        )

    var sqrt_beta2 = scalar_sqrt(beta2)
    var one_minus_sqrt_beta2 = 1.0 - sqrt_beta2

    # First moment: m' = beta1 * m + (1 - beta1) * d * g
    var beta1_t = full_like(m, beta1)
    var one_minus_beta1_d = full_like(m, (1.0 - beta1) * d)
    var new_m = add_simd(
        multiply_simd(beta1_t, m),
        multiply_simd(one_minus_beta1_d, gradients),
    )

    # Second moment: v' = beta2 * v + (1 - beta2) * d^2 * g^2
    var beta2_t = full_like(v, beta2)
    var one_minus_beta2_d2 = full_like(v, (1.0 - beta2) * d * d)
    var grad_sq = multiply_simd(gradients, gradients)
    var new_v = add_simd(
        multiply_simd(beta2_t, v),
        multiply_simd(one_minus_beta2_d2, grad_sq),
    )

    # Distance numerator: r' = sqrt(b2)*r + (1-sqrt(b2))*gamma*d^2*<g, x0 - x>
    var displacement = subtract_simd(x0, params)
    var inner = _scalar_sum(multiply_simd(gradients, displacement))
    var new_r = sqrt_beta2 * r + one_minus_sqrt_beta2 * gamma * d * d * inner

    # Distance accumulator: s' = sqrt(b2)*s + (1-sqrt(b2))*gamma*d^2*g
    var sqrt_beta2_t = full_like(s, sqrt_beta2)
    var s_grad_coeff = full_like(s, one_minus_sqrt_beta2 * gamma * d * d)
    var new_s = add_simd(
        multiply_simd(sqrt_beta2_t, s),
        multiply_simd(s_grad_coeff, gradients),
    )

    # Distance estimate: d_hat = r' / ||s'||_1 ; d' = max(d, d_hat), capped.
    var s_l1 = _scalar_sum(elementwise_abs(new_s))
    var d_hat = 0.0
    if s_l1 != 0.0:
        d_hat = new_r / s_l1
    var new_d = d
    if d_hat > new_d:
        new_d = d_hat
    # growth_rate cap: d' must not exceed growth_rate * d (natural warmup).
    var d_cap = growth_rate * d
    if new_d > d_cap:
        new_d = d_cap

    # Parameter update: x' = x - gamma * d * m' / (sqrt(v') + d * eps)
    # NOTE: uses the CURRENT d (not d'), matching Algorithm 4.
    var denom = add_simd(sqrt(new_v), full_like(new_v, d * epsilon))
    var scaled_m = multiply_simd(full_like(new_m, gamma * d), new_m)
    var update = divide_simd(scaled_m, denom)
    var new_params = subtract_simd(params, update)

    return (new_params, new_m, new_v, new_s, new_r, new_d)


def prodigy_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    m: AnyTensor,
    v: AnyTensor,
    s: AnyTensor,
    x0: AnyTensor,
    r: Float64,
    d: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor, Float64, Float64]:
    """Simplified Prodigy step with default hyperparameters (Mishchenko & Defazio 2023).

    Convenience wrapper around `prodigy_step` for the common case: gamma = 1.0
    (no learning rate to tune), betas (0.9, 0.999), epsilon 1e-8, and the growth
    cap disabled. The caller still manages the m/v/s buffers, the x0 reference,
    and the scalars r and d (initialize r = 0.0, d = d0 such as 1e-6, x0 =
    params on step 1).

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        m: First-moment buffer. Initialize to zeros_like(params).
        v: Second-moment buffer. Initialize to zeros_like(params).
        s: Distance-numerator accumulator. Initialize to zeros_like(params).
        x0: Initial parameter vector (= params on step 1).
        r: Scalar distance numerator from the previous step (0.0 on step 1).
        d: Scalar distance estimate from the previous step (d0 on step 1).

    Returns:
        Tuple of (new_params, new_m, new_v, new_s, new_r, new_d).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return prodigy_step(params, gradients, m, v, s, x0, r, d)


def init_prodigy_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the prodigy optimizer.

    Returns a `List[List[AnyTensor]]` with outer length == `len(params_list)` (one entry per parameter) and inner length == 3 (one entry per state buffer the optimizer threads across calls).

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
        for _ in range(3):
            per.append(zeros(p.shape(), d))
        all_states.append(per^)
    return all_states^
