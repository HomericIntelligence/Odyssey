"""SF-NorMuon optimizer (schedule-free spectral).

SF-NorMuon combines Schedule-Free anytime iterate averaging (three coupled
sequences y/z/x — Defazio et al. 2024) with NorMuon's spectral geometry
(row-wise-normalized Newton-Schulz orthogonalization of the momentum — the
polar factor of the update direction; Li et al. 2025, arXiv:2510.05491). The averaged
iterate `x` is a good checkpoint at ANY step (no training-horizon schedule),
while the update DIRECTION is the polar / orthogonalized momentum instead of the
raw gradient. This is the schedule-free wrapper around the NorMuon base
optimizer: the `z` fast-sequence update below is the NorMuon step (orthogonalize
+ row-normalize), and the y/z/x averaging is layered on top.

Three sequences are maintained (per step t, 1-indexed):

    y_t     = beta*x_t + (1-beta)*z_t                (schedule-free query point)
    g_t     = grad( f, y_t )                          (caller supplies this)
    m_t     = mu*m_{t-1} + (1-mu)*g_t                 (momentum EMA)
    P_t     = rownorm( newton_schulz(m_t) )           (polar + per-row L2 norm)
    z_{t+1} = (1 - lambda_wd)*z_t - eta*P_t           (WEIGHT DECAY ON z_t)
    c_{t+1} = (r + 1)/(t + r + 1)                     (averaging weight)
    x_{t+1} = (1 - c_{t+1})*x_t + c_{t+1}*z_{t+1}      (running average of z)

`z` (fast NorMuon sequence), `x` (running-average / eval-checkpoint buffer), and
`m` (momentum EMA) are the caller-managed state; `y` is recomputed each step and
is where the caller must evaluate the gradient. On the FIRST step, initialize
`z = x = params` (so `y_1 = params`) and `m = zeros_like(params)`. The step
returns the advanced `(z_{t+1}, x_{t+1}, m_t, y_{t+1})` — the next query point
`y_{t+1}` is returned for convenience so the caller can evaluate the next
gradient there.

CRITICAL — weight decay on z_t, NOT y_t:
    The weight-decay term `(1 - lambda_wd)*z_t` MUST be applied to the fast
    iterate `z_t`, never to the query point `y_t` or the average `x_t`. Decaying
    the query point diverges over long horizons (the schedule-free coupling
    re-injects the decayed magnitude into the average). This is the documented
    failure mode of naively composing schedule-free + weight decay; the guard is
    structural (the decay only ever multiplies `z` below).

NorMuon polar factor:
    `P_t = rownorm(newton_schulz(m_t))` reuses Odyssey's shared
    `newton_schulz_orthogonalize` (muon.mojo — Jordan's tuned quintic, 5 steps,
    Frobenius pre-normalization, transpose-on-shorter-dimension) for the
    orthogonalization, then divides each ROW by its L2 norm (axis=0, matching
    NorMuon's `_normalize_tensor_by_axis`). Because Newton-Schulz here does not
    converge exactly to an orthogonal matrix (singular values land in
    ~[0.68, 1.13]), "polar" is approximate — the same approximation NorMuon and
    Muon use deliberately; an approximately-orthogonal, row-normalized direction
    is all the update needs.

Matrix-shaped params only:
    Like Muon/NorMuon, this operates on rank-2 (matrix) parameters. It raises on
    non-2D inputs. For biases/embeddings/norm scales, use AdamW.

dtypes:
    Implemented against the float32/float64 SIMD elementwise API
    (`add_simd`/`subtract_simd`/`multiply_simd`) plus the shared Newton-Schulz
    helper (matmul/transpose). All buffers must share the params dtype; the step
    raises on a params/gradients dtype OR shape mismatch. float16/bfloat16 are
    NOT validated here — the Newton-Schulz Gram-matrix accumulation is
    numerically fragile in low precision, so callers should orthogonalize in
    float32 or float64. The per-row L2 reduction reads/writes through the
    tensor's OWN dtype (`_get_float64`/`_set_float64` dispatch on `self._dtype`),
    so no dtype is hardcoded in the reduction loop.

Reference:
    Schedule-Free: Defazio, A., Yang, X. A., Mehta, H., Mishchenko, K., Khaled,
    A., & Cutkosky, A. (2024). The Road Less Scheduled. NeurIPS 2024.
    arXiv:2405.15682. https://github.com/facebookresearch/schedule_free
    NorMuon: "NorMuon: Making Muon more efficient and scalable" (Li et al.,
    2025, arXiv:2510.05491). Built on the canonical Muon optimizer from
    Jordan et al. 2024 — https://kellerjordan.github.io/posts/muon/ — with
    per-axis normalization for stability at higher learning rates.
    The SF-NorMuon composition (schedule-free query point + row-normalized polar
    update with weight decay on the fast iterate) follows the update rule pinned
    in tracking issue mvillmow/Random#79 [OPT-15]; no single primary paper
    describes the exact composition, so the rule is [derived from the issue].
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like, zeros_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)
from odyssey.training.optimizers.muon import newton_schulz_orthogonalize


def _row_normalize(tensor: AnyTensor, eps: Float64 = 1e-8) raises -> AnyTensor:
    """Divide each row of a 2D tensor by its L2 norm (NorMuon axis=0).

    Mirrors NorMuon's `_normalize_tensor_by_axis(axis=0)`: for each row i,
    norm_i = sqrt(sum_j t[i,j]^2 + eps), then t[i,:] /= norm_i. The scalar
    reduction reads/writes through `_get_float64`/`_set_float64`, which dispatch
    on the tensor's OWN dtype — nothing is hardcoded to float64.

    Args:
        tensor: 2D tensor to row-normalize.
        eps: Small epsilon folded under the sqrt for numerical stability.

    Returns:
        A new tensor of the same shape with each row L2-normalized.

    Raises:
        Error: If tensor is not 2D or eps is not positive.
    """
    var shape = tensor.shape()
    if len(shape) != 2:
        raise Error("SF-NorMuon row-normalize requires a 2D tensor")
    if eps <= 0.0:
        raise Error("eps must be positive")

    var m = shape[0]
    var n = shape[1]
    var result = zeros_like(tensor)

    for i in range(m):
        var sum_sq = 0.0
        for j in range(n):
            var val = tensor._get_float64(i * n + j)
            sum_sq += val * val
        var norm_val = scalar_sqrt(sum_sq + eps)
        for j in range(n):
            var val = tensor._get_float64(i * n + j)
            result._set_float64(i * n + j, val / norm_val)

    return result


def sf_normuon_step(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    momentum: AnyTensor,
    step: Int,
    learning_rate: Float64,
    beta: Float64 = 0.9,
    mu: Float64 = 0.95,
    weight_decay: Float64 = 0.0,
    weight_power: Float64 = 0.0,
    eps: Float64 = 1e-8,
    ns_steps: Int = 5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single SF-NorMuon step - pure functional.

    Returns `(new_z, new_x, new_m, y_next)`: the advanced fast sequence
    `z_{t+1}`, the advanced running average `x_{t+1}` (the eval/checkpoint
    buffer), the momentum EMA `m_t`, and the NEXT gradient query point
    `y_{t+1} = beta*x_{t+1} + (1-beta)*z_{t+1}`. Caller manages all state,
    including the integer step `t`.

    IMPORTANT: `gradients` must be the gradient evaluated at the CURRENT query
    point `y_t = beta*x_t + (1-beta)*z_t`, NOT at `z_t` or `x_t`. The caller
    forms `y_t` (returned as `y_next` by the previous call) and evaluates the
    gradient there. On the FIRST step, initialize `z = x = params` (so
    `y_1 = params`) and `momentum = zeros_like(params)`; `params` is used only
    for the shape/dtype guards (the live state is `z`/`x`/`momentum`).

    Weight decay applies to the fast iterate `z_t` ONLY (see the module
    docstring's CRITICAL note): `z_{t+1} = (1-weight_decay)*z_t - lr*P_t`.

    Args:
        params: Model parameters (shape/dtype guards; live state is z/x/m).
            Initialize `z = x = params` on step 1.
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast NorMuon-sequence buffer `z_t`. Init to `params`.
        x: Running-average buffer `x_t` (eval/checkpoint). Init to `params`.
        momentum: Momentum-EMA buffer `m_{t-1}`. Init to `zeros_like(params)`.
        step: 1-indexed step counter `t`; drives `c_{t+1} = (r+1)/(t+r+1)`.
        learning_rate: Step size `eta` for the fast `z` update (NorMuon ~0.05).
        beta: Interpolation between `x` and `z` at the query point (default 0.9).
        mu: Momentum EMA coefficient (default 0.95).
        weight_decay: `lambda_wd`, applied to `z_t` only (default 0.0).
        weight_power: Averaging weight power `r` (default 0.0 -> uniform average
            `c_{t+1} = 1/(t+1)`; larger `r` down-weights early iterates).
        eps: Epsilon for the per-row L2 normalization (default 1e-8).
        ns_steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Tuple of (new_z, new_x, new_momentum, y_next).

    Raises:
        Error: If params are not 2D, shapes/dtypes mismatch, or eps<=0.
    """
    # Validation (mirror NorMuon / schedule-free guards).
    var shape = params.shape()
    if len(shape) != 2:
        raise Error(
            "SF-NorMuon only supports 2D (matrix-shaped) parameters; for"
            " biases/embeddings use AdamW"
        )
    if params.shape() != gradients.shape():
        raise Error("SF-NorMuon: params and gradients must have the same shape")
    if params.shape() != z.shape():
        raise Error("SF-NorMuon: params and z must have the same shape")
    if params.shape() != x.shape():
        raise Error("SF-NorMuon: params and x must have the same shape")
    if params.shape() != momentum.shape():
        raise Error("SF-NorMuon: params and momentum must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error("SF-NorMuon: params and gradients must have the same dtype")
    if eps <= 0.0:
        raise Error("eps must be positive")

    # Momentum EMA: m_t = mu * m_{t-1} + (1 - mu) * g_t
    var mu_t = full_like(momentum, mu)
    var one_minus_mu = full_like(momentum, 1.0 - mu)
    var new_m = add_simd(
        multiply_simd(mu_t, momentum),
        multiply_simd(one_minus_mu, gradients),
    )

    # Polar factor: P_t = rownorm( newton_schulz(m_t) ).
    var m_orth = newton_schulz_orthogonalize(new_m, steps=ns_steps)
    var P = _row_normalize(m_orth, eps=eps)

    # Fast sequence with weight decay ON z_t:
    #   z_{t+1} = (1 - lambda_wd) * z_t - lr * P_t
    var keep = full_like(z, 1.0 - weight_decay)
    var lr_t = full_like(P, learning_rate)
    var new_z = subtract_simd(
        multiply_simd(keep, z),
        multiply_simd(lr_t, P),
    )

    # Averaging weight: c_{t+1} = (r + 1) / (t + r + 1)
    var c = (weight_power + 1.0) / (Float64(step) + weight_power + 1.0)

    # Running average: x_{t+1} = (1 - c) * x_t + c * z_{t+1}
    var one_minus_c = full_like(x, 1.0 - c)
    var c_t = full_like(new_z, c)
    var new_x = add_simd(
        multiply_simd(one_minus_c, x),
        multiply_simd(c_t, new_z),
    )

    # Next query point: y_{t+1} = beta * x_{t+1} + (1 - beta) * z_{t+1}
    var beta_t = full_like(new_x, beta)
    var one_minus_beta = full_like(new_z, 1.0 - beta)
    var y_next = add_simd(
        multiply_simd(beta_t, new_x),
        multiply_simd(one_minus_beta, new_z),
    )

    return (new_z, new_x, new_m, y_next)


def sf_normuon_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    momentum: AnyTensor,
    step: Int,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Simplified SF-NorMuon step with default hyperparameters.

    Convenience wrapper around `sf_normuon_step` with beta=0.9, mu=0.95,
    weight_decay=0.0, uniform averaging (weight_power=0.0), eps=1e-8, and
    ns_steps=5. Caller still manages the z/x/momentum buffers and the integer
    step t. On step 1, initialize `z = x = params` and
    `momentum = zeros_like(params)`.

    Args:
        params: Model parameters (shape/dtype guards; live state is z/x/m).
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast-sequence buffer `z_t`. Init to `params`.
        x: Running-average buffer `x_t`. Init to `params`.
        momentum: Momentum-EMA buffer. Init to `zeros_like(params)`.
        step: 1-indexed step counter `t`.
        learning_rate: Step size `eta` (NorMuon ~0.05).

    Returns:
        Tuple of (new_z, new_x, new_momentum, y_next).

    Raises:
        Error: If params are not 2D or shapes/dtypes mismatch.
    """
    return sf_normuon_step(
        params,
        gradients,
        z,
        x,
        momentum,
        step,
        learning_rate=learning_rate,
        beta=0.9,
        mu=0.95,
        weight_decay=0.0,
        weight_power=0.0,
        eps=1e-8,
        ns_steps=5,
    )


def init_sf_normuon_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the sf_normuon optimizer.

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
