"""Schedule-Free optimizer (online iterate averaging — anytime).

Schedule-Free replaces explicit learning-rate schedules with online iterate
averaging + interpolation over THREE coupled sequences. There is no fixed
training horizon: the averaged iterate `x` is a good checkpoint at ANY step, so
you get optimal-checkpoint behavior at every step instead of only at the end of
a tuned schedule.

The method maintains two persisted buffers, the fast sequence `z` and the
running average `x`, and forms a third derived point `y` — the query point where
the gradient is evaluated. Concretely, per step t (1-indexed):

    y_t     = (1 - beta) * z_t + beta * x_t          (gradient query point)
    g_t     = grad( f, y_t )                          (caller supplies this)
    z_{t+1} = z_t - gamma * g_t                       (fast SGD-style sequence)
    c_{t+1} = (r + 1) / (t + r + 1)                   (averaging weight)
    x_{t+1} = (1 - c_{t+1}) * x_t + c_{t+1} * z_{t+1} (running average of z)

`z` and `x` are the caller-managed state; `y` is recomputed each step. The model
does its forward/backward pass at `y` (the "train" buffer), while `x` is the
"eval"/checkpoint buffer — the .train()/.eval() buffer switch in the reference
implementations. On the first step the state is initialized `z_1 = x_1 = params`
(so `y_1 = params`); this pure-functional `schedule_free_step` advances the state
and also returns the NEXT query point `y_{t+1}` for convenience.

With the weight-power `r = 0` the schedule `c_{t+1} = (r+1)/(t+r+1) = 1/(t+1)`
reduces `x` to the plain uniform average of the `z` iterates; larger `r`
down-weights early iterates (a polynomial-decay weighting). `beta` (~0.9–0.95)
is the interpolation between the fast and averaged sequences at the query point.

Key characteristics:
    - Two persisted buffers (`z`, `x`) — same memory as SGD+momentum, no
      second-moment tensor. This is the Schedule-Free wrapper around SGD; a
      base optimizer other than SGD would replace the `z` update below.
    - Anytime: `x` is a valid checkpoint at every step; no horizon/schedule.
    - The integer step `t` is caller-managed (drives the averaging weight
      `c_{t+1}`); there is no EMA-style bias correction.
    - dtypes: implemented against the float32/float64 elementwise SIMD API
      (`add_simd`/`subtract_simd`/`multiply_simd` on matching-dtype tensors).
      All buffers must share the params dtype; the step raises on a
      params/gradients dtype OR shape mismatch (see the guards below). No mixed
      dtypes: passing e.g. float32 params with float16 gradients raises.

Reference:
    Defazio, A., Yang, X. A., Mehta, H., Mishchenko, K., Khaled, A., &
    Cutkosky, A. (2024). The Road Less Scheduled. Advances in Neural
    Information Processing Systems (NeurIPS) 2024.
    arXiv preprint arXiv:2405.15682
    https://github.com/facebookresearch/schedule_free
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)


def schedule_free_step(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    step: Int,
    learning_rate: Float64,
    beta: Float64 = 0.9,
    weight_power: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single Schedule-Free (SGD) optimization step - pure functional.

    Returns `(new_z, new_x, y_next)`: the advanced fast sequence `z_{t+1}`, the
    advanced running average `x_{t+1}` (the eval/checkpoint buffer), and the
    NEXT gradient query point `y_{t+1} = (1-beta)*z_{t+1} + beta*x_{t+1}` (the
    train buffer the caller should evaluate the next gradient at). Caller manages
    all state, including the integer step `t`.

    IMPORTANT: `gradients` must be the gradient evaluated at the CURRENT query
    point `y_t = (1-beta)*z_t + beta*x_t`, NOT at `z_t` or `x_t`. The caller is
    responsible for having formed `y_t` (returned as `y_next` by the previous
    call) and evaluated the gradient there. `params` is accepted only for the
    shape/dtype guards and is otherwise unused (the state lives in `z`/`x`); on
    the FIRST step, initialize both `z` and `x` to `params` so that
    `y_1 = params`.

    Args:
        params: Model parameters (used for shape/dtype guards; the live state is
            `z`/`x`). Initialize `z = x = params` on step 1.
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast-sequence buffer `z_t` (SGD-style iterate). Init to `params`.
        x: Running-average buffer `x_t` (eval/checkpoint iterate). Init to
            `params`.
        step: 1-indexed step counter `t`; drives the averaging weight
            `c_{t+1} = (r+1)/(t+r+1)`.
        learning_rate: Step size `gamma` for the fast `z` update.
        beta: Interpolation between `z` and `x` at the query point (default 0.9;
            paper's useful range ~0.9–0.95).
        weight_power: Averaging weight power `r` (default 0.0 → uniform average
            `c_{t+1} = 1/(t+1)`; larger `r` down-weights early iterates).

    Returns:
        Tuple of (new_z, new_x, y_next).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    if params.shape() != gradients.shape():
        raise Error(
            "schedule_free_step: params and gradients must have the same shape"
        )
    if params.shape() != z.shape():
        raise Error("schedule_free_step: params and z must have the same shape")
    if params.shape() != x.shape():
        raise Error("schedule_free_step: params and x must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error(
            "schedule_free_step: params and gradients must have the same dtype"
        )
    if params.dtype() != z.dtype():
        raise Error("schedule_free_step: params and z must have the same dtype")
    if params.dtype() != x.dtype():
        raise Error("schedule_free_step: params and x must have the same dtype")

    # Fast sequence: z_{t+1} = z_t - gamma * g_t
    var lr_t = full_like(z, learning_rate)
    var new_z = subtract_simd(z, multiply_simd(lr_t, gradients))

    # Averaging weight: c_{t+1} = (r + 1) / (t + r + 1)
    var c = (weight_power + 1.0) / (Float64(step) + weight_power + 1.0)

    # Running average: x_{t+1} = (1 - c) * x_t + c * z_{t+1}
    var one_minus_c = full_like(x, 1.0 - c)
    var c_t = full_like(new_z, c)
    var new_x = add_simd(
        multiply_simd(one_minus_c, x),
        multiply_simd(c_t, new_z),
    )

    # Next query point: y_{t+1} = (1 - beta) * z_{t+1} + beta * x_{t+1}
    var one_minus_beta = full_like(new_z, 1.0 - beta)
    var beta_t = full_like(new_x, beta)
    var y_next = add_simd(
        multiply_simd(one_minus_beta, new_z),
        multiply_simd(beta_t, new_x),
    )

    return (new_z, new_x, y_next)


def schedule_free_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    z: AnyTensor,
    x: AnyTensor,
    step: Int,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Simplified Schedule-Free step with default hyperparameters.

    Convenience wrapper around `schedule_free_step` with the default
    interpolation `beta = 0.9` and uniform averaging (`weight_power = 0.0`,
    i.e. `c_{t+1} = 1/(t+1)`). Caller still manages the `z`/`x` buffers and the
    integer step `t`. On step 1, initialize `z = x = params` (see
    `schedule_free_step`).

    Args:
        params: Model parameters (shape/dtype guards; live state is `z`/`x`).
        gradients: Gradient of the loss w.r.t. the query point `y_t`.
        z: Fast-sequence buffer `z_t`. Init to `params`.
        x: Running-average buffer `x_t`. Init to `params`.
        step: 1-indexed step counter `t`.
        learning_rate: Step size `gamma`.

    Returns:
        Tuple of (new_z, new_x, y_next).

    Raises:
        Error: If tensor shapes or dtypes don't match.
    """
    return schedule_free_step(
        params,
        gradients,
        z,
        x,
        step,
        learning_rate=learning_rate,
        beta=0.9,
        weight_power=0.0,
    )


def init_schedule_free_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the schedule_free optimizer.

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
