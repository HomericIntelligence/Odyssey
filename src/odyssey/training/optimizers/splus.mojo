"""SPlus optimizer — a Stable whitening optimizer (sign-in-eigenbasis).

SPlus (Frans, Levine & Abbeel, 2025) is a whitening optimizer in the Shampoo/SOAP
family that fixes the long-cache divergence of naive Shampoo. Instead of dividing
the eigenbasis-projected update by the square-root of the accumulated eigenvalues
(which becomes unstable when the preconditioner is cached across many steps), SPlus
takes the ELEMENT-WISE SIGN of the projected update — a bounded, instantaneous
normalization that caps the maximum spectral magnitude. It adds shape-aware scaling
for width-invariant learning-rate transfer, and evaluates on an EMA (iterate-
averaged) copy of the parameters while the live parameters move at a high LR.

For a 2-D weight `W (R×C)` it maintains, exactly like SOAP, two Kronecker factors —
a left factor `L (R×R) = EMA(g gᵀ)` and a right factor `M (C×C) = EMA(gᵀ g)` — and
their eigenvector bases `Q_L`, `Q_R`, refreshed every `precondition_frequency`
steps. Each step:

    L        = β2 L + (1-β2) g gᵀ            # left Kronecker factor
    M        = β2 M + (1-β2) gᵀ g            # right Kronecker factor
    Q_L, Q_R = eig(L), eig(M)                # historical eigenbasis (refreshed)
    m        = β1 m + (1-β1) g               # first moment on the RAW grad
    m'       = Q_Lᵀ m Q_R                   # project the first moment
    u        = Q_L sign(m') Q_Rᵀ            # SIGN in the eigenbasis, rotated back
                                             # (deadzone: |m'| < sign_eps -> 0)
    scale    = 2 / (R + C)                    # shape-aware scaling
    W        = W - lr*scale*u - lr*wd*W       # live params (decoupled weight decay)
    W_ema    = ema_rate W_ema + (1-ema_rate) W  # the sequence you EVALUATE on

The sign step (`sign(m')`) is what replaces SOAP's `m' / (√v + ε)`: it bounds the
per-coordinate update to ±1 in the eigenbasis, which is the "stable" in SPlus. The
live parameters `W` train fast; downstream evaluation uses the slowly-averaged
`W_ema`.

**Numerics:** Odyssey's matmul raises on mixed dtypes, and f32 preconditioner math
against f64 state produces garbage (the SOAP lesson). All state and matrix math here
are float64; the caller keeps every buffer in float64 and casts the final delta back
to the parameter dtype at the call site. Pure-functional: the caller owns all state
(exp_avg, the two Kronecker factors, the two eigenbases, and the EMA params) and
threads it across steps.

Reference:
    Frans, K., Levine, S., & Abbeel, P. (2025). A Stable Whitening Optimizer for
    Efficient Neural Network Training. arXiv:2506.07254. NeurIPS 2025.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, zeros_like
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, subtract_simd
from odyssey.core.eigen import symmetric_eigh


def splus_step(
    params: AnyTensor,
    params_ema: AnyTensor,
    gradients: AnyTensor,
    exp_avg: AnyTensor,
    gg_left: AnyTensor,
    gg_right: AnyTensor,
    q_left: AnyTensor,
    q_right: AnyTensor,
    step: Int,
    learning_rate: Float64,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.99,
    ema_rate: Float64 = 0.999,
    weight_decay: Float64 = 0.0,
    precondition_frequency: Int = 100,
    sign_eps: Float64 = 1e-12,
) raises -> Tuple[
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
]:
    """Perform a single SPlus step for a 2-D parameter — pure functional.

    The caller owns all state and passes it in each step. `step` is 1-indexed
    (increment BEFORE calling, matching SOAP's `group['step'] += 1` convention). On
    `step == 1` the eigenbases are built from the freshly-updated Kronecker factors;
    on later steps they are refreshed only when `step % precondition_frequency == 0`.

    Args:
        params: Live model parameter `W` (rank-2, R×C; trains at the high LR).
        params_ema: EMA (iterate-averaged) copy of the params (R×C; initialize to a
            copy of `params`). This is the sequence to evaluate on.
        gradients: Gradient `g` (R×C).
        exp_avg: First-moment buffer `m` (R×C; zeros initially).
        gg_left: Left Kronecker factor `L` (R×R; zeros initially).
        gg_right: Right Kronecker factor `M` (C×C; zeros initially).
        q_left: Left eigenbasis `Q_L` (R×R; zeros initially — built on step 1).
        q_right: Right eigenbasis `Q_R` (C×C; zeros initially — built on step 1).
        step: 1-indexed global step (increment before calling).
        learning_rate: Base learning rate (SPlus uses a width-invariant, high LR).
        beta1: First-moment decay (default 0.9).
        beta2: Kronecker-factor EMA decay (default 0.99).
        ema_rate: Iterate-averaging decay for `params_ema` (default 0.999).
        weight_decay: Decoupled weight-decay factor on the live params (default 0.0).
        precondition_frequency: Eigenbasis refresh period (default 100).
        sign_eps: Deadzone for the eigenbasis sign step — projected components with
            magnitude below `sign_eps` map to 0 instead of an arbitrary ±1 (default
            1e-12). This suppresses the sign of numerically-zero eigen-directions
            (e.g. every off-diagonal on step 1, where the freshly-built eigenbasis
            exactly diagonalizes the gradient), keeping the update well-defined and
            eigensolver-independent.

    Returns:
        Tuple `(new_params, new_params_ema, new_exp_avg, new_gg_left, new_gg_right,
        new_q_left, new_q_right)`.

    Raises:
        Error: If params is not rank-2.
    """
    if params.ndim() != 2:
        raise Error("splus_step requires a rank-2 (matrix) parameter")

    var shape = params.shape()
    var R = shape[0]
    var C = shape[1]

    # --- 1. Update the Kronecker factors: L = EMA(g gᵀ), M = EMA(gᵀ g). ---
    var g_gt = matmul(gradients, transpose(gradients, None))  # R×R
    var gt_g = matmul(transpose(gradients, None), gradients)  # C×C
    var w = 1.0 - beta2
    var new_gg_left = add_simd(_scale(gg_left, beta2), _scale(g_gt, w))
    var new_gg_right = add_simd(_scale(gg_right, beta2), _scale(gt_g, w))

    # --- 2. Eigenbasis: build on step 1, refresh every precondition_frequency. ---
    var new_q_left = q_left
    var new_q_right = q_right
    var need_eig = step == 1 or (step % precondition_frequency == 0)
    if need_eig:
        var el = symmetric_eigh(new_gg_left)
        var er = symmetric_eigh(new_gg_right)
        # symmetric_eigh returns ascending eigenvalues; flip columns to descending
        # (matching SOAP / torch.flip(dims=[1])).
        new_q_left = _flip_columns(el[1])
        new_q_right = _flip_columns(er[1])

    # --- 3. First moment on the RAW gradient. ---
    var new_exp_avg = add_simd(
        _scale(exp_avg, beta1), _scale(gradients, 1.0 - beta1)
    )

    # --- 4. Project the first moment into the eigenbasis, then SIGN it. ---
    var m_proj = matmul(
        matmul(transpose(new_q_left, None), new_exp_avg), new_q_right
    )
    var u_proj = _sign_tensor(m_proj, sign_eps)

    # --- 5. Rotate the signed update back: u = Q_L sign(m') Q_Rᵀ. ---
    var u = matmul(matmul(new_q_left, u_proj), transpose(new_q_right, None))

    # --- 6. Shape-aware scaling by the layer's dimensional ratio. ---
    var scale = 2.0 / (Float64(R) + Float64(C))

    # --- 7. Live params (decoupled weight decay), then EMA-average. ---
    var new_params = subtract_simd(params, _scale(u, learning_rate * scale))
    if weight_decay != 0.0:
        new_params = subtract_simd(
            new_params, _scale(params, learning_rate * weight_decay)
        )
    var new_params_ema = add_simd(
        _scale(params_ema, ema_rate), _scale(new_params, 1.0 - ema_rate)
    )

    return (
        new_params,
        new_params_ema,
        new_exp_avg,
        new_gg_left,
        new_gg_right,
        new_q_left,
        new_q_right,
    )


def init_splus_state(
    params: AnyTensor,
) raises -> Tuple[
    AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor
]:
    """Allocate SPlus state for a 2-D parameter `W (R×C)`.

    Returns `(params_ema, exp_avg, gg_left, gg_right, q_left, q_right)` — the EMA
    param copy (R×C, seeded to a copy of `params`), the first-moment buffer (R×C,
    zeros), the Kronecker factors (R×R, C×C, zeros), and the eigenbases (R×R, C×C,
    zeros — populated on the first `splus_step` call). All float64.

    Args:
        params: The 2-D parameter whose shape defines the state shapes.

    Returns:
        Tuple of six float64 state tensors.

    Raises:
        Error: If params is not rank-2.
    """
    if params.ndim() != 2:
        raise Error("init_splus_state requires a rank-2 (matrix) parameter")
    var shape = params.shape()
    var R = shape[0]
    var C = shape[1]
    # EMA sequence starts at the initial params (a fresh float64 copy).
    var params_ema = zeros([R, C], DType.float64)
    for i in range(R * C):
        params_ema.store[DType.float64](i, params.load[DType.float64](i))
    var exp_avg = zeros([R, C], DType.float64)
    var gg_left = zeros([R, R], DType.float64)
    var gg_right = zeros([C, C], DType.float64)
    var q_left = zeros([R, R], DType.float64)
    var q_right = zeros([C, C], DType.float64)
    return (params_ema, exp_avg, gg_left, gg_right, q_left, q_right)


# ---- small elementwise helpers (fresh-tensor, float64) --------------------------


def _scale(x: AnyTensor, s: Float64) raises -> AnyTensor:
    """Elementwise scalar multiply (fresh tensor)."""
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, x.load[DType.float64](i) * s)
    return out


def _sign_tensor(x: AnyTensor, eps: Float64) raises -> AnyTensor:
    """Elementwise sign with a deadzone: |v| < eps -> 0, v > 0 -> +1, v < 0 -> -1.

    The deadzone suppresses the sign of numerically-zero projected components (the
    eigen-directions carrying no momentum energy). Without it, `sign` of a ~1e-18
    value returns an arbitrary ±1 whose sign differs between eigensolvers.
    """
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        var v = x.load[DType.float64](i)
        var av = v if v >= 0.0 else -v
        var s = 0.0
        if av >= eps:
            s = 1.0 if v > 0.0 else -1.0
        out.store[DType.float64](i, s)
    return out


def _flip_columns(m: AnyTensor) raises -> AnyTensor:
    """Reverse the column order of a rank-2 matrix (torch.flip(dims=[1]))."""
    var shape = m.shape()
    var rows = shape[0]
    var cols = shape[1]
    var out = zeros([rows, cols], DType.float64)
    for r in range(rows):
        for c in range(cols):
            out.store[DType.float64](
                r * cols + c, m.load[DType.float64](r * cols + (cols - 1 - c))
            )
    return out
