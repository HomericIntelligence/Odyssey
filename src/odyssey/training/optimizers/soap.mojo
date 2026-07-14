"""SOAP optimizer — ShampoO with Adam in the Preconditioner eigenbasis.

SOAP (Vyas et al., 2024) runs Adam in the slowly-varying eigenbasis of Shampoo's
Kronecker preconditioners. For a 2-D weight `W (R×C)` it maintains two Kronecker
factors — a left factor `L (R×R) = EMA(g gᵀ)` and a right factor `M (C×C) = EMA(gᵀ g)`
— and their eigenvector bases `Q_L`, `Q_R`. Each step rotates the gradient into that
eigenbasis, runs Adam there, and rotates the update back:

    g'      = Q_Lᵀ g Q_R                     # project into the eigenbasis
    m       = β1 m + (1-β1) g                # Adam first moment  (RAW grad)
    v       = β2 v + (1-β2) g'²              # Adam second moment (PROJECTED grad)
    m'      = Q_Lᵀ m Q_R                     # project the first moment
    u       = Q_L (m' / (√v + ε)) Q_Rᵀ       # Adam update, rotated back
    W       = W - step_size * u - lr*wd*W    # decoupled weight decay

with `step_size = lr * √(1-β2ᵗ)/(1-β1ᵗ)` when bias correction is on. The eigenbases
are refreshed every `precondition_frequency` steps.

**Fidelity note:** the reference (`pytorch_optimizer.SOAP`) refreshes the eigenbasis
with a power-iteration + QR approximation for speed; this implementation refreshes it
with a full symmetric eigendecomposition (`symmetric_eigh`), which yields the exact
eigenbasis the QR variant approximates. Handles 2-D (matrix) parameters — the shape
all Kronecker-preconditioned weights take. Pure-functional: the caller owns all state
(exp_avg, exp_avg_sq, the two Kronecker factors, and the two eigenbases) and threads
it across steps.

Reference:
    Vyas, N., Morwani, D., Zhao, R., et al. (2024). SOAP: Improving and Stabilizing
    Shampoo using Adam. arXiv:2409.11321.
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, zeros_like
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, subtract_simd, multiply_simd
from odyssey.core.eigen import symmetric_eigh


def soap_step(
    params: AnyTensor,
    gradients: AnyTensor,
    exp_avg: AnyTensor,
    exp_avg_sq: AnyTensor,
    gg_left: AnyTensor,
    gg_right: AnyTensor,
    q_left: AnyTensor,
    q_right: AnyTensor,
    step: Int,
    learning_rate: Float64,
    beta1: Float64 = 0.95,
    beta2: Float64 = 0.95,
    shampoo_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.01,
    precondition_frequency: Int = 10,
    epsilon: Float64 = 1e-8,
    correct_bias: Bool = True,
) raises -> Tuple[
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
]:
    """Perform a single SOAP step for a 2-D parameter — pure functional.

    The caller owns all state and passes it in each step. `step` is 1-indexed
    (increment BEFORE calling, matching the reference's `group['step'] += 1` before
    the parameter loop). On `step == 1` the eigenbases are built from the freshly-
    updated Kronecker factors; on later steps they are refreshed only when
    `step % precondition_frequency == 0`.

    Args:
        params: Model parameter `W` (rank-2, R×C).
        gradients: Gradient `g` (R×C).
        exp_avg: Adam first-moment buffer `m` (R×C; zeros initially).
        exp_avg_sq: Adam second-moment buffer `v` (R×C; zeros initially).
        gg_left: Left Kronecker factor `L` (R×R; zeros initially).
        gg_right: Right Kronecker factor `M` (C×C; zeros initially).
        q_left: Left eigenbasis `Q_L` (R×R; zeros initially — built on step 1).
        q_right: Right eigenbasis `Q_R` (C×C; zeros initially — built on step 1).
        step: 1-indexed global step (increment before calling).
        learning_rate: Base learning rate.
        beta1: Adam first-moment decay (default 0.95).
        beta2: Adam second-moment decay (default 0.95).
        shampoo_beta: EMA decay for the Kronecker factors (default 0.95).
        weight_decay: Decoupled weight-decay factor (default 0.01).
        precondition_frequency: Eigenbasis refresh period (default 10).
        epsilon: Denominator stabilizer (default 1e-8).
        correct_bias: Apply Adam bias correction to the step size (default True).

    Returns:
        Tuple `(new_params, new_exp_avg, new_exp_avg_sq, new_gg_left, new_gg_right,
        new_q_left, new_q_right)`.

    Raises:
        Error: If params is not rank-2.
    """
    if params.ndim() != 2:
        raise Error("soap_step requires a rank-2 (matrix) parameter")

    # --- 1. Update the Kronecker factors: L = EMA(g gᵀ), M = EMA(gᵀ g). ---
    var g_gt = matmul(gradients, transpose(gradients, None))  # R×R
    var gt_g = matmul(transpose(gradients, None), gradients)  # C×C
    var w = 1.0 - shampoo_beta
    var new_gg_left = add_simd(_scale(gg_left, shampoo_beta), _scale(g_gt, w))
    var new_gg_right = add_simd(_scale(gg_right, shampoo_beta), _scale(gt_g, w))

    # --- 2. Eigenbasis: build on step 1, refresh every precondition_frequency. ---
    var new_q_left = q_left
    var new_q_right = q_right
    var need_eig = step == 1 or (step % precondition_frequency == 0)
    if need_eig:
        var el = symmetric_eigh(new_gg_left)
        var er = symmetric_eigh(new_gg_right)
        # symmetric_eigh returns ascending eigenvalues; the reference flips columns
        # (torch.flip(q, dims=[1])) so the basis is descending. Match that.
        new_q_left = _flip_columns(el[1])
        new_q_right = _flip_columns(er[1])

    # --- 3. Project gradient into the eigenbasis: g' = Q_Lᵀ g Q_R. ---
    var g_proj = matmul(
        matmul(transpose(new_q_left, None), gradients), new_q_right
    )

    # --- 4. Adam moments: m on RAW grad, v on PROJECTED grad squared. ---
    var new_exp_avg = add_simd(
        _scale(exp_avg, beta1), _scale(gradients, 1.0 - beta1)
    )
    var g_proj_sq = multiply_simd(g_proj, g_proj)
    var new_exp_avg_sq = add_simd(
        _scale(exp_avg_sq, beta2), _scale(g_proj_sq, 1.0 - beta2)
    )

    # --- 5. Rotated Adam update, projected back: Q_L (m' / (√v + ε)) Q_Rᵀ. ---
    var denom = _add_scalar(_sqrt_tensor(new_exp_avg_sq), epsilon)
    var m_proj = matmul(
        matmul(transpose(new_q_left, None), new_exp_avg), new_q_right
    )
    var norm_grad_proj = _div(m_proj, denom)
    var norm_grad = matmul(
        matmul(new_q_left, norm_grad_proj), transpose(new_q_right, None)
    )

    # --- 6. Step size (Adam bias correction) + decoupled weight decay. ---
    var step_size = learning_rate
    if correct_bias:
        var bc1 = 1.0 - _pow(beta1, step)
        var bc2_sq = scalar_sqrt(1.0 - _pow(beta2, step))
        step_size = step_size * bc2_sq / bc1

    var new_params = subtract_simd(params, _scale(norm_grad, step_size))
    if weight_decay != 0.0:
        new_params = subtract_simd(
            new_params, _scale(params, learning_rate * weight_decay)
        )

    return (
        new_params,
        new_exp_avg,
        new_exp_avg_sq,
        new_gg_left,
        new_gg_right,
        new_q_left,
        new_q_right,
    )


def init_soap_state(
    params: AnyTensor,
) raises -> Tuple[
    AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor, AnyTensor
]:
    """Allocate zeroed SOAP state for a 2-D parameter `W (R×C)`.

    Returns `(exp_avg, exp_avg_sq, gg_left, gg_right, q_left, q_right)` — the Adam
    moments (R×C), the Kronecker factors (R×R, C×C), and the eigenbases (R×R, C×C),
    all zeros. The eigenbases are populated on the first `soap_step` call.

    Args:
        params: The 2-D parameter whose shape defines the state shapes.

    Returns:
        Tuple of six zeroed float64 state tensors.

    Raises:
        Error: If params is not rank-2.
    """
    if params.ndim() != 2:
        raise Error("init_soap_state requires a rank-2 (matrix) parameter")
    var shape = params.shape()
    var R = shape[0]
    var C = shape[1]
    var exp_avg = zeros([R, C], DType.float64)
    var exp_avg_sq = zeros([R, C], DType.float64)
    var gg_left = zeros([R, R], DType.float64)
    var gg_right = zeros([C, C], DType.float64)
    var q_left = zeros([R, R], DType.float64)
    var q_right = zeros([C, C], DType.float64)
    return (exp_avg, exp_avg_sq, gg_left, gg_right, q_left, q_right)


# ---- small elementwise helpers (fresh-tensor, float64) --------------------------


def _scale(x: AnyTensor, s: Float64) raises -> AnyTensor:
    """Elementwise scalar multiply (fresh tensor)."""
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, x.load[DType.float64](i) * s)
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


def _sqrt_tensor(x: AnyTensor) raises -> AnyTensor:
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, scalar_sqrt(x.load[DType.float64](i)))
    return out


def _add_scalar(x: AnyTensor, s: Float64) raises -> AnyTensor:
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, x.load[DType.float64](i) + s)
    return out


def _div(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    var out = zeros_like(a)
    var n = a.numel()
    for i in range(n):
        out.store[DType.float64](
            i, a.load[DType.float64](i) / b.load[DType.float64](i)
        )
    return out


def _pow(base: Float64, exp: Int) raises -> Float64:
    var r = 1.0
    for _ in range(exp):
        r = r * base
    return r
