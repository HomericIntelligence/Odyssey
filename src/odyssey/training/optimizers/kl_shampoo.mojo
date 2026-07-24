"""KL-Shampoo optimizer — Adam-free stable Shampoo via KL-divergence minimization.

KL-Shampoo (Lin et al., 2025) recasts Shampoo's Kronecker-factor estimation as
covariance estimation under Kullback–Leibler (KL) divergence minimization instead
of the Frobenius-norm view underlying standard Shampoo. The KL objective respects
the symmetric-positive-definite (SPD) constraint on the preconditioner factors,
which the Frobenius norm ignores. The practical payoff: KL-Shampoo stabilizes
WITHOUT the in-eigenbasis Adam moments that SOAP relies on, eliminating Adam's
first/second-moment memory overhead while matching or exceeding SOAP.

For a 2-D weight `W (R×C)` KL-Shampoo maintains two SPD Kronecker factors — a left
factor `S_A (R×R)` (row covariance) and a right factor `S_B (C×C)` (column
covariance). The distinguishing feature is the COUPLED update: each factor is
updated with the gradient whitened by the *cross* factor's inverse, which is the
stationarity condition (∂KL/∂S_A = 0, ∂KL/∂S_B = 0) of the KL objective rather than
the raw Gram outer products `G Gᵀ` / `Gᵀ G` that standard Shampoo accumulates.

Idealized KL-Shampoo update (paper Eq. 5, per step):

    S_A ← (1 − β) · S_A + (β / d_B) · G  S_B⁻¹  Gᵀ            # R×R, uses OLD S_B
    S_B ← (1 − β) · S_B + (β / d_A) · Gᵀ S_A⁻¹  G             # C×C, uses OLD S_A
    W   ← W − γ · S_A^{-1/2}  G  S_B^{-1/2}                    # preconditioned step

where `d_A = R`, `d_B = C`, β is the preconditioner moving-average weight, and γ is
the learning rate. Both factor updates read the OLD (pre-update) inverses, so the
step is order-independent and matches the coupled fixed-point form.

**Root note (why -1/2, not Shampoo's -1/4):** because each factor already absorbs
the cross-factor whitening in its update, the preconditioner is `S_A^{-1/2} ⊗
S_B^{-1/2}` — an inverse SQUARE root per factor. Standard Shampoo accumulates raw
`G Gᵀ` and therefore uses the inverse FOURTH root `G^{-1/(2k)}` (k=2 ⇒ -1/4). Do
not port Shampoo's -1/4 here.

**Numerics (SOAP lesson) — float64 params only:** Odyssey's `matmul` raises on
mixed dtypes and f32 preconditioner math with f64 state produces garbage, so ALL
state, the gradient, and the preconditioned delta are float64, and the parameter
subtraction (`subtract_simd`) requires the delta and `params` to share a dtype.
The params must therefore be **float64** — an f32 param raises "Cannot subtract
tensors with different dtypes" at the final update (same posture as SOAP; there is
no f32 fast path). The factor inverses and inverse-square-roots
are computed from a single symmetric eigendecomposition (`symmetric_eigh`) per
factor — `S⁻¹ = Q diag(1/λ) Qᵀ` and `S^{-1/2} = Q diag(λ^{-1/2}) Qᵀ` — which is the
same exact-eigenbasis approach SOAP uses (not Newton-Schulz), and is exact and
stable for the small SPD factors that arise here. A `ridge` floor is added to the
eigenvalues before inversion to keep S⁻¹ finite when a factor is near-singular
(e.g. the identity-initialized factor on early steps).

Pure-functional: the caller owns both factors and threads them across steps.

Reference:
    Lin, W., Lowe, S. C., Dangel, F., Eschenhagen, R., Xu, Z., & Grosse, R. B.
    (2025). Understanding and Improving Shampoo and SOAP via Kullback–Leibler
    Minimization. arXiv:2509.03378.
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import eye, zeros, zeros_like
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, subtract_simd
from odyssey.core.eigen import symmetric_eigh


def is_kl_shampoo_eligible(params: AnyTensor) -> Bool:
    """Check whether a parameter is eligible for KL-Shampoo optimization.

    KL-Shampoo, like Shampoo, is designed for matrix-shaped parameters (rank-2
    tensors) with both dimensions >= 2. Biases, embeddings, and other vector /
    scalar parameters are not eligible.

    Args:
        params: Tensor to check for eligibility.

    Returns:
        True if params is a matrix with both dimensions >= 2, else False.
    """
    if params.ndim() != 2:
        return False
    var shape = params.shape()
    return shape[0] >= 2 and shape[1] >= 2


def _init_kl_shampoo_state_single_matrix(
    params: AnyTensor,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Allocate KL-Shampoo state for a 2-D parameter `W (R×C)`.

    Returns `(S_A, S_B)` — the left (R×R) and right (C×C) Kronecker factors,
    each initialized to the IDENTITY (float64). The identity init keeps the
    cross-factor inverse `S_B⁻¹` / `S_A⁻¹` well-defined on the very first step
    (a zero init would make the coupled Eq. 5 update singular).

    Args:
        params: The 2-D parameter whose shape defines the factor shapes.

    Returns:
        Tuple `(S_A, S_B)` of two identity float64 matrices (R×R and C×C).

    Raises:
        Error: If params is not rank-2, or either dimension is < 2 (a degenerate
            1×N / N×1 matrix is not KL-Shampoo-eligible).
    """
    if params.ndim() != 2:
        raise Error(
            "init_kl_shampoo_state requires a rank-2 (matrix) parameter"
        )
    var shape = params.shape()
    var R = shape[0]
    var C = shape[1]
    if R < 2 or C < 2:
        raise Error(
            "init_kl_shampoo_state requires both dimensions >= 2 (a degenerate"
            " 1×N / N×1 matrix is not KL-Shampoo-eligible)"
        )
    var S_A = eye(R, R, 0, DType.float64)
    var S_B = eye(C, C, 0, DType.float64)
    return (S_A, S_B)


def kl_shampoo_step(
    params: AnyTensor,
    gradients: AnyTensor,
    s_a: AnyTensor,
    s_b: AnyTensor,
    learning_rate: Float64,
    beta: Float64 = 0.95,
    weight_decay: Float64 = 0.0,
    ridge: Float64 = 1e-8,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single KL-Shampoo step for a 2-D parameter — pure functional.

    Implements the idealized KL-Shampoo update (paper Eq. 5): the two Kronecker
    factors are updated with the gradient whitened by the OLD cross-factor inverse
    (the KL stationarity condition), then the parameter takes an inverse-square-root
    preconditioned step. The caller owns `s_a` / `s_b` and threads them across
    steps; both are initialized to the identity via `init_kl_shampoo_state`.

    All state is float64 and every root/inverse is computed in float64 (Odyssey's
    `matmul` raises on mixed dtypes). The preconditioned delta is float64, and the
    final `subtract_simd(params, delta)` requires `params` to be **float64** too —
    an f32 param raises "Cannot subtract tensors with different dtypes" (same
    float64-only posture as SOAP; there is no f32 fast path).

    Args:
        params: Model parameter `W` (rank-2, R×C, float64).
        gradients: Gradient `G` (R×C, same shape as params).
        s_a: Left Kronecker factor `S_A` (R×R; identity initially).
        s_b: Right Kronecker factor `S_B` (C×C; identity initially).
        learning_rate: Base learning rate γ.
        beta: Preconditioner moving-average weight β (default 0.95).
        weight_decay: Decoupled weight-decay factor (default 0.0).
        ridge: Eigenvalue floor added before inversion for numerical stability
            (default 1e-8).

    Returns:
        Tuple `(new_params, new_s_a, new_s_b)`.

    Raises:
        Error: If params is not rank-2, either dimension is < 2, or shapes /
            dtypes are inconsistent (including a non-float64 param, which raises
            at the final subtraction).
    """
    if params.ndim() != 2:
        raise Error("kl_shampoo_step requires a rank-2 (matrix) parameter")
    if params.shape()[0] < 2 or params.shape()[1] < 2:
        raise Error(
            "kl_shampoo_step requires both dimensions >= 2 (a degenerate 1×N /"
            " N×1 matrix is not KL-Shampoo-eligible)"
        )
    if params.shape() != gradients.shape():
        raise Error(
            "kl_shampoo_step: params and gradients must have same shape"
        )
    if params.dtype() != gradients.dtype():
        raise Error(
            "kl_shampoo_step: params and gradients must have same dtype"
        )

    var shape = params.shape()
    var R = shape[0]
    var C = shape[1]
    var s_a_shape = s_a.shape()
    var s_b_shape = s_b.shape()
    if s_a_shape[0] != R or s_a_shape[1] != R:
        raise Error("kl_shampoo_step: S_A must be R×R")
    if s_b_shape[0] != C or s_b_shape[1] != C:
        raise Error("kl_shampoo_step: S_B must be C×C")

    # Copy gradient into a fresh float64 tensor so all preconditioner math stays
    # in f64 (the SOAP mixed-dtype lesson). The final delta is float64, so the
    # subtract_simd against params requires params to be float64 too (an f32 param
    # raises there — this optimizer is float64-only, like SOAP).
    var g64 = _to_f64(gradients)
    var gt = transpose(g64, None)

    # --- Inverses of the OLD factors (KL coupled update reads pre-update state). ---
    var s_a_inv = _sym_inv(s_a, ridge)  # R×R
    var s_b_inv = _sym_inv(s_b, ridge)  # C×C

    # --- Factor updates (Eq. 5): whiten G by the CROSS factor's inverse. ---
    # S_A ← (1−β) S_A + (β / d_B) · G S_B⁻¹ Gᵀ         (d_B = C)
    var g_sbinv = matmul(g64, s_b_inv)  # R×C
    var a_term = matmul(g_sbinv, gt)  # R×R
    var new_s_a = add_simd(
        _scale(s_a, 1.0 - beta), _scale(a_term, beta / Float64(C))
    )
    # S_B ← (1−β) S_B + (β / d_A) · Gᵀ S_A⁻¹ G         (d_A = R)
    var gt_sainv = matmul(gt, s_a_inv)  # C×R
    var b_term = matmul(gt_sainv, g64)  # C×C
    var new_s_b = add_simd(
        _scale(s_b, 1.0 - beta), _scale(b_term, beta / Float64(R))
    )

    # --- Preconditioned step: W ← W − γ · S_A^{-1/2} G S_B^{-1/2}. ---
    # Inverse square roots use the UPDATED factors (the current preconditioner).
    var s_a_inv_sqrt = _sym_inv_sqrt(new_s_a, ridge)  # R×R
    var s_b_inv_sqrt = _sym_inv_sqrt(new_s_b, ridge)  # C×C
    var left = matmul(s_a_inv_sqrt, g64)  # R×C
    var precond = matmul(left, s_b_inv_sqrt)  # R×C

    var new_params = subtract_simd(params, _scale(precond, learning_rate))
    if weight_decay != 0.0:
        new_params = subtract_simd(
            new_params, _scale(params, learning_rate * weight_decay)
        )

    return (new_params, new_s_a, new_s_b)


def kl_shampoo_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    s_a: AnyTensor,
    s_b: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Convenience wrapper around `kl_shampoo_step` with default hyperparameters.

    Uses β = 0.95, no weight decay, and the default ridge floor. The caller still
    owns `s_a` / `s_b` and threads them across steps.

    Args:
        params: Model parameter `W` (rank-2, R×C).
        gradients: Gradient `G` (R×C).
        s_a: Left Kronecker factor `S_A` (R×R).
        s_b: Right Kronecker factor `S_B` (C×C).
        learning_rate: Base learning rate γ.

    Returns:
        Tuple `(new_params, new_s_a, new_s_b)`.

    Raises:
        Error: If shapes or dtypes are inconsistent.
    """
    return kl_shampoo_step(
        params,
        gradients,
        s_a,
        s_b,
        learning_rate,
        beta=0.95,
        weight_decay=0.0,
        ridge=1e-8,
    )


# ---- small helpers (fresh-tensor, float64) --------------------------------------


def _to_f64(x: AnyTensor) raises -> AnyTensor:
    """Copy a tensor into a fresh float64 tensor of the same shape."""
    var shape_list = List[Int]()
    var sh = x.shape()
    for i in range(x.ndim()):
        shape_list.append(sh[i])
    var out = zeros(shape_list, DType.float64)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, x.load[DType.float64](i))
    return out


def _scale(x: AnyTensor, s: Float64) raises -> AnyTensor:
    """Elementwise scalar multiply (fresh float64 tensor)."""
    var out = zeros_like(x)
    var n = x.numel()
    for i in range(n):
        out.store[DType.float64](i, x.load[DType.float64](i) * s)
    return out


def _reconstruct(q: AnyTensor, d: List[Float64], n: Int) raises -> AnyTensor:
    """Form `Q diag(d) Qᵀ` for an N×N eigenbasis Q and eigenvalue-map list d."""
    # Scale the columns of Q by d, then multiply by Qᵀ.
    var qd = zeros([n, n], DType.float64)
    for r in range(n):
        for c in range(n):
            qd.store[DType.float64](
                r * n + c, q.load[DType.float64](r * n + c) * d[c]
            )
    return matmul(qd, transpose(q, None))


def _sym_inv(s: AnyTensor, ridge: Float64) raises -> AnyTensor:
    """Inverse of a symmetric PSD matrix: `Q diag(1/(λ+ridge)) Qᵀ` (float64)."""
    var n = s.shape()[0]
    var e = symmetric_eigh(s)
    var vals = e[0]
    var q = e[1]
    var inv_vals = List[Float64]()
    for i in range(n):
        inv_vals.append(1.0 / (vals.load[DType.float64](i) + ridge))
    return _reconstruct(q, inv_vals, n)


def _sym_inv_sqrt(s: AnyTensor, ridge: Float64) raises -> AnyTensor:
    """Inverse square root of a symmetric PSD matrix: `Q diag((λ+ridge)^{-1/2}) Qᵀ`.
    """
    var n = s.shape()[0]
    var e = symmetric_eigh(s)
    var vals = e[0]
    var q = e[1]
    var inv_sqrt_vals = List[Float64]()
    for i in range(n):
        var lam = vals.load[DType.float64](i) + ridge
        inv_sqrt_vals.append(1.0 / scalar_sqrt(lam))
    return _reconstruct(q, inv_sqrt_vals, n)


def init_kl_shampoo_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter KL-Shampoo state buffers (matrix-only).

    For rank-2 matrix params (both dims >= 2) emits two identity float64 Kronecker factors `[S_A, S_B]`. For non-matrix params emits an empty list (caller routes via AdamW).

    Args:
        params_list: Model parameters.
        force_f64: Ignored -- this optimizer always emits float64 state.

    Returns:
        A list of state buffer lists in the same order as `params_list`. Matrix params get `[S_A, S_B]`; non-matrix get `[]`.
    """
    var all_states: List[List[AnyTensor]] = []
    for i in range(len(params_list)):
        var p = params_list[i]
        var per: List[AnyTensor] = []
        if p.ndim() == 2:
            var sh = p.shape()
            var R = sh[0]
            var C = sh[1]
            if R >= 2 and C >= 2:
                var unpacked = _init_kl_shampoo_state_single_matrix(p)
                per.append(unpacked[0])
                per.append(unpacked[1])
        all_states.append(per^)
    return all_states^
