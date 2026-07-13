"""Shampoo optimizer with two-sided matrix preconditioning via Newton-Schulz.

This module provides the Shampoo optimizer for matrix-shaped parameters. Shampoo applies
a two-sided matrix preconditioning via Newton-Schulz iteration to compute the inverse
fourth root of accumulated Gram matrices, which provides adaptive step sizes that improve
conditioning compared to SGD with momentum.

Key Concepts:
    Shampoo maintains two accumulator matrices L ∈ R^{m×m} and R ∈ R^{n×n} that
    track the left and right Gram matrices of gradient history:
        L_t = β·L_{t-1} + G G^T   (left Gram matrix)
        R_t = β·R_{t-1} + G^T G   (right Gram matrix)

    Note on the β factor: the original Gupta et al. 2018 paper accumulates a pure
    running SUM (β = 1: L_t = L_{t-1} + G Gᵀ). The exponential-moving-average form
    used here (β < 1) is the Scalable-Shampoo variant of Anil et al. 2020, which
    keeps the accumulators bounded for long training runs. This is a standard,
    well-documented variant — not a deviation from the preconditioner math.

    The preconditioner matrices are the inverse fourth roots:
        L_precond = L^{-1/4}
        R_precond = R^{-1/4}

    The -1/4 power is the Gupta et al. G^{-1/(2k)} rule for an order-k tensor: a
    parameter matrix is an order-2 tensor (k = 2), so the per-dimension power is
    -1/(2·2) = -1/4.

    The preconditioned gradient is computed as:
        g_precond = L_precond @ G @ R_precond

    This adaptive preconditioning reduces the condition number of the parameter
    update, leading to faster convergence in practice.

Calling Convention (Important):
    Shampoo uses an asymmetric calling convention that differs from typical optimizers:
    - initialize_shampoo_state() returns THREE buffers: (L, R, momentum)
    - shampoo_step() accepts FIVE state arguments: (params, gradients, L, R, momentum)
    - shampoo_step() returns FOUR state outputs: (params_new, L_new, R_new, momentum_new)
    - The CALLER continues to hold and manage the params tensor itself

    This design avoids a 5-tuple in/out signature which would be awkward in Mojo.

    Example usage:
        var (L, R, m) = initialize_shampoo_state(params)
        # ... training loop:
        var (params, L, R, m) = shampoo_step(params, gradients, L, R, m, learning_rate=0.01)

Preconditioner Stability:
    The Gram matrix accumulators L and R can grow without bound if ||gradients|| is large.
    To prevent numerical instability, _clamp_precond_norm() enforces:
        ||L||_F ≤ max_norm  and  ||R||_F ≤ max_norm

    Clamping a PSD matrix by a positive scalar preserves both symmetry (scalar scaling
    commutes with transpose) and positive semi-definiteness (eigenvalues scale by the
    same positive scalar). The returned accumulators L_new/R_new are UNCLAMPED — clamping
    only affects the input to newton_schulz_inv_fourth_root(). This allows caller flexibility
    in managing accumulator growth if desired.

Newton-Schulz Convergence:
    A single cubic Newton-Schulz iteration (coefficients (15I − 10·YZ + 3·YZ²)/8)
    converges to the inverse SQUARE root M^{-1/2}, not the inverse fourth root.
    newton_schulz_inv_fourth_root() therefore composes two inverse-square-root stages
    (S = M^{-1/2}; H = M·S = M^{1/2}; R = H^{-1/2} = M^{-1/4}). It then verifies
    ||M · Y^4 − I||_F ≤ convergence_tol against the ORIGINAL M (not the normalized
    M_norm) and raises Error if the tolerance is exceeded. The default tolerance
    (5e-2) is empirically calibrated for fp32 + 8 iterations. Tighter tolerances can
    be requested per-call for mathematical tests where the answer is analytically
    known (e.g., diagonal matrices).

Reference:
    Gupta, Koren, Singer 2018, "Shampoo: Preconditioned Stochastic Tensor
    Optimization", https://arxiv.org/abs/1802.09568
    Anil et al. 2020, "Scalable Second Order Optimization for Deep Learning",
    https://arxiv.org/abs/2002.09018
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import eye, zeros_like, full_like
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.matrix import matmul, transpose
from odyssey.training.optimizers.optimizer_utils import (
    compute_tensor_norm,
)


def is_shampoo_eligible(params: AnyTensor) -> Bool:
    """Check if a parameter tensor is eligible for Shampoo optimization.

    Shampoo is designed for matrix-shaped parameters (rank-2 tensors). It is not
    applicable to embeddings, biases, or other scalar/vector parameters.

    A parameter is eligible if:
    - Rank is exactly 2 (a matrix)
    - Both dimensions >= 2 (avoids degenerate cases where one dimension is 1)

    Args:
        params: Tensor to check for Shampoo eligibility.

    Returns:
        True if params is a matrix with both dimensions >= 2, False otherwise.

    Example:
        ```mojo
        var weight = zeros([784, 128], DType.float32)
        assert is_shampoo_eligible(weight)  # True

        var bias = zeros([128], DType.float32)
        assert not is_shampoo_eligible(bias)  # False
        ```
    """
    if params.ndim() != 2:
        return False

    var shape = params.shape()
    var rows = shape[0]
    var cols = shape[1]

    return rows >= 2 and cols >= 2


def initialize_shampoo_state(
    params: AnyTensor,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Initialize Shampoo optimizer state buffers.

    Creates three state buffers that must be passed to shampoo_step():
    - L: Identity matrix [m, m] accumulating G @ G^T
    - R: Identity matrix [n, n] accumulating G^T @ G
    - momentum: Zero tensor [m, n] for momentum accumulation

    Args:
        params: Parameter tensor [m, n] to initialize state for.

    Returns:
        Tuple of (L, R, momentum) state tensors.

    Raises:
        Error: If params is not rank-2.

    Note:
        The caller continues to hold the params tensor. Initialize_shampoo_state
        returns only the three state buffers (L, R, momentum).
    """
    if params.ndim() != 2:
        raise Error(
            "initialize_shampoo_state requires rank-2 tensor, got ndim: "
            + String(params.ndim())
        )

    var shape = params.shape()
    var m = shape[0]
    var n = shape[1]
    var dtype = params.dtype()

    var L = eye(m, m, 0, dtype)
    var R = eye(n, n, 0, dtype)
    var momentum = zeros_like(params)

    return (L, R, momentum)


def _trace_sum_diag(M: AnyTensor) raises -> Float64:
    """Compute the trace of a square matrix by summing diagonal entries.

    Args:
        M: Square matrix tensor (rank-2, shape [n, n]).

    Returns:
        Sum of diagonal entries M[0,0] + M[1,1] + ... + M[n-1,n-1].

    Raises:
        Error: If M is not rank-2 or not square.
    """
    if M.ndim() != 2:
        raise Error(
            "_trace_sum_diag requires rank-2 tensor, got ndim: "
            + String(M.ndim())
        )

    var shape = M.shape()
    var m = shape[0]
    var n = shape[1]

    if m != n:
        raise Error(
            "_trace_sum_diag requires square matrix, got ["
            + String(m)
            + ", "
            + String(n)
            + "]"
        )

    var total: Float64 = 0.0
    for i in range(m):
        var diag_entry = M._get_float64(i * m + i)
        total += diag_entry

    return total


def _clamp_precond_norm(
    M: AnyTensor, max_norm: Float64 = 1e6
) raises -> AnyTensor:
    """Clamp the Frobenius norm of a PSD matrix to a maximum value.

    Divides M by max(1.0, ||M||_F / max_norm) to ensure ||M_clamped||_F ≤ max_norm.

    Clamping preserves PSD structure: scaling a PSD matrix by a positive scalar
    preserves both symmetry and positive semi-definiteness.

    Args:
        M: Input tensor (typically a symmetric PSD matrix).
        max_norm: Maximum Frobenius norm (default: 1e6).

    Returns:
        Clamped tensor with ||clamped||_F ≤ max_norm.

    Raises:
        Error: If M is not rank-2.
    """
    if M.ndim() != 2:
        raise Error(
            "_clamp_precond_norm requires rank-2 tensor, got ndim: "
            + String(M.ndim())
        )

    var norm = compute_tensor_norm(M)
    var norm_ratio = norm / max_norm
    var scale_factor = 1.0
    if norm_ratio > 1.0:
        scale_factor = norm_ratio

    var scale_tensor = full_like(M, scale_factor)
    var M_clamped = divide_simd(M, scale_tensor)

    return M_clamped


def _newton_schulz_inv_sqrt(
    M: AnyTensor,
    steps: Int = 8,
    eps: Float64 = 1e-10,
    convergence_tol: Float64 = 5e-2,
) raises -> AnyTensor:
    """Compute the inverse square root of a symmetric PSD matrix via Newton-Schulz.

    Implements the cubically-convergent coupled Newton-Schulz iteration to compute
    Y such that Y^2 ≈ M^{-1} (i.e. Y ≈ M^{-1/2}). All products in the iteration are
    matrix products (matmul), never element-wise.

    Scalar reduction (M symmetric ⇒ all matrices commute; work per eigenvalue λ):
        y0 = 1, z0 = λ;  x = y·z, t = (15 − 10·x + 3·x²)/8, y ← y·t, z ← t·z.
        zₖ = λ·yₖ always ⇒ xₖ = λ·yₖ² → 1, so y → λ^{-1/2}.  (Verified: λ=2 → 0.707.)

    Because the iteration is run on the trace-normalized matrix M_norm = M / c, the
    output is M_norm^{-1/2}; the post-rescale by c^{-1/2} recovers M^{-1/2} exactly:
        M^{-1/2} = (c · M_norm)^{-1/2} = c^{-1/2} · M_norm^{-1/2}.

    Algorithm:
        1. Normalize: M_norm = M / (trace(M) + eps)  [λ_max(M_norm) ≤ 1 for PSD M]
        2. Initialize: Y_0 = I, Z_0 = M_norm
        3. For each iteration:
            - YZ = matmul(Y, Z)
            - YZ2 = matmul(YZ, YZ)
            - T = (15I − 10·YZ + 3·YZ2) / 8  [elementwise on same-shape matrices]
            - Y ← matmul(Y, T)
            - Z ← matmul(T, Z)
        4. Post-rescale: Y *= c^{-1/2}
        5. Verify convergence against the ORIGINAL M: ||M · Y^2 − I||_F ≤ tol

    Args:
        M: Symmetric PSD matrix.
        steps: Number of Newton-Schulz iterations (default: 8).
        eps: Numerical stability floor (default: 1e-10).
        convergence_tol: Maximum allowed residual ||M·Y^2 - I||_F (default: 5e-2).

    Returns:
        Matrix Y such that Y^2 ≈ M^{-1}.

    Raises:
        Error: If M is not rank-2, not square, or if convergence check fails.

    Note:
        The convergence check uses the ORIGINAL M (not M_norm) so that the residual
        is exactly ||M · Y^2 − I||; checking against M_norm would be off by a factor
        of 1/c and silently mask divergence.
    """
    if M.ndim() != 2:
        raise Error(
            "_newton_schulz_inv_sqrt requires rank-2 tensor, got ndim: "
            + String(M.ndim())
        )

    var shape = M.shape()
    var m = shape[0]
    var n = shape[1]

    if m != n:
        raise Error(
            "_newton_schulz_inv_sqrt requires square matrix, got ["
            + String(m)
            + ", "
            + String(n)
            + "]"
        )

    var dtype = M.dtype()

    # Step 1: Normalize M_norm = M / (trace(M) + eps).
    # Normalizing by the FULL trace (not trace/m) guarantees λ_max(M_norm) ≤ 1 for any
    # PSD M, since λ_max ≤ Σλ = trace. This keeps every eigenvalue inside the basin of
    # convergence of the cubic Newton-Schulz iteration (which only converges for
    # λ ≲ 2-3). Normalizing by trace/m instead sets the MEAN eigenvalue to 1, allowing
    # λ_max(M_norm) up to m, which pushes ill-conditioned accumulators out of the basin
    # and produces inf/NaN. The post-rescale 1/sqrt(c) is exact for any scalar c.
    var trace_M = _trace_sum_diag(M)
    var c = trace_M + eps
    var c_safe = max(c, eps)  # Floor on c for safety
    var c_tensor = full_like(M, c_safe)
    var M_norm = divide_simd(M, c_tensor)

    # Step 2: Initialize Y = I, Z = M_norm
    var Y = eye(m, m, 0, dtype)
    var Z = M_norm

    # Create identity matrix once for use in the loop
    var I = eye(m, m, 0, dtype)

    # Step 3: Coupled Newton-Schulz iteration (inverse square root)
    for _ in range(steps):
        # YZ = matmul(Y, Z)  # MATRIX
        var YZ = matmul(Y, Z)

        # YZ2 = matmul(YZ, YZ)  # MATRIX
        var YZ2 = matmul(YZ, YZ)

        # T = (15I − 10·YZ + 3·YZ2) / 8  # elementwise on same-shape matrices
        var fifteen_I = multiply_simd(full_like(I, 15.0), I)
        var ten_YZ = multiply_simd(full_like(YZ, 10.0), YZ)
        var three_YZ2 = multiply_simd(full_like(YZ2, 3.0), YZ2)

        var fifteen_I_minus_ten_YZ = subtract_simd(fifteen_I, ten_YZ)
        var numerator = add_simd(fifteen_I_minus_ten_YZ, three_YZ2)
        var eight_tensor = full_like(numerator, 8.0)
        var T = divide_simd(numerator, eight_tensor)

        # Y ← matmul(Y, T)  # MATRIX
        Y = matmul(Y, T)

        # Z ← matmul(T, Z)  # MATRIX
        Z = matmul(T, Z)

    # Step 4: Post-rescale Y *= c^{-1/2}  (Y currently ≈ M_norm^{-1/2})
    var sqrt_c = scalar_sqrt(c_safe)
    var rescale_factor = 1.0 / sqrt_c
    var rescale_tensor = full_like(Y, rescale_factor)
    Y = multiply_simd(Y, rescale_tensor)

    # Step 5: Convergence assertion against the ORIGINAL M: ||M · Y^2 − I||_F
    var Y2 = matmul(Y, Y)  # MATRIX
    var MY2 = matmul(M, Y2)  # MATRIX
    var residual = subtract_simd(MY2, I)
    var residual_norm = compute_tensor_norm(residual)

    # Use `not (<=)` rather than `>` so that a NaN residual (from a diverged
    # iteration) is treated as failure: `nan > tol` is False, but `nan <= tol` is
    # also False, so `not (nan <= tol)` is True and we raise instead of silently
    # returning a NaN-filled matrix.
    if not (residual_norm <= convergence_tol):
        raise Error(
            "_newton_schulz_inv_sqrt failed to converge; residual_norm: "
            + String(residual_norm)
            + ", tolerance: "
            + String(convergence_tol)
        )

    return Y


def newton_schulz_inv_fourth_root(
    M: AnyTensor,
    steps: Int = 8,
    eps: Float64 = 1e-10,
    convergence_tol: Float64 = 5e-2,
) raises -> AnyTensor:
    """Compute the inverse fourth root of a symmetric PSD matrix via Newton-Schulz.

    Computes Y such that Y^4 ≈ M^{-1} (i.e. Y ≈ M^{-1/4}), which is the per-dimension
    preconditioner power for an order-2 (matrix) tensor in Shampoo (Gupta et al. 2018:
    G^{-1/2k} for an order-k tensor, here k=2 ⇒ -1/4).

    The fourth root is obtained by composing the inverse-square-root iteration twice,
    exploiting commutativity of the symmetric PSD matrix with its own roots:

        S = M^{-1/2}              = _newton_schulz_inv_sqrt(M)
        H = M · S = M^{1/2}       (M and M^{-1/2} commute)
        R = H^{-1/2} = M^{-1/4}   = _newton_schulz_inv_sqrt(H)

    A single cubic Newton-Schulz iteration with the (15I − 10·YZ + 3·YZ²)/8 coefficients
    converges to the inverse SQUARE root, not the fourth root, so a direct application
    would compute M^{-1/2}. Composition is required to reach M^{-1/4}.

    Args:
        M: Symmetric PSD matrix.
        steps: Number of Newton-Schulz iterations per inv-sqrt stage (default: 8).
        eps: Numerical stability floor (default: 1e-10).
        convergence_tol: Maximum allowed residual ||M·Y^4 - I||_F (default: 5e-2).
                         Tighter values (e.g., 1e-2) can be passed for mathematical tests.

    Returns:
        Matrix Y such that Y^4 ≈ M^{-1}.

    Raises:
        Error: If M is not rank-2, not square, or if convergence check fails.
    """
    if M.ndim() != 2:
        raise Error(
            "newton_schulz_inv_fourth_root requires rank-2 tensor, got ndim: "
            + String(M.ndim())
        )

    var shape = M.shape()
    var m = shape[0]
    var n = shape[1]

    if m != n:
        raise Error(
            "newton_schulz_inv_fourth_root requires square matrix, got ["
            + String(m)
            + ", "
            + String(n)
            + "]"
        )

    var dtype = M.dtype()

    # Stage 1: S = M^{-1/2}. Use a looser internal tolerance so the intermediate
    # inv-sqrt does not over-constrain; the final fourth-root residual is the
    # authoritative check below.
    var S = _newton_schulz_inv_sqrt(
        M, steps=steps, eps=eps, convergence_tol=1.0
    )

    # H = M · S = M^{1/2}  (M and M^{-1/2} commute, so the product is symmetric PSD).
    var H = matmul(M, S)

    # Stage 2: R = H^{-1/2} = M^{-1/4}.
    var R = _newton_schulz_inv_sqrt(
        H, steps=steps, eps=eps, convergence_tol=1.0
    )

    # Final convergence assertion against the ORIGINAL M: ||M · R^4 − I||_F
    var I = eye(m, m, 0, dtype)
    var R2 = matmul(R, R)  # MATRIX
    var R4 = matmul(R2, R2)  # MATRIX
    var MR4 = matmul(M, R4)  # MATRIX
    var residual = subtract_simd(MR4, I)
    var residual_norm = compute_tensor_norm(residual)

    # `not (<=)` so a NaN residual (diverged iteration) raises instead of passing.
    if not (residual_norm <= convergence_tol):
        raise Error(
            "newton_schulz_inv_fourth_root failed to converge; residual_norm: "
            + String(residual_norm)
            + ", tolerance: "
            + String(convergence_tol)
        )

    return R


def shampoo_step(
    params: AnyTensor,
    gradients: AnyTensor,
    L: AnyTensor,
    R: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    beta_precond: Float64 = 0.95,
    beta_momentum: Float64 = 0.95,
    weight_decay: Float64 = 0.0,
    ns_steps: Int = 8,
    eps: Float64 = 1e-10,
    max_precond_norm: Float64 = 1e6,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single Shampoo optimization step - pure functional.

    Returns new parameters and new state buffers. Caller manages all state.

    Shampoo applies two-sided matrix preconditioning via Newton-Schulz iteration
    to compute inverse fourth roots of accumulated Gram matrices. This provides
    adaptive step sizes that improve conditioning.

    **Important: Calling convention differs from most optimizers.**
    The caller holds params directly; this function returns (params_new, L_new,
    R_new, momentum_new). See module docstring for example usage.

    Args:
        params: Model parameters [m, n] (must be rank-2 matrix).
        gradients: Gradients of loss w.r.t. params (same shape as params).
        L: Left accumulator [m, m] tracking G @ G^T.
        R: Right accumulator [n, n] tracking G^T @ G.
        momentum: Momentum buffer [m, n] (use zeros_like(params) initially).
        learning_rate: Step size for parameter updates.
        beta_precond: EMA decay for Gram matrix accumulators (default: 0.95).
        beta_momentum: EMA decay for momentum buffer (default: 0.95).
        weight_decay: L2 regularization factor (default: 0.0).
        ns_steps: Number of Newton-Schulz iterations (default: 8).
        eps: Numerical stability floor (default: 1e-10).
        max_precond_norm: Maximum Frobenius norm for Gram accumulators (default: 1e6).

    Returns:
        Tuple of (params_new, L_new, R_new, momentum_new). Note that L_new and
        R_new are UNCLAMPED accumulators; clamping only affects the inverse-root
        computation internally.

    Raises:
        Error: If shapes or dtypes are invalid, or if Newton-Schulz convergence fails.

    Note:
        The returned L_new and R_new are the raw EMA accumulators without the
        clamping applied. This allows the caller flexibility in managing accumulator
        growth. The convergence assertion in Newton-Schulz will fail if the
        accumulators become ill-conditioned.
    """
    # Step 1: Validate shapes and dtypes
    var p_shape = params.shape()
    var g_shape = gradients.shape()
    var L_shape = L.shape()
    var R_shape = R.shape()
    var _ = momentum.shape()

    if params.ndim() != 2:
        raise Error(
            "shampoo_step requires rank-2 params, got ndim: "
            + String(params.ndim())
        )

    if p_shape != g_shape:
        raise Error("params and gradients shapes must match")

    if params.dtype() != gradients.dtype():
        raise Error("params and gradients dtypes must match")

    var m = p_shape[0]
    var n = p_shape[1]

    if L_shape[0] != m or L_shape[1] != m:
        raise Error(
            "L shape mismatch: expected ["
            + String(m)
            + ", "
            + String(m)
            + "], got ["
            + String(L_shape[0])
            + ", "
            + String(L_shape[1])
            + "]"
        )

    if R_shape[0] != n or R_shape[1] != n:
        raise Error(
            "R shape mismatch: expected ["
            + String(n)
            + ", "
            + String(n)
            + "], got ["
            + String(R_shape[0])
            + ", "
            + String(R_shape[1])
            + "]"
        )

    # Step 2: Transpose gradients
    var Gt = transpose(gradients, None)

    # Step 3-4: Update L and R with EMA of Gram matrices
    # L_new = β·L + G @ G^T
    var GGt = matmul(gradients, Gt)  # MATRIX
    var beta_L = full_like(L, beta_precond)
    var beta_L_scaled = multiply_simd(beta_L, L)
    var L_new = add_simd(beta_L_scaled, GGt)

    # R_new = β·R + G^T @ G
    var GtG = matmul(Gt, gradients)  # MATRIX
    var beta_R = full_like(R, beta_precond)
    var beta_R_scaled = multiply_simd(beta_R, R)
    var R_new = add_simd(beta_R_scaled, GtG)

    # Step 5-6: Clamp for numerical stability (only affects inverse-root computation)
    var L_clamped = _clamp_precond_norm(L_new, max_norm=max_precond_norm)
    var R_clamped = _clamp_precond_norm(R_new, max_norm=max_precond_norm)

    # Step 7-8: Compute inverse fourth roots
    var L_inv_root = newton_schulz_inv_fourth_root(
        L_clamped, steps=ns_steps, eps=eps
    )
    var R_inv_root = newton_schulz_inv_fourth_root(
        R_clamped, steps=ns_steps, eps=eps
    )

    # Step 9: Precondition gradient
    # precond_grad = L^{-1/4} @ G @ R^{-1/4}
    var L_G = matmul(L_inv_root, gradients)  # MATRIX
    var precond_grad = matmul(L_G, R_inv_root)  # MATRIX

    # Step 10: Momentum update
    # momentum_new = β·momentum + precond_grad
    var beta_m = full_like(momentum, beta_momentum)
    var beta_m_scaled = multiply_simd(beta_m, momentum)
    var momentum_new = add_simd(beta_m_scaled, precond_grad)

    # Step 11: Parameter update
    # update = lr · momentum_new
    var lr_tensor = full_like(momentum_new, learning_rate)
    var update = multiply_simd(lr_tensor, momentum_new)
    var params_after = subtract_simd(params, update)

    # Step 12: Weight decay (if needed)
    var params_new: AnyTensor
    if weight_decay > 0.0:
        var wd_scale = learning_rate * weight_decay
        var wd_tensor = full_like(params_after, wd_scale)
        var wd_term = multiply_simd(wd_tensor, params_after)
        params_new = subtract_simd(params_after, wd_term)
    else:
        params_new = params_after

    # Step 13: Return new state (L_new and R_new are UNCLAMPED accumulators)
    return (params_new, L_new, R_new, momentum_new)


def shampoo_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    L: AnyTensor,
    R: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor, AnyTensor]:
    """Convenience wrapper around shampoo_step with default hyperparameters.

    Args:
        params: Model parameters [m, n].
        gradients: Gradients of loss w.r.t. params.
        L: Left accumulator [m, m].
        R: Right accumulator [n, n].
        momentum: Momentum buffer [m, n].
        learning_rate: Step size.

    Returns:
        Tuple of (params_new, L_new, R_new, momentum_new).
    """
    return shampoo_step(
        params,
        gradients,
        L,
        R,
        momentum,
        learning_rate,
        beta_precond=0.95,
        beta_momentum=0.95,
        weight_decay=0.0,
        ns_steps=8,
        eps=1e-10,
        max_precond_norm=1e6,
    )
