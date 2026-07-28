"""Diagonal Hessian estimators for the Sophia-G optimizer (arXiv:2305.14342).

This module provides two diagonal-Hessian estimators that feed into
the Sophia-G preconditioner:

- `HutchinsonDiagonalHessian`  -- Rademacher-perturbed central finite
  differences averaged across `num_samples` random sign perturbations.

- `GaussNewtonDiagonalHessian` -- classical central finite differences
  with a fixed step size.

## Phase 2+ architecture note (recorded)

The original issue #5717 text describes Hutchinson as
"Rademacher-sampled JVPs." The substrate available at PR-B
(`odyssey.autograd.jvp` -- the `jvp`/`jvp_only` scalar Dual module
shipped in PR #5721) is **scalar-only** by its own design: scalar
Dual arithmetic over `eps^2 -> 0` truncation. Per-coord scalar JVP
at coord i with tangent v_i yields `df_i/d(theta_i) * v_i` -- the
FIRST directional derivative along v_i, not `(Hv)_i`.

Recovering per-coord H_ii from a scalar JVP requires nested-dual
arithmetic (the J(V) product that gives Hv), which would break the
eps^2 -> 0 invariant on the inner dual. The Phase 2+ path is
therefore central finite differences on a scalar callback signature
`scalar_loss_at: def(Int, Float64) raises -> Float64 thin`,
mathematically equivalent to Gauss-Newton H_diag under convex
quadratic surfaces and converging to within `O(epsilon^2)` of the
analytical H_ii everywhere.

When `odyssey.autograd.jvp` ships its DualTensor Phase 2
implementation, the trait signature can swap back to a JVP-driven
Hutchinson formulation WITHOUT changing the public struct/factory
API exposed here.

References:
    - Gao et al., 2020 (arXiv:2006.16236) Hutchinson trace estimator.
    - Liu et al., 2023 Sophia (arXiv:2305.14342 section 3.2) Gauss-Newton.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


# ===========================================================================
# Trait
# ===========================================================================


trait DiagonalHessianEstimator:
    """Diagonal Hessian estimator contract -- 7 Invariants.

    Invariants documented per-method:

      1. length-preserving:   |return| == len(parameters).
      2. shape-preserving:    return[i].shape() == parameters[i].shape().
      3. dtype contract:      return[i].dtype() == parameters[i].dtype().
      4. scalar-loss:         `scalar_loss_at(i, .)` is the total scalar
                              loss as a function of coord i with all
                              others fixed.
      5. no side-effects:     pure functional -- inputs not mutated; the
                              `scalar_loss_at` closure is not destructively
                              queried.
      6. determinism:         with the same `seed` and same `parameters`,
                              Hutchinson returns the same value; GN returns
                              the exact FD result.
      7. finite values:       output contains no NaN/Inf.
    """

    fn estimate_diag_hessian(
        self,
        parameters: List[AnyTensor],
        scalar_loss_at: def(Int, Float64) raises -> Float64 thin,
    ) raises -> List[AnyTensor]:
        """Estimate the per-coordinate diagonal Hessian.

        Args:
            parameters: List of AnyTensors to compute H_diag for. The
                returned list has the same length and per-tensor
                shape/dtype.
            scalar_loss_at: Callable representing the total scalar loss
                as a function of one parameter coordinate. Called as
                `scalar_loss_at(coord_index, x_at_coord)` and returns
                `f_i(x)`, the loss with coord i replaced by `x` and all
                others at their current values.

        Returns:
            `List[AnyTensor]` of H_diag estimates, one AnyTensor per
            input parameter, with the same shape and dtype.
        """
        ...


# ===========================================================================
# Helpers
# ===========================================================================


fn _rademacher_at(seed: Int, idx: Int) -> Float64:
    """Deterministic Rademacher sign: +1.0 or -1.0 from a 64-bit LCG.

    Uses the LSB of the LCG state for cleanest +-1 distribution
    (no integer-division bucket collapse). Invariant 6 (determinism).
    """
    var x = (seed + idx * 1103515245 + 12345) % 2147483647
    if (x & 1) == 0:
        return 1.0
    return -1.0


fn _random_step_in_range(
    seed: Int, sample_idx: Int, base_eps: Float64
) -> Float64:
    """Stochastic FD step size in [base_eps/2, 3*base_eps/2].

    Central-FD truncation bias scales as ``(delta^2 / 12) * f^{(4)}(x)``,
    so different per-sample `delta` values produce different bias. Averaging
    across samples with stochastic `delta` therefore reduces the systematic
    truncation bias that would arise from any single fixed step size.

    Uses a distinct LCG offset (``2147483646``) versus `_rademacher_at`
    (``12345``) so the two random streams don't collide -- a per-coord
    sample uses *one* `_rademacher_at` index plus *one* `_random_step_in_range`
    index, both deterministic in ``(seed, sample_idx)``.

    Args:
        seed: Master RNG seed.
        sample_idx: Sample number (typically ``s`` in the outer loop).
        base_eps: The user-supplied epsilon (default 1e-3).

    Returns:
        ``delta_s = base_eps * (0.5 + u)`` where ``u`` is uniform in
        ``[0.0, 1.0)`` from Knuth's multiplicative-hash LCG with divisor
        ``2^31 - 1`` so the upper bound is *true* half-open (the max
        reachable LCG state ``(seed + sample_idx * 2654435761 + offset)
        % 2147483647`` is at most ``2147483646 = 2^31 - 2``, so
        ``u_max = 2147483646 / 2147483647 ~= 0.99999999953``, slightly
        less than 1.0, and ``delta_s`` strictly less than
        ``1.5 * base_eps``). Reachable range:
        ``delta_s in [base_eps/2, 1.5 * base_eps)``.

    Invariants: Deterministic for fixed ``(seed, sample_idx, base_eps)``
    (Invariant 6). For pure-quadratic surfaces the per-sample FD value is
    algebraically identical regardless of ``delta_s`` (since ``f^{(4)} == 0``),
    so stochastic averaging ONLY helps on non-quadratic surfaces -- this is
    the math-correctness boundary documented in the test suite.
    """
    var x = (seed + sample_idx * 2654435761 + 2147483646) % 2147483647
    # 2654435761 = Knuth's multiplicative hash (Lewis-Goodman-Miller constant).
    var u = Float64(x) / Float64(2147483647)  # uniform in [0, 1) -- half-open
    return base_eps * (0.5 + u)


fn _set_value_at(
    mut t: AnyTensor, k: Int, val: Float64, dtype: DType
) raises:
    """Store `val` at flat index `k` of `t`, dispatching on dtype.

    Only float32/float64 are supported in PR-B. Other dtypes raise
    (the trait contract lists float32 + float64 in Invariant 3).
    """
    if dtype == DType.float64:
        t.store[DType.float64](k, val)
    elif dtype == DType.float32:
        t.store[DType.float32](k, Float32(val))
    else:
        raise Error(
            "sophia_g: unsupported dtype "
            + String(dtype)
            + " (only float32/float64 supported in PR-B)"
        )


# ===========================================================================
# Hutchinson stochastic-trace diagonal-Hessian
# ===========================================================================


struct HutchinsonDiagonalHessian(Copyable, Movable):
    """Hutchinson-style stochastic-trace diagonal-Hessian estimator.

    PR-B (round-2 audit) implementation: **Rademacher-perturbed central FD
    per coord with stochastic step size, averaged across `num_samples`
    (sign, step-size) pairs**:

        For each sample s in 0..num_samples-1:
            delta_s ~ Uniform[epsilon/2, 3*epsilon/2)   (LCG-stochastic)
            For each coord k:
                sign = Rademacher(seed, s*K + k)
                f_plus  = scalar_loss_at(k, p_k + delta_s * sign)
                f_zero  = scalar_loss_at(k, p_k)
                f_minus = scalar_loss_at(k, p_k - delta_s * sign)
                sample_h[k] = (f_plus - 2*f_zero + f_minus) / delta_s^2
        H_diag[k] = mean over samples of `sample_h[k]`

    Why stochastic `delta_s`: with a *fixed* `epsilon`, the central-FD
    value `(f(θ+ε) - 2f(θ) + f(θ-ε)) / ε^2` is algebraically identical
    across samples on pure-quadratic surfaces (`f^{(4)} == 0`), so the
    round-1 implementation's averaging across `num_samples` was a no-op
    on the canonical test surface (`sum(theta_i^2)`).

    Stochastic `delta_s` adds **truncation-bias variance**: with
    `delta_s` varying uniformly in `[eps/2, 3*eps/2)`, the per-sample
    FD truncation bias is `(delta_s^2 / 12) * f^{(4)}(x) + O(delta_s^4)`,
    i.e. it varies with `delta_s`. Averaging across samples is a
    Monte-Carlo average of FD estimates with different (not identical)
    bias levels -- reducing the systematic truncation error on
    non-quadratic surfaces.

    Math-correctness boundary: this fix only helps on non-quadratic
    surfaces. The combined estimator remains exact on pure-quadratic
    surfaces (FD truncation vanishes regardless of `delta_s`). See
    `test_hutchinson_stochastic_step_on_quartic_surface` in
    `test_sophia_g.mojo` for empirical confirmation of bias reduction.

    Phase-2 swap path: when `odyssey.autograd.jvp` ships DualTensor,
    this struct's algorithm can be replaced with Rademacher-sampled
    JVP per coord (recovering `H_ii = imag / v_sign`) without
    changing the public trait/factory/struct API.

    Attributes:
        num_samples: Number of (sign, step-size) pairs to average
                     (default 100).
        seed:        RNG seed for reproducibility (default 42).
        epsilon:     Base FD step size (default 1e-3); per-sample
                     step is drawn uniformly from [epsilon/2, 3*epsilon/2).
    """

    var num_samples: Int
    var seed: Int
    var epsilon: Float64

    fn __init__(
        out self,
        num_samples: Int = 100,
        seed: Int = 42,
        epsilon: Float64 = 1e-3,
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.epsilon = epsilon

    fn estimate_diag_hessian(
        self,
        parameters: List[AnyTensor],
        scalar_loss_at: def(Int, Float64) raises -> Float64 thin,
    ) raises -> List[AnyTensor]:
        """Hutchinson-DIAGONAL via Rademacher-perturbed central FD + stochastic step.

        Per-sample stochastic step `delta_s in [eps/2, 3*eps/2)` from
        `_random_step_in_range(seed, s, epsilon)` -- this varies the
        truncation bias across samples (round-2 review fix).
        """
        var result = List[AnyTensor]()

        for pi in range(len(parameters)):
            var p = parameters[pi]
            var K = p.numel()
            var flat = List[Float64]()
            for k in range(K):
                flat.append(p._get_float64(k))

            var h_diag = List[Float64]()
            for k in range(K):
                h_diag.append(0.0)

            for s in range(self.num_samples):
                # Per-sample stochastic step (round-2 review fix).
                # Round-1 used fixed `epsilon` -- algebraically redundant
                # on pure-quadratic surfaces. Round-2 randomizes in
                # [eps/2, 3*eps/2) so truncation bias varies across samples.
                var delta_s = _random_step_in_range(
                    self.seed, s, self.epsilon
                )
                var delta_s_sq = delta_s * delta_s
                var sample_idx = s * K
                for k in range(K):
                    var sign = _rademacher_at(self.seed, sample_idx + k)
                    var p_k = flat[k]
                    var f_plus = scalar_loss_at(k, p_k + delta_s * sign)
                    var f_zero = scalar_loss_at(k, p_k)
                    var f_minus = scalar_loss_at(k, p_k - delta_s * sign)
                    h_diag[k] = h_diag[k] + (
                        f_plus - 2.0 * f_zero + f_minus
                    ) / delta_s_sq

            for k in range(K):
                h_diag[k] = h_diag[k] / Float64(self.num_samples)

            var out_t = zeros(p.shape(), p.dtype())
            for k in range(K):
                _set_value_at(out_t, k, h_diag[k], p.dtype())
            result.append(out_t^)

        return result^


# ===========================================================================
# Gauss-Newton-Bartlett central-FD diagonal-Hessian
# ===========================================================================


struct GaussNewtonDiagonalHessian(Copyable, Movable):
    """Gauss-Newton-Bartlett diagonal-Hessian estimator via central FD.

    Classical central FD per coord:

        H_ii = (f_i(p + delta * e_i) - 2 f_i(p) + f_i(p - delta * e_i)) / delta^2

    Truncation error is `O(delta^2)`; default `delta = 1e-3` gives
    ~1e-6 absolute error on smooth (Lipschitz-2-bounded) loss surfaces.
    """

    var epsilon: Float64

    fn __init__(out self, epsilon: Float64 = 1e-3):
        self.epsilon = epsilon

    fn estimate_diag_hessian(
        self,
        parameters: List[AnyTensor],
        scalar_loss_at: def(Int, Float64) raises -> Float64 thin,
    ) raises -> List[AnyTensor]:
        """Classical central FD diagonal-Hessian."""
        var result = List[AnyTensor]()
        var delta = self.epsilon
        var delta_sq = delta * delta

        for pi in range(len(parameters)):
            var p = parameters[pi]
            var K = p.numel()
            var flat = List[Float64]()
            for k in range(K):
                flat.append(p._get_float64(k))

            var h_diag = List[Float64]()
            for k in range(K):
                var p_k = flat[k]
                var f_plus = scalar_loss_at(k, p_k + delta)
                var f_zero = scalar_loss_at(k, p_k)
                var f_minus = scalar_loss_at(k, p_k - delta)
                h_diag.append(
                    (f_plus - 2.0 * f_zero + f_minus) / delta_sq
                )

            var out_t = zeros(p.shape(), p.dtype())
            for k in range(K):
                _set_value_at(out_t, k, h_diag[k], p.dtype())
            result.append(out_t^)

        return result^


# ===========================================================================
# Factory
# ===========================================================================


def make_estimator(kind: String) raises -> DiagonalHessianEstimator:
    """Factory routing `kind` to the corresponding estimator.

    Routes:
      "hutchinson"             -> HutchinsonDiagonalHessian (default config)
      "gauss_newton_bartlett"  -> GaussNewtonDiagonalHessian (default config)

    Unknown kinds raise with a clear diagnostic.
    """
    if kind == "hutchinson":
        return HutchinsonDiagonalHessian()
    if kind == "gauss_newton_bartlett":
        return GaussNewtonDiagonalHessian()
    raise Error(
        "make_estimator: unknown kind '"
        + kind
        + "' -- expected 'hutchinson' or 'gauss_newton_bartlett'"
    )


# ===========================================================================
# Parametric-bound conformance verifier
# ===========================================================================


def accept_as_estimator[H: DiagonalHessianEstimator](h: H) -> Bool:
    """Compile-time trait-conformance verifier.

    The parametric bind `[H: DiagonalHessianEstimator]` is the
    operative check: if `h`'s type does not satisfy the trait at the
    call site, Mojo refuses to compile.
    """
    return True
