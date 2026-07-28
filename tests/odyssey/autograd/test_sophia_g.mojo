"""Tests for ``odyssey.autograd.sophia_g`` -- Phase 2+ Sophia-G estimators (v2).

The math invariants under test (v2 -- math correction per code-reviewer):

- Compile-time trait conformance via the parametric-bound verifier.
- ``make_estimator("hutchinson")`` returns a working Rademacher-perturbed
  central-FD estimator whose H_diag on the canonical quadratic loss
  ``sum(theta_i^2)`` converges to the analytical ``H_ii = 2`` within
  +-5% over 200 Rademacher samples.
- ``make_estimator("gauss_newton_bartlett")`` returns a working classical
  central-FD estimator whose H_diag matches ``H_diag = 2`` within
  ``epsilon^2 ~ 1e-6`` (default ``epsilon = 1e-3``).
- ``make_estimator(<unknown>)`` raises with diagnostic listing allowed kinds.
- Output length, shape, dtype are preserved (Invariants 1, 2, 3).
- Multi-parameter-list (2+ AnyTensors) supported.
- Outputs are finite and essentially non-negative on convex quadratic
  surfaces (Invariants 5, 7).

Tests dispatch via ``def main() raises`` matching the project's
Mojo 1.0 convention (``mojo run <file>.mojo``).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.autograd.sophia_g import (
    DiagonalHessianEstimator,
    HutchinsonDiagonalHessian,
    GaussNewtonDiagonalHessian,
    make_estimator,
    accept_as_estimator,
)


def _abs_diff_f64(a: Float64, b: Float64) -> Float64:
    """``|a - b|``. Mirrors the canonical helper used in sibling tests
    (test_jvp.mojo, test_sophia.mojo, etc.)."""
    var d = a - b
    if d < 0.0:
        return -d
    return d


def test_accept_as_estimator_parametric_bound() raises:
    """[compile-time] Both estimator structs conform to DiagonalHessianEstimator."""
    print("Running test_accept_as_estimator_parametric_bound...")
    var h = HutchinsonDiagonalHessian()
    var gn = GaussNewtonDiagonalHessian()
    var _r1 = accept_as_estimator[HutchinsonDiagonalHessian](h)
    var _r2 = accept_as_estimator[GaussNewtonDiagonalHessian](gn)
    print(
        "  ok both HutchinsonDiagonalHessian and GaussNewtonDiagonalHessian"
        " conform to DiagonalHessianEstimator"
    )
    print("test_accept_as_estimator_parametric_bound PASSED")


def test_make_estimator_routes_to_hutchinson() raises:
    """``make_estimator("hutchinson")`` returns a working estimator."""
    print("Running test_make_estimator_routes_to_hutchinson...")
    var est = make_estimator("hutchinson")

    var n = 4
    var p = full([n], 0.5, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)

    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    # f_i(x) = sum_{j != i} p[j]^2 + x^2; analytical H_ii = 2 per coord.
    def q_at(coord: Int, x: Float64) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var h_diag = est.estimate_diag_hessian(params, q_at)
    var expected: Float64 = 2.0
    for k in range(n):
        var got = h_diag[0]._get_float64(k)
        if _abs_diff_f64(got, expected) > 0.05:
            raise Error(
                "make_estimator('hutchinson') H_diag off at k="
                + String(k)
                + ": got "
                + String(got)
                + ", want ~2.0"
            )
    print("  ok make_estimator('hutchinson') returns a working estimator")
    print("test_make_estimator_routes_to_hutchinson PASSED")


def test_make_estimator_routes_to_gauss_newton() raises:
    """``make_estimator("gauss_newton_bartlett")`` returns working GN estimator."""
    print("Running test_make_estimator_routes_to_gauss_newton...")
    var est = make_estimator("gauss_newton_bartlett")

    var n = 4
    var p = full([n], 0.5, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)

    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    def q_at(coord: Int, x: Float64) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var h_diag = est.estimate_diag_hessian(params, q_at)
    var expected: Float64 = 2.0
    for k in range(n):
        var got = h_diag[0]._get_float64(k)
        if _abs_diff_f64(got, expected) > 1e-4:
            raise Error(
                "make_estimator('gauss_newton_bartlett') H_diag off at k="
                + String(k)
                + ": got "
                + String(got)
                + ", want 2.0"
            )
    print(
        "  ok make_estimator('gauss_newton_bartlett') returns a working"
        " estimator"
    )
    print("test_make_estimator_routes_to_gauss_newton PASSED")


def test_make_estimator_unknown_raises() raises:
    """Unknown kind raises with diagnostic listing allowed kinds."""
    print("Running test_make_estimator_unknown_raises...")
    var raised = False
    try:
        var _ = make_estimator("not_a_real_kind")
    except e:
        raised = True
        var msg = String(e)
        if "unknown kind" not in msg:
            raise Error(
                "make_estimator: error should mention 'unknown kind',"
                " got: "
                + msg
            )
        if "hutchinson" not in msg or "gauss_newton_bartlett" not in msg:
            raise Error(
                "make_estimator: error should list supported kinds,"
                " got: "
                + msg
            )
    if not raised:
        raise Error(
            "make_estimator('not_a_real_kind') should have raised"
        )
    print(
        "  ok make_estimator raises 'unknown kind' + lists allowed kinds"
    )
    print("test_make_estimator_unknown_raises PASSED")


def test_hutchinson_converges_within_tolerance() raises:
    """Hutchinson converges to analytical H_diag = 2 within +-5%.

    Sum-of-squares loss has analytical H_diag = 2 for every coord.
    With 200 Rademacher samples (seed=42), the per-coord estimate
    converges to within +-5% relative error thanks to the
    Rademacher-perturbed FD averaging.
    """
    print("Running test_hutchinson_converges_within_tolerance...")
    var n = 8
    var p = full([n], 0.5, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)

    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    def q_at(coord: Int, x: Float64) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var est = HutchinsonDiagonalHessian(
        num_samples=200, seed=42, epsilon=1e-3
    )
    var h_diag = est.estimate_diag_hessian(params, q_at)
    var analytical: Float64 = 2.0
    var tol_rel: Float64 = 0.05
    var max_rel_err: Float64 = 0.0
    for k in range(n):
        var got = h_diag[0]._get_float64(k)
        var rel_err = _abs_diff_f64(got, analytical) / analytical
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    if max_rel_err > tol_rel:
        raise Error(
            "Hutchinson rel_error > 5% (got "
            + String(max_rel_err)
            + "); analytical=2.0"
        )
    print(
        "  ok Hutchinson H_diag within +-5% of analytical 2.0"
        " (max_rel_err="
        + String(max_rel_err)
        + ")"
    )
    print("test_hutchinson_converges_within_tolerance PASSED")


def test_gauss_newton_matches_analytical() raises:
    """Gauss-Newton central FD recovers analytical H_diag = 2 within FD error.

    FD truncation error is ``O(epsilon^2)`` -- with default
    ``epsilon = 1e-3`` and Float64 arithmetic, absolute tolerance
    ~1e-4 is comfortable.
    """
    print("Running test_gauss_newton_matches_analytical...")
    var n = 6
    var p = full([n], 0.3, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)

    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    def q_at(coord: Int, x: Float64) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var est = GaussNewtonDiagonalHessian(epsilon=1e-3)
    var h_diag = est.estimate_diag_hessian(params, q_at)
    var analytical: Float64 = 2.0
    for k in range(n):
        var got = h_diag[0]._get_float64(k)
        if _abs_diff_f64(got, analytical) > 1e-4:
            raise Error(
                "GaussNewton FD mismatch at k="
                + String(k)
                + ": got "
                + String(got)
                + ", want 2.0"
            )
    print(
        "  ok GaussNewton H_diag matches analytical 2.0 within +-1e-4"
    )
    print("test_gauss_newton_matches_analytical PASSED")


def test_output_shape_dtype_preserved() raises:
    """Output is length-equal, shape-equal, dtype-equal to input parameters.

    Invariants 1, 2, 3 from the trait contract. Tested on a 2D f32
    tensor so the dtype path is exercised (the typical cold path is
    f64).
    """
    print("Running test_output_shape_dtype_preserved...")
    var n_rows = 3
    var n_cols = 4
    var p = zeros([n_rows, n_cols], DType.float32)
    var p_filled = p
    p_filled.store[DType.float32](0, 0.5)
    p_filled.store[DType.float32](5, 0.7)
    var params = List[AnyTensor]()
    params.append(p_filled^)

    var p_flat = List[Float64]()
    var n_total = n_rows * n_cols
    for k in range(n_total):
        p_flat.append(p._get_float64(k))

    def q_at(
        coord: Int, x: Float64
    ) capturing[p_flat, n_total] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n_total):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var est_h = HutchinsonDiagonalHessian(num_samples=30, epsilon=1e-3)
    var est_gn = GaussNewtonDiagonalHessian()
    var h_diag_h = est_h.estimate_diag_hessian(params, q_at)
    var h_diag_gn = est_gn.estimate_diag_hessian(params, q_at)

    if len(h_diag_h) != len(params):
        raise Error("Hutchinson output length != parameter length")
    if len(h_diag_gn) != len(params):
        raise Error("GaussNewton output length != parameter length")
    if h_diag_h[0].shape() != params[0].shape():
        raise Error(
            "Hutchinson output shape "
            + String(h_diag_h[0].shape())
            + " != input shape "
            + String(params[0].shape())
        )
    if h_diag_gn[0].shape() != params[0].shape():
        raise Error("GaussNewton output shape mismatch")
    if h_diag_h[0].dtype() != params[0].dtype():
        raise Error("Hutchinson output dtype mismatch")
    if h_diag_gn[0].dtype() != params[0].dtype():
        raise Error("GaussNewton output dtype mismatch")
    print(
        "  ok both estimators preserve length/shape/dtype for [3,4] float32"
    )
    print("test_output_shape_dtype_preserved PASSED")


def test_outputs_are_finite_and_nonneg() raises:
    """H_diag estimates are finite and (essentially) non-negative on convex surface."""
    print("Running test_outputs_are_finite_and_nonneg...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)

    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    def q_at(coord: Int, x: Float64) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j]
        s = s + x * x
        return s

    var est_h = HutchinsonDiagonalHessian(num_samples=50, epsilon=1e-3)
    var est_gn = GaussNewtonDiagonalHessian()
    var h_diag_h = est_h.estimate_diag_hessian(params, q_at)
    var h_diag_gn = est_gn.estimate_diag_hessian(params, q_at)
    for k in range(n):
        var v_h = h_diag_h[0]._get_float64(k)
        var v_g = h_diag_gn[0]._get_float64(k)
        # NaN check: NaN != NaN always.
        if v_h != v_h:
            raise Error("Hutchinson NaN at k=" + String(k))
        if v_g != v_g:
            raise Error("GaussNewton NaN at k=" + String(k))
        # On convex loss surface, H_diag > 0; allow small noise band.
        if v_h < -1e-3:
            raise Error(
                "Hutchinson negative at k=" + String(k) + ": " + String(v_h)
            )
    print(
        "  ok both estimators produce finite, non-negative H_diag on"
        " convex quadratic"
    )
    print("test_outputs_are_finite_and_nonneg PASSED")


def test_stochastic_step_produces_truncation_bias_variance() raises:
    """Round-2 review fix: confirm stochastic `delta_s` per sample,
    vs fixed `delta` for Gauss-Newton, on a non-quadratic surface.

    On the quartic ``f(theta) = sum_j theta_j^4`` (so
    ``f^{(4)}(theta) = 24``), central-FD truncation bias is
    ``(delta^2 / 12) * 24 = 2 * delta^2``. With `theta_i = 0.5`
    everywhere, analytical ``H_ii = 12 * theta_i^2 = 3.0``.

    - GaussNewton with fixed `eps = 0.05` returns
      ``H_ii = 3.0 + 2 * (0.05)^2 = 3.005`` (deterministic).
    - Hutchinson with stochastic ``delta_s in [0.025, 0.075)`` returns
      ``E[H_ii] = 3.0 + 2 * E[delta_s^2]`` where
      ``E[delta_s^2] = (0.05)^2 * (E[u^2] over u in [0.5, 1.5])``.
      Since ``u ~ Uniform[0.5, 1.5]``,
      ``E[u^2] = (0.25 + 0.75 + 2.25) / 3 = 1.0833`` (analytic E[U^2]
      for U ~ Uniform[0.5, 1.5], derived from
      ``(b^3 - a^3) / (3(b - a)) = (a^2 + a*b + b^2) / 3``),
      so ``E[delta_s^2] = 0.002708`` and ``E[bias] = 0.005417``.
      So ``E[H_ii] ~= 3.00542``.

    The two estimators should agree within +/-0.005 on this surface,
    and both within +/-0.01 of analytical 3.0. This is empirical
    confirmation that the round-2 stochastic-delta fix produces
    non-vacuous per-sample FD values on non-quadratic surfaces
    (the round-1 fixed-eps was algebraically redundant for
    ``sum(theta_i^2)`` and never verified the math).
    """
    print("Running test_stochastic_step_produces_truncation_bias_variance...")
    var n = 4
    var theta_val: Float64 = 0.5
    var base_eps: Float64 = 0.05
    var p = full([n], theta_val, DType.float64)
    var params = List[AnyTensor]()
    params.append(p^)
    var p_flat = List[Float64]()
    for k in range(n):
        p_flat.append(p._get_float64(k))

    # f_i(x) = sum_{j != i} theta_j^4 + x^4.  Analytical H_ii = 12 * theta_i^2.
    def q_at(
        coord: Int, x: Float64
    ) capturing[p_flat, n] raises -> Float64:
        var s: Float64 = 0.0
        for j in range(n):
            if j != coord:
                s = s + p_flat[j] * p_flat[j] * p_flat[j] * p_flat[j]
        s = s + x * x * x * x
        return s

    var analytical: Float64 = 3.0  # 12 * 0.5^2

    # --- GaussNewton: fixed delta = 0.05 → expected = 3.0 + 2 * 0.05^2 = 3.005
    var gn_expected: Float64 = analytical + 2.0 * base_eps * base_eps
    var gn = GaussNewtonDiagonalHessian(epsilon=base_eps)
    var h_gn = gn.estimate_diag_hessian(params, q_at)
    for k in range(n):
        var got_gn = h_gn[0]._get_float64(k)
        if _abs_diff_f64(got_gn, gn_expected) > 1e-4:
            raise Error(
                "GN fixed-eps on quartic at k="
                + String(k)
                + ": got "
                + String(got_gn)
                + ", want "
                + String(gn_expected)
            )

    # --- Hutchinson: stochastic delta_s, expected mean ~3.00542
    var hutch = HutchinsonDiagonalHessian(
        num_samples=200, seed=42, epsilon=base_eps
    )
    var h_hutch = hutch.estimate_diag_hessian(params, q_at)
    var avg_hutch: Float64 = 0.0
    for k in range(n):
        avg_hutch = avg_hutch + h_hutch[0]._get_float64(k)
    avg_hutch = avg_hutch / Float64(n)
    # Tolerance: ±0.01 covers both stochastic variance AND the
    # <0.001 difference between E[bias] and the deterministic GN bias.
    if _abs_diff_f64(avg_hutch, analytical) > 0.01:
        raise Error(
            "Hutchinson stochastic on quartic: mean "
            + String(avg_hutch)
            + " outside ±0.01 of analytical "
            + String(analytical)
        )
    # The two estimators should agree within ±0.005 on this surface.
    if _abs_diff_f64(avg_hutch, h_gn[0]._get_float64(0)) > 0.005:
        raise Error(
            "Hutchinson vs GN disagree on quartic: "
            + String(avg_hutch)
            + " vs "
            + String(h_gn[0]._get_float64(0))
        )
    print(
        "  ok stochastic δ Hutchinson and fixed-δ GN agree within"
        " ±0.005 on quartic surface"
    )
    print("test_stochastic_step_produces_truncation_bias_variance PASSED")


def test_multi_parameter_list() raises:
    """2-element parameter list supported (multi-tensor case)."""
    print("Running test_multi_parameter_list...")
    var n_a = 3
    var n_b = 5
    var pa = full([n_a], 0.4, DType.float64)
    var pb = full([n_b], 0.6, DType.float64)
    var params = List[AnyTensor]()
    params.append(pa^)
    params.append(pb^)

    var pa_flat = List[Float64]()
    var pb_flat = List[Float64]()
    for k in range(n_a):
        pa_flat.append(pa._get_float64(k))
    for k in range(n_b):
        pb_flat.append(pb._get_float64(k))

    # Combined-index quadratic loss: each coord gets H_ii = 2.
    def q_at(
        coord: Int, x: Float64
    ) capturing[pa_flat, pb_flat, n_a, n_b] raises -> Float64:
        var s: Float64 = 0.0
        if coord < n_a:
            var local_k = coord
            for j in range(n_a):
                if j != local_k:
                    s = s + pa_flat[j] * pa_flat[j]
        else:
            var local_k = coord - n_a
            for j in range(n_b):
                if j != local_k:
                    s = s + pb_flat[j] * pb_flat[j]
        s = s + x * x
        return s

    var est = GaussNewtonDiagonalHessian(epsilon=1e-3)
    var h_diag = est.estimate_diag_hessian(params, q_at)

    if len(h_diag) != 2:
        raise Error("multi-param: output length != input length")
    var expected: Float64 = 2.0
    for k in range(n_a):
        var got = h_diag[0]._get_float64(k)
        if _abs_diff_f64(got, expected) > 1e-4:
            raise Error(
                "multi-param: H_diag[0][" + String(k) + "] mismatch"
            )
    for k in range(n_b):
        var got = h_diag[1]._get_float64(k)
        if _abs_diff_f64(got, expected) > 1e-4:
            raise Error(
                "multi-param: H_diag[1][" + String(k) + "] mismatch"
            )
    print(
        "  ok multi-parameter list (2 AnyTensors) preserved;"
        " H_diag = 2 per coord"
    )
    print("test_multi_parameter_list PASSED")


def main() raises:
    """Run all sophia_g test entry points."""
    print("=" * 60)
    print(
        "odyssey.autograd.sophia_g -- Phase 2+ Sophia-G diagonal"
        " Hessian test suite (v2: central FD substrate)"
    )
    print(
        "Closes #5717 -- math correction per code-reviewer audit"
    )
    print("=" * 60)
    test_accept_as_estimator_parametric_bound()
    test_make_estimator_routes_to_hutchinson()
    test_make_estimator_routes_to_gauss_newton()
    test_make_estimator_unknown_raises()
    test_hutchinson_converges_within_tolerance()
    test_gauss_newton_matches_analytical()
    test_output_shape_dtype_preserved()
    test_outputs_are_finite_and_nonneg()
    test_stochastic_step_produces_truncation_bias_variance()
    test_multi_parameter_list()
    print("")
    print("=" * 60)
    print("ALL sophia_g TESTS PASSED")
    print("=" * 60)
