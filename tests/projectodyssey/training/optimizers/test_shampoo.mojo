"""Unit tests for Shampoo optimizer.

Tests cover:
- Eligibility checking (rejects non-2D, degenerate cases)
- State initialization (L, R identity matrices with correct shapes)
- Dtype and shape validation
- Trace helper correctness
- Clamping preserves symmetry
- Newton-Schulz inverse fourth root (identity, diagonal, non-diagonal cases)
- Convergence assertion on divergence
- Non-square parameter support (regression for #5487)
- Genuine descent on quadratic objectives
"""

from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    zeros_like,
    randn,
    full,
    full_like,
    ones_like,
)
from projectodyssey.training.optimizers.shampoo import (
    is_shampoo_eligible,
    initialize_shampoo_state,
    _trace_sum_diag,
    _clamp_precond_norm,
    newton_schulz_inv_fourth_root,
    shampoo_step,
    shampoo_step_simple,
)
from projectodyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from projectodyssey.core.matrix import matmul, transpose
from projectodyssey.tensor.tensor_creation import eye
from projectodyssey.training.optimizers.optimizer_utils import (
    compute_tensor_norm,
)
from std.math import sqrt as scalar_sqrt, abs as math_abs, isnan


def _require(cond: Bool, msg: String) raises:
    """Raise-based check that works regardless of debug_assert build flags.

    The stdlib `assert` lowers to `debug_assert`, which is compiled out unless
    assertions are explicitly enabled. Numerical-correctness tests must NOT rely on
    it, or they silently pass (e.g. a NaN loss would slip through `assert nan < x`).
    """
    if not cond:
        raise Error("CHECK FAILED: " + msg)


def _require_close(
    value: Float64, target: Float64, tol: Float64, msg: String
) raises:
    """Raise-based approximate-equality check that also rejects NaN."""
    if isnan(value):
        raise Error("CHECK FAILED (NaN): " + msg + "; got NaN")
    if not (math_abs(value - target) <= tol):
        raise Error(
            "CHECK FAILED: "
            + msg
            + "; expected "
            + String(target)
            + " ± "
            + String(tol)
            + ", got "
            + String(value)
        )


def test_is_shampoo_eligible() raises:
    """Test that is_shampoo_eligible correctly identifies valid parameters."""
    print("Running test_is_shampoo_eligible...")

    # Test 2D with both dims >= 2: should be True
    var params_ok = zeros([8, 16], DType.float32)
    assert is_shampoo_eligible(params_ok), "Should accept [8,16]"

    # Test 1D: should be False
    var params_1d = zeros([10], DType.float32)
    assert not is_shampoo_eligible(params_1d), "Should reject [10]"

    # Test 3D: should be False
    var params_3d = zeros([2, 3, 4], DType.float32)
    assert not is_shampoo_eligible(params_3d), "Should reject [2,3,4]"

    # Test degenerate [1,5]: should be False (one dim < 2)
    var params_degen1 = zeros([1, 5], DType.float32)
    assert not is_shampoo_eligible(params_degen1), "Should reject [1,5]"

    # Test degenerate [5,1]: should be False (one dim < 2)
    var params_degen2 = zeros([5, 1], DType.float32)
    assert not is_shampoo_eligible(params_degen2), "Should reject [5,1]"

    # Test [2,3]: should be True (both >= 2, non-square)
    var params_nonsquare = zeros([2, 3], DType.float32)
    assert is_shampoo_eligible(params_nonsquare), "Should accept [2,3]"

    print("test_is_shampoo_eligible PASSED")


def test_initialize_shampoo_state_shapes() raises:
    """Test that initialize_shampoo_state creates correctly-shaped buffers."""
    print("Running test_initialize_shampoo_state_shapes...")

    var params = zeros([4, 7], DType.float32)
    var (L, R, momentum) = initialize_shampoo_state(params)

    # Check shapes
    var L_shape = L.shape()
    var R_shape = R.shape()
    var m_shape = momentum.shape()

    assert L_shape[0] == 4 and L_shape[1] == 4, "L should be [4,4]"
    assert R_shape[0] == 7 and R_shape[1] == 7, "R should be [7,7]"
    assert m_shape[0] == 4 and m_shape[1] == 7, "momentum should be [4,7]"

    # Check that L and R are initialized to identity
    # L[0,0] should be 1.0
    var L_00 = L._get_float64(0)
    assert math_abs(L_00 - 1.0) < 1e-5, "L[0,0] should be 1.0"

    # L[0,1] should be 0.0
    var L_01 = L._get_float64(1)
    assert math_abs(L_01 - 0.0) < 1e-5, "L[0,1] should be 0.0"

    # R[3,3] should be 1.0
    var R_33 = R._get_float64(7 * 3 + 3)
    assert math_abs(R_33 - 1.0) < 1e-5, "R[3,3] should be 1.0"

    print("test_initialize_shampoo_state_shapes PASSED")


def test_reject_non_2d_params() raises:
    """Test that shampoo_step rejects non-2D parameters."""
    print("Running test_reject_non_2d_params...")

    # Test 1D rejection
    var params_1d = zeros([10], DType.float32)
    var grad_1d = zeros([10], DType.float32)
    var (L, R, m) = initialize_shampoo_state(zeros([4, 4], DType.float32))

    try:
        var _ = shampoo_step(params_1d, grad_1d, L, R, m, learning_rate=0.01)
        raise Error("Should have rejected 1D params")
    except e:
        print("  ✓ Correctly rejected 1D: " + String(e))

    # Test 3D rejection
    var params_3d = zeros([2, 3, 4], DType.float32)
    var grad_3d = zeros([2, 3, 4], DType.float32)

    try:
        var _ = shampoo_step(params_3d, grad_3d, L, R, m, learning_rate=0.01)
        raise Error("Should have rejected 3D params")
    except e:
        print("  ✓ Correctly rejected 3D: " + String(e))

    print("test_reject_non_2d_params PASSED")


def test_reject_dtype_mismatch() raises:
    """Test that shampoo_step rejects dtype mismatches."""
    print("Running test_reject_dtype_mismatch...")

    var params = zeros([4, 4], DType.float32)
    var grad = zeros([4, 4], DType.float16)  # Mismatch!
    var (L, R, m) = initialize_shampoo_state(params)

    try:
        var _ = shampoo_step(params, grad, L, R, m, learning_rate=0.01)
        raise Error("Should have rejected dtype mismatch")
    except e:
        print("  ✓ Correctly rejected dtype mismatch: " + String(e))

    print("test_reject_dtype_mismatch PASSED")


def test_reject_wrong_L_R_shapes() raises:
    """Test that shampoo_step rejects incorrect L and R shapes."""
    print("Running test_reject_wrong_L_R_shapes...")

    var params = zeros([4, 5], DType.float32)
    var grad = zeros([4, 5], DType.float32)
    var (L_ok, R_ok, m) = initialize_shampoo_state(params)

    # Test wrong L shape
    var L_wrong = zeros([5, 5], DType.float32)  # Should be [4,4]
    try:
        var _ = shampoo_step(params, grad, L_wrong, R_ok, m, learning_rate=0.01)
        raise Error("Should have rejected wrong L shape")
    except e:
        print("  ✓ Correctly rejected wrong L: " + String(e))

    # Test wrong R shape
    var R_wrong = zeros([4, 4], DType.float32)  # Should be [5,5]
    try:
        var _ = shampoo_step(params, grad, L_ok, R_wrong, m, learning_rate=0.01)
        raise Error("Should have rejected wrong R shape")
    except e:
        print("  ✓ Correctly rejected wrong R: " + String(e))

    print("test_reject_wrong_L_R_shapes PASSED")


def test_trace_sum_diag() raises:
    """Test that _trace_sum_diag correctly sums diagonal entries."""
    print("Running test_trace_sum_diag...")

    # Test identity matrix: trace should be 4.0
    var I = eye(4, 4, 0, DType.float32)
    var trace_I = _trace_sum_diag(I)
    assert math_abs(trace_I - 4.0) < 1e-5, "trace(I_4) should be 4.0"

    # Test constant diagonal matrix: diag(2.0) has trace 8.0
    var M = multiply_simd(
        full([4, 4], 2.0, DType.float32), eye(4, 4, 0, DType.float32)
    )
    var trace_M = _trace_sum_diag(M)
    assert math_abs(trace_M - 8.0) < 1e-5, "trace(diag(2)) should be 8.0"

    print("test_trace_sum_diag PASSED")


def test_clamp_precond_norm_preserves_symmetric() raises:
    """Test that _clamp_precond_norm preserves symmetry of PSD matrices."""
    print("Running test_clamp_precond_norm_preserves_symmetric...")

    # Build a symmetric PSD matrix: M = A @ A^T
    var A = randn([3, 3], DType.float32, seed=7)
    var M = matmul(transpose(A, None), A)

    # Clamp it
    var M_clamped = _clamp_precond_norm(M, max_norm=1e6)

    # Check symmetry: ||M_clamped - M_clamped^T||_F < 1e-5
    var M_T = transpose(M_clamped, None)
    var diff = subtract_simd(M_clamped, M_T)
    var diff_norm = compute_tensor_norm(diff)

    assert (
        diff_norm < 1e-5
    ), "Clamp should preserve symmetry; got norm: " + String(diff_norm)

    print("test_clamp_precond_norm_preserves_symmetric PASSED")


def test_inv_fourth_root_identity() raises:
    """Test Newton-Schulz inverse fourth root on identity matrix."""
    print("Running test_inv_fourth_root_identity...")

    var M = eye(5, 5, 0, DType.float32)
    var Y = newton_schulz_inv_fourth_root(M, steps=8, convergence_tol=1e-2)

    # Y should be close to identity
    var diff = subtract_simd(Y, M)
    var diff_norm = compute_tensor_norm(diff)

    _require(
        diff_norm < 1e-3,
        "Y^4 = I should give Y ≈ I; got diff_norm: " + String(diff_norm),
    )

    print("test_inv_fourth_root_identity PASSED")


def test_inv_fourth_root_diagonal() raises:
    """Test Newton-Schulz inverse fourth root on diagonal matrix."""
    print("Running test_inv_fourth_root_diagonal...")

    # Diagonal matrix with entries 16.0: (16.0)^(1/4) = 2.0
    var diag_val = full([3, 3], 16.0, DType.float32)
    var I = eye(3, 3, 0, DType.float32)
    var M = multiply_simd(diag_val, I)

    var Y = newton_schulz_inv_fourth_root(M, steps=8, convergence_tol=1e-2)

    # Check diagonal entries: should be close to 0.5 (1 / 2.0)
    for i in range(3):
        var Y_ii = Y._get_float64(i * 3 + i)
        _require_close(
            Y_ii, 0.5, 5e-3, "Y[" + String(i) + "," + String(i) + "] (16^-1/4)"
        )

    # Check off-diagonal entries: should be close to 0
    for i in range(3):
        for j in range(3):
            if i != j:
                var Y_ij = Y._get_float64(i * 3 + j)
                _require_close(
                    Y_ij, 0.0, 5e-3, "Y[" + String(i) + "," + String(j) + "]"
                )

    print("test_inv_fourth_root_diagonal PASSED")


def test_inv_fourth_root_distinct_eigenvalues() raises:
    """Discriminating test: distinct diagonal entries so M_norm != I.

    With M = diag(1, 16, 81), the trace-normalized matrix is NOT the identity, so
    this case distinguishes the inverse FOURTH root from the inverse SQUARE root:
      - inverse fourth root: diag(1^{-1/4}, 16^{-1/4}, 81^{-1/4}) = diag(1, 0.5, 1/3)
      - inverse square root: diag(1^{-1/2}, 16^{-1/2}, 81^{-1/2}) = diag(1, 0.25, 1/9)
    The expected 0.5 / 0.333 values only hold for the genuine fourth root.
    """
    print("Running test_inv_fourth_root_distinct_eigenvalues...")

    var M = zeros([3, 3], DType.float32)
    M[0] = 1.0  # M[0,0]
    M[3 + 1] = 16.0  # M[1,1]
    M[6 + 2] = 81.0  # M[2,2]

    var Y = newton_schulz_inv_fourth_root(M, steps=12, convergence_tol=5e-2)

    var expected = [1.0, 0.5, 1.0 / 3.0]
    for i in range(3):
        var Y_ii = Y._get_float64(i * 3 + i)
        _require_close(
            Y_ii,
            expected[i],
            1e-2,
            "Y["
            + String(i)
            + ","
            + String(i)
            + "] inverse FOURTH root (not -1/2)",
        )

    print("test_inv_fourth_root_distinct_eigenvalues PASSED")


def test_inv_fourth_root_non_diagonal() raises:
    """Test Newton-Schulz inverse fourth root on non-diagonal PSD matrix.

    This is the critical matmul-vs-elementwise tripwire test.
    """
    print("Running test_inv_fourth_root_non_diagonal...")

    # Build non-diagonal PSD: M = A^T @ A where A is not diagonal
    var A = add_simd(
        full([3, 3], 0.5, DType.float32), eye(3, 3, 0, DType.float32)
    )
    var M = matmul(transpose(A, None), A)

    var Y = newton_schulz_inv_fourth_root(M, steps=12, convergence_tol=5e-2)

    # Compute Y^4 = (Y^2)^2
    var Y2 = matmul(Y, Y)
    var Y4 = matmul(Y2, Y2)

    # Check: M @ Y^4 ≈ I
    var MY4 = matmul(M, Y4)
    var I = eye(3, 3, 0, DType.float32)
    var residual = subtract_simd(MY4, I)
    var residual_norm = compute_tensor_norm(residual)

    _require(
        residual_norm < 1e-1,
        "M @ Y^4 should be close to I; got residual norm: "
        + String(residual_norm),
    )

    print("test_inv_fourth_root_non_diagonal PASSED")


def test_inv_fourth_root_divergence_raises() raises:
    """Test that Newton-Schulz raises when convergence fails."""
    print("Running test_inv_fourth_root_divergence_raises...")

    # Feed an ill-conditioned matrix (high values can cause divergence with few steps)
    var M = full([3, 3], 1e15, DType.float32)

    try:
        var _ = newton_schulz_inv_fourth_root(M, steps=1, convergence_tol=1e-5)
        raise Error("Should have raised on divergence")
    except e:
        if "failed to converge" in String(e):
            print("  ✓ Correctly raised convergence error: " + String(e))
        else:
            raise Error("Unexpected error: " + String(e))

    print("test_inv_fourth_root_divergence_raises PASSED")


def test_non_square_params_shape() raises:
    """Regression test for #5487: non-square params [2,3].

    This case raised "Shapes are not broadcast-compatible" in the broken version.
    """
    print("Running test_non_square_params_shape...")

    var params = randn([2, 3], DType.float32, seed=42)
    var grad = randn([2, 3], DType.float32, seed=43)
    var (L, R, m) = initialize_shampoo_state(params)

    # This should not raise
    var (p_new, L_new, R_new, m_new) = shampoo_step(
        params, grad, L, R, m, learning_rate=0.01
    )

    # Verify returned shapes
    var p_shape = p_new.shape()
    var L_shape = L_new.shape()
    var R_shape = R_new.shape()
    var m_shape = m_new.shape()

    assert p_shape[0] == 2 and p_shape[1] == 3, "params shape mismatch"
    assert L_shape[0] == 2 and L_shape[1] == 2, "L shape mismatch"
    assert R_shape[0] == 3 and R_shape[1] == 3, "R shape mismatch"
    assert m_shape[0] == 2 and m_shape[1] == 3, "momentum shape mismatch"

    print("test_non_square_params_shape PASSED")


def test_descent_on_quadratic() raises:
    """Test genuine descent on quadratic objective."""
    print("Running test_descent_on_quadratic...")

    var W = randn([4, 4], DType.float32, seed=42)
    var (L, R, momentum) = initialize_shampoo_state(W)

    # Gradient of sum(W*W) is 2*W
    var loss_0 = compute_tensor_norm(W) * compute_tensor_norm(W)
    print("  Initial loss: " + String(loss_0))

    var W_t = W
    var L_t = L
    var R_t = R
    var m_t = momentum
    var loss_25 = 0.0
    var loss_50 = 0.0

    for step in range(50):
        # Gradient: 2*W
        var two = full_like(W_t, 2.0)
        var grad = multiply_simd(two, W_t)

        # Step
        var (W_new, L_new, R_new, m_new) = shampoo_step(
            W_t, grad, L_t, R_t, m_t, learning_rate=0.01
        )

        W_t = W_new
        L_t = L_new
        R_t = R_new
        m_t = m_new

        if step == 24:
            loss_25 = compute_tensor_norm(W_t) * compute_tensor_norm(W_t)
            print("  Loss at step 25: " + String(loss_25))

        if step == 49:
            loss_50 = compute_tensor_norm(W_t) * compute_tensor_norm(W_t)
            print("  Loss at step 50: " + String(loss_50))

    _require(not isnan(loss_50), "loss diverged to NaN")
    _require(loss_50 < loss_0, "Should have descended below initial loss")
    _require(
        loss_50 <= 0.5 * loss_0,
        "Should have descended to ≤50% of initial loss; got "
        + String(loss_50)
        + " vs "
        + String(loss_0),
    )
    _require(loss_50 < loss_25, "Should continue descending")

    print("test_descent_on_quadratic PASSED")


def test_descent_on_non_square() raises:
    """Test descent on non-square parameters."""
    print("Running test_descent_on_non_square...")

    var W = randn([2, 3], DType.float32, seed=42)
    var (L, R, momentum) = initialize_shampoo_state(W)

    # Gradient: 2*W
    var loss_0 = compute_tensor_norm(W) * compute_tensor_norm(W)
    print("  Initial loss: " + String(loss_0))

    var W_t = W
    var L_t = L
    var R_t = R
    var m_t = momentum
    var loss_50 = 0.0

    for step in range(50):
        var two = full_like(W_t, 2.0)
        var grad = multiply_simd(two, W_t)

        var (W_new, L_new, R_new, m_new) = shampoo_step(
            W_t, grad, L_t, R_t, m_t, learning_rate=0.01
        )

        W_t = W_new
        L_t = L_new
        R_t = R_new
        m_t = m_new

        if step == 49:
            loss_50 = compute_tensor_norm(W_t) * compute_tensor_norm(W_t)
            print("  Loss at step 50: " + String(loss_50))

    _require(not isnan(loss_50), "loss diverged to NaN")
    _require(loss_50 < loss_0, "Should have descended below initial loss")

    print("test_descent_on_non_square PASSED")


def test_shampoo_step_simple() raises:
    """Smoke test for the convenience wrapper."""
    print("Running test_shampoo_step_simple...")

    var params = zeros([8, 16], DType.float32)
    var grad = zeros([8, 16], DType.float32)
    var (L, R, m) = initialize_shampoo_state(params)

    # Should not raise
    var (p_new, _, _, _) = shampoo_step_simple(
        params, grad, L, R, m, learning_rate=0.01
    )

    var p_shape = p_new.shape()
    _require(p_shape[0] == 8 and p_shape[1] == 16, "Shape mismatch")

    print("test_shampoo_step_simple PASSED")


def main() raises:
    """Run all tests."""
    test_is_shampoo_eligible()
    test_initialize_shampoo_state_shapes()
    test_reject_non_2d_params()
    test_reject_dtype_mismatch()
    test_reject_wrong_L_R_shapes()
    test_trace_sum_diag()
    test_clamp_precond_norm_preserves_symmetric()
    test_inv_fourth_root_identity()
    test_inv_fourth_root_diagonal()
    test_inv_fourth_root_distinct_eigenvalues()
    test_inv_fourth_root_non_diagonal()
    test_inv_fourth_root_divergence_raises()
    test_non_square_params_shape()
    test_descent_on_quadratic()
    test_descent_on_non_square()
    test_shampoo_step_simple()
    print("")
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
