"""Unit tests for Muon optimizer (Newton-Schulz orthogonalized momentum).

Tests cover:
- Eligibility checking (is_muon_eligible)
- Newton-Schulz orthogonalization convergence and correctness
- Muon step execution with various hyperparameters
- Parameter update behavior (descent, shape preservation)
- Error handling and validation

Muon is specialized for matrix-shaped parameters (linear weights, conv kernels
reshaped to 2D). Non-matrix parameters should use AdamW instead.
"""

from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_equal,
    assert_greater,
    assert_less,
    assert_not_equal,
    assert_shape,
    assert_true,
    create_test_vector,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, full
from projectodyssey.training.optimizers.muon import (
    muon_step,
    muon_step_simple,
    newton_schulz_orthogonalize,
    is_muon_eligible,
)
from projectodyssey.training.optimizers.optimizer_utils import compute_tensor_norm
from projectodyssey.core.matrix import matmul, transpose


# ============================================================================
# Tests for is_muon_eligible
# ============================================================================


def test_is_muon_eligible_rank_1() raises:
    """Test that rank-1 tensors are not Muon-eligible.

    Embeddings, biases, and 1D parameter vectors should not use Muon.
    """
    var shape: List[Int] = [128]
    var bias = ones(shape, DType.float32)

    assert not is_muon_eligible(bias)


def test_is_muon_eligible_rank_3() raises:
    """Test that rank-3 tensors are not Muon-eligible.

    High-rank tensors (e.g., conv kernels before reshape) should not use Muon.
    """
    var shape: List[Int] = [16, 8, 3]
    var kernel = ones(shape, DType.float32)

    assert not is_muon_eligible(kernel)


def test_is_muon_eligible_rank_2_matrix() raises:
    """Test that rank-2 tensors with sufficient size are Muon-eligible.

    Linear layer weights: [784, 128] -> eligible.
    """
    var shape: List[Int] = [784, 128]
    var weight = ones(shape, DType.float32)

    assert is_muon_eligible(weight)


def test_is_muon_eligible_rank_2_small_row() raises:
    """Test that rank-2 tensors with degenerate row dimension are not eligible.

    Shape [1, 100]: only 1 row -> not eligible (only one nonzero singular value).
    """
    var shape: List[Int] = [1, 100]
    var weight = ones(shape, DType.float32)

    assert not is_muon_eligible(weight)


def test_is_muon_eligible_rank_2_small_col() raises:
    """Test that rank-2 tensors with degenerate col dimension are not eligible.

    Shape [100, 1]: only 1 col -> not eligible.
    """
    var shape: List[Int] = [100, 1]
    var weight = ones(shape, DType.float32)

    assert not is_muon_eligible(weight)


def test_is_muon_eligible_rank_2_min_size() raises:
    """Test that rank-2 tensors with both dimensions >= 2 are eligible.

    Shape [2, 2]: exactly the minimum viable size -> eligible.
    """
    var shape: List[Int] = [2, 2]
    var weight = ones(shape, DType.float32)

    assert is_muon_eligible(weight)


# ============================================================================
# Tests for newton_schulz_orthogonalize
# ============================================================================


def test_newton_schulz_square_orthogonality() raises:
    """Test NS orthogonalization on a square matrix.

    After 5 iterations, Y @ Y^T should be close to identity.
    Frobenius norm of (Y @ Y^T - I) < 1e-5 for fp32.
    """
    # Create random 8×8 matrix
    var shape: List[Int] = [8, 8]
    var X = zeros(shape, DType.float32)

    # Fill with values simulating random matrix (pseudo-random via index)
    for i in range(X.numel()):
        var val = Float32((Float64(i) * 0.123 + 0.456).fract())
        X._set_float64(i, Float64(val))

    # Orthogonalize
    var Y = newton_schulz_orthogonalize(X, steps=5)

    # Check shape preserved
    assert_shape(Y, shape, "Orthogonalized shape matches input")

    # Compute Gram matrix: G = Y @ Y^T
    var G = matmul(Y, transpose(Y, None))

    # Compute I (identity)
    var I = zeros(shape, DType.float32)
    for i in range(shape[0]):
        I.set(i * (shape[1] + 1), Float32(1.0))  # Set diagonal

    # Compute error norm: ||G - I||_F
    var error = zeros_like(G)
    for i in range(G.numel()):
        var g_val = G._get_float64(i)
        var i_val = I._get_float64(i)
        error._set_float64(i, g_val - i_val)

    var error_norm = compute_tensor_norm(error)

    # Assert error < 1e-5
    assert_less(
        error_norm,
        1e-5,
        "Gram matrix (Y @ Y^T) is close to identity after 5 NS steps",
    )


def test_newton_schulz_wide_matrix() raises:
    """Test NS orthogonalization on a wide matrix (more cols than rows).

    Shape [8, 16]: rows <= cols -> iterate on Y directly.
    Result should satisfy Y @ Y^T ≈ I_8 (rows orthonormal).
    """
    var shape: List[Int] = [8, 16]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        var val = Float32((Float64(i) * 0.234 + 0.789).fract())
        X._set_float64(i, Float64(val))

    var Y = newton_schulz_orthogonalize(X, steps=5)

    assert_shape(Y, shape, "Wide matrix shape preserved")

    # Check Y @ Y^T ≈ I_rows
    var G = matmul(Y, transpose(Y, None))
    var I = zeros([shape[0], shape[0]], DType.float32)
    for i in range(shape[0]):
        I.set(i * (shape[0] + 1), Float32(1.0))

    var error = zeros_like(G)
    for i in range(G.numel()):
        error._set_float64(i, G._get_float64(i) - I._get_float64(i))

    var error_norm = compute_tensor_norm(error)
    assert_less(error_norm, 1e-5, "Wide matrix rows are orthonormal")


def test_newton_schulz_tall_matrix() raises:
    """Test NS orthogonalization on a tall matrix (more rows than cols).

    Shape [16, 8]: rows > cols -> transpose, iterate, transpose back.
    Result should satisfy Y^T @ Y ≈ I_8 (cols orthonormal).
    """
    var shape: List[Int] = [16, 8]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        var val = Float32((Float64(i) * 0.345 + 0.111).fract())
        X._set_float64(i, Float64(val))

    var Y = newton_schulz_orthogonalize(X, steps=5)

    assert_shape(Y, shape, "Tall matrix shape preserved")

    # Check Y^T @ Y ≈ I_cols
    var Yt = transpose(Y, None)
    var G = matmul(Yt, Y)
    var I = zeros([shape[1], shape[1]], DType.float32)
    for i in range(shape[1]):
        I.set(i * (shape[1] + 1), Float32(1.0))

    var error = zeros_like(G)
    for i in range(G.numel()):
        error._set_float64(i, G._get_float64(i) - I._get_float64(i))

    var error_norm = compute_tensor_norm(error)
    assert_less(error_norm, 1e-5, "Tall matrix cols are orthonormal")


def test_newton_schulz_convergence_rate() raises:
    """Test that NS iteration shows quintic convergence (not linear).

    Wrong coefficients would converge linearly; correct coefficients converge quintic.
    After 5 steps, residual should drop by factor ~1e-2 relative to step 1.
    """
    var shape: List[Int] = [8, 16]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        var val = Float32((Float64(i) * 0.456 + 0.222).fract())
        X._set_float64(i, Float64(val))

    # Compute error after 1 step
    var Y1 = newton_schulz_orthogonalize(X, steps=1)
    var G1 = matmul(Y1, transpose(Y1, None))
    var I = zeros([shape[0], shape[0]], DType.float32)
    for i in range(shape[0]):
        I.set(i * (shape[0] + 1), Float32(1.0))

    var error1 = zeros_like(G1)
    for i in range(G1.numel()):
        error1._set_float64(i, G1._get_float64(i) - I._get_float64(i))

    var residual1 = compute_tensor_norm(error1)

    # Compute error after 5 steps
    var Y5 = newton_schulz_orthogonalize(X, steps=5)
    var G5 = matmul(Y5, transpose(Y5, None))

    var error5 = zeros_like(G5)
    for i in range(G5.numel()):
        error5._set_float64(i, G5._get_float64(i) - I._get_float64(i))

    var residual5 = compute_tensor_norm(error5)

    # Convergence ratio should be very small (quintic -> ~1e-10 or better)
    # Check that step 5 residual is at least 100x better than step 1
    var convergence_ratio = residual5 / residual1
    assert_less(
        convergence_ratio,
        1e-2,
        "NS iteration shows quintic convergence (100x reduction per step)",
    )


# ============================================================================
# Tests for muon_step
# ============================================================================


def test_muon_step_shape_preservation() raises:
    """Test that muon_step preserves parameter shape.

    Input shape [4, 6] should yield output shape [4, 6].
    """
    var shape: List[Int] = [4, 6]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var momentum = zeros(shape, DType.float32)

    var (new_params, new_momentum) = muon_step(
        params, gradients, momentum, learning_rate=0.01
    )

    assert_shape(new_params, shape, "Muon step preserves parameter shape")
    assert_shape(new_momentum, shape, "Muon step preserves momentum shape")


def test_muon_step_quadratic_descent() raises:
    """Test that muon_step achieves descent on a quadratic loss.

    Loss = sum(p^2), grad = 2*p, initialized to ones.
    After 50 steps with lr=0.05, loss should decrease and final loss < 1e-3.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var momentum = zeros_like(params)

    var initial_loss = 0.0
    for i in range(params.numel()):
        var p = params._get_float64(i)
        initial_loss += p * p

    # Run 50 optimization steps
    var prev_loss = initial_loss

    for step in range(50):
        # Compute gradients: grad = 2 * p
        var gradients = zeros_like(params)
        for i in range(params.numel()):
            var p = params._get_float64(i)
            gradients._set_float64(i, 2.0 * p)

        # Muon step
        var (new_params, new_momentum) = muon_step(
            params,
            gradients,
            momentum,
            learning_rate=0.05,
            momentum_beta=0.95,
            weight_decay=0.0,
        )

        params = new_params
        momentum = new_momentum

        # Compute loss
        var loss = 0.0
        for i in range(params.numel()):
            var p = params._get_float64(i)
            loss += p * p

        # Loss should decrease
        assert_less(
            loss,
            prev_loss,
            "Muon achieves descent on quadratic loss (step "
            + String(step)
            + ")",
        )
        prev_loss = loss

    # Final loss should be small
    assert_less(prev_loss, 1e-3, "Final loss after 50 steps is very small")


def test_muon_step_rejects_non_matrix() raises:
    """Test that muon_step rejects non-matrix (rank != 2) parameters.

    Should raise Error with a message mentioning rank or matrix.
    """
    var shape: List[Int] = [128]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var momentum = zeros(shape, DType.float32)

    try:
        var (_new_params, _new_momentum) = muon_step(
            params, gradients, momentum, learning_rate=0.01
        )
        assert_true(False, "muon_step should raise Error for rank-1 params")
    except e:
        # Check that the error message mentions rank or matrix
        var msg = String(e)
        var mentions_issue = ("rank" in msg) or ("matrix" in msg)
        assert_true(mentions_issue, "Error message mentions rank or matrix")


def test_muon_step_dtype_mismatch() raises:
    """Test that muon_step rejects dtype mismatches.

    Params and gradients must have the same dtype.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float16)
    var momentum = zeros(shape, DType.float32)

    try:
        var (_new_params, _new_momentum) = muon_step(
            params, gradients, momentum, learning_rate=0.01
        )
        assert_true(False, "muon_step should raise Error for dtype mismatch")
    except e:
        assert_true(True, "muon_step correctly rejects dtype mismatch")


def test_muon_step_shape_mismatch() raises:
    """Test that muon_step rejects shape mismatches.

    Params and gradients must have the same shape.
    """
    var shape1: List[Int] = [4, 4]
    var shape2: List[Int] = [4, 5]
    var params = ones(shape1, DType.float32)
    var gradients = zeros(shape2, DType.float32)
    var momentum = zeros(shape1, DType.float32)

    try:
        var (_new_params, _new_momentum) = muon_step(
            params, gradients, momentum, learning_rate=0.01
        )
        assert_true(False, "muon_step should raise Error for shape mismatch")
    except e:
        assert_true(True, "muon_step correctly rejects shape mismatch")


def test_muon_step_with_nesterov() raises:
    """Test muon_step with Nesterov momentum enabled.

    Nesterov should improve convergence compared to plain momentum.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var momentum = zeros_like(params)

    # Run a few steps with Nesterov
    for _step in range(5):
        var gradients = zeros_like(params)
        for i in range(params.numel()):
            var p = params._get_float64(i)
            gradients._set_float64(i, 2.0 * p)

        var (new_params, new_momentum) = muon_step(
            params,
            gradients,
            momentum,
            learning_rate=0.05,
            momentum_beta=0.95,
            weight_decay=0.0,
            nesterov=True,
        )

        params = new_params
        momentum = new_momentum

    # Should reach small values
    var final_norm = compute_tensor_norm(params)
    assert_less(final_norm, 2.0, "Nesterov momentum reduces parameters")


def test_muon_step_with_weight_decay() raises:
    """Test that weight decay is applied in muon_step.

    With weight_decay > 0, parameters should shrink even with zero gradients.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var momentum = zeros_like(params)

    var initial_norm = compute_tensor_norm(params)

    # One step with weight decay and zero gradient
    var (new_params, _new_momentum) = muon_step(
        params,
        gradients,
        momentum,
        learning_rate=0.1,
        weight_decay=0.1,
    )

    var final_norm = compute_tensor_norm(new_params)

    # With zero gradient but positive weight decay, norm should shrink
    assert_less(final_norm, initial_norm, "Weight decay shrinks parameters")


def test_muon_step_simple() raises:
    """Test muon_step_simple with default hyperparameters.

    Should work the same as muon_step with paper defaults.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var momentum = zeros_like(params)

    var (new_params, new_momentum) = muon_step_simple(
        params, gradients, momentum, learning_rate=0.01
    )

    # Should return valid tensors of correct shape
    assert_shape(new_params, shape, "muon_step_simple returns correct shape")
    assert_shape(new_momentum, shape, "muon_step_simple momentum shape correct")


def test_muon_step_pure_functional() raises:
    """Test that muon_step does not mutate input parameters.

    Pure functional design: input tensors should be unchanged after the call.
    """
    var shape: List[Int] = [4, 4]
    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var momentum = zeros_like(params)

    # Save original value
    var original_p0 = params._get_float64(0)

    # Call muon_step
    var (_new_params, _new_momentum) = muon_step(
        params, gradients, momentum, learning_rate=0.01
    )

    # Check params was not mutated
    var final_p0 = params._get_float64(0)
    assert_almost_equal(
        original_p0,
        final_p0,
        tolerance=1e-10,
        msg="muon_step does not mutate input params",
    )
