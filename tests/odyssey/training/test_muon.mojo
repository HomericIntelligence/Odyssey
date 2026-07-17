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

from tests.odyssey.conftest import (
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
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import (
    zeros,
    ones,
    zeros_like,
    full,
)
from odyssey.training.optimizers.muon import (
    muon_step,
    muon_step_simple,
    newton_schulz_orthogonalize,
    is_muon_eligible,
)
from odyssey.training.optimizers.optimizer_utils import (
    compute_tensor_norm,
)
from odyssey.core.matrix import matmul, transpose


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
#
# What Jordan's Newton-Schulz variant actually guarantees:
# The quintic coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are tuned for
# SPEED, not for asymptotic convergence to exact orthogonality. Jordan's
# writeup (https://kellerjordan.github.io/posts/muon/) states the iteration
# drives every singular value into roughly [0.68, 1.13] and then OSCILLATES in
# that band — it never converges to sv == 1, no matter how many steps you run
# (verified in fp64 numpy: a Gaussian 8x8 input gives ||YY^T - I||_F ~= 0.62
# after 5 steps and ~= 1.27 after 50). Therefore Gram ~= I to 1e-5 is NOT a
# property of a correct implementation, and tests must assert the sv band
# instead:
#   - sv in [s_min, s_max] implies ||YY^T - I||_F <= sqrt(r)*max|sv^2 - 1|;
#     for [0.68, 1.13] and r = 8 that bound is ~1.52 (we assert < 1.6).
#   - G_ii = ||e_i^T Y||^2 is bounded by [sv_min^2, sv_max^2] ~ [0.46, 1.28]
#     (we assert [0.40, 1.40]).
# The previous 1e-5 assertions could never pass; they were invisible only
# because the CI harness swallowed the failure (PR #5625).
#
# Input matrices: the old "pseudo-random via index" affine fill
# ((i*k + b) % 1.0) is nearly low-rank (the 8x8 case had singular values down
# to ~2.5e-18) — NS cannot lift a zero singular value, so those inputs can
# never orthogonalize regardless of coefficients. _hash_unit below is a
# lowbias32-style integer hash: deterministic, bit-exact reproducible in
# numpy, and well-conditioned (cond ~ 3-30 for these shapes).
# ============================================================================


def _hash_unit(i: Int, seed: UInt32) -> Float64:
    """Deterministic well-spread pseudo-random value in [-0.5, 0.5).

    Lowbias32-style integer mixing hash of the flat index. Unlike an affine
    fill ((i*k + b) % 1.0), consecutive indices produce uncorrelated values,
    so the resulting matrices are well-conditioned. Bit-exact reproducible:
    (i*747796405 + 2891336453 + seed) mod 2^32, then xorshift-multiply mixing.
    """
    var h: UInt32 = UInt32(i) * 747796405 + 2891336453 + seed
    h ^= h >> 16
    h *= 0x7FEB352D
    h ^= h >> 15
    h *= 0x846CA68B
    h ^= h >> 16
    return Float64(h) / 4294967296.0 - 0.5


def _gram_identity_residual(
    G: AnyTensor, dim: Int
) raises -> Float64:
    """Return ||G - I_dim||_F for a square Gram matrix G."""
    var error = zeros_like(G)
    for i in range(G.numel()):
        var target = 0.0
        if i % (dim + 1) == 0:
            target = 1.0
        error._set_float64(i, G._get_float64(i) - target)
    return compute_tensor_norm(error)


def test_newton_schulz_square_orthogonality() raises:
    """Test NS orthogonalization on a well-conditioned square matrix.

    Asserts the property Jordan's NS variant actually guarantees after 5
    steps: singular values in ~[0.68, 1.13] (NOT exact orthogonality — see
    the block comment above). Checked via ||YY^T - I||_F < 1.6 (sv-band
    bound ~1.52; numpy fp32 reference for this exact input: ~0.97) and Gram
    diagonal entries in [0.40, 1.40] (sv-band bound [0.46, 1.28]).
    """
    var shape: List[Int] = [8, 8]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        X._set_float64(i, _hash_unit(i, 0))

    # Orthogonalize
    var Y = newton_schulz_orthogonalize(X, steps=5)

    # Check shape preserved
    assert_shape(Y, shape, "Orthogonalized shape matches input")

    # Compute Gram matrix: G = Y @ Y^T
    var G = matmul(Y, transpose(Y, None))

    var error_norm = _gram_identity_residual(G, shape[0])
    assert_less(
        error_norm,
        1.6,
        "||YY^T - I||_F within the sv-band bound after 5 NS steps",
    )

    # Every Gram diagonal entry (= squared row norm) inside the sv^2 band
    for r in range(shape[0]):
        var diag = G._get_float64(r * (shape[0] + 1))
        assert_greater(diag, 0.40, "Gram diagonal above sv_min^2 band")
        assert_less(diag, 1.40, "Gram diagonal below sv_max^2 band")


def test_newton_schulz_wide_matrix() raises:
    """Test NS orthogonalization on a wide matrix (more cols than rows).

    Shape [8, 16]: rows <= cols -> iterate on Y directly. Asserts the sv-band
    property on the row Gram matrix Y @ Y^T (numpy fp32 reference for this
    input: ||G - I_8||_F ~= 1.07, diagonals in [0.47, 1.25]).
    """
    var shape: List[Int] = [8, 16]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        X._set_float64(i, _hash_unit(i, 7))

    var Y = newton_schulz_orthogonalize(X, steps=5)

    assert_shape(Y, shape, "Wide matrix shape preserved")

    # Check Y @ Y^T lands in the sv band around I_rows
    var G = matmul(Y, transpose(Y, None))
    var error_norm = _gram_identity_residual(G, shape[0])
    assert_less(error_norm, 1.6, "Wide matrix rows near-orthonormal (sv band)")

    for r in range(shape[0]):
        var diag = G._get_float64(r * (shape[0] + 1))
        assert_greater(diag, 0.40, "Wide Gram diagonal above sv_min^2 band")
        assert_less(diag, 1.40, "Wide Gram diagonal below sv_max^2 band")


def test_newton_schulz_tall_matrix() raises:
    """Test NS orthogonalization on a tall matrix (more rows than cols).

    Shape [16, 8]: rows > cols -> transpose, iterate, transpose back. Asserts
    the sv-band property on the column Gram matrix Y^T @ Y (numpy fp32
    reference for this input: ||G - I_8||_F ~= 0.94, diagonals in [0.47, 1.26]).
    """
    var shape: List[Int] = [16, 8]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        X._set_float64(i, _hash_unit(i, 13))

    var Y = newton_schulz_orthogonalize(X, steps=5)

    assert_shape(Y, shape, "Tall matrix shape preserved")

    # Check Y^T @ Y lands in the sv band around I_cols
    var Yt = transpose(Y, None)
    var G = matmul(Yt, Y)
    var error_norm = _gram_identity_residual(G, shape[1])
    assert_less(error_norm, 1.6, "Tall matrix cols near-orthonormal (sv band)")

    for r in range(shape[1]):
        var diag = G._get_float64(r * (shape[1] + 1))
        assert_greater(diag, 0.40, "Tall Gram diagonal above sv_min^2 band")
        assert_less(diag, 1.40, "Tall Gram diagonal below sv_max^2 band")


def test_newton_schulz_convergence_progress() raises:
    """Test that more NS steps move the Gram matrix closer to identity.

    Jordan's tuned quintic trades asymptotic contraction for speed: once
    singular values reach the [0.68, 1.13] band it stops contracting (it
    oscillates), so a "100x per step" expectation is wrong BY DESIGN. The
    guaranteed property is monotone progress from 1 step to 5 steps while
    entering the band. numpy fp32 reference for this input: residual ratio
    (5 steps / 1 step) ~= 0.82; assert < 0.95.
    """
    var shape: List[Int] = [8, 16]
    var X = zeros(shape, DType.float32)

    for i in range(X.numel()):
        X._set_float64(i, _hash_unit(i, 29))

    # Residual after 1 step
    var Y1 = newton_schulz_orthogonalize(X, steps=1)
    var G1 = matmul(Y1, transpose(Y1, None))
    var residual1 = _gram_identity_residual(G1, shape[0])

    # Residual after 5 steps
    var Y5 = newton_schulz_orthogonalize(X, steps=5)
    var G5 = matmul(Y5, transpose(Y5, None))
    var residual5 = _gram_identity_residual(G5, shape[0])

    var convergence_ratio = residual5 / residual1
    assert_less(
        convergence_ratio,
        0.95,
        "5 NS steps land closer to the sv band than 1 step",
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
    """Test that muon_step achieves monotone descent on a quadratic loss.

    Loss = sum(p^2), grad = 2*p, initialized to ones.

    Muon's update is ORTHOGONALIZED and then scaled by a constant
    (lr * 0.2 * max(R,C)/sqrt(RC)), so — like sign-SGD — its step magnitude is
    independent of the gradient magnitude. Descent on a quadratic is therefore
    LINEAR per step, not exponential: each element moves by ~lr*scale*|u| ~
    0.005/step here, so 50 steps from p=1 cannot reach loss < 1e-3 (the old
    assertion; impossible by design, hidden by the swallowed-failure harness).
    numpy fp32 simulation of this exact loop: 16.0 -> 13.335, strictly
    monotone. Assert monotone decrease every step and final loss < 14.0.
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

    # Final loss reflects 50 constant-magnitude orthogonalized steps
    # (numpy fp32 reference: 13.335; linear descent, see docstring)
    assert_less(
        prev_loss,
        14.0,
        "Loss after 50 constant-magnitude Muon steps matches linear descent",
    )


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

    5 constant-magnitude Muon steps (each ~lr*scale*|u| ~ 0.005 per element)
    from ones(4,4) (norm 4.0) can only reduce the norm to ~3.965 (numpy fp32
    reference) — the old < 2.0 assertion was impossible by design (see
    test_muon_step_quadratic_descent). Assert the norm strictly decreased.
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

    # Norm strictly decreased from the initial 4.0
    # (numpy fp32 reference after 5 steps: ~3.965)
    var final_norm = compute_tensor_norm(params)
    assert_less(final_norm, 3.99, "Nesterov momentum reduces parameter norm")


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
        message="muon_step does not mutate input params",
    )


def main() raises:
    """Run all Muon optimizer tests."""
    print("=" * 60)
    print("Muon Optimizer Test Suite")
    print("=" * 60)

    test_is_muon_eligible_rank_1()
    test_is_muon_eligible_rank_3()
    test_is_muon_eligible_rank_2_matrix()
    test_is_muon_eligible_rank_2_small_row()
    test_is_muon_eligible_rank_2_small_col()
    test_is_muon_eligible_rank_2_min_size()
    test_newton_schulz_square_orthogonality()
    test_newton_schulz_wide_matrix()
    test_newton_schulz_tall_matrix()
    test_newton_schulz_convergence_progress()
    test_muon_step_shape_preservation()
    test_muon_step_quadratic_descent()
    test_muon_step_rejects_non_matrix()
    test_muon_step_dtype_mismatch()
    test_muon_step_shape_mismatch()
    test_muon_step_with_nesterov()
    test_muon_step_with_weight_decay()
    test_muon_step_simple()
    test_muon_step_pure_functional()

    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
