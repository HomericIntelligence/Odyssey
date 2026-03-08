"""Tests for optimizer base gradient zeroing and clipping functionality.

Tests the OptimizerBase gradient zeroing and clipping:
- Gradient zeroing implementation
- SGD and Adam zero_grad methods
- Optimizer state preservation across zero_grad
- Gradient clipping by global norm (no clipping, with clipping, multiple params)

Split from test_optimizer_base.mojo to comply with ADR-009 (≤10 fn test_ per file).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizer_base.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from tests.shared.conftest import assert_almost_equal
from shared.core.extensor import ExTensor, zeros
from shared.autograd import Variable, GradientTape, SGD, Adam, AdaGrad, RMSprop
from shared.autograd.optimizer_base import (
    zero_grad_impl,
    clip_gradients_by_global_norm,
)


# ============================================================================
# Gradient Zeroing Tests
# ============================================================================


fn test_zero_grad_implementation() raises:
    """Test that zero_grad clears the gradient tape."""
    var tape = GradientTape()

    # Create a variable to register with the tape (gets auto-assigned ID)
    var shape: List[Int] = [1]
    var data = ExTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add a gradient for this variable
    var grad = ExTensor(shape, DType.float32)
    grad._set_float64(0, 1.0)
    tape.registry.set_grad(var_id, grad^)

    # Verify gradient exists
    assert_true(
        tape.registry.has_gradient(var_id), "Gradient should exist before clear"
    )

    # Clear gradients using implementation
    zero_grad_impl(tape)

    # Verify gradient was cleared
    assert_true(
        not tape.registry.has_gradient(var_id), "Gradient should be cleared"
    )


fn test_sgd_zero_grad() raises:
    """Test SGD zero_grad method clears tape."""
    var optimizer = SGD(learning_rate=0.01)
    var tape = GradientTape()

    # Create a variable to register with tape
    var shape: List[Int] = [1]
    var data = ExTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add gradient
    var grad = ExTensor(shape, DType.float32)
    tape.registry.set_grad(var_id, grad^)

    # Clear gradients
    optimizer.zero_grad(tape)

    # Verify cleared
    assert_true(
        not tape.registry.has_gradient(var_id),
        "SGD zero_grad should clear tape",
    )


fn test_adam_zero_grad() raises:
    """Test Adam zero_grad method clears tape."""
    var optimizer = Adam()
    var tape = GradientTape()

    # Create a variable to register with tape
    var shape: List[Int] = [1]
    var data = ExTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add gradient
    var grad = ExTensor(shape, DType.float32)
    tape.registry.set_grad(var_id, grad^)

    # Clear gradients
    optimizer.zero_grad(tape)

    # Verify cleared
    assert_true(
        not tape.registry.has_gradient(var_id),
        "Adam zero_grad should clear tape",
    )


fn test_zero_grad_preserves_optimizer_state() raises:
    """Test zero_grad clears tape but preserves optimizer internal state."""
    var optimizer = Adam()
    var tape = GradientTape()
    tape.enable()

    # Create a parameter
    var shape: List[Int] = [2]
    var data = ExTensor(shape, DType.float32)
    data._set_float64(0, 1.0)
    data._set_float64(1, 2.0)

    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add gradient
    var grad = ExTensor(shape, DType.float32)
    grad._set_float64(0, 0.1)
    grad._set_float64(1, 0.2)
    tape.registry.set_grad(var_id, grad^)

    # Take a step (initializes moment buffers)
    optimizer.step(parameters, tape)

    # Verify moment buffers were initialized
    assert_equal(len(optimizer.m_buffers), 1, "Should have 1 m_buffer")
    assert_equal(len(optimizer.v_buffers), 1, "Should have 1 v_buffer")

    # Clear gradients
    optimizer.zero_grad(tape)

    # Verify moment buffers are preserved
    assert_equal(len(optimizer.m_buffers), 1, "m_buffers should be preserved")
    assert_equal(len(optimizer.v_buffers), 1, "v_buffers should be preserved")


# ============================================================================
# Gradient Clipping Tests
# ============================================================================


fn test_clip_gradients_no_clipping_needed() raises:
    """Test gradient clipping when norm is below threshold."""
    var tape = GradientTape()
    tape.enable()

    # Create parameter
    var shape: List[Int] = [3]
    var data = ExTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add small gradient (norm < 5.0)
    var grad = ExTensor(shape, DType.float32)
    grad._set_float64(0, 0.1)
    grad._set_float64(1, 0.1)
    grad._set_float64(2, 0.1)
    tape.registry.set_grad(var_id, grad^)

    # Clip with max_norm=5.0 (should not clip)
    var original_norm = clip_gradients_by_global_norm(
        parameters, tape, max_norm=5.0
    )

    # Verify original norm
    var expected_norm = (0.1**2 + 0.1**2 + 0.1**2) ** 0.5
    assert_almost_equal(original_norm, expected_norm, tolerance=1e-6)

    # Verify gradients unchanged
    var clipped_grad = tape.registry.get_grad(var_id)
    assert_almost_equal(clipped_grad._get_float64(0), 0.1, tolerance=1e-6)
    assert_almost_equal(clipped_grad._get_float64(1), 0.1, tolerance=1e-6)
    assert_almost_equal(clipped_grad._get_float64(2), 0.1, tolerance=1e-6)


fn test_clip_gradients_with_clipping() raises:
    """Test gradient clipping when norm exceeds threshold."""
    var tape = GradientTape()
    tape.enable()

    # Create parameter
    var shape: List[Int] = [3]
    var data = ExTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add large gradient (norm > 1.0)
    var grad = ExTensor(shape, DType.float32)
    grad._set_float64(0, 3.0)
    grad._set_float64(1, 4.0)
    grad._set_float64(2, 0.0)
    tape.registry.set_grad(var_id, grad^)

    # Original norm = sqrt(9 + 16) = 5.0
    # Clip to max_norm=1.0
    var original_norm = clip_gradients_by_global_norm(
        parameters, tape, max_norm=1.0
    )

    # Verify original norm
    assert_almost_equal(original_norm, 5.0, tolerance=1e-6)

    # Verify gradients were scaled down by factor of 1.0/5.0 = 0.2
    var clipped_grad = tape.registry.get_grad(var_id)
    assert_almost_equal(
        clipped_grad._get_float64(0), 0.6, tolerance=1e-6
    )  # 3.0 * 0.2
    assert_almost_equal(
        clipped_grad._get_float64(1), 0.8, tolerance=1e-6
    )  # 4.0 * 0.2
    assert_almost_equal(clipped_grad._get_float64(2), 0.0, tolerance=1e-10)

    # Verify new norm is max_norm
    var new_norm_squared = (
        clipped_grad._get_float64(0) ** 2
        + clipped_grad._get_float64(1) ** 2
        + clipped_grad._get_float64(2) ** 2
    )
    var new_norm = new_norm_squared**0.5
    assert_almost_equal(new_norm, 1.0, tolerance=1e-6)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run optimizer base gradient zeroing and clipping tests."""
    print("Running gradient zeroing tests...")
    test_zero_grad_implementation()
    test_sgd_zero_grad()
    test_adam_zero_grad()
    test_zero_grad_preserves_optimizer_state()

    print("Running gradient clipping tests...")
    test_clip_gradients_no_clipping_needed()
    test_clip_gradients_with_clipping()

    print("\nAll optimizer base part2 tests passed! ✓")
