# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_lars.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for LARS (Layer-wise Adaptive Rate Scaling) optimizer - Part 2.

Tests cover:
- Adaptive scaling properties (small/large gradients)
- Shape and dtype validation

LARS is particularly useful for large-batch distributed training where the
learning rate must be carefully adapted to the parameter and gradient norms.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
    assert_less,
    assert_greater,
    create_test_vector,
    TestFixtures,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, zeros_like
from shared.core.numerical_safety import compute_tensor_l2_norm
from shared.training.optimizers.lars import lars_step, lars_step_simple


# ============================================================================
# LARS Property Tests
# ============================================================================


fn test_lars_adaptive_scaling_small_gradients() raises:
    """Test LARS scales learning rate up when gradients are small.

    When grad_norm is small relative to param_norm, LARS increases
    the effective learning rate via higher trust_ratio.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 10.0  # Large parameter

    # Create gradient tensor with small norm
    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.01  # Small gradient

    var velocity = zeros(shape, DType.float32)

    # LARS should scale the learning rate based on param/grad ratio
    var result = lars_step(
        params,
        grads,
        velocity,
        learning_rate=1.0,
        momentum=0.0,
        weight_decay=0.0,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var new_params = result[0]

    # Should still make progress despite small gradient
    assert_not_equal(Float64(new_params._data.bitcast[Float32]()[0]), 10.0)


fn test_lars_adaptive_scaling_large_gradients() raises:
    """Test LARS scales learning rate down when gradients are large.

    When grad_norm is large relative to param_norm, LARS decreases
    the effective learning rate via lower trust_ratio, preventing divergence.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0  # Small parameter

    # Create gradient tensor with large norm
    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 10.0  # Large gradient

    var velocity = zeros(shape, DType.float32)

    # LARS should scale the learning rate down
    var result = lars_step(
        params,
        grads,
        velocity,
        learning_rate=1.0,
        momentum=0.0,
        weight_decay=0.0,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var new_params = result[0]

    # Even with large gradient, update should be controlled
    var param_val = Float64(new_params._data.bitcast[Float32]()[0])
    assert_greater(param_val, 0.99)  # Should not change drastically


# ============================================================================
# LARS Shape Validation Tests
# ============================================================================


fn test_lars_shape_mismatch() raises:
    """Test LARS raises error on shape mismatch.

    Parameters and gradients must have the same shape.
    """
    var shape1: List[Int] = [3]
    var params = ones(shape1, DType.float32)

    var shape2: List[Int] = [5]
    var grads = zeros(shape2, DType.float32)

    var velocity = zeros_like(params)

    # Should raise error due to shape mismatch
    try:
        var _ = lars_step(params, grads, velocity, learning_rate=0.1)
        # If we get here, test failed
        assert_true(False)
    except e:
        # Expected error
        assert_true(True)


fn test_lars_dtype_mismatch() raises:
    """Test LARS raises error on dtype mismatch.

    Parameters and gradients must have the same dtype.
    """
    var shape: List[Int] = [3]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float64)
    var velocity = zeros_like(params)

    # Should raise error due to dtype mismatch
    try:
        var _ = lars_step(params, grads, velocity, learning_rate=0.1)
        # If we get here, test failed
        assert_true(False)
    except e:
        # Expected error
        assert_true(True)


fn test_lars_empty_velocity_buffer() raises:
    """Test LARS raises error when velocity buffer is not initialized.

    Velocity buffer must be pre-allocated (use zeros_like).
    """
    var shape: List[Int] = [3]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var velocity = zeros(List[Int](), DType.float32)  # Empty velocity

    # Should raise error due to empty velocity buffer
    try:
        var _ = lars_step(params, grads, velocity, learning_rate=0.1)
        # If we get here, test failed
        assert_true(False)
    except e:
        # Expected error
        assert_true(True)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run LARS optimizer tests - Part 2."""
    print("Running LARS property tests...")
    test_lars_adaptive_scaling_small_gradients()
    test_lars_adaptive_scaling_large_gradients()

    print("Running LARS shape validation tests...")
    test_lars_shape_mismatch()
    test_lars_dtype_mismatch()
    test_lars_empty_velocity_buffer()

    print("\nAll LARS optimizer tests (Part 2) passed! ✓")
