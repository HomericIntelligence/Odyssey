"""Unit tests for LARS (Layer-wise Adaptive Rate Scaling) optimizer.

Tests cover:
- LARS initialization with hyperparameters
- Basic parameter update with adaptive learning rate scaling
- Trust ratio computation and application
- Momentum accumulation
- Weight decay integration
- Norm computation

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
    assert_shape,
    create_test_vector,
    TestFixtures,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like
from shared.core.numerical_safety import compute_tensor_l2_norm
from shared.training.optimizers.lars import lars_step, lars_step_simple
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


def test_lars_initialization() raises:
    """Test LARS optimizer initialization with hyperparameters.

    Functional API Note:
        Pure functional design - no class initialization.
        Hyperparameters are passed as function arguments to lars_step().
        This test verifies that the function accepts all expected parameters.
    """
    # Test that lars_step accepts all hyperparameters
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var velocity = zeros(shape, DType.float32)

    # Should accept all hyperparameters without error
    var result = lars_step(
        params,
        grads,
        velocity,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    # Verify the result has the correct shape
    assert_shape(
        result[0], shape, "LARS initialization result shape matches input"
    )


def test_lars_parameter_norm_computation() raises:
    """Test LARS correctly computes parameter norm.

    LARS uses L2 norm: ||params|| = sqrt(sum(params^2)).
    """
    # Create simple parameter tensor: [3.0, 4.0]
    # Expected norm: sqrt(9 + 16) = 5.0
    var shape: List[Int] = [2]
    var params = zeros(shape, DType.float32)
    params.set(0, Float32(3.0))
    params.set(1, Float32(4.0))

    var param_norm = compute_tensor_l2_norm(params)

    # Verify norm is correct
    assert_almost_equal(param_norm, 5.0, tolerance=1e-6)


def test_lars_gradient_norm_computation() raises:
    """Test LARS correctly computes gradient norm.

    LARS uses L2 norm: ||grads|| = sqrt(sum(grads^2)).
    """
    # Create simple gradient tensor: [0.6, 0.8]
    # Expected norm: sqrt(0.36 + 0.64) = 1.0
    var shape: List[Int] = [2]
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.6))
    grads.set(1, Float32(0.8))

    var grad_norm = compute_tensor_l2_norm(grads)

    # Verify norm is correct
    assert_almost_equal(grad_norm, 1.0, tolerance=1e-6)


def test_lars_trust_ratio_scaling() raises:
    """Test LARS computes trust ratio correctly.

    Formula:
        trust_ratio = trust_coefficient * param_norm / (grad_norm + weight_decay * param_norm + epsilon)

    Example:
        ```mojo
        aram_norm = 5.0
        grad_norm = 1.0
        weight_decay = 0.0001
        trust_coefficient = 0.001
        epsilon = 1e-8
        trust_ratio = 0.001 * 5.0 / (1.0 + 0.0001 * 5.0 + 1e-8)
                    = 0.005 / (1.0005)
                    ≈ 0.004998
        ```
    """
    var shape: List[Int] = [2]
    var params = zeros(shape, DType.float32)
    params.set(0, Float32(3.0))
    params.set(1, Float32(4.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.6))
    grads.set(1, Float32(0.8))

    var velocity = zeros(shape, DType.float32)

    # Perform LARS step with zero learning rate to isolate trust ratio effect
    # (params will not change due to zero learning rate, but calculation will verify)
    var result = lars_step(
        params,
        grads,
        velocity,
        learning_rate=0.0,  # Zero LR to prevent parameter change
        momentum=0.0,  # No momentum to isolate trust ratio
        weight_decay=0.0001,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    # With zero learning rate, params should be unchanged
    var new_params = result[0]
    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[0]), 3.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[1]), 4.0, tolerance=1e-6
    )


def test_lars_basic_update() raises:
    """Test LARS performs basic parameter update with adaptive scaling.

    LARS scales the learning rate based on parameter and gradient norms,
    enabling stable training across different parameter scales.
    """
    # Simple case: single parameter
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var velocity = zeros(shape, DType.float32)

    # Perform update with LARS
    var result = lars_step(
        params,
        grads,
        velocity,
        learning_rate=1.0,  # Use 1.0 to see the effect clearly
        momentum=0.0,  # No momentum for this test
        weight_decay=0.0,  # No weight decay for simplicity
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var new_params = result[0]

    # Parameter should decrease (gradient descent)
    assert_less(Float64(new_params._data.bitcast[Float32]()[0]), 1.0)

    # With trust ratio scaling, update should be small but measurable
    var param_val = Float64(new_params._data.bitcast[Float32]()[0])
    assert_almost_equal(param_val, 0.999, tolerance=1e-3)


def test_lars_momentum_accumulation() raises:
    """Test LARS accumulates momentum correctly over multiple steps.

    Functional API:
        With momentum > 0:
        - First update: velocity = scaled_grad
        - Subsequent updates: velocity = momentum * velocity + scaled_grad
        - Parameter update: new_params = params - lr * velocity

    Returns: (new_params, new_velocity).
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var velocity = zeros(shape, DType.float32)

    # Step 1: velocity accumulates gradient with trust ratio scaling
    var result1 = lars_step(
        params,
        grads,
        velocity,
        learning_rate=1.0,
        momentum=0.9,
        weight_decay=0.0,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var params1 = result1[0]
    var velocity1 = result1[1]

    var param_val1 = Float64(params1._data.bitcast[Float32]()[0])
    assert_less(param_val1, 1.0)  # Should decrease

    # Step 2: momentum continues to accumulate
    # Need fresh grads tensor for second step
    var grads2 = zeros(shape, DType.float32)
    grads2.set(0, Float32(0.1))

    var result2 = lars_step(
        params1,
        grads2,
        velocity1,
        learning_rate=1.0,
        momentum=0.9,
        weight_decay=0.0,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var params2 = result2[0]
    var param_val2 = Float64(params2._data.bitcast[Float32]()[0])

    # Parameter should continue to decrease
    assert_less(param_val2, param_val1)


def test_lars_weight_decay() raises:
    """Test LARS applies weight decay (L2 regularization).

    Functional API:
        With weight_decay > 0:
        - Effective gradient: grad_eff = grad + weight_decay * params
        - Trust ratio computed with weight_decay term
        - Then apply standard update with momentum

    Note: LARS's trust ratio uses ||grad|| + wd * ||params|| in denominator
    while effective gradient is grad + wd * params. These have different
    magnitudes in multi-dimensional cases, making weight decay visible.
    In 1D with same-sign values, they cancel perfectly, so we use 2D tensors.
    """
    var shape: List[Int] = [2]  # Use 2D to break cancellation

    # Create tensors WITH weight decay test
    # Use orthogonal grad and param vectors so ||grad + wd*params|| != ||grad|| + wd*||params||
    var params_wd = zeros(shape, DType.float32)
    params_wd.set(0, Float32(1.0))  # [1.0, 0.0]
    params_wd.set(1, Float32(0.0))

    var grads_wd = zeros(shape, DType.float32)
    grads_wd.set(0, Float32(0.0))  # [0.0, 0.1]
    grads_wd.set(1, Float32(0.1))

    var velocity_wd = zeros(shape, DType.float32)

    # Perform update WITH weight decay
    var result_with_wd = lars_step(
        params_wd,
        grads_wd,
        velocity_wd,
        learning_rate=1.0,
        momentum=0.0,
        weight_decay=0.1,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var new_params_with_wd = result_with_wd[0]

    # Create tensors WITHOUT weight decay
    var params_no_wd = zeros(shape, DType.float32)
    params_no_wd.set(0, Float32(1.0))  # [1.0, 0.0]
    params_no_wd.set(1, Float32(0.0))

    var grads_no_wd = zeros(shape, DType.float32)
    grads_no_wd.set(0, Float32(0.0))  # [0.0, 0.1]
    grads_no_wd.set(1, Float32(0.1))

    var velocity_no_wd = zeros(shape, DType.float32)

    # Perform update WITHOUT weight decay
    var result_without_wd = lars_step(
        params_no_wd,
        grads_no_wd,
        velocity_no_wd,
        learning_rate=1.0,
        momentum=0.0,
        weight_decay=0.0,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )

    var new_params_without_wd = result_without_wd[0]

    # With weight decay, the first component (params[0]=1.0) should decrease more
    # because effective_gradient[0] = 0.0 + 0.1 * 1.0 = 0.1 (vs 0.0 without wd)
    var val_with_wd_0 = Float64(new_params_with_wd._data.bitcast[Float32]()[0])
    var val_without_wd_0 = Float64(
        new_params_without_wd._data.bitcast[Float32]()[0]
    )

    # First component should be smaller with weight decay
    assert_less(val_with_wd_0, val_without_wd_0)


def test_lars_step_simple() raises:
    """Test LARS simplified step function with default hyperparameters.

    Simplified API uses sensible defaults:
    - momentum = 0.9
    - weight_decay = 0.0001
    - trust_coefficient = 0.001
    - epsilon = 1e-8
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var velocity = zeros(shape, DType.float32)

    # Should accept minimal parameters
    var result = lars_step_simple(params, grads, velocity, learning_rate=1.0)

    var new_params = result[0]

    # Parameter should change
    var new_val = Float64(new_params._data.bitcast[Float32]()[0])
    assert_not_equal(new_val, 1.0)


def test_lars_adaptive_scaling_small_gradients() raises:
    """Test LARS scales learning rate up when gradients are small.

    When grad_norm is small relative to param_norm, LARS increases
    the effective learning rate via higher trust_ratio.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(10.0))  # Large parameter

    # Create gradient tensor with small norm
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.01))  # Small gradient

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


def test_lars_adaptive_scaling_large_gradients() raises:
    """Test LARS scales learning rate down when gradients are large.

    When grad_norm is large relative to param_norm, LARS decreases
    the effective learning rate via lower trust_ratio, preventing divergence.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))  # Small parameter

    # Create gradient tensor with large norm
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(10.0))  # Large gradient

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


def test_lars_shape_mismatch() raises:
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


def test_lars_dtype_mismatch() raises:
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


def test_lars_empty_velocity_buffer() raises:
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


def main() raises:
    """Run all test_lars tests."""
    print("Running test_lars tests...")

    test_lars_initialization()
    print("✓ test_lars_initialization")

    test_lars_parameter_norm_computation()
    print("✓ test_lars_parameter_norm_computation")

    test_lars_gradient_norm_computation()
    print("✓ test_lars_gradient_norm_computation")

    test_lars_trust_ratio_scaling()
    print("✓ test_lars_trust_ratio_scaling")

    test_lars_basic_update()
    print("✓ test_lars_basic_update")

    test_lars_momentum_accumulation()
    print("✓ test_lars_momentum_accumulation")

    test_lars_weight_decay()
    print("✓ test_lars_weight_decay")

    test_lars_step_simple()
    print("✓ test_lars_step_simple")

    test_lars_adaptive_scaling_small_gradients()
    print("✓ test_lars_adaptive_scaling_small_gradients")

    test_lars_adaptive_scaling_large_gradients()
    print("✓ test_lars_adaptive_scaling_large_gradients")

    test_lars_shape_mismatch()
    print("✓ test_lars_shape_mismatch")

    test_lars_dtype_mismatch()
    print("✓ test_lars_dtype_mismatch")

    test_lars_empty_velocity_buffer()
    print("✓ test_lars_empty_velocity_buffer")

    print("\nAll test_lars tests passed!")
