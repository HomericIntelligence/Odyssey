# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for optimizer implementations - Part 1 (SGD + Adam initialization).

Tests cover:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation) - initialization and basic update

Following TDD principles - these tests define the expected API
and numerical behavior for implementation in Issue #49.

Note: Tests have been adapted from class-based API to pure functional API
as per architecture decision to use functional design throughout shared library.

Split from test_optimizers.mojo per ADR-009 to avoid Mojo heap corruption
in Mojo v0.26.1 (libKGENCompilerRTShared.so JIT fault under high test load).
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    assert_shape,
    create_test_vector,
    TestFixtures,
)
from shared.core.extensor import AnyTensor, zeros, ones, zeros_like
from shared.training.optimizers.sgd import sgd_step, sgd_step_simple
from shared.training.optimizers.adam import adam_step, adam_step_simple


# ============================================================================
# SGD Tests
# ============================================================================


fn test_sgd_initialization() raises:
    """Test SGD optimizer initialization with hyperparameters.

    Functional API Note:
        Pure functional design - no class initialization.
        Hyperparameters are passed as function arguments to sgd_step().
        This test verifies that the function accepts all expected parameters.
    """
    # Test that sgd_step accepts all hyperparameters
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var velocity = zeros(shape, DType.float32)

    # Should accept all hyperparameters without error
    var result = sgd_step(
        params,
        grads,
        velocity,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0001,
    )

    # Verify the result has the correct shape
    assert_shape(result[0], shape, "SGD step result shape matches input")


fn test_sgd_basic_update() raises:
    """Test SGD performs basic parameter update without momentum.

    Functional API:
        sgd_step_simple(params, grads, learning_rate) -> new_params
        - Returns new parameters (pure functional)
        - Formula: new_params = params - lr * grads

    This is a CRITICAL test that defines the core SGD behavior.
    """
    # Initial parameters: [1.0, 2.0, 3.0]
    var shape: List[Int] = [3]
    var params = ones(shape, DType.float32)

    # Manually set values: [1.0, 2.0, 3.0]
    params._data.bitcast[Float32]()[0] = 1.0
    params._data.bitcast[Float32]()[1] = 2.0
    params._data.bitcast[Float32]()[2] = 3.0

    # Gradients: [0.1, 0.2, 0.3]
    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1
    grads._data.bitcast[Float32]()[1] = 0.2
    grads._data.bitcast[Float32]()[2] = 0.3

    # Perform update with lr=0.1
    var new_params = sgd_step_simple(params, grads, learning_rate=0.1)

    # Expected: new_params = params - lr * grads
    # [1.0 - 0.1*0.1, 2.0 - 0.1*0.2, 3.0 - 0.1*0.3]
    # = [0.99, 1.98, 2.97]
    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[0]), 0.99, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[1]), 1.98, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[2]), 2.97, tolerance=1e-6
    )


fn test_sgd_momentum_accumulation() raises:
    """Test SGD accumulates momentum correctly over multiple steps.

    Functional API:
        With momentum > 0:
        - First update: velocity = grad
        - Subsequent updates: velocity = momentum * velocity + grad
        - Parameter update: new_params = params - lr * velocity
        - Returns: (new_params, new_velocity)

    This is a CRITICAL test for momentum-based training.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var velocity = zeros(shape, DType.float32)

    # Step 1: velocity = grad = 0.1
    # update = lr * velocity = 0.1 * 0.1 = 0.01
    # params = 1.0 - 0.01 = 0.99
    var result = sgd_step(
        params, grads, velocity, learning_rate=0.1, momentum=0.9
    )
    params = result[0]
    velocity = result[1]

    assert_almost_equal(
        Float64(params._data.bitcast[Float32]()[0]), 0.99, tolerance=1e-6
    )

    # Step 2: velocity = 0.9 * 0.1 + 0.1 = 0.19
    # update = 0.1 * 0.19 = 0.019
    # params = 0.99 - 0.019 = 0.971
    result = sgd_step(params, grads, velocity, learning_rate=0.1, momentum=0.9)
    params = result[0]
    velocity = result[1]

    assert_almost_equal(
        Float64(params._data.bitcast[Float32]()[0]), 0.971, tolerance=1e-5
    )


fn test_sgd_weight_decay() raises:
    """Test SGD applies weight decay (L2 regularization).

    Functional API:
        With weight_decay > 0:
        - Effective gradient: grad = grad + weight_decay * params
        - Then apply standard update.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var velocity = zeros(shape, DType.float32)

    # Effective grad = 0.1 + 0.01 * 1.0 = 0.11
    # update = 0.1 * 0.11 = 0.011
    # params = 1.0 - 0.011 = 0.989
    var result = sgd_step(
        params, grads, velocity, learning_rate=0.1, weight_decay=0.01
    )
    var new_params = result[0]

    assert_almost_equal(
        Float64(new_params._data.bitcast[Float32]()[0]), 0.989, tolerance=1e-6
    )


fn test_sgd_nesterov_momentum() raises:
    """Test SGD with Nesterov momentum (lookahead).

    Not applicable to pure functional design - Nesterov momentum requires
    computing gradients at a different point (lookahead position), which
    would require the gradient computation to be part of the optimizer.

    In the functional design, gradient computation is external to the
    optimizer function, so Nesterov momentum is deferred.
    """
    pass  # Deferred - not applicable to pure functional design


fn test_sgd_zero_grad() raises:
    """Test SGD clears optimizer state (if needed).

    Not applicable to pure functional design - there is no internal state
    to clear. In the functional API, the caller manages all state (velocity
    buffers, etc.), so gradient clearing is the caller's responsibility.
    """
    pass  # Not applicable - no internal state in functional design


# ============================================================================
# Adam Tests
# ============================================================================


fn test_adam_initialization() raises:
    """Test Adam optimizer initialization.

    Functional API Note:
        Pure functional design - no class initialization.
        Hyperparameters are passed as function arguments to adam_step().
        This test verifies that the function accepts all expected parameters.
    """
    # Test that adam_step accepts all hyperparameters
    var shape = List[Int]()
    shape.append(3)
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Should accept all hyperparameters without error
    var result = adam_step(
        params,
        grads,
        m,
        v,
        t=1,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    # Verify the result has the correct shape
    assert_shape(result[0], shape, "Adam initialization result shape matches input")


fn test_adam_parameter_update() raises:
    """Test Adam performs correct parameter update.

    Functional API:
        Adam maintains two moments:
        - m (first moment, momentum)
        - v (second moment, RMSprop)

        Update formulas:
        - m = beta1 * m + (1 - beta1) * grad
        - v = beta2 * v + (1 - beta2) * grad^2
        - m_hat = m / (1 - beta1^t)  # Bias correction
        - v_hat = v / (1 - beta2^t)  # Bias correction
        - params = params - lr * m_hat / (sqrt(v_hat) + epsilon)

    This is a CRITICAL test for Adam correctness.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # First step (t=1):
    # m = 0.9 * 0 + 0.1 * 0.1 = 0.01
    # v = 0.999 * 0 + 0.001 * 0.01 = 0.00001
    # m_hat = 0.01 / (1 - 0.9) = 0.1
    # v_hat = 0.00001 / (1 - 0.999) = 0.01
    # update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 0.001
    var result = adam_step(
        params,
        grads,
        m,
        v,
        t=1,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    params = result[0]
    m = result[1]
    v = result[2]

    # Parameter should decrease from 1.0
    # Exact value ≈ 0.999 (1.0 - 0.001)
    assert_less(params._data.bitcast[Float32]()[0], 1.0)
    assert_almost_equal(
        params._data.bitcast[Float32]()[0], 0.999, tolerance=1e-3
    )


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run SGD and Adam initialization tests."""
    print("Running SGD tests...")
    test_sgd_initialization()
    test_sgd_basic_update()
    test_sgd_momentum_accumulation()
    test_sgd_weight_decay()
    test_sgd_nesterov_momentum()
    test_sgd_zero_grad()

    print("Running Adam tests...")
    test_adam_initialization()
    test_adam_parameter_update()

    print("\nAll optimizer part 1 tests passed! ✓")
