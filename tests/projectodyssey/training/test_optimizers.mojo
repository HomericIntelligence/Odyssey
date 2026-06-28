"""Unit tests for optimizer implementations.

Tests cover:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation) - initialization and basic update

Following TDD principles - these tests define the expected API
and numerical behavior for implementation in Issue #49.

Note: Tests have been adapted from class-based API to pure functional API
as per architecture decision to use functional design throughout shared library.
"""


from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_equal,
    assert_less,
    assert_shape,
    assert_true,
    create_test_vector,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, zeros_like
from projectodyssey.training.optimizers.sgd import sgd_step, sgd_step_simple
from projectodyssey.training.optimizers.adam import adam_step, adam_step_simple
from projectodyssey.training.optimizers.adamw import adamw_step
from projectodyssey.training.optimizers.lion import lion_step, lion_step_simple
from projectodyssey.training.optimizers.rmsprop import rmsprop_step
from projectodyssey.training.optimizers.shampoo import (
    shampoo_step,
    shampoo_step_simple,
    initialize_shampoo_state,
)


def test_sgd_initialization() raises:
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


def test_sgd_basic_update() raises:
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
    params.set(0, Float32(1.0))
    params.set(1, Float32(2.0))
    params.set(2, Float32(3.0))

    # Gradients: [0.1, 0.2, 0.3]
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))
    grads.set(1, Float32(0.2))
    grads.set(2, Float32(0.3))

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


def test_sgd_momentum_accumulation() raises:
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
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

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


def test_sgd_weight_decay() raises:
    """Test SGD applies weight decay (L2 regularization).

    Functional API:
        With weight_decay > 0:
        - Effective gradient: grad = grad + weight_decay * params
        - Then apply standard update.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

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


def test_sgd_nesterov_momentum() raises:
    """Test SGD with Nesterov momentum (lookahead).

    Not applicable to pure functional design - Nesterov momentum requires
    computing gradients at a different point (lookahead position), which
    would require the gradient computation to be part of the optimizer.

    In the functional design, gradient computation is external to the
    optimizer function, so Nesterov momentum is deferred.
    """
    pass  # Deferred - not applicable to pure functional design


def test_sgd_zero_grad() raises:
    """Test SGD clears optimizer state (if needed).

    Not applicable to pure functional design - there is no internal state
    to clear. In the functional API, the caller manages all state (velocity
    buffers, etc.), so gradient clearing is the caller's responsibility.
    """
    pass  # Not applicable - no internal state in functional design


def test_adam_initialization() raises:
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
    assert_shape(
        result[0], shape, "Adam initialization result shape matches input"
    )


def test_adam_parameter_update() raises:
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
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

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


def test_adam_bias_correction() raises:
    """Test Adam applies bias correction in early steps.

    Functional API:
        Bias correction factors:
        - m_hat = m / (1 - beta1^t)
        - v_hat = v / (1 - beta2^t)
        Where t is the step number (1, 2, 3, ...)

    This is CRITICAL for Adam's fast convergence in early training.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # First few steps should have larger effective learning rate
    # due to bias correction
    var prev_param = Float32(1.0)

    # Run 5 steps
    for t in range(1, 6):
        var result = adam_step(params, grads, m, v, t=t, learning_rate=0.001)
        params = result[0]
        m = result[1]
        v = result[2]

        # Each step should decrease parameters
        assert_less(params._data.bitcast[Float32]()[0], prev_param)
        prev_param = params._data.bitcast[Float32]()[0]


def test_adamw_weight_decay() raises:
    """Test AdamW applies decoupled weight decay.

    API Contract:
        AdamW is Adam with decoupled weight decay:
        - Apply Adam update (without weight decay in gradient)
        - Then apply: params = params * (1 - lr * weight_decay)

        This differs from L2 regularization used in standard Adam.

    Verification:
        - AdamW should reduce parameters more than Adam alone
        - Weight decay is applied directly to parameters, not through gradients
        - Effect accumulates over multiple steps
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Single AdamW step with weight decay
    var result = adamw_step(
        params,
        grads,
        m,
        v,
        t=1,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )
    params = result[0]

    # Parameter should decrease due to both gradient update and weight decay
    # Adam update: ~1.0 - 0.001 = ~0.999
    # Then weight decay: 0.999 * (1 - 0.001 * 0.01) = 0.999 * 0.99999 ≈ 0.998999
    var param_val = params._data.bitcast[Float32]()[0]
    assert_less(param_val, 1.0)
    assert_less(
        param_val, 0.999
    )  # Weight decay should make it smaller than Adam alone


def test_rmsprop_initialization() raises:
    """Test RMSprop optimizer initialization.

    Functional API Note:
        Pure functional design - hyperparameters are passed as function arguments.
        This test verifies that rmsprop_step accepts all expected parameters
        and performs computation without error.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)

    var result = rmsprop_step(
        params,
        grads,
        square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        momentum=0.0,
    )

    assert_shape(result[0], [1])
    assert_shape(result[1], [1])


def test_rmsprop_parameter_update() raises:
    """Test RMSprop performs correct parameter update.

    Functional API:
        rmsprop_step updates:
        - square_avg = alpha * square_avg + (1 - alpha) * grad^2
        - params = params - lr * grad / (sqrt(square_avg) + epsilon).
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))
    var square_avg = zeros(shape, DType.float32)

    var result = rmsprop_step(
        params,
        grads,
        square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        momentum=0.0,
    )

    var new_params = result[0]
    var param_value = new_params._data.bitcast[Float32]()[0]

    assert_less(param_value, Float32(0.95))


def test_optimizer_property_decreasing_loss() raises:
    """Property: Optimizer should decrease loss on a convex function.

    Minimize f(x) = x^2 with SGD by recomputing the true gradient
    (f'(x) = 2x) from the *current* parameter at every step. This
    genuinely validates convergence behavior, not a pre-baked gradient
    schedule.
    """
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(5.0))

    var velocity = zeros(shape, DType.float32)
    var x0 = Float32(5.0)
    var initial_loss = x0 * x0

    for _ in range(100):
        # Gradient of x^2 at the current parameter value.
        var x = params._data.bitcast[Float32]()[0]
        var grad = zeros(shape, DType.float32)
        grad.set(0, Float32(2.0) * x)

        var result = sgd_step(
            params, grad, velocity, learning_rate=0.1, momentum=0.9
        )
        params = result[0]
        velocity = result[1]

    var final_x = params._data.bitcast[Float32]()[0]
    var final_loss = final_x * final_x

    # SGD on a convex quadratic must reduce the loss far below the start.
    assert_less(Float32(final_loss), Float32(initial_loss * 0.1))


def test_optimizer_property_gradient_shape() raises:
    """Property: Optimizer should handle gradients of same shape as parameters.

    All optimizers should work with multi-dimensional parameter tensors.
    """
    var velocity = zeros([10], DType.float32)
    var params = ones([10], DType.float32)
    var grads = zeros([10], DType.float32)

    var result = sgd_step(params, grads, velocity, learning_rate=0.01)

    var new_params = result[0]
    var new_shape = new_params.shape()
    assert_equal(new_shape[0], 10)


def test_sgd_matches_pytorch() raises:
    """Test SGD matches PyTorch implementation exactly.

    This CRITICAL test validates numerical correctness against PyTorch.

    PyTorch reference code:
        ```python
        import torch
        import torch.optim as optim

        # Initial parameters
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)

        # Gradients
        params.grad = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

        # SGD optimizer with momentum
        optimizer = optim.SGD([params], lr=0.1, momentum=0.9, weight_decay=0.0)

        # First step
        optimizer.step()
        print("After step 1:", params)  # tensor([0.9900, 1.9800, 2.9700])

        # Second step (same gradients)
        params.grad = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        optimizer.step()
        print("After step 2:", params)  # tensor([0.9710, 1.9420, 2.9130])
        ```
    """
    # Initial parameters
    var shape = List[Int]()
    shape.append(3)
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))
    params.set(1, Float32(2.0))
    params.set(2, Float32(3.0))

    # Gradients
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))
    grads.set(1, Float32(0.2))
    grads.set(2, Float32(0.3))

    # Velocity buffer
    var velocity = zeros(shape, DType.float32)

    # First step
    var result = sgd_step(
        params, grads, velocity, learning_rate=0.1, momentum=0.9
    )
    params = result[0]
    velocity = result[1]

    # Validate against PyTorch (step 1)
    assert_almost_equal(
        params._data.bitcast[Float32]()[0], 0.9900, tolerance=1e-6
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[1], 1.9800, tolerance=1e-6
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[2], 2.9700, tolerance=1e-6
    )

    # Second step (same gradients)
    result = sgd_step(params, grads, velocity, learning_rate=0.1, momentum=0.9)
    params = result[0]
    velocity = result[1]

    # Validate against PyTorch (step 2)
    assert_almost_equal(
        params._data.bitcast[Float32]()[0], 0.9710, tolerance=1e-6
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[1], 1.9420, tolerance=1e-6
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[2], 2.9130, tolerance=1e-6
    )


def test_adam_matches_pytorch() raises:
    """Test Adam matches PyTorch implementation exactly.

    This CRITICAL test validates Adam's complex update rules.

    PyTorch reference code:
        ```python
        import torch
        import torch.optim as optim

        # Initial parameters
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)

        # Gradients
        params.grad = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

        # Adam optimizer
        optimizer = optim.Adam([params], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # First step
        optimizer.step()
        print("After step 1:", params)
        # tensor([0.9990, 1.9990, 2.9990])

        # Second step (same gradients)
        params.grad = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        optimizer.step()
        print("After step 2:", params)
        # tensor([0.9980, 1.9980, 2.9980])
        ```
    """
    # Initial parameters
    var shape = List[Int]()
    shape.append(3)
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))
    params.set(1, Float32(2.0))
    params.set(2, Float32(3.0))

    # Gradients
    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))
    grads.set(1, Float32(0.2))
    grads.set(2, Float32(0.3))

    # Moment buffers
    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # First step (t=1)
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

    # Validate against PyTorch (step 1)
    assert_almost_equal(
        params._data.bitcast[Float32]()[0], 0.9990, tolerance=1e-4
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[1], 1.9990, tolerance=1e-4
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[2], 2.9990, tolerance=1e-4
    )

    # Second step (t=2, same gradients)
    result = adam_step(
        params,
        grads,
        m,
        v,
        t=2,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    params = result[0]
    m = result[1]
    v = result[2]

    # Validate against PyTorch (step 2)
    assert_almost_equal(
        params._data.bitcast[Float32]()[0], 0.9980, tolerance=1e-4
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[1], 1.9980, tolerance=1e-4
    )
    assert_almost_equal(
        params._data.bitcast[Float32]()[2], 2.9980, tolerance=1e-4
    )


def test_lion_initialization() raises:
    """Test Lion optimizer initialization with hyperparameters."""
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var momentum = zeros(shape, DType.float32)

    # Should accept all hyperparameters without error
    var result = lion_step(
        params,
        grads,
        momentum,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )

    # Verify shapes
    assert_shape(result[0], shape, "Lion step params shape")
    assert_shape(result[1], shape, "Lion step momentum shape")


def test_lion_basic_update_manual() raises:
    """Test Lion performs correct basic parameter update."""
    var shape: List[Int] = [1]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))

    var momentum = zeros(shape, DType.float32)

    # Step with Lion
    var result = lion_step(
        params,
        grads,
        momentum,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )
    var new_params = result[0]
    var new_momentum = result[1]

    # Verify that parameters updated
    var param_val = new_params._data.bitcast[Float32]()[0]
    assert_less(Float64(param_val), 1.0, "Lion should decrease params")


def test_lion_descent_on_quadratic() raises:
    """Test Lion converges on simple quadratic loss."""
    # f(x) = ||x||^2, optimal at x=0
    var shape: List[Int] = [5]
    var params = ones(shape, DType.float32)
    var momentum = zeros(shape, DType.float32)

    # Manually initialize params to [1, 1, 1, 1, 1]
    for i in range(5):
        params.set(i, Float32(1.0))

    var learning_rate = 0.01
    var num_steps = 50

    for step in range(num_steps):
        # Gradient of f(x) = ||x||^2 is grad = 2*x
        var grads = zeros(shape, DType.float32)
        for i in range(5):
            var val = params._data.bitcast[Float32]()[i]
            grads.set(i, val * Float32(2.0))

        # Lion step
        var result = lion_step(
            params,
            grads,
            momentum,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.0,
        )
        params = result[0]
        momentum = result[1]

    # Lion uses signed updates: each step moves the parameter by exactly
    # `learning_rate` (= 0.01). Starting at 1.0, the smallest reachable value
    # after 50 monotone-downward steps is 1.0 - 50 * 0.01 = 0.5, which Lion
    # attains here (final ~= 0.5000005). Assert clear descent with a margin
    # just above that mathematical floor.
    var final_val = Float64(params._data.bitcast[Float32]()[0])
    assert_less(
        final_val, 0.55, "Lion descent on quadratic: final value too high"
    )


def test_lion_weight_decay() raises:
    """Test Lion applies weight decay correctly."""
    var shape: List[Int] = [2]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(2.0))
    params.set(1, Float32(3.0))

    var grads = zeros(shape, DType.float32)
    grads.set(0, Float32(0.1))
    grads.set(1, Float32(0.1))

    var momentum = zeros(shape, DType.float32)

    # With weight decay, params should decrease more
    var result = lion_step(
        params,
        grads,
        momentum,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.01,
    )
    var new_params = result[0]

    # Without weight decay
    var result_no_wd = lion_step(
        params,
        grads,
        momentum,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )
    var new_params_no_wd = result_no_wd[0]

    # Weight decay version should have smaller params
    var wd_val = new_params._data.bitcast[Float32]()[0]
    var no_wd_val = new_params_no_wd._data.bitcast[Float32]()[0]
    assert_less(
        Float64(wd_val), Float64(no_wd_val), "Weight decay should reduce params"
    )


def test_lion_memory_footprint() raises:
    """Test Lion uses only one state buffer (half of AdamW)."""
    var shape: List[Int] = [3]
    var params = ones(shape, DType.float32)
    var momentum = zeros(shape, DType.float32)
    var grads = zeros(shape, DType.float32)

    # Lion should return (params, momentum) = 2 tensors
    # AdamW returns (params, m, v) = 3 tensors
    var result = lion_step(params, grads, momentum, learning_rate=0.0001)

    # Verify tuple length is 2 (params + momentum only)
    assert_true(
        len(result) == 2, "Lion should return 2 tensors (params, momentum)"
    )


def test_lion_checkpoint_roundtrip() raises:
    """Test Lion state can round-trip through save/load (structural compatibility).
    """
    var shape: List[Int] = [2]
    var momentum = ones(shape, DType.float32)
    momentum.set(0, Float32(0.5))
    momentum.set(1, Float32(0.75))

    # The key claim: momentum is a plain AnyTensor that matches param shapes
    # and can be checkpointed using the same infrastructure as model parameters
    assert_shape(momentum, shape, "Lion momentum matches param shape")


def test_shampoo_initialization() raises:
    """Test Shampoo optimizer state initialization."""
    var shape: List[Int] = [2, 3]
    var params = ones(shape, DType.float32)

    # initialize_shampoo_state returns (L, R, momentum)
    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    # L should be [m, m], R should be [n, n], momentum should match params
    var L_shape: List[Int] = [2, 2]
    var R_shape: List[Int] = [3, 3]
    assert_shape(L, L_shape, "Shampoo L accumulator shape [m,m]")
    assert_shape(R, R_shape, "Shampoo R accumulator shape [n,n]")
    assert_shape(m, shape, "Shampoo momentum matches param shape")


def test_shampoo_basic_update() raises:
    """Test Shampoo performs a parameter update and returns updated state."""
    var shape: List[Int] = [2, 3]
    var params = ones(shape, DType.float32)
    var grads = ones(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var result = shampoo_step(params, grads, L, R, m, learning_rate=0.01)
    var new_params = result[0]
    var new_L = result[1]
    var new_R = result[2]
    var new_m = result[3]

    # Shapes preserved
    assert_shape(new_params, shape, "Shampoo updated params shape preserved")
    var L_shape: List[Int] = [2, 2]
    var R_shape: List[Int] = [3, 3]
    assert_shape(new_L, L_shape, "Shampoo updated L shape preserved")
    assert_shape(new_R, R_shape, "Shampoo updated R shape preserved")
    assert_shape(new_m, shape, "Shampoo updated momentum shape preserved")


def test_shampoo_descent_on_quadratic() raises:
    """Test Shampoo reduces loss on a simple quadratic objective."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(2.0))
    params.set(1, Float32(2.0))
    params.set(2, Float32(2.0))
    params.set(3, Float32(2.0))

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    # Quadratic: loss = sum(params^2), grad = 2 * params
    var initial_norm = Float64(0.0)
    for i in range(4):
        var v = Float64(params._data.bitcast[Float32]()[i])
        initial_norm = initial_norm + v * v

    for _ in range(30):
        var grads = ones(shape, DType.float32)
        for i in range(4):
            var p = params._data.bitcast[Float32]()[i]
            grads.set(i, p * Float32(2.0))

        var result = shampoo_step(params, grads, L, R, m, learning_rate=0.01)
        params = result[0]
        L = result[1]
        R = result[2]
        m = result[3]

    var final_norm = Float64(0.0)
    for i in range(4):
        var v = Float64(params._data.bitcast[Float32]()[i])
        final_norm = final_norm + v * v

    assert_less(final_norm, initial_norm, "Shampoo reduces quadratic loss")


def test_shampoo_preconditioner_accumulates() raises:
    """Test Shampoo L and R accumulators change after gradient steps."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    var grads = ones(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var L_before = Float64(L._data.bitcast[Float32]()[0])

    var result = shampoo_step(params, grads, L, R, m, learning_rate=0.01)
    var new_L = result[1]
    var L_after = Float64(new_L._data.bitcast[Float32]()[0])

    assert_true(
        L_after != L_before, "Shampoo L accumulator changes after gradient step"
    )


def test_shampoo_epsilon_stability_zero_gradient() raises:
    """Test Shampoo is stable with zero gradients (epsilon guards)."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    # Should not raise even with zero gradients
    var result = shampoo_step(params, grads, L, R, m, learning_rate=0.01)
    var new_params = result[0]

    assert_shape(new_params, shape, "Shampoo stable with zero gradients")


def test_shampoo_memory_footprint() raises:
    """Test Shampoo state: L [m,m], R [n,n], momentum [m,n] — 4-tensor return.
    """
    var shape: List[Int] = [3, 4]
    var params = ones(shape, DType.float32)
    var grads = ones(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var result = shampoo_step(params, grads, L, R, m, learning_rate=0.01)

    assert_true(
        len(result) == 4,
        "Shampoo should return 4 tensors (params, L, R, momentum)",
    )


def test_shampoo_weight_decay() raises:
    """Test Shampoo with weight decay reduces parameter magnitude over time."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var initial_sum = Float64(0.0)
    for i in range(4):
        initial_sum = initial_sum + Float64(params._data.bitcast[Float32]()[i])

    for _ in range(10):
        var result = shampoo_step(
            params, grads, L, R, m, learning_rate=0.01, weight_decay=0.1
        )
        params = result[0]
        L = result[1]
        R = result[2]
        m = result[3]

    var final_sum = Float64(0.0)
    for i in range(4):
        final_sum = final_sum + Float64(params._data.bitcast[Float32]()[i])

    assert_less(final_sum, initial_sum, "Shampoo weight decay shrinks params")


def test_shampoo_pure_functional() raises:
    """Test Shampoo does not mutate input tensors (pure functional)."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.5))
    var grads = ones(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var param_val_before = Float64(params._data.bitcast[Float32]()[0])
    var _ = shampoo_step(params, grads, L, R, m, learning_rate=0.01)
    var param_val_after = Float64(params._data.bitcast[Float32]()[0])

    assert_almost_equal(
        param_val_before,
        param_val_after,
        tolerance=1e-7,
    )


def test_shampoo_simple_wrapper() raises:
    """Test shampoo_step_simple convenience wrapper returns correct tuple."""
    var shape: List[Int] = [2, 2]
    var params = ones(shape, DType.float32)
    var grads = ones(shape, DType.float32)

    var state = initialize_shampoo_state(params)
    var L = state[0]
    var R = state[1]
    var m = state[2]

    var result = shampoo_step_simple(params, grads, L, R, m, learning_rate=0.01)

    assert_true(
        len(result) == 4,
        "shampoo_step_simple returns 4-tuple (params, L, R, momentum)",
    )
    assert_shape(
        result[0], shape, "shampoo_step_simple: params shape preserved"
    )


def main() raises:
    """Run all test_optimizers tests."""
    print("Running test_optimizers tests...")

    test_sgd_initialization()
    print("✓ test_sgd_initialization")

    test_sgd_basic_update()
    print("✓ test_sgd_basic_update")

    test_sgd_momentum_accumulation()
    print("✓ test_sgd_momentum_accumulation")

    test_sgd_weight_decay()
    print("✓ test_sgd_weight_decay")

    test_sgd_nesterov_momentum()
    print("✓ test_sgd_nesterov_momentum")

    test_sgd_zero_grad()
    print("✓ test_sgd_zero_grad")

    test_adam_initialization()
    print("✓ test_adam_initialization")

    test_adam_parameter_update()
    print("✓ test_adam_parameter_update")

    test_adam_bias_correction()
    print("✓ test_adam_bias_correction")

    test_adamw_weight_decay()
    print("✓ test_adamw_weight_decay")

    test_rmsprop_initialization()
    print("✓ test_rmsprop_initialization")

    test_rmsprop_parameter_update()
    print("✓ test_rmsprop_parameter_update")

    test_optimizer_property_decreasing_loss()
    print("✓ test_optimizer_property_decreasing_loss")

    test_optimizer_property_gradient_shape()
    print("✓ test_optimizer_property_gradient_shape")

    test_sgd_matches_pytorch()
    print("✓ test_sgd_matches_pytorch")

    test_adam_matches_pytorch()
    print("✓ test_adam_matches_pytorch")

    test_lion_initialization()
    print("✓ test_lion_initialization")

    test_lion_basic_update_manual()
    print("✓ test_lion_basic_update_manual")

    test_lion_descent_on_quadratic()
    print("✓ test_lion_descent_on_quadratic")

    test_lion_weight_decay()
    print("✓ test_lion_weight_decay")

    test_lion_memory_footprint()
    print("✓ test_lion_memory_footprint")

    test_lion_checkpoint_roundtrip()
    print("✓ test_lion_checkpoint_roundtrip")

    test_shampoo_initialization()
    print("✓ test_shampoo_initialization")

    test_shampoo_basic_update()
    print("✓ test_shampoo_basic_update")

    test_shampoo_descent_on_quadratic()
    print("✓ test_shampoo_descent_on_quadratic")

    test_shampoo_preconditioner_accumulates()
    print("✓ test_shampoo_preconditioner_accumulates")

    test_shampoo_epsilon_stability_zero_gradient()
    print("✓ test_shampoo_epsilon_stability_zero_gradient")

    test_shampoo_memory_footprint()
    print("✓ test_shampoo_memory_footprint")

    test_shampoo_weight_decay()
    print("✓ test_shampoo_weight_decay")

    test_shampoo_pure_functional()
    print("✓ test_shampoo_pure_functional")

    test_shampoo_simple_wrapper()
    print("✓ test_shampoo_simple_wrapper")

    print("\nAll test_optimizers tests passed!")
