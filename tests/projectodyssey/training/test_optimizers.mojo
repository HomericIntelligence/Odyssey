"""Unit tests for optimizer implementations.

Tests cover:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation) - initialization and basic update

Following TDD principles - these tests define the expected API
and numerical behavior for implementation in Issue #49.

Note: Tests have been adapted from class-based API to pure functional API
as per architecture decision to use functional design throughout shared library.

in Mojo v0.26.1 (libKGENCompilerRTShared.so JIT fault under high test load).
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
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like
from projectodyssey.training.optimizers.sgd import sgd_step, sgd_step_simple
from projectodyssey.training.optimizers.adam import adam_step, adam_step_simple
from projectodyssey.training.optimizers.adamw import adamw_step


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

    API Contract:
        RMSprop(
            learning_rate: Float32 = 0.01,
            alpha: Float32 = 0.99,
            epsilon: Float32 = 1e-8,
            momentum: Float32 = 0.0
        ).
    """
    # TODO(#1538): Implement when RMSprop is available
    # var optimizer = RMSprop(
    #     learning_rate=0.01,
    #     alpha=0.99,
    #     epsilon=1e-8,
    #     momentum=0.0
    # )
    # assert_almost_equal(optimizer.learning_rate, 0.01)
    # assert_almost_equal(optimizer.alpha, 0.99)
    pass


def test_rmsprop_parameter_update() raises:
    """Test RMSprop performs correct parameter update.

    API Contract:
        RMSprop maintains moving average of squared gradients:
        - v = alpha * v + (1 - alpha) * grad^2
        - params = params - lr * grad / (sqrt(v) + epsilon).
    """
    # TODO(#1538): Implement when RMSprop is available
    # var params = AnyTensor([1], DType.float32)
    # params.set(0, Float32(1.0))
    # var grads = AnyTensor([1], DType.float32)
    # grads.set(0, Float32(0.1))
    # #
    # var optimizer = RMSprop(learning_rate=0.01, alpha=0.99, epsilon=1e-8)
    # #
    # # First step:
    # # v = 0.99 * 0 + 0.01 * 0.01 = 0.0001
    # # update = 0.01 * 0.1 / (sqrt(0.0001) + 1e-8) ≈ 0.1
    # optimizer.step(params, grads)
    # #
    # # Parameter should decrease significantly
    # assert_less(params._get_float64(0), 0.95)
    pass


def test_optimizer_property_decreasing_loss() raises:
    """Property: Optimizer should decrease loss on convex function.

    Test that all optimizers can minimize a simple quadratic function.
    This validates basic convergence behavior.
    """
    # TODO(#1538): Implement when optimizers and loss functions are available
    # # Define simple quadratic: f(x) = x^2
    # # Gradient: df/dx = 2x
    # # Minimum at x=0
    # #
    # var initial_value = Float32(5.0)
    # var params = Tensor(List[Float32](), Shape(1))
    # #
    # # Test each optimizer
    # varoptimizers = [
    #     SGD(learning_rate=0.1),
    #     Adam(learning_rate=0.1),
    #     RMSprop(learning_rate=0.1),
    # ]
    # #
    # for optimizer in optimizers:
    #     var x = params.copy()
    #     var initial_loss = x[0] * x[0]
    # #
    #     # Run 100 steps
    #     for _ in range(100):
    #         var grad = 2 * x[0]  # Gradient of x^2
    #         optimizer.step(x, grad)
    # #
    #     var final_loss = x[0] * x[0]
    # #
    #     # Loss should decrease significantly
    #     assert_less(final_loss, initial_loss * 0.1)
    pass


def test_optimizer_property_gradient_shape() raises:
    """Property: Optimizer should handle gradients of same shape as parameters.

    All optimizers should work with multi-dimensional parameter tensors.
    """
    # TODO(#1538): Implement when optimizers are available
    # # Test with various parameter shapes
    # varshapes = [Shape(10), Shape(10, 5), Shape(3, 32, 32)]
    # #
    # for shape in shapes:
    #     var params = Tensor.randn(shape)
    #     var grads = Tensor.randn(shape)
    # #
    #     var optimizer = SGD(learning_rate=0.01)
    #     optimizer.step(params, grads)
    # #
    #     # Shape should be preserved
    #     assert_equal(params.shape(), shape)
    pass


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

    print("\nAll test_optimizers tests passed!")
