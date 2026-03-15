# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for optimizer implementations - Part 2 (Adam advanced, AdamW, RMSprop, properties).

Tests cover:
- Adam (Adaptive Moment Estimation) - bias correction and PyTorch validation
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)
- Property-based tests
- Numerical accuracy tests vs PyTorch

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
    create_test_vector,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like
from shared.training.optimizers.sgd import sgd_step, sgd_step_simple
from shared.training.optimizers.adam import adam_step, adam_step_simple
from shared.training.optimizers.adamw import adamw_step


# ============================================================================
# Adam Tests (continued)
# ============================================================================


fn test_adam_bias_correction() raises:
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
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

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


# ============================================================================
# AdamW Tests
# ============================================================================


fn test_adamw_weight_decay() raises:
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
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

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
    assert_less(param_val, 0.999)  # Weight decay should make it smaller than Adam alone


# ============================================================================
# RMSprop Tests
# ============================================================================


fn test_rmsprop_initialization() raises:
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


fn test_rmsprop_parameter_update() raises:
    """Test RMSprop performs correct parameter update.

    API Contract:
        RMSprop maintains moving average of squared gradients:
        - v = alpha * v + (1 - alpha) * grad^2
        - params = params - lr * grad / (sqrt(v) + epsilon).
    """
    # TODO(#1538): Implement when RMSprop is available
    # var params = ExTensor([1], DType.float32)
    # params._data.bitcast[Float32]()[0] = 1.0
    # var grads = ExTensor([1], DType.float32)
    # grads._data.bitcast[Float32]()[0] = 0.1
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


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_optimizer_property_decreasing_loss() raises:
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


fn test_optimizer_property_gradient_shape() raises:
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


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================


fn test_sgd_matches_pytorch() raises:
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
    params._data.bitcast[Float32]()[0] = 1.0
    params._data.bitcast[Float32]()[1] = 2.0
    params._data.bitcast[Float32]()[2] = 3.0

    # Gradients
    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1
    grads._data.bitcast[Float32]()[1] = 0.2
    grads._data.bitcast[Float32]()[2] = 0.3

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


fn test_adam_matches_pytorch() raises:
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
    params._data.bitcast[Float32]()[0] = 1.0
    params._data.bitcast[Float32]()[1] = 2.0
    params._data.bitcast[Float32]()[2] = 3.0

    # Gradients
    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1
    grads._data.bitcast[Float32]()[1] = 0.2
    grads._data.bitcast[Float32]()[2] = 0.3

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


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run Adam advanced, AdamW, RMSprop, property, and numerical accuracy tests.
    """
    print("Running Adam advanced tests...")
    test_adam_bias_correction()

    print("Running AdamW tests...")
    test_adamw_weight_decay()

    print("Running RMSprop tests...")
    test_rmsprop_initialization()
    test_rmsprop_parameter_update()

    print("Running property-based tests...")
    test_optimizer_property_decreasing_loss()
    test_optimizer_property_gradient_shape()

    print("Running numerical accuracy tests...")
    test_sgd_matches_pytorch()
    test_adam_matches_pytorch()

    print("\nAll optimizer part 2 tests passed! ✓")
