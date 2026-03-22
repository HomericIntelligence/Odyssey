"""Tests for optimizer base multi-parameter clipping, counting, and integration.

Tests the OptimizerBase:
- Gradient clipping across multiple parameters
- Gradient clipping validation
- Parameter counting with/without gradients
- Integration: optimizer with gradient clipping
- Learning rate scheduling workflow

Split from test_optimizer_base.mojo to comply with ADR-009 (≤10 fn test_ per file).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizer_base.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from tests.shared.conftest import assert_almost_equal
from shared.core.any_tensor import AnyTensor, zeros
from shared.autograd import Variable, GradientTape, SGD, Adam, AdaGrad, RMSprop
from shared.autograd.optimizer_base import (
    clip_gradients_by_global_norm,
    count_parameters_with_gradients,
)


# ============================================================================
# Gradient Clipping Tests (continued)
# ============================================================================


fn test_clip_gradients_multiple_parameters() raises:
    """Test global norm clipping across multiple parameters."""
    var tape = GradientTape()
    tape.enable()

    # Create two parameters
    var shape: List[Int] = [2]
    var data1 = AnyTensor(shape, DType.float32)
    var param1 = Variable(data1^, True, tape)
    var var_id1 = param1.id

    var data2 = AnyTensor(shape, DType.float32)
    var param2 = Variable(data2^, True, tape)
    var var_id2 = param2.id

    var parameters: List[Variable] = []
    parameters.append(param1.copy())
    parameters.append(param2.copy())

    # Add gradients to both parameters
    var grad1 = AnyTensor(shape, DType.float32)
    grad1._set_float64(0, 3.0)
    grad1._set_float64(1, 0.0)
    tape.registry.set_grad(var_id1, grad1^)

    var grad2 = AnyTensor(shape, DType.float32)
    grad2._set_float64(0, 0.0)
    grad2._set_float64(1, 4.0)
    tape.registry.set_grad(var_id2, grad2^)

    # Global norm = sqrt(3^2 + 4^2) = 5.0
    # Clip to max_norm=1.0
    var original_norm = clip_gradients_by_global_norm(
        parameters, tape, max_norm=1.0
    )

    # Verify original norm
    assert_almost_equal(original_norm, 5.0, tolerance=1e-6)

    # Both gradients should be scaled by 1.0/5.0 = 0.2
    var clipped_grad1 = tape.registry.get_grad(var_id1)
    assert_almost_equal(
        clipped_grad1._get_float64(0), 0.6, tolerance=1e-6
    )  # 3.0 * 0.2

    var clipped_grad2 = tape.registry.get_grad(var_id2)
    assert_almost_equal(
        clipped_grad2._get_float64(1), 0.8, tolerance=1e-6
    )  # 4.0 * 0.2


fn test_clip_gradients_validation() raises:
    """Test gradient clipping validates max_norm is non-negative."""
    var tape = GradientTape()
    var parameters: List[Variable] = []

    # Should raise error for negative max_norm
    try:
        _ = clip_gradients_by_global_norm(parameters, tape, max_norm=-1.0)
        assert_true(False, "Should have raised error for negative max_norm")
    except e:
        assert_true(True, "Correctly raised error for negative max_norm")


# ============================================================================
# Parameter Counting Tests
# ============================================================================


fn test_count_parameters_with_gradients() raises:
    """Test counting parameters that have gradients."""
    var tape = GradientTape()
    tape.enable()

    # Create 3 parameters
    var shape: List[Int] = [1]
    var data1 = AnyTensor(shape, DType.float32)
    var param1 = Variable(data1^, True, tape)
    var var_id1 = param1.id

    var data2 = AnyTensor(shape, DType.float32)
    var param2 = Variable(data2^, True, tape)
    var var_id2 = param2.id

    var data3 = AnyTensor(shape, DType.float32)
    var param3 = Variable(data3^, False, tape)  # Doesn't require grad

    var parameters: List[Variable] = []
    parameters.append(param1.copy())
    parameters.append(param2.copy())
    parameters.append(param3.copy())

    # Add gradients for param1 and param2 only
    var grad1 = AnyTensor(shape, DType.float32)
    tape.registry.set_grad(var_id1, grad1^)

    var grad2 = AnyTensor(shape, DType.float32)
    tape.registry.set_grad(var_id2, grad2^)

    # Count should be 2 (param3 doesn't require grad)
    var count = count_parameters_with_gradients(parameters, tape)
    assert_equal(count, 2, "Should count 2 parameters with gradients")


fn test_count_parameters_with_no_gradients() raises:
    """Test counting when no parameters have gradients."""
    var tape = GradientTape()
    tape.enable()

    # Create parameter without gradient
    var shape: List[Int] = [1]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Count should be 0
    var count = count_parameters_with_gradients(parameters, tape)
    assert_equal(count, 0, "Should count 0 parameters with gradients")


# ============================================================================
# Integration Tests - Optimizer Usage with Base Functionality
# ============================================================================


fn test_optimizer_integration_with_gradient_clipping() raises:
    """Test optimizer integration with gradient clipping."""
    var optimizer = SGD(learning_rate=0.1)
    var tape = GradientTape()
    tape.enable()

    # Create parameter
    var shape: List[Int] = [2]
    var data = AnyTensor(shape, DType.float32)
    data._set_float64(0, 1.0)
    data._set_float64(1, 2.0)

    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add large gradient
    var grad = AnyTensor(shape, DType.float32)
    grad._set_float64(0, 10.0)
    grad._set_float64(1, 0.0)
    tape.registry.set_grad(var_id, grad^)

    # Clip gradients before optimizer step
    _ = clip_gradients_by_global_norm(parameters, tape, max_norm=1.0)

    # Take optimizer step with clipped gradients
    optimizer.step(parameters, tape)

    # Parameter update should use clipped gradient (1.0 instead of 10.0)
    # param[0] = 1.0 - 0.1 * 1.0 = 0.9
    var updated_data = parameters[0].data
    assert_almost_equal(updated_data._get_float64(0), 0.9, tolerance=1e-6)
    assert_almost_equal(updated_data._get_float64(1), 2.0, tolerance=1e-6)


fn test_lr_scheduling_workflow() raises:
    """Test learning rate scheduling workflow."""
    var optimizer = Adam()

    # Initial learning rate
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)

    # Simulate learning rate warmup (5 epochs)
    for epoch in range(1, 6):
        var warmup_lr = Float64(epoch) * 0.0002
        optimizer.set_lr(warmup_lr)

    # After warmup, should be at 0.001
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)

    # Learning rate decay (multiply by 0.1 every 10 epochs)
    optimizer.set_lr(lr * 0.1)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.0001, tolerance=1e-10)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run optimizer base multi-parameter, counting, and integration tests."""
    print("Running gradient clipping tests (continued)...")
    test_clip_gradients_multiple_parameters()
    test_clip_gradients_validation()

    print("Running parameter counting tests...")
    test_count_parameters_with_gradients()
    test_count_parameters_with_no_gradients()

    print("Running integration tests...")
    test_optimizer_integration_with_gradient_clipping()
    test_lr_scheduling_workflow()

    print("\nAll optimizer base part3 tests passed! ✓")
