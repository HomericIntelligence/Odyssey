"""Tests for optimizer base learning rate functionality.

Tests the OptimizerBase learning rate get/set methods and validation:
- Learning rate get/set for SGD, Adam, AdaGrad, RMSprop
- Learning rate validation


# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from std.testing import (
    assert_equal,
    assert_true,
)
from tests.projectodyssey.conftest import assert_almost_equal
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
)
from projectodyssey.autograd.variable import Variable
from projectodyssey.autograd.tape import GradientTape
from projectodyssey.autograd.optimizers import SGD, Adam, AdaGrad, RMSprop
from projectodyssey.autograd.optimizer_base import (
    clip_gradients_by_global_norm,
    count_parameters_with_gradients,
    validate_learning_rate,
    zero_grad_impl,
)


def test_sgd_get_set_lr() raises:
    """Test SGD learning rate get/set methods."""
    var optimizer = SGD(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.001)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)


def test_adam_get_set_lr() raises:
    """Test Adam learning rate get/set methods."""
    var optimizer = Adam(learning_rate=0.001)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.0001)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.0001, tolerance=1e-10)


def test_adagrad_get_set_lr() raises:
    """Test AdaGrad learning rate get/set methods."""
    var optimizer = AdaGrad(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.005)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.005, tolerance=1e-10)


def test_rmsprop_get_set_lr() raises:
    """Test RMSprop learning rate get/set methods."""
    var optimizer = RMSprop(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.02)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.02, tolerance=1e-10)


def test_set_lr_validation() raises:
    """Test that set_lr validates learning rate is positive."""
    var optimizer = SGD(learning_rate=0.01)

    # Should raise error for non-positive learning rate
    try:
        optimizer.set_lr(0.0)
        assert_true(False, "Should have raised error for lr=0.0")
    except e:
        assert_true(True, "Correctly raised error for lr=0.0")

    try:
        optimizer.set_lr(-0.01)
        assert_true(False, "Should have raised error for negative lr")
    except e:
        assert_true(True, "Correctly raised error for negative lr")


def test_validate_learning_rate_function() raises:
    """Test the validate_learning_rate utility function."""
    # Should not raise for positive values
    validate_learning_rate(0.01)
    validate_learning_rate(0.001)
    validate_learning_rate(1.0)

    # Should raise for non-positive values
    try:
        validate_learning_rate(0.0)
        assert_true(False, "Should have raised error for lr=0.0")
    except e:
        assert_true(True, "Correctly raised error for lr=0.0")

    try:
        validate_learning_rate(-0.01)
        assert_true(False, "Should have raised error for negative lr")
    except e:
        assert_true(True, "Correctly raised error for negative lr")


def test_zero_grad_implementation() raises:
    """Test that zero_grad clears the gradient tape."""
    var tape = GradientTape()

    # Create a variable to register with the tape (gets auto-assigned ID)
    var shape: List[Int] = [1]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add a gradient for this variable
    var grad = AnyTensor(shape, DType.float32)
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


def test_sgd_zero_grad() raises:
    """Test SGD zero_grad method clears tape."""
    var optimizer = SGD(learning_rate=0.01)
    var tape = GradientTape()

    # Create a variable to register with tape
    var shape: List[Int] = [1]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add gradient
    var grad = AnyTensor(shape, DType.float32)
    tape.registry.set_grad(var_id, grad^)

    # Clear gradients
    optimizer.zero_grad(tape)

    # Verify cleared
    assert_true(
        not tape.registry.has_gradient(var_id),
        "SGD zero_grad should clear tape",
    )


def test_adam_zero_grad() raises:
    """Test Adam zero_grad method clears tape."""
    var optimizer = Adam()
    var tape = GradientTape()

    # Create a variable to register with tape
    var shape: List[Int] = [1]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id

    # Add gradient
    var grad = AnyTensor(shape, DType.float32)
    tape.registry.set_grad(var_id, grad^)

    # Clear gradients
    optimizer.zero_grad(tape)

    # Verify cleared
    assert_true(
        not tape.registry.has_gradient(var_id),
        "Adam zero_grad should clear tape",
    )


def test_zero_grad_preserves_optimizer_state() raises:
    """Test zero_grad clears tape but preserves optimizer internal state."""
    var optimizer = Adam()
    var tape = GradientTape()
    tape.enable()

    # Create a parameter
    var shape: List[Int] = [2]
    var data = AnyTensor(shape, DType.float32)
    data._set_float64(0, 1.0)
    data._set_float64(1, 2.0)

    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add gradient
    var grad = AnyTensor(shape, DType.float32)
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


def test_clip_gradients_no_clipping_needed() raises:
    """Test gradient clipping when norm is below threshold."""
    var tape = GradientTape()
    tape.enable()

    # Create parameter
    var shape: List[Int] = [3]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add small gradient (norm < 5.0)
    var grad = AnyTensor(shape, DType.float32)
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


def test_clip_gradients_with_clipping() raises:
    """Test gradient clipping when norm exceeds threshold."""
    var tape = GradientTape()
    tape.enable()

    # Create parameter
    var shape: List[Int] = [3]
    var data = AnyTensor(shape, DType.float32)
    var param = Variable(data^, True, tape)
    var var_id = param.id
    var parameters: List[Variable] = []
    parameters.append(param.copy())

    # Add large gradient (norm > 1.0)
    var grad = AnyTensor(shape, DType.float32)
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


def test_clip_gradients_multiple_parameters() raises:
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


def test_clip_gradients_validation() raises:
    """Test gradient clipping validates max_norm is non-negative."""
    var tape = GradientTape()
    var parameters: List[Variable] = []

    # Should raise error for negative max_norm
    try:
        _ = clip_gradients_by_global_norm(parameters, tape, max_norm=-1.0)
        assert_true(False, "Should have raised error for negative max_norm")
    except e:
        assert_true(True, "Correctly raised error for negative max_norm")


def test_count_parameters_with_gradients() raises:
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


def test_count_parameters_with_no_gradients() raises:
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


def test_optimizer_integration_with_gradient_clipping() raises:
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


def test_lr_scheduling_workflow() raises:
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


def main() raises:
    """Run all test_optimizer_base tests."""
    print("Running test_optimizer_base tests...")

    test_sgd_get_set_lr()
    print("✓ test_sgd_get_set_lr")

    test_adam_get_set_lr()
    print("✓ test_adam_get_set_lr")

    test_adagrad_get_set_lr()
    print("✓ test_adagrad_get_set_lr")

    test_rmsprop_get_set_lr()
    print("✓ test_rmsprop_get_set_lr")

    test_set_lr_validation()
    print("✓ test_set_lr_validation")

    test_validate_learning_rate_function()
    print("✓ test_validate_learning_rate_function")

    test_zero_grad_implementation()
    print("✓ test_zero_grad_implementation")

    test_sgd_zero_grad()
    print("✓ test_sgd_zero_grad")

    test_adam_zero_grad()
    print("✓ test_adam_zero_grad")

    test_zero_grad_preserves_optimizer_state()
    print("✓ test_zero_grad_preserves_optimizer_state")

    test_clip_gradients_no_clipping_needed()
    print("✓ test_clip_gradients_no_clipping_needed")

    test_clip_gradients_with_clipping()
    print("✓ test_clip_gradients_with_clipping")

    test_clip_gradients_multiple_parameters()
    print("✓ test_clip_gradients_multiple_parameters")

    test_clip_gradients_validation()
    print("✓ test_clip_gradients_validation")

    test_count_parameters_with_gradients()
    print("✓ test_count_parameters_with_gradients")

    test_count_parameters_with_no_gradients()
    print("✓ test_count_parameters_with_no_gradients")

    test_optimizer_integration_with_gradient_clipping()
    print("✓ test_optimizer_integration_with_gradient_clipping")

    test_lr_scheduling_workflow()
    print("✓ test_lr_scheduling_workflow")

    print("\nAll test_optimizer_base tests passed!")
