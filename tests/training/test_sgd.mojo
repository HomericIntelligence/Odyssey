"""Tests for SGD optimizer.

This module tests the SGD optimizer implementations:
- sgd_step_simple (basic SGD)
- sgd_step (SGD with momentum and weight decay)
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import subtract, multiply
from shared.training.optimizers import sgd_step_simple, sgd_step


fn test_sgd_step_simple_basic() raises:
    """Test basic SGD parameter update."""
    print("Testing basic SGD step...")

    var shape = List[Int]()
    shape.append(3)
    var params = ExTensor(shape, DType.float32)
    var gradients = ExTensor(shape, DType.float32)

    # Initial params: [1, 2, 3]
    params._set_float64(0, 1.0)
    params._set_float64(1, 2.0)
    params._set_float64(2, 3.0)

    # Gradients: [0.1, 0.2, 0.3]
    gradients._set_float64(0, 0.1)
    gradients._set_float64(1, 0.2)
    gradients._set_float64(2, 0.3)

    var learning_rate = 0.1

    # Update: params = params - lr * gradients
    # Expected: [1 - 0.1*0.1, 2 - 0.1*0.2, 3 - 0.1*0.3]
    #         = [0.99, 1.98, 2.97]

    var updated = sgd_step_simple(params, gradients, learning_rate)

    var p0 = updated._get_float64(0)
    var p1 = updated._get_float64(1)
    var p2 = updated._get_float64(2)

    print("  Original params: [1.0, 2.0, 3.0]")
    print("  Gradients: [0.1, 0.2, 0.3]")
    print("  Learning rate:", learning_rate)
    print("  Updated params: [", p0, ",", p1, ",", p2, "]")

    # Check values
    if abs(p0 - 0.99) > 0.001:
        raise Error("Updated param[0] should be 0.99")
    if abs(p1 - 1.98) > 0.001:
        raise Error("Updated param[1] should be 1.98")
    if abs(p2 - 2.97) > 0.001:
        raise Error("Updated param[2] should be 2.97")

    print("  ✓ Basic SGD step test passed")


fn test_sgd_step_zero_gradients() raises:
    """Test SGD with zero gradients (params should not change)."""
    print("Testing SGD with zero gradients...")

    var shape = List[Int]()
    shape.append(3)
    var params = ExTensor(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)

    # Initial params: [1, 2, 3]
    params._set_float64(0, 1.0)
    params._set_float64(1, 2.0)
    params._set_float64(2, 3.0)

    var learning_rate = 0.1

    # With zero gradients, params should not change
    var updated = sgd_step_simple(params, gradients, learning_rate)

    var p0 = updated._get_float64(0)
    var p1 = updated._get_float64(1)
    var p2 = updated._get_float64(2)

    print("  Original params: [1.0, 2.0, 3.0]")
    print("  Zero gradients applied")
    print("  Updated params: [", p0, ",", p1, ",", p2, "]")

    # Check params unchanged
    if abs(p0 - 1.0) > 0.001:
        raise Error("Param[0] should be unchanged")
    if abs(p1 - 2.0) > 0.001:
        raise Error("Param[1] should be unchanged")
    if abs(p2 - 3.0) > 0.001:
        raise Error("Param[2] should be unchanged")

    print("  ✓ Zero gradients test passed")


fn test_sgd_step_large_learning_rate() raises:
    """Test SGD with large learning rate."""
    print("Testing SGD with large learning rate...")

    var shape = List[Int]()
    shape.append(2)
    var params = ExTensor(shape, DType.float32)
    var gradients = ExTensor(shape, DType.float32)

    # Initial params: [10, 20]
    params._set_float64(0, 10.0)
    params._set_float64(1, 20.0)

    # Gradients: [1, 2]
    gradients._set_float64(0, 1.0)
    gradients._set_float64(1, 2.0)

    var learning_rate = 5.0  # Large LR

    # Update: [10 - 5*1, 20 - 5*2] = [5, 10]
    var updated = sgd_step_simple(params, gradients, learning_rate)

    var p0 = updated._get_float64(0)
    var p1 = updated._get_float64(1)

    print("  Large learning rate:", learning_rate)
    print("  Updated params: [", p0, ",", p1, "]")

    if abs(p0 - 5.0) > 0.001:
        raise Error("Param[0] should be 5.0")
    if abs(p1 - 10.0) > 0.001:
        raise Error("Param[1] should be 10.0")

    print("  ✓ Large learning rate test passed")


fn test_sgd_step_negative_gradients() raises:
    """Test SGD with negative gradients (params should increase)."""
    print("Testing SGD with negative gradients...")

    var shape = List[Int]()
    shape.append(2)
    var params = ExTensor(shape, DType.float32)
    var gradients = ExTensor(shape, DType.float32)

    # Initial params: [1, 2]
    params._set_float64(0, 1.0)
    params._set_float64(1, 2.0)

    # Negative gradients: [-1, -2]
    gradients._set_float64(0, -1.0)
    gradients._set_float64(1, -2.0)

    var learning_rate = 0.1

    # Update: [1 - 0.1*(-1), 2 - 0.1*(-2)] = [1.1, 2.2]
    var updated = sgd_step_simple(params, gradients, learning_rate)

    var p0 = updated._get_float64(0)
    var p1 = updated._get_float64(1)

    print("  Negative gradients: [-1.0, -2.0]")
    print("  Updated params: [", p0, ",", p1, "]")

    # Params should increase with negative gradients
    if abs(p0 - 1.1) > 0.001:
        raise Error("Param[0] should be 1.1")
    if abs(p1 - 2.2) > 0.001:
        raise Error("Param[1] should be 2.2")

    print("  ✓ Negative gradients test passed")


fn test_sgd_step_with_weight_decay() raises:
    """Test SGD with L2 weight decay."""
    print("Testing SGD with weight decay...")

    var shape = List[Int]()
    shape.append(2)
    var params = ExTensor(shape, DType.float32)
    var gradients = ExTensor(shape, DType.float32)
    var velocity = zeros(shape, DType.float32)  # Not used, but required

    # Initial params: [1, 2]
    params._set_float64(0, 1.0)
    params._set_float64(1, 2.0)

    # Gradients: [0, 0]
    gradients._set_float64(0, 0.0)
    gradients._set_float64(1, 0.0)

    var learning_rate = 0.1
    var momentum = 0.0
    var weight_decay = 0.01

    # With weight decay: effective_grad = grad + weight_decay * params
    # = [0 + 0.01*1, 0 + 0.01*2] = [0.01, 0.02]
    # Update: [1 - 0.1*0.01, 2 - 0.1*0.02] = [0.999, 1.998]

    var result = sgd_step(
        params, gradients, velocity, learning_rate, momentum, weight_decay
    )
    var updated = result[0]

    var p0 = updated._get_float64(0)
    var p1 = updated._get_float64(1)

    print("  Weight decay:", weight_decay)
    print("  Updated params: [", p0, ",", p1, "]")

    # Check that weight decay was applied (params decreased slightly)
    if p0 >= 1.0:
        raise Error("Param[0] should decrease with weight decay")
    if p1 >= 2.0:
        raise Error("Param[1] should decrease with weight decay")

    print("  ✓ Weight decay test passed")


fn test_sgd_multi_step_convergence() raises:
    """Test multiple SGD steps converge towards target."""
    print("Testing multi-step SGD convergence...")

    var shape = List[Int]()
    shape.append(1)
    var params = ExTensor(shape, DType.float32)

    # Start at params = 10, try to reach target = 0
    params._set_float64(0, 10.0)
    var target = 0.0
    var learning_rate = 0.1

    print("  Initial param:", params._get_float64(0))

    # Run 20 steps
    for step in range(20):
        # Gradient = 2 * (params - target) (MSE gradient)
        var current = params._get_float64(0)
        var error = current - target
        var grad = 2.0 * error

        var gradients = ExTensor(shape, DType.float32)
        gradients._set_float64(0, grad)

        # Update
        params = sgd_step_simple(params, gradients, learning_rate)

    var final_param = params._get_float64(0)
    print("  Final param after 20 steps:", final_param)

    # Should be much closer to 0
    if abs(final_param) > 1.0:
        raise Error("Param should converge closer to target")

    # Should be less than initial value
    if final_param >= 10.0:
        raise Error("Param should decrease towards target")

    print("  ✓ Multi-step convergence test passed")


fn run_all_tests() raises:
    """Run all SGD optimizer tests."""
    print("=" * 60)
    print("SGD Optimizer Test Suite")
    print("=" * 60)

    test_sgd_step_simple_basic()
    test_sgd_step_zero_gradients()
    test_sgd_step_large_learning_rate()
    test_sgd_step_negative_gradients()
    test_sgd_step_with_weight_decay()
    test_sgd_multi_step_convergence()

    print("=" * 60)
    print("All SGD optimizer tests passed! ✓")
    print("=" * 60)


fn main() raises:
    """Entry point for SGD tests."""
    run_all_tests()
