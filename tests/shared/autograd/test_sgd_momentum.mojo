"""Tests for SGD optimizer with momentum implementation.

Tests momentum velocity storage and update rules for SGD optimizer.
Verifies:
- Velocity buffer initialization to zeros
- Momentum accumulation across steps
- Convergence behavior with and without momentum
- Parameter updates match expected formulas
"""

from std.testing import assert_true
from tests.shared.conftest import assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.autograd.variable import Variable
from shared.autograd.tape import GradientTape
from shared.autograd.optimizers import SGD


def test_sgd_basic() raises:
    """Test basic SGD without momentum."""
    # Create a simple 1D parameter
    var shape: List[Int] = [1]
    var param_data = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)

    # Create variable with gradient tracking
    var tape = GradientTape()
    tape.enable()

    var param = Variable(param_data, True, tape)
    var param_id = param.id

    # Create gradient manually for testing
    var grad = zeros(shape, DType.float32)
    grad._set_float64(0, 0.5)  # gradient = 0.5

    # Set gradient in tape registry
    tape.registry.set_grad(param_id, grad)

    # Create optimizer without momentum
    var optimizer = SGD(learning_rate=0.1, momentum=0.0)

    # Collect parameters
    var params: List[Variable] = []
    params.append(param.copy())

    # Perform one step
    optimizer.step(params, tape)

    # Expected update: param = 1.0 - 0.1 * 0.5 = 0.95
    var expected = 0.95
    var actual = Float64(params[0].data._get_float64(0))
    assert_almost_equal(actual, expected, tolerance=1e-6)


def test_sgd_momentum_init() raises:
    """Test that velocity buffers are initialized to zeros."""
    # Create parameter
    var shape: List[Int] = [2]
    var param_data = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)
    param_data._set_float64(1, 2.0)

    # Create variable
    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)
    var param_id = param.id

    # Create gradient
    var grad = zeros(shape, DType.float32)
    grad._set_float64(0, 0.1)
    grad._set_float64(1, 0.2)
    tape.registry.set_grad(param_id, grad)

    # Create optimizer with momentum
    var optimizer = SGD(learning_rate=0.01, momentum=0.9)

    var params: List[Variable] = []
    params.append(param.copy())

    # First step should initialize velocity
    optimizer.step(params, tape)

    # Check that velocity buffer was created
    assert_true(
        len(optimizer.velocities) == 1, "One velocity buffer should be created"
    )
    assert_true(
        optimizer._initialized, "Optimizer should be marked as initialized"
    )


def test_sgd_momentum_accumulation() raises:
    """Test momentum accumulation across multiple steps."""
    # Create parameter
    var shape: List[Int] = [1]
    var param_data = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)

    # Create variable
    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)
    var param_id = param.id

    # Create optimizer with momentum
    var optimizer = SGD(learning_rate=0.1, momentum=0.9)

    var params: List[Variable] = []
    params.append(param.copy())

    # Step 1: gradient = 1.0
    # v_1 = 0.9 * 0 + 1.0 = 1.0
    # param = 1.0 - 0.1 * 1.0 = 0.9
    var grad1 = zeros(shape, DType.float32)
    grad1._set_float64(0, 1.0)
    tape.registry.set_grad(param_id, grad1)
    optimizer.step(params, tape)
    tape.registry.clear()

    var param_after_step1 = Float64(params[0].data._get_float64(0))
    assert_almost_equal(param_after_step1, 0.9, tolerance=1e-6)

    # Step 2: gradient = 1.0
    # v_2 = 0.9 * 1.0 + 1.0 = 1.9
    # param = 0.9 - 0.1 * 1.9 = 0.71
    var grad2 = zeros(shape, DType.float32)
    grad2._set_float64(0, 1.0)
    tape.registry.set_grad(param_id, grad2)
    optimizer.step(params, tape)
    tape.registry.clear()

    var param_after_step2 = Float64(params[0].data._get_float64(0))
    assert_almost_equal(param_after_step2, 0.71, tolerance=1e-6)

    # Step 3: gradient = 1.0
    # v_3 = 0.9 * 1.9 + 1.0 = 2.71
    # param = 0.71 - 0.1 * 2.71 = 0.439
    var grad3 = zeros(shape, DType.float32)
    grad3._set_float64(0, 1.0)
    tape.registry.set_grad(param_id, grad3)
    optimizer.step(params, tape)

    var param_after_step3 = Float64(params[0].data._get_float64(0))
    assert_almost_equal(param_after_step3, 0.439, tolerance=1e-5)


def test_sgd_momentum_vs_vanilla() raises:
    """Test that momentum converges faster than vanilla SGD."""
    # Both optimizers should converge to same point, but momentum faster
    var shape: List[Int] = [1]

    # Vanilla SGD
    var param_vanilla = zeros(shape, DType.float32)
    param_vanilla._set_float64(0, 10.0)

    var tape_vanilla = GradientTape()
    tape_vanilla.enable()
    var var_vanilla = Variable(param_vanilla, True, tape_vanilla)
    var id_vanilla = var_vanilla.id

    var optimizer_vanilla = SGD(learning_rate=0.01, momentum=0.0)
    var params_vanilla: List[Variable] = []
    params_vanilla.append(var_vanilla.copy())

    # Momentum SGD
    var param_momentum = zeros(shape, DType.float32)
    param_momentum._set_float64(0, 10.0)

    var tape_momentum = GradientTape()
    tape_momentum.enable()
    var var_momentum = Variable(param_momentum, True, tape_momentum)
    var id_momentum = var_momentum.id

    var optimizer_momentum = SGD(learning_rate=0.01, momentum=0.9)
    var params_momentum: List[Variable] = []
    params_momentum.append(var_momentum.copy())

    # Run 5 steps with same gradients
    for _ in range(5):
        # Vanilla step
        var grad = zeros(shape, DType.float32)
        grad._set_float64(0, 1.0)  # Constant gradient
        tape_vanilla.registry.set_grad(id_vanilla, grad)
        optimizer_vanilla.step(params_vanilla, tape_vanilla)
        tape_vanilla.registry.clear()

        # Momentum step
        var grad_m = zeros(shape, DType.float32)
        grad_m._set_float64(0, 1.0)
        tape_momentum.registry.set_grad(id_momentum, grad_m)
        optimizer_momentum.step(params_momentum, tape_momentum)
        tape_momentum.registry.clear()

    var final_vanilla = Float64(params_vanilla[0].data._get_float64(0))
    var final_momentum = Float64(params_momentum[0].data._get_float64(0))

    # Momentum should have moved more (descended faster)
    assert_true(
        final_momentum < final_vanilla,
        "Momentum should converge faster than vanilla SGD",
    )


def test_sgd_zero_momentum() raises:
    """Test that momentum=0 behaves like standard SGD."""
    var shape: List[Int] = [1]

    # Create parameter
    var param_data = zeros(shape, DType.float32)
    param_data._set_float64(0, 5.0)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)
    var param_id = param.id

    var optimizer = SGD(learning_rate=0.1, momentum=0.0)

    var params: List[Variable] = []
    params.append(param.copy())

    # Step with gradient = 2.0
    var grad = zeros(shape, DType.float32)
    grad._set_float64(0, 2.0)
    tape.registry.set_grad(param_id, grad)
    optimizer.step(params, tape)

    # Expected: 5.0 - 0.1 * 2.0 = 4.8
    var result = Float64(params[0].data._get_float64(0))
    assert_almost_equal(result, 4.8, tolerance=1e-6)


def test_sgd_multiple_parameters() raises:
    """Test momentum with multiple parameters."""
    var shape: List[Int] = [2]

    # Create two parameters
    var param1_data = zeros(shape, DType.float32)
    param1_data._set_float64(0, 1.0)
    param1_data._set_float64(1, 2.0)

    var param2_data = zeros(shape, DType.float32)
    param2_data._set_float64(0, 3.0)
    param2_data._set_float64(1, 4.0)

    var tape = GradientTape()
    tape.enable()

    var param1 = Variable(param1_data, True, tape)
    var param2 = Variable(param2_data, True, tape)

    var id1 = param1.id
    var id2 = param2.id

    var optimizer = SGD(learning_rate=0.1, momentum=0.9)

    var params: List[Variable] = []
    params.append(param1.copy())
    params.append(param2.copy())

    # Step 1
    var grad1 = zeros(shape, DType.float32)
    grad1._set_float64(0, 0.5)
    grad1._set_float64(1, 0.5)
    tape.registry.set_grad(id1, grad1)

    var grad2 = zeros(shape, DType.float32)
    grad2._set_float64(0, 1.0)
    grad2._set_float64(1, 1.0)
    tape.registry.set_grad(id2, grad2)

    optimizer.step(params, tape)
    tape.registry.clear()

    # Check both parameters were updated
    var p1_val = Float64(params[0].data._get_float64(0))
    var p2_val = Float64(params[1].data._get_float64(0))

    # Expected: param1[0] = 1.0 - 0.1 * 0.5 = 0.95
    # Expected: param2[0] = 3.0 - 0.1 * 1.0 = 2.9
    assert_almost_equal(p1_val, 0.95, tolerance=1e-6)
    assert_almost_equal(p2_val, 2.9, tolerance=1e-6)


def main() raises:
    """Run all SGD momentum tests."""
    print("Running SGD momentum tests...")
    test_sgd_basic()
    test_sgd_momentum_init()
    test_sgd_momentum_accumulation()
    test_sgd_momentum_vs_vanilla()
    test_sgd_zero_momentum()
    test_sgd_multiple_parameters()
    print("All SGD momentum tests passed!")
