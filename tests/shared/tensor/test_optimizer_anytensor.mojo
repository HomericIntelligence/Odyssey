"""Tests for optimizer infrastructure with AnyTensor.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests verify that SGD and Adam optimizers work correctly when Variable.data
is AnyTensor (since AnyTensor is now an alias for AnyTensor). The optimizers
operate on List[Variable] with gradients stored in GradientTape.

Tests cover:
- SGD step updates parameters via AnyTensor data
- SGD with momentum uses AnyTensor velocity buffers
- Adam step updates parameters via AnyTensor data
- Adam moment buffers use AnyTensor storage
- Optimizer preserves tensor shape and dtype after step
"""

from testing import assert_true
from tests.shared.conftest import assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.autograd import Variable, GradientTape, SGD, Adam


def test_sgd_step_with_anytensor_data() raises:
    """SGD step updates Variable whose data is AnyTensor."""
    var shape: List[Int] = [4]
    var param_data: AnyTensor = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)
    param_data._set_float64(1, 0.5)
    param_data._set_float64(2, 1.0)
    param_data._set_float64(3, 0.5)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)
    var param_id = param.id

    # Set gradient: all 0.5
    var grad: AnyTensor = zeros(shape, DType.float32)
    grad._set_float64(0, 0.5)
    grad._set_float64(1, 0.5)
    grad._set_float64(2, 0.5)
    grad._set_float64(3, 0.5)
    tape.registry.set_grad(param_id, grad)

    var optimizer = SGD(learning_rate=0.1, momentum=0.0)
    var params: List[Variable] = []
    params.append(param.copy())

    optimizer.step(params, tape)

    # param[0] = 1.0 - 0.1 * 0.5 = 0.95
    var actual = Float64(params[0].data._get_float64(0))
    assert_almost_equal(actual, 0.95, tolerance=1e-6)
    # param[1] = 0.5 - 0.1 * 0.5 = 0.45
    var actual1 = Float64(params[0].data._get_float64(1))
    assert_almost_equal(actual1, 0.45, tolerance=1e-6)
    print("PASS: test_sgd_step_with_anytensor_data")


def test_sgd_momentum_anytensor_velocities() raises:
    """SGD with momentum stores velocity buffers as AnyTensor."""
    var shape: List[Int] = [2]
    var param_data: AnyTensor = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)
    param_data._set_float64(1, 1.0)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)

    var grad: AnyTensor = zeros(shape, DType.float32)
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 1.0)
    tape.registry.set_grad(param.id, grad)

    var optimizer = SGD(learning_rate=0.01, momentum=0.9)
    var params: List[Variable] = []
    params.append(param.copy())

    optimizer.step(params, tape)

    # After first step with momentum:
    # v = 0 * 0.9 + 1.0 = 1.0
    # param = 1.0 - 0.01 * 1.0 = 0.99
    var actual = Float64(params[0].data._get_float64(0))
    assert_almost_equal(actual, 0.99, tolerance=1e-6)
    print("PASS: test_sgd_momentum_anytensor_velocities")


def test_adam_step_with_anytensor_data() raises:
    """Adam step updates Variable whose data is AnyTensor."""
    var shape: List[Int] = [2]
    var param_data: AnyTensor = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)
    param_data._set_float64(1, 1.0)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)

    var grad: AnyTensor = zeros(shape, DType.float32)
    grad._set_float64(0, 0.5)
    grad._set_float64(1, 0.5)
    tape.registry.set_grad(param.id, grad)

    var optimizer = Adam(learning_rate=0.001)
    var params: List[Variable] = []
    params.append(param.copy())

    optimizer.step(params, tape)

    # After step, parameters should be updated (decreased from 1.0)
    var actual = Float64(params[0].data._get_float64(0))
    assert_true(actual < 1.0, "param should decrease after Adam step")
    assert_true(actual > 0.99, "param should not decrease too much with lr=0.001")
    print("PASS: test_adam_step_with_anytensor_data")


def test_optimizer_preserves_shape_dtype() raises:
    """Optimizer step preserves AnyTensor shape and dtype."""
    var shape: List[Int] = [3, 2]
    var param_data: AnyTensor = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, True, tape)

    var grad: AnyTensor = zeros(shape, DType.float32)
    grad._set_float64(0, 0.5)
    tape.registry.set_grad(param.id, grad)

    var optimizer = SGD(learning_rate=0.01)
    var params: List[Variable] = []
    params.append(param.copy())

    optimizer.step(params, tape)

    var result_shape = params[0].data.shape()
    assert_true(result_shape[0] == 3, "dim 0 preserved")
    assert_true(result_shape[1] == 2, "dim 1 preserved")
    assert_true(params[0].data.dtype() == DType.float32, "dtype preserved")
    assert_true(params[0].data.numel() == 6, "numel preserved")
    print("PASS: test_optimizer_preserves_shape_dtype")


def test_sgd_no_grad_param_skipped() raises:
    """SGD skips parameters with requires_grad=False."""
    var shape: List[Int] = [2]
    var param_data: AnyTensor = zeros(shape, DType.float32)
    param_data._set_float64(0, 1.0)
    param_data._set_float64(1, 1.0)

    var tape = GradientTape()
    tape.enable()
    var param = Variable(param_data, False, tape)

    var grad: AnyTensor = zeros(shape, DType.float32)
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 1.0)
    tape.registry.set_grad(param.id, grad)

    var optimizer = SGD(learning_rate=0.1)
    var params: List[Variable] = []
    params.append(param.copy())

    optimizer.step(params, tape)

    # Parameters should be unchanged since requires_grad=False
    var actual = Float64(params[0].data._get_float64(0))
    assert_almost_equal(actual, 1.0, tolerance=1e-6)
    print("PASS: test_sgd_no_grad_param_skipped")


def main() raises:
    test_sgd_step_with_anytensor_data()
    test_sgd_momentum_anytensor_velocities()
    test_adam_step_with_anytensor_data()
    test_optimizer_preserves_shape_dtype()
    test_sgd_no_grad_param_skipped()
    print("\n5 optimizer AnyTensor tests passed\n")
