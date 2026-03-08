# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_backward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for linear layer backward passes."""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, ones_like
from shared.core.linear import linear, linear_backward
from shared.testing import (
    check_gradient,
)


fn test_linear_backward_shapes() raises:
    """Test that linear_backward returns correct gradient shapes."""
    var batch = 2
    var in_features = 10
    var out_features = 5

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = ones(input_shape, DType.float32)

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = List[Int]()
    grad_out_shape.append(batch)
    grad_out_shape.append(out_features)
    var grad_output = ones(grad_out_shape, DType.float32)

    var grads = linear_backward(grad_output, x, weights)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_features)

    var gw_shape = grads.grad_weights.shape()
    assert_equal(gw_shape[0], out_features)
    assert_equal(gw_shape[1], in_features)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_features)


fn test_linear_backward_numerical() raises:
    """Test linear_backward with numerical gradient checking."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    var x = ones(input_shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var weight_shape = List[Int]()
    weight_shape.append(2)
    weight_shape.append(2)
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.5
    weights._data.bitcast[Float32]()[1] = 0.3
    weights._data.bitcast[Float32]()[2] = 0.2
    weights._data.bitcast[Float32]()[3] = 0.4

    var grad_out_shape = List[Int]()
    grad_out_shape.append(1)
    grad_out_shape.append(2)
    var grad_output = ones(grad_out_shape, DType.float32)

    var grads = linear_backward(grad_output, x, weights)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 2)

    assert_almost_equal(
        grads.grad_input._data.bitcast[Float32]()[0],
        Float32(0.7),
        tolerance=1e-5,
    )
    assert_almost_equal(
        grads.grad_input._data.bitcast[Float32]()[1],
        Float32(0.7),
        tolerance=1e-5,
    )


fn test_linear_backward_batch() raises:
    """Test linear_backward with batch size > 1."""
    var batch = 3
    var in_features = 4
    var out_features = 2

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = ones(input_shape, DType.float32)

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = List[Int]()
    grad_out_shape.append(batch)
    grad_out_shape.append(out_features)
    var grad_output = ones(grad_out_shape, DType.float32)

    var grads = linear_backward(grad_output, x, weights)

    assert_equal(grads.grad_input.shape()[0], batch)
    assert_equal(grads.grad_weights.shape()[0], out_features)
    assert_equal(grads.grad_bias.shape()[0], out_features)


fn test_linear_backward_gradient() raises:
    """Test linear backward with numerical gradient checking."""
    var batch = 2
    var in_features = 3
    var out_features = 2

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = zeros(input_shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.4
    weights._data.bitcast[Float32]()[1] = 0.2
    weights._data.bitcast[Float32]()[2] = -0.3
    weights._data.bitcast[Float32]()[3] = 0.6
    weights._data.bitcast[Float32]()[4] = -0.2
    weights._data.bitcast[Float32]()[5] = 0.5

    fn forward(inp: ExTensor) raises -> ExTensor:
        var bias_shape = List[Int]()
        bias_shape.append(out_features)
        var bias = zeros(bias_shape, DType.float32)
        bias._data.bitcast[Float32]()[0] = 0.3
        bias._data.bitcast[Float32]()[1] = -0.2
        return linear(inp, weights, bias)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = linear_backward(grad_out, inp, weights)
        return grads.grad_input

    var output = forward(x)
    var grad_output = ones_like(output)
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=5e-4)


fn main() raises:
    """Run linear backward tests."""
    print("Running linear backward tests...")
    test_linear_backward_shapes()
    test_linear_backward_numerical()
    test_linear_backward_batch()
    test_linear_backward_gradient()
    print("All linear backward tests passed!")
