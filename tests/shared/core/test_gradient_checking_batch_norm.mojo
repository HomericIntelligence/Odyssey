# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checking.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Parametrized gradient checking tests for batch norm backward pass.

Tests batch_norm2d_backward across multiple batch sizes to catch edge cases
where batch_size=1 exhibits degenerate behavior (zero variance, undefined gradients).

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import assert_true
from shared.testing import check_gradients
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.normalization import batch_norm2d, batch_norm2d_backward


fn test_batch_norm_gradient_batch_size_1() raises:
    """Test batch_norm2d gradient checking with batch_size=1 (degenerate case)."""
    # batch_size=1 has zero variance (N-1 = 0), gradients are clamped
    var shape = List[Int]()
    shape.append(1)  # batch_size=1 (degenerate)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = ones(shape, DType.float32)

    var gamma_shape = List[Int]()
    gamma_shape.append(2)  # channels
    var gamma = ones(gamma_shape, DType.float32)

    var beta_shape = List[Int]()
    beta_shape.append(2)  # channels
    var beta = zeros(beta_shape, DType.float32)

    var mean_shape = List[Int]()
    mean_shape.append(2)  # channels
    var running_mean = zeros(mean_shape, DType.float32)
    var running_var = ones(mean_shape, DType.float32)

    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    fn backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "batch_norm gradient check failed for batch_size=1")


fn test_batch_norm_gradient_batch_size_2() raises:
    """Test batch_norm2d gradient checking with batch_size=2."""
    var shape = List[Int]()
    shape.append(2)  # batch_size=2 (normal case)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = ones(shape, DType.float32)

    var gamma_shape = List[Int]()
    gamma_shape.append(2)  # channels
    var gamma = ones(gamma_shape, DType.float32)

    var beta_shape = List[Int]()
    beta_shape.append(2)  # channels
    var beta = zeros(beta_shape, DType.float32)

    var mean_shape = List[Int]()
    mean_shape.append(2)  # channels
    var running_mean = zeros(mean_shape, DType.float32)
    var running_var = ones(mean_shape, DType.float32)

    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    fn backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "batch_norm gradient check failed for batch_size=2")


fn test_batch_norm_gradient_batch_size_4() raises:
    """Test batch_norm2d gradient checking with batch_size=4."""
    var shape = List[Int]()
    shape.append(4)  # batch_size=4 (larger batch)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = ones(shape, DType.float32)

    var gamma_shape = List[Int]()
    gamma_shape.append(2)  # channels
    var gamma = ones(gamma_shape, DType.float32)

    var beta_shape = List[Int]()
    beta_shape.append(2)  # channels
    var beta = zeros(beta_shape, DType.float32)

    var mean_shape = List[Int]()
    mean_shape.append(2)  # channels
    var running_mean = zeros(mean_shape, DType.float32)
    var running_var = ones(mean_shape, DType.float32)

    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    fn backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "batch_norm gradient check failed for batch_size=4")


fn test_batch_norm_gamma_gradient_batch_size_2() raises:
    """Test batch_norm2d gamma gradient with batch_size=2."""
    var shape = List[Int]()
    shape.append(2)  # batch_size=2
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = ones(shape, DType.float32)

    var gamma_shape = List[Int]()
    gamma_shape.append(2)  # channels
    var gamma = ones(gamma_shape, DType.float32)

    var beta_shape = List[Int]()
    beta_shape.append(2)  # channels
    var beta = zeros(beta_shape, DType.float32)

    var mean_shape = List[Int]()
    mean_shape.append(2)  # channels
    var running_mean = zeros(mean_shape, DType.float32)
    var running_var = ones(mean_shape, DType.float32)

    fn forward(g: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(input, g, beta, running_mean, running_var, training=True)
        return result[0]

    fn backward(grad_out: AnyTensor, g: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, input, g, running_mean, running_var, training=True
        )
        return result[1]

    var passed = check_gradients(forward, backward, gamma)
    assert_true(passed, "batch_norm gamma gradient check failed for batch_size=2")


fn main() raises:
    print("Running batch norm gradient checking tests...")
    test_batch_norm_gradient_batch_size_1()
    test_batch_norm_gradient_batch_size_2()
    test_batch_norm_gradient_batch_size_4()
    test_batch_norm_gamma_gradient_batch_size_2()
    print("All batch norm gradient checking tests passed!")
