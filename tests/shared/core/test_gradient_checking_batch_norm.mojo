# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checking.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Parametrised gradient checking tests for batch norm backward pass.

Tests batch_norm2d_backward across multiple batch sizes using non-uniform
grad_output to avoid the pathological cancellation where uniform grad_output
(ones_like) causes analytical gradients to be exactly zero due to the
normalization identity sum(x_norm) = 0. See issue #3282.

Uses check_gradient() instead of check_gradients() because check_gradient()
accepts a custom grad_output parameter, allowing non-uniform upstream gradients.

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.testing.gradient_checker import check_gradient
from shared.testing.assertions import assert_true
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.normalization import batch_norm2d, batch_norm2d_backward
from shared.core.arithmetic import multiply
from shared.core.reduction import sum as reduce_sum


def _make_non_uniform_grad_output(output: AnyTensor) raises -> AnyTensor:
    """Create non-uniform grad_output that avoids batch norm cancellation.

    Pattern: Float32(i % 4) * 0.25 - 0.3 gives [-0.3, -0.05, 0.2, 0.45, ...]
    This breaks the sum(x_norm) = 0 symmetry that causes degenerate gradients.
    """
    var grad_output = zeros(output.shape(), output._dtype)
    for i in range(output.numel()):
        var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
        grad_output.set(i, Float32(val))
    return grad_output^


def _make_non_uniform_input(shape: List[Int]) raises -> AnyTensor:
    """Create non-uniform input to avoid zero-variance degenerate case."""
    var input = zeros(shape, DType.float32)
    for i in range(input.numel()):
        input.set(i, Float32(Float32(i) * Float32(0.1) + Float32(0.1)))
    return input^


def test_batch_norm_gradient_batch_size_1() raises:
    """Test batch_norm2d gradient checking with batch_size=1 (degenerate case).

    batch_size=1 is degenerate for batch norm (zero variance). With non-uniform
    grad_output, the gradient should still be numerically verifiable.
    """
    var shape = List[Int]()
    shape.append(1)  # batch_size=1 (degenerate)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = _make_non_uniform_input(shape)

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

    # Compute forward to get output shape for grad_output
    def forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    def backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var output = forward(input)
    var grad_output = _make_non_uniform_grad_output(output)

    # Use check_gradient which accepts custom grad_output
    check_gradient(forward, backward, input, grad_output, rtol=1e-2, atol=1e-2)


def test_batch_norm_gradient_batch_size_2() raises:
    """Test batch_norm2d gradient checking with batch_size=2."""
    var shape = List[Int]()
    shape.append(2)  # batch_size=2 (normal case)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = _make_non_uniform_input(shape)

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

    def forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    def backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var output = forward(input)
    var grad_output = _make_non_uniform_grad_output(output)

    check_gradient(forward, backward, input, grad_output, rtol=1e-2, atol=1e-2)


def test_batch_norm_gradient_batch_size_4() raises:
    """Test batch_norm2d gradient checking with batch_size=4."""
    var shape = List[Int]()
    shape.append(4)  # batch_size=4 (larger batch)
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = _make_non_uniform_input(shape)

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

    def forward(x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
        return result[0]

    def backward(grad_out: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, x, gamma, running_mean, running_var, training=True
        )
        return result[0]

    var output = forward(input)
    var grad_output = _make_non_uniform_grad_output(output)

    check_gradient(forward, backward, input, grad_output, rtol=1e-2, atol=1e-2)


def test_batch_norm_gamma_gradient_batch_size_2() raises:
    """Test batch_norm2d gamma gradient with batch_size=2."""
    var shape = List[Int]()
    shape.append(2)  # batch_size=2
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var input = _make_non_uniform_input(shape)

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

    def forward(g: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d(input, g, beta, running_mean, running_var, training=True)
        return result[0]

    def backward(grad_out: AnyTensor, g: AnyTensor) raises escaping -> AnyTensor:
        var result = batch_norm2d_backward(
            grad_out, input, g, running_mean, running_var, training=True
        )
        return result[1]

    var output = forward(gamma)
    var grad_output = _make_non_uniform_grad_output(output)

    check_gradient(forward, backward, gamma, grad_output, rtol=1e-2, atol=1e-2)


def main() raises:
    print("Running batch norm gradient checking tests...")
    test_batch_norm_gradient_batch_size_1()
    test_batch_norm_gradient_batch_size_2()
    test_batch_norm_gradient_batch_size_4()
    test_batch_norm_gamma_gradient_batch_size_2()
    print("All batch norm gradient checking tests passed!")
