# ADR-009: This file is intentionally limited to ≤8 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_depthwise_conv_grad_check.
# See docs/adr/ADR-009-heap-corruption-workaround.md
"""Gradient checking tests for depthwise_conv2d_backward (part 1).

Tests grad_input correctness via numerical gradient checking across
stride/padding configurations and multi-channel cases.
Follow-up from issue #3233.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.conv import depthwise_conv2d, depthwise_conv2d_backward
from shared.testing import check_gradient


fn test_depthwise_conv2d_backward_shapes() raises:
    """Test that depthwise_conv2d_backward returns correct gradient shapes."""
    var channels = 2
    var in_h = 6
    var in_w = 6
    var kh = 3
    var kw = 3

    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    var output = depthwise_conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    var grads = depthwise_conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], channels)
    assert_equal(gi_shape[2], in_h)
    assert_equal(gi_shape[3], in_w)

    var gk_shape = grads.grad_weights.shape()
    assert_equal(gk_shape[0], channels)
    assert_equal(gk_shape[1], 1)
    assert_equal(gk_shape[2], kh)
    assert_equal(gk_shape[3], kw)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], channels)


fn test_depthwise_conv2d_backward_stride1_padding0_grad_input() raises:
    """Test grad_input via numerical gradient check: stride=1, padding=0.

    Input (1,1,4,4), kernel (1,1,3,3). Uses non-uniform input and
    non-uniform grad_output to avoid pathological cancellation.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward_input(inp: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(inp, kernel, bias, stride=1, padding=0)

    fn backward_input(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, inp, kernel, stride=1, padding=0)
        return grads.grad_input

    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_backward_stride2_grad_input() raises:
    """Test grad_input via numerical gradient check: stride=2, padding=0.

    Input (1,1,8,8), kernel (1,1,3,3).
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(8)
    input_shape.append(8)
    var x = zeros(input_shape, DType.float32)

    for i in range(64):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.05

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward_input(inp: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(inp, kernel, bias, stride=2, padding=0)

    fn backward_input(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, inp, kernel, stride=2, padding=0)
        return grads.grad_input

    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_backward_padding1_grad_input() raises:
    """Test grad_input via numerical gradient check: stride=1, padding=1.

    Input (1,1,4,4), kernel (1,1,3,3).
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward_input(inp: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(inp, kernel, bias, stride=1, padding=1)

    fn backward_input(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, inp, kernel, stride=1, padding=1)
        return grads.grad_input

    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_backward_multichannel_grad_input() raises:
    """Test grad_input via numerical gradient check: 3 channels.

    Input (1,3,4,4), kernel (3,1,3,3). Each channel processed independently.
    """
    var channels = 3
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(channels)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    for i in range(channels * 16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.05

    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(channels * 9):
        kernel._data.bitcast[Float32]()[i] = Float32(i % 9) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    fn forward_input(inp: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(inp, kernel, bias, stride=1, padding=0)

    fn backward_input(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, inp, kernel, stride=1, padding=0)
        return grads.grad_input

    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2)


fn run_all_tests() raises:
    test_depthwise_conv2d_backward_shapes()
    test_depthwise_conv2d_backward_stride1_padding0_grad_input()
    test_depthwise_conv2d_backward_stride2_grad_input()
    test_depthwise_conv2d_backward_padding1_grad_input()
    test_depthwise_conv2d_backward_multichannel_grad_input()


fn main() raises:
    run_all_tests()
