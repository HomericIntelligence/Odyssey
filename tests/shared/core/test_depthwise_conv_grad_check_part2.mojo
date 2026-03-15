# ADR-009: This file is intentionally limited to ≤8 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_depthwise_conv_grad_check.
# See docs/adr/ADR-009-heap-corruption-workaround.md
"""Gradient checking tests for depthwise_conv2d_backward (part 2).

Tests grad_weights and grad_bias correctness via numerical gradient checking,
plus a full pipeline test verifying all three gradient fields.
Follow-up from issue #3233.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.conv import depthwise_conv2d, depthwise_conv2d_backward
from shared.testing import check_gradient


fn test_depthwise_conv2d_backward_grad_weights_numerical() raises:
    """Test grad_weights via numerical gradient check: stride=1, padding=0.

    Perturbs kernel with x held fixed. Input (1,1,4,4), kernel (1,1,3,3).
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

    # Perturb kernel, hold x fixed
    fn forward_weights(k: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(x, k, bias, stride=1, padding=0)

    fn backward_weights(grad_out: ExTensor, k: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, x, k, stride=1, padding=0)
        return grads.grad_weights

    var output = forward_weights(kernel)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_weights, backward_weights, kernel, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_backward_grad_bias_value() raises:
    """Test grad_bias = sum(grad_output, dims=[batch, H, W]) per channel.

    Uses a 1x2x2x2 case where the analytical value is easily verified.
    Two channels, each with a 1x1 output so grad_bias = grad_output summed
    over batch=1, out_H=1, out_W=1 → equals the single grad_output element.
    """
    var channels = 2

    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = zeros(input_shape, DType.float32)

    for i in range(channels * 9):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.1

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

    # stride=1, padding=0, kernel=3x3, input=3x3 → output is 1x1 per channel
    var output = depthwise_conv2d(x, kernel, bias, stride=1, padding=0)

    # grad_output: channel 0 gets 0.5, channel 1 gets -0.3
    var grad_output = zeros_like(output)
    grad_output._data.bitcast[Float32]()[0] = Float32(0.5)
    grad_output._data.bitcast[Float32]()[1] = Float32(-0.3)

    var grads = depthwise_conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    # grad_bias[c] = sum of grad_output over batch and spatial dims for channel c
    # Since output is 1x1 per channel: grad_bias[c] = grad_output[0, c, 0, 0]
    assert_almost_equal(
        grads.grad_bias._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        grads.grad_bias._data.bitcast[Float32]()[1], Float32(-0.3), tolerance=1e-5
    )


fn test_depthwise_conv2d_backward_grad_weights_multichannel() raises:
    """Test grad_weights via numerical gradient check: 2 channels, kernel 2x2.

    Input (1,2,4,4), kernel (2,1,2,2). Verifies per-channel kernel gradients.
    """
    var channels = 2
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
    kernel_shape.append(2)
    kernel_shape.append(2)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(channels * 4):
        kernel._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.1 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)

    fn forward_weights(k: ExTensor) raises -> ExTensor:
        return depthwise_conv2d(x, k, bias, stride=1, padding=0)

    fn backward_weights(grad_out: ExTensor, k: ExTensor) raises -> ExTensor:
        var grads = depthwise_conv2d_backward(grad_out, x, k, stride=1, padding=0)
        return grads.grad_weights

    var output = forward_weights(kernel)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    check_gradient(forward_weights, backward_weights, kernel, grad_output, rtol=1e-2, atol=1e-2)


fn test_depthwise_conv2d_backward_full_gradient_pipeline() raises:
    """Test full forward + backward pipeline: all three gradient fields non-zero.

    Runs depthwise_conv2d forward then depthwise_conv2d_backward and verifies
    that grad_input, grad_weights, and grad_bias all have correct shapes and
    contain non-zero values (non-trivial gradients).
    """
    var channels = 2
    var in_h = 5
    var in_w = 5
    var kh = 3
    var kw = 3

    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = zeros(input_shape, DType.float32)

    for i in range(channels * in_h * in_w):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.2

    var kernel_shape = List[Int]()
    kernel_shape.append(channels)
    kernel_shape.append(1)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = zeros(kernel_shape, DType.float32)

    for i in range(channels * kh * kw):
        kernel._data.bitcast[Float32]()[i] = Float32(i % 9) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, DType.float32)
    bias._data.bitcast[Float32]()[0] = Float32(0.1)
    bias._data.bitcast[Float32]()[1] = Float32(-0.1)

    var output = depthwise_conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = zeros_like(output)
    for i in range(output.numel()):
        grad_output._data.bitcast[Float32]()[i] = Float32(i % 4) * 0.25 - 0.3

    var grads = depthwise_conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    # Verify shapes
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

    # Verify all three gradient fields contain non-zero values
    var gi_nonzero = False
    for i in range(channels * in_h * in_w):
        if grads.grad_input._data.bitcast[Float32]()[i] != Float32(0.0):
            gi_nonzero = True
            break
    assert_equal(gi_nonzero, True)

    var gk_nonzero = False
    for i in range(channels * kh * kw):
        if grads.grad_weights._data.bitcast[Float32]()[i] != Float32(0.0):
            gk_nonzero = True
            break
    assert_equal(gk_nonzero, True)

    var gb_nonzero = False
    for i in range(channels):
        if grads.grad_bias._data.bitcast[Float32]()[i] != Float32(0.0):
            gb_nonzero = True
            break
    assert_equal(gb_nonzero, True)


fn run_all_tests() raises:
    test_depthwise_conv2d_backward_grad_weights_numerical()
    test_depthwise_conv2d_backward_grad_bias_value()
    test_depthwise_conv2d_backward_grad_weights_multichannel()
    test_depthwise_conv2d_backward_full_gradient_pipeline()


fn main() raises:
    run_all_tests()
