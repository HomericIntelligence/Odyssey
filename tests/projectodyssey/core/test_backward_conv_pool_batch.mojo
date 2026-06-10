"""Tests for conv2d backward pass with batch>1.

Tests cover:
- grad_bias accumulation across batch dimension (batch>1)
- grad_weights accumulation across batch dimension (batch>1)
- grad_input correctness per batch item
"""

from tests.projectodyssey.conftest import assert_almost_equal, assert_equal
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.core.conv import conv2d, conv2d_backward


def test_conv2d_backward_batched_grad_bias() raises:
    """Verify grad_bias accumulates correctly over batch dimension.

    Config: batch=2, in_channels=3, out_channels=8, spatial=3x3, kernel=3x3
    stride=1, padding=0 -> output shape: (2, 8, 1, 1)
    All-ones setup:
      grad_bias[oc] = batch * out_H * out_W = 2 * 1 * 1 = 2.0
    """
    var batch = 2
    var in_channels = 3
    var out_channels = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward: output shape (2, 8, 1, 1)
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_bias = result.grad_bias

    # grad_bias[oc] = batch * out_H * out_W = 2.0 for all oc
    var grad_bias_data = grad_bias._data.bitcast[Float32]()
    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias_data[oc],
            Float32(2.0),
            tolerance=1e-4,
        )


def test_conv2d_backward_batched_grad_weights() raises:
    """Verify grad_weights accumulates correctly over batch dimension.

    Config: batch=2, in_channels=3, out_channels=8, spatial=3x3, kernel=3x3
    stride=1, padding=0 -> output shape: (2, 8, 1, 1)
    All-ones setup:
      grad_weights[oc, ic, kh, kw] = sum over (batch, oh, ow) of
        grad_output[b, oc, oh, ow] * x[b, ic, oh+kh, ow+kw]
        = 2 batch items * 1.0 * 1.0 = 2.0 for every weight position
    """
    var batch = 2
    var in_channels = 3
    var out_channels = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward: output shape (2, 8, 1, 1)
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_weights = result.grad_weights
    var grad_input = result.grad_input

    # grad_weights[oc, ic, kh, kw] = 2.0 for all indices (sum over batch=2)
    var n_weights = out_channels * in_channels * kH * kW
    var grad_weights_data = grad_weights._data.bitcast[Float32]()
    for i in range(n_weights):
        assert_almost_equal(
            grad_weights_data[i],
            Float32(2.0),
            tolerance=1e-4,
        )

    # grad_input[b, ic, ih, iw] = out_channels * kernel(1.0) * grad_output(1.0)
    # = 8.0 for all positions and both batch items
    var n_inputs = batch * in_channels * 3 * 3
    var grad_input_data = grad_input._data.bitcast[Float32]()
    for i in range(n_inputs):
        assert_almost_equal(
            grad_input_data[i],
            Float32(out_channels),
            tolerance=1e-4,
        )


def main() raises:
    """Run batched conv2d backward tests."""
    print("Running batched conv2d backward tests (batch>1)...")
    test_conv2d_backward_batched_grad_bias()
    print("✓ test_conv2d_backward_batched_grad_bias")
    test_conv2d_backward_batched_grad_weights()
    print("✓ test_conv2d_backward_batched_grad_weights")
    print("All batched conv2d backward tests passed.")
