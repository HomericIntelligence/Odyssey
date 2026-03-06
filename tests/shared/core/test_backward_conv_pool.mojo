"""Tests for conv2d and pooling backward passes.

Note: Split from test_backward.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
)
from shared.testing import check_gradient


fn test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 2
    var in_channels = 3
    var out_channels = 4
    var in_h = 8
    var in_w = 8
    var kh = 3
    var kw = 3

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var grad_output = ones_like(output)
    var grads = conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_h)
    assert_equal(gi_shape[3], in_w)

    var gk_shape = grads.grad_weights.shape()
    assert_equal(gk_shape[0], out_channels)
    assert_equal(gk_shape[1], in_channels)
    assert_equal(gk_shape[2], kh)
    assert_equal(gk_shape[3], kw)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


fn test_conv2d_backward_with_stride() raises:
    """Test conv2d_backward with stride > 1."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    var output = conv2d(x, kernel, bias, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grads = conv2d_backward(grad_output, x, kernel, stride=2, padding=0)

    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 1)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


fn test_maxpool2d_backward_shapes() raises:
    """Test that maxpool2d_backward returns correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


fn test_maxpool2d_backward_gradient_routing() raises:
    """Test that maxpool2d_backward routes gradients only to max positions."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = zeros(input_shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0

    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


fn test_avgpool2d_backward_gradient_distribution() raises:
    """Test that avgpool2d_backward distributes gradients equally."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = ones(input_shape, DType.float32)

    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)
    var grad_input = avgpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(0.25), tolerance=1e-5
    )


fn main() raises:
    """Run conv2d and pooling backward tests."""
    print("Running conv2d and pooling backward tests...")
    test_conv2d_backward_shapes()
    test_conv2d_backward_with_stride()
    test_maxpool2d_backward_shapes()
    test_maxpool2d_backward_gradient_routing()
    test_avgpool2d_backward_gradient_distribution()
    print("All conv2d and pooling backward tests passed!")
