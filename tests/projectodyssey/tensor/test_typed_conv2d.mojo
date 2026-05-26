"""Tests for Parameterized Conv2dLayer[dtype].

TDD tests for Phase 4a (PR 6, epic #4998): parameterize non-Module layers.
Conv2dLayer becomes Conv2dLayer[dtype: DType = DType.float32] with
Tensor[dtype] forward interface.

Tests cover:
- Default dtype (float32) constructor
- Explicit float64 parameterization
- Weight shape (out_ch, in_ch, kH, kW)
- Bias shape (out_ch,)
- Forward pass preserves dtype
- Forward pass output shape
- Parameters list
- Backward pass dtype preservation
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.core.layers.conv2d import Conv2dLayer
from projectodyssey.tensor.tensor import Tensor


def test_conv2d_default_dtype() raises:
    """Conv2dLayer defaults to float32 weights."""
    var layer = Conv2dLayer(
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3
    )
    assert_true(
        layer.weight.get_dtype() == DType.float32,
        "weight dtype should be float32",
    )
    assert_true(
        layer.bias.get_dtype() == DType.float32, "bias dtype should be float32"
    )
    print("PASS: test_conv2d_default_dtype")


def test_conv2d_float64() raises:
    """Conv2dLayer[DType.float64] uses float64 tensors."""
    var layer = Conv2dLayer[DType.float64](
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3
    )
    assert_true(
        layer.weight.get_dtype() == DType.float64,
        "weight dtype should be float64",
    )
    assert_true(
        layer.bias.get_dtype() == DType.float64, "bias dtype should be float64"
    )
    print("PASS: test_conv2d_float64")


def test_conv2d_weight_shape() raises:
    """Weight has shape (out_channels, in_channels, kernel_h, kernel_w)."""
    var layer = Conv2dLayer(
        in_channels=3, out_channels=16, kernel_h=5, kernel_w=5
    )
    var s = layer.weight.shape()
    assert_true(len(s) == 4, "weight should be 4D")
    assert_true(s[0] == 16, "out_channels")
    assert_true(s[1] == 3, "in_channels")
    assert_true(s[2] == 5, "kernel_h")
    assert_true(s[3] == 5, "kernel_w")
    assert_true(layer.weight.numel() == 16 * 3 * 5 * 5, "weight numel")
    print("PASS: test_conv2d_weight_shape")


def test_conv2d_bias_shape() raises:
    """Bias has shape (out_channels,) and is initialized to zeros."""
    var layer = Conv2dLayer(
        in_channels=3, out_channels=16, kernel_h=3, kernel_w=3
    )
    assert_true(
        layer.bias.numel() == 16, "bias should have out_channels elements"
    )
    # Bias initialized to 0.0
    assert_almost_equal(Float32(layer.bias[0]), Float32(0.0), atol=1e-6)
    assert_almost_equal(Float32(layer.bias[15]), Float32(0.0), atol=1e-6)
    print("PASS: test_conv2d_bias_shape")


def test_conv2d_forward_typed() raises:
    """Forward pass accepts and returns Tensor[dtype]."""
    var layer = Conv2dLayer[DType.float32](
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3, padding=1
    )
    var input = Tensor[DType.float32]([1, 1, 5, 5])  # NCHW
    # Set some input values using FP-representable values
    input._data[0] = Scalar[DType.float32](0.5)
    input._data[1] = Scalar[DType.float32](1.0)
    input._data[12] = Scalar[DType.float32](1.5)
    var output = layer.forward(input.as_any())
    assert_true(
        output.get_dtype() == DType.float32, "output dtype should be float32"
    )
    # With padding=1, stride=1, kernel 3x3: output = [1, 2, 5, 5]
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim")
    assert_true(s[1] == 2, "out_channels dim")
    assert_true(s[2] == 5, "height dim (padding=1 preserves)")
    assert_true(s[3] == 5, "width dim (padding=1 preserves)")
    print("PASS: test_conv2d_forward_typed")


def test_conv2d_forward_output_shape_no_padding() raises:
    """Forward pass computes correct output shape without padding."""
    var layer = Conv2dLayer(
        in_channels=1, out_channels=4, kernel_h=3, kernel_w=3
    )
    var input = Tensor[DType.float32]([1, 1, 8, 8])
    var output = layer.forward(input.as_any())
    # out_h = (8 + 0 - 3) // 1 + 1 = 6
    # out_w = (8 + 0 - 3) // 1 + 1 = 6
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim")
    assert_true(s[1] == 4, "out_channels dim")
    assert_true(s[2] == 6, "out_height: (8 - 3) // 1 + 1 = 6")
    assert_true(s[3] == 6, "out_width: (8 - 3) // 1 + 1 = 6")
    print("PASS: test_conv2d_forward_output_shape_no_padding")


def test_conv2d_parameters_typed() raises:
    """Parameters() returns List with weight and bias."""
    var layer = Conv2dLayer(
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3
    )
    var params = layer.parameters()
    assert_true(len(params) == 2, "should have 2 parameters (weight, bias)")
    assert_true(params[0].numel() == 2 * 1 * 3 * 3, "weight numel")
    assert_true(params[1].numel() == 2, "bias numel")
    print("PASS: test_conv2d_parameters_typed")


def test_conv2d_backward_typed() raises:
    """Backward pass returns gradient tensors."""
    var layer = Conv2dLayer[DType.float32](
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3, padding=1
    )
    var input = Tensor[DType.float32]([1, 1, 5, 5])
    for i in range(input.numel()):
        input._data[i] = Scalar[DType.float32](0.5)
    var output = layer.forward(input.as_any())
    var grad_output = Tensor[DType.float32](output.shape())
    for i in range(grad_output.numel()):
        grad_output._data[i] = Scalar[DType.float32](1.0)
    var grads = layer.backward(grad_output.as_any(), input.as_any())
    var grad_input = grads[0]
    var grad_weight = grads[1]
    var grad_bias = grads[2]
    # grad_input should match input shape
    assert_true(grad_input.numel() == input.numel(), "grad_input numel")
    assert_true(grad_input.get_dtype() == DType.float32, "grad_input dtype")
    # grad_weight should match weight shape
    assert_true(
        grad_weight.numel() == layer.weight.numel(), "grad_weight numel"
    )
    # grad_bias should match bias shape
    assert_true(grad_bias.numel() == layer.bias.numel(), "grad_bias numel")
    print("PASS: test_conv2d_backward_typed")


def test_conv2d_stride() raises:
    """Conv2d with stride=2 halves spatial dimensions."""
    var layer = Conv2dLayer(
        in_channels=1, out_channels=2, kernel_h=3, kernel_w=3, stride=2
    )
    var input = Tensor[DType.float32]([1, 1, 8, 8])
    var output = layer.forward(input.as_any())
    # out_h = (8 + 0 - 3) // 2 + 1 = 3
    # out_w = (8 + 0 - 3) // 2 + 1 = 3
    var s = output.shape()
    assert_true(s[2] == 3, "out_height with stride=2")
    assert_true(s[3] == 3, "out_width with stride=2")
    print("PASS: test_conv2d_stride")


def main() raises:
    test_conv2d_default_dtype()
    test_conv2d_float64()
    test_conv2d_weight_shape()
    test_conv2d_bias_shape()
    test_conv2d_forward_typed()
    test_conv2d_forward_output_shape_no_padding()
    test_conv2d_parameters_typed()
    test_conv2d_backward_typed()
    test_conv2d_stride()
    print("All 9 typed conv2d tests passed!")
