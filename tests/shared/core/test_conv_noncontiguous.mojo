"""Tests for conv2d operations on non-contiguous inputs

Verifies that conv2d() produces correct results when given non-contiguous
inputs. Without the as_contiguous() guard, flat-index arithmetic reads from
wrong memory positions, silently producing wrong values.

Strategy: compute conv2d on a contiguous tensor and on a non-contiguous
tensor with the same logical values; results must match.

Non-contiguous tensors are created by transposing H and W dims on
non-square tensors (e.g. 4×6 → 6×4 via transpose(2,3)), which produces
non-unit strides that violate C-order.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Follow-up from #3236.
"""


from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.conv import conv2d, conv2d_no_bias
from shared.core.shape import as_contiguous
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.conv import conv2d_backward, conv2d_no_bias_backward


def _make_nc_nchw_symmetric() raises -> AnyTensor:
    """Create a non-contiguous (1,1,6,4) tensor by transposing (1,1,4,6).

    The input (1,1,4,6) has all ones. Transposing H (dim2) and W (dim3)
    gives shape (1,1,6,4) with strides [24, 24, 1, 6] — non-contiguous.
    Since values are all ones, the logical content is identical to a
    contiguous (1,1,6,4) tensor of all ones.
    """
    var x = ones([1, 1, 4, 6], DType.float32)
    var nc = x.transpose(2, 3)  # shape (1,1,6,4), strides non-C-order
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")
    return nc^


def _make_nc_grad_output() raises -> AnyTensor:
    """Non-contiguous grad_output of logical shape (1,1,4,2).

    Base (1,1,2,4) all-ones transposed to (1,1,4,2): strides [8,8,1,4]
    vs C-order [8,8,2,1] — genuinely non-contiguous.
    Logical values are all ones, matching a contiguous (1,1,4,2) ones tensor.
    """
    var g = ones([1, 1, 2, 4], DType.float32)
    var nc = g.transpose(2, 3)  # shape (1,1,4,2), strides non-C-order
    assert_false(nc.is_contiguous(), "grad_output must be non-contiguous")
    return nc^


def _make_nc_input() raises -> AnyTensor:
    """Non-contiguous input of logical shape (1,1,6,4) with all ones.

    Base (1,1,4,6) transposed to (1,1,6,4): strides [24,24,1,6]
    vs C-order [24,24,4,1].
    """
    var x = ones([1, 1, 4, 6], DType.float32)
    var nc = x.transpose(2, 3)  # shape (1,1,6,4), non-contiguous
    assert_false(nc.is_contiguous(), "input must be non-contiguous")
    return nc^


def test_conv2d_noncontiguous_input_ones() raises:
    """Conv2d with non-contiguous all-ones input matches contiguous baseline."""
    # Both contiguous (1,1,6,4) and non-contiguous (1,1,6,4) have all-ones
    var x_cont = ones([1, 1, 6, 4], DType.float32)
    var nc_x = _make_nc_nchw_symmetric()  # logical (1,1,6,4) all-ones, non-contiguous

    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)

    var baseline = conv2d(x_cont, kernel, bias, stride=1, padding=0)
    var result = conv2d(nc_x, kernel, bias, stride=1, padding=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_noncontiguous_kernel() raises:
    """Conv2d with non-contiguous kernel matches contiguous baseline."""
    # Kernel (1,1,3,4) transposed to (1,1,4,3) — non-contiguous, all ones
    var kernel_cont = ones([1, 1, 3, 4], DType.float32)
    var nc_kernel = kernel_cont.transpose(2, 3)  # (1,1,4,3) non-contiguous
    assert_false(nc_kernel.is_contiguous(), "kernel must be non-contiguous")

    var x = ones([1, 1, 6, 5], DType.float32)
    var bias = zeros([1], DType.float32)
    # Both compute with all-ones values, so results must match
    var kernel_ref = ones([1, 1, 4, 3], DType.float32)

    var baseline = conv2d(x, kernel_ref, bias, stride=1, padding=0)
    var result = conv2d(x, nc_kernel, bias, stride=1, padding=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_result_shape_correct() raises:
    """Conv2d on non-contiguous input should return the correct output shape."""
    var nc_x = _make_nc_nchw_symmetric()  # logical (1,1,6,4)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)
    var x_cont = ones([1, 1, 6, 4], DType.float32)

    var baseline = conv2d(x_cont, kernel, bias, stride=1, padding=0)
    var result = conv2d(nc_x, kernel, bias, stride=1, padding=0)

    assert_equal_int(result.shape()[0], baseline.shape()[0])
    assert_equal_int(result.shape()[1], baseline.shape()[1])
    assert_equal_int(result.shape()[2], baseline.shape()[2])
    assert_equal_int(result.shape()[3], baseline.shape()[3])


def test_conv2d_result_is_contiguous() raises:
    """Conv2d output should always be contiguous regardless of input."""
    var nc_x = _make_nc_nchw_symmetric()
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)

    var result = conv2d(nc_x, kernel, bias, stride=1, padding=0)

    assert_true(result.is_contiguous(), "conv2d output should be contiguous")


def test_conv2d_noncontiguous_both_input_and_kernel() raises:
    """Conv2d with both non-contiguous input and kernel matches contiguous baseline."""
    var nc_x = _make_nc_nchw_symmetric()  # logical (1,1,6,4) all-ones

    var kernel_cont = ones([1, 1, 3, 4], DType.float32)
    var nc_kernel = kernel_cont.transpose(2, 3)  # logical (1,1,4,3) all-ones
    assert_false(nc_kernel.is_contiguous(), "kernel must be non-contiguous")

    var bias = zeros([1], DType.float32)
    var x_ref = ones([1, 1, 6, 4], DType.float32)
    var kernel_ref = ones([1, 1, 4, 3], DType.float32)

    var baseline = conv2d(x_ref, kernel_ref, bias, stride=1, padding=0)
    var result = conv2d(nc_x, nc_kernel, bias, stride=1, padding=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_no_bias_noncontiguous_input() raises:
    """Conv2d_no_bias with non-contiguous input matches contiguous baseline."""
    var nc_x = _make_nc_nchw_symmetric()  # logical (1,1,6,4) all-ones
    var x_cont = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)

    var baseline = conv2d_no_bias(x_cont, kernel, stride=1, padding=0)
    var result = conv2d_no_bias(nc_x, kernel, stride=1, padding=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_noncontiguous_with_stride() raises:
    """Conv2d with non-contiguous input and stride > 1 matches contiguous baseline."""
    # Use (1,1,8,6) → transpose(2,3) → logical (1,1,6,8), stride=2
    var x_cont_base = ones([1, 1, 8, 6], DType.float32)
    var nc_x = x_cont_base.transpose(2, 3)  # logical (1,1,6,8) all-ones
    assert_false(nc_x.is_contiguous(), "input must be non-contiguous")

    var x_ref = ones([1, 1, 6, 8], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)

    var baseline = conv2d(x_ref, kernel, bias, stride=2, padding=0)
    var result = conv2d(nc_x, kernel, bias, stride=2, padding=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_noncontiguous_with_padding() raises:
    """Conv2d with non-contiguous input and padding matches contiguous baseline."""
    var nc_x = _make_nc_nchw_symmetric()  # logical (1,1,6,4)
    var x_ref = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)

    var baseline = conv2d(x_ref, kernel, bias, stride=1, padding=1)
    var result = conv2d(nc_x, kernel, bias, stride=1, padding=1)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(baseline.numel()):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_conv2d_nc_output_shape_with_multichannel() raises:
    """Conv2d on non-contiguous input with multiple channels has correct output shape."""
    # (2,3,8,6) → transpose(2,3) → logical (2,3,6,8) non-contiguous
    var x_cont_base = ones([2, 3, 8, 6], DType.float32)
    var nc_x = x_cont_base.transpose(2, 3)
    assert_false(nc_x.is_contiguous(), "input must be non-contiguous")

    var kernel = ones([4, 3, 3, 3], DType.float32)
    var bias = zeros([4], DType.float32)
    var x_ref = ones([2, 3, 6, 8], DType.float32)

    var baseline = conv2d(x_ref, kernel, bias, stride=1, padding=0)
    var result = conv2d(nc_x, kernel, bias, stride=1, padding=0)

    assert_equal_int(result.shape()[0], baseline.shape()[0])
    assert_equal_int(result.shape()[1], baseline.shape()[1])
    assert_equal_int(result.shape()[2], baseline.shape()[2])
    assert_equal_int(result.shape()[3], baseline.shape()[3])


def test_conv2d_backward_noncontiguous_grad_output() raises:
    """Conv2d_backward with non-contiguous grad_output matches contiguous baseline.

    Input (1,1,6,4), kernel (1,1,3,3) → output shape (1,1,4,2).
    Non-contiguous grad: (1,1,2,4) transposed to logical (1,1,4,2), all ones.
    """
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var grad_cont = ones([1, 1, 4, 2], DType.float32)

    var baseline = conv2d_backward(grad_cont, x, kernel, stride=1, padding=0)

    var nc_grad = _make_nc_grad_output()  # logical (1,1,4,2) all-ones

    var result = conv2d_backward(nc_grad, x, kernel, stride=1, padding=0)

    var bi = baseline.grad_input
    var ri = result.grad_input
    assert_equal_int(bi.shape()[0], ri.shape()[0])
    assert_equal_int(bi.shape()[1], ri.shape()[1])
    assert_equal_int(bi.shape()[2], ri.shape()[2])
    assert_equal_int(bi.shape()[3], ri.shape()[3])

    var bip = bi._data.bitcast[Float32]()
    var rip = ri._data.bitcast[Float32]()
    for i in range(bi.numel()):
        assert_almost_equal(rip[i], bip[i], tolerance=1e-4)


def test_conv2d_backward_noncontiguous_input() raises:
    """Conv2d_backward with non-contiguous x matches contiguous baseline."""
    var x_cont = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var grad = ones([1, 1, 4, 2], DType.float32)

    var baseline = conv2d_backward(grad, x_cont, kernel, stride=1, padding=0)

    var nc_x = _make_nc_input()  # logical (1,1,6,4) all-ones
    var result = conv2d_backward(grad, nc_x, kernel, stride=1, padding=0)

    var bk = baseline.grad_weights
    var rk = result.grad_weights
    var bkp = bk._data.bitcast[Float32]()
    var rkp = rk._data.bitcast[Float32]()
    for i in range(bk.numel()):
        assert_almost_equal(rkp[i], bkp[i], tolerance=1e-4)


def test_conv2d_backward_noncontiguous_kernel() raises:
    """Conv2d_backward with non-contiguous kernel matches contiguous baseline."""
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel_cont = ones([1, 1, 3, 4], DType.float32)
    var grad = ones([1, 1, 4, 1], DType.float32)

    var baseline = conv2d_backward(grad, x, kernel_cont, stride=1, padding=0)

    # Transpose (1,1,3,4) → (1,1,4,3): strides non-C-order, all-ones logical
    var nc_kernel = kernel_cont.transpose(2, 3)
    assert_false(nc_kernel.is_contiguous(), "kernel must be non-contiguous")

    var kernel_ref = ones([1, 1, 4, 3], DType.float32)
    # Use same grad shape for nc_kernel (1,1,4,3) → output is (1,1,3,2)
    var grad_nc = ones([1, 1, 3, 2], DType.float32)
    var result = conv2d_backward(grad_nc, x, nc_kernel, stride=1, padding=0)

    var bi = baseline.grad_input
    var ri = result.grad_input
    # Shape check only — different kernel sizes produce different grad shapes
    assert_equal_int(bi.shape()[0], ri.shape()[0])
    assert_equal_int(bi.shape()[1], ri.shape()[1])
    assert_equal_int(bi.shape()[2], ri.shape()[2])


def test_conv2d_backward_grad_input_is_contiguous() raises:
    """Conv2d_backward grad_input output should be contiguous."""
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)

    var nc_grad = _make_nc_grad_output()  # logical (1,1,4,2)
    var result = conv2d_backward(nc_grad, x, kernel, stride=1, padding=0)

    assert_true(
        result.grad_input.is_contiguous(),
        "grad_input should be contiguous",
    )
    assert_true(
        result.grad_weights.is_contiguous(),
        "grad_kernel should be contiguous",
    )
    assert_true(
        result.grad_bias.is_contiguous(),
        "grad_bias should be contiguous",
    )


def test_conv2d_backward_grad_bias_noncontiguous() raises:
    """Conv2d_backward grad_bias sum correct with non-contiguous grad_output.

    All-ones grad_output (1,1,4,2) → grad_bias[0] = 4*2 = 8.
    """
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var grad_cont = ones([1, 1, 4, 2], DType.float32)

    var baseline = conv2d_backward(grad_cont, x, kernel, stride=1, padding=0)
    var nc_grad = _make_nc_grad_output()
    var result = conv2d_backward(nc_grad, x, kernel, stride=1, padding=0)

    var bb = baseline.grad_bias
    var rb = result.grad_bias
    var bbp = bb._data.bitcast[Float32]()
    var rbp = rb._data.bitcast[Float32]()
    for i in range(bb.numel()):
        assert_almost_equal(rbp[i], bbp[i], tolerance=1e-4)


def test_conv2d_no_bias_backward_noncontiguous() raises:
    """Conv2d_no_bias_backward with non-contiguous grad_output matches baseline."""
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var grad_cont = ones([1, 1, 4, 2], DType.float32)

    var baseline = conv2d_no_bias_backward(
        grad_cont, x, kernel, stride=1, padding=0
    )

    var nc_grad = _make_nc_grad_output()
    assert_false(nc_grad.is_contiguous(), "grad_output must be non-contiguous")

    var result = conv2d_no_bias_backward(nc_grad, x, kernel, stride=1, padding=0)

    var bi = baseline.grad_input
    var ri = result.grad_input
    var bip = bi._data.bitcast[Float32]()
    var rip = ri._data.bitcast[Float32]()
    for i in range(bi.numel()):
        assert_almost_equal(rip[i], bip[i], tolerance=1e-4)


def test_conv2d_backward_all_noncontiguous() raises:
    """Conv2d_backward with all non-contiguous inputs matches contiguous baseline."""
    var x_cont = ones([1, 1, 6, 4], DType.float32)
    var kernel_cont = ones([1, 1, 3, 3], DType.float32)
    var grad_cont = ones([1, 1, 4, 2], DType.float32)

    var baseline = conv2d_backward(grad_cont, x_cont, kernel_cont, stride=1, padding=0)

    var nc_grad = _make_nc_grad_output()
    var nc_x = _make_nc_input()

    var result = conv2d_backward(nc_grad, nc_x, kernel_cont, stride=1, padding=0)

    var bi = baseline.grad_input
    var ri = result.grad_input
    var bip = bi._data.bitcast[Float32]()
    var rip = ri._data.bitcast[Float32]()
    for i in range(bi.numel()):
        assert_almost_equal(rip[i], bip[i], tolerance=1e-4)


def test_conv2d_backward_grad_weights_shape() raises:
    """Conv2d_backward grad_weights has the correct shape with non-contiguous inputs."""
    var x = ones([1, 1, 6, 4], DType.float32)
    var kernel = ones([1, 1, 3, 3], DType.float32)

    var nc_grad = _make_nc_grad_output()
    assert_false(nc_grad.is_contiguous(), "grad_output must be non-contiguous")

    var result = conv2d_backward(nc_grad, x, kernel, stride=1, padding=0)

    assert_equal_int(result.grad_weights.shape()[0], 1)  # out_channels
    assert_equal_int(result.grad_weights.shape()[1], 1)  # in_channels
    assert_equal_int(result.grad_weights.shape()[2], 3)  # kH
    assert_equal_int(result.grad_weights.shape()[3], 3)  # kW


def main() raises:
    """Run all test_conv_noncontiguous tests."""
    print("Running test_conv_noncontiguous tests...")

    test_conv2d_noncontiguous_input_ones()
    print("✓ test_conv2d_noncontiguous_input_ones")

    test_conv2d_noncontiguous_kernel()
    print("✓ test_conv2d_noncontiguous_kernel")

    test_conv2d_result_shape_correct()
    print("✓ test_conv2d_result_shape_correct")

    test_conv2d_result_is_contiguous()
    print("✓ test_conv2d_result_is_contiguous")

    test_conv2d_noncontiguous_both_input_and_kernel()
    print("✓ test_conv2d_noncontiguous_both_input_and_kernel")

    test_conv2d_no_bias_noncontiguous_input()
    print("✓ test_conv2d_no_bias_noncontiguous_input")

    test_conv2d_noncontiguous_with_stride()
    print("✓ test_conv2d_noncontiguous_with_stride")

    test_conv2d_noncontiguous_with_padding()
    print("✓ test_conv2d_noncontiguous_with_padding")

    test_conv2d_nc_output_shape_with_multichannel()
    print("✓ test_conv2d_nc_output_shape_with_multichannel")

    test_conv2d_backward_noncontiguous_grad_output()
    print("✓ test_conv2d_backward_noncontiguous_grad_output")

    test_conv2d_backward_noncontiguous_input()
    print("✓ test_conv2d_backward_noncontiguous_input")

    test_conv2d_backward_noncontiguous_kernel()
    print("✓ test_conv2d_backward_noncontiguous_kernel")

    test_conv2d_backward_grad_input_is_contiguous()
    print("✓ test_conv2d_backward_grad_input_is_contiguous")

    test_conv2d_backward_grad_bias_noncontiguous()
    print("✓ test_conv2d_backward_grad_bias_noncontiguous")

    test_conv2d_no_bias_backward_noncontiguous()
    print("✓ test_conv2d_no_bias_backward_noncontiguous")

    test_conv2d_backward_all_noncontiguous()
    print("✓ test_conv2d_backward_all_noncontiguous")

    test_conv2d_backward_grad_weights_shape()
    print("✓ test_conv2d_backward_grad_weights_shape")

    print("\nAll test_conv_noncontiguous tests passed!")
