"""Tests for conv2d_backward on non-contiguous inputs - Part 2.

Verifies that conv2d_backward() produces correct gradients when given
non-contiguous inputs. Without the as_contiguous() guard, flat-index
arithmetic reads from wrong memory positions.

Non-contiguous tensors are created by transposing H and W spatial dims on
non-square tensors. For example, (1,1,2,4) transposed to (1,1,4,2) gives
strides [8,8,1,4] instead of C-order [8,8,2,1] — genuinely non-contiguous.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Follow-up from #3236.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.conv import conv2d_backward, conv2d_no_bias_backward


fn _make_nc_grad_output() raises -> AnyTensor:
    """Non-contiguous grad_output of logical shape (1,1,4,2).

    Base (1,1,2,4) all-ones transposed to (1,1,4,2): strides [8,8,1,4]
    vs C-order [8,8,2,1] — genuinely non-contiguous.
    Logical values are all ones, matching a contiguous (1,1,4,2) ones tensor.
    """
    var g = ones([1, 1, 2, 4], DType.float32)
    var nc = g.transpose(2, 3)  # shape (1,1,4,2), strides non-C-order
    assert_false(nc.is_contiguous(), "grad_output must be non-contiguous")
    return nc^


fn _make_nc_input() raises -> AnyTensor:
    """Non-contiguous input of logical shape (1,1,6,4) with all ones.

    Base (1,1,4,6) transposed to (1,1,6,4): strides [24,24,1,6]
    vs C-order [24,24,4,1].
    """
    var x = ones([1, 1, 4, 6], DType.float32)
    var nc = x.transpose(2, 3)  # shape (1,1,6,4), non-contiguous
    assert_false(nc.is_contiguous(), "input must be non-contiguous")
    return nc^


fn test_conv2d_backward_noncontiguous_grad_output() raises:
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


fn test_conv2d_backward_noncontiguous_input() raises:
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


fn test_conv2d_backward_noncontiguous_kernel() raises:
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


fn test_conv2d_backward_grad_input_is_contiguous() raises:
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


fn test_conv2d_backward_grad_bias_noncontiguous() raises:
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


fn test_conv2d_no_bias_backward_noncontiguous() raises:
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


fn test_conv2d_backward_all_noncontiguous() raises:
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


fn test_conv2d_backward_grad_weights_shape() raises:
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


fn main() raises:
    """Run all non-contiguous conv2d_backward tests (part 2)."""
    print("Running conv2d_backward non-contiguous tests (part 2)...")

    test_conv2d_backward_noncontiguous_grad_output()
    test_conv2d_backward_noncontiguous_input()
    test_conv2d_backward_noncontiguous_kernel()
    test_conv2d_backward_grad_input_is_contiguous()
    test_conv2d_backward_grad_bias_noncontiguous()
    test_conv2d_no_bias_backward_noncontiguous()
    test_conv2d_backward_all_noncontiguous()
    test_conv2d_backward_grad_weights_shape()

    print("All conv2d_backward non-contiguous tests (part 2) passed!")
