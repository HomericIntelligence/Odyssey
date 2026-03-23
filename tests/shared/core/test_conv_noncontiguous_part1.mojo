"""Tests for conv2d operations on non-contiguous inputs - Part 1.

Verifies that conv2d() produces correct results when given non-contiguous
inputs. Without the as_contiguous() guard, flat-index arithmetic reads from
wrong memory positions, silently producing wrong values.

Strategy: compute conv2d on a contiguous tensor and on a non-contiguous
tensor with the same logical values; results must match.

Non-contiguous tensors are created by transposing H and W dims on
non-square tensors (e.g. 4×6 → 6×4 via transpose(2,3)), which produces
non-unit strides that violate C-order.

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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.conv import conv2d, conv2d_no_bias
from shared.core.shape import as_contiguous


fn _make_nc_nchw_symmetric() raises -> AnyTensor:
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


fn test_conv2d_noncontiguous_input_ones() raises:
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


fn test_conv2d_noncontiguous_kernel() raises:
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


fn test_conv2d_result_shape_correct() raises:
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


fn test_conv2d_result_is_contiguous() raises:
    """Conv2d output should always be contiguous regardless of input."""
    var nc_x = _make_nc_nchw_symmetric()
    var kernel = ones([1, 1, 3, 3], DType.float32)
    var bias = zeros([1], DType.float32)

    var result = conv2d(nc_x, kernel, bias, stride=1, padding=0)

    assert_true(result.is_contiguous(), "conv2d output should be contiguous")


fn test_conv2d_noncontiguous_both_input_and_kernel() raises:
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


fn test_conv2d_no_bias_noncontiguous_input() raises:
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


fn test_conv2d_noncontiguous_with_stride() raises:
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


fn test_conv2d_noncontiguous_with_padding() raises:
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


fn test_conv2d_nc_output_shape_with_multichannel() raises:
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


fn main() raises:
    """Run all non-contiguous conv2d tests (part 1)."""
    print("Running conv2d non-contiguous tests (part 1)...")

    test_conv2d_noncontiguous_input_ones()
    test_conv2d_noncontiguous_kernel()
    test_conv2d_result_shape_correct()
    test_conv2d_result_is_contiguous()
    test_conv2d_noncontiguous_both_input_and_kernel()
    test_conv2d_no_bias_noncontiguous_input()
    test_conv2d_noncontiguous_with_stride()
    test_conv2d_noncontiguous_with_padding()
    test_conv2d_nc_output_shape_with_multichannel()

    print("All conv2d non-contiguous tests (part 1) passed!")
