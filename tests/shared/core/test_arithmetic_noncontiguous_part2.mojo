"""Tests for arithmetic operations on non-contiguous tensors - Part 2.

Verifies broadcasting with non-contiguous inputs. Broadcasting already
computed strides from the logical shape, but the flat-buffer read was
using the wrong (gap-filled) memory layout. These tests confirm the
as_contiguous() guard fixes broadcast cases too.

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
from shared.core.extensor import ExTensor, zeros, ones, full, arange
from shared.core.arithmetic import add, subtract, multiply
from shared.core.shape import transpose_view, as_contiguous


fn _make_nc_2x3() raises -> ExTensor:
    """Non-contiguous 3×2 tensor (logical values: 0,3,1,4,2,5)."""
    var base = arange(0.0, 6.0, 1.0, DType.float32)
    var shaped = base.reshape([2, 3])
    var nc = transpose_view(shaped)
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")
    return nc^


fn test_add_noncontiguous_shape_match() raises:
    """Non-contiguous tensors with matching shapes should add correctly."""
    var nc_a = _make_nc_2x3()
    var nc_b = _make_nc_2x3()

    var result = add(nc_a, nc_b)

    # Each element doubled
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(6.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(10.0), tolerance=1e-5)


fn test_subtract_noncontiguous_shape_match() raises:
    """Non-contiguous tensors with matching shapes should subtract correctly."""
    var nc_a = _make_nc_2x3()
    var nc_b = _make_nc_2x3()

    var result = subtract(nc_a, nc_b)

    # a - a = 0 for all elements
    var ptr = result._data.bitcast[Float32]()
    for i in range(6):
        assert_almost_equal(ptr[i], Float32(0.0), tolerance=1e-5)


fn test_multiply_noncontiguous_shape_match() raises:
    """Non-contiguous tensors with matching shapes should multiply correctly."""
    var nc_a = _make_nc_2x3()
    var nc_b = _make_nc_2x3()

    var result = multiply(nc_a, nc_b)

    # Each element squared: 0*0, 3*3, 1*1, 4*4, 2*2, 5*5
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(9.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(16.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(25.0), tolerance=1e-5)


fn test_add_broadcast_noncontiguous_1d() raises:
    """Non-contiguous 2D + contiguous 1D broadcast should work correctly."""
    # nc shape: (3, 2), contiguous 1D shape: (2,) → broadcasts to (3, 2)
    var nc = _make_nc_2x3()  # logical flat: 0,3,1,4,2,5
    var b = full([2], 1.0, DType.float32)  # add 1 to each row

    var result = add(nc, b)

    # Expected: nc[i,j] + b[j] = nc[i,j] + 1
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-5)   # 0+1
    assert_almost_equal(ptr[1], Float32(4.0), tolerance=1e-5)   # 3+1
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)   # 1+1
    assert_almost_equal(ptr[3], Float32(5.0), tolerance=1e-5)   # 4+1
    assert_almost_equal(ptr[4], Float32(3.0), tolerance=1e-5)   # 2+1
    assert_almost_equal(ptr[5], Float32(6.0), tolerance=1e-5)   # 5+1


fn test_add_broadcast_contiguous_to_noncontiguous() raises:
    """Contiguous 1D + non-contiguous 2D broadcast should work correctly."""
    var b = full([2], 1.0, DType.float32)
    var nc = _make_nc_2x3()

    var result = add(b, nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(6.0), tolerance=1e-5)


fn test_result_shape_preserved() raises:
    """Result of non-contiguous op should have correct shape."""
    var nc = _make_nc_2x3()  # shape (3, 2)
    var rhs = ones([3, 2], DType.float32)

    var result = add(nc, rhs)

    assert_equal_int(result.shape()[0], 3)
    assert_equal_int(result.shape()[1], 2)


fn test_noncontiguous_add_matches_contiguous_baseline() raises:
    """Non-contiguous result must match contiguous baseline computation."""
    # Build contiguous equivalent of the logical 3x2 transposed tensor
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0; lp[1] = 3.0
    lp[2] = 1.0; lp[3] = 4.0
    lp[4] = 2.0; lp[5] = 5.0

    var rhs = full([3, 2], 10.0, DType.float32)

    # Contiguous baseline
    var baseline = add(logical, rhs)

    # Non-contiguous path
    var nc = _make_nc_2x3()
    var result = add(nc, rhs)

    # Compare element by element
    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(6):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-5)


fn test_multiply_broadcast_noncontiguous_lhs() raises:
    """Non-contiguous 2D * contiguous 1D broadcast should work."""
    var nc = _make_nc_2x3()  # logical flat: 0,3,1,4,2,5
    var b = full([2], 2.0, DType.float32)

    var result = multiply(nc, b)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)    # 0*2
    assert_almost_equal(ptr[1], Float32(6.0), tolerance=1e-5)    # 3*2
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)    # 1*2
    assert_almost_equal(ptr[3], Float32(8.0), tolerance=1e-5)    # 4*2
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)    # 2*2
    assert_almost_equal(ptr[5], Float32(10.0), tolerance=1e-5)   # 5*2


fn test_noncontiguous_result_is_always_contiguous() raises:
    """Results of all binary ops on non-contiguous inputs should be contiguous."""
    var nc = _make_nc_2x3()
    var rhs = ones([3, 2], DType.float32)

    var r_add = add(nc, rhs)
    var r_sub = subtract(nc, rhs)
    var r_mul = multiply(nc, rhs)

    assert_true(r_add.is_contiguous(), "add result should be contiguous")
    assert_true(r_sub.is_contiguous(), "subtract result should be contiguous")
    assert_true(r_mul.is_contiguous(), "multiply result should be contiguous")


fn main() raises:
    """Run all non-contiguous arithmetic tests (part 2)."""
    print("Running arithmetic non-contiguous tests (part 2)...")

    test_add_noncontiguous_shape_match()
    test_subtract_noncontiguous_shape_match()
    test_multiply_noncontiguous_shape_match()
    test_add_broadcast_noncontiguous_1d()
    test_add_broadcast_contiguous_to_noncontiguous()
    test_result_shape_preserved()
    test_noncontiguous_add_matches_contiguous_baseline()
    test_multiply_broadcast_noncontiguous_lhs()
    test_noncontiguous_result_is_always_contiguous()

    print("All arithmetic non-contiguous tests (part 2) passed!")
