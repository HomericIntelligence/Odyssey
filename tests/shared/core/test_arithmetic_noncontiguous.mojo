"""Tests for arithmetic operations on non-contiguous tensors

Verifies that add, subtract, multiply, divide produce correct results
when given non-contiguous inputs (e.g., from transpose_view). Without
the as_contiguous() guard these operations silently returned wrong values.

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
from shared.core.arithmetic import add, subtract, multiply, divide
from shared.core.matrix import transpose_view
from shared.core.shape import as_contiguous
from shared.core.arithmetic import add, subtract, multiply


def _make_noncontiguous_2x3() raises -> AnyTensor:
    """Create a non-contiguous 3×2 tensor by transposing a 2×3.

    The logical values (row-major for 3×2) are:
        row 0: [0.0, 3.0]
        row 1: [1.0, 4.0]
        row 2: [2.0, 5.0]
    """
    # Allocate a 2×3 contiguous tensor with sequential values
    var base = arange(0.0, 6.0, 1.0, DType.float32)
    var shaped = base.reshape([2, 3])
    # transpose_view swaps axes → shape (3, 2), non-contiguous strides
    var nc = transpose_view(shaped)
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")
    return nc^


def _make_nc_2x3() raises -> AnyTensor:
    """Non-contiguous 3×2 tensor (logical values: 0,3,1,4,2,5)."""
    var base = arange(0.0, 6.0, 1.0, DType.float32)
    var shaped = base.reshape([2, 3])
    var nc = transpose_view(shaped)
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")
    return nc^


def test_add_noncontiguous_lhs() raises:
    """Non-contiguous lhs + contiguous rhs should produce correct values."""
    var nc = _make_noncontiguous_2x3()       # shape (3, 2), non-contiguous
    var rhs = full([3, 2], 10.0, DType.float32)  # contiguous

    var result = add(nc, rhs)

    # nc logical layout (row-major for 3x2): 0,3,1,4,2,5 → after +10: 10,13,11,14,12,15
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(10.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(13.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(11.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(14.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(12.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(15.0), tolerance=1e-5)


def test_add_noncontiguous_rhs() raises:
    """Contiguous lhs + non-contiguous rhs should produce correct values."""
    var lhs = full([3, 2], 10.0, DType.float32)
    var nc = _make_noncontiguous_2x3()

    var result = add(lhs, nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(10.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(13.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(11.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(14.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(12.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(15.0), tolerance=1e-5)


def test_add_both_noncontiguous() raises:
    """Both inputs non-contiguous should produce correct values."""
    var nc_a = _make_noncontiguous_2x3()  # logical values: 0,3,1,4,2,5
    var nc_b = _make_noncontiguous_2x3()  # same

    var result = add(nc_a, nc_b)

    # Expected: each logical element doubled
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(6.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(10.0), tolerance=1e-5)


def test_subtract_noncontiguous_lhs() raises:
    """Non-contiguous lhs - contiguous rhs should produce correct values."""
    var nc = _make_noncontiguous_2x3()
    var rhs = ones([3, 2], DType.float32)

    var result = subtract(nc, rhs)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(-1.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(4.0), tolerance=1e-5)


def test_subtract_noncontiguous_rhs() raises:
    """Contiguous lhs - non-contiguous rhs should produce correct values."""
    var lhs = ones([3, 2], DType.float32)
    var nc = _make_noncontiguous_2x3()

    var result = subtract(lhs, nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(-2.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(-3.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(-1.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(-4.0), tolerance=1e-5)


def test_multiply_noncontiguous_lhs() raises:
    """Non-contiguous lhs * contiguous rhs should produce correct values."""
    var nc = _make_noncontiguous_2x3()
    var rhs = full([3, 2], 2.0, DType.float32)

    var result = multiply(nc, rhs)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(6.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(10.0), tolerance=1e-5)


def test_multiply_noncontiguous_rhs() raises:
    """Contiguous lhs * non-contiguous rhs should produce correct values."""
    var lhs = full([3, 2], 2.0, DType.float32)
    var nc = _make_noncontiguous_2x3()

    var result = multiply(lhs, nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(6.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(10.0), tolerance=1e-5)


def test_divide_noncontiguous_lhs() raises:
    """Non-contiguous lhs / contiguous rhs should produce correct values."""
    var nc = _make_noncontiguous_2x3()  # logical values: 0,3,1,4,2,5
    # Use values starting from 1 to avoid division by zero
    var base = arange(1.0, 7.0, 1.0, DType.float32)
    var rhs_cont = full([3, 2], 2.0, DType.float32)

    var result = divide(nc, rhs_cont)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(1.5), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(0.5), tolerance=1e-5)
    assert_almost_equal(ptr[3], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[4], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[5], Float32(2.5), tolerance=1e-5)


def test_divide_noncontiguous_rhs() raises:
    """Contiguous lhs / non-contiguous rhs should produce correct values."""
    # Avoid division by zero: use lhs values aligned with nc logical layout
    # nc logical flat: 0,3,1,4,2,5  - index 0 would be div-by-zero
    # Use a non-contiguous from 1..6 instead
    var base2 = arange(1.0, 7.0, 1.0, DType.float32)
    var shaped2 = base2.reshape([2, 3])
    var nc = transpose_view(shaped2)  # logical values: 1,4,2,5,3,6
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")

    var lhs = full([3, 2], 12.0, DType.float32)
    var result = divide(lhs, nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(12.0), tolerance=1e-4)  # 12/1
    assert_almost_equal(ptr[1], Float32(3.0), tolerance=1e-4)   # 12/4
    assert_almost_equal(ptr[2], Float32(6.0), tolerance=1e-4)   # 12/2
    assert_almost_equal(ptr[3], Float32(2.4), tolerance=1e-4)   # 12/5
    assert_almost_equal(ptr[4], Float32(4.0), tolerance=1e-4)   # 12/3
    assert_almost_equal(ptr[5], Float32(2.0), tolerance=1e-4)   # 12/6


def test_result_is_contiguous() raises:
    """Result of ops on non-contiguous inputs should be contiguous."""
    var nc = _make_noncontiguous_2x3()
    var rhs = ones([3, 2], DType.float32)

    var result = add(nc, rhs)

    assert_true(result.is_contiguous(), "result should always be contiguous")


def test_add_noncontiguous_shape_match() raises:
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


def test_subtract_noncontiguous_shape_match() raises:
    """Non-contiguous tensors with matching shapes should subtract correctly."""
    var nc_a = _make_nc_2x3()
    var nc_b = _make_nc_2x3()

    var result = subtract(nc_a, nc_b)

    # a - a = 0 for all elements
    var ptr = result._data.bitcast[Float32]()
    for i in range(6):
        assert_almost_equal(ptr[i], Float32(0.0), tolerance=1e-5)


def test_multiply_noncontiguous_shape_match() raises:
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


def test_add_broadcast_noncontiguous_1d() raises:
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


def test_add_broadcast_contiguous_to_noncontiguous() raises:
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


def test_result_shape_preserved() raises:
    """Result of non-contiguous op should have correct shape."""
    var nc = _make_nc_2x3()  # shape (3, 2)
    var rhs = ones([3, 2], DType.float32)

    var result = add(nc, rhs)

    assert_equal_int(result.shape()[0], 3)
    assert_equal_int(result.shape()[1], 2)


def test_noncontiguous_add_matches_contiguous_baseline() raises:
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


def test_multiply_broadcast_noncontiguous_lhs() raises:
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


def test_noncontiguous_result_is_always_contiguous() raises:
    """Results of all binary ops on non-contiguous inputs should be contiguous."""
    var nc = _make_nc_2x3()
    var rhs = ones([3, 2], DType.float32)

    var r_add = add(nc, rhs)
    var r_sub = subtract(nc, rhs)
    var r_mul = multiply(nc, rhs)

    assert_true(r_add.is_contiguous(), "add result should be contiguous")
    assert_true(r_sub.is_contiguous(), "subtract result should be contiguous")
    assert_true(r_mul.is_contiguous(), "multiply result should be contiguous")


def main() raises:
    """Run all test_arithmetic_noncontiguous tests."""
    print("Running test_arithmetic_noncontiguous tests...")

    test_add_noncontiguous_lhs()
    print("✓ test_add_noncontiguous_lhs")

    test_add_noncontiguous_rhs()
    print("✓ test_add_noncontiguous_rhs")

    test_add_both_noncontiguous()
    print("✓ test_add_both_noncontiguous")

    test_subtract_noncontiguous_lhs()
    print("✓ test_subtract_noncontiguous_lhs")

    test_subtract_noncontiguous_rhs()
    print("✓ test_subtract_noncontiguous_rhs")

    test_multiply_noncontiguous_lhs()
    print("✓ test_multiply_noncontiguous_lhs")

    test_multiply_noncontiguous_rhs()
    print("✓ test_multiply_noncontiguous_rhs")

    test_divide_noncontiguous_lhs()
    print("✓ test_divide_noncontiguous_lhs")

    test_divide_noncontiguous_rhs()
    print("✓ test_divide_noncontiguous_rhs")

    test_result_is_contiguous()
    print("✓ test_result_is_contiguous")

    test_add_noncontiguous_shape_match()
    print("✓ test_add_noncontiguous_shape_match")

    test_subtract_noncontiguous_shape_match()
    print("✓ test_subtract_noncontiguous_shape_match")

    test_multiply_noncontiguous_shape_match()
    print("✓ test_multiply_noncontiguous_shape_match")

    test_add_broadcast_noncontiguous_1d()
    print("✓ test_add_broadcast_noncontiguous_1d")

    test_add_broadcast_contiguous_to_noncontiguous()
    print("✓ test_add_broadcast_contiguous_to_noncontiguous")

    test_result_shape_preserved()
    print("✓ test_result_shape_preserved")

    test_noncontiguous_add_matches_contiguous_baseline()
    print("✓ test_noncontiguous_add_matches_contiguous_baseline")

    test_multiply_broadcast_noncontiguous_lhs()
    print("✓ test_multiply_broadcast_noncontiguous_lhs")

    test_noncontiguous_result_is_always_contiguous()
    print("✓ test_noncontiguous_result_is_always_contiguous")

    print("\nAll test_arithmetic_noncontiguous tests passed!")
