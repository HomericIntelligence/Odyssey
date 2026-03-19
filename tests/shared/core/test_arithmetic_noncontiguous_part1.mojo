"""Tests for arithmetic operations on non-contiguous tensors - Part 1.

Verifies that add, subtract, multiply, divide produce correct results
when given non-contiguous inputs (e.g., from transpose_view). Without
the as_contiguous() guard these operations silently returned wrong values.

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
from shared.core.arithmetic import add, subtract, multiply, divide
from shared.core.matrix import transpose_view
from shared.core.shape import as_contiguous


fn _make_noncontiguous_2x3() raises -> ExTensor:
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


fn test_add_noncontiguous_lhs() raises:
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


fn test_add_noncontiguous_rhs() raises:
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


fn test_add_both_noncontiguous() raises:
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


fn test_subtract_noncontiguous_lhs() raises:
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


fn test_subtract_noncontiguous_rhs() raises:
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


fn test_multiply_noncontiguous_lhs() raises:
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


fn test_multiply_noncontiguous_rhs() raises:
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


fn test_divide_noncontiguous_lhs() raises:
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


fn test_divide_noncontiguous_rhs() raises:
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


fn test_result_is_contiguous() raises:
    """Result of ops on non-contiguous inputs should be contiguous."""
    var nc = _make_noncontiguous_2x3()
    var rhs = ones([3, 2], DType.float32)

    var result = add(nc, rhs)

    assert_true(result.is_contiguous(), "result should always be contiguous")


fn main() raises:
    """Run all non-contiguous arithmetic tests (part 1)."""
    print("Running arithmetic non-contiguous tests (part 1)...")

    test_add_noncontiguous_lhs()
    test_add_noncontiguous_rhs()
    test_add_both_noncontiguous()
    test_subtract_noncontiguous_lhs()
    test_subtract_noncontiguous_rhs()
    test_multiply_noncontiguous_lhs()
    test_multiply_noncontiguous_rhs()
    test_divide_noncontiguous_lhs()
    test_divide_noncontiguous_rhs()
    test_result_is_contiguous()

    print("All arithmetic non-contiguous tests (part 1) passed!")
