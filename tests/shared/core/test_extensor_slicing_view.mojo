# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split per ADR-009. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor slice() view semantics (#3799).

Verifies that slice() returns a zero-copy view using view_with_strides,
element access on sliced views returns correct values, writes through
views affect the original, and slice + transpose composition works.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, arange
from tests.shared.conftest import assert_true, assert_false, assert_almost_equal, assert_equal


# ============================================================================
# slice() view semantics tests
# ============================================================================


def test_slice_returns_view() raises:
    """The slice() method returns a tensor with _is_view == True."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var s = t.slice(0, 5)
    assert_true(s._is_view)


def test_slice_view_correct_values() raises:
    """A slice() view has correct element values."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var s = t.slice(3, 7)

    assert_equal(s.numel(), 4)
    assert_almost_equal(Float64(s[0]), 3.0, tolerance=1e-5)
    assert_almost_equal(Float64(s[1]), 4.0, tolerance=1e-5)
    assert_almost_equal(Float64(s[2]), 5.0, tolerance=1e-5)
    assert_almost_equal(Float64(s[3]), 6.0, tolerance=1e-5)


def test_slice_view_mutates_original() raises:
    """Writing to a slice view affects the original tensor."""
    var t = zeros([6], DType.float32)
    var s = t.slice(2, 5)

    s[0] = 99.0
    s[1] = 88.0
    s[2] = 77.0

    # The original tensor's elements 2, 3, 4 should reflect the writes
    assert_almost_equal(Float64(t[2]), 99.0, tolerance=1e-5)
    assert_almost_equal(Float64(t[3]), 88.0, tolerance=1e-5)
    assert_almost_equal(Float64(t[4]), 77.0, tolerance=1e-5)


def test_slice_view_axis1() raises:
    """The slice() method along axis=1 on a 2D tensor returns correct shape and values."""
    # 4x6 tensor with sequential values [0..23]
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t2d = t.reshape([4, 6])

    var s = t2d.slice(1, 4, axis=1)

    var shape = s.shape()
    assert_equal(shape[0], 4)
    assert_equal(shape[1], 3)

    # s[0] (flat index 0) -> row 0, col 1 of original (value 1)
    assert_almost_equal(Float64(s[0]), 1.0, tolerance=1e-5)
    # s[1] (flat index 1) -> row 0, col 2 of original (value 2)
    assert_almost_equal(Float64(s[1]), 2.0, tolerance=1e-5)


def test_slice_view_axis_bounds_error() raises:
    """The slice() method raises on invalid axis."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var raised = False
    try:
        var _ = t2d.slice(0, 1, axis=5)
    except:
        raised = True
    assert_true(raised)


def test_slice_view_index_bounds_error() raises:
    """The slice() method raises when end index exceeds dimension size."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var raised = False
    try:
        var _ = t.slice(0, 10)
    except:
        raised = True
    assert_true(raised)


def test_slice_on_transposed_view() raises:
    """Applying slice() on a transposed view returns correct values via _nd_index_to_flat_offset."""
    # 3x4 tensor: values 0..11
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])
    # Transpose to shape (4, 3)
    var tx = t2d.transpose(0, 1)
    # slice(1, 3) along axis=0 gives rows 1 and 2 of the (4,3) transposed view
    # i.e., original columns 1 and 2
    var s = tx.slice(1, 3, axis=0)

    var shape = s.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)

    # s[0] (flat index 0) -> coord (0,0) in shape (2,3), which in tx is row 1, col 0
    # tx strides are [1, 4] (transposed from original [4,1])
    # tx[1,0] -> element at byte offset 1*1 + 0*4 = 1 -> original element 1 = value 1
    assert_almost_equal(Float64(s[0]), 1.0, tolerance=1e-5)
    # s[1] -> coord (0,1) in (2,3) -> tx[1,1] -> 1*1 + 1*4 = 5 -> value 5
    assert_almost_equal(Float64(s[1]), 5.0, tolerance=1e-5)
    # s[2] -> coord (0,2) in (2,3) -> tx[1,2] -> 1*1 + 2*4 = 9 -> value 9
    assert_almost_equal(Float64(s[2]), 9.0, tolerance=1e-5)
    # s[3] -> coord (1,0) in (2,3) -> tx[2,0] -> 2*1 + 0*4 = 2 -> value 2
    assert_almost_equal(Float64(s[3]), 2.0, tolerance=1e-5)


def main() raises:
    """Run all slice() view semantics tests."""
    test_slice_returns_view()
    test_slice_view_correct_values()
    test_slice_view_mutates_original()
    test_slice_view_axis1()
    test_slice_view_axis_bounds_error()
    test_slice_view_index_bounds_error()
    test_slice_on_transposed_view()
    print("All slice() view semantics tests passed!")
