# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split per ADR-009. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor view_with_strides primitive and _nd_index_to_flat_offset (#3799)."""

from shared.core.extensor import ExTensor, zeros, ones, arange
from tests.shared.conftest import assert_true, assert_false, assert_almost_equal, assert_equal


# ============================================================================
# view_with_strides Tests
# ============================================================================


fn test_view_with_strides_is_view() raises:
    """The view_with_strides method returns a tensor marked as a view."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])

    var new_shape = t2d.shape()
    var new_strides = List[Int]()
    new_strides.append(4)
    new_strides.append(1)
    var v = t2d.view_with_strides(new_shape, new_strides)

    assert_true(v._is_view)


fn test_view_with_strides_shape_strides() raises:
    """The view_with_strides method sets shape and strides from arguments."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])

    var new_shape = List[Int]()
    new_shape.append(4)
    new_shape.append(3)
    var new_strides = List[Int]()
    new_strides.append(1)
    new_strides.append(4)
    var v = t2d.view_with_strides(new_shape, new_strides)

    var shape = v.shape()
    assert_equal(shape[0], 4)
    assert_equal(shape[1], 3)
    assert_equal(v._strides[0], 1)
    assert_equal(v._strides[1], 4)


fn test_view_with_strides_numel() raises:
    """The view_with_strides method computes numel from new shape."""
    var t = arange(0.0, 24.0, 1.0, DType.float32)

    var new_shape = List[Int]()
    new_shape.append(2)
    new_shape.append(3)
    new_shape.append(4)
    var new_strides = List[Int]()
    new_strides.append(12)
    new_strides.append(4)
    new_strides.append(1)
    var v = t.view_with_strides(new_shape, new_strides)

    assert_equal(v.numel(), 24)


fn test_view_with_strides_shares_data() raises:
    """The view_with_strides method shares underlying data with original."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)

    var new_shape = List[Int]()
    new_shape.append(2)
    new_shape.append(3)
    var new_strides = List[Int]()
    new_strides.append(3)
    new_strides.append(1)
    var v = t.view_with_strides(new_shape, new_strides)

    # Both point to same data — raw pointer equality
    assert_equal(Int(v._data), Int(t._data))


# ============================================================================
# _nd_index_to_flat_offset Tests
# ============================================================================


fn test_nd_index_to_flat_offset_contiguous() raises:
    """For a contiguous tensor, offset equals index * dtype_size."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])

    var dtype_size = t2d._get_dtype_size()
    for i in range(t2d.numel()):
        assert_equal(t2d._nd_index_to_flat_offset(i), i * dtype_size)


fn test_nd_index_to_flat_offset_transposed() raises:
    """For a transposed 2D tensor, offset uses permuted strides."""
    # 2x3 tensor: row 0 = [0,1,2], row 1 = [3,4,5]
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var tx = t2d.transpose(0, 1)  # shape (3,2), strides [1, 3]

    # tx[0,0] -> element 0 of original (value 0)
    # tx[0,1] -> element 3 of original (value 3)
    # tx[1,0] -> element 1 of original (value 1)
    # C-order flat index 0 in tx shape (3,2): coord (0,0) -> offset = 0*1 + 0*3 = 0
    # flat index 1 in tx: coord (0,1) -> offset = 0*1 + 1*3 = 3
    var dtype_size = tx._get_dtype_size()
    assert_equal(tx._nd_index_to_flat_offset(0), 0 * dtype_size)
    assert_equal(tx._nd_index_to_flat_offset(1), 3 * dtype_size)
    assert_equal(tx._nd_index_to_flat_offset(2), 1 * dtype_size)
    assert_equal(tx._nd_index_to_flat_offset(3), 4 * dtype_size)


fn test_view_with_strides_element_read_transposed() raises:
    """Element read via __getitem__ on transposed view returns correct values."""
    # 2x3 tensor filled with [0..5]
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var tx = t2d.transpose(0, 1)  # shape (3, 2)

    # tx flat index 0 -> coord (0,0) in (3,2) -> original [0,0] = 0
    assert_almost_equal(Float64(tx[0]), 0.0, tolerance=1e-5)
    # tx flat index 1 -> coord (0,1) in (3,2) -> original [1,0] = 3
    assert_almost_equal(Float64(tx[1]), 3.0, tolerance=1e-5)
    # tx flat index 2 -> coord (1,0) in (3,2) -> original [0,1] = 1
    assert_almost_equal(Float64(tx[2]), 1.0, tolerance=1e-5)
    # tx flat index 5 -> coord (2,1) in (3,2) -> original [1,2] = 5
    assert_almost_equal(Float64(tx[5]), 5.0, tolerance=1e-5)


fn main() raises:
    """Run all view_with_strides and _nd_index_to_flat_offset tests."""
    test_view_with_strides_is_view()
    test_view_with_strides_shape_strides()
    test_view_with_strides_numel()
    test_view_with_strides_shares_data()
    test_nd_index_to_flat_offset_contiguous()
    test_nd_index_to_flat_offset_transposed()
    test_view_with_strides_element_read_transposed()
    print("All view_with_strides and _nd_index_to_flat_offset tests passed!")
