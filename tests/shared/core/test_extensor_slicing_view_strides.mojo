# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Tests for AnyTensor view/stride semantics via transpose and reshape.

The methods view_with_strides() and _nd_index_to_flat_offset() were removed
from AnyTensor. These tests validate the same stride-based view behaviour
using the current public API: transpose(), reshape(), view() from shape.mojo,
and multi-dimensional __getitem__.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, arange
from shared.core.shape import view
from tests.shared.conftest import assert_true, assert_false, assert_almost_equal, assert_equal


# ============================================================================
# transpose view tests (replacements for view_with_strides tests)
# ============================================================================


def test_transpose_is_view() raises:
    """Transpose() returns a tensor marked as a view."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var tx = t2d.transpose(0, 1)  # shape (3, 2)

    assert_true(tx._is_view)


def test_transpose_shape_strides() raises:
    """Transpose() permutes shape and strides correctly."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])  # strides [4, 1]
    var tx = t2d.transpose(0, 1)  # shape (4, 3), strides [1, 4]

    var shape = tx.shape()
    assert_equal(shape[0], 4)
    assert_equal(shape[1], 3)
    assert_equal(tx._strides[0], 1)
    assert_equal(tx._strides[1], 4)


def test_reshape_preserves_numel() raises:
    """Reshape() preserves numel across shape changes."""
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t3d = t.reshape([2, 3, 4])

    assert_equal(t3d.numel(), 24)

    var shape = t3d.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 4)


def test_transpose_shares_data() raises:
    """Transpose() shares underlying data pointer with original."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var tx = t2d.transpose(0, 1)

    # Both point to same data — raw pointer equality
    assert_equal(Int(tx._data), Int(t2d._data))


# ============================================================================
# Stride-aware element access tests
# (replacements for _nd_index_to_flat_offset tests)
# ============================================================================


def test_contiguous_element_access() raises:
    """Contiguous tensor element access via multi-dim indices is correct."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])

    # t2d[i, j] should equal i * 4 + j
    for i in range(3):
        for j in range(4):
            var idx = List[Int]()
            idx.append(i)
            idx.append(j)
            var expected = Float32(i * 4 + j)
            assert_almost_equal(Float64(t2d[idx]), Float64(expected), tolerance=1e-5)


def test_transposed_element_access() raises:
    """Transposed tensor element access via multi-dim indices uses permuted strides."""
    # 2x3 tensor: row 0 = [0,1,2], row 1 = [3,4,5]
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2d = t.reshape([2, 3])
    var tx = t2d.transpose(0, 1)  # shape (3,2), strides [1, 3]

    # tx[0,0] -> original [0,0] = 0
    var idx00 = List[Int]()
    idx00.append(0)
    idx00.append(0)
    assert_almost_equal(Float64(tx[idx00]), 0.0, tolerance=1e-5)

    # tx[0,1] -> original [1,0] = 3
    var idx01 = List[Int]()
    idx01.append(0)
    idx01.append(1)
    assert_almost_equal(Float64(tx[idx01]), 3.0, tolerance=1e-5)

    # tx[1,0] -> original [0,1] = 1
    var idx10 = List[Int]()
    idx10.append(1)
    idx10.append(0)
    assert_almost_equal(Float64(tx[idx10]), 1.0, tolerance=1e-5)

    # tx[2,1] -> original [1,2] = 5
    var idx21 = List[Int]()
    idx21.append(2)
    idx21.append(1)
    assert_almost_equal(Float64(tx[idx21]), 5.0, tolerance=1e-5)


def test_view_creates_zero_copy() raises:
    """View() from shape.mojo creates a zero-copy tensor with new shape."""
    var t = ones([2, 3], DType.float32)
    var v = view(t, [6])

    assert_equal(v.numel(), 6)
    var shape = v.shape()
    assert_equal(len(shape), 1)
    assert_equal(shape[0], 6)

    # All elements should be 1.0
    for i in range(6):
        assert_almost_equal(Float64(v[i]), 1.0, tolerance=1e-5)


def main() raises:
    """Run all view/stride tests."""
    test_transpose_is_view()
    test_transpose_shape_strides()
    test_reshape_preserves_numel()
    test_transpose_shares_data()
    test_contiguous_element_access()
    test_transposed_element_access()
    test_view_creates_zero_copy()
    print("All view/stride tests passed!")
