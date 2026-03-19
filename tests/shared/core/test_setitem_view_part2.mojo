# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_setitem_view.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for flat-index get/set on transposed and slice views (Part 2 of 2).

Covers:
- __getitem__ flat index on transposed view returns correct value
- __setitem__ flat index on transposed view writes correct element
- Write-back via flat index on slice() view
- No corruption of adjacent elements after flat view write

Issue: #4076 — Verify multi-dim __setitem__ works with non-contiguous (view) tensors
"""

from shared.core.extensor import ExTensor, zeros, ones, arange
from tests.shared.conftest import (
    assert_true,
    assert_almost_equal,
    assert_equal,
    assert_value_at,
)


# ============================================================================
# Slice view setup and basic tests
# ============================================================================


fn test_setitem_view_has_correct_setup() raises:
    """Verify that sliced tensors are views with non-identity strides.

    This test confirms the test infrastructure is correct before testing
    write behavior.
    """
    # Create a 1D tensor and slice it
    var original = arange(0.0, 10.0, 1.0, DType.float32)
    var view = original[2:8]

    # Verify the view has correct properties
    if not view._is_view:
        raise Error("Sliced tensor should be a view")

    if view.numel() != 6:
        raise Error("Slice [2:8] should have 6 elements")

    # Element values should be correct
    assert_value_at(view, 0, 2.0, message="view[0] should be 2.0")
    assert_value_at(view, 5, 7.0, message="view[5] should be 7.0")


fn test_setitem_view_1d_writes_correctly() raises:
    """Test that __setitem__ on a 1D view writes to correct parent position.

    When we write to a sliced tensor, the write should affect the correct
    position in the original tensor (offset by slice start).
    """
    var original = zeros([10], DType.float32)
    var view = original[2:5]

    # Write to the view
    view[0] = 99.0
    view[1] = 88.0
    view[2] = 77.0

    # Verify writes affected the correct positions in the original
    assert_value_at(original, 0, 0.0, message="original[0] should be 0.0 (before slice)")
    assert_value_at(original, 2, 99.0, message="original[2] should be 99.0 (from view[0])")
    assert_value_at(original, 3, 88.0, message="original[3] should be 88.0 (from view[1])")
    assert_value_at(original, 4, 77.0, message="original[4] should be 77.0 (from view[2])")
    assert_value_at(original, 5, 0.0, message="original[5] should be 0.0 (after slice)")


fn test_setitem_view_multidim_writes_correctly() raises:
    """Test __setitem__ on a multi-dimensional sliced tensor.

    Slice a 2D tensor and verify writes go to the correct parent positions.
    """
    var original = zeros([3, 4], DType.float32)
    var view = original[1:3, 1:3]

    # Write to the view (it should be 2x2)
    if view.shape()[0] != 2 or view.shape()[1] != 2:
        raise Error("View shape should be (2, 2)")

    # Set using List[Int] multi-index
    var idx1 = [0, 0]
    var idx2 = [0, 1]
    var idx3 = [1, 0]
    var idx4 = [1, 1]

    view.__setitem__(idx1, 11.0)
    view.__setitem__(idx2, 12.0)
    view.__setitem__(idx3, 21.0)
    view.__setitem__(idx4, 22.0)

    # Verify positions in original tensor
    # view[0,0] = original[1,1], view[0,1] = original[1,2],
    # view[1,0] = original[2,1], view[1,1] = original[2,2]
    var idx_11 = [1, 1]
    var idx_12 = [1, 2]
    var idx_21 = [2, 1]
    var idx_22 = [2, 2]

    assert_value_at(original, 0, 0.0, message="original[0,0] should be 0.0 (outside slice)")
    var val_11 = original.__getitem__(idx_11)
    var val_12 = original.__getitem__(idx_12)
    var val_21 = original.__getitem__(idx_21)
    var val_22 = original.__getitem__(idx_22)

    if val_11 != Float32(11.0):
        raise Error("original[1,1] expected 11.0, got " + String(val_11))
    if val_12 != Float32(12.0):
        raise Error("original[1,2] expected 12.0, got " + String(val_12))
    if val_21 != Float32(21.0):
        raise Error("original[2,1] expected 21.0, got " + String(val_21))
    if val_22 != Float32(22.0):
        raise Error("original[2,2] expected 22.0, got " + String(val_22))


fn test_setitem_view_does_not_corrupt_adjacent_elements() raises:
    """Verify that writing to a view doesn't corrupt adjacent memory.

    Write to a sliced tensor and verify that elements outside the slice
    remain unchanged.
    """
    var original = ones([10], DType.float32)
    var view = original[3:7]

    # Write to view
    for i in range(4):
        view[i] = Float32(99.0)

    # Check elements outside the slice are unchanged
    for i in range(3):
        assert_value_at(original, i, 1.0, message=String("original[") + String(i) + String("] should be 1.0"))

    for i in range(7, 10):
        assert_value_at(original, i, 1.0, message=String("original[") + String(i) + String("] should be 1.0"))

    # Check elements inside the slice are updated
    for i in range(3, 7):
        assert_value_at(original, i, 99.0, message=String("original[") + String(i) + String("] should be 99.0"))


# ============================================================================
# Transpose/non-contiguous view property tests
# ============================================================================


fn test_transpose_creates_noncontiguous_view() raises:
    """Verify that transpose() produces _is_view=True with non-identity strides."""
    var t = zeros([3, 4], DType.float32)
    var v = t.transpose(0, 1)
    assert_true(v._is_view, "transpose should produce _is_view=True")
    assert_true(not v.is_contiguous(), "transpose should be non-contiguous")
    # shape (4, 3) with original row-major strides [4,1] swapped to [1,4]
    assert_equal(v._shape[0], 4, "transposed dim 0 should be 4")
    assert_equal(v._shape[1], 3, "transposed dim 1 should be 3")
    assert_equal(v._strides[0], 1, "transposed stride[0] should be 1")
    assert_equal(v._strides[1], 4, "transposed stride[1] should be 4")


fn test_slice_creates_view() raises:
    """Verify that slice() produces _is_view=True."""
    var t = zeros([6], DType.float32)
    var v = t.slice(2, 5, axis=0)
    assert_true(v._is_view, "slice() should produce _is_view=True")


# ============================================================================
# __getitem__ flat index on transposed view
# ============================================================================


fn test_getitem_flat_on_transposed_view() raises:
    """__getitem__ with flat index on a transposed view returns correct value.

    Original (2,3) tensor with values 0..5:
      [[0, 1, 2],
       [3, 4, 5]]

    Transposed to (3,2) with strides [1, 3]:
      [[0, 3],
       [1, 4],
       [2, 5]]

    Flat index 0 on (3,2) = logical [0,0] -> original [0,0] = 0.0
    Flat index 1 on (3,2) = logical [0,1] -> original [1,0] = 3.0
    Flat index 4 on (3,2) = logical [2,0] -> original [0,2] = 2.0
    """
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var mat = t.reshape([2, 3])
    var v = mat.transpose(0, 1)  # shape (3, 2), strides [1, 3]

    assert_almost_equal(Float64(v[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[2]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[3]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[4]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[5]), 5.0, tolerance=1e-6)


fn test_getitem_flat_on_transposed_3x4_view() raises:
    """__getitem__ on a (3,4) -> transpose -> (4,3) view returns correct values.

    Original (3,4) with values 0..11, strides [4,1].
    Transposed (4,3) strides [1,4]:
      v[i,j] corresponds to original [j,i]
    Flat index k in (4,3): dim_idx_0 = k//3, dim_idx_1 = k%3
      buffer_pos = dim_idx_0 * stride[0] + dim_idx_1 * stride[1]
                 = dim_idx_0 * 1 + dim_idx_1 * 4
    """
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var mat = t.reshape([3, 4])
    var v = mat.transpose(0, 1)  # shape (4, 3), strides [1, 4]

    # Verify selected elements
    # flat 0 = [0,0] -> buf 0*1+0*4=0 -> mat[0,0]=0
    assert_almost_equal(Float64(v[0]), 0.0, tolerance=1e-6)
    # flat 1 = [0,1] -> buf 0*1+1*4=4 -> mat[1,0]=4
    assert_almost_equal(Float64(v[1]), 4.0, tolerance=1e-6)
    # flat 2 = [0,2] -> buf 0*1+2*4=8 -> mat[2,0]=8
    assert_almost_equal(Float64(v[2]), 8.0, tolerance=1e-6)
    # flat 3 = [1,0] -> buf 1*1+0*4=1 -> mat[0,1]=1
    assert_almost_equal(Float64(v[3]), 1.0, tolerance=1e-6)
    # flat 5 = [1,2] -> buf 1*1+2*4=9 -> mat[2,1]=9
    assert_almost_equal(Float64(v[5]), 9.0, tolerance=1e-6)
    # flat 11 = [3,2] -> buf 3*1+2*4=11 -> mat[2,3]=11
    assert_almost_equal(Float64(v[11]), 11.0, tolerance=1e-6)


# ============================================================================
# __setitem__ flat index on transposed view
# ============================================================================


fn test_setitem_flat_on_transposed_view_writes_correct_element() raises:
    """Flat __setitem__ on a transposed view writes the correct element.

    Original (2,3) tensor, shape [2,3], strides [3,1].
    Transposed to (3,2), strides [1,3].
    Write at flat index 0 of view -> v[0,0] -> mat[0,0] (buffer pos 0).
    Write at flat index 3 of view -> v[1,1] -> mat[1,1] (buffer pos 4).
    """
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var mat = t.reshape([2, 3])
    var v = mat.transpose(0, 1)  # shape (3, 2), strides [1, 3]

    # Write 99.0 at flat index 0 of the view -> v[0,0] = mat[0,0] (buffer pos 0)
    v[0] = 99.0
    assert_almost_equal(Float64(v[0]), 99.0, tolerance=1e-6)
    # mat flat index 0 = mat[0,0] = buffer pos 0
    assert_almost_equal(Float64(mat[0]), 99.0, tolerance=1e-6)

    # Write 77.0 at flat index 3 of the view -> v[1,1] = mat[1,1] (buffer pos 4)
    v[3] = 77.0
    assert_almost_equal(Float64(v[3]), 77.0, tolerance=1e-6)
    # mat flat index 4 = mat[1,1] = buffer pos 4
    assert_almost_equal(Float64(mat[4]), 77.0, tolerance=1e-6)


fn test_setitem_flat_view_all_elements() raises:
    """Write to each flat index of transposed view and verify correct buffer update.

    Uses (2,3) -> transpose -> (3,2).  Strides [1,3].
    flat k: dim_idx_0 = k//2, dim_idx_1 = k%2 -> buf_pos = dim_idx_0*1 + dim_idx_1*3
    """
    var mat = zeros([2, 3], DType.float32)
    var v = mat.transpose(0, 1)  # shape (3, 2), strides [1, 3]

    # Write distinct values at each flat index
    v[0] = 0.0
    v[1] = 10.0
    v[2] = 20.0
    v[3] = 30.0
    v[4] = 40.0
    v[5] = 50.0

    # Verify via view reads
    assert_almost_equal(Float64(v[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[1]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[2]), 20.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[3]), 30.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[4]), 40.0, tolerance=1e-6)
    assert_almost_equal(Float64(v[5]), 50.0, tolerance=1e-6)

    # Verify correct buffer positions in mat (shape [2,3], strides [3,1])
    # flat k on (3,2) view: dim0 = k//2, dim1 = k%2 -> buf = dim0*1 + dim1*3
    # k=0: dim0=0,dim1=0 -> buf=0  -> mat flat idx 0 should be 0
    # k=1: dim0=0,dim1=1 -> buf=3  -> mat flat idx 3 should be 10
    # k=2: dim0=1,dim1=0 -> buf=1  -> mat flat idx 1 should be 20
    # k=3: dim0=1,dim1=1 -> buf=4  -> mat flat idx 4 should be 30
    # k=4: dim0=2,dim1=0 -> buf=2  -> mat flat idx 2 should be 40
    # k=5: dim0=2,dim1=1 -> buf=5  -> mat flat idx 5 should be 50
    assert_almost_equal(Float64(mat[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[3]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[1]), 20.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[4]), 30.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[2]), 40.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[5]), 50.0, tolerance=1e-6)


fn test_setitem_flat_view_does_not_corrupt_neighbors() raises:
    """Writing one flat-index element on a view leaves other elements unchanged."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var mat = t.reshape([2, 3])
    var v = mat.transpose(0, 1)  # shape (3, 2), strides [1, 3]

    # Overwrite only flat index 2 -> v[1,0] -> buf pos = 1*1 + 0*3 = 1 -> mat[0,1]=1->55
    v[2] = 55.0

    # Only position 2 in the view should be changed
    assert_almost_equal(Float64(v[0]), 0.0, tolerance=1e-6)  # v[0,0] = mat[0,0]
    assert_almost_equal(Float64(v[1]), 3.0, tolerance=1e-6)  # v[0,1] = mat[1,0]
    assert_almost_equal(Float64(v[2]), 55.0, tolerance=1e-6)  # changed
    assert_almost_equal(Float64(v[3]), 4.0, tolerance=1e-6)  # v[1,1] = mat[1,1]
    assert_almost_equal(Float64(v[4]), 2.0, tolerance=1e-6)  # v[2,0] = mat[0,2]
    assert_almost_equal(Float64(v[5]), 5.0, tolerance=1e-6)  # v[2,1] = mat[1,2]


fn test_setitem_on_transposed_view_updates_original() raises:
    """Writes via transposed view are visible through the original tensor."""
    var mat = zeros([3, 4], DType.float32)
    var v = mat.transpose(0, 1)  # shape (4, 3), strides [1, 4]

    # flat 0 on (4,3) = [0,0] -> buf 0*1+0*4=0 -> mat flat 0
    v[0] = 100.0
    assert_almost_equal(Float64(mat[0]), 100.0, tolerance=1e-6)

    # flat 5 on (4,3) = [1,2] -> dim0=5//3=1, dim1=5%3=2 -> buf 1*1+2*4=9 -> mat flat 9
    v[5] = 200.0
    assert_almost_equal(Float64(mat[9]), 200.0, tolerance=1e-6)

    # flat 11 on (4,3) = [3,2] -> dim0=11//3=3, dim1=11%3=2 -> buf 3*1+2*4=11 -> mat flat 11
    v[11] = 300.0
    assert_almost_equal(Float64(mat[11]), 300.0, tolerance=1e-6)


# ============================================================================
# Slice view write-back
# ============================================================================


fn test_setitem_on_slice_view_writes_to_parent() raises:
    """Writing to a slice() view propagates to the correct positions in the original."""
    var t = zeros([6], DType.float32)
    var v = t.slice(2, 5, axis=0)  # view of t[2:5], _is_view=True

    v[0] = 10.0
    v[1] = 20.0
    v[2] = 30.0

    # Verify original t was updated at positions 2, 3, 4
    assert_almost_equal(Float64(t[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(t[1]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(t[2]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(t[3]), 20.0, tolerance=1e-6)
    assert_almost_equal(Float64(t[4]), 30.0, tolerance=1e-6)
    assert_almost_equal(Float64(t[5]), 0.0, tolerance=1e-6)


# ============================================================================
# Entry point
# ============================================================================


fn main() raises:
    print("Running __setitem__ view tensor tests (Part 2)...")

    print("  test_getitem_flat_on_transposed_view...")
    test_getitem_flat_on_transposed_view()

    print("  test_getitem_flat_on_transposed_3x4_view...")
    test_getitem_flat_on_transposed_3x4_view()

    print("  test_setitem_flat_on_transposed_view_writes_correct_element...")
    test_setitem_flat_on_transposed_view_writes_correct_element()

    print("  test_setitem_flat_view_all_elements...")
    test_setitem_flat_view_all_elements()

    print("  test_setitem_flat_view_does_not_corrupt_neighbors...")
    test_setitem_flat_view_does_not_corrupt_neighbors()

    print("  test_setitem_on_transposed_view_updates_original...")
    test_setitem_on_transposed_view_updates_original()

    print("  test_setitem_on_slice_view_writes_to_parent...")
    test_setitem_on_slice_view_writes_to_parent()

    print("All __setitem__ view tests (Part 2) passed!")
