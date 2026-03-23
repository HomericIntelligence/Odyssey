# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_setitem_view.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for __setitem__ and __getitem__ on view (non-contiguous) tensors (Part 1 of 2).

Covers:
- Sliced 1D and multi-dim view tensor write-back
- No corruption of adjacent elements after view write
- Transpose/non-contiguous view property tests
- Integer dtype view

Issue: #4076 — Verify multi-dim __setitem__ works with non-contiguous (view) tensors
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, arange
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
    assert_value_at(view, 0, 2.0, "view[0] should be 2.0")
    assert_value_at(view, 5, 7.0, "view[5] should be 7.0")


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
    assert_value_at(original, 0, 0.0, "original[0] should be 0.0 (before slice)")
    assert_value_at(original, 2, 99.0, "original[2] should be 99.0 (from view[0])")
    assert_value_at(original, 3, 88.0, "original[3] should be 88.0 (from view[1])")
    assert_value_at(original, 4, 77.0, "original[4] should be 77.0 (from view[2])")
    assert_value_at(original, 5, 0.0, "original[5] should be 0.0 (after slice)")


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

    assert_value_at(original, 0, 0.0, "original[0,0] should be 0.0 (outside slice)")
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
        assert_value_at(original, i, 1.0, f"original[{i}] should be 1.0")

    for i in range(7, 10):
        assert_value_at(original, i, 1.0, f"original[{i}] should be 1.0")

    # Check elements inside the slice are updated
    for i in range(3, 7):
        assert_value_at(original, i, 99.0, f"original[{i}] should be 99.0")


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
# Integer dtype view
# ============================================================================


fn test_setitem_view_int32_dtype() raises:
    """__setitem__ flat index on a transposed int32 view writes correct element."""
    var mat = zeros([2, 3], DType.int32)
    var v = mat.transpose(0, 1)  # shape (3, 2), strides [1, 3]

    # flat 1 on (3,2) = [0,1] -> buf 0*1+1*3=3 -> mat flat 3 = mat[1,0]
    v[1] = 88.0
    assert_almost_equal(Float64(v[1]), 88.0, tolerance=1e-6)
    assert_almost_equal(Float64(mat[3]), 88.0, tolerance=1e-6)


# ============================================================================
# Entry point
# ============================================================================


fn main() raises:
    print("Running __setitem__ view tensor tests (Part 1)...")

    # Slice view tests
    print("  test_setitem_view_has_correct_setup...")
    test_setitem_view_has_correct_setup()

    print("  test_setitem_view_1d_writes_correctly...")
    test_setitem_view_1d_writes_correctly()

    print("  test_setitem_view_multidim_writes_correctly...")
    test_setitem_view_multidim_writes_correctly()

    print("  test_setitem_view_does_not_corrupt_adjacent_elements...")
    test_setitem_view_does_not_corrupt_adjacent_elements()

    # Transpose/non-contiguous view tests
    print("  test_transpose_creates_noncontiguous_view...")
    test_transpose_creates_noncontiguous_view()

    print("  test_slice_creates_view...")
    test_slice_creates_view()

    print("  test_setitem_view_int32_dtype...")
    test_setitem_view_int32_dtype()

    print("All __setitem__ view tests (Part 1) passed!")
