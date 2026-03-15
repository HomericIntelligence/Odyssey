"""Tests for __setitem__ on view/non-contiguous tensors.

Verifies that __setitem__ correctly handles writes to view tensors (tensors
with non-identity strides that share memory with parent tensors). Tests cover
both the case where the fix is implemented and documents current behavior.

See Issue #4076 for context.
"""

from shared.core import ExTensor, zeros, ones, arange
from tests.shared.conftest import assert_value_at


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
    var original = zeros(List[Int](10), DType.float32)
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
    var original = zeros(List[Int](3, 4), DType.float32)
    var view = original[1:3, 1:3]

    # Write to the view (it should be 2x2)
    if view.shape()[0] != 2 or view.shape()[1] != 2:
        raise Error("View shape should be (2, 2)")

    # Set using List[Int] multi-index
    var idx1 = List[Int](0, 0)
    var idx2 = List[Int](0, 1)
    var idx3 = List[Int](1, 0)
    var idx4 = List[Int](1, 1)

    view.__setitem__(idx1, 11.0)
    view.__setitem__(idx2, 12.0)
    view.__setitem__(idx3, 21.0)
    view.__setitem__(idx4, 22.0)

    # Verify positions in original tensor
    # view[0,0] = original[1,1], view[0,1] = original[1,2],
    # view[1,0] = original[2,1], view[1,1] = original[2,2]
    var idx_11 = List[Int](1, 1)
    var idx_12 = List[Int](1, 2)
    var idx_21 = List[Int](2, 1)
    var idx_22 = List[Int](2, 2)

    assert_value_at(original, 0, 0.0, "original[0,0] should be 0.0 (outside slice)")
    assert_value_at(original.__getitem__(idx_11), Float32(11.0), "original[1,1]")
    assert_value_at(original.__getitem__(idx_12), Float32(12.0), "original[1,2]")
    assert_value_at(original.__getitem__(idx_21), Float32(21.0), "original[2,1]")
    assert_value_at(original.__getitem__(idx_22), Float32(22.0), "original[2,2]")


fn test_setitem_view_does_not_corrupt_adjacent_elements() raises:
    """Verify that writing to a view doesn't corrupt adjacent memory.

    Write to a sliced tensor and verify that elements outside the slice
    remain unchanged.
    """
    var original = ones(List[Int](10), DType.float32)
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
# Entry point
# ============================================================================


fn main() raises:
    print("Running __setitem__ view tensor tests...")

    print("  test_setitem_view_has_correct_setup...")
    test_setitem_view_has_correct_setup()

    print("  test_setitem_view_1d_writes_correctly...")
    test_setitem_view_1d_writes_correctly()

    print("  test_setitem_view_multidim_writes_correctly...")
    test_setitem_view_multidim_writes_correctly()

    print("  test_setitem_view_does_not_corrupt_adjacent_elements...")
    test_setitem_view_does_not_corrupt_adjacent_elements()

    print("All __setitem__ view tests passed!")
