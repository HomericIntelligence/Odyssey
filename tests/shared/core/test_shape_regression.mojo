"""Tests for List[Int] constructor bugs in shape.mojo.

This test file demonstrates the bugs in shape.mojo functions that use the
unsafe pattern= List[Int]() followed by list[i] = value.

These tests SHOULD FAIL before the fixes are applied, demonstrating the bug.
After fixing, they should PASS.

Bugs tested:
- Line 48: reshape() - var final_shape = List[Int]()
- Line 121: squeeze(dim) - var new_shape : List[Int] = [ndim - 1]
- Line 141: squeeze() - var new_shape = List[Int]()
- Line 177: unsqueeze() - var new_shape = List[Int]()
- Line 296: concatenate() - var result_shape = List[Int]()
"""

# Import AnyTensor and shape operations
from shared.tensor.any_tensor import AnyTensor, ones, zeros, arange
from shared.core.shape import reshape, squeeze, unsqueeze, flatten, concatenate

# Import test helpers
from tests.shared.conftest import (
    assert_dim,
    assert_numel,
    assert_value_at,
    assert_all_close,
)


# ============================================================================
# Test reshape() bug (Line 48)
# ============================================================================


fn test_reshape_with_inferred_dimension() raises:
    """Test reshape with -1 dimension (triggers List constructor bug at line 48).

    Bug: Line 48-59 uses List[Int]() then indexes it.
    This crashes because the list has undefined size.
    """
    # Create a 1D tensor with 12 elements
    var a = arange(0.0, 12.0, 1.0, DType.float32)  # Shape (12,)

    # Reshape to (3, -1) should infer -1 as 4
    var shape = List[Int]()
    shape.append(3)
    shape.append(-1)  # Should infer 4

    # This will crash due to the bug at line 48-59
    var b = reshape(a, shape)

    # If we get here, the bug is fixed
    assert_dim(b, 2, "Result should be 2D")
    assert_numel(b, 12, "Result should have 12 elements")


fn test_reshape_explicit_shape() raises:
    """Test reshape with explicit dimensions (triggers List constructor bug at line 48).

    Bug: Line 62-65 uses List[Int]() then indexes it.
    This crashes because the list has undefined size.
    """
    # Create a 1D tensor with 12 elements
    var a = arange(0.0, 12.0, 1.0, DType.float32)  # Shape (12,)

    # Reshape to (3, 4) explicitly
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    # This will crash due to the bug at line 62-65
    var b = reshape(a, shape)

    # If we get here, the bug is fixed
    assert_dim(b, 2, "Result should be 2D")
    assert_numel(b, 12, "Result should have 12 elements")


# ============================================================================
# Test squeeze() bugs (Lines 121, 141)
# ============================================================================


fn test_squeeze_specific_dimension() raises:
    """Test squeeze with specific dimension (triggers bug at line 121).

    Bug: Line 121 uses List[Int](ndim - 1) then indexes at line 125.
    This crashes because the list has undefined size.
    """
    # Create tensor with shape (1, 3, 1, 4)
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(1)
    shape.append(4)
    var a = ones(shape, DType.float32)

    # Squeeze dimension 0 (size 1)
    # This will crash due to bug at line 121-125
    var b = squeeze(a, 0)

    # If we get here, the bug is fixed
    assert_dim(b, 3, "Result should be 3D")
    assert_numel(b, 12, "Result should have 12 elements")


fn test_squeeze_all_dimensions() raises:
    """Test squeeze all size-1 dimensions (triggers bug at line 141).

    Bug: Line 141 uses List[Int]() then indexes at line 145.
    This crashes because the list has undefined size.
    """
    # Create tensor with shape (1, 3, 1, 4)
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(1)
    shape.append(4)
    var a = ones(shape, DType.float32)

    # Squeeze all size-1 dimensions (default behavior)
    # This will crash due to bug at line 141-145
    var b = squeeze(a)

    # If we get here, the bug is fixed
    assert_dim(b, 2, "Result should be 2D (squeezed 2 dims)")
    assert_numel(b, 12, "Result should have 12 elements")


# ============================================================================
# Test unsqueeze() bug (Line 177)
# ============================================================================


fn test_unsqueeze_add_dimension() raises:
    """Test unsqueeze to add dimension (triggers bug at line 177).

    Bug: Line 177 uses List[Int]() then indexes at line 181/183.
    This crashes because the list has undefined size.
    """
    # Create tensor with shape (3, 4)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)

    # Add dimension at position 0 -> (1, 3, 4)
    # This will crash due to bug at line 177-183
    var b = unsqueeze(a, 0)

    # If we get here, the bug is fixed
    assert_dim(b, 3, "Result should be 3D")
    assert_numel(b, 12, "Result should have 12 elements")


fn test_unsqueeze_negative_index() raises:
    """Test unsqueeze with negative index (triggers bug at line 177).

    Bug: Line 177 uses List[Int]() then indexes at line 181/183.
    This crashes because the list has undefined size.
    """
    # Create tensor with shape (3, 4)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)

    # Add dimension at end -> (3, 4, 1)
    # This will crash due to bug at line 177-183
    var b = unsqueeze(a, -1)

    # If we get here, the bug is fixed
    assert_dim(b, 3, "Result should be 3D")
    assert_numel(b, 12, "Result should have 12 elements")


# ============================================================================
# Test concatenate() bug (Line 296)
# ============================================================================


fn test_concatenate_along_axis() raises:
    """Test concatenate tensors (triggers bug at line 296).

    Bug: Line 296 uses List[Int]() then indexes at line 299/301.
    This crashes because the list has undefined size.
    """
    # Create two tensors to concatenate
    var shape1 = List[Int]()
    shape1.append(2)
    shape1.append(3)
    var a = ones(shape1, DType.float32)  # 2x3

    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(3)
    var b = ones(shape2, DType.float32)  # 3x3

    # Concatenate along axis 0
    var tensors: List[AnyTensor] = []
    tensors.append(a)
    tensors.append(b)

    # This will crash due to bug at line 296-301
    var c = concatenate(tensors, axis=0)

    # If we get here, the bug is fixed
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 15, "Result should be 5x3 (15 elements)")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run all shape.mojo bug tests.

    These tests demonstrate the List[Int] constructor bugs in shape.mojo.
    They SHOULD FAIL before fixes are applied, and PASS after.
    """
    print("Running shape.mojo List[Int] constructor bug tests...")
    print("WARNING: These tests may crash before fixes are applied!")
    print("")

    # reshape() bugs
    print("  Testing reshape() bugs...")
    try:
        test_reshape_with_inferred_dimension()
        print("    ✓ reshape with -1 dimension")
    except e:
        print("    ✗ reshape with -1 dimension CRASHED:", String(e))

    try:
        test_reshape_explicit_shape()
        print("    ✓ reshape explicit shape")
    except e:
        print("    ✗ reshape explicit shape CRASHED:", String(e))

    # squeeze() bugs
    print("  Testing squeeze() bugs...")
    try:
        test_squeeze_specific_dimension()
        print("    ✓ squeeze specific dimension")
    except e:
        print("    ✗ squeeze specific dimension CRASHED:", String(e))

    try:
        test_squeeze_all_dimensions()
        print("    ✓ squeeze all dimensions")
    except e:
        print("    ✗ squeeze all dimensions CRASHED:", String(e))

    # unsqueeze() bugs
    print("  Testing unsqueeze() bugs...")
    try:
        test_unsqueeze_add_dimension()
        print("    ✓ unsqueeze add dimension")
    except e:
        print("    ✗ unsqueeze add dimension CRASHED:", String(e))

    try:
        test_unsqueeze_negative_index()
        print("    ✓ unsqueeze negative index")
    except e:
        print("    ✗ unsqueeze negative index CRASHED:", String(e))

    # concatenate() bug
    print("  Testing concatenate() bugs...")
    try:
        test_concatenate_along_axis()
        print("    ✓ concatenate along axis")
    except e:
        print("    ✗ concatenate along axis CRASHED:", String(e))

    print("")
    print("shape.mojo bug tests completed!")
    print("If any tests crashed, the bugs are still present.")
    print("After fixing, all tests should pass.")
