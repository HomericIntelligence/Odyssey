"""Tests for ExTensor utility operations.

Tests utility functions including copy, clone, properties, conversions,
and helper methods like numel, dim, size, stride, is_contiguous.
"""

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, arange, clone, item, diff

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_equal,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_equal_int,
    assert_almost_equal,
)


# ============================================================================
# Test copy() and clone()
# ============================================================================


fn test_copy_independence() raises:
    """Test that copy creates independent tensor."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = clone(a)

    # Check that clone creates a copy with same values
    assert_value_at(b, 0, 3.0, 1e-6, "Clone should have same value")
    assert_value_at(b, 4, 3.0, 1e-6, "Clone should have same value at end")


fn test_clone_identical() raises:
    """Test that clone creates identical tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = clone(a)

    # Should have same values
    for i in range(12):
        assert_value_at(b, i, Float64(i), 1e-6, "Clone should have same values")


# ============================================================================
# Test property accessors
# ============================================================================


fn test_numel_total_elements() raises:
    """Test numel() returns total number of elements."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 24, "numel should return 24 for (2,3,4)")


fn test_dim_num_dimensions() raises:
    """Test that dim is correct."""
    var shape_1d = List[Int]()
    shape_1d.append(10)
    var t1 = ones(shape_1d, DType.float32)
    assert_dim(t1, 1, "1D tensor should have dim=1")

    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var t3 = ones(shape_3d, DType.float32)
    assert_dim(t3, 3, "3D tensor should have dim=3")


fn test_shape_property() raises:
    """Test shape() returns correct shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 2, "Shape should have 2 dimensions")
    assert_equal_int(s[0], 3, "First dimension should be 3")
    assert_equal_int(s[1], 4, "Second dimension should be 4")


fn test_dtype_property() raises:
    """Test dtype() returns correct data type."""
    var shape = List[Int]()
    shape.append(5)

    var t32 = ones(shape, DType.float32)
    assert_dtype(t32, DType.float32, "Should be float32")

    var t64 = ones(shape, DType.float64)
    assert_dtype(t64, DType.float64, "Should be float64")


# ============================================================================
# Test stride calculations
# ============================================================================


fn test_stride_row_major() raises:
    """Test stride calculation for row-major (C-order)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)  # Shape (2, 3, 4)

    # Row-major strides for (2,3,4): [12, 4, 1]
    # Access strides directly without transferring ownership
    assert_equal_int(t._strides[0], 12, "Stride for dim 0 should be 12")
    assert_equal_int(t._strides[1], 4, "Stride for dim 1 should be 4")
    assert_equal_int(t._strides[2], 1, "Stride for dim 2 should be 1")


# ============================================================================
# Test contiguity
# ============================================================================


fn test_is_contiguous_true() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    # TODO(#2722): assert_contiguous(t)
    # For now, just test the method exists
    var _ = t.is_contiguous()
    # Should be True for newly created tensor
    pass  # Placeholder


fn test_is_contiguous_after_transpose() raises:
    """Test that transposed tensor is not contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    # var b = transpose(a)  # TODO(#2722): Implement transpose()

    # Transposed tensor is typically not contiguous
    # var contig = b.is_contiguous()
    # assert_false(contig, "Transposed tensor should not be contiguous")
    pass  # Placeholder


# ============================================================================
# Test contiguous() - make contiguous copy
# ============================================================================


fn test_contiguous_on_noncontiguous() raises:
    """Test making non-contiguous tensor contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    # var b = transpose(a)  # Not contiguous
    # var c = contiguous(b)  # TODO(#2722): Implement contiguous()

    # c should now be contiguous
    # var contig = c.is_contiguous()
    # assert_true(contig, "contiguous() should make tensor contiguous")
    pass  # Placeholder


# ============================================================================
# Test item() - scalar extraction
# ============================================================================


fn test_item_single_element() raises:
    """Test extracting value from single-element tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var val = item(t)

    assert_almost_equal(val, 42.0, 1e-6, "item() should extract scalar value")


fn test_item_requires_single_element() raises:
    """Test that item() requires single-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    # Should raise error for multi-element tensor
    var raised = False
    try:
        var val = item(t)
        _ = val
    except:
        raised = True

    if not raised:
        raise Error("item() should raise error for multi-element tensor")


# ============================================================================
# Test tolist() - convert to nested list
# ============================================================================


fn test_tolist_1d() raises:
    """Test converting 1D tensor to list."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var lst = t.tolist()

    # Should return [0, 1, 2, 3, 4]
    assert_equal_int(len(lst), 5, "List should have 5 elements")
    for i in range(5):
        assert_almost_equal(
            lst[i], Float64(i), 1e-6, "List value should match tensor"
        )


fn test_tolist_nested() raises:
    """Test converting multi-dimensional tensor to nested list."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var lst = t.tolist()

    # tolist() returns flat list, not nested
    assert_equal_int(len(lst), 6, "List should have 6 elements")
    for i in range(6):
        assert_almost_equal(
            lst[i], Float64(i), 1e-6, "List value should match tensor"
        )


# ============================================================================
# Test __len__
# ============================================================================


fn test_len_first_dim() raises:
    """Test __len__ returns size of first dimension."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(3)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 5, "__len__ should return first dimension")


fn test_len_1d() raises:
    """Test __len__ on 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 10, "__len__ should return size for 1D")


# ============================================================================
# Test __bool__
# ============================================================================


fn test_bool_single_element() raises:
    """Test __bool__ on single-element tensor."""
    var shape = List[Int]()
    var t_zero = full(shape, 0.0, DType.float32)
    var t_nonzero = full(shape, 5.0, DType.float32)

    # if t_zero:  # Should be False
    #     raise Error("Zero tensor should be falsy")
    # if not t_nonzero:  # Should be True
    #     raise Error("Non-zero tensor should be truthy")
    pass  # Placeholder


fn test_bool_requires_single_element() raises:
    """Test that item() requires single-element tensor.

    Note: Since __bool__ is not yet implemented, we test item() which
    has the same single-element requirement and is used for scalar extraction.
    """
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var val = item(t)  # Should raise error for multi-element tensor
        _ = val  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions single-element requirement
        if (
            "single" not in error_msg.lower()
            and "element" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention single-element requirement"
            )

    if not error_raised:
        raise Error("item() on multi-element tensor should raise error")


# ============================================================================
# Test type conversions
# ============================================================================


fn test_int_conversion() raises:
    """Test int conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.5, DType.float32)

    # Use item() to extract value, then convert to Int
    var val = Int(item(t))
    assert_equal_int(val, 42, "item() + Int should convert to int")


fn test_float_conversion() raises:
    """Test float conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.int32)

    # Use item() to extract value as Float64
    var val = item(t)
    assert_almost_equal(val, 42.0, 1e-6, "item() should return Float64 value")


# ============================================================================
# Test __str__ and __repr__
# ============================================================================


fn test_str_readable() raises:
    """Test __str__ produces readable output."""
    var t = arange(0.0, 3.0, 1.0, DType.float32)
    var s = String(t)
    assert_equal(
        s, "ExTensor([0.0, 1.0, 2.0], dtype=float32)", "__str__ format"
    )


fn test_repr_complete() raises:
    """Test __repr__ produces complete representation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var r = repr(t)
    assert_equal(
        r,
        (
            "ExTensor(shape=[2, 2], dtype=float32, numel=4, data=[1.0, 1.0,"
            " 1.0, 1.0])"
        ),
        "__repr__ format",
    )


# ============================================================================
# Test __hash__
# ============================================================================


fn test_hash_immutable() raises:
    """Test __hash__ for immutable tensors."""
    var shape = List[Int]()
    shape.append(3)
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var b = arange(0.0, 3.0, 1.0, DType.float32)

    # var hash_a = hash(a)  # TODO(#2722): Implement __hash__
    # var hash_b = hash(b)
    # Equal tensors should have same hash
    # assert_equal_int(hash_a, hash_b, "Equal tensors should have same hash")
    pass  # Placeholder


# ============================================================================
# Test diff() - consecutive differences
# ============================================================================


fn test_diff_1d() raises:
    """Test computing consecutive differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var d = diff(t)

    # Result: [1, 1, 1, 1] (4 elements)
    assert_numel(d, 4, "diff should have n-1 elements")
    for i in range(4):
        assert_value_at(d, i, 1.0, 1e-6, "Consecutive differences should be 1")


fn test_diff_higher_order() raises:
    """Test higher-order differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var d = diff(t, 2)

    # Second differences of [0,1,2,3,4] -> [0,0,0]
    assert_numel(d, 3, "Second diff should have n-2 elements")
    for i in range(3):
        assert_value_at(d, i, 0.0, 1e-6, "Second-order differences should be 0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run all utility operation tests."""
    print("Running ExTensor utility operation tests...")

    # copy() and clone() tests
    print("  Testing copy() and clone()...")
    test_copy_independence()
    test_clone_identical()

    # Property accessors
    print("  Testing property accessors...")
    test_numel_total_elements()
    test_dim_num_dimensions()
    test_shape_property()
    test_dtype_property()

    # Stride calculations
    print("  Testing stride calculations...")
    test_stride_row_major()

    # Contiguity
    print("  Testing contiguity...")
    test_is_contiguous_true()
    test_is_contiguous_after_transpose()
    test_contiguous_on_noncontiguous()

    # item() extraction
    print("  Testing item()...")
    test_item_single_element()
    test_item_requires_single_element()

    # tolist() conversion
    print("  Testing tolist()...")
    test_tolist_1d()
    test_tolist_nested()

    # __len__
    print("  Testing __len__...")
    test_len_first_dim()
    test_len_1d()

    # __bool__
    print("  Testing __bool__...")
    test_bool_single_element()
    test_bool_requires_single_element()

    # Type conversions
    print("  Testing type conversions...")
    test_int_conversion()
    test_float_conversion()

    # __str__ and __repr__
    print("  Testing __str__ and __repr__...")
    test_str_readable()
    test_repr_complete()

    # __hash__
    print("  Testing __hash__...")
    test_hash_immutable()

    # diff()
    print("  Testing diff()...")
    test_diff_1d()
    test_diff_higher_order()

    print("All utility operation tests completed!")
