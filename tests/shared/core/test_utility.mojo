"""Tests for ExTensor utility operations.

Tests utility functions including copy, clone, properties, conversions,
and helper methods like numel, dim, size, stride, is_contiguous.
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    clone,
    item,
    diff,
    as_contiguous,
    transpose_view,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_equal,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_equal_int,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
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

    assert_true(t.is_contiguous(), "Newly created tensor should be contiguous")


fn test_is_contiguous_after_transpose() raises:
    """Test that a transposed tensor is not contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = a.transpose(0, 1)
    assert_false(
        b.is_contiguous(), "Transposed tensor should not be contiguous"
    )


# ============================================================================
# Test contiguous() - make contiguous copy
# ============================================================================


fn test_contiguous_on_noncontiguous() raises:
    """Test making non-contiguous tensor contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = a.reshape(shape)
    var t = b.transpose(0, 1)
    assert_false(
        t.is_contiguous(), "Transposed tensor should not be contiguous"
    )

    # as_contiguous() should produce a contiguous copy
    var c = as_contiguous(t)
    assert_true(
        c.is_contiguous(), "as_contiguous() result should be contiguous"
    )

    # Result should have row-major strides [4, 1] for shape (4, 3) after transpose
    assert_equal_int(c._strides[0], 3, "Stride for dim 0 should be 3 (rows)")
    assert_equal_int(c._strides[1], 1, "Stride for dim 1 should be 1")

    # Verify element values are correctly reordered per transpose stride mapping.
    # Original (3,4) row-major: a[i,j] = i*4 + j (values 0..11)
    # After transpose to (4,3), reading row-major: t[j,i] = a[i,j]
    # Row 0 of transpose = col 0 of original: 0, 4, 8
    assert_almost_equal(c._get_float64(0), 0.0, 1e-6, "c[0,0] should be 0")
    assert_almost_equal(c._get_float64(1), 4.0, 1e-6, "c[0,1] should be 4")
    assert_almost_equal(c._get_float64(2), 8.0, 1e-6, "c[0,2] should be 8")
    # Row 1 of transpose = col 1 of original: 1, 5, 9
    assert_almost_equal(c._get_float64(3), 1.0, 1e-6, "c[1,0] should be 1")
    assert_almost_equal(c._get_float64(4), 5.0, 1e-6, "c[1,1] should be 5")
    assert_almost_equal(c._get_float64(5), 9.0, 1e-6, "c[1,2] should be 9")
    # Row 2 of transpose = col 2 of original: 2, 6, 10
    assert_almost_equal(c._get_float64(6), 2.0, 1e-6, "c[2,0] should be 2")
    assert_almost_equal(c._get_float64(7), 6.0, 1e-6, "c[2,1] should be 6")
    assert_almost_equal(c._get_float64(8), 10.0, 1e-6, "c[2,2] should be 10")
    # Row 3 of transpose = col 3 of original: 3, 7, 11
    assert_almost_equal(c._get_float64(9), 3.0, 1e-6, "c[3,0] should be 3")
    assert_almost_equal(c._get_float64(10), 7.0, 1e-6, "c[3,1] should be 7")
    assert_almost_equal(c._get_float64(11), 11.0, 1e-6, "c[3,2] should be 11")


fn test_as_contiguous_values_correct() raises:
    """Test that as_contiguous() copies correct element values from non-contiguous views.

    This is a regression test for the bug where the non-contiguous branch used
    flat index offset (i * dtype_size) instead of stride-based multi-dimensional
    indexing, causing silently wrong results for transposed/sliced tensors.

    For a (3,4) tensor with values 0..11 reshaped, the transpose (4,3) should have:
      row 0: [0, 4, 8]    (original column 0)
      row 1: [1, 5, 9]    (original column 1)
      row 2: [2, 6, 10]   (original column 2)
      row 3: [3, 7, 11]   (original column 3)
    """
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = a.reshape(shape)
    var t = transpose_view(b)  # shape (4, 3), non-contiguous

    var c = as_contiguous(t)

    # Verify shape (4, 3)
    var c_shape = c.shape()
    assert_equal_int(
        c_shape[0], 4, "as_contiguous transpose: dim 0 should be 4"
    )
    assert_equal_int(
        c_shape[1], 3, "as_contiguous transpose: dim 1 should be 3"
    )

    # Expected values: column-major read of original 0..11 row-major (3,4)
    # Row 0 of transpose = col 0 of original: 0, 4, 8
    assert_almost_equal(c._get_float64(0), 0.0, 1e-6, "c[0,0] should be 0")
    assert_almost_equal(c._get_float64(1), 4.0, 1e-6, "c[0,1] should be 4")
    assert_almost_equal(c._get_float64(2), 8.0, 1e-6, "c[0,2] should be 8")
    # Row 1 of transpose = col 1 of original: 1, 5, 9
    assert_almost_equal(c._get_float64(3), 1.0, 1e-6, "c[1,0] should be 1")
    assert_almost_equal(c._get_float64(4), 5.0, 1e-6, "c[1,1] should be 5")
    assert_almost_equal(c._get_float64(5), 9.0, 1e-6, "c[1,2] should be 9")
    # Row 2 of transpose = col 2 of original: 2, 6, 10
    assert_almost_equal(c._get_float64(6), 2.0, 1e-6, "c[2,0] should be 2")
    assert_almost_equal(c._get_float64(7), 6.0, 1e-6, "c[2,1] should be 6")
    assert_almost_equal(c._get_float64(8), 10.0, 1e-6, "c[2,2] should be 10")
    # Row 3 of transpose = col 3 of original: 3, 7, 11
    assert_almost_equal(c._get_float64(9), 3.0, 1e-6, "c[3,0] should be 3")
    assert_almost_equal(c._get_float64(10), 7.0, 1e-6, "c[3,1] should be 7")
    assert_almost_equal(c._get_float64(11), 11.0, 1e-6, "c[3,2] should be 11")


fn test_as_contiguous_3d_values_correct() raises:
    """Test as_contiguous() on a 3D non-contiguous tensor with permuted strides.

    This tests that stride-based indexing works correctly for arbitrary dimensions,
    not just 2D cases. Creates a 3D tensor and verifies that the contiguous copy
    correctly maps elements even with complex stride patterns.

    For a (2,3,4) tensor reshaped from 0..23 row-major, we transpose dimensions
    and verify the result is contiguous and has correct element values.
    """
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)

    # Create a 3D tensor with values 0..23 in row-major order
    var a_flat = arange(0.0, 24.0, 1.0, DType.float32)
    var a = a_flat.reshape(shape_3d)

    # Transpose to create non-contiguous view: (2,3,4) -> (4,3,2)
    # This swaps dimensions 0 and 2
    var t = a.transpose(2, 0)
    assert_false(t.is_contiguous(), "Transposed 3D tensor should not be contiguous")

    # Make contiguous
    var c = as_contiguous(t)
    assert_true(c.is_contiguous(), "as_contiguous() result should be contiguous")

    # Verify shape (4,3,2) after transpose
    var c_shape = c.shape()
    assert_equal_int(c_shape[0], 4, "Shape[0] should be 4 after transpose")
    assert_equal_int(c_shape[1], 3, "Shape[1] should be 3 after transpose")
    assert_equal_int(c_shape[2], 2, "Shape[2] should be 2 after transpose")

    # Verify totalcontent (all 24 elements)
    assert_equal_int(c.numel(), 24, "Contiguous 3D tensor should have 24 elements")

    # Verify that the transpose correctly maps elements from original (2,3,4) to result (4,3,2)
    # Original element at [i,j,k] goes to transposed position [k,j,i]
    # So we check a few cross-section values to ensure stride-based indexing is correct

    # Original [0,0,0]=0 -> transposed [0,0,0]
    assert_almost_equal(c._get_float64(0), 0.0, 1e-6, "First element should be 0")

    # Original [0,0,3]=3 -> transposed [3,0,0]
    # Transposed flat index [3,0,0]: 3*6 + 0*2 + 0 = 18
    assert_almost_equal(c._get_float64(18), 3.0, 1e-6, "Element from [0,0,3]")

    # Original [1,2,3]=23 -> transposed [3,2,1]
    # Transposed flat index [3,2,1]: 3*6 + 2*2 + 1 = 23
    assert_almost_equal(c._get_float64(23), 23.0, 1e-6, "Element from [1,2,3]")

    # Original [1,0,0]=12 -> transposed [0,0,1]
    # Transposed flat index [0,0,1]: 0*6 + 0*2 + 1 = 1
    assert_almost_equal(c._get_float64(1), 12.0, 1e-6, "Element from [1,0,0]")


# ============================================================================
# Test transpose() - edge cases
# ============================================================================


fn test_transpose_1d_tensor_raises() raises:
    """Test that transpose on 1D tensor raises an error.

    The error message should indicate that transpose requires at least 2 dimensions.
    """
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var result = t.transpose(0, 1)
    except e:
        error_raised = True

    assert_true(
        error_raised, "transpose on 1D tensor should raise an error"
    )


fn test_transpose_out_of_range_dim0() raises:
    """Test that transpose with out-of-range dim0 raises an error."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var result = t.transpose(5, 1)  # dim0=5 is out of range
    except e:
        error_raised = True

    assert_true(
        error_raised, "transpose with out-of-range dim0 should raise an error"
    )


fn test_transpose_out_of_range_dim1() raises:
    """Test that transpose with out-of-range dim1 raises an error."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var result = t.transpose(0, 10)  # dim1=10 is out of range
    except e:
        error_raised = True

    assert_true(
        error_raised, "transpose with out-of-range dim1 should raise an error"
    )


fn test_transpose_same_dim_identity() raises:
    """Test that transpose with dim0 == dim1 is a no-op (identity swap).

    When dim0 == dim1, the operation should be a no-op: shape and strides
    should remain unchanged, but it should still return a view.
    """
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = a.reshape(shape)

    # Transpose with same dimension (should be identity)
    var result = b.transpose(0, 0)

    # Shape should be unchanged
    var result_shape = result.shape()
    assert_equal_int(result_shape[0], 3, "Shape dim 0 should be 3")
    assert_equal_int(result_shape[1], 4, "Shape dim 1 should be 4")

    # Strides should be unchanged
    assert_equal_int(result._strides[0], b._strides[0], "Stride[0] should be unchanged")
    assert_equal_int(result._strides[1], b._strides[1], "Stride[1] should be unchanged")

    # Values should be identical
    assert_almost_equal(
        result._get_float64(0), 0.0, 1e-6, "Value at [0,0] should match"
    )
    assert_almost_equal(
        result._get_float64(5), 5.0, 1e-6, "Value at [1,1] should match"
    )


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
    except e:
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
# Test __setitem__
# ============================================================================


fn test_setitem_valid_index() raises:
    """Test setting value at valid flat index, verified with __getitem__."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t[1] = 9.5
    assert_value_at(t, 1, 9.5, 1e-6, "__setitem__ should set value at index 1")
    # Other elements unchanged
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0.0")
    assert_value_at(t, 2, 0.0, 1e-6, "Element 2 should remain 0.0")


fn test_setitem_integer_dtype() raises:
    """Test setting value on integer dtype tensor."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.int32)
    t[2] = 7.0
    assert_value_at(
        t, 2, 7.0, 1e-6, "__setitem__ should set value on int32 tensor"
    )
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0")


fn test_setitem_out_of_bounds() raises:
    """Test that __setitem__ raises error for out-of-bounds index."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)

    var error_raised = False
    try:
        t[5] = 1.0
    except e:
        error_raised = True
        assert_equal(String(e), "Index out of bounds")

    if not error_raised:
        raise Error("__setitem__ should raise error for out-of-bounds index")


fn test_setitem_negative_index() raises:
    """Test that __setitem__ raises error for negative index."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)

    var raised = False
    try:
        t[-1] = 1.0
    except e:
        raised = True

    if not raised:
        raise Error("__setitem__ should raise error for negative index")


fn test_getitem_negative_index() raises:
    """Test that __getitem__ raises error for negative index."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)

    var raised = False
    try:
        var _ = t[-1]
    except e:
        raised = True

    if not raised:
        raise Error("__getitem__ should raise error for negative index")


# ============================================================================
# Test __bool__
# ============================================================================


fn test_bool_single_element() raises:
    """Test __bool__ on single-element tensor."""
    var shape = List[Int]()
    var t_zero = full(shape, 0.0, DType.float32)
    var t_nonzero = full(shape, 5.0, DType.float32)

    if t_zero:
        raise Error("Zero tensor should be falsy")
    if not t_nonzero:
        raise Error("Non-zero tensor should be truthy")


fn test_bool_requires_single_element() raises:
    """Test that __bool__ raises for multi-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var val = t.__bool__()  # Should raise error for multi-element tensor
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
        raise Error("__bool__ on multi-element tensor should raise error")


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
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var b = arange(0.0, 3.0, 1.0, DType.float32)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a), Int(hash_b), "Equal tensors should have same hash"
    )


fn test_hash_different_values_differ() raises:
    """Test that tensors with different values produce different hashes."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float64)
    var b = full(shape, 2.0, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "Tensors with different values should have different hashes"
        )


fn test_hash_large_values() raises:
    """Test that large float values hash consistently without Int overflow."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1e15, DType.float64)
    var b = full(shape, 1e15, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a), Int(hash_b), "Large values must hash consistently"
    )


fn test_hash_small_values_distinguish() raises:
    """Test that small but distinct float values produce different hashes."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1e-7, DType.float64)
    var b = full(shape, 2e-7, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "Distinct small values should have different hashes with bitcast"
        )


fn test_hash_different_dtypes_differ() raises:
    """Test that same logical values with different dtypes produce different hashes.

    The hash implementation includes dtype ordinal to distinguish tensors,
    so a float32 and float64 tensor with identical values must hash differently.
    """
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 1.0, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "float32 and float64 tensors with same values should have different"
            " hashes"
        )


fn test_hash_different_shapes_differ() raises:
    """Test that tensors with same data but different shapes produce different hashes.
    """
    # Create [3] tensor with values [1, 2, 3]
    var t1 = arange(1.0, 4.0, 1.0, DType.float32)

    # Create [1, 3] tensor with values [1, 2, 3]
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var t2 = arange(1.0, 4.0, 1.0, DType.float32)
    t2 = t2.reshape(shape)

    var hash_1 = hash(t1)
    var hash_2 = hash(t2)
    if hash_1 == hash_2:
        raise Error(
            "Tensors with same data but different shapes should have different"
            " hashes"
        )


fn test_hash_same_values_different_dtype() raises:
    """Test that tensors with same values but different dtypes produce different hashes.

    The dtype ordinal is included in the hash, so float32 and float64 tensors
    with identical numeric values must produce different hashes.
    """
    var shape = List[Int]()
    shape.append(1)
    var a_f32 = full(shape, 1.0, DType.float32)
    var a_f64 = full(shape, 1.0, DType.float64)

    var hash_f32 = hash(a_f32)
    var hash_f64 = hash(a_f64)
    if hash_f32 == hash_f64:
        raise Error(
            "Tensors with same values but different dtypes should have"
            " different hashes"
        )


fn test_hash_integer_dtype_consistent() raises:
    """Test __hash__ for integer-typed tensors produces consistent hashes.

    _get_float64 casts integer values to Float64 before hashing. Two separate
    tensors with identical integer values must produce the same hash.
    """
    var a = arange(0.0, 4.0, 1.0, DType.int32)
    var b = arange(0.0, 4.0, 1.0, DType.int32)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a),
        Int(hash_b),
        "Integer-typed tensors with same values should have same hash",
    )


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
    test_as_contiguous_values_correct()
    test_as_contiguous_3d_values_correct()

    # transpose() edge cases
    print("  Testing transpose() edge cases...")
    test_transpose_1d_tensor_raises()
    test_transpose_out_of_range_dim0()
    test_transpose_out_of_range_dim1()
    test_transpose_same_dim_identity()

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

    # __setitem__
    print("  Testing __setitem__...")
    test_setitem_valid_index()
    test_setitem_integer_dtype()
    test_setitem_out_of_bounds()
    test_setitem_negative_index()

    # __getitem__
    print("  Testing __getitem__...")
    test_getitem_negative_index()

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
    test_hash_different_values_differ()
    test_hash_large_values()
    test_hash_small_values_distinguish()
    test_hash_different_dtypes_differ()
    test_hash_different_shapes_differ()
    test_hash_integer_dtype_consistent()
    test_hash_same_values_different_dtype()

    # diff()
    print("  Testing diff()...")
    test_diff_1d()
    test_diff_higher_order()

    print("All utility operation tests completed!")
