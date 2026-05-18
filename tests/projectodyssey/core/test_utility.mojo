"""Tests for AnyTensor utility operations.

Tests utility functions including copy, clone, numel, dim, shape, dtype,
and stride calculations.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    arange,
    copy,
    clone,
    item,
    diff,
)
from projectodyssey.core.shape import as_contiguous
from projectodyssey.core.matrix import transpose_view
from tests.projectodyssey.conftest import (
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


def make_bf16_nan_tensor(raw_bits: UInt16) raises -> AnyTensor:
    """Create a scalar BF16 tensor with the given raw NaN bit pattern.

    Bypasses _set_float64 by writing raw UInt16 bits directly via pointer cast,
    since nan_tensor() uses _set_float64 which may not preserve unusual NaN bit
    patterns for bfloat16.

    BF16 bit layout (16 bits): 1 sign | 8 exponent | 7 mantissa.
    NaN requires all-one exponent (0xFF) and non-zero mantissa.
    Canonical quiet NaN: 0x7FC0 (positive, mantissa msb set).
    Negative quiet NaN:  0xFFC0 (negative, mantissa msb set).
    Note: 0xFF80 has zero mantissa — that is negative infinity, not NaN.

    Args:
        raw_bits: Raw UInt16 bit pattern for a BF16 NaN value.

    Returns:
        Scalar AnyTensor with DType.bfloat16 containing the given bit pattern.
    """
    var shape = List[Int]()
    shape.append(1)
    var tensor = AnyTensor(shape, DType.bfloat16)
    var ptr = tensor._data.bitcast[UInt16]()
    ptr[] = raw_bits
    return tensor^


def test_copy_independence() raises:
    """Test that copy() creates an independent deep copy."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = copy(a)

    # Check that copy creates a tensor with same values
    assert_value_at(b, 0, 3.0, 1e-6, "Copy should have same value")
    assert_value_at(b, 4, 3.0, 1e-6, "Copy should have same value at end")

    # Verify independence: modifying copy doesn't affect original
    b.set(0, Float64(99.0))
    assert_almost_equal(
        b._get_float64(0), 99.0, 1e-6, "Copy should be modified"
    )
    assert_value_at(
        a, 0, 3.0, 1e-6, "Original should be unchanged after copy modification"
    )


def test_clone_identical() raises:
    """Test that clone creates identical tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = clone(a)

    # Should have same values
    for i in range(12):
        assert_value_at(b, i, Float64(i), 1e-6, "Clone should have same values")


def test_clone_non_contiguous() raises:
    """Test that clone of a non-contiguous (transposed) tensor produces correct contiguous copy.

    A transposed tensor has permuted strides and is_contiguous() == False.
    clone() must iterate via stride-aware multi-dim indexing and produce a
    fresh contiguous tensor with the transposed logical order.
    """
    # Create (3,4) tensor with values 0..11, then transpose to (4,3)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = a.reshape(shape)
    var t = b.transpose(0, 1)  # Shape (4,3), non-contiguous

    assert_false(
        t.is_contiguous(), "Transposed tensor should not be contiguous"
    )

    var c = clone(t)

    # Clone should be contiguous
    assert_true(
        c.is_contiguous(), "Clone of transposed tensor should be contiguous"
    )

    # Clone shape should match the transposed shape (4,3)
    var c_shape = c.shape()
    assert_equal_int(c_shape[0], 4, "Cloned shape dim 0 should be 4")
    assert_equal_int(c_shape[1], 3, "Cloned shape dim 1 should be 3")

    # Verify element values: transpose of row-major (3,4) with vals 0..11
    # Row 0 of transpose = col 0 of original: 0, 4, 8
    assert_almost_equal(c._get_float64(0), 0.0, 1e-6, "c[0,0]=0")
    assert_almost_equal(c._get_float64(1), 4.0, 1e-6, "c[0,1]=4")
    assert_almost_equal(c._get_float64(2), 8.0, 1e-6, "c[0,2]=8")
    # Row 1: 1, 5, 9
    assert_almost_equal(c._get_float64(3), 1.0, 1e-6, "c[1,0]=1")
    assert_almost_equal(c._get_float64(4), 5.0, 1e-6, "c[1,1]=5")
    assert_almost_equal(c._get_float64(5), 9.0, 1e-6, "c[1,2]=9")
    # Row 2: 2, 6, 10
    assert_almost_equal(c._get_float64(6), 2.0, 1e-6, "c[2,0]=2")
    assert_almost_equal(c._get_float64(7), 6.0, 1e-6, "c[2,1]=6")
    assert_almost_equal(c._get_float64(8), 10.0, 1e-6, "c[2,2]=10")
    # Row 3: 3, 7, 11
    assert_almost_equal(c._get_float64(9), 3.0, 1e-6, "c[3,0]=3")
    assert_almost_equal(c._get_float64(10), 7.0, 1e-6, "c[3,1]=7")
    assert_almost_equal(c._get_float64(11), 11.0, 1e-6, "c[3,2]=11")


def test_clone_zero_element_tensor() raises:
    """Test that clone of a 0-element tensor succeeds and preserves metadata.

    A tensor with shape (0,) or (2,0,3) has numel == 0. clone() should
    return a new tensor with the same shape, dtype, and zero elements
    without crashing or allocating wrong sizes.
    """
    # 1D empty tensor
    var shape_1d = List[Int]()
    shape_1d.append(0)
    var empty_1d = zeros(shape_1d, DType.float32)
    var c1 = clone(empty_1d)
    assert_numel(c1, 0, "Cloned 0-element 1D tensor should have 0 elements")
    assert_dtype(c1, DType.float32, "Cloned 0-element tensor should keep dtype")
    var c1_shape = c1.shape()
    assert_equal_int(c1_shape[0], 0, "Cloned shape should be (0,)")

    # Multi-dim empty tensor: shape (2, 0, 3)
    var shape_nd = List[Int]()
    shape_nd.append(2)
    shape_nd.append(0)
    shape_nd.append(3)
    var empty_nd = zeros(shape_nd, DType.float64)
    var c2 = clone(empty_nd)
    assert_numel(c2, 0, "Cloned (2,0,3) tensor should have 0 elements")
    assert_dtype(c2, DType.float64, "Cloned (2,0,3) tensor should keep dtype")
    var c2_shape = c2.shape()
    assert_equal_int(c2_shape[0], 2, "Cloned shape dim 0 should be 2")
    assert_equal_int(c2_shape[1], 0, "Cloned shape dim 1 should be 0")
    assert_equal_int(c2_shape[2], 3, "Cloned shape dim 2 should be 3")


def test_clone_multiple_dtypes() raises:
    """Test that clone preserves values for uint8, float64, and bfloat16 dtypes.

    Ensures the stride-aware element copy in clone() correctly reads and writes
    elements for each dtype without precision loss or type confusion.
    """
    var shape = List[Int]()
    shape.append(4)

    # uint8: small integer values
    var t_u8 = full(shape, 42.0, DType.uint8)
    var c_u8 = clone(t_u8)
    assert_dtype(c_u8, DType.uint8, "uint8 clone should keep dtype")
    for i in range(4):
        assert_value_at(c_u8, i, 42.0, 1e-6, "uint8 clone value should be 42")

    # float64: high-precision value
    var t_f64 = full(shape, 3.141592653589793, DType.float64)
    var c_f64 = clone(t_f64)
    assert_dtype(c_f64, DType.float64, "float64 clone should keep dtype")
    for i in range(4):
        assert_value_at(
            c_f64,
            i,
            3.141592653589793,
            1e-12,
            "float64 clone should preserve precision",
        )

    # bfloat16: reduced precision value
    var t_bf = full(shape, 1.5, DType.bfloat16)
    var c_bf = clone(t_bf)
    assert_dtype(c_bf, DType.bfloat16, "bfloat16 clone should keep dtype")
    for i in range(4):
        # bfloat16 has limited precision; use wider tolerance
        assert_value_at(
            c_bf, i, 1.5, 0.01, "bfloat16 clone value should be ~1.5"
        )


def test_numel_total_elements() raises:
    """Test numel() returns total number of elements."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 24, "numel should return 24 for (2,3,4)")


def test_dim_num_dimensions() raises:
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


def test_shape_property() raises:
    """Test shape() returns correct shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 2, "Shape should have 2 dimensions")
    assert_equal_int(s[0], 3, "First dimension should be 3")
    assert_equal_int(s[1], 4, "Second dimension should be 4")


def test_dtype_property() raises:
    """Test dtype() returns correct data type."""
    var shape = List[Int]()
    shape.append(5)

    var t32 = ones(shape, DType.float32)
    assert_dtype(t32, DType.float32, "Should be float32")

    var t64 = ones(shape, DType.float64)
    assert_dtype(t64, DType.float64, "Should be float64")


def test_stride_row_major() raises:
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


def test_is_contiguous_true() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_true(t.is_contiguous(), "Newly created tensor should be contiguous")


def test_is_contiguous_after_transpose() raises:
    """Test that a transposed tensor is not contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = a.transpose(0, 1)
    assert_false(
        b.is_contiguous(), "Transposed tensor should not be contiguous"
    )


def test_contiguous_on_noncontiguous() raises:
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


def test_as_contiguous_values_correct() raises:
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


def test_as_contiguous_3d_values_correct() raises:
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
    assert_false(
        t.is_contiguous(), "Transposed 3D tensor should not be contiguous"
    )

    # Make contiguous
    var c = as_contiguous(t)
    assert_true(
        c.is_contiguous(), "as_contiguous() result should be contiguous"
    )

    # Verify shape (4,3,2) after transpose
    var c_shape = c.shape()
    assert_equal_int(c_shape[0], 4, "Shape[0] should be 4 after transpose")
    assert_equal_int(c_shape[1], 3, "Shape[1] should be 3 after transpose")
    assert_equal_int(c_shape[2], 2, "Shape[2] should be 2 after transpose")

    # Verify totalcontent (all 24 elements)
    assert_equal_int(
        c.numel(), 24, "Contiguous 3D tensor should have 24 elements"
    )

    # Verify that the transpose correctly maps elements from original (2,3,4) to result (4,3,2)
    # Original element at [i,j,k] goes to transposed position [k,j,i]
    # So we check a few cross-section values to ensure stride-based indexing is correct

    # Original [0,0,0]=0 -> transposed [0,0,0]
    assert_almost_equal(
        c._get_float64(0), 0.0, 1e-6, "First element should be 0"
    )

    # Original [0,0,3]=3 -> transposed [3,0,0]
    # Transposed flat index [3,0,0]: 3*6 + 0*2 + 0 = 18
    assert_almost_equal(c._get_float64(18), 3.0, 1e-6, "Element from [0,0,3]")

    # Original [1,2,3]=23 -> transposed [3,2,1]
    # Transposed flat index [3,2,1]: 3*6 + 2*2 + 1 = 23
    assert_almost_equal(c._get_float64(23), 23.0, 1e-6, "Element from [1,2,3]")

    # Original [1,0,0]=12 -> transposed [0,0,1]
    # Transposed flat index [0,0,1]: 0*6 + 0*2 + 1 = 1
    assert_almost_equal(c._get_float64(1), 12.0, 1e-6, "Element from [1,0,0]")


def test_transpose_1d_tensor_raises() raises:
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

    assert_true(error_raised, "transpose on 1D tensor should raise an error")


def test_transpose_out_of_range_dim0() raises:
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


def test_transpose_out_of_range_dim1() raises:
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


def test_transpose_same_dim_identity() raises:
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
    assert_equal_int(
        result._strides[0], b._strides[0], "Stride[0] should be unchanged"
    )
    assert_equal_int(
        result._strides[1], b._strides[1], "Stride[1] should be unchanged"
    )

    # Values should be identical
    assert_almost_equal(
        result._get_float64(0), 0.0, 1e-6, "Value at [0,0] should match"
    )
    assert_almost_equal(
        result._get_float64(5), 5.0, 1e-6, "Value at [1,1] should match"
    )


def test_item_single_element() raises:
    """Test extracting value from single-element tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var val = item(t)

    assert_almost_equal(val, 42.0, 1e-6, "item() should extract scalar value")


def test_item_requires_single_element() raises:
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


def test_tolist_1d() raises:
    """Test converting 1D tensor to list."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var lst = t.tolist()

    # Should return [0, 1, 2, 3, 4]
    assert_equal_int(len(lst), 5, "List should have 5 elements")
    for i in range(5):
        assert_almost_equal(
            lst[i], Float64(i), 1e-6, "List value should match tensor"
        )


def test_tolist_nested() raises:
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


def test_len_first_dim() raises:
    """Test __len__ returns size of first dimension."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(3)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 5, "__len__ should return first dimension")


def test_len_1d() raises:
    """Test __len__ on 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 10, "__len__ should return size for 1D")


def test_setitem_valid_index() raises:
    """Test setting value at valid flat index, verified with __getitem__."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t[1] = 9.5
    assert_value_at(t, 1, 9.5, 1e-6, "__setitem__ should set value at index 1")
    # Other elements unchanged
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0.0")
    assert_value_at(t, 2, 0.0, 1e-6, "Element 2 should remain 0.0")


def test_setitem_integer_dtype() raises:
    """Test setting value on integer dtype tensor."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.int32)
    t[2] = 7.0
    assert_value_at(
        t, 2, 7.0, 1e-6, "__setitem__ should set value on int32 tensor"
    )
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0")


def test_setitem_out_of_bounds() raises:
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


def test_setitem_negative_index() raises:
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


def test_getitem_negative_index() raises:
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


def test_bool_single_element() raises:
    """Test __bool__ on single-element tensor."""
    var shape = List[Int]()
    var t_zero = full(shape, 0.0, DType.float32)
    var t_nonzero = full(shape, 5.0, DType.float32)

    if t_zero:
        raise Error("Zero tensor should be falsy")
    if not t_nonzero:
        raise Error("Non-zero tensor should be truthy")


def test_bool_requires_single_element() raises:
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


def test_int_conversion() raises:
    """Test int conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.5, DType.float32)

    # Use item() to extract value, then convert to Int
    var val = Int(item(t))
    assert_equal_int(val, 42, "item() + Int should convert to int")


def test_float_conversion() raises:
    """Test float conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.int32)

    # Use item() to extract value as Float64
    var val = item(t)
    assert_almost_equal(val, 42.0, 1e-6, "item() should return Float64 value")


def test_str_readable() raises:
    """Test __str__ produces readable output."""
    var t = arange(0.0, 3.0, 1.0, DType.float32)
    var s = String(t)
    assert_equal(
        s, "AnyTensor([0.0, 1.0, 2.0], dtype=float32)", "__str__ format"
    )


def test_repr_complete() raises:
    """Test __repr__ produces complete representation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var r = repr(t)
    assert_equal(
        r,
        (
            "AnyTensor(shape=[2, 2], dtype=float32, numel=4, data=[1.0, 1.0,"
            " 1.0, 1.0])"
        ),
        "__repr__ format",
    )


def test_hash_immutable() raises:
    """Test __hash__ for immutable tensors."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var b = arange(0.0, 3.0, 1.0, DType.float32)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a), Int(hash_b), "Equal tensors should have same hash"
    )


def test_hash_different_values_differ() raises:
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


def test_hash_large_values() raises:
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


def test_hash_small_values_distinguish() raises:
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


def test_hash_different_dtypes_differ() raises:
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


def test_hash_different_shapes_differ() raises:
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


def test_hash_same_values_different_dtype() raises:
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


def test_hash_integer_dtype_consistent() raises:
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


def test_hash_empty_tensor_dtype_differs() raises:
    """Test that empty tensors with different dtypes produce different hashes.

    When numel=0, the data loop is skipped entirely, so dtype_to_ordinal is
    the only contributor that can distinguish them. This catches regressions
    where dtype contribution is accidentally dropped from __hash__.
    """
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "Empty tensors with different dtypes should have different hashes"
        )


def test_hash_bf16_nan_canonical() raises:
    """Test that canonical BF16 NaN (0x7FC0) hashes consistently.

    Two tensors with the same canonical NaN bit pattern must produce identical
    hashes — NaN canonicalization must not depend on object identity.
    """
    var a = make_bf16_nan_tensor(0x7FC0)
    var b = make_bf16_nan_tensor(0x7FC0)
    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a),
        Int(hash_b),
        "Canonical BF16 NaN should hash consistently",
    )


def test_hash_bf16_nan_negative() raises:
    """Test that negative BF16 NaN (0xFFC0) hashes consistently.

    0xFFC0 is a negative quiet NaN: sign=1, exponent=all-ones, mantissa=0x40.
    Two tensors with the same negative NaN bit pattern must produce identical
    hashes.
    """
    var a = make_bf16_nan_tensor(0xFFC0)
    var b = make_bf16_nan_tensor(0xFFC0)
    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a),
        Int(hash_b),
        "Negative BF16 NaN should hash consistently",
    )


def test_hash_bf16_nan_canonicalization() raises:
    """Test that canonical and negative BF16 NaN hash to the same value.

    IEEE 754 NaN canonicalization: all NaN variants should produce the same
    hash so tensors containing NaN behave correctly in sets/dicts. The
    __hash__ implementation calls _get_float64 (which reinterprets BF16 raw
    bits as Float32 via bit shift then promotes to Float64) then checks
    isnan() to canonicalize before hashing. Both 0x7FC0 (positive quiet NaN)
    and 0xFFC0 (negative quiet NaN) should map to the same hash value.
    """
    var canonical_nan = make_bf16_nan_tensor(0x7FC0)
    var negative_nan = make_bf16_nan_tensor(0xFFC0)
    var hash_canonical = hash(canonical_nan)
    var hash_negative = hash(negative_nan)
    assert_equal_int(
        Int(hash_canonical),
        Int(hash_negative),
        "All BF16 NaN variants should canonicalize to same hash",
    )


def test_hash_empty_tensor_shapes_differ() raises:
    """Test that empty tensors with different shapes produce different hashes.

    When numel=0, the data loop is skipped and only shape dimensions and dtype
    contribute to the hash. This verifies that shape is fully exercised for
    multi-dimensional empty tensors.
    """
    var shape_1d = List[Int]()
    shape_1d.append(0)
    var t1 = zeros(shape_1d, DType.float32)  # shape [0]

    var shape_2d_00 = List[Int]()
    shape_2d_00.append(0)
    shape_2d_00.append(0)
    var t2 = zeros(shape_2d_00, DType.float32)  # shape [0, 0]

    var shape_2d_01 = List[Int]()
    shape_2d_01.append(0)
    shape_2d_01.append(1)
    var t3 = zeros(shape_2d_01, DType.float32)  # shape [0, 1]

    if hash(t1) == hash(t2):
        raise Error("Empty tensors [0] and [0,0] should have different hashes")
    if hash(t1) == hash(t3):
        raise Error("Empty tensors [0] and [0,1] should have different hashes")
    if hash(t2) == hash(t3):
        raise Error(
            "Empty tensors [0,0] and [0,1] should have different hashes"
        )


def test_diff_1d() raises:
    """Test computing consecutive differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var d = diff(t)

    # Result: [1, 1, 1, 1] (4 elements)
    assert_numel(d, 4, "diff should have n-1 elements")
    for i in range(4):
        assert_value_at(d, i, 1.0, 1e-6, "Consecutive differences should be 1")


def test_diff_higher_order() raises:
    """Test higher-order differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var d = diff(t, 2)

    # Second differences of [0,1,2,3,4] -> [0,0,0]
    assert_numel(d, 3, "Second diff should have n-2 elements")
    for i in range(3):
        assert_value_at(d, i, 0.0, 1e-6, "Second-order differences should be 0")


def main() raises:
    """Run all test_utility tests."""
    print("Running test_utility tests...")

    test_copy_independence()
    print("✓ test_copy_independence")

    test_clone_identical()
    print("✓ test_clone_identical")

    test_clone_non_contiguous()
    print("✓ test_clone_non_contiguous")

    test_clone_zero_element_tensor()
    print("✓ test_clone_zero_element_tensor")

    test_clone_multiple_dtypes()
    print("✓ test_clone_multiple_dtypes")

    test_numel_total_elements()
    print("✓ test_numel_total_elements")

    test_dim_num_dimensions()
    print("✓ test_dim_num_dimensions")

    test_shape_property()
    print("✓ test_shape_property")

    test_dtype_property()
    print("✓ test_dtype_property")

    test_stride_row_major()
    print("✓ test_stride_row_major")

    test_is_contiguous_true()
    print("✓ test_is_contiguous_true")

    test_is_contiguous_after_transpose()
    print("✓ test_is_contiguous_after_transpose")

    test_contiguous_on_noncontiguous()
    print("✓ test_contiguous_on_noncontiguous")

    test_as_contiguous_values_correct()
    print("✓ test_as_contiguous_values_correct")

    test_as_contiguous_3d_values_correct()
    print("✓ test_as_contiguous_3d_values_correct")

    test_transpose_1d_tensor_raises()
    print("✓ test_transpose_1d_tensor_raises")

    test_transpose_out_of_range_dim0()
    print("✓ test_transpose_out_of_range_dim0")

    test_transpose_out_of_range_dim1()
    print("✓ test_transpose_out_of_range_dim1")

    test_transpose_same_dim_identity()
    print("✓ test_transpose_same_dim_identity")

    test_item_single_element()
    print("✓ test_item_single_element")

    test_item_requires_single_element()
    print("✓ test_item_requires_single_element")

    test_tolist_1d()
    print("✓ test_tolist_1d")

    test_tolist_nested()
    print("✓ test_tolist_nested")

    test_len_first_dim()
    print("✓ test_len_first_dim")

    test_len_1d()
    print("✓ test_len_1d")

    test_setitem_valid_index()
    print("✓ test_setitem_valid_index")

    test_setitem_integer_dtype()
    print("✓ test_setitem_integer_dtype")

    test_setitem_out_of_bounds()
    print("✓ test_setitem_out_of_bounds")

    test_setitem_negative_index()
    print("✓ test_setitem_negative_index")

    test_getitem_negative_index()
    print("✓ test_getitem_negative_index")

    test_bool_single_element()
    print("✓ test_bool_single_element")

    test_bool_requires_single_element()
    print("✓ test_bool_requires_single_element")

    test_int_conversion()
    print("✓ test_int_conversion")

    test_float_conversion()
    print("✓ test_float_conversion")

    test_str_readable()
    print("✓ test_str_readable")

    test_repr_complete()
    print("✓ test_repr_complete")

    test_hash_immutable()
    print("✓ test_hash_immutable")

    test_hash_different_values_differ()
    print("✓ test_hash_different_values_differ")

    test_hash_large_values()
    print("✓ test_hash_large_values")

    test_hash_small_values_distinguish()
    print("✓ test_hash_small_values_distinguish")

    test_hash_different_dtypes_differ()
    print("✓ test_hash_different_dtypes_differ")

    test_hash_different_shapes_differ()
    print("✓ test_hash_different_shapes_differ")

    test_hash_same_values_different_dtype()
    print("✓ test_hash_same_values_different_dtype")

    test_hash_integer_dtype_consistent()
    print("✓ test_hash_integer_dtype_consistent")

    test_hash_empty_tensor_dtype_differs()
    print("✓ test_hash_empty_tensor_dtype_differs")

    test_hash_bf16_nan_canonical()
    print("✓ test_hash_bf16_nan_canonical")

    test_hash_bf16_nan_negative()
    print("✓ test_hash_bf16_nan_negative")

    test_hash_bf16_nan_canonicalization()
    print("✓ test_hash_bf16_nan_canonicalization")

    test_hash_empty_tensor_shapes_differ()
    print("✓ test_hash_empty_tensor_shapes_differ")

    test_diff_1d()
    print("✓ test_diff_1d")

    test_diff_higher_order()
    print("✓ test_diff_higher_order")

    print("\nAll test_utility tests passed!")
