"""Tests for AnyTensor utility operations - Part 1: copy/clone and property accessors.

Tests utility functions including copy, clone, numel, dim, shape, dtype,
and stride calculations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utility.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange, clone, item, diff
from shared.core.shape import as_contiguous
from shared.core.matrix import transpose_view

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


fn test_clone_non_contiguous() raises:
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
    assert_true(c.is_contiguous(), "Clone of transposed tensor should be contiguous")

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


fn test_clone_zero_element_tensor() raises:
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


fn test_clone_multiple_dtypes() raises:
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
            c_f64, i, 3.141592653589793, 1e-12, "float64 clone should preserve precision"
        )

    # bfloat16: reduced precision value
    var t_bf = full(shape, 1.5, DType.bfloat16)
    var c_bf = clone(t_bf)
    assert_dtype(c_bf, DType.bfloat16, "bfloat16 clone should keep dtype")
    for i in range(4):
        # bfloat16 has limited precision; use wider tolerance
        assert_value_at(c_bf, i, 1.5, 0.01, "bfloat16 clone value should be ~1.5")


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
# Main test runner
# ============================================================================


fn main() raises:
    """Run utility operation tests - Part 1: copy/clone and property accessors.
    """
    print("Running AnyTensor utility operation tests (Part 1)...")

    # copy() and clone() tests
    print("  Testing copy() and clone()...")
    test_copy_independence()
    test_clone_identical()
    test_clone_non_contiguous()
    test_clone_zero_element_tensor()
    test_clone_multiple_dtypes()

    # Property accessors
    print("  Testing property accessors...")
    test_numel_total_elements()
    test_dim_num_dimensions()
    test_shape_property()
    test_dtype_property()

    # Stride calculations
    print("  Testing stride calculations...")
    test_stride_row_major()

    print("All utility operation tests (Part 1) completed!")
