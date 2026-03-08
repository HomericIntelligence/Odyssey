"""Tests for ExTensor utility operations - Part 1: copy/clone and property accessors.

Tests utility functions including copy, clone, numel, dim, shape, dtype,
and stride calculations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utility.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
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
# Main test runner
# ============================================================================


fn main() raises:
    """Run utility operation tests - Part 1: copy/clone and property accessors."""
    print("Running ExTensor utility operation tests (Part 1)...")

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

    print("All utility operation tests (Part 1) completed!")
