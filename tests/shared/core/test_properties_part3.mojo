"""Tests for AnyTensor stride (3D), contiguity, and dimension properties (Part 3 of 5).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_properties.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange, eye

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_equal_int,
    assert_value_at,
)


# ============================================================================
# Test stride calculations (continued)
# ============================================================================


fn test_strides_3d_row_major() raises:
    """Test stride calculation for 3D tensor (row-major)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 3, "3D tensor should have 3 strides")
    assert_equal_int(strides[0], 12, "Stride 0 should be 12 (3*4)")
    assert_equal_int(strides[1], 4, "Stride 1 should be 4")
    assert_equal_int(strides[2], 1, "Stride 2 should be 1")


# ============================================================================
# Test contiguity
# ============================================================================


fn test_contiguous_new_tensor() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_true(t.is_contiguous(), "Newly created tensor should be contiguous")


fn test_contiguous_1d() raises:
    """Test that 1D tensors are contiguous."""
    var shape = List[Int]()
    shape.append(100)
    var t = arange(0.0, 100.0, 1.0, DType.float32)

    assert_true(t.is_contiguous(), "1D tensor should be contiguous")


fn test_contiguous_scalar() raises:
    """Test that scalar tensors are contiguous."""
    var shape = List[Int]()
    var t = full(shape, 5.0, DType.float32)

    assert_true(t.is_contiguous(), "Scalar tensor should be contiguous")


# ============================================================================
# Test dimension queries
# ============================================================================


fn test_dim_1d() raises:
    """Test dim for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    assert_dim(t, 1, "1D tensor should have dim=1")


fn test_dim_2d() raises:
    """Test dim for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_dim(t, 2, "2D tensor should have dim=2")


fn test_dim_3d() raises:
    """Test dim for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_dim(t, 3, "3D tensor should have dim=3")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run stride 3D, contiguity, and dimension property tests (Part 3)."""
    print(
        "Running AnyTensor stride, contiguity, and dimension tests (Part 3)..."
    )

    # Stride 3D test
    print("  Testing 3D stride calculation...")
    test_strides_3d_row_major()

    # Contiguity tests
    print("  Testing contiguity...")
    test_contiguous_new_tensor()
    test_contiguous_1d()
    test_contiguous_scalar()

    # Dimension tests
    print("  Testing dimension queries...")
    test_dim_1d()
    test_dim_2d()
    test_dim_3d()

    print("All Part 3 tests completed!")
