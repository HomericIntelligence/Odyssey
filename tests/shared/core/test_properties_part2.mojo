"""Tests for AnyTensor dtype bool and numel properties (Part 2 of 5).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_properties.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full, arange, eye

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
# Test dtype property (continued)
# ============================================================================


fn test_dtype_bool() raises:
    """Test dtype property for bool tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.bool)

    assert_dtype(t, DType.bool, "Should be bool")


# ============================================================================
# Test numel property
# ============================================================================


fn test_numel_1d() raises:
    """Test numel for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    assert_numel(t, 10, "1D tensor with 10 elements")


fn test_numel_2d() raises:
    """Test numel for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 12, "2D tensor with 12 elements (3*4)")


fn test_numel_3d() raises:
    """Test numel for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 24, "3D tensor with 24 elements (2*3*4)")


fn test_numel_scalar() raises:
    """Test numel for scalar tensor."""
    var shape = List[Int]()
    var t = full(shape, 1.0, DType.float32)

    assert_numel(t, 1, "Scalar tensor has 1 element")


fn test_numel_empty() raises:
    """Test numel for empty tensor."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor has 0 elements")


# ============================================================================
# Test stride calculations
# ============================================================================


fn test_strides_1d() raises:
    """Test stride calculation for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 1, "1D tensor should have 1 stride")
    assert_equal_int(strides[0], 1, "1D stride should be 1")


fn test_strides_2d_row_major() raises:
    """Test stride calculation for 2D tensor (row-major)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 2, "2D tensor should have 2 strides")
    assert_equal_int(strides[0], 4, "Outer stride should be 4 (row length)")
    assert_equal_int(strides[1], 1, "Inner stride should be 1")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run dtype bool, numel, and stride property tests (Part 2)."""
    print("Running AnyTensor dtype bool, numel, and stride tests (Part 2)...")

    # DType bool test
    print("  Testing dtype bool...")
    test_dtype_bool()

    # Numel property tests
    print("  Testing numel property...")
    test_numel_1d()
    test_numel_2d()
    test_numel_3d()
    test_numel_scalar()
    test_numel_empty()

    # Stride tests
    print("  Testing stride calculations...")
    test_strides_1d()
    test_strides_2d_row_major()

    print("All Part 2 tests completed!")
