"""Tests for AnyTensor shape and dtype properties (Part 1 of 5).

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
# Test shape property
# ============================================================================


fn test_shape_1d() raises:
    """Test shape property for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 1, "1D tensor should have 1 dimension in shape")
    assert_equal_int(s[0], 10, "First dimension should be 10")


fn test_shape_2d() raises:
    """Test shape property for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 2, "2D tensor should have 2 dimensions")
    assert_equal_int(s[0], 3, "First dimension should be 3")
    assert_equal_int(s[1], 4, "Second dimension should be 4")


fn test_shape_3d() raises:
    """Test shape property for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 3, "3D tensor should have 3 dimensions")
    assert_equal_int(s[0], 2, "Dim 0 should be 2")
    assert_equal_int(s[1], 3, "Dim 1 should be 3")
    assert_equal_int(s[2], 4, "Dim 2 should be 4")


fn test_shape_scalar() raises:
    """Test shape property for scalar (0D) tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 0, "Scalar tensor should have 0 dimensions")


# ============================================================================
# Test dtype property
# ============================================================================


fn test_dtype_float32() raises:
    """Test dtype property for float32 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    assert_dtype(t, DType.float32, "Should be float32")


fn test_dtype_float64() raises:
    """Test dtype property for float64 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float64)

    assert_dtype(t, DType.float64, "Should be float64")


fn test_dtype_int32() raises:
    """Test dtype property for int32 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.int32)

    assert_dtype(t, DType.int32, "Should be int32")


fn test_dtype_int64() raises:
    """Test dtype property for int64 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.int64)

    assert_dtype(t, DType.int64, "Should be int64")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run shape and dtype property tests (Part 1)."""
    print("Running AnyTensor shape and dtype property tests (Part 1)...")

    # Shape property tests
    print("  Testing shape property...")
    test_shape_1d()
    test_shape_2d()
    test_shape_3d()
    test_shape_scalar()

    # DType property tests
    print("  Testing dtype property...")
    test_dtype_float32()
    test_dtype_float64()
    test_dtype_int32()
    test_dtype_int64()

    print("All Part 1 tests completed!")
