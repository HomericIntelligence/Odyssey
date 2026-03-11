# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor creation operations - Part 5: dtype support and edge cases.

Tests dtype support across creation operations and edge cases.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and creation operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test dtype support
# ============================================================================


fn test_creation_float16() raises:
    """Test creation operations with float16 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float16)
    assert_dtype(t, DType.float16, "zeros should support float16")


fn test_creation_float32() raises:
    """Test creation operations with float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)
    assert_dtype(t, DType.float32, "ones should support float32")


fn test_creation_float64() raises:
    """Test creation operations with float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = full(shape, 3.14, DType.float64)
    assert_dtype(t, DType.float64, "full should support float64")


fn test_creation_int8() raises:
    """Test creation operations with int8 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.int8)
    assert_dtype(t, DType.int8, "zeros should support int8")


fn test_creation_int32() raises:
    """Test creation operations with int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.int32)
    assert_dtype(t, DType.int32, "ones should support int32")


fn test_creation_uint8() raises:
    """Test creation operations with uint8 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = full(shape, 255.0, DType.uint8)
    assert_dtype(t, DType.uint8, "full should support uint8")


fn test_creation_bool() raises:
    """Test creation operations with bool dtype."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.bool)
    assert_dtype(t, DType.bool, "zeros should support bool")


# ============================================================================
# Test edge cases
# ============================================================================


fn test_creation_0d_scalar() raises:
    """Test creating 0D scalar tensor."""
    var shape = List[Int]()
    var t = zeros(shape, DType.float32)

    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_value_at(t, 0, 0.0, 1e-8, "0D tensor value")


fn test_creation_very_large_1d() raises:
    """Test creating very large 1D tensor."""
    var shape = List[Int]()
    shape.append(1000000)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 1000000, "Large 1D tensor should have 1000000 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "Large tensor first element")
    assert_value_at(t, 999999, 0.0, 1e-8, "Large tensor last element")


fn test_creation_high_dimensional() raises:
    """Test creating tensor with many dimensions (e.g., 8D)."""
    var shape = List[Int](length=8, fill=2)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 8, "8D tensor should have 8 dimensions")
    assert_numel(t, 256, "8D tensor (2x2x2x2x2x2x2x2) should have 256 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run dtype support and edge case creation tests."""
    print(
        "Running ExTensor creation tests - Part 5: dtype support and edge"
        " cases..."
    )

    # dtype tests
    test_creation_float16()
    test_creation_float32()
    test_creation_float64()
    test_creation_int8()
    test_creation_int32()
    test_creation_uint8()
    test_creation_bool()

    # Edge case tests
    test_creation_0d_scalar()
    test_creation_very_large_1d()
    test_creation_high_dimensional()

    print("All Part 5 creation tests completed!")
