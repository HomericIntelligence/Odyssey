# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor creation operations - Part 5: dtype support.

Tests dtype support across creation operations.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
Edge cases moved to test_creation_edge_cases.mojo."""

# Import ExTensor and creation operations
from shared.core.extensor import ExTensor, zeros, ones, full

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
# Main test runner
# ============================================================================


fn main() raises:
    """Run dtype support creation tests."""
    print("Running ExTensor creation tests - Part 5: dtype support...")

    test_creation_float16()
    test_creation_float32()
    test_creation_float64()
    test_creation_int8()
    test_creation_int32()
    test_creation_uint8()
    test_creation_bool()

    print("All Part 5 creation tests completed!")
