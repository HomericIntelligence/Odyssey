# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor creation operations - Part 2: full(), empty(), and from_array() placeholders.

Tests full() and empty() creation functions, plus from_array() placeholders.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and creation operations
from shared.core.extensor import ExTensor, full, empty

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
# Test full()
# ============================================================================


fn test_full_positive_value() raises:
    """Test creating tensor filled with positive value."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = full(shape, 5.5, DType.float32)

    assert_numel(t, 12, "full should have 12 elements")
    assert_dtype(t, DType.float32, "full should have float32 dtype")
    assert_all_values(t, 5.5, 1e-6, "full should contain all 5.5 values")


fn test_full_negative_value() raises:
    """Test creating tensor filled with negative value."""
    var shape = List[Int]()
    shape.append(10)
    var t = full(shape, -3.14, DType.float64)

    assert_numel(t, 10, "full should have 10 elements")
    assert_dtype(t, DType.float64, "full should have float64 dtype")
    assert_all_values(t, -3.14, 1e-8, "full should contain all -3.14 values")


fn test_full_zero_value() raises:
    """Test creating tensor filled with zero (should match zeros)."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var t = full(shape, 0.0, DType.float32)

    assert_numel(t, 25, "full with 0.0 should have 25 elements")
    assert_all_values(t, 0.0, 1e-8, "full with 0.0 should match zeros")


fn test_full_large_value() raises:
    """Test creating tensor filled with large value."""
    var shape = List[Int]()
    shape.append(100)
    var t = full(shape, 999999.0, DType.float32)

    assert_numel(t, 100, "full should have 100 elements")
    assert_all_values(t, 999999.0, 1e-2, "full should contain large value")


# ============================================================================
# Test empty()
# ============================================================================


fn test_empty_allocates_memory() raises:
    """Test that empty() allocates memory without initialization."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(10)
    var t = empty(shape, DType.float32)

    # Verify tensor is created with correct shape and dtype
    # Don't check values (they are undefined/uninitialized)
    assert_numel(t, 50, "empty should allocate correct size")
    assert_dtype(t, DType.float32, "empty should have correct dtype")
    assert_dim(t, 2, "empty should have correct dimensions")


fn test_empty_1d() raises:
    """Test creating empty 1D tensor."""
    var shape = List[Int]()
    shape.append(100)
    var t = empty(shape, DType.float64)

    assert_numel(t, 100, "empty 1D should have 100 elements")
    assert_dim(t, 1, "empty 1D should have 1 dimension")
    assert_dtype(t, DType.float64, "empty should have float64 dtype")


fn test_empty_2d() raises:
    """Test creating empty 2D tensor."""
    var shape = List[Int]()
    shape.append(8)
    shape.append(8)
    var t = empty(shape, DType.int32)

    assert_numel(t, 64, "empty 2D should have 64 elements")
    assert_dim(t, 2, "empty 2D should have 2 dimensions")
    assert_dtype(t, DType.int32, "empty should have int32 dtype")


# ============================================================================
# Test from_array() (placeholders - not yet implemented, see #3013)
# ============================================================================


fn test_from_array_1d() raises:
    """Test creating tensor from 1D array.

    NOTE(#3013): from_array() is not yet implemented. This test is a
    placeholder for array-to-tensor conversion. Current workaround
    is to use arange(), zeros(), or manual element initialization.
    """
    pass


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run full(), empty(), and from_array() placeholder creation tests."""
    print(
        "Running ExTensor creation tests - Part 2: full(), empty(),"
        " from_array()..."
    )

    # full() tests
    test_full_positive_value()
    test_full_negative_value()
    test_full_zero_value()
    test_full_large_value()

    # empty() tests
    test_empty_allocates_memory()
    test_empty_1d()
    test_empty_2d()

    # from_array() placeholder tests
    test_from_array_1d()

    print("All Part 2 creation tests completed!")
