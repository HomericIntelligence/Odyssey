"""Tests for ExTensor scalar dim, value access, and creation patterns (Part 4 of 5).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_properties.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, arange, eye

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
# Test dimension queries (continued)
# ============================================================================


fn test_dim_scalar() raises:
    """Test dim for scalar (0D) tensor."""
    var shape = List[Int]()
    var t = full(shape, 1.0, DType.float32)

    assert_dim(t, 0, "Scalar tensor should have dim=0")


# ============================================================================
# Test value access patterns
# ============================================================================


fn test_value_access_1d() raises:
    """Test accessing values in 1D tensor."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    assert_value_at(t, 0, 0.0, 1e-6, "First element")
    assert_value_at(t, 2, 2.0, 1e-6, "Middle element")
    assert_value_at(t, 4, 4.0, 1e-6, "Last element")


fn test_value_access_2d_row_major() raises:
    """Test accessing values in 2D tensor (row-major order)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    # Should be: [[0, 1, 2], [3, 4, 5]]

    assert_value_at(t, 0, 0.0, 1e-6, "Element [0,0]")
    assert_value_at(t, 2, 2.0, 1e-6, "Element [0,2]")
    assert_value_at(t, 3, 3.0, 1e-6, "Element [1,0]")
    assert_value_at(t, 5, 5.0, 1e-6, "Element [1,2]")


fn test_value_access_identity() raises:
    """Test accessing values in identity matrix."""
    var t = eye(3, 3, 0, DType.float32)

    # Diagonal elements should be 1.0
    assert_value_at(t, 0, 1.0, 1e-6, "Diagonal [0,0]")
    assert_value_at(t, 4, 1.0, 1e-6, "Diagonal [1,1]")
    assert_value_at(t, 8, 1.0, 1e-6, "Diagonal [2,2]")

    # Off-diagonal should be 0.0
    assert_value_at(t, 1, 0.0, 1e-6, "Off-diagonal [0,1]")
    assert_value_at(t, 3, 0.0, 1e-6, "Off-diagonal [1,0]")


# ============================================================================
# Test special tensor creation patterns
# ============================================================================


fn test_all_zeros_pattern() raises:
    """Test that zeros creates all zero values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = zeros(shape, DType.float32)

    for i in range(9):
        assert_value_at(t, i, 0.0, 1e-8, "All elements should be 0")


fn test_all_ones_pattern() raises:
    """Test that ones creates all one values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = ones(shape, DType.float32)

    for i in range(9):
        assert_value_at(t, i, 1.0, 1e-8, "All elements should be 1")


fn test_full_pattern() raises:
    """Test that full creates uniform values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    var t = full(shape, 7.5, DType.float32)

    for i in range(8):
        assert_value_at(t, i, 7.5, 1e-6, "All elements should be 7.5")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run scalar dim, value access, and creation pattern tests (Part 4)."""
    print("Running ExTensor scalar dim, value access, and creation pattern tests (Part 4)...")

    # Scalar dim test
    print("  Testing scalar dimension query...")
    test_dim_scalar()

    # Value access tests
    print("  Testing value access patterns...")
    test_value_access_1d()
    test_value_access_2d_row_major()
    test_value_access_identity()

    # Pattern tests
    print("  Testing creation patterns...")
    test_all_zeros_pattern()
    test_all_ones_pattern()
    test_full_pattern()

    print("All Part 4 tests completed!")
