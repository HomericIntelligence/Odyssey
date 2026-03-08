# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for squeeze and unsqueeze edge cases.

Tests edge cases for squeeze and unsqueeze operations including:
- Squeeze with size-1 dimensions
- Squeeze with no size-1 dimensions
- Unsqueeze at various positions
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange
from shared.core.shape import reshape, squeeze, unsqueeze, concatenate, stack

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_shape,
)


# ============================================================================
# Test squeeze edge cases
# ============================================================================


fn test_squeeze_size_one_dim() raises:
    """Squeeze [1, 3, 1, 4] -> [3, 4]."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(1)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Should have 3*4=12 elements")


fn test_squeeze_all_size_one() raises:
    """Squeeze [1, 1, 1] -> []."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)
    shape.append(1)
    var t = full(shape, 42.0, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_numel(result, 1, "Should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_squeeze_no_size_one() raises:
    """Squeeze [2, 3, 4] with no size-1 dims."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    # Should remain unchanged if no size-1 dimensions
    assert_dim(result, 3, "Result should remain 3D")
    assert_numel(result, 24, "Should preserve elements")


fn test_squeeze_1d_with_size_one() raises:
    """Squeeze [1] -> []."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 99.0, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_value_at(result, 0, 99.0, 1e-6, "Value preserved")


# ============================================================================
# Test unsqueeze edge cases
# ============================================================================


fn test_unsqueeze_scalar() raises:
    """Unsqueeze [] at dim 0 -> [1]."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    var result = unsqueeze(t, 0)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 1, "Should have 1 element")


fn test_unsqueeze_1d() raises:
    """Unsqueeze [3] at dim 0 -> [1, 3]."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, 2.0, DType.float32)

    var result = unsqueeze(t, 0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 3, "Should have 3 elements")


fn test_unsqueeze_1d_at_end() raises:
    """Unsqueeze [3] at dim 1 -> [3, 1]."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 1)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 3, "Should have 3 elements")


fn test_unsqueeze_2d() raises:
    """Unsqueeze [2, 3] at dim 1 -> [2, 1, 3]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 1)

    assert_dim(result, 3, "Result should be 3D")
    assert_numel(result, 6, "Should have 6 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run squeeze and unsqueeze edge case tests."""
    print("Running squeeze and unsqueeze edge case tests...")

    # Squeeze edge cases
    print("  Testing squeeze edge cases...")
    test_squeeze_size_one_dim()
    test_squeeze_all_size_one()
    test_squeeze_no_size_one()
    test_squeeze_1d_with_size_one()

    # Unsqueeze edge cases
    print("  Testing unsqueeze edge cases...")
    test_unsqueeze_scalar()
    test_unsqueeze_1d()
    test_unsqueeze_1d_at_end()
    test_unsqueeze_2d()

    print("All squeeze and unsqueeze edge case tests completed!")
