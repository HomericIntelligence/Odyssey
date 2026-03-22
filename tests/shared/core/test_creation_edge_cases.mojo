# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation_part5.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor creation operations - Edge cases.

Tests edge cases like scalar creation, very large tensors, and high-dimensional tensors.
Split from test_creation_part5.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import AnyTensor and creation operations
from shared.core.extensor import AnyTensor, zeros

# Import test helpers
from tests.shared.conftest import (
    assert_dim,
    assert_numel,
    assert_value_at,
)


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
    """Run creation edge case tests."""
    print("Running AnyTensor creation tests - Edge cases...")

    test_creation_0d_scalar()
    test_creation_very_large_1d()
    test_creation_high_dimensional()

    print("All edge case creation tests completed!")
