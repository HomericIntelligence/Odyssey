# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_fixtures.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for shared.testing.fixtures module - Part 3: Tensor value assertions.

Tests assert_tensor_dtype_invalid, assert_tensor_all_finite, and
assert_tensor_not_all_zeros helpers.
"""

from testing import assert_true, assert_equal
from shared.testing.fixtures import (
    assert_tensor_dtype,
    assert_tensor_all_finite,
    assert_tensor_not_all_zeros,
)
from shared.core.extensor import ones, zeros


fn test_assert_tensor_dtype_invalid() raises:
    """Test assert_tensor_dtype with mismatched dtype."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(not assert_tensor_dtype(tensor, DType.float64))


fn test_assert_tensor_all_finite_valid() raises:
    """Test assert_tensor_all_finite with finite values."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_all_finite(tensor))


fn test_assert_tensor_not_all_zeros_valid() raises:
    """Test assert_tensor_not_all_zeros with non-zero values."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_not_all_zeros(tensor))


fn test_assert_tensor_not_all_zeros_invalid() raises:
    """Test assert_tensor_not_all_zeros with all zeros."""
    var tensor = zeros([32, 10], DType.float32)
    assert_true(not assert_tensor_not_all_zeros(tensor))


fn main() raises:
    """Run all fixture tests - Part 3."""
    test_assert_tensor_dtype_invalid()
    test_assert_tensor_all_finite_valid()
    test_assert_tensor_not_all_zeros_valid()
    test_assert_tensor_not_all_zeros_invalid()

    print("All tests passed!")
