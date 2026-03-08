# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_extensor_unary_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor __neg__ and __pos__ unary operators."""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_neg_basic() raises:
    """Test __neg__: negation -tensor."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -3.0, tolerance=1e-6
        )


fn test_neg_negative_values() raises:
    """Test __neg__: negation of negative values."""
    var a = full([2, 3], -5.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 5.0, tolerance=1e-6
        )


fn test_neg_zeros() raises:
    """Test __neg__: negation of zeros."""
    var a = zeros([2, 3], DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 0.0, tolerance=1e-6
        )


fn test_pos_basic() raises:
    """Test __pos__: positive +tensor (returns copy)."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.0, tolerance=1e-6
        )
    assert_equal(result.numel(), a.numel())


fn test_pos_preserves_values() raises:
    """Test __pos__: positive preserves values including negative."""
    var a = full([3, 2], -2.5, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -2.5, tolerance=1e-6
        )


fn main() raises:
    """Run all __neg__ and __pos__ operator tests."""
    test_neg_basic()
    test_neg_negative_values()
    test_neg_zeros()
    test_pos_basic()
    test_pos_preserves_values()
    print("All neg/pos operator tests passed!")
