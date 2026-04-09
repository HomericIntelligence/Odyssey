"""Tests for tensor dtype/numel/dim/value assertion functions.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_assertions.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)
from shared.tensor.any_tensor import ones


def test_assert_dtype_tensor_passes() raises:
    """Test assert_dtype with matching dtype."""
    var tensor = ones([3, 4], DType.float32)
    assert_dtype(tensor, DType.float32)


def test_assert_dtype_tensor_fails() raises:
    """Test assert_dtype with mismatched dtype."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dtype(tensor, DType.float64)
    except:
        failed = True
    assert_true(failed, "assert_dtype should raise error on dtype mismatch")


def test_assert_numel_tensor_passes() raises:
    """Test assert_numel with matching element count."""
    var tensor = ones([3, 4], DType.float32)
    assert_numel(tensor, 12)


def test_assert_numel_tensor_fails() raises:
    """Test assert_numel with mismatched element count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_numel(tensor, 10)
    except:
        failed = True
    assert_true(failed, "assert_numel should raise error on numel mismatch")


def test_assert_dim_tensor_passes() raises:
    """Test assert_dim with matching dimension count."""
    var tensor = ones([3, 4, 5], DType.float32)
    assert_dim(tensor, 3)


def test_assert_dim_tensor_fails() raises:
    """Test assert_dim with mismatched dimension count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dim(tensor, 3)
    except:
        failed = True
    assert_true(failed, "assert_dim should raise error on dimension mismatch")


def test_assert_value_at_passes() raises:
    """Test assert_value_at with matching value."""
    var tensor = ones([3, 4], DType.float32)
    assert_value_at(tensor, 0, 1.0, tolerance=1e-6)


def test_assert_value_at_fails() raises:
    """Test assert_value_at with non-matching value."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_value_at(tensor, 0, 2.0, tolerance=1e-6)
    except:
        failed = True
    assert_true(failed, "assert_value_at should raise error on value mismatch")


def main() raises:
    """Run tensor property assertion tests."""
    test_assert_dtype_tensor_passes()
    test_assert_dtype_tensor_fails()
    test_assert_numel_tensor_passes()
    test_assert_numel_tensor_fails()
    test_assert_dim_tensor_passes()
    test_assert_dim_tensor_fails()
    test_assert_value_at_passes()
    test_assert_value_at_fails()
    print("All tensor property assertion tests passed!")
