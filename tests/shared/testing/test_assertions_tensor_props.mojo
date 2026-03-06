"""Tests for tensor dtype/numel/dim/value assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)
from shared.core import ones


fn test_assert_dtype_tensor_passes() raises:
    """Test assert_dtype with matching dtype."""
    var tensor = ones([3, 4], DType.float32)
    assert_dtype(tensor, DType.float32)


fn test_assert_dtype_tensor_fails() raises:
    """Test assert_dtype with mismatched dtype."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dtype(tensor, DType.float64)
    except:
        failed = True
    assert_true(failed, "assert_dtype should raise error on dtype mismatch")


fn test_assert_numel_tensor_passes() raises:
    """Test assert_numel with matching element count."""
    var tensor = ones([3, 4], DType.float32)
    assert_numel(tensor, 12)


fn test_assert_numel_tensor_fails() raises:
    """Test assert_numel with mismatched element count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_numel(tensor, 10)
    except:
        failed = True
    assert_true(failed, "assert_numel should raise error on numel mismatch")


fn test_assert_dim_tensor_passes() raises:
    """Test assert_dim with matching dimension count."""
    var tensor = ones([3, 4, 5], DType.float32)
    assert_dim(tensor, 3)


fn test_assert_dim_tensor_fails() raises:
    """Test assert_dim with mismatched dimension count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dim(tensor, 3)
    except:
        failed = True
    assert_true(failed, "assert_dim should raise error on dimension mismatch")


fn test_assert_value_at_passes() raises:
    """Test assert_value_at with matching value."""
    var tensor = ones([3, 4], DType.float32)
    assert_value_at(tensor, 0, 1.0, tolerance=1e-6)


fn test_assert_value_at_fails() raises:
    """Test assert_value_at with non-matching value."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_value_at(tensor, 0, 2.0, tolerance=1e-6)
    except:
        failed = True
    assert_true(failed, "assert_value_at should raise error on value mismatch")


fn main() raises:
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
