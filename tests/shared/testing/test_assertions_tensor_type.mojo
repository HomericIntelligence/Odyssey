"""Tests for type assertion functions.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_assertions.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Note: Split from test_assertions_tensor_values.mojo to satisfy ADR-009 ≤8
fn test_ target per file.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_type,
    assert_not_equal_tensor,
)
from shared.core import ones, full


fn test_assert_type_int() raises:
    """Test assert_type with int value."""
    var value: Int = 42
    assert_type[Int](value, "Int")


fn test_assert_type_float() raises:
    """Test assert_type with float value."""
    var value: Float32 = 3.14
    assert_type[Float32](value, "Float32")


fn test_assert_not_equal_tensor_fails() raises:
    """Test assert_not_equal_tensor with equal tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_not_equal_tensor(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_not_equal_tensor should raise error on equal tensors"
    )


fn main() raises:
    """Run type assertion tests."""
    test_assert_type_int()
    test_assert_type_float()
    test_assert_not_equal_tensor_fails()
    print("All type assertion tests passed!")
