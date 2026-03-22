"""Tests for AnyTensor comparison operations - Part 1: equal and not_equal.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_comparison_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests comparison operations following the Array API Standard:
equal, not_equal.
All operations return boolean tensors (DType.bool).
"""

# Import AnyTensor and comparison operations
from shared.core.extensor import AnyTensor, full, ones, zeros
from shared.core.comparison import equal, not_equal

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test equal()
# ============================================================================


fn test_equal_same_values() raises:
    """Test equal with identical values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    assert_numel(c, 5, "Result should have 5 elements")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "Equal values should return True")


fn test_equal_different_values() raises:
    """Test equal with different values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "Different values should return False")


fn test_equal_with_dunder() raises:
    """Test equal using == operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a == b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a == b should work via __eq__")


# ============================================================================
# Test not_equal()
# ============================================================================


fn test_not_equal_same_values() raises:
    """Test not_equal with identical values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = not_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(
            c, i, 0.0, 1e-6, "Equal values should return False for !="
        )


fn test_not_equal_different_values() raises:
    """Test not_equal with different values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = not_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(
            c, i, 1.0, 1e-6, "Different values should return True for !="
        )


fn test_not_equal_with_dunder() raises:
    """Test not_equal using != operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a != b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a != b should work via __ne__")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run equal and not_equal comparison operation tests."""
    print("Running AnyTensor equal/not_equal comparison operation tests...")

    # equal() tests
    print("  Testing equal()...")
    test_equal_same_values()
    test_equal_different_values()
    test_equal_with_dunder()

    # not_equal() tests
    print("  Testing not_equal()...")
    test_not_equal_same_values()
    test_not_equal_different_values()
    test_not_equal_with_dunder()

    print("All equal/not_equal comparison operation tests completed!")
