"""Tests for AnyTensor comparison operations - Part 2: less and less_equal.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_comparison_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests comparison operations following the Array API Standard:
less, less_equal.
All operations return boolean tensors (DType.bool).
"""

# Import AnyTensor and comparison operations
from shared.core.any_tensor import AnyTensor, full, ones, zeros
from shared.core.comparison import less, less_equal

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test less()
# ============================================================================


fn test_less_true() raises:
    """Test less when first tensor has smaller values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "2.0 < 3.0 should be True")


fn test_less_false() raises:
    """Test less when first tensor has larger values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "5.0 < 3.0 should be False")


fn test_less_with_dunder() raises:
    """Test less using < operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a < b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a < b should work via __lt__")


# ============================================================================
# Test less_equal()
# ============================================================================


fn test_less_equal_true_less() raises:
    """Test less_equal when values are less."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "2.0 <= 3.0 should be True")


fn test_less_equal_true_equal() raises:
    """Test less_equal when values are equal."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3.0 <= 3.0 should be True")


fn test_less_equal_with_dunder() raises:
    """Test less_equal using <= operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a <= b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a <= b should work via __le__")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run less and less_equal comparison operation tests."""
    print("Running AnyTensor less/less_equal comparison operation tests...")

    # less() tests
    print("  Testing less()...")
    test_less_true()
    test_less_false()
    test_less_with_dunder()

    # less_equal() tests
    print("  Testing less_equal()...")
    test_less_equal_true_less()
    test_less_equal_true_equal()
    test_less_equal_with_dunder()

    print("All less/less_equal comparison operation tests completed!")
