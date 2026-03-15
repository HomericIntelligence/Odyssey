"""Tests for ExTensor comparison operations - Part 3: greater, greater_equal, negatives.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_comparison_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests comparison operations following the Array API Standard:
greater, greater_equal, and edge cases with negative values.
All operations return boolean tensors (DType.bool).
"""

# Import ExTensor and comparison operations
from shared.core.extensor import ExTensor, full, ones, zeros
from shared.core.comparison import greater, greater_equal, less

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test greater()
# ============================================================================


fn test_greater_true() raises:
    """Test greater when first tensor has larger values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "5.0 > 3.0 should be True")


fn test_greater_false() raises:
    """Test greater when first tensor has smaller values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "2.0 > 3.0 should be False")


fn test_greater_with_dunder() raises:
    """Test greater using > operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a > b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a > b should work via __gt__")


# ============================================================================
# Test greater_equal()
# ============================================================================


fn test_greater_equal_true_greater() raises:
    """Test greater_equal when values are greater."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "5.0 >= 3.0 should be True")


fn test_greater_equal_true_equal() raises:
    """Test greater_equal when values are equal."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3.0 >= 3.0 should be True")


fn test_greater_equal_with_dunder() raises:
    """Test greater_equal using >= operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a >= b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a >= b should work via __ge__")


# ============================================================================
# Test with negative values
# ============================================================================


fn test_comparison_with_negatives() raises:
    """Test comparisons with negative values."""
    var shape = List[Int]()
    shape.append(5)

    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, -5.0, DType.float32)

    # -2.0 > -5.0 should be True
    var c_greater = greater(a, b)
    for i in range(5):
        assert_value_at(c_greater, i, 1.0, 1e-6, "-2.0 > -5.0 should be True")

    # -2.0 < -5.0 should be False
    var c_less = less(a, b)
    for i in range(5):
        assert_value_at(c_less, i, 0.0, 1e-6, "-2.0 < -5.0 should be False")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run greater, greater_equal, and negative value comparison operation tests.
    """
    print(
        "Running ExTensor greater/greater_equal comparison operation tests..."
    )

    # greater() tests
    print("  Testing greater()...")
    test_greater_true()
    test_greater_false()
    test_greater_with_dunder()

    # greater_equal() tests
    print("  Testing greater_equal()...")
    test_greater_equal_true_greater()
    test_greater_equal_true_equal()
    test_greater_equal_with_dunder()

    # Negative values
    print("  Testing with negative values...")
    test_comparison_with_negatives()

    print("All greater/greater_equal comparison operation tests completed!")
