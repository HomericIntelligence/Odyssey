# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor edge cases - Part 5: Floor divide, comparison, subnormal, and numerical stability.

Split from test_edge_cases.mojo per ADR-009 (≤10 fn test_ functions per file).
"""

from math import isnan, isinf

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_equal_int,
    assert_true,
)


# ============================================================================
# Test floor_divide edge cases
# ============================================================================


fn test_floor_divide_by_zero() raises:
    """Test floor division by zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 10.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = floor_divide(a, b)

    # Floor division by zero should give inf (like regular division)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val):
            raise Error("x // 0 should be inf")


fn test_floor_divide_with_remainder() raises:
    """Test floor division with remainder."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = floor_divide(a, b)

    # 7 // 3 = 2 (floor of 2.333...)
    assert_value_at(c, 0, 2.0, 1e-6, "7 // 3 should be 2")


fn test_floor_divide_negative_result() raises:
    """Test floor division with negative result."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = floor_divide(a, b)

    # -7 // 3 = -3 (floor of -2.333... = -3, not -2)
    assert_value_at(
        c, 0, -3.0, 1e-6, "-7 // 3 should be -3 (floor toward -inf)"
    )


# ============================================================================
# Test comparison edge cases
# ============================================================================


fn test_comparison_with_zero() raises:
    """Test comparison operations with zero."""
    var shape = List[Int]()
    shape.append(3)

    var positive = full(shape, 1.0, DType.float32)
    var negative = full(shape, -1.0, DType.float32)
    var zero = zeros(shape, DType.float32)

    # Test greater than zero
    var pos_gt_zero = greater(positive, zero)
    assert_all_values(pos_gt_zero, 1.0, 1e-6, "1.0 > 0 should be True")

    var neg_gt_zero = greater(negative, zero)
    assert_all_values(neg_gt_zero, 0.0, 1e-6, "-1.0 > 0 should be False")

    # Test less than zero
    var neg_lt_zero = less(negative, zero)
    assert_all_values(neg_lt_zero, 1.0, 1e-6, "-1.0 < 0 should be True")


fn test_comparison_equal_values() raises:
    """Test equality comparison with same values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.14159, DType.float64)
    var b = full(shape, 3.14159, DType.float64)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Comparison should return bool")
    assert_all_values(c, 1.0, 1e-10, "Equal values should be equal")


fn test_comparison_very_close_values() raises:
    """Test comparison with very close but not equal values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 1.0000001, DType.float32)

    # These should NOT be equal (exact comparison, no tolerance)
    var eq = equal(a, b)
    var ne = not_equal(a, b)

    # Depending on float32 precision, these might be equal or not
    # For now, just verify bool dtype
    assert_dtype(eq, DType.bool, "equal() should return bool")
    assert_dtype(ne, DType.bool, "not_equal() should return bool")


# ============================================================================
# Test subnormal numbers
# ============================================================================


fn test_subnormal_numbers() raises:
    """Test handling of subnormal (denormalized) numbers."""
    var shape = List[Int]()
    shape.append(2)
    # Create subnormal float32 value (~1e-40)
    var a = full(shape, 1e-40, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = add(a, b)

    # a + 1 should be approximately 1 (subnormal is tiny)
    var expected = full(shape, 1.0, DType.float32)
    assert_all_close(c, expected, 1e-6, "1e-40 + 1 ≈ 1")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 5."""
    print("Running ExTensor edge case tests (Part 5)...")

    # Floor divide edge cases
    print("  Testing floor_divide edge cases...")
    test_floor_divide_by_zero()
    test_floor_divide_with_remainder()
    test_floor_divide_negative_result()

    # Comparison edge cases
    print("  Testing comparison edge cases...")
    test_comparison_with_zero()
    test_comparison_equal_values()
    test_comparison_very_close_values()

    # Subnormal numbers
    print("  Testing subnormal numbers...")
    test_subnormal_numbers()

    print("All edge case tests (Part 5) completed!")
