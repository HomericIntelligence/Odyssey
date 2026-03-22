# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor edge cases - Part 4: Modulo and power edge cases.

Split from test_edge_cases.mojo per ADR-009 (≤10 fn test_ functions per file).
"""

from math import isnan, isinf

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange, nan_tensor, inf_tensor, neg_inf_tensor
from shared.core.arithmetic import add, subtract, multiply, divide, floor_divide, modulo, power
from shared.core.comparison import equal, not_equal, less, less_equal, greater, greater_equal

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
# Test modulo edge cases
# ============================================================================


fn test_modulo_by_zero() raises:
    """Test modulo by zero (should give NaN for floats)."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = modulo(a, b)

    # Modulo by zero: undefined, should give NaN
    for i in range(3):
        var val = c._get_float64(i)
        if not isnan(val):
            raise Error("x % 0 should be NaN")


fn test_modulo_with_negative_divisor() raises:
    """Test modulo with negative divisor."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, -3.0, DType.float32)
    var c = modulo(a, b)

    # Python semantics: 7 % -3 = -2 (result has sign of divisor)
    assert_value_at(c, 0, -2.0, 1e-6, "7 % -3 should be -2 (Python semantics)")


fn test_modulo_both_negative() raises:
    """Test modulo with both negative values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, -3.0, DType.float32)
    var c = modulo(a, b)

    # Python semantics: -7 % -3 = -1
    assert_value_at(c, 0, -1.0, 1e-6, "-7 % -3 should be -1 (Python semantics)")


# ============================================================================
# Test power edge cases
# ============================================================================


fn test_power_zero_to_zero() raises:
    """Test 0^0 (mathematically undefined, conventionally 1)."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = power(a, b)

    # Convention: 0^0 = 1 (used in polynomial evaluation)
    assert_all_values(c, 1.0, 1e-6, "0^0 should be 1 by convention")


fn test_power_negative_base_even() raises:
    """Test negative base with even exponent."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = power(a, b)

    # (-2)^2 = 4
    assert_value_at(c, 0, 4.0, 1e-6, "(-2)^2 should be 4")


fn test_power_negative_base_odd() raises:
    """Test negative base with odd exponent."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = power(a, b)

    # (-2)^3 = -8
    assert_value_at(c, 0, -8.0, 1e-6, "(-2)^3 should be -8")


fn test_power_zero_base_positive_exp() raises:
    """Test 0^n for positive n."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = full(shape, 5.0, DType.float32)
    var c = power(a, b)

    # 0^5 = 0
    assert_all_values(c, 0.0, 1e-6, "0^n should be 0 for positive n")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 4."""
    print("Running AnyTensor edge case tests (Part 4)...")

    # Modulo edge cases
    print("  Testing modulo edge cases...")
    test_modulo_by_zero()
    test_modulo_with_negative_divisor()
    test_modulo_both_negative()

    # Power edge cases
    print("  Testing power edge cases...")
    test_power_zero_to_zero()
    test_power_negative_base_even()
    test_power_negative_base_odd()
    test_power_zero_base_positive_exp()

    print("All edge case tests (Part 4) completed!")
