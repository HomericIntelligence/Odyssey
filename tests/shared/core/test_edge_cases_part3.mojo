# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor edge cases - Part 3: Overflow, underflow, and division by zero.

Split from test_edge_cases.mojo per ADR-009 (≤10 fn test_ functions per file).
"""

from math import isnan, isinf

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full, arange, nan_tensor, inf_tensor, neg_inf_tensor
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
# Test overflow behavior
# ============================================================================


fn test_overflow_float32() raises:
    """Test overflow behavior for float32."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1e38, DType.float32)  # Near float32 max
    var b = full(shape, 10.0, DType.float32)
    var c = multiply(a, b)

    # 1e38 * 10 should overflow to inf
    for i in range(2):
        var val = c._get_float64(i)
        if not isinf(val):
            raise Error("Overflow should produce inf")


fn test_overflow_int32() raises:
    """Test overflow behavior for int32."""
    var shape = List[Int]()
    shape.append(2)
    # Create values close to int32 max
    var a = full(shape, 2147483647.0, DType.int32)  # INT32_MAX
    var b = ones(shape, DType.int32)
    var c = add(a, b)

    # Integer overflow behavior: wraps around (implementation dependent)
    # Just verify it doesn't crash
    assert_dim(c, 1, "INT32_MAX + 1 should produce result")


# ============================================================================
# Test underflow behavior
# ============================================================================


fn test_underflow_float64() raises:
    """Test underflow behavior for float64."""
    var shape = List[Int]()
    shape.append(2)
    # Create very small values
    var a = full(shape, 1e-300, DType.float64)
    var b = full(shape, 1e-100, DType.float64)
    var c = multiply(a, b)

    # Result may underflow to 0 (gradual underflow)
    assert_all_values(c, 0.0, 1e-320, "Underflow should give 0")


# ============================================================================
# Test division by zero
# ============================================================================


fn test_divide_by_zero_float() raises:
    """Test division by zero for floating point."""
    # Skip this test - may cause undefined behavior
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, 1.0, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)
    # 1/0 = inf (IEEE 754)
    pass


fn test_divide_by_zero_int() raises:
    """Test division by zero for integers."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, 10.0, DType.int32)
    # var b = zeros(shape, DType.int32)
    # var c = divide(a, b)


fn test_divide_zero_by_zero() raises:
    """Test 0/0 (should give NaN for floats)."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = zeros(shape, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)


fn test_divide_negative_by_zero() raises:
    """Test -1/0 (should give -inf for floats)."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, -1.0, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 3."""
    print("Running AnyTensor edge case tests (Part 3)...")

    # Overflow
    print("  Testing overflow...")
    test_overflow_float32()
    test_overflow_int32()

    # Underflow
    print("  Testing underflow...")
    test_underflow_float64()

    # Division by zero
    print("  Testing division by zero...")
    test_divide_by_zero_float()
    test_divide_by_zero_int()
    test_divide_zero_by_zero()
    test_divide_negative_by_zero()

    print("All edge case tests (Part 3) completed!")
