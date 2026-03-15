# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor edge cases - Part 2: NaN handling and infinity handling.

Split from test_edge_cases.mojo per ADR-009 (≤10 fn test_ functions per file).
"""

from math import isnan, isinf

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange, nan_tensor, inf_tensor, neg_inf_tensor
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
# Test NaN handling
# ============================================================================


fn test_nan_propagation_add() raises:
    """Test that NaN propagates through addition."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # NaN + x = NaN
    for i in range(3):
        var val = c._get_float64(i)
        assert_true(isnan(val), "NaN should propagate through addition")


fn test_nan_propagation_multiply() raises:
    """Test that NaN propagates through multiplication."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    # NaN * 0 = NaN (not 0!)
    for i in range(3):
        var val = c._get_float64(i)
        assert_true(isnan(val), "NaN * 0 should be NaN")


fn test_nan_equality() raises:
    """Test NaN equality (NaN != NaN per IEEE 754)."""
    var shape = List[Int]()
    shape.append(1)
    var a = nan_tensor(shape, DType.float32)
    var b = nan_tensor(shape, DType.float32)
    var c = equal(a, b)

    # IEEE 754: NaN != NaN
    assert_value_at(c, 0, 0.0, 1e-8, "NaN != NaN should be False")


# ============================================================================
# Test infinity handling
# ============================================================================


fn test_inf_arithmetic() raises:
    """Test arithmetic with infinity."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # inf + x = inf (except inf + (-inf) = NaN)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val < 0:
            raise Error("inf + 1 should be inf")


fn test_inf_multiplication() raises:
    """Test infinity multiplication."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = multiply(a, b)

    # inf * x = inf (for positive x)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val < 0:
            raise Error("inf * 2 should be inf")


fn test_inf_times_zero() raises:
    """Test infinity times zero (should give NaN)."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    # inf * 0 = NaN (indeterminate form)
    for i in range(3):
        var val = c._get_float64(i)
        if not isnan(val):
            raise Error("inf * 0 should be NaN")


fn test_negative_inf() raises:
    """Test negative infinity."""
    var shape = List[Int]()
    shape.append(3)
    var a = neg_inf_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # -inf + x = -inf (for finite x)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val > 0:
            raise Error("-inf + 1 should be -inf")


fn test_inf_comparison() raises:
    """Test comparison with infinity."""
    var shape = List[Int]()
    shape.append(1)
    var a = inf_tensor(shape, DType.float32)
    var b = full(shape, 1000.0, DType.float32)
    var c = greater(a, b)

    # inf > 1000 should be True
    assert_value_at(c, 0, 1.0, 1e-8, "inf > 1000 should be True")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 2."""
    print("Running ExTensor edge case tests (Part 2)...")

    # NaN handling
    print("  Testing NaN handling...")
    test_nan_propagation_add()
    test_nan_propagation_multiply()
    test_nan_equality()

    # Infinity handling
    print("  Testing infinity handling...")
    test_inf_arithmetic()
    test_inf_multiplication()
    test_inf_times_zero()
    test_negative_inf()
    test_inf_comparison()

    print("All edge case tests (Part 2) completed!")
