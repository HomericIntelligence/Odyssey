# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operation edge cases - Part 2: log edge cases.

Tests edge cases for log operations including:
- log of zero, one, negative numbers
- log of small positive numbers
- log of e (natural logarithm verification)
"""

from math import isnan, isinf, log

# Import AnyTensor and operations
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.elementwise import (
    log as log_op,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_true,
)


# ============================================================================
# Test log edge cases
# ============================================================================


fn test_log_of_zero() raises:
    """Llog(0) should be -inf."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    var result = log_op(t)

    # log(0) = -inf
    for i in range(3):
        var val = result._get_float64(i)
        if not isinf(val) or val > 0:
            raise Error("log(0) should be -inf")


fn test_log_of_one() raises:
    """Llog(1) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = log_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "log(1) should be 0")


fn test_log_of_negative() raises:
    """Llog(-1) should be NaN (IEEE 754)."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = log_op(t)

    # log of negative should be NaN
    for i in range(3):
        var val = result._get_float64(i)
        assert_true(isnan(val), "log(-1) should be NaN")


fn test_log_of_small_positive() raises:
    """Llog of very small positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 1e-10, DType.float32)
    var result = log_op(t)

    # log(1e-10) should be large negative value
    var val = result._get_float64(0)
    if val > -20.0:
        raise Error("log(1e-10) should be large negative")


fn test_log_of_e() raises:
    """Llog(e) should be 1 (natural logarithm)."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 2.718281828, DType.float64)
    var result = log_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "log(e) should be 1")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run log edge case tests."""
    print("Running log operation edge case tests (part 2)...")

    # log edge cases
    print("  Testing log edge cases...")
    test_log_of_zero()
    test_log_of_one()
    test_log_of_negative()
    test_log_of_small_positive()
    test_log_of_e()

    print("All log edge case tests completed!")
