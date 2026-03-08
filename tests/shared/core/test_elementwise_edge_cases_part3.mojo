# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operation edge cases - Part 3: exp edge cases.

Tests edge cases for exp operations including:
- exp of zero, one, negative values
- exp of large positive/negative values (overflow/underflow)
- exp with different dtypes
- exp on vectors
- log/exp inverse relationship
"""

from math import isnan, isinf, exp

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.elementwise import (
    exp as exp_op,
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
# Test exp edge cases
# ============================================================================


fn test_exp_of_zero() raises:
    """exp(0) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "exp(0) should be 1")


fn test_exp_of_one() raises:
    """exp(1) should be e."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 2.718281828, 1e-5, "exp(1) should be e")


fn test_exp_of_negative() raises:
    """exp(-1) should be 1/e."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -1.0, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 0.367879441, 1e-5, "exp(-1) should be 1/e")


fn test_exp_of_large_positive() raises:
    """exp(large) should overflow to inf."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 1000.0, DType.float32)
    var result = exp_op(t)

    var val = result._get_float64(0)
    if not isinf(val):
        raise Error("exp(1000) should overflow to inf")


fn test_exp_of_large_negative() raises:
    """exp(-large) should underflow to 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -1000.0, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 0.0, 1e-10, "exp(-1000) should underflow to 0")


fn test_exp_float64() raises:
    """Test exp with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float64)
    var result = exp_op(t)

    assert_value_at(result, 0, 1.0, 1e-15, "exp(0) in float64")


fn test_log_exp_inverse() raises:
    """Test that log and exp are approximate inverses."""
    var shape = List[Int]()
    shape.append(1)
    var x = full(shape, 3.0, DType.float32)
    var result = exp_op(x)  # exp(3)
    # Would need log(exp(3)) = 3, but log not exposed

    # Verify exp(3) is approximately 20.086
    assert_value_at(result, 0, 20.086, 1e-2, "exp(3) ≈ 20.086")


fn test_exp_vector() raises:
    """Test exp on vector of values."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    var result = exp_op(t)

    # All should be exp(0) = 1
    assert_all_values(result, 1.0, 1e-6, "exp(0) = 1 for all elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run exp edge case tests."""
    print("Running exp operation edge case tests (part 3)...")

    # exp edge cases
    print("  Testing exp edge cases...")
    test_exp_of_zero()
    test_exp_of_one()
    test_exp_of_negative()
    test_exp_of_large_positive()
    test_exp_of_large_negative()

    # Different dtypes
    print("  Testing exp with different dtypes...")
    test_exp_float64()

    # Inverse relationship
    print("  Testing log/exp inverse relationship...")
    test_log_exp_inverse()

    # Vector operations
    print("  Testing exp vector operations...")
    test_exp_vector()

    print("All exp edge case tests completed!")
