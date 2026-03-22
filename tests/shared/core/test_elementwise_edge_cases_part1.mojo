# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operation edge cases - Part 1: sqrt edge cases.

Tests edge cases for sqrt operations including:
- sqrt of zero, one, negative numbers
- sqrt of small and large positive numbers
- sqrt with different dtypes
- sqrt on vectors
"""

from math import isnan, isinf, sqrt

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.elementwise import (
    sqrt as sqrt_op,
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
# Test sqrt edge cases
# ============================================================================


fn test_sqrt_of_zero() raises:
    """Ssqrt(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "sqrt(0) should be 0")


fn test_sqrt_of_one() raises:
    """Ssqrt(1) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "sqrt(1) should be 1")


fn test_sqrt_of_negative() raises:
    """Ssqrt(-1) should return NaN (IEEE 754 behavior)."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = sqrt_op(t)

    # sqrt of negative should be NaN
    for i in range(3):
        var val = result._get_float64(i)
        assert_true(isnan(val), "sqrt(-1) should be NaN")


fn test_sqrt_of_small_positive() raises:
    """Ssqrt of small positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 0.25, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 0.5, 1e-5, "sqrt(0.25) should be 0.5")


fn test_sqrt_of_large_positive() raises:
    """Ssqrt of large positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 10000.0, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 100.0, 1e-3, "sqrt(10000) should be 100")


fn test_sqrt_numerical_stability() raises:
    """Test sqrt numerical stability with typical values."""
    var shape = List[Int]()
    shape.append(5)
    var t = full(shape, 2.0, DType.float32)
    var result = sqrt_op(t)

    # sqrt(2) should be approximately 1.414...
    var expected = full(shape, 1.414213562, DType.float32)
    assert_all_close(result, expected, 1e-4, "sqrt(2) numerical stability")


fn test_sqrt_float64() raises:
    """Test sqrt with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 4.0, DType.float64)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 2.0, 1e-10, "sqrt(4.0) in float64")


fn test_sqrt_vector() raises:
    """Test sqrt on vector of values."""
    var shape = List[Int]()
    shape.append(4)
    var vals = List[Float32]()
    vals.append(1.0)
    vals.append(4.0)
    vals.append(9.0)
    vals.append(16.0)
    var t = AnyTensor(shape, DType.float32)
    for i in range(4):
        t._set_float32(i, vals[i])

    var result = sqrt_op(t)

    # Check approximate results
    assert_value_at(result, 0, 1.0, 1e-5, "sqrt(1) = 1")
    assert_value_at(result, 1, 2.0, 1e-5, "sqrt(4) = 2")
    assert_value_at(result, 2, 3.0, 1e-5, "sqrt(9) = 3")
    assert_value_at(result, 3, 4.0, 1e-5, "sqrt(16) = 4")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run sqrt edge case tests."""
    print("Running sqrt operation edge case tests (part 1)...")

    # sqrt edge cases
    print("  Testing sqrt edge cases...")
    test_sqrt_of_zero()
    test_sqrt_of_one()
    test_sqrt_of_negative()
    test_sqrt_of_small_positive()
    test_sqrt_of_large_positive()

    # Numerical stability
    print("  Testing sqrt numerical stability...")
    test_sqrt_numerical_stability()

    # Different dtypes
    print("  Testing sqrt with different dtypes...")
    test_sqrt_float64()

    # Vector operations
    print("  Testing sqrt vector operations...")
    test_sqrt_vector()

    print("All sqrt edge case tests completed!")
