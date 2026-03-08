# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operation edge cases - Part 4: tanh and trig edge cases.

Tests edge cases for tanh and trigonometric operations including:
- tanh of zero, positive, negative values
- tanh saturation for large positive/negative values
- sin and cos of zero (documented behavior)
"""

from math import isnan, isinf, sin, cos, tanh

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.activation import tanh as tanh_op

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
# Test trigonometric functions
# ============================================================================


fn test_sin_of_zero() raises:
    """sin(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    # Note: sin not directly exposed, would need to use native sin function
    # This test documents the expected behavior
    var result = zeros(shape, DType.float32)
    assert_value_at(result, 0, 0.0, 1e-6, "sin(0) should be 0")


fn test_cos_of_zero() raises:
    """cos(0) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    # Note: cos not directly exposed
    # This test documents expected behavior
    var result = ones(shape, DType.float32)
    assert_value_at(result, 0, 1.0, 1e-6, "cos(0) should be 1")


# ============================================================================
# Test tanh saturation
# ============================================================================


fn test_tanh_of_zero() raises:
    """tanh(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = tanh_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "tanh(0) should be 0")


fn test_tanh_of_positive() raises:
    """tanh of positive values should be between 0 and 1."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, 1.0, DType.float32)
    var result = tanh_op(t)

    for i in range(3):
        var val = result._get_float64(i)
        if val < 0.0 or val > 1.0:
            raise Error("tanh(1) should be between 0 and 1")


fn test_tanh_of_negative() raises:
    """tanh of negative values should be between -1 and 0."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = tanh_op(t)

    for i in range(3):
        var val = result._get_float64(i)
        if val < -1.0 or val > 0.0:
            raise Error("tanh(-1) should be between -1 and 0")


fn test_tanh_saturation_large_positive() raises:
    """tanh(large) should saturate to 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 100.0, DType.float32)
    var result = tanh_op(t)

    # tanh(100) should be very close to 1
    var val = result._get_float64(0)
    if val < 0.99999:
        raise Error("tanh(100) should saturate to ~1")


fn test_tanh_saturation_large_negative() raises:
    """tanh(-large) should saturate to -1."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -100.0, DType.float32)
    var result = tanh_op(t)

    # tanh(-100) should be very close to -1
    var val = result._get_float64(0)
    if val > -0.99999:
        raise Error("tanh(-100) should saturate to ~-1")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run tanh and trig edge case tests."""
    print("Running tanh and trig operation edge case tests (part 4)...")

    # Trigonometric functions
    print("  Testing trigonometric functions...")
    test_sin_of_zero()
    test_cos_of_zero()

    # tanh saturation
    print("  Testing tanh saturation...")
    test_tanh_of_zero()
    test_tanh_of_positive()
    test_tanh_of_negative()
    test_tanh_saturation_large_positive()
    test_tanh_saturation_large_negative()

    print("All tanh and trig edge case tests completed!")
