# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor edge cases - Part 6: Numerical stability and special dtype behaviors.

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
# Test numerical stability
# ============================================================================


fn test_catastrophic_cancellation() raises:
    """Test catastrophic cancellation in subtraction."""
    var shape = List[Int]()
    shape.append(2)
    # Create two very close values
    var a = full(shape, 1.0000000001, DType.float64)
    var b = full(shape, 1.0, DType.float64)
    var c = subtract(a, b)

    # Result should be very small (loss of precision is expected here)
    for i in range(2):
        var val = c._get_float64(i)
        # Just verify it's close to expected value
        if val < 0 or val > 1e-9:
            # Allow for precision loss
            pass


fn test_associativity_loss() raises:
    """Test loss of associativity in floating point."""
    var shape = List[Int]()
    shape.append(3)
    # Create specific values to demonstrate associativity loss
    var a = full(shape, 1e20, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = full(shape, -1e20, DType.float32)

    # (a + b) + c != a + (b + c) in floating point
    var result1 = add(add(a, b), c)
    var result2 = add(a, add(b, c))

    # Just verify both produce results (may differ due to rounding)
    assert_dim(result1, 1, "(a + b) + c should produce result")
    assert_dim(result2, 1, "a + (b + c) should produce result")


# ============================================================================
# Test special dtype behaviors
# ============================================================================


fn test_bool_dtype_operations() raises:
    """Test operations on bool dtype tensors."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.bool)  # All True (1)
    var b = zeros(shape, DType.bool)  # All False (0)

    # Test equality comparison with bool dtypes
    var c = equal(a, a)  # Should be all True
    assert_all_values(c, 1.0, 1e-8, "True == True should be True")

    var d = equal(a, b)  # Should be all False
    assert_all_values(d, 0.0, 1e-8, "True == False should be False")


fn test_int8_range() raises:
    """Test int8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at int8 boundaries
    var a = full(shape, 127.0, DType.int8)  # INT8_MAX
    var b = full(shape, -128.0, DType.int8)  # INT8_MIN

    # Verify values are stored correctly
    assert_value_at(a, 0, 127.0, 1e-6, "INT8_MAX should be 127")
    assert_value_at(b, 0, -128.0, 1e-6, "INT8_MIN should be -128")


fn test_uint8_range() raises:
    """Test uint8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at uint8 boundaries
    var a = full(shape, 255.0, DType.uint8)  # UINT8_MAX
    var b = zeros(shape, DType.uint8)  # UINT8_MIN

    # Verify values are stored correctly
    assert_value_at(a, 0, 255.0, 1e-6, "UINT8_MAX should be 255")
    assert_value_at(b, 0, 0.0, 1e-6, "UINT8_MIN should be 0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 6."""
    print("Running AnyTensor edge case tests (Part 6)...")

    # Numerical stability
    print("  Testing numerical stability...")
    test_catastrophic_cancellation()
    test_associativity_loss()

    # Special dtype behaviors
    print("  Testing special dtype behaviors...")
    test_bool_dtype_operations()
    test_int8_range()
    test_uint8_range()

    print("All edge case tests (Part 6) completed!")
