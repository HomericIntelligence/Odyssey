"""Tests for ExTensor element-wise mathematical operations - Part 5: clip and dtype.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import (
    ones,
    full,
    arange,
    abs,
    sign,
    exp,
    log,
    sqrt,
    clip,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test clip() - Clip values to range
# ============================================================================


fn test_clip_basic() raises:
    """Test clip with basic range."""
    # Create array: [1, 2, 3, 4, 5]
    var a = arange(1.0, 6.0, 1.0, DType.float32)
    var b = clip(a, 2.0, 4.0)

    # Expected: [2, 2, 3, 4, 4]
    assert_value_at(b, 0, 2.0, 1e-6, "clip(1, 2, 4) = 2")
    assert_value_at(b, 1, 2.0, 1e-6, "clip(2, 2, 4) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "clip(3, 2, 4) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "clip(4, 2, 4) = 4")
    assert_value_at(b, 4, 4.0, 1e-6, "clip(5, 2, 4) = 4")


fn test_clip_all_below() raises:
    """Test clip when all values below min."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = clip(a, 5.0, 10.0)

    # All values should be clipped to min
    assert_all_values(b, 5.0, 1e-6, "clip(1, 5, 10) should be 5")


fn test_clip_all_above() raises:
    """Test clip when all values above max."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 20.0, DType.float32)
    var b = clip(a, 5.0, 10.0)

    # All values should be clipped to max
    assert_all_values(b, 10.0, 1e-6, "clip(20, 5, 10) should be 10")


# ============================================================================
# Test dtype preservation for all operations
# ============================================================================


fn test_operations_preserve_dtype() raises:
    """Test that all operations preserve dtype."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float64)

    # All operations should preserve float64
    var b_abs = abs(a)
    assert_dtype(b_abs, DType.float64, "abs should preserve dtype")

    var b_sign = sign(a)
    assert_dtype(b_sign, DType.float64, "sign should preserve dtype")

    var b_exp = exp(a)
    assert_dtype(b_exp, DType.float64, "exp should preserve dtype")

    var b_log = log(a)
    assert_dtype(b_log, DType.float64, "log should preserve dtype")

    var b_sqrt = sqrt(a)
    assert_dtype(b_sqrt, DType.float64, "sqrt should preserve dtype")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run clip and dtype preservation element-wise math tests."""
    print("Running ExTensor element-wise math tests (Part 5: clip, dtype)...")

    # clip() tests
    print("  Testing clip()...")
    test_clip_basic()
    test_clip_all_below()
    test_clip_all_above()

    # dtype preservation
    print("  Testing dtype preservation...")
    test_operations_preserve_dtype()

    print("All element-wise math tests (Part 5) completed!")
