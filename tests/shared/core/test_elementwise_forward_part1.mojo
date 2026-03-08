"""Tests for ExTensor element-wise mathematical operations - Part 1: abs and sign.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    abs,
    sign,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test abs() - Absolute value
# ============================================================================


fn test_abs_positive() raises:
    """Test abs with positive values."""
    var shape = List[Int]()
    shape.append(5)
    var a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    var b = abs(a)

    # Positive values should remain unchanged
    assert_value_at(b, 0, 1.0, 1e-6, "abs(1) = 1")
    assert_value_at(b, 1, 2.0, 1e-6, "abs(2) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "abs(3) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "abs(4) = 4")
    assert_value_at(b, 4, 5.0, 1e-6, "abs(5) = 5")


fn test_abs_negative() raises:
    """Test abs with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -3.0, DType.float32)
    var b = abs(a)

    # Should convert to positive
    assert_all_values(b, 3.0, 1e-6, "abs(negative) = positive")


fn test_abs_mixed() raises:
    """Test abs with mixed positive/negative values."""
    # Create array: [-2, -1, 0, 1, 2]
    var a = arange(-2.0, 3.0, 1.0, DType.float32)
    var b = abs(a)

    # Expected: [2, 1, 0, 1, 2]
    assert_value_at(b, 0, 2.0, 1e-6, "abs(-2) = 2")
    assert_value_at(b, 1, 1.0, 1e-6, "abs(-1) = 1")
    assert_value_at(b, 2, 0.0, 1e-6, "abs(0) = 0")
    assert_value_at(b, 3, 1.0, 1e-6, "abs(1) = 1")
    assert_value_at(b, 4, 2.0, 1e-6, "abs(2) = 2")


fn test_abs_preserves_dtype() raises:
    """Test that abs preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = abs(a)

    assert_dtype(b, DType.float64, "abs should preserve float64")


# ============================================================================
# Test sign() - Sign of values
# ============================================================================


fn test_sign_positive() raises:
    """Test sign with positive values."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = sign(a)

    # Positive values should give +1
    assert_all_values(b, 1.0, 1e-6, "sign(positive) = 1")


fn test_sign_negative() raises:
    """Test sign with negative values."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, -5.0, DType.float32)
    var b = sign(a)

    # Negative values should give -1
    assert_all_values(b, -1.0, 1e-6, "sign(negative) = -1")


fn test_sign_zero() raises:
    """Test sign with zero values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = sign(a)

    # Zero should give 0
    assert_all_values(b, 0.0, 1e-6, "sign(0) = 0")


fn test_sign_mixed() raises:
    """Test sign with mixed values."""
    # Create array: [-2, -1, 0, 1, 2]
    var a = arange(-2.0, 3.0, 1.0, DType.float32)
    var b = sign(a)

    # Expected: [-1, -1, 0, 1, 1]
    assert_value_at(b, 0, -1.0, 1e-6, "sign(-2) = -1")
    assert_value_at(b, 1, -1.0, 1e-6, "sign(-1) = -1")
    assert_value_at(b, 2, 0.0, 1e-6, "sign(0) = 0")
    assert_value_at(b, 3, 1.0, 1e-6, "sign(1) = 1")
    assert_value_at(b, 4, 1.0, 1e-6, "sign(2) = 1")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run abs and sign element-wise math tests."""
    print("Running ExTensor element-wise math tests (Part 1: abs, sign)...")

    # abs() tests
    print("  Testing abs()...")
    test_abs_positive()
    test_abs_negative()
    test_abs_mixed()
    test_abs_preserves_dtype()

    # sign() tests
    print("  Testing sign()...")
    test_sign_positive()
    test_sign_negative()
    test_sign_zero()
    test_sign_mixed()

    print("All element-wise math tests (Part 1) completed!")
