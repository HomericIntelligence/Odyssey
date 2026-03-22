"""Tests for AnyTensor element-wise mathematical operations - Part 3: sqrt and sin.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.elementwise import sqrt, sin

# Import test helpers
from tests.shared.conftest import (
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test sqrt() - Square root
# ============================================================================


fn test_sqrt_perfect_squares() raises:
    """Test sqrt with perfect squares."""
    # Create array: [1, 4, 9, 16, 25]
    var shape = List[Int]()
    shape.append(5)
    var a = AnyTensor(shape, DType.float32)
    a._set_float64(0, 1.0)
    a._set_float64(1, 4.0)
    a._set_float64(2, 9.0)
    a._set_float64(3, 16.0)
    a._set_float64(4, 25.0)
    var b = sqrt(a)

    # Expected: [1, 2, 3, 4, 5]
    assert_value_at(b, 0, 1.0, 1e-6, "sqrt(1) = 1")
    assert_value_at(b, 1, 2.0, 1e-6, "sqrt(4) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "sqrt(9) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "sqrt(16) = 4")
    assert_value_at(b, 4, 5.0, 1e-6, "sqrt(25) = 5")


fn test_sqrt_zero() raises:
    """Test sqrt(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = sqrt(a)

    # sqrt(0) = 0
    assert_all_values(b, 0.0, 1e-6, "sqrt(0) should be 0")


fn test_sqrt_one() raises:
    """Test sqrt(1) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = sqrt(a)

    # sqrt(1) = 1
    assert_all_values(b, 1.0, 1e-6, "sqrt(1) should be 1")


fn test_sqrt_two() raises:
    """Test sqrt(2) ≈ 1.41421."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 2.0, DType.float32)
    var b = sqrt(a)

    # sqrt(2) ≈ 1.41421
    assert_value_at(b, 0, 1.41421, 1e-4, "sqrt(2) should be ~1.414")


# ============================================================================
# Test sin() - Sine
# ============================================================================


fn test_sin_zero() raises:
    """Test sin(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = sin(a)

    # sin(0) = 0
    assert_all_values(b, 0.0, 1e-6, "sin(0) should be 0")


fn test_sin_pi_over_2() raises:
    """Test sin(π/2) = 1."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.5708, DType.float32)  # π/2 ≈ 1.5708
    var b = sin(a)

    # sin(π/2) = 1
    assert_value_at(b, 0, 1.0, 1e-4, "sin(π/2) should be 1")


fn test_sin_pi() raises:
    """Test sin(π) ≈ 0."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 3.14159, DType.float32)  # π
    var b = sin(a)

    # sin(π) ≈ 0
    assert_value_at(b, 0, 0.0, 1e-5, "sin(π) should be ~0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run sqrt and sin element-wise math tests."""
    print("Running AnyTensor element-wise math tests (Part 3: sqrt, sin)...")

    # sqrt() tests
    print("  Testing sqrt()...")
    test_sqrt_perfect_squares()
    test_sqrt_zero()
    test_sqrt_one()
    test_sqrt_two()

    # sin() tests
    print("  Testing sin()...")
    test_sin_zero()
    test_sin_pi_over_2()
    test_sin_pi()

    print("All element-wise math tests (Part 3) completed!")
