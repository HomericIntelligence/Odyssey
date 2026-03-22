"""Tests for AnyTensor element-wise mathematical operations - Part 4: cos and tanh.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import zeros, ones, full
from shared.core.elementwise import cos, tanh

# Import test helpers
from tests.shared.conftest import (
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test cos() - Cosine
# ============================================================================


fn test_cos_zero() raises:
    """Test cos(0) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = cos(a)

    # cos(0) = 1
    assert_all_values(b, 1.0, 1e-6, "cos(0) should be 1")


fn test_cos_pi_over_2() raises:
    """Test cos(π/2) ≈ 0."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.5708, DType.float32)  # π/2
    var b = cos(a)

    # cos(π/2) ≈ 0
    assert_value_at(b, 0, 0.0, 1e-4, "cos(π/2) should be ~0")


fn test_cos_pi() raises:
    """Test cos(π) = -1."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 3.14159, DType.float32)  # π
    var b = cos(a)

    # cos(π) = -1
    assert_value_at(b, 0, -1.0, 1e-4, "cos(π) should be ~-1")


# ============================================================================
# Test tanh() - Hyperbolic tangent
# ============================================================================


fn test_tanh_zero() raises:
    """Test tanh(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = tanh(a)

    # tanh(0) = 0
    assert_all_values(b, 0.0, 1e-6, "tanh(0) should be 0")


fn test_tanh_large_positive() raises:
    """Test tanh(large) ≈ 1."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 10.0, DType.float32)
    var b = tanh(a)

    # tanh(10) ≈ 1 (saturates)
    assert_value_at(b, 0, 1.0, 1e-5, "tanh(large) should be ~1")


fn test_tanh_large_negative() raises:
    """Test tanh(-large) ≈ -1."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -10.0, DType.float32)
    var b = tanh(a)

    # tanh(-10) ≈ -1 (saturates)
    assert_value_at(b, 0, -1.0, 1e-5, "tanh(-large) should be ~-1")


fn test_tanh_small_values() raises:
    """Test tanh with small values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 0.5, DType.float32)
    var b = tanh(a)

    # tanh(0.5) ≈ 0.46212
    assert_value_at(b, 0, 0.46212, 1e-4, "tanh(0.5) should be ~0.462")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run cos and tanh element-wise math tests."""
    print("Running AnyTensor element-wise math tests (Part 4: cos, tanh)...")

    # cos() tests
    print("  Testing cos()...")
    test_cos_zero()
    test_cos_pi_over_2()
    test_cos_pi()

    # tanh() tests
    print("  Testing tanh()...")
    test_tanh_zero()
    test_tanh_large_positive()
    test_tanh_large_negative()
    test_tanh_small_values()

    print("All element-wise math tests (Part 4) completed!")
