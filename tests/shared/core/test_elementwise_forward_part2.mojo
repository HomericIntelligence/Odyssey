"""Tests for AnyTensor element-wise mathematical operations - Part 2: exp and log.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import zeros, ones, full
from shared.core.elementwise import exp, log

# Import test helpers
from tests.shared.conftest import (
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test exp() - Exponential
# ============================================================================


fn test_exp_zeros() raises:
    """Test exp(0) = 1."""
    var shape = List[Int]()
    shape.append(5)
    var a = zeros(shape, DType.float32)
    var b = exp(a)

    # exp(0) = 1
    assert_all_values(b, 1.0, 1e-6, "exp(0) should be 1")


fn test_exp_ones() raises:
    """Test exp(1) ≈ 2.71828."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = exp(a)

    # exp(1) ≈ e ≈ 2.71828
    assert_all_values(b, 2.71828, 1e-5, "exp(1) should be approximately e")


fn test_exp_small_values() raises:
    """Test exp with small values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 0.5, DType.float32)
    var b = exp(a)

    # exp(0.5) ≈ 1.64872
    assert_value_at(b, 0, 1.64872, 1e-4, "exp(0.5) should be ~1.649")


fn test_exp_negative() raises:
    """Test exp with negative values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -1.0, DType.float32)
    var b = exp(a)

    # exp(-1) ≈ 0.36788 (1/e)
    assert_value_at(b, 0, 0.36788, 1e-4, "exp(-1) should be ~0.368")


# ============================================================================
# Test log() - Natural logarithm
# ============================================================================


fn test_log_one() raises:
    """Test log(1) = 0."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = log(a)

    # log(1) = 0
    assert_all_values(b, 0.0, 1e-6, "log(1) should be 0")


fn test_log_e() raises:
    """Test log(e) = 1."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 2.71828, DType.float32)  # e
    var b = log(a)

    # log(e) = 1
    assert_value_at(b, 0, 1.0, 1e-4, "log(e) should be 1")


fn test_log_powers_of_2() raises:
    """Test log with powers of 2."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 2.0, DType.float32)
    var b = log(a)

    # log(2) ≈ 0.69315
    assert_value_at(b, 0, 0.69315, 1e-4, "log(2) should be ~0.693")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run exp and log element-wise math tests."""
    print("Running AnyTensor element-wise math tests (Part 2: exp, log)...")

    # exp() tests
    print("  Testing exp()...")
    test_exp_zeros()
    test_exp_ones()
    test_exp_small_values()
    test_exp_negative()

    # log() tests
    print("  Testing log()...")
    test_log_one()
    test_log_e()
    test_log_powers_of_2()

    print("All element-wise math tests (Part 2) completed!")
