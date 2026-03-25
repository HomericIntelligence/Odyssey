"""Tests for AnyTensor element-wise mathematical operations

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_forward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""


from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.elementwise import abs, sign
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)
from shared.tensor.any_tensor import zeros, ones, full
from shared.core.elementwise import exp, log
from tests.shared.conftest import (
    assert_value_at,
    assert_all_values,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.elementwise import sqrt, sin
from shared.core.elementwise import cos, tanh
from shared.tensor.any_tensor import ones, full, arange
from shared.core.elementwise import abs, sign, exp, log, sqrt, clip
from tests.shared.conftest import (
    assert_dtype,
    assert_value_at,
    assert_all_values,
)


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


fn main() raises:
    """Run all test_elementwise_forward tests."""
    print("Running test_elementwise_forward tests...")

    test_abs_positive()
    print("✓ test_abs_positive")

    test_abs_negative()
    print("✓ test_abs_negative")

    test_abs_mixed()
    print("✓ test_abs_mixed")

    test_abs_preserves_dtype()
    print("✓ test_abs_preserves_dtype")

    test_sign_positive()
    print("✓ test_sign_positive")

    test_sign_negative()
    print("✓ test_sign_negative")

    test_sign_zero()
    print("✓ test_sign_zero")

    test_sign_mixed()
    print("✓ test_sign_mixed")

    test_exp_zeros()
    print("✓ test_exp_zeros")

    test_exp_ones()
    print("✓ test_exp_ones")

    test_exp_small_values()
    print("✓ test_exp_small_values")

    test_exp_negative()
    print("✓ test_exp_negative")

    test_log_one()
    print("✓ test_log_one")

    test_log_e()
    print("✓ test_log_e")

    test_log_powers_of_2()
    print("✓ test_log_powers_of_2")

    test_sqrt_perfect_squares()
    print("✓ test_sqrt_perfect_squares")

    test_sqrt_zero()
    print("✓ test_sqrt_zero")

    test_sqrt_one()
    print("✓ test_sqrt_one")

    test_sqrt_two()
    print("✓ test_sqrt_two")

    test_sin_zero()
    print("✓ test_sin_zero")

    test_sin_pi_over_2()
    print("✓ test_sin_pi_over_2")

    test_sin_pi()
    print("✓ test_sin_pi")

    test_cos_zero()
    print("✓ test_cos_zero")

    test_cos_pi_over_2()
    print("✓ test_cos_pi_over_2")

    test_cos_pi()
    print("✓ test_cos_pi")

    test_tanh_zero()
    print("✓ test_tanh_zero")

    test_tanh_large_positive()
    print("✓ test_tanh_large_positive")

    test_tanh_large_negative()
    print("✓ test_tanh_large_negative")

    test_tanh_small_values()
    print("✓ test_tanh_small_values")

    test_clip_basic()
    print("✓ test_clip_basic")

    test_clip_all_below()
    print("✓ test_clip_all_below")

    test_clip_all_above()
    print("✓ test_clip_all_above")

    test_operations_preserve_dtype()
    print("✓ test_operations_preserve_dtype")

    print("\nAll test_elementwise_forward tests passed!")
