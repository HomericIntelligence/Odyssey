"""Tests for modulo edge cases and power operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- modulo: negative dividend, fractional values
- power: shapes, values, integer exponent, zero exponent, one exponent, negative base
"""

from tests.shared.conftest import (
    assert_all_values,
    assert_almost_equal,
    assert_equal_int,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import (
    modulo,
    power,
)


fn test_modulo_negative_dividend() raises:
    """Test modulo with negative dividend."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = modulo(a, b)

    # Python semantics: -7 % 3 = 2 (not -1)
    assert_all_values(
        c, 2.0, 1e-6, "-7.0 % 3.0 should be 2.0 (Python semantics)"
    )


fn test_modulo_fractional() raises:
    """Test modulo with fractional values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.5, DType.float32)
    var b = full(shape, 2.5, DType.float32)
    var c = modulo(a, b)

    assert_all_values(c, 0.0, 1e-6, "7.5 % 2.5 should be 0.0")


fn test_power_shapes() raises:
    """Test that power returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = power(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_power_values() raises:
    """Test that power computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    a._data.bitcast[Float32]()[2] = 4.0

    b._data.bitcast[Float32]()[0] = 3.0
    b._data.bitcast[Float32]()[1] = 2.0
    b._data.bitcast[Float32]()[2] = 0.5

    var result = power(a, b)

    # 2^3 = 8, 3^2 = 9, 4^0.5 = 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(9.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_power_integer_exponent() raises:
    """Test power with small integer exponent."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 8.0, 1e-6, "2.0 ** 3.0 should be 8.0")


fn test_power_zero_exponent() raises:
    """Test power with zero exponent."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 0.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 1.0, 1e-6, "x ** 0 should be 1.0")


fn test_power_one_exponent() raises:
    """Test power with exponent of one."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.5, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 7.5, 1e-6, "x ** 1 should be x")


fn test_power_negative_base() raises:
    """Test power with negative base."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 4.0, 1e-6, "(-2.0) ** 2.0 should be 4.0")


fn main() raises:
    """Run modulo edge cases and power tests."""
    print("Running modulo edge cases and power tests (part 6)...")

    test_modulo_negative_dividend()
    print("    ✓ test_modulo_negative_dividend")
    test_modulo_fractional()
    print("    ✓ test_modulo_fractional")
    test_power_shapes()
    print("    ✓ test_power_shapes")
    test_power_values()
    print("    ✓ test_power_values")
    test_power_integer_exponent()
    print("    ✓ test_power_integer_exponent")
    test_power_zero_exponent()
    print("    ✓ test_power_zero_exponent")
    test_power_one_exponent()
    print("    ✓ test_power_one_exponent")
    test_power_negative_base()
    print("    ✓ test_power_negative_base")

    print("\nAll arithmetic part 6 tests passed! (8 tests)")
