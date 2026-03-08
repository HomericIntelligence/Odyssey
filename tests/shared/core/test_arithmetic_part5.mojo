"""Tests for floor division and modulo operations (shapes, values, basic cases).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- floor_divide: shapes, values, same shape, positive, negative
- modulo: shapes, values, positive
"""

from tests.shared.conftest import (
    assert_all_values,
    assert_almost_equal,
    assert_equal_int,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import (
    floor_divide,
    modulo,
)


fn test_floor_divide_shapes() raises:
    """Test that floor_divide returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = floor_divide(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_floor_divide_values() raises:
    """Test that floor_divide computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 7.0
    a._data.bitcast[Float32]()[1] = 8.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = floor_divide(a, b)

    # 7 // 2 = 3, 8 // 3 = 2, 9 // 4 = 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_floor_divide_same_shape() raises:
    """Test floor division with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, 3.0, 1e-6, "7.0 // 2.0 should be 3.0")


fn test_floor_divide_positive() raises:
    """Test floor division with positive values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 9.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, 2.0, 1e-6, "9.0 // 4.0 should be 2.0")


fn test_floor_divide_negative() raises:
    """Test floor division with negative dividend."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, -4.0, 1e-6, "-7.0 // 2.0 should be -4.0")


fn test_modulo_shapes() raises:
    """Test that modulo returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = modulo(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_modulo_values() raises:
    """Test that modulo computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 7.0
    a._data.bitcast[Float32]()[1] = 8.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 3.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = modulo(a, b)

    # 7 % 3 = 1, 8 % 5 = 3, 9 % 4 = 1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


fn test_modulo_positive() raises:
    """Test modulo with positive values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = modulo(a, b)

    assert_all_values(c, 1.0, 1e-6, "7.0 % 3.0 should be 1.0")


fn main() raises:
    """Run floor division and modulo (basic) tests."""
    print("Running floor divide and modulo tests (part 5)...")

    test_floor_divide_shapes()
    print("    ✓ test_floor_divide_shapes")
    test_floor_divide_values()
    print("    ✓ test_floor_divide_values")
    test_floor_divide_same_shape()
    print("    ✓ test_floor_divide_same_shape")
    test_floor_divide_positive()
    print("    ✓ test_floor_divide_positive")
    test_floor_divide_negative()
    print("    ✓ test_floor_divide_negative")
    test_modulo_shapes()
    print("    ✓ test_modulo_shapes")
    test_modulo_values()
    print("    ✓ test_modulo_values")
    test_modulo_positive()
    print("    ✓ test_modulo_positive")

    print("\nAll arithmetic part 5 tests passed! (8 tests)")
