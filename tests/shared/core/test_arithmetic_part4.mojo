"""Tests for arithmetic division operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Basic divide operation: shapes, values
- Same-shape division
- Edge cases: divide by one, divide by two, negative values
- Backward pass for divide
"""

from tests.shared.conftest import (
    assert_all_values,
    assert_almost_equal,
    assert_equal_int,
    assert_numel,
    assert_dtype,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import (
    divide,
    divide_backward,
)


fn test_divide_shapes() raises:
    """Test that divide returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = divide(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_divide_values() raises:
    """Test that divide computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 10.0
    a._data.bitcast[Float32]()[1] = 20.0
    a._data.bitcast[Float32]()[2] = 30.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 4.0
    b._data.bitcast[Float32]()[2] = 5.0

    var result = divide(a, b)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(6.0), tolerance=1e-5
    )


fn test_divide_same_shape() raises:
    """Test dividing two tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = divide(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 3.0, 1e-6, "6.0 / 2.0 should be 3.0")


fn test_divide_by_one() raises:
    """Test dividing by one (identity)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var b = ones(shape, DType.float32)
    var c = divide(a, b)

    assert_all_values(c, 7.5, 1e-6, "x / 1 should be x")


fn test_divide_by_two() raises:
    """Test dividing by two."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 10.0, DType.float64)
    var b = full(shape, 2.0, DType.float64)
    var c = divide(a, b)

    assert_dtype(c, DType.float64, "Should preserve float64")
    assert_all_values(c, 5.0, 1e-8, "10.0 / 2.0 should be 5.0")


fn test_divide_negative() raises:
    """Test dividing negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = divide(a, b)

    assert_all_values(c, -3.0, 1e-6, "-6.0 / 2.0 should be -3.0")


fn test_divide_backward() raises:
    """Test divide backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 10.0
    a._data.bitcast[Float32]()[1] = 20.0
    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 4.0

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # Gradient of divide: d/da = 1/b, d/db = -a/b^2
    # d/da[0] = 1/2 = 0.5
    # d/da[1] = 1/4 = 0.25
    # d/db[0] = -10/4 = -2.5
    # d/db[1] = -20/16 = -1.25

    assert_almost_equal(
        grad_a._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        grad_a._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-2.5), tolerance=1e-4
    )
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[1], Float32(-1.25), tolerance=1e-4
    )


fn main() raises:
    """Run division arithmetic tests."""
    print("Running arithmetic division tests (part 4)...")

    test_divide_shapes()
    print("    ✓ test_divide_shapes")
    test_divide_values()
    print("    ✓ test_divide_values")
    test_divide_same_shape()
    print("    ✓ test_divide_same_shape")
    test_divide_by_one()
    print("    ✓ test_divide_by_one")
    test_divide_by_two()
    print("    ✓ test_divide_by_two")
    test_divide_negative()
    print("    ✓ test_divide_negative")
    test_divide_backward()
    print("    ✓ test_divide_backward")

    print("\nAll arithmetic part 4 tests passed! (7 tests)")
