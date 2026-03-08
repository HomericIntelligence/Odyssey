"""Tests for arithmetic multiplication operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Basic multiply operation: shapes, values
- Same-shape multiplication (1D and 2D)
- Edge cases: by zero, by one, negative values
- Backward pass for multiply
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
    multiply,
    multiply_backward,
)


fn test_multiply_shapes() raises:
    """Test that multiply returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = multiply(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_multiply_values() raises:
    """Test that multiply computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    a._data.bitcast[Float32]()[2] = 4.0

    b._data.bitcast[Float32]()[0] = 5.0
    b._data.bitcast[Float32]()[1] = 6.0
    b._data.bitcast[Float32]()[2] = 7.0

    var result = multiply(a, b)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(10.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(18.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(28.0), tolerance=1e-5
    )


fn test_multiply_same_shape_1d() raises:
    """Test multiplying two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 2.5, DType.float32)
    var c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 10.0, 1e-6, "4.0 * 2.5 should be 10.0")


fn test_multiply_same_shape_2d() raises:
    """Test multiplying two 2D tensors with same shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 3.0, DType.float64)
    var b = full(shape, 1.5, DType.float64)
    var c = multiply(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 4.5, 1e-8, "3.0 * 1.5 should be 4.5")


fn test_multiply_by_zero() raises:
    """Test multiplying by zero (should give all zeros)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 99.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, 0.0, 1e-8, "x * 0 should equal 0")


fn test_multiply_by_one() raises:
    """Test multiplying by one (should not change values)."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 7.5, DType.float32)
    var b = ones(shape, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, 7.5, 1e-6, "x * 1 should equal x")


fn test_multiply_negative() raises:
    """Test multiplying with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, -6.0, 1e-6, "-3.0 * 2.0 should be -6.0")


fn test_multiply_backward() raises:
    """Test multiply backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # Gradient of multiply: d/da = b, d/db = a
    assert_almost_equal(
        grad_a._data.bitcast[Float32]()[0], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_a._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[1], Float32(3.0), tolerance=1e-5
    )


fn main() raises:
    """Run multiplication arithmetic tests."""
    print("Running arithmetic multiplication tests (part 3)...")

    test_multiply_shapes()
    print("    ✓ test_multiply_shapes")
    test_multiply_values()
    print("    ✓ test_multiply_values")
    test_multiply_same_shape_1d()
    print("    ✓ test_multiply_same_shape_1d")
    test_multiply_same_shape_2d()
    print("    ✓ test_multiply_same_shape_2d")
    test_multiply_by_zero()
    print("    ✓ test_multiply_by_zero")
    test_multiply_by_one()
    print("    ✓ test_multiply_by_one")
    test_multiply_negative()
    print("    ✓ test_multiply_negative")
    test_multiply_backward()
    print("    ✓ test_multiply_backward")

    print("\nAll arithmetic part 3 tests passed! (8 tests)")
