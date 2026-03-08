"""Tests for arithmetic subtraction operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Basic subtract operation: shapes, values
- Same-shape subtraction (1D and 2D)
- Edge cases: zeros, negative results
- Backward pass for subtract
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
    subtract,
    subtract_backward,
)


fn test_subtract_shapes() raises:
    """Test that subtract returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = subtract(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_subtract_values() raises:
    """Test that subtract computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 5.0
    a._data.bitcast[Float32]()[1] = 7.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = subtract(a, b)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-5
    )


fn test_subtract_same_shape_1d() raises:
    """Test subtracting two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = subtract(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 4.0, 1e-6, "7.0 - 3.0 should be 4.0")


fn test_subtract_same_shape_2d() raises:
    """Test subtracting two 2D tensors with same shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 10.0, DType.float64)
    var b = full(shape, 2.5, DType.float64)
    var c = subtract(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 7.5, 1e-8, "10.0 - 2.5 should be 7.5")


fn test_subtract_zeros() raises:
    """Test subtracting zeros (should not change values)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 9.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = subtract(a, b)

    assert_all_values(c, 9.0, 1e-6, "x - 0 should equal x")


fn test_subtract_negative_result() raises:
    """Test subtraction resulting in negative values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)
    var c = subtract(a, b)

    assert_all_values(c, -2.0, 1e-6, "3.0 - 5.0 should be -2.0")


fn test_subtract_backward() raises:
    """Test subtract backward pass."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # Gradient of subtract: d/da = +1, d/db = -1
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-5
        )


fn main() raises:
    """Run subtraction arithmetic tests."""
    print("Running arithmetic subtraction tests (part 2)...")

    test_subtract_shapes()
    print("    ✓ test_subtract_shapes")
    test_subtract_values()
    print("    ✓ test_subtract_values")
    test_subtract_same_shape_1d()
    print("    ✓ test_subtract_same_shape_1d")
    test_subtract_same_shape_2d()
    print("    ✓ test_subtract_same_shape_2d")
    test_subtract_zeros()
    print("    ✓ test_subtract_zeros")
    test_subtract_negative_result()
    print("    ✓ test_subtract_negative_result")
    test_subtract_backward()
    print("    ✓ test_subtract_backward")

    print("\nAll arithmetic part 2 tests passed! (7 tests)")
