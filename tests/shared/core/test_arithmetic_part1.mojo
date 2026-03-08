"""Tests for arithmetic addition operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Basic add operation: shapes, values
- Same-shape addition (1D and 2D)
- Edge cases: zeros, negative values
- Backward pass for add
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
    add,
    add_backward,
)


fn test_add_shapes() raises:
    """Test that add returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = add(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


fn test_add_values() raises:
    """Test that add computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0

    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = add(a, b)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(7.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(9.0), tolerance=1e-5
    )


fn test_add_same_shape_1d() raises:
    """Test adding two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = add(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 5.0, 1e-6, "2.0 + 3.0 should be 5.0")


fn test_add_same_shape_2d() raises:
    """Test adding two 2D tensors with same shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float64)
    var b = full(shape, 2.5, DType.float64)
    var c = add(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 3.5, 1e-8, "1.0 + 2.5 should be 3.5")


fn test_add_zeros() raises:
    """Test adding zeros (should not change values)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 7.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = add(a, b)

    assert_all_values(c, 7.0, 1e-6, "x + 0 should equal x")


fn test_add_negative_values() raises:
    """Test adding negative values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, -5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = add(a, b)

    assert_all_values(c, -2.0, 1e-6, "-5.0 + 3.0 should be -2.0")


fn test_add_backward() raises:
    """Test add backward pass."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # Gradient of add is just pass-through
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


fn main() raises:
    """Run addition arithmetic tests."""
    print("Running arithmetic addition tests (part 1)...")

    test_add_shapes()
    print("    ✓ test_add_shapes")
    test_add_values()
    print("    ✓ test_add_values")
    test_add_same_shape_1d()
    print("    ✓ test_add_same_shape_1d")
    test_add_same_shape_2d()
    print("    ✓ test_add_same_shape_2d")
    test_add_zeros()
    print("    ✓ test_add_zeros")
    test_add_negative_values()
    print("    ✓ test_add_negative_values")
    test_add_backward()
    print("    ✓ test_add_backward")

    print("\nAll arithmetic part 1 tests passed! (7 tests)")
