"""Tests for arithmetic addition operations.

Tests cover:
- Basic add operation: shapes, values
- Same-shape addition (1D and 2D)
- Edge cases: zeros, negative values
- Backward pass for add
"""


from tests.projectodyssey.conftest import (
    assert_all_values,
    assert_almost_equal,
    assert_dim,
    assert_dtype,
    assert_equal_int,
    assert_numel,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full, ones, zeros
from projectodyssey.core.arithmetic import (
    add,
    add_backward,
    divide,
    divide_backward,
    floor_divide,
    modulo,
    multiply,
    multiply_backward,
    power,
    subtract,
    subtract_backward,
)


def test_add_shapes() raises:
    """Test that add returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = add(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_add_values() raises:
    """Test that add computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(1.0))
    a.set(1, Float32(2.0))
    a.set(2, Float32(3.0))

    b.set(0, Float32(4.0))
    b.set(1, Float32(5.0))
    b.set(2, Float32(6.0))

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


def test_add_same_shape_1d() raises:
    """Test adding two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = add(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 5.0, 1e-6, "2.0 + 3.0 should be 5.0")


def test_add_same_shape_2d() raises:
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


def test_add_zeros() raises:
    """Test adding zeros (should not change values)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 7.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = add(a, b)

    assert_all_values(c, 7.0, 1e-6, "x + 0 should equal x")


def test_add_negative_values() raises:
    """Test adding negative values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, -5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = add(a, b)

    assert_all_values(c, -2.0, 1e-6, "-5.0 + 3.0 should be -2.0")


def test_add_backward() raises:
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


def test_subtract_shapes() raises:
    """Test that subtract returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = subtract(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_subtract_values() raises:
    """Test that subtract computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(5.0))
    a.set(1, Float32(7.0))
    a.set(2, Float32(9.0))

    b.set(0, Float32(2.0))
    b.set(1, Float32(3.0))
    b.set(2, Float32(4.0))

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


def test_subtract_same_shape_1d() raises:
    """Test subtracting two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = subtract(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 4.0, 1e-6, "7.0 - 3.0 should be 4.0")


def test_subtract_same_shape_2d() raises:
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


def test_subtract_zeros() raises:
    """Test subtracting zeros (should not change values)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 9.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = subtract(a, b)

    assert_all_values(c, 9.0, 1e-6, "x - 0 should equal x")


def test_subtract_negative_result() raises:
    """Test subtraction resulting in negative values."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)
    var c = subtract(a, b)

    assert_all_values(c, -2.0, 1e-6, "3.0 - 5.0 should be -2.0")


def test_subtract_backward() raises:
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


def test_multiply_shapes() raises:
    """Test that multiply returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = multiply(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_multiply_values() raises:
    """Test that multiply computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(2.0))
    a.set(1, Float32(3.0))
    a.set(2, Float32(4.0))

    b.set(0, Float32(5.0))
    b.set(1, Float32(6.0))
    b.set(2, Float32(7.0))

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


def test_multiply_same_shape_1d() raises:
    """Test multiplying two 1D tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 2.5, DType.float32)
    var c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 10.0, 1e-6, "4.0 * 2.5 should be 10.0")


def test_multiply_same_shape_2d() raises:
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


def test_multiply_by_zero() raises:
    """Test multiplying by zero (should give all zeros)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 99.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, 0.0, 1e-8, "x * 0 should equal 0")


def test_multiply_by_one() raises:
    """Test multiplying by one (should not change values)."""
    var shape = List[Int]()
    shape.append(10)
    var a = full(shape, 7.5, DType.float32)
    var b = ones(shape, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, 7.5, 1e-6, "x * 1 should equal x")


def test_multiply_negative() raises:
    """Test multiplying with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = multiply(a, b)

    assert_all_values(c, -6.0, 1e-6, "-3.0 * 2.0 should be -6.0")


def test_multiply_backward() raises:
    """Test multiply backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a.set(0, Float32(2.0))
    a.set(1, Float32(3.0))
    b.set(0, Float32(4.0))
    b.set(1, Float32(5.0))

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


def test_divide_shapes() raises:
    """Test that divide returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = divide(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_divide_values() raises:
    """Test that divide computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(10.0))
    a.set(1, Float32(20.0))
    a.set(2, Float32(30.0))

    b.set(0, Float32(2.0))
    b.set(1, Float32(4.0))
    b.set(2, Float32(5.0))

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


def test_divide_same_shape() raises:
    """Test dividing two tensors with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = divide(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 3.0, 1e-6, "6.0 / 2.0 should be 3.0")


def test_divide_by_one() raises:
    """Test dividing by one (identity)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var b = ones(shape, DType.float32)
    var c = divide(a, b)

    assert_all_values(c, 7.5, 1e-6, "x / 1 should be x")


def test_divide_by_two() raises:
    """Test dividing by two."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 10.0, DType.float64)
    var b = full(shape, 2.0, DType.float64)
    var c = divide(a, b)

    assert_dtype(c, DType.float64, "Should preserve float64")
    assert_all_values(c, 5.0, 1e-8, "10.0 / 2.0 should be 5.0")


def test_divide_negative() raises:
    """Test dividing negative values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = divide(a, b)

    assert_all_values(c, -3.0, 1e-6, "-6.0 / 2.0 should be -3.0")


def test_divide_backward() raises:
    """Test divide backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a.set(0, Float32(10.0))
    a.set(1, Float32(20.0))
    b.set(0, Float32(2.0))
    b.set(1, Float32(4.0))

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


def test_floor_divide_shapes() raises:
    """Test that floor_divide returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = floor_divide(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_floor_divide_values() raises:
    """Test that floor_divide computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(7.0))
    a.set(1, Float32(8.0))
    a.set(2, Float32(9.0))

    b.set(0, Float32(2.0))
    b.set(1, Float32(3.0))
    b.set(2, Float32(4.0))

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


def test_floor_divide_same_shape() raises:
    """Test floor division with same shape."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, 3.0, 1e-6, "7.0 // 2.0 should be 3.0")


def test_floor_divide_positive() raises:
    """Test floor division with positive values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 9.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, 2.0, 1e-6, "9.0 // 4.0 should be 2.0")


def test_floor_divide_negative() raises:
    """Test floor division with negative dividend."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = floor_divide(a, b)

    assert_all_values(c, -4.0, 1e-6, "-7.0 // 2.0 should be -4.0")


def test_modulo_shapes() raises:
    """Test that modulo returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = modulo(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_modulo_values() raises:
    """Test that modulo computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(7.0))
    a.set(1, Float32(8.0))
    a.set(2, Float32(9.0))

    b.set(0, Float32(3.0))
    b.set(1, Float32(5.0))
    b.set(2, Float32(4.0))

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


def test_modulo_positive() raises:
    """Test modulo with positive values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = modulo(a, b)

    assert_all_values(c, 1.0, 1e-6, "7.0 % 3.0 should be 1.0")


def test_modulo_negative_dividend() raises:
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


def test_modulo_fractional() raises:
    """Test modulo with fractional values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.5, DType.float32)
    var b = full(shape, 2.5, DType.float32)
    var c = modulo(a, b)

    assert_all_values(c, 0.0, 1e-6, "7.5 % 2.5 should be 0.0")


def test_power_shapes() raises:
    """Test that power returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = power(a, b)

    assert_equal_int(result.shape()[0], 4)
    assert_equal_int(result.shape()[1], 10)


def test_power_values() raises:
    """Test that power computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(2.0))
    a.set(1, Float32(3.0))
    a.set(2, Float32(4.0))

    b.set(0, Float32(3.0))
    b.set(1, Float32(2.0))
    b.set(2, Float32(0.5))

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


def test_power_integer_exponent() raises:
    """Test power with small integer exponent."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 8.0, 1e-6, "2.0 ** 3.0 should be 8.0")


def test_power_zero_exponent() raises:
    """Test power with zero exponent."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 0.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 1.0, 1e-6, "x ** 0 should be 1.0")


def test_power_one_exponent() raises:
    """Test power with exponent of one."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.5, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 7.5, 1e-6, "x ** 1 should be x")


def test_power_negative_base() raises:
    """Test power with negative base."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = power(a, b)

    assert_all_values(c, 4.0, 1e-6, "(-2.0) ** 2.0 should be 4.0")


def test_dunder_add() raises:
    """Test __add__ operator overloading (a + b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a + b

    assert_all_values(c, 5.0, 1e-6, "a + b should work via __add__")


def test_dunder_sub() raises:
    """Test __sub__ operator overloading (a - b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a - b

    assert_all_values(c, 4.0, 1e-6, "a - b should work via __sub__")


def test_dunder_mul() raises:
    """Test __mul__ operator overloading (a * b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a * b

    assert_all_values(c, 8.0, 1e-6, "a * b should work via __mul__")


def test_chained_operations() raises:
    """Test chained operations with multiple operators."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = full(shape, 1.0, DType.float32)

    # (a + b) * c = (2 + 3) * 1 = 5
    var result = (a + b) * c
    assert_all_values(result, 5.0, 1e-6, "(2 + 3) * 1 should be 5")


def test_complex_expression() raises:
    """Test complex expression with multiple operations."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    var result = a + b * c
    assert_all_values(result, 7.0, 1e-6, "1 + 2 * 3 should be 7")


def test_add_preserves_dtype_float32() raises:
    """Test that add preserves float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dtype(c, DType.float32, "Result should preserve float32 dtype")


def test_add_preserves_dtype_float64() raises:
    """Test that add preserves float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = add(a, b)

    assert_dtype(c, DType.float64, "Result should preserve float64 dtype")


def test_multiply_preserves_dtype_int32() raises:
    """Test that multiply preserves int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.int32)
    var b = full(shape, 2.0, DType.int32)
    var c = multiply(a, b)

    assert_dtype(c, DType.int32, "Result should preserve int32 dtype")


def test_add_preserves_shape_1d() raises:
    """Test that add preserves 1D shape."""
    var shape = List[Int]()
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dim(c, 1, "Result should be 1D")
    assert_numel(c, 10, "Result should have 10 elements")


def test_add_preserves_shape_3d() raises:
    """Test that add preserves 3D shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 24, "Result should have 24 elements")


def test_add_mismatched_shapes_raises_error() raises:
    """Test that add with mismatched shapes raises error."""
    var shape_a = List[Int]()
    shape_a.append(5)
    var shape_b = List[Int]()
    shape_b.append(10)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = add(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions shape/broadcast incompatibility
        if (
            "broadcast" not in error_msg.lower()
            and "shape" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention shape or broadcast compatibility"
            )

    if not error_raised:
        raise Error(
            "add with mismatched non-broadcastable shapes should raise error"
        )


def test_multiply_mismatched_shapes_raises_error() raises:
    """Test that multiply with mismatched shapes raises error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = multiply(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions shape/broadcast incompatibility
        if (
            "broadcast" not in error_msg.lower()
            and "shape" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention shape or broadcast compatibility"
            )

    if not error_raised:
        raise Error(
            "multiply with mismatched non-broadcastable shapes should raise"
            " error"
        )


def test_add_mismatched_dtypes_raises_error() raises:
    """Test that add with mismatched dtypes raises error."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = add(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions dtype mismatch
        if "dtype" not in error_msg.lower():
            raise Error("Error message should mention dtype mismatch")

    if not error_raised:
        raise Error("add with mismatched dtypes should raise error")


def main() raises:
    """Run all test_arithmetic tests."""
    print("Running test_arithmetic tests...")

    test_add_shapes()
    print("✓ test_add_shapes")

    test_add_values()
    print("✓ test_add_values")

    test_add_same_shape_1d()
    print("✓ test_add_same_shape_1d")

    test_add_same_shape_2d()
    print("✓ test_add_same_shape_2d")

    test_add_zeros()
    print("✓ test_add_zeros")

    test_add_negative_values()
    print("✓ test_add_negative_values")

    test_add_backward()
    print("✓ test_add_backward")

    test_subtract_shapes()
    print("✓ test_subtract_shapes")

    test_subtract_values()
    print("✓ test_subtract_values")

    test_subtract_same_shape_1d()
    print("✓ test_subtract_same_shape_1d")

    test_subtract_same_shape_2d()
    print("✓ test_subtract_same_shape_2d")

    test_subtract_zeros()
    print("✓ test_subtract_zeros")

    test_subtract_negative_result()
    print("✓ test_subtract_negative_result")

    test_subtract_backward()
    print("✓ test_subtract_backward")

    test_multiply_shapes()
    print("✓ test_multiply_shapes")

    test_multiply_values()
    print("✓ test_multiply_values")

    test_multiply_same_shape_1d()
    print("✓ test_multiply_same_shape_1d")

    test_multiply_same_shape_2d()
    print("✓ test_multiply_same_shape_2d")

    test_multiply_by_zero()
    print("✓ test_multiply_by_zero")

    test_multiply_by_one()
    print("✓ test_multiply_by_one")

    test_multiply_negative()
    print("✓ test_multiply_negative")

    test_multiply_backward()
    print("✓ test_multiply_backward")

    test_divide_shapes()
    print("✓ test_divide_shapes")

    test_divide_values()
    print("✓ test_divide_values")

    test_divide_same_shape()
    print("✓ test_divide_same_shape")

    test_divide_by_one()
    print("✓ test_divide_by_one")

    test_divide_by_two()
    print("✓ test_divide_by_two")

    test_divide_negative()
    print("✓ test_divide_negative")

    test_divide_backward()
    print("✓ test_divide_backward")

    test_floor_divide_shapes()
    print("✓ test_floor_divide_shapes")

    test_floor_divide_values()
    print("✓ test_floor_divide_values")

    test_floor_divide_same_shape()
    print("✓ test_floor_divide_same_shape")

    test_floor_divide_positive()
    print("✓ test_floor_divide_positive")

    test_floor_divide_negative()
    print("✓ test_floor_divide_negative")

    test_modulo_shapes()
    print("✓ test_modulo_shapes")

    test_modulo_values()
    print("✓ test_modulo_values")

    test_modulo_positive()
    print("✓ test_modulo_positive")

    test_modulo_negative_dividend()
    print("✓ test_modulo_negative_dividend")

    test_modulo_fractional()
    print("✓ test_modulo_fractional")

    test_power_shapes()
    print("✓ test_power_shapes")

    test_power_values()
    print("✓ test_power_values")

    test_power_integer_exponent()
    print("✓ test_power_integer_exponent")

    test_power_zero_exponent()
    print("✓ test_power_zero_exponent")

    test_power_one_exponent()
    print("✓ test_power_one_exponent")

    test_power_negative_base()
    print("✓ test_power_negative_base")

    test_dunder_add()
    print("✓ test_dunder_add")

    test_dunder_sub()
    print("✓ test_dunder_sub")

    test_dunder_mul()
    print("✓ test_dunder_mul")

    test_chained_operations()
    print("✓ test_chained_operations")

    test_complex_expression()
    print("✓ test_complex_expression")

    test_add_preserves_dtype_float32()
    print("✓ test_add_preserves_dtype_float32")

    test_add_preserves_dtype_float64()
    print("✓ test_add_preserves_dtype_float64")

    test_multiply_preserves_dtype_int32()
    print("✓ test_multiply_preserves_dtype_int32")

    test_add_preserves_shape_1d()
    print("✓ test_add_preserves_shape_1d")

    test_add_preserves_shape_3d()
    print("✓ test_add_preserves_shape_3d")

    test_add_mismatched_shapes_raises_error()
    print("✓ test_add_mismatched_shapes_raises_error")

    test_multiply_mismatched_shapes_raises_error()
    print("✓ test_multiply_mismatched_shapes_raises_error")

    test_add_mismatched_dtypes_raises_error()
    print("✓ test_add_mismatched_dtypes_raises_error")

    print("\nAll test_arithmetic tests passed!")
