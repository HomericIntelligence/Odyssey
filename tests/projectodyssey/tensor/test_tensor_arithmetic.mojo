"""Tests for AnyTensor arithmetic operations.

Tests cover:
- add: Element-wise addition via AnyTensor
- subtract: Element-wise subtraction via AnyTensor
- multiply: Element-wise multiplication via AnyTensor
- divide: Element-wise division via AnyTensor
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    ones as any_ones,
    full as any_full,
)
from projectodyssey.core.arithmetic import add, subtract, multiply, divide


def test_add() raises:
    """Add preserves dtype and computes correct values."""
    var a = any_full([2, 2], 0.5, DType.float32)
    var b = any_full([2, 2], 1.0, DType.float32)
    var c = add(a, b)
    assert_true(c.dtype() == DType.float32, "dtype should be float32")
    assert_true(c.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 1.5, atol=1e-6, msg="0.5 + 1.0 = 1.5"
        )
    print("PASS: test_add")


def test_add_zeros() raises:
    """Add with zeros is identity."""
    var a = any_full([3], 1.5, DType.float32)
    var b = any_full([3], 0.0, DType.float32)
    var c = add(a, b)
    for i in range(3):
        assert_almost_equal(Float64(c[i]), 1.5, atol=1e-6, msg="x + 0 = x")
    print("PASS: test_add_zeros")


def test_subtract() raises:
    """Subtract computes correct values."""
    var a = any_full([2, 2], 1.5, DType.float32)
    var b = any_full([2, 2], 0.5, DType.float32)
    var c = subtract(a, b)
    assert_true(c.dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 1.0, atol=1e-6, msg="1.5 - 0.5 = 1.0"
        )
    print("PASS: test_subtract")


def test_multiply() raises:
    """Multiply computes correct values."""
    var a = any_full([2, 2], 0.5, DType.float32)
    var b = any_full([2, 2], 1.5, DType.float32)
    var c = multiply(a, b)
    assert_true(c.dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 0.75, atol=1e-6, msg="0.5 * 1.5 = 0.75"
        )
    print("PASS: test_multiply")


def test_multiply_ones() raises:
    """Multiply with ones is identity."""
    var a = any_full([3], 0.25, DType.float32)
    var b = any_ones([3], DType.float32)
    var c = multiply(a, b)
    for i in range(3):
        assert_almost_equal(Float64(c[i]), 0.25, atol=1e-6, msg="x * 1 = x")
    print("PASS: test_multiply_ones")


def test_divide() raises:
    """Divide computes correct values."""
    var a = any_full([2, 2], 1.5, DType.float32)
    var b = any_full([2, 2], 0.5, DType.float32)
    var c = divide(a, b)
    assert_true(c.dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 3.0, atol=1e-6, msg="1.5 / 0.5 = 3.0"
        )
    print("PASS: test_divide")


def test_divide_by_ones() raises:
    """Divide by ones is identity."""
    var a = any_full([3], 0.25, DType.float32)
    var b = any_ones([3], DType.float32)
    var c = divide(a, b)
    for i in range(3):
        assert_almost_equal(Float64(c[i]), 0.25, atol=1e-6, msg="x / 1 = x")
    print("PASS: test_divide_by_ones")


def test_arithmetic_float64() raises:
    """Arithmetic operations work with float64 dtype."""
    var a = any_full([2], 1.0, DType.float64)
    var b = any_full([2], 0.5, DType.float64)
    var c = add(a, b)
    assert_true(c.dtype() == DType.float64, "dtype should be float64")
    for i in range(2):
        assert_almost_equal(
            Float64(c[i]), 1.5, atol=1e-10, msg="1.0 + 0.5 = 1.5"
        )
    print("PASS: test_arithmetic_float64")


def main() raises:
    test_add()
    test_add_zeros()
    test_subtract()
    test_multiply()
    test_multiply_ones()
    test_divide()
    test_divide_by_ones()
    test_arithmetic_float64()
    print("All test_tensor_arithmetic tests passed!")
