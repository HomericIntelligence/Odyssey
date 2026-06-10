"""Tests for AnyTensor arithmetic operations and precision.

Tests cover:
- AnyTensor add/subtract/multiply/divide correctness
- Float64 precision preservation
- Shape preservation in arithmetic
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    zeros as any_zeros,
    ones as any_ones,
    full as any_full,
)
from projectodyssey.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
)


def test_add_correctness() raises:
    """AnyTensor add produces correct element-wise results."""
    var a = any_full([3, 4], 0.5, DType.float32)
    var b = any_full([3, 4], 1.0, DType.float32)
    var result = add(a, b)

    assert_true(result.numel() == 12, "numel should be 12")
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            1.5,
            atol=1e-7,
            msg="0.5 + 1.0 = 1.5",
        )
    print("PASS: test_add_correctness")


def test_subtract_correctness() raises:
    """AnyTensor subtract produces correct element-wise results."""
    var a = any_full([2, 3], 1.5, DType.float32)
    var b = any_full([2, 3], 0.5, DType.float32)
    var result = subtract(a, b)

    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            1.0,
            atol=1e-7,
            msg="1.5 - 0.5 = 1.0",
        )
    print("PASS: test_subtract_correctness")


def test_multiply_correctness() raises:
    """AnyTensor multiply produces correct element-wise results."""
    var a = any_full([2, 3], 0.5, DType.float32)
    var b = any_full([2, 3], 1.5, DType.float32)
    var result = multiply(a, b)

    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            0.75,
            atol=1e-7,
            msg="0.5 * 1.5 = 0.75",
        )
    print("PASS: test_multiply_correctness")


def test_divide_correctness() raises:
    """AnyTensor divide produces correct element-wise results."""
    var a = any_full([2, 3], 1.5, DType.float32)
    var b = any_full([2, 3], 0.5, DType.float32)
    var result = divide(a, b)

    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            3.0,
            atol=1e-7,
            msg="1.5 / 0.5 = 3.0",
        )
    print("PASS: test_divide_correctness")


def test_add_float64_precision() raises:
    """Float64 add preserves full precision."""
    var a = any_full([2], 3.141592653589793, DType.float64)
    var b = any_full([2], 1.0e-15, DType.float64)
    var result = add(a, b)
    assert_almost_equal(
        Float64(result[0]),
        3.141592653589793 + 1.0e-15,
        atol=1e-18,
        msg="float64 add should preserve full precision",
    )
    print("PASS: test_add_float64_precision")


def test_subtract_float64_precision() raises:
    """Float64 subtract preserves full precision."""
    var a = any_full([2], 3.141592653589793, DType.float64)
    var b = any_full([2], 1.0e-15, DType.float64)
    var result = subtract(a, b)
    assert_almost_equal(
        Float64(result[0]),
        3.141592653589793 - 1.0e-15,
        atol=1e-18,
        msg="float64 subtract should preserve full precision",
    )
    print("PASS: test_subtract_float64_precision")


def test_multiply_divide_inverse() raises:
    """Multiply then divide returns original values."""
    var a = any_full([4], 1.5, DType.float32)
    var b = any_full([4], 0.5, DType.float32)
    var product = multiply(a, b)
    var result = divide(product, b)
    for i in range(4):
        assert_almost_equal(
            Float64(result[i]),
            1.5,
            atol=1e-6,
            msg="(a * b) / b should return a",
        )
    print("PASS: test_multiply_divide_inverse")


def test_add_negative_values() raises:
    """Add handles negative values correctly."""
    var a = any_full([3], -1.0, DType.float32)
    var b = any_full([3], -0.5, DType.float32)
    var result = add(a, b)
    for i in range(3):
        assert_almost_equal(
            Float64(result[i]),
            -1.5,
            atol=1e-6,
            msg="-1.0 + (-0.5) = -1.5",
        )
    print("PASS: test_add_negative_values")


def test_arithmetic_preserves_shape() raises:
    """Arithmetic preserves tensor shape."""
    var a = any_ones([2, 3, 4], DType.float32)
    var b = any_ones([2, 3, 4], DType.float32)
    var result = add(a, b)
    var shape = result.shape()
    assert_true(len(shape) == 3, "result should be 3D")
    assert_true(shape[0] == 2, "dim 0 should be 2")
    assert_true(shape[1] == 3, "dim 1 should be 3")
    assert_true(shape[2] == 4, "dim 2 should be 4")
    print("PASS: test_arithmetic_preserves_shape")


def main() raises:
    test_add_correctness()
    test_subtract_correctness()
    test_multiply_correctness()
    test_divide_correctness()
    test_add_float64_precision()
    test_subtract_float64_precision()
    test_multiply_divide_inverse()
    test_add_negative_values()
    test_arithmetic_preserves_shape()
    print("All test_tensor_arithmetic_native tests passed!")
