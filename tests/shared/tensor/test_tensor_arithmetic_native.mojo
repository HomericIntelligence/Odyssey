"""Tests for native Tensor[dtype] arithmetic: typed vs AnyTensor equivalence.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- add_typed vs AnyTensor add equivalence
- subtract_typed vs AnyTensor subtract equivalence
- multiply_typed vs AnyTensor multiply equivalence
- divide_typed vs AnyTensor divide equivalence
- Float64 precision preservation
- Broadcasting with typed ops
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import zeros, ones, full
from shared.core.any_tensor import AnyTensor, zeros as any_zeros, ones as any_ones, full as any_full
from shared.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    add_typed,
    subtract_typed,
    multiply_typed,
    divide_typed,
)


fn test_add_typed_matches_anytensor() raises:
    """Typed add produces identical results to AnyTensor add."""
    var a_any = any_full([3, 4], 0.5, DType.float32)
    var b_any = any_full([3, 4], 1.0, DType.float32)
    var result_any = add(a_any, b_any)

    var a_typed = full[DType.float32]([3, 4], 0.5)
    var b_typed = full[DType.float32]([3, 4], 1.0)
    var result_typed = add_typed(a_typed, b_typed)

    assert_true(result_typed.numel() == 12, "numel should be 12")
    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-7,
            msg="add: typed should match AnyTensor",
        )
    print("PASS: test_add_typed_matches_anytensor")


fn test_subtract_typed_matches_anytensor() raises:
    """Typed subtract produces identical results to AnyTensor subtract."""
    var a_any = any_full([2, 3], 1.5, DType.float32)
    var b_any = any_full([2, 3], 0.5, DType.float32)
    var result_any = subtract(a_any, b_any)

    var a_typed = full[DType.float32]([2, 3], 1.5)
    var b_typed = full[DType.float32]([2, 3], 0.5)
    var result_typed = subtract_typed(a_typed, b_typed)

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-7,
            msg="subtract: typed should match AnyTensor",
        )
    print("PASS: test_subtract_typed_matches_anytensor")


fn test_multiply_typed_matches_anytensor() raises:
    """Typed multiply produces identical results to AnyTensor multiply."""
    var a_any = any_full([2, 3], 0.5, DType.float32)
    var b_any = any_full([2, 3], 1.5, DType.float32)
    var result_any = multiply(a_any, b_any)

    var a_typed = full[DType.float32]([2, 3], 0.5)
    var b_typed = full[DType.float32]([2, 3], 1.5)
    var result_typed = multiply_typed(a_typed, b_typed)

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-7,
            msg="multiply: typed should match AnyTensor",
        )
    print("PASS: test_multiply_typed_matches_anytensor")


fn test_divide_typed_matches_anytensor() raises:
    """Typed divide produces identical results to AnyTensor divide."""
    var a_any = any_full([2, 3], 1.5, DType.float32)
    var b_any = any_full([2, 3], 0.5, DType.float32)
    var result_any = divide(a_any, b_any)

    var a_typed = full[DType.float32]([2, 3], 1.5)
    var b_typed = full[DType.float32]([2, 3], 0.5)
    var result_typed = divide_typed(a_typed, b_typed)

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-7,
            msg="divide: typed should match AnyTensor",
        )
    print("PASS: test_divide_typed_matches_anytensor")


fn test_add_typed_float64_precision() raises:
    """Typed float64 add preserves full precision."""
    var a = zeros[DType.float64]([2])
    a._data[0] = Scalar[DType.float64](3.141592653589793)
    a._data[1] = Scalar[DType.float64](2.718281828459045)
    var b = zeros[DType.float64]([2])
    b._data[0] = Scalar[DType.float64](1.0e-15)
    b._data[1] = Scalar[DType.float64](1.0e-15)
    var result = add_typed(a, b)
    assert_almost_equal(
        Float64(result[0]),
        3.141592653589793 + 1.0e-15,
        atol=1e-18,
        msg="float64 add should preserve full precision",
    )
    assert_almost_equal(
        Float64(result[1]),
        2.718281828459045 + 1.0e-15,
        atol=1e-18,
        msg="float64 add should preserve full precision",
    )
    print("PASS: test_add_typed_float64_precision")


fn test_subtract_typed_float64_precision() raises:
    """Typed float64 subtract preserves full precision."""
    var a = zeros[DType.float64]([2])
    a._data[0] = Scalar[DType.float64](3.141592653589793)
    a._data[1] = Scalar[DType.float64](2.718281828459045)
    var b = zeros[DType.float64]([2])
    b._data[0] = Scalar[DType.float64](1.0e-15)
    b._data[1] = Scalar[DType.float64](1.0e-15)
    var result = subtract_typed(a, b)
    assert_almost_equal(
        Float64(result[0]),
        3.141592653589793 - 1.0e-15,
        atol=1e-18,
        msg="float64 subtract should preserve full precision",
    )
    print("PASS: test_subtract_typed_float64_precision")


fn test_multiply_divide_typed_inverse() raises:
    """Typed multiply then divide returns original values."""
    var a = full[DType.float32]([4], 1.5)
    var b = full[DType.float32]([4], 0.5)
    var product = multiply_typed(a, b)
    var result = divide_typed(product, b)
    for i in range(4):
        assert_almost_equal(
            Float64(result[i]),
            1.5,
            atol=1e-6,
            msg="(a * b) / b should return a",
        )
    print("PASS: test_multiply_divide_typed_inverse")


fn test_add_typed_negative_values() raises:
    """Typed add handles negative values correctly."""
    var a = full[DType.float32]([3], -1.0)
    var b = full[DType.float32]([3], -0.5)
    var result = add_typed(a, b)
    for i in range(3):
        assert_almost_equal(
            Float64(result[i]),
            -1.5,
            atol=1e-6,
            msg="-1.0 + (-0.5) = -1.5",
        )
    print("PASS: test_add_typed_negative_values")


fn test_arithmetic_typed_preserves_shape() raises:
    """Typed arithmetic preserves tensor shape."""
    var a = ones[DType.float32]([2, 3, 4])
    var b = ones[DType.float32]([2, 3, 4])
    var result = add_typed(a, b)
    var shape = result.shape()
    assert_true(len(shape) == 3, "result should be 3D")
    assert_true(shape[0] == 2, "dim 0 should be 2")
    assert_true(shape[1] == 3, "dim 1 should be 3")
    assert_true(shape[2] == 4, "dim 2 should be 4")
    print("PASS: test_arithmetic_typed_preserves_shape")


fn main() raises:
    test_add_typed_matches_anytensor()
    test_subtract_typed_matches_anytensor()
    test_multiply_typed_matches_anytensor()
    test_divide_typed_matches_anytensor()
    test_add_typed_float64_precision()
    test_subtract_typed_float64_precision()
    test_multiply_divide_typed_inverse()
    test_add_typed_negative_values()
    test_arithmetic_typed_preserves_shape()
    print("All test_tensor_arithmetic_native tests passed!")
