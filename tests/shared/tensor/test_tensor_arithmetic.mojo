"""Tests for typed Tensor[dtype] arithmetic operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- add_typed[dt]: Element-wise addition
- subtract_typed[dt]: Element-wise subtraction
- multiply_typed[dt]: Element-wise multiplication
- divide_typed[dt]: Element-wise division
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import ones, full
from shared.core.arithmetic import add_typed, subtract_typed, multiply_typed, divide_typed


fn test_add_typed() raises:
    """add preserves dtype and computes correct values."""
    var a = full[DType.float32]([2, 2], 0.5)
    var b = full[DType.float32]([2, 2], 1.0)
    var c = add_typed(a, b)
    assert_true(c.get_dtype() == DType.float32, "dtype should be float32")
    assert_true(c.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 1.5, atol=1e-6, msg="0.5 + 1.0 = 1.5"
        )
    print("PASS: test_add_typed")


fn test_add_zeros() raises:
    """add with zeros is identity."""
    var a = full[DType.float32]([3], 1.5)
    var b = full[DType.float32]([3], 0.0)
    var c = add_typed(a, b)
    for i in range(3):
        assert_almost_equal(
            Float64(c[i]), 1.5, atol=1e-6, msg="x + 0 = x"
        )
    print("PASS: test_add_zeros")


fn test_subtract_typed() raises:
    """subtract computes correct values."""
    var a = full[DType.float32]([2, 2], 1.5)
    var b = full[DType.float32]([2, 2], 0.5)
    var c = subtract_typed(a, b)
    assert_true(c.get_dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 1.0, atol=1e-6, msg="1.5 - 0.5 = 1.0"
        )
    print("PASS: test_subtract_typed")


fn test_multiply_typed() raises:
    """multiply computes correct values."""
    var a = full[DType.float32]([2, 2], 0.5)
    var b = full[DType.float32]([2, 2], 1.5)
    var c = multiply_typed(a, b)
    assert_true(c.get_dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 0.75, atol=1e-6, msg="0.5 * 1.5 = 0.75"
        )
    print("PASS: test_multiply_typed")


fn test_multiply_ones() raises:
    """multiply with ones is identity."""
    var a = full[DType.float32]([3], 0.25)
    var b = ones[DType.float32]([3])
    var c = multiply_typed(a, b)
    for i in range(3):
        assert_almost_equal(
            Float64(c[i]), 0.25, atol=1e-6, msg="x * 1 = x"
        )
    print("PASS: test_multiply_ones")


fn test_divide_typed() raises:
    """divide computes correct values."""
    var a = full[DType.float32]([2, 2], 1.5)
    var b = full[DType.float32]([2, 2], 0.5)
    var c = divide_typed(a, b)
    assert_true(c.get_dtype() == DType.float32, "dtype should be float32")
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 3.0, atol=1e-6, msg="1.5 / 0.5 = 3.0"
        )
    print("PASS: test_divide_typed")


fn test_divide_by_ones() raises:
    """divide by ones is identity."""
    var a = full[DType.float32]([3], 0.25)
    var b = ones[DType.float32]([3])
    var c = divide_typed(a, b)
    for i in range(3):
        assert_almost_equal(
            Float64(c[i]), 0.25, atol=1e-6, msg="x / 1 = x"
        )
    print("PASS: test_divide_by_ones")


fn test_arithmetic_float64() raises:
    """Arithmetic operations work with float64 dtype."""
    var a = full[DType.float64]([2], 1.0)
    var b = full[DType.float64]([2], 0.5)
    var c = add_typed(a, b)
    assert_true(c.get_dtype() == DType.float64, "dtype should be float64")
    for i in range(2):
        assert_almost_equal(
            Float64(c[i]), 1.5, atol=1e-10, msg="1.0 + 0.5 = 1.5"
        )
    print("PASS: test_arithmetic_float64")


fn main() raises:
    test_add_typed()
    test_add_zeros()
    test_subtract_typed()
    test_multiply_typed()
    test_multiply_ones()
    test_divide_typed()
    test_divide_by_ones()
    test_arithmetic_float64()
    print("All test_tensor_arithmetic tests passed!")
