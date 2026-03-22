"""Tests for Tensor[dtype] typed arithmetic operation overloads.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- add element-wise
- subtract element-wise
- multiply element-wise
- divide element-wise
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.core.arithmetic import add, subtract, multiply, divide


fn test_add_typed() raises:
    """Element-wise addition."""
    var a = Tensor[DType.float32]([3])
    var b = Tensor[DType.float32]([3])
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    b[0] = 10.0
    b[1] = 20.0
    b[2] = 30.0

    var r = add(a, b)
    assert_true(r.numel() == 3, "numel should be 3")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](11.0), msg="val 0")
    assert_almost_equal(r[1], Scalar[DType.float32](22.0), msg="val 1")
    assert_almost_equal(r[2], Scalar[DType.float32](33.0), msg="val 2")
    print("PASS: test_add_typed")


fn test_subtract_typed() raises:
    """Element-wise subtraction."""
    var a = Tensor[DType.float32]([3])
    var b = Tensor[DType.float32]([3])
    a[0] = 10.0
    a[1] = 20.0
    a[2] = 30.0
    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0

    var r = subtract(a, b)
    assert_true(r.numel() == 3, "numel should be 3")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](9.0), msg="val 0")
    assert_almost_equal(r[1], Scalar[DType.float32](18.0), msg="val 1")
    assert_almost_equal(r[2], Scalar[DType.float32](27.0), msg="val 2")
    print("PASS: test_subtract_typed")


fn test_multiply_typed() raises:
    """Element-wise multiplication."""
    var a = Tensor[DType.float32]([3])
    var b = Tensor[DType.float32]([3])
    a[0] = 2.0
    a[1] = 3.0
    a[2] = 4.0
    b[0] = 5.0
    b[1] = 6.0
    b[2] = 7.0

    var r = multiply(a, b)
    assert_true(r.numel() == 3, "numel should be 3")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](10.0), msg="val 0")
    assert_almost_equal(r[1], Scalar[DType.float32](18.0), msg="val 1")
    assert_almost_equal(r[2], Scalar[DType.float32](28.0), msg="val 2")
    print("PASS: test_multiply_typed")


fn test_divide_typed() raises:
    """Element-wise division."""
    var a = Tensor[DType.float32]([3])
    var b = Tensor[DType.float32]([3])
    a[0] = 10.0
    a[1] = 20.0
    a[2] = 30.0
    b[0] = 2.0
    b[1] = 4.0
    b[2] = 5.0

    var r = divide(a, b)
    assert_true(r.numel() == 3, "numel should be 3")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](5.0), msg="val 0")
    assert_almost_equal(r[1], Scalar[DType.float32](5.0), msg="val 1")
    assert_almost_equal(r[2], Scalar[DType.float32](6.0), msg="val 2")
    print("PASS: test_divide_typed")


fn main() raises:
    test_add_typed()
    test_subtract_typed()
    test_multiply_typed()
    test_divide_typed()
