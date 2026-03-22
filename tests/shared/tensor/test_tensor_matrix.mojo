"""Tests for typed Tensor[dtype] matrix and reduction operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- matmul[dt]: Matrix multiplication
- transpose[dt]: Matrix transpose
- sum[dt]: Tensor reduction (sum)
- mean[dt]: Tensor reduction (mean)
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import ones, full, zeros
from shared.core.matrix import matmul, transpose
from shared.core.reduction import sum, mean


fn test_matmul_typed() raises:
    """matmul computes correct matrix product."""
    # [2, 3] @ [3, 2] = [2, 2]
    var a = ones[DType.float32]([2, 3])
    var b = ones[DType.float32]([3, 2])
    var c = matmul(a, b)
    assert_true(c.dtype() == DType.float32, "dtype should be float32")
    var s = c.shape()
    assert_true(s[0] == 2, "output rows should be 2")
    assert_true(s[1] == 2, "output cols should be 2")
    # Each element = sum of 3 ones = 3.0
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 3.0, atol=1e-6, msg="ones @ ones = 3"
        )
    print("PASS: test_matmul_typed")


fn test_matmul_identity() raises:
    """matmul with identity-like matrix preserves values."""
    var a = full[DType.float32]([2, 2], 0.5)
    var eye = Tensor[DType.float32]([2, 2])
    eye._data[0] = Scalar[DType.float32](1.0)
    eye._data[1] = Scalar[DType.float32](0.0)
    eye._data[2] = Scalar[DType.float32](0.0)
    eye._data[3] = Scalar[DType.float32](1.0)
    var c = matmul(a, eye)
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 0.5, atol=1e-6, msg="A @ I = A"
        )
    print("PASS: test_matmul_identity")


fn test_transpose_typed() raises:
    """transpose swaps dimensions and preserves dtype."""
    var t = Tensor[DType.float32]([2, 3])
    # Fill with 0..5
    for i in range(6):
        t._data[i] = Scalar[DType.float32](Float32(i))
    var r = transpose(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    var s = r.shape()
    assert_true(s[0] == 3, "transposed rows should be 3")
    assert_true(s[1] == 2, "transposed cols should be 2")
    print("PASS: test_transpose_typed")


fn test_sum_all_typed() raises:
    """sum over all elements computes correct total."""
    var t = full[DType.float32]([2, 3], 0.5)
    var s = sum(t)
    assert_true(s.dtype() == DType.float32, "dtype should be float32")
    # sum of 6 elements, each 0.5 = 3.0
    assert_almost_equal(
        Float64(s[0]), 3.0, atol=1e-5, msg="sum of 6 * 0.5 = 3.0"
    )
    print("PASS: test_sum_all_typed")


fn test_sum_axis_typed() raises:
    """sum along axis computes correct result."""
    var t = ones[DType.float32]([2, 3])
    var s = sum(t, axis=1)
    var shape = s.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    # Each row sums to 3.0
    for i in range(2):
        assert_almost_equal(
            Float64(s[i]), 3.0, atol=1e-5, msg="row sum of ones = 3"
        )
    print("PASS: test_sum_axis_typed")


fn test_mean_all_typed() raises:
    """mean over all elements computes correct average."""
    var t = full[DType.float32]([2, 3], 1.5)
    var m = mean(t)
    assert_true(m.dtype() == DType.float32, "dtype should be float32")
    # mean of 6 elements all 1.5 = 1.5
    assert_almost_equal(
        Float64(m[0]), 1.5, atol=1e-5, msg="mean of all 1.5 = 1.5"
    )
    print("PASS: test_mean_all_typed")


fn test_mean_axis_typed() raises:
    """mean along axis computes correct result."""
    var t = Tensor[DType.float32]([2, 2])
    t._data[0] = Scalar[DType.float32](1.0)
    t._data[1] = Scalar[DType.float32](0.0)
    t._data[2] = Scalar[DType.float32](0.0)
    t._data[3] = Scalar[DType.float32](1.0)
    var m = mean(t, axis=1)
    var shape = m.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(m[i]), 0.5, atol=1e-5, msg="row mean = 0.5"
        )
    print("PASS: test_mean_axis_typed")


fn test_sum_float64() raises:
    """sum works with float64 dtype."""
    var t = full[DType.float64]([3], 1.0)
    var s = sum(t)
    assert_true(s.dtype() == DType.float64, "dtype should be float64")
    assert_almost_equal(
        Float64(s[0]), 3.0, atol=1e-10, msg="sum of 3 * 1.0 = 3.0"
    )
    print("PASS: test_sum_float64")


fn main() raises:
    test_matmul_typed()
    test_matmul_identity()
    test_transpose_typed()
    test_sum_all_typed()
    test_sum_axis_typed()
    test_mean_all_typed()
    test_mean_axis_typed()
    test_sum_float64()
    print("All test_tensor_matrix tests passed!")
