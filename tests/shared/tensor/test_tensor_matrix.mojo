"""Tests for AnyTensor matrix and reduction operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- matmul: Matrix multiplication via AnyTensor
- transpose: Matrix transpose via AnyTensor
- sum: Tensor reduction (sum) via AnyTensor
- mean: Tensor reduction (mean) via AnyTensor
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, ones as any_ones, full as any_full, zeros as any_zeros
from shared.core.matrix import matmul, transpose
from shared.core.reduction import sum, mean


def test_matmul() raises:
    """Matmul computes correct matrix product."""
    # [2, 3] @ [3, 2] = [2, 2]
    var a = any_ones([2, 3], DType.float32)
    var b = any_ones([3, 2], DType.float32)
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
    print("PASS: test_matmul")


def test_matmul_identity() raises:
    """Matmul with identity-like matrix preserves values."""
    var a = any_full([2, 2], 0.5, DType.float32)
    var eye = any_zeros([2, 2], DType.float32)
    eye[0] = 1.0
    eye[3] = 1.0
    var c = matmul(a, eye)
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 0.5, atol=1e-6, msg="A @ I = A"
        )
    print("PASS: test_matmul_identity")


def test_transpose() raises:
    """Transpose swaps dimensions and preserves dtype."""
    var t = any_zeros([2, 3], DType.float32)
    for i in range(6):
        t[i] = Float32(i)
    var r = transpose(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    var s = r.shape()
    assert_true(s[0] == 3, "transposed rows should be 3")
    assert_true(s[1] == 2, "transposed cols should be 2")
    print("PASS: test_transpose")


def test_sum_all() raises:
    """Sum over all elements computes correct total."""
    var t = any_full([2, 3], 0.5, DType.float32)
    var s = sum(t)
    assert_true(s.dtype() == DType.float32, "dtype should be float32")
    # sum of 6 elements, each 0.5 = 3.0
    assert_almost_equal(
        Float64(s[0]), 3.0, atol=1e-5, msg="sum of 6 * 0.5 = 3.0"
    )
    print("PASS: test_sum_all")


def test_sum_axis() raises:
    """Sum along axis computes correct result."""
    var t = any_ones([2, 3], DType.float32)
    var s = sum(t, axis=1)
    var shape = s.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    # Each row sums to 3.0
    for i in range(2):
        assert_almost_equal(
            Float64(s[i]), 3.0, atol=1e-5, msg="row sum of ones = 3"
        )
    print("PASS: test_sum_axis")


def test_mean_all() raises:
    """Mean over all elements computes correct average."""
    var t = any_full([2, 3], 1.5, DType.float32)
    var m = mean(t)
    assert_true(m.dtype() == DType.float32, "dtype should be float32")
    # mean of 6 elements all 1.5 = 1.5
    assert_almost_equal(
        Float64(m[0]), 1.5, atol=1e-5, msg="mean of all 1.5 = 1.5"
    )
    print("PASS: test_mean_all")


def test_mean_axis() raises:
    """Mean along axis computes correct result."""
    var t = any_zeros([2, 2], DType.float32)
    t[0] = 1.0
    t[1] = 0.0
    t[2] = 0.0
    t[3] = 1.0
    var m = mean(t, axis=1)
    var shape = m.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(m[i]), 0.5, atol=1e-5, msg="row mean = 0.5"
        )
    print("PASS: test_mean_axis")


def test_sum_float64() raises:
    """Sum works with float64 dtype."""
    var t = any_full([3], 1.0, DType.float64)
    var s = sum(t)
    assert_true(s.dtype() == DType.float64, "dtype should be float64")
    assert_almost_equal(
        Float64(s[0]), 3.0, atol=1e-10, msg="sum of 3 * 1.0 = 3.0"
    )
    print("PASS: test_sum_float64")


def main() raises:
    test_matmul()
    test_matmul_identity()
    test_transpose()
    test_sum_all()
    test_sum_axis()
    test_mean_all()
    test_mean_axis()
    test_sum_float64()
    print("All test_tensor_matrix tests passed!")
