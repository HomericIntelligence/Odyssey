"""Tests for Tensor[dtype] typed matrix and reduction overloads.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- matmul[dt] matrix multiply
- transpose[dt] swap dimensions
- sum[dt] sum all elements
- mean[dt] mean of elements
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.core.matrix import matmul, transpose
from shared.core.reduction import sum, mean


fn test_matmul_typed() raises:
    """Matmul typed overload computes matrix product."""
    # 2x2 @ 2x2
    var a = Tensor[DType.float32]([2, 2])
    a[0] = Float32(1.0)
    a[1] = Float32(0.5)
    a[2] = Float32(0.25)
    a[3] = Float32(1.0)

    var b = Tensor[DType.float32]([2, 2])
    b[0] = Float32(1.0)
    b[1] = Float32(0.0)
    b[2] = Float32(0.0)
    b[3] = Float32(1.0)

    var result = matmul(a, b)
    var s = result.shape()
    assert_true(s[0] == 2 and s[1] == 2, "shape preserved")
    # Identity multiplication: result should equal a
    assert_almost_equal(Float64(result[0]), 1.0, atol=1e-6)
    assert_almost_equal(Float64(result[1]), 0.5, atol=1e-6)
    assert_almost_equal(Float64(result[2]), 0.25, atol=1e-6)
    assert_almost_equal(Float64(result[3]), 1.0, atol=1e-6)
    print("PASS: test_matmul_typed")


fn test_transpose_typed() raises:
    """Transpose typed overload swaps dimensions."""
    var t = Tensor[DType.float32]([2, 3])
    # Fill with distinguishable values
    t[0] = Float32(1.0)
    t[1] = Float32(0.5)
    t[2] = Float32(0.25)
    t[3] = Float32(1.5)
    t[4] = Float32(-1.0)
    t[5] = Float32(-0.5)

    var result = transpose(t)
    var s = result.shape()
    assert_true(s[0] == 3, "transposed dim 0")
    assert_true(s[1] == 2, "transposed dim 1")
    # Check transposed values: result[col, row] == original[row, col]
    # Original [0,0]=1.0 -> result[0,0]=1.0
    assert_almost_equal(Float64(result[0]), 1.0, atol=1e-6)
    # Original [1,0]=1.5 -> result[0,1]=1.5
    assert_almost_equal(Float64(result[1]), 1.5, atol=1e-6)
    # Original [0,1]=0.5 -> result[1,0]=0.5
    assert_almost_equal(Float64(result[2]), 0.5, atol=1e-6)
    print("PASS: test_transpose_typed")


fn test_sum_typed() raises:
    """Sum typed overload sums all elements."""
    var t = Tensor[DType.float32]([4])
    t[0] = Float32(0.5)
    t[1] = Float32(1.0)
    t[2] = Float32(1.5)
    t[3] = Float32(0.25)

    var result = sum(t)
    # Sum of all: 0.5 + 1.0 + 1.5 + 0.25 = 3.25
    assert_almost_equal(Float64(result[0]), 3.25, atol=1e-6)
    print("PASS: test_sum_typed")


fn test_mean_typed() raises:
    """Mean typed overload computes mean of all elements."""
    var t = Tensor[DType.float32]([4])
    t[0] = Float32(0.5)
    t[1] = Float32(1.0)
    t[2] = Float32(1.5)
    t[3] = Float32(0.25)

    var result = mean(t)
    # Mean: 3.25 / 4 = 0.8125
    assert_almost_equal(Float64(result[0]), 0.8125, atol=1e-6)
    print("PASS: test_mean_typed")


fn main() raises:
    test_matmul_typed()
    test_transpose_typed()
    test_sum_typed()
    test_mean_typed()
