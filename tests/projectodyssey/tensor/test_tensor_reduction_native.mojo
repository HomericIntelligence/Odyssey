"""Tests for AnyTensor reduction operations and precision.

Tests cover:
- AnyTensor sum, mean, max_reduce, min_reduce correctness
- Known values: sum([1,2,3])=6, mean([2,4,6])=4
- Axis reduction
- Float64 precision preservation
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    ones as any_ones,
    full as any_full,
    zeros as any_zeros,
)
from projectodyssey.core.reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
)


def test_sum_correctness() raises:
    """AnyTensor sum produces correct results."""
    var a = any_full([2, 3], 0.5, DType.float32)
    var result = sum(a)

    # 6 elements * 0.5 = 3.0
    assert_almost_equal(
        Float64(result[0]),
        3.0,
        atol=1e-5,
        msg="sum of 6 * 0.5 = 3.0",
    )
    print("PASS: test_sum_correctness")


def test_mean_correctness() raises:
    """AnyTensor mean produces correct results."""
    var a = any_full([2, 3], 1.5, DType.float32)
    var result = mean(a)

    # mean of all 1.5 = 1.5
    assert_almost_equal(
        Float64(result[0]),
        1.5,
        atol=1e-5,
        msg="mean of all 1.5 = 1.5",
    )
    print("PASS: test_mean_correctness")


def test_sum_known_values() raises:
    """Sum([1,2,3]) = 6."""
    var t = any_zeros([3], DType.float32)
    t[0] = 1.0
    t[1] = 2.0
    t[2] = 3.0
    var result = sum(t)
    assert_almost_equal(
        Float64(result[0]),
        6.0,
        atol=1e-5,
        msg="sum([1,2,3]) = 6",
    )
    print("PASS: test_sum_known_values")


def test_mean_known_values() raises:
    """Mean([2,4,6]) = 4."""
    var t = any_zeros([3], DType.float32)
    t[0] = 2.0
    t[1] = 4.0
    t[2] = 6.0
    var result = mean(t)
    assert_almost_equal(
        Float64(result[0]),
        4.0,
        atol=1e-5,
        msg="mean([2,4,6]) = 4",
    )
    print("PASS: test_mean_known_values")


def test_sum_axis() raises:
    """Sum along axis 1 for a [2, 3] tensor."""
    var t = any_ones([2, 3], DType.float32)
    var result = sum(t, axis=1)
    var shape = result.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(result[i]),
            3.0,
            atol=1e-5,
            msg="row sum of ones = 3",
        )
    print("PASS: test_sum_axis")


def test_mean_axis() raises:
    """Mean along axis 1 for known values."""
    var t = any_zeros([2, 2], DType.float32)
    t[0] = 1.0
    t[1] = 0.0
    t[2] = 0.0
    t[3] = 1.0
    var result = mean(t, axis=1)
    var shape = result.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(result[i]),
            0.5,
            atol=1e-5,
            msg="row mean = 0.5",
        )
    print("PASS: test_mean_axis")


def test_max_reduce() raises:
    """Max_reduce returns maximum element."""
    var t = any_zeros([4], DType.float32)
    t[0] = -1.0
    t[1] = 0.5
    t[2] = 1.5
    t[3] = 0.0
    var result = max_reduce(t)
    assert_almost_equal(
        Float64(result[0]),
        1.5,
        atol=1e-6,
        msg="max of [-1, 0.5, 1.5, 0] = 1.5",
    )
    print("PASS: test_max_reduce")


def test_min_reduce() raises:
    """Min_reduce returns minimum element."""
    var t = any_zeros([4], DType.float32)
    t[0] = -1.0
    t[1] = 0.5
    t[2] = 1.5
    t[3] = 0.0
    var result = min_reduce(t)
    assert_almost_equal(
        Float64(result[0]),
        -1.0,
        atol=1e-6,
        msg="min of [-1, 0.5, 1.5, 0] = -1",
    )
    print("PASS: test_min_reduce")


def test_sum_float64_precision() raises:
    """Float64 sum preserves full precision."""
    var t = any_zeros([3], DType.float64)
    t[0] = 3.141592653589793
    t[1] = 2.718281828459045
    t[2] = 1.0e-15
    var result = sum(t)
    var expected = 3.141592653589793 + 2.718281828459045 + 1.0e-15
    assert_almost_equal(
        Float64(result[0]),
        expected,
        atol=1e-14,
        msg="float64 sum should preserve full precision",
    )
    print("PASS: test_sum_float64_precision")


def main() raises:
    test_sum_correctness()
    test_mean_correctness()
    test_sum_known_values()
    test_mean_known_values()
    test_sum_axis()
    test_mean_axis()
    test_max_reduce()
    test_min_reduce()
    test_sum_float64_precision()
    print("All test_tensor_reduction_native tests passed!")
