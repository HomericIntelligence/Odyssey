"""Tests for native Tensor[dtype] reduction: typed vs AnyTensor equivalence.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- sum_typed vs AnyTensor sum equivalence
- mean_typed vs AnyTensor mean equivalence
- max_reduce_typed, min_reduce_typed
- Known values: sum([1,2,3])=6, mean([2,4,6])=4
- Float64 precision preservation
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import zeros, ones, full
from shared.core.any_tensor import AnyTensor, ones as any_ones, full as any_full
from shared.core.reduction import (
    sum,
    mean,
    sum_typed,
    mean_typed,
    max_reduce_typed,
    min_reduce_typed,
)


fn test_sum_typed_matches_anytensor() raises:
    """Typed sum produces identical results to AnyTensor sum."""
    var a_any = any_full([2, 3], 0.5, DType.float32)
    var result_any = sum(a_any)

    var a_typed = full[DType.float32]([2, 3], 0.5)
    var result_typed = sum_typed(a_typed)

    assert_almost_equal(
        Float64(result_any[0]),
        Float64(result_typed[0]),
        atol=1e-5,
        msg="sum: typed should match AnyTensor",
    )
    # 6 elements * 0.5 = 3.0
    assert_almost_equal(
        Float64(result_typed[0]),
        3.0,
        atol=1e-5,
        msg="sum of 6 * 0.5 = 3.0",
    )
    print("PASS: test_sum_typed_matches_anytensor")


fn test_mean_typed_matches_anytensor() raises:
    """Typed mean produces identical results to AnyTensor mean."""
    var a_any = any_full([2, 3], 1.5, DType.float32)
    var result_any = mean(a_any)

    var a_typed = full[DType.float32]([2, 3], 1.5)
    var result_typed = mean_typed(a_typed)

    assert_almost_equal(
        Float64(result_any[0]),
        Float64(result_typed[0]),
        atol=1e-5,
        msg="mean: typed should match AnyTensor",
    )
    # mean of all 1.5 = 1.5
    assert_almost_equal(
        Float64(result_typed[0]),
        1.5,
        atol=1e-5,
        msg="mean of all 1.5 = 1.5",
    )
    print("PASS: test_mean_typed_matches_anytensor")


fn test_sum_typed_known_values() raises:
    """sum([1,2,3]) = 6 for typed tensors."""
    var t = Tensor[DType.float32]([3])
    t._data[0] = Scalar[DType.float32](1.0)
    t._data[1] = Scalar[DType.float32](2.0)
    t._data[2] = Scalar[DType.float32](3.0)
    var result = sum_typed(t)
    assert_almost_equal(
        Float64(result[0]),
        6.0,
        atol=1e-5,
        msg="sum([1,2,3]) = 6",
    )
    print("PASS: test_sum_typed_known_values")


fn test_mean_typed_known_values() raises:
    """mean([2,4,6]) = 4 for typed tensors."""
    var t = Tensor[DType.float32]([3])
    t._data[0] = Scalar[DType.float32](2.0)
    t._data[1] = Scalar[DType.float32](4.0)
    t._data[2] = Scalar[DType.float32](6.0)
    var result = mean_typed(t)
    assert_almost_equal(
        Float64(result[0]),
        4.0,
        atol=1e-5,
        msg="mean([2,4,6]) = 4",
    )
    print("PASS: test_mean_typed_known_values")


fn test_sum_typed_axis() raises:
    """sum along axis 1 for a [2, 3] tensor."""
    var t = ones[DType.float32]([2, 3])
    var result = sum_typed(t, axis=1)
    var shape = result.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(result[i]),
            3.0,
            atol=1e-5,
            msg="row sum of ones = 3",
        )
    print("PASS: test_sum_typed_axis")


fn test_mean_typed_axis() raises:
    """mean along axis 1 for known values."""
    var t = Tensor[DType.float32]([2, 2])
    t._data[0] = Scalar[DType.float32](1.0)
    t._data[1] = Scalar[DType.float32](0.0)
    t._data[2] = Scalar[DType.float32](0.0)
    t._data[3] = Scalar[DType.float32](1.0)
    var result = mean_typed(t, axis=1)
    var shape = result.shape()
    assert_true(shape[0] == 2, "reduced shape should be [2]")
    for i in range(2):
        assert_almost_equal(
            Float64(result[i]),
            0.5,
            atol=1e-5,
            msg="row mean = 0.5",
        )
    print("PASS: test_mean_typed_axis")


fn test_max_reduce_typed() raises:
    """max_reduce returns maximum element."""
    var t = Tensor[DType.float32]([4])
    t._data[0] = Scalar[DType.float32](-1.0)
    t._data[1] = Scalar[DType.float32](0.5)
    t._data[2] = Scalar[DType.float32](1.5)
    t._data[3] = Scalar[DType.float32](0.0)
    var result = max_reduce_typed(t)
    assert_almost_equal(
        Float64(result[0]),
        1.5,
        atol=1e-6,
        msg="max of [-1, 0.5, 1.5, 0] = 1.5",
    )
    print("PASS: test_max_reduce_typed")


fn test_min_reduce_typed() raises:
    """min_reduce returns minimum element."""
    var t = Tensor[DType.float32]([4])
    t._data[0] = Scalar[DType.float32](-1.0)
    t._data[1] = Scalar[DType.float32](0.5)
    t._data[2] = Scalar[DType.float32](1.5)
    t._data[3] = Scalar[DType.float32](0.0)
    var result = min_reduce_typed(t)
    assert_almost_equal(
        Float64(result[0]),
        -1.0,
        atol=1e-6,
        msg="min of [-1, 0.5, 1.5, 0] = -1",
    )
    print("PASS: test_min_reduce_typed")


fn test_sum_typed_float64_precision() raises:
    """Typed float64 sum preserves full precision."""
    var t = Tensor[DType.float64]([3])
    t._data[0] = Scalar[DType.float64](3.141592653589793)
    t._data[1] = Scalar[DType.float64](2.718281828459045)
    t._data[2] = Scalar[DType.float64](1.0e-15)
    var result = sum_typed(t)
    var expected = 3.141592653589793 + 2.718281828459045 + 1.0e-15
    assert_almost_equal(
        Float64(result[0]),
        expected,
        atol=1e-14,
        msg="float64 sum should preserve full precision",
    )
    print("PASS: test_sum_typed_float64_precision")


fn main() raises:
    test_sum_typed_matches_anytensor()
    test_mean_typed_matches_anytensor()
    test_sum_typed_known_values()
    test_mean_typed_known_values()
    test_sum_typed_axis()
    test_mean_typed_axis()
    test_max_reduce_typed()
    test_min_reduce_typed()
    test_sum_typed_float64_precision()
    print("All test_tensor_reduction_native tests passed!")
