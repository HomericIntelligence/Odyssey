"""Tests that AnyTensor operations correctly dispatch to typed cores.

Tests cover:
- AnyTensor add/subtract/multiply dispatch produces correct results
- AnyTensor exp/log dispatch produces correct results
- AnyTensor matmul dispatch produces correct results
- Multiple dtype dispatch (float32, float64)
- dtype mismatch raises error
- Conversion round-trip via as_tensor/as_any
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.tensor import Tensor
from projectodyssey.tensor.factories import (
    ones as typed_ones,
    full as typed_full,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    full as any_full,
    ones as any_ones,
    zeros as any_zeros,
)
from projectodyssey.core.arithmetic import add, subtract, multiply
from projectodyssey.core.elementwise import exp, log
from projectodyssey.core.matrix import matmul
from projectodyssey.core.reduction import sum


def test_anytensor_add_dispatch() raises:
    """AnyTensor add dispatches correctly and produces correct results."""
    var a = any_full([3, 4], 0.5, DType.float32)
    var b = any_ones([3, 4], DType.float32)
    var result = add(a, b)
    assert_true(result.numel() == 12, "result numel should be 12")
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            1.5,
            atol=1e-6,
            msg="0.5 + 1.0 = 1.5",
        )
    print("PASS: test_anytensor_add_dispatch")


def test_anytensor_subtract_multiply_dispatch() raises:
    """AnyTensor subtract and multiply dispatch correctly."""
    var a = any_full([4], 1.5, DType.float32)
    var b = any_full([4], 0.5, DType.float32)

    var sub_result = subtract(a, b)
    for i in range(4):
        assert_almost_equal(
            Float64(sub_result[i]),
            1.0,
            atol=1e-6,
            msg="1.5 - 0.5 = 1.0",
        )

    var mul_result = multiply(a, b)
    for i in range(4):
        assert_almost_equal(
            Float64(mul_result[i]),
            0.75,
            atol=1e-6,
            msg="1.5 * 0.5 = 0.75",
        )
    print("PASS: test_anytensor_subtract_multiply_dispatch")


def test_anytensor_exp_log_dispatch() raises:
    """AnyTensor exp and log dispatch correctly."""
    var t = any_full([4], 1.0, DType.float32)
    var exp_result = exp(t)
    for i in range(4):
        assert_almost_equal(
            Float64(exp_result[i]),
            2.718281828,
            atol=1e-4,
            msg="exp(1) ~ e",
        )

    var t2 = any_full([4], 1.0, DType.float32)
    var log_result = log(t2)
    for i in range(4):
        assert_almost_equal(
            Float64(log_result[i]),
            0.0,
            atol=1e-6,
            msg="log(1) = 0",
        )
    print("PASS: test_anytensor_exp_log_dispatch")


def test_anytensor_matmul_dispatch() raises:
    """AnyTensor matmul dispatch produces correct shape and values."""
    var a = any_ones([2, 3], DType.float32)
    var b = any_ones([3, 4], DType.float32)
    var result = matmul(a, b)
    var shape = result.shape()
    assert_true(shape[0] == 2, "result rows should be 2")
    assert_true(shape[1] == 4, "result cols should be 4")
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            3.0,
            atol=1e-6,
            msg="ones @ ones = 3",
        )
    print("PASS: test_anytensor_matmul_dispatch")


def test_anytensor_float64_dispatch() raises:
    """AnyTensor operations dispatch correctly with float64 dtype."""
    var a = any_full([3], 1.0, DType.float64)
    var b = any_full([3], 0.5, DType.float64)
    var result = add(a, b)
    assert_true(
        result.dtype() == DType.float64,
        "result dtype should be float64",
    )
    for i in range(3):
        assert_almost_equal(
            Float64(result[i]),
            1.5,
            atol=1e-10,
            msg="float64: 1.0 + 0.5 = 1.5",
        )
    print("PASS: test_anytensor_float64_dispatch")


def test_anytensor_dtype_mismatch_raises() raises:
    """AnyTensor add with mismatched dtypes raises error."""
    var a = any_full([3], 1.0, DType.float32)
    var b = any_full([3], 1.0, DType.float64)
    var raised = False
    try:
        var result = add(a, b)
        _ = result  # suppress unused warning
    except:
        raised = True
    assert_true(raised, "add with mismatched dtypes should raise")
    print("PASS: test_anytensor_dtype_mismatch_raises")


def test_conversion_roundtrip_preserves_values() raises:
    """Tensor -> AnyTensor -> Tensor round-trip preserves data."""
    var t1 = Tensor[DType.float32]([4])
    t1._data[0] = Scalar[DType.float32](1.5)
    t1._data[1] = Scalar[DType.float32](-0.5)
    t1._data[2] = Scalar[DType.float32](0.0)
    t1._data[3] = Scalar[DType.float32](0.25)

    # Round-trip: Tensor -> AnyTensor -> Tensor
    var any_t = t1.as_any()
    var t2 = any_t.as_tensor[DType.float32]()

    for i in range(4):
        assert_almost_equal(
            Float64(t1[i]),
            Float64(t2[i]),
            atol=1e-7,
            msg="round-trip should preserve values",
        )
    print("PASS: test_conversion_roundtrip_preserves_values")


def test_conversion_roundtrip_preserves_shape() raises:
    """Tensor -> AnyTensor -> Tensor round-trip preserves shape."""
    var t = typed_ones[DType.float32]([2, 3, 4])
    var any_t = t.as_any()
    var t2 = any_t.as_tensor[DType.float32]()
    var s1 = t.shape()
    var s2 = t2.shape()
    assert_true(len(s1) == len(s2), "ndims should match")
    for i in range(len(s1)):
        assert_true(s1[i] == s2[i], "shape dim should match")
    print("PASS: test_conversion_roundtrip_preserves_shape")


def test_anytensor_sum_dispatch() raises:
    """AnyTensor sum dispatch produces correct results."""
    var t = any_full([2, 3], 0.5, DType.float32)
    var result = sum(t)
    # 6 elements * 0.5 = 3.0
    assert_almost_equal(
        Float64(result[0]),
        3.0,
        atol=1e-5,
        msg="sum of 6 * 0.5 = 3.0",
    )
    print("PASS: test_anytensor_sum_dispatch")


def main() raises:
    test_anytensor_add_dispatch()
    test_anytensor_subtract_multiply_dispatch()
    test_anytensor_exp_log_dispatch()
    test_anytensor_matmul_dispatch()
    test_anytensor_float64_dispatch()
    test_anytensor_dtype_mismatch_raises()
    test_conversion_roundtrip_preserves_values()
    test_conversion_roundtrip_preserves_shape()
    test_anytensor_sum_dispatch()
    print("All test_anytensor_dispatch tests passed!")
