"""Tests for AnyTensor elementwise operations and precision.

Tests cover:
- AnyTensor exp, log, sqrt, abs, sin, cos correctness
- Edge cases: exp(0)=1, log(1)=0, sqrt(4)=2, abs(-3)=3
- Float64 precision preservation
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    full as any_full,
    zeros as any_zeros,
)
from projectodyssey.core.elementwise import (
    exp,
    log,
    sqrt,
    abs,
    sin,
    cos,
)


def test_exp_correctness() raises:
    """AnyTensor exp produces correct results."""
    var a = any_full([6], 0.5, DType.float32)
    var result = exp(a)

    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            1.6487212707,
            atol=1e-4,
            msg="exp(0.5) ~ 1.6487",
        )
    print("PASS: test_exp_correctness")


def test_log_correctness() raises:
    """AnyTensor log produces correct results."""
    var a = any_full([6], 1.5, DType.float32)
    var result = log(a)

    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            0.4054651081,
            atol=1e-4,
            msg="log(1.5) ~ 0.4055",
        )
    print("PASS: test_log_correctness")


def test_sqrt_abs_correctness() raises:
    """AnyTensor sqrt and abs produce correct results."""
    # sqrt
    var s = any_full([4], 0.25, DType.float32)
    var sr = sqrt(s)
    for i in range(4):
        assert_almost_equal(
            Float64(sr[i]),
            0.5,
            atol=1e-6,
            msg="sqrt(0.25) = 0.5",
        )

    # abs
    var a = any_full([4], -1.5, DType.float32)
    var ar = abs(a)
    for i in range(4):
        assert_almost_equal(
            Float64(ar[i]),
            1.5,
            atol=1e-6,
            msg="abs(-1.5) = 1.5",
        )
    print("PASS: test_sqrt_abs_correctness")


def test_sin_cos_correctness() raises:
    """AnyTensor sin and cos produce correct results."""
    var a = any_full([4], 0.5, DType.float32)
    var sin_r = sin(a)
    var cos_r = cos(a)

    for i in range(4):
        assert_almost_equal(
            Float64(sin_r[i]),
            0.4794255386,
            atol=1e-4,
            msg="sin(0.5) ~ 0.4794",
        )
        assert_almost_equal(
            Float64(cos_r[i]),
            0.8775825619,
            atol=1e-4,
            msg="cos(0.5) ~ 0.8776",
        )
    print("PASS: test_sin_cos_correctness")


def test_exp_edge_cases() raises:
    """Exp(0)=1 and exp(1)~2.71828."""
    var t = any_zeros([2], DType.float32)
    t[0] = 0.0
    t[1] = 1.0
    var r = exp(t)
    assert_almost_equal(Float64(r[0]), 1.0, atol=1e-6, msg="exp(0) = 1")
    assert_almost_equal(Float64(r[1]), 2.718281828, atol=1e-4, msg="exp(1) ~ e")
    print("PASS: test_exp_edge_cases")


def test_log_sqrt_edge_cases() raises:
    """Log(1)=0 and sqrt(4)=2."""
    # log(1) = 0
    var t1 = any_full([2], 1.0, DType.float32)
    var r1 = log(t1)
    for i in range(2):
        assert_almost_equal(Float64(r1[i]), 0.0, atol=1e-6, msg="log(1) = 0")

    # sqrt(4) = 2
    var t2 = any_full([2], 4.0, DType.float32)
    var r2 = sqrt(t2)
    for i in range(2):
        assert_almost_equal(Float64(r2[i]), 2.0, atol=1e-6, msg="sqrt(4) = 2")
    print("PASS: test_log_sqrt_edge_cases")


def test_abs_edge_cases() raises:
    """Abs(-3)=3, abs(0)=0, abs(1.5)=1.5."""
    var t = any_zeros([3], DType.float32)
    t[0] = -3.0
    t[1] = 0.0
    t[2] = 1.5
    var r = abs(t)
    assert_almost_equal(Float64(r[0]), 3.0, atol=1e-6, msg="abs(-3) = 3")
    assert_almost_equal(Float64(r[1]), 0.0, atol=1e-6, msg="abs(0) = 0")
    assert_almost_equal(Float64(r[2]), 1.5, atol=1e-6, msg="abs(1.5) = 1.5")
    print("PASS: test_abs_edge_cases")


def test_exp_float64_precision() raises:
    """Float64 exp preserves full precision."""
    var t = any_zeros([2], DType.float64)
    t[0] = 1.0
    t[1] = 0.5
    var r = exp(t)
    assert_almost_equal(
        Float64(r[0]),
        2.718281828459045,
        atol=1e-14,
        msg="float64 exp(1) should preserve full precision",
    )
    assert_almost_equal(
        Float64(r[1]),
        1.6487212707001282,
        atol=1e-14,
        msg="float64 exp(0.5) should preserve full precision",
    )
    print("PASS: test_exp_float64_precision")


def test_log_float64_precision() raises:
    """Float64 log preserves full precision."""
    var t = any_full([1], 2.718281828459045, DType.float64)
    var r = log(t)
    assert_almost_equal(
        Float64(r[0]),
        1.0,
        atol=1e-14,
        msg="float64 log(e) should be 1.0",
    )
    print("PASS: test_log_float64_precision")


def main() raises:
    test_exp_correctness()
    test_log_correctness()
    test_sqrt_abs_correctness()
    test_sin_cos_correctness()
    test_exp_edge_cases()
    test_log_sqrt_edge_cases()
    test_abs_edge_cases()
    test_exp_float64_precision()
    test_log_float64_precision()
    print("All test_tensor_elementwise_native tests passed!")
