"""Tests for AnyTensor elementwise and activation operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- exp: Element-wise exponential via AnyTensor
- relu: ReLU activation via AnyTensor
- sigmoid: Sigmoid activation via AnyTensor
"""

from testing import assert_true, assert_almost_equal
from shared.core.any_tensor import AnyTensor, full as any_full, zeros as any_zeros
from shared.core.elementwise import exp, log, sqrt, abs, sin, cos
from shared.core.activation import relu, sigmoid


fn test_exp() raises:
    """exp preserves dtype and computes correct values."""
    var t = any_zeros([4], DType.float32)
    var r = exp(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # exp(0) = 1.0
    for i in range(4):
        assert_almost_equal(
            Float64(r[i]), 1.0, atol=1e-6, msg="exp(0) = 1"
        )
    print("PASS: test_exp")


fn test_exp_one() raises:
    """exp(1.0) computes e."""
    var t = any_full([2], 1.0, DType.float32)
    var r = exp(t)
    for i in range(2):
        assert_almost_equal(
            Float64(r[i]),
            2.718281828,
            atol=1e-4,
            msg="exp(1) ~ 2.71828",
        )
    print("PASS: test_exp_one")


fn test_log() raises:
    """log computes correct values."""
    var t = any_full([3], 1.0, DType.float32)
    var r = log(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # log(1) = 0.0
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.0, atol=1e-6, msg="log(1) = 0"
        )
    print("PASS: test_log")


fn test_sqrt() raises:
    """sqrt computes correct values."""
    var t = any_full([3], 0.25, DType.float32)
    var r = sqrt(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sqrt(0.25) = 0.5"
        )
    print("PASS: test_sqrt")


fn test_abs() raises:
    """abs computes correct values for negative inputs."""
    var t = any_full([3], -1.5, DType.float32)
    var r = abs(t)
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 1.5, atol=1e-6, msg="abs(-1.5) = 1.5"
        )
    print("PASS: test_abs")


fn test_relu() raises:
    """relu zeros out negatives and preserves positives."""
    var t = any_zeros([4], DType.float32)
    t[0] = -1.0
    t[1] = 0.0
    t[2] = 0.5
    t[3] = 1.5
    var r = relu(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    assert_almost_equal(
        Float64(r[0]), 0.0, atol=1e-6, msg="relu(-1) = 0"
    )
    assert_almost_equal(
        Float64(r[1]), 0.0, atol=1e-6, msg="relu(0) = 0"
    )
    assert_almost_equal(
        Float64(r[2]), 0.5, atol=1e-6, msg="relu(0.5) = 0.5"
    )
    assert_almost_equal(
        Float64(r[3]), 1.5, atol=1e-6, msg="relu(1.5) = 1.5"
    )
    print("PASS: test_relu")


fn test_sigmoid() raises:
    """sigmoid maps 0 to 0.5."""
    var t = any_zeros([3], DType.float32)
    var r = sigmoid(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # sigmoid(0) = 0.5
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sigmoid(0) = 0.5"
        )
    print("PASS: test_sigmoid")


fn test_sin_cos() raises:
    """sin and cos compute correct values."""
    var t = any_zeros([2], DType.float32)
    var s = sin(t)
    var c = cos(t)
    # sin(0) = 0, cos(0) = 1
    for i in range(2):
        assert_almost_equal(
            Float64(s[i]), 0.0, atol=1e-6, msg="sin(0) = 0"
        )
        assert_almost_equal(
            Float64(c[i]), 1.0, atol=1e-6, msg="cos(0) = 1"
        )
    print("PASS: test_sin_cos")


fn main() raises:
    test_exp()
    test_exp_one()
    test_log()
    test_sqrt()
    test_abs()
    test_relu()
    test_sigmoid()
    test_sin_cos()
    print("All test_tensor_elementwise tests passed!")
