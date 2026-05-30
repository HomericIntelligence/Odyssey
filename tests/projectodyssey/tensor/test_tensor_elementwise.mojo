"""Tests for AnyTensor elementwise and activation operations.

Tests cover:
- exp: Element-wise exponential via AnyTensor
- relu: ReLU activation via AnyTensor
- sigmoid: Sigmoid activation via AnyTensor
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full as any_full, zeros as any_zeros
from projectodyssey.core.elementwise import exp, log, sqrt, abs, sin, cos
from projectodyssey.core.activation import relu, sigmoid


def test_exp() raises:
    """Exp preserves dtype and computes correct values."""
    var t = any_zeros([4], DType.float32)
    var r = exp(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # exp(0) = 1.0
    for i in range(4):
        assert_almost_equal(Float64(r[i]), 1.0, atol=1e-6, msg="exp(0) = 1")
    print("PASS: test_exp")


def test_exp_one() raises:
    """Exp(1.0) computes e."""
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


def test_log() raises:
    """Log computes correct values."""
    var t = any_full([3], 1.0, DType.float32)
    var r = log(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # log(1) = 0.0
    for i in range(3):
        assert_almost_equal(Float64(r[i]), 0.0, atol=1e-6, msg="log(1) = 0")
    print("PASS: test_log")


def test_sqrt() raises:
    """Sqrt computes correct values."""
    var t = any_full([3], 0.25, DType.float32)
    var r = sqrt(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sqrt(0.25) = 0.5"
        )
    print("PASS: test_sqrt")


def test_abs() raises:
    """Abs computes correct values for negative inputs."""
    var t = any_full([3], -1.5, DType.float32)
    var r = abs(t)
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 1.5, atol=1e-6, msg="abs(-1.5) = 1.5"
        )
    print("PASS: test_abs")


def test_relu() raises:
    """Relu zeros out negatives and preserves positives."""
    var t = any_zeros([4], DType.float32)
    t[0] = -1.0
    t[1] = 0.0
    t[2] = 0.5
    t[3] = 1.5
    var r = relu(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    assert_almost_equal(Float64(r[0]), 0.0, atol=1e-6, msg="relu(-1) = 0")
    assert_almost_equal(Float64(r[1]), 0.0, atol=1e-6, msg="relu(0) = 0")
    assert_almost_equal(Float64(r[2]), 0.5, atol=1e-6, msg="relu(0.5) = 0.5")
    assert_almost_equal(Float64(r[3]), 1.5, atol=1e-6, msg="relu(1.5) = 1.5")
    print("PASS: test_relu")


def test_sigmoid() raises:
    """Sigmoid maps 0 to 0.5."""
    var t = any_zeros([3], DType.float32)
    var r = sigmoid(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # sigmoid(0) = 0.5
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sigmoid(0) = 0.5"
        )
    print("PASS: test_sigmoid")


def test_sin_cos() raises:
    """Sin and cos compute correct values."""
    var t = any_zeros([2], DType.float32)
    var s = sin(t)
    var c = cos(t)
    # sin(0) = 0, cos(0) = 1
    for i in range(2):
        assert_almost_equal(Float64(s[i]), 0.0, atol=1e-6, msg="sin(0) = 0")
        assert_almost_equal(Float64(c[i]), 1.0, atol=1e-6, msg="cos(0) = 1")
    print("PASS: test_sin_cos")


def main() raises:
    test_exp()
    test_exp_one()
    test_log()
    test_sqrt()
    test_abs()
    test_relu()
    test_sigmoid()
    test_sin_cos()
    print("All test_tensor_elementwise tests passed!")
