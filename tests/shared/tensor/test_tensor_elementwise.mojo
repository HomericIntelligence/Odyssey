"""Tests for typed Tensor[dtype] elementwise and activation operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- exp[dt]: Element-wise exponential
- relu[dt]: ReLU activation
- sigmoid[dt]: Sigmoid activation
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import full, zeros
from shared.core.elementwise import exp, log, sqrt, abs, sin, cos
from shared.core.activation import relu, sigmoid


fn test_exp_typed() raises:
    """exp preserves dtype and computes correct values."""
    var t = zeros[DType.float32]([4])
    var r = exp(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # exp(0) = 1.0
    for i in range(4):
        assert_almost_equal(
            Float64(r[i]), 1.0, atol=1e-6, msg="exp(0) = 1"
        )
    print("PASS: test_exp_typed")


fn test_exp_one() raises:
    """exp(1.0) computes e."""
    var t = full[DType.float32]([2], 1.0)
    var r = exp(t)
    for i in range(2):
        assert_almost_equal(
            Float64(r[i]),
            2.718281828,
            atol=1e-4,
            msg="exp(1) ~ 2.71828",
        )
    print("PASS: test_exp_one")


fn test_log_typed() raises:
    """log computes correct values."""
    var t = full[DType.float32]([3], 1.0)
    var r = log(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # log(1) = 0.0
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.0, atol=1e-6, msg="log(1) = 0"
        )
    print("PASS: test_log_typed")


fn test_sqrt_typed() raises:
    """sqrt computes correct values."""
    var t = full[DType.float32]([3], 0.25)
    var r = sqrt(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sqrt(0.25) = 0.5"
        )
    print("PASS: test_sqrt_typed")


fn test_abs_typed() raises:
    """abs computes correct values for negative inputs."""
    var t = full[DType.float32]([3], -1.5)
    var r = abs(t)
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 1.5, atol=1e-6, msg="abs(-1.5) = 1.5"
        )
    print("PASS: test_abs_typed")


fn test_relu_typed() raises:
    """relu zeros out negatives and preserves positives."""
    var t = Tensor[DType.float32]([4])
    t._data[0] = Scalar[DType.float32](-1.0)
    t._data[1] = Scalar[DType.float32](0.0)
    t._data[2] = Scalar[DType.float32](0.5)
    t._data[3] = Scalar[DType.float32](1.5)
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
    print("PASS: test_relu_typed")


fn test_sigmoid_typed() raises:
    """sigmoid maps 0 to 0.5."""
    var t = zeros[DType.float32]([3])
    var r = sigmoid(t)
    assert_true(r.dtype() == DType.float32, "dtype should be float32")
    # sigmoid(0) = 0.5
    for i in range(3):
        assert_almost_equal(
            Float64(r[i]), 0.5, atol=1e-6, msg="sigmoid(0) = 0.5"
        )
    print("PASS: test_sigmoid_typed")


fn test_sin_cos_typed() raises:
    """sin and cos compute correct values."""
    var t = zeros[DType.float32]([2])
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
    print("PASS: test_sin_cos_typed")


fn main() raises:
    test_exp_typed()
    test_exp_one()
    test_log_typed()
    test_sqrt_typed()
    test_abs_typed()
    test_relu_typed()
    test_sigmoid_typed()
    test_sin_cos_typed()
    print("All test_tensor_elementwise tests passed!")
