"""Tests for Tensor[dtype] typed elementwise and activation overloads.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- exp[dt] on known values
- log[dt] on known values
- sqrt[dt] on known values
- relu[dt] zeroes negatives
- sigmoid[dt] of 0 is 0.5
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.core.elementwise import exp, log, sqrt
from shared.core.activation import relu, sigmoid
from math import exp as math_exp, log as math_log, sqrt as math_sqrt


fn test_exp_typed() raises:
    """Exp typed overload computes element-wise exponential."""
    var t = Tensor[DType.float32]([3])
    t[0] = Float32(0.0)
    t[1] = Float32(0.5)
    t[2] = Float32(1.0)
    var result = exp(t)
    assert_true(result.numel() == 3, "numel preserved")
    assert_almost_equal(
        Float64(result[0]), Float64(math_exp(Float32(0.0))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[1]), Float64(math_exp(Float32(0.5))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[2]), Float64(math_exp(Float32(1.0))), atol=1e-6
    )
    print("PASS: test_exp_typed")


fn test_log_typed() raises:
    """Log typed overload computes element-wise natural log."""
    var t = Tensor[DType.float32]([3])
    t[0] = Float32(0.5)
    t[1] = Float32(1.0)
    t[2] = Float32(1.5)
    var result = log(t)
    assert_true(result.numel() == 3, "numel preserved")
    assert_almost_equal(
        Float64(result[0]), Float64(math_log(Float32(0.5))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[1]), Float64(math_log(Float32(1.0))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[2]), Float64(math_log(Float32(1.5))), atol=1e-6
    )
    print("PASS: test_log_typed")


fn test_sqrt_typed() raises:
    """Sqrt typed overload computes element-wise square root."""
    var t = Tensor[DType.float32]([3])
    t[0] = Float32(0.25)
    t[1] = Float32(1.0)
    t[2] = Float32(1.5)
    var result = sqrt(t)
    assert_true(result.numel() == 3, "numel preserved")
    assert_almost_equal(
        Float64(result[0]), Float64(math_sqrt(Float32(0.25))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[1]), Float64(math_sqrt(Float32(1.0))), atol=1e-6
    )
    assert_almost_equal(
        Float64(result[2]), Float64(math_sqrt(Float32(1.5))), atol=1e-6
    )
    print("PASS: test_sqrt_typed")


fn test_relu_typed() raises:
    """ReLU typed overload zeroes negative values."""
    var t = Tensor[DType.float32]([4])
    t[0] = Float32(-1.0)
    t[1] = Float32(-0.5)
    t[2] = Float32(0.0)
    t[3] = Float32(1.5)
    var result = relu(t)
    assert_true(result.numel() == 4, "numel preserved")
    assert_almost_equal(Float64(result[0]), 0.0, atol=1e-6)
    assert_almost_equal(Float64(result[1]), 0.0, atol=1e-6)
    assert_almost_equal(Float64(result[2]), 0.0, atol=1e-6)
    assert_almost_equal(Float64(result[3]), 1.5, atol=1e-6)
    print("PASS: test_relu_typed")


fn test_sigmoid_typed() raises:
    """Sigmoid typed overload of 0 is 0.5."""
    var t = Tensor[DType.float32]([1])
    t[0] = Float32(0.0)
    var result = sigmoid(t)
    assert_true(result.numel() == 1, "numel preserved")
    assert_almost_equal(Float64(result[0]), 0.5, atol=1e-6)
    print("PASS: test_sigmoid_typed")


fn main() raises:
    test_exp_typed()
    test_log_typed()
    test_sqrt_typed()
    test_relu_typed()
    test_sigmoid_typed()
