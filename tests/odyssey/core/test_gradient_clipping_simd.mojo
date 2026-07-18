"""Tests for SIMD-vectorized gradient clipping functions."""

from odyssey.core import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from tests.odyssey.conftest import (
    assert_true,
)
from std.math import sqrt, abs
from std.collections import List


def test_clip_grad_value_f32() raises:
    """Test clip_grad_value_ with float32."""
    var grad = zeros([4], DType.float32)
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 2.0)
    grad._set_float64(2, 3.0)
    grad._set_float64(3, -4.0)
    clip_grad_value_(grad, 2.5)

    assert_true(abs(grad._get_float64(0) - 1.0) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 2.0) < 1e-6)
    assert_true(abs(grad._get_float64(2) - 2.5) < 1e-6)
    assert_true(abs(grad._get_float64(3) - (-2.5)) < 1e-6)


def test_clip_grad_value_f64() raises:
    """Test clip_grad_value_ with float64."""
    var grad = zeros([4], DType.float64)
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 2.0)
    grad._set_float64(2, 3.0)
    grad._set_float64(3, -4.0)
    clip_grad_value_(grad, 2.5)

    assert_true(abs(grad._get_float64(0) - 1.0) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 2.0) < 1e-6)
    assert_true(abs(grad._get_float64(2) - 2.5) < 1e-6)
    assert_true(abs(grad._get_float64(3) - (-2.5)) < 1e-6)


def test_clip_grad_norm_f32() raises:
    """Test clip_grad_norm_ with float32."""
    var grad = zeros([2], DType.float32)
    grad._set_float64(0, 3.0)
    grad._set_float64(1, 4.0)
    var norm = clip_grad_norm_(grad, 2.5)

    # Original norm: sqrt(9 + 16) = 5.0
    assert_true(abs(norm - 5.0) < 1e-6)

    # After clipping: scale by 2.5/5.0 = 0.5
    assert_true(abs(grad._get_float64(0) - 1.5) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 2.0) < 1e-6)


def test_clip_grad_norm_f64() raises:
    """Test clip_grad_norm_ with float64."""
    var grad = zeros([2], DType.float64)
    grad._set_float64(0, 3.0)
    grad._set_float64(1, 4.0)
    var norm = clip_grad_norm_(grad, 2.5)

    # Original norm: sqrt(9 + 16) = 5.0
    assert_true(abs(norm - 5.0) < 1e-6)

    # After clipping: scale by 2.5/5.0 = 0.5
    assert_true(abs(grad._get_float64(0) - 1.5) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 2.0) < 1e-6)


def test_clip_grad_global_norm_f32() raises:
    """Test clip_grad_global_norm_ with float32."""
    var grad1 = zeros([2], DType.float32)
    grad1._set_float64(0, 3.0)
    grad1._set_float64(1, 4.0)
    var grad2 = zeros([2], DType.float32)
    var grads: List[AnyTensor] = []
    grads.append(grad1)
    grads.append(grad2)

    var global_norm = clip_grad_global_norm_(grads, 2.5)

    # Original norm: sqrt(9 + 16 + 0 + 0) = 5.0
    assert_true(abs(global_norm - 5.0) < 1e-6)

    # After clipping: scale by 2.5/5.0 = 0.5
    assert_true(abs(grads[0]._get_float64(0) - 1.5) < 1e-6)
    assert_true(abs(grads[0]._get_float64(1) - 2.0) < 1e-6)


def test_clip_grad_global_norm_f64() raises:
    """Test clip_grad_global_norm_ with float64."""
    var grad1 = zeros([2], DType.float64)
    grad1._set_float64(0, 3.0)
    grad1._set_float64(1, 4.0)
    var grad2 = zeros([2], DType.float64)
    var grads: List[AnyTensor] = []
    grads.append(grad1)
    grads.append(grad2)

    var global_norm = clip_grad_global_norm_(grads, 2.5)

    # Original norm: sqrt(9 + 16 + 0 + 0) = 5.0
    assert_true(abs(global_norm - 5.0) < 1e-6)

    # After clipping: scale by 2.5/5.0 = 0.5
    assert_true(abs(grads[0]._get_float64(0) - 1.5) < 1e-6)
    assert_true(abs(grads[0]._get_float64(1) - 2.0) < 1e-6)


def test_clip_grad_value_no_clipping_f32() raises:
    """Test clip_grad_value_ when clipping is not needed (float32)."""
    var grad = zeros([3], DType.float32)
    grad._set_float64(0, 0.5)
    grad._set_float64(1, 1.0)
    grad._set_float64(2, 1.5)
    clip_grad_value_(grad, 2.0)

    assert_true(abs(grad._get_float64(0) - 0.5) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 1.0) < 1e-6)
    assert_true(abs(grad._get_float64(2) - 1.5) < 1e-6)


def test_clip_grad_norm_no_clipping_f32() raises:
    """Test clip_grad_norm_ when clipping is not needed (float32)."""
    var grad = zeros([2], DType.float32)
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 2.0)
    var norm = clip_grad_norm_(grad, 10.0)

    # norm = sqrt(1 + 4) = sqrt(5) ≈ 2.236
    var expected_norm = sqrt(5.0)
    assert_true(abs(norm - expected_norm) < 1e-6)

    # No clipping should occur
    assert_true(abs(grad._get_float64(0) - 1.0) < 1e-6)
    assert_true(abs(grad._get_float64(1) - 2.0) < 1e-6)


def test_clip_grad_global_norm_no_clipping_f64() raises:
    """Test clip_grad_global_norm_ when clipping is not needed (float64)."""
    var grad1 = zeros([2], DType.float64)
    grad1._set_float64(0, 1.0)
    grad1._set_float64(1, 2.0)
    var grad2 = zeros([2], DType.float64)
    var grads: List[AnyTensor] = []
    grads.append(grad1)
    grads.append(grad2)

    var global_norm = clip_grad_global_norm_(grads, 10.0)

    var expected_norm = sqrt(5.0)
    assert_true(abs(global_norm - expected_norm) < 1e-6)

    # No clipping should occur
    assert_true(abs(grads[0]._get_float64(0) - 1.0) < 1e-6)
    assert_true(abs(grads[0]._get_float64(1) - 2.0) < 1e-6)


def main() raises:
    """Run all tests."""
    test_clip_grad_value_f32()
    test_clip_grad_value_f64()
    test_clip_grad_norm_f32()
    test_clip_grad_norm_f64()
    test_clip_grad_global_norm_f32()
    test_clip_grad_global_norm_f64()
    test_clip_grad_value_no_clipping_f32()
    test_clip_grad_norm_no_clipping_f32()
    test_clip_grad_global_norm_no_clipping_f64()
    print("✓ All SIMD gradient clipping tests passed")
