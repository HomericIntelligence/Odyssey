# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_anytensor_unary_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor __abs__ operator and combined unary/binary operations."""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


def test_abs_positive_values() raises:
    """Test __abs__: absolute value of positive numbers."""
    var a = full([2, 3], 3.5, DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.5, tolerance=1e-6
        )


def test_abs_negative_values() raises:
    """Test __abs__: absolute value of negative numbers."""
    var a = full([2, 3], -3.5, DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.5, tolerance=1e-6
        )


def test_abs_mixed_values() raises:
    """Test __abs__: absolute value with mixed positive/negative."""
    var a = zeros([4], DType.float32)
    a._set_float32(0, Float32(2.0))
    a._set_float32(1, Float32(-3.0))
    a._set_float32(2, Float32(0.0))
    a._set_float32(3, Float32(-1.5))
    var result = a.__abs__()
    assert_almost_equal(Float64(result._get_float32(0)), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(1)), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(2)), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(3)), 1.5, tolerance=1e-6)


def test_abs_zeros() raises:
    """Test __abs__: absolute value of zeros."""
    var a = zeros([2, 3], DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 0.0, tolerance=1e-6
        )


def test_combined_unary_binary_ops() raises:
    """Test combining unary and binary operators."""
    var a = full([2, 2], 2.0, DType.float32)
    var b = full([2, 2], -3.0, DType.float32)
    var abs_a = a.__abs__()
    var abs_b = b.__abs__()
    var result = abs_a + abs_b
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 5.0, tolerance=1e-6
        )


def test_double_negation() raises:
    """Test double negation: -(-a) == a."""
    var a = full([2, 2], 3.0, DType.float32)
    var result = -(-a)
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.0, tolerance=1e-6
        )


def test_operators_preserve_shape() raises:
    """Test that all operators preserve tensor shape."""
    var shape: List[Int] = [3, 4, 2]
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var add_result = a + b
    assert_equal(len(add_result.shape()), 3)
    var c = zeros(shape, DType.float32)
    c += b
    assert_equal(len(c.shape()), 3)
    var neg_result = -a
    assert_equal(len(neg_result.shape()), 3)
    var pos_result = +a
    assert_equal(len(pos_result.shape()), 3)
    var abs_result = a.__abs__()
    assert_equal(len(abs_result.shape()), 3)


def main() raises:
    """Run all __abs__ and combined operator tests."""
    test_abs_positive_values()
    test_abs_negative_values()
    test_abs_mixed_values()
    test_abs_zeros()
    test_combined_unary_binary_ops()
    test_double_negation()
    test_operators_preserve_shape()
    print("All abs and combined operator tests passed!")
