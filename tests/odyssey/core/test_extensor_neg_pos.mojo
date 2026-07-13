"""Tests for AnyTensor __neg__ and __pos__ unary operators."""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, full
from tests.odyssey.conftest import (
    assert_true,
    assert_almost_equal,
    assert_equal,
)


def test_neg_basic() raises:
    """Test __neg__: negation -tensor."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -3.0, tolerance=1e-6
        )


def test_neg_negative_values() raises:
    """Test __neg__: negation of negative values."""
    var a = full([2, 3], -5.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 5.0, tolerance=1e-6
        )


def test_neg_zeros() raises:
    """Test __neg__: negation of zeros."""
    var a = zeros([2, 3], DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 0.0, tolerance=1e-6
        )


def test_pos_basic() raises:
    """Test __pos__: positive +tensor (returns copy)."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.0, tolerance=1e-6
        )
    assert_equal(result.numel(), a.numel())


def test_pos_preserves_values() raises:
    """Test __pos__: positive preserves values including negative."""
    var a = full([3, 2], -2.5, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -2.5, tolerance=1e-6
        )


def test_unary_ops_preserve_shape() raises:
    """Test that unary operators preserve tensor shape."""
    var shape: List[Int] = [3, 4, 2]
    var a = zeros(shape, DType.float32)
    var neg_result = -a
    assert_equal(len(neg_result.shape()), 3)
    var pos_result = +a
    assert_equal(len(pos_result.shape()), 3)


def main() raises:
    """Run all __neg__ and __pos__ operator tests."""
    test_neg_basic()
    test_neg_negative_values()
    test_neg_zeros()
    test_pos_basic()
    test_pos_preserves_values()
    test_unary_ops_preserve_shape()
    print("All neg/pos operator tests passed!")
