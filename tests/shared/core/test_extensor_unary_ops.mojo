"""Tests for ExTensor unary operators (__neg__, __pos__, __abs__) and combinations.

Note: Split from test_extensor_operators.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_neg_basic() raises:
    """Test __neg__: negation -tensor."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), -3.0, tolerance=1e-6)


fn test_neg_negative_values() raises:
    """Test __neg__: negation of negative values."""
    var a = full([2, 3], -5.0, DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 5.0, tolerance=1e-6)


fn test_neg_zeros() raises:
    """Test __neg__: negation of zeros."""
    var a = zeros([2, 3], DType.float32)
    var result = -a
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 0.0, tolerance=1e-6)


fn test_pos_basic() raises:
    """Test __pos__: positive +tensor (returns copy)."""
    var a = full([2, 3], 3.0, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 3.0, tolerance=1e-6)
    assert_equal(result.numel(), a.numel())


fn test_pos_preserves_values() raises:
    """Test __pos__: positive preserves values including negative."""
    var a = full([3, 2], -2.5, DType.float32)
    var result = +a
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), -2.5, tolerance=1e-6)


fn test_abs_positive_values() raises:
    """Test __abs__: absolute value of positive numbers."""
    var a = full([2, 3], 3.5, DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 3.5, tolerance=1e-6)


fn test_abs_negative_values() raises:
    """Test __abs__: absolute value of negative numbers."""
    var a = full([2, 3], -3.5, DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 3.5, tolerance=1e-6)


fn test_abs_mixed_values() raises:
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


fn test_abs_zeros() raises:
    """Test __abs__: absolute value of zeros."""
    var a = zeros([2, 3], DType.float32)
    var result = a.__abs__()
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 0.0, tolerance=1e-6)


fn test_combined_unary_binary_ops() raises:
    """Test combining unary and binary operators."""
    var a = full([2, 2], 2.0, DType.float32)
    var b = full([2, 2], -3.0, DType.float32)
    var abs_a = a.__abs__()
    var abs_b = b.__abs__()
    var result = abs_a + abs_b
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 5.0, tolerance=1e-6)


fn test_double_negation() raises:
    """Test double negation: -(-a) == a."""
    var a = full([2, 2], 3.0, DType.float32)
    var result = -(-a)
    for i in range(result.numel()):
        assert_almost_equal(Float64(result._get_float32(i)), 3.0, tolerance=1e-6)


fn test_operators_preserve_shape() raises:
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


fn main() raises:
    """Run unary and combined operator tests."""
    test_neg_basic()
    test_neg_negative_values()
    test_neg_zeros()
    test_pos_basic()
    test_pos_preserves_values()
    test_abs_positive_values()
    test_abs_negative_values()
    test_abs_mixed_values()
    test_abs_zeros()
    test_combined_unary_binary_ops()
    test_double_negation()
    test_operators_preserve_shape()
    print("All unary operator tests passed!")
