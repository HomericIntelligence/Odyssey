"""Unit tests for ExTensor operator overloads (#2386).

Tests cover:
- Reflected operators (__radd__, __rsub__, __rmul__, __rtruediv__)
- In-place operators (__iadd__, __isub__, __imul__, __itruediv__)
- Unary operators (__neg__, __pos__, __abs__)

Following TDD principles - these tests verify operator implementation correctness.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Reflected Operators Tests
# ============================================================================


fn test_radd_tensors() raises:
    """Test __radd__: reflected addition a + b = b + a (commutative)."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    # Both should produce the same result (commutative)
    var result1 = a + b
    var result2 = b + a

    # Verify both tensors have same shape
    assert_equal(len(result1.shape()), len(result2.shape()))
    assert_equal(result1.numel(), result2.numel())

    # Verify values: 2.0 + 3.0 = 5.0
    for i in range(result1.numel()):
        assert_almost_equal(
            Float64(result1._get_float32(i)), 5.0, tolerance=1e-6
        )
        assert_almost_equal(
            Float64(result2._get_float32(i)), 5.0, tolerance=1e-6
        )


fn test_rsub_tensors() raises:
    """Test __rsub__: reflected subtraction (order matters)."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 5.0, DType.float32)

    # Different orders should give different results (non-commutative)
    var result1 = a - b  # 2.0 - 5.0 = -3.0
    var result2 = b - a  # 5.0 - 2.0 = 3.0

    # Verify values
    for i in range(result1.numel()):
        assert_almost_equal(
            Float64(result1._get_float32(i)), -3.0, tolerance=1e-6
        )
        assert_almost_equal(
            Float64(result2._get_float32(i)), 3.0, tolerance=1e-6
        )


fn test_rmul_tensors() raises:
    """Test __rmul__: reflected multiplication (commutative)."""
    var a = full([3, 2], 2.0, DType.float32)
    var b = full([3, 2], 3.0, DType.float32)

    # Both should produce the same result (commutative)
    var result1 = a * b
    var result2 = b * a

    # Verify values: 2.0 * 3.0 = 6.0
    for i in range(result1.numel()):
        assert_almost_equal(
            Float64(result1._get_float32(i)), 6.0, tolerance=1e-6
        )
        assert_almost_equal(
            Float64(result2._get_float32(i)), 6.0, tolerance=1e-6
        )


fn test_rtruediv_tensors() raises:
    """Test __rtruediv__: reflected division (order matters)."""
    var a = full([2, 2], 2.0, DType.float32)
    var b = full([2, 2], 8.0, DType.float32)

    # Different orders should give different results (non-commutative)
    var result1 = a / b  # 2.0 / 8.0 = 0.25
    var result2 = b / a  # 8.0 / 2.0 = 4.0

    # Verify values
    for i in range(result1.numel()):
        assert_almost_equal(
            Float64(result1._get_float32(i)), 0.25, tolerance=1e-6
        )
        assert_almost_equal(
            Float64(result2._get_float32(i)), 4.0, tolerance=1e-6
        )


# ============================================================================
# In-Place Operators Tests
# ============================================================================


fn test_iadd_basic() raises:
    """Test __iadd__: in-place addition tensor += other."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    # In-place add
    a += b

    # Verify result: 2.0 + 3.0 = 5.0
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 5.0, tolerance=1e-6)


fn test_isub_basic() raises:
    """Test __isub__: in-place subtraction tensor -= other."""
    var a = full([2, 3], 5.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)

    # In-place subtract
    a -= b

    # Verify result: 5.0 - 2.0 = 3.0
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 3.0, tolerance=1e-6)


fn test_imul_basic() raises:
    """Test __imul__: in-place multiplication tensor *= other."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    # In-place multiply
    a *= b

    # Verify result: 2.0 * 3.0 = 6.0
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 6.0, tolerance=1e-6)


fn test_itruediv_basic() raises:
    """Test __itruediv__: in-place division tensor /= other."""
    var a = full([2, 3], 8.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)

    # In-place divide
    a /= b

    # Verify result: 8.0 / 2.0 = 4.0
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 4.0, tolerance=1e-6)


fn test_inplace_operators_chain() raises:
    """Test chaining multiple in-place operators."""
    var a = full([2, 2], 10.0, DType.float32)
    var b = full([2, 2], 2.0, DType.float32)
    var c = full([2, 2], 3.0, DType.float32)

    # Chain operations: a = ((10 / 2) * 3) = 15
    a /= b
    a *= c

    # Verify result: 15.0
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 15.0, tolerance=1e-6)


# ============================================================================
# Unary Operators Tests
# ============================================================================


fn test_neg_basic() raises:
    """Test __neg__: negation -tensor."""
    var a = full([2, 3], 3.0, DType.float32)

    # Negate
    var result = -a

    # Verify result: -3.0
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -3.0, tolerance=1e-6
        )


fn test_neg_negative_values() raises:
    """Test __neg__: negation of negative values."""
    var a = full([2, 3], -5.0, DType.float32)

    # Negate negative value
    var result = -a

    # Verify result: 5.0
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 5.0, tolerance=1e-6
        )


fn test_neg_zeros() raises:
    """Test __neg__: negation of zeros."""
    var a = zeros([2, 3], DType.float32)

    # Negate zeros
    var result = -a

    # Verify result: -0.0 (still zero in most implementations)
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 0.0, tolerance=1e-6
        )


fn test_pos_basic() raises:
    """Test __pos__: positive +tensor (returns copy)."""
    var a = full([2, 3], 3.0, DType.float32)

    # Positive (copy)
    var result = +a

    # Verify result is equal to original
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.0, tolerance=1e-6
        )

    # Verify shapes match
    assert_equal(result.numel(), a.numel())


fn test_pos_preserves_values() raises:
    """Test __pos__: positive preserves values including negative."""
    var a = full([3, 2], -2.5, DType.float32)

    # Positive (copy)
    var result = +a

    # Verify result preserves negative values
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), -2.5, tolerance=1e-6
        )


fn test_abs_positive_values() raises:
    """Test __abs__: absolute value of positive numbers."""
    var a = full([2, 3], 3.5, DType.float32)

    # Absolute value
    var result = a.__abs__()

    # Verify result: 3.5
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.5, tolerance=1e-6
        )


fn test_abs_negative_values() raises:
    """Test __abs__: absolute value of negative numbers."""
    var a = full([2, 3], -3.5, DType.float32)

    # Absolute value
    var result = a.__abs__()

    # Verify result: 3.5
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.5, tolerance=1e-6
        )


fn test_abs_mixed_values() raises:
    """Test __abs__: absolute value with mixed positive/negative."""
    var a = zeros([4], DType.float32)

    # Set mixed values
    a._set_float32(0, Float32(2.0))
    a._set_float32(1, Float32(-3.0))
    a._set_float32(2, Float32(0.0))
    a._set_float32(3, Float32(-1.5))

    # Absolute value
    var result = a.__abs__()

    # Verify results
    assert_almost_equal(Float64(result._get_float32(0)), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(1)), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(2)), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(result._get_float32(3)), 1.5, tolerance=1e-6)


fn test_abs_zeros() raises:
    """Test __abs__: absolute value of zeros."""
    var a = zeros([2, 3], DType.float32)

    # Absolute value
    var result = a.__abs__()

    # Verify result: still zeros
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 0.0, tolerance=1e-6
        )


# ============================================================================
# Combined Operator Tests
# ============================================================================


fn test_combined_unary_binary_ops() raises:
    """Test combining unary and binary operators."""
    var a = full([2, 2], 2.0, DType.float32)
    var b = full([2, 2], -3.0, DType.float32)

    # Compute: a.__abs__() + b.__abs__() = 2.0 + 3.0 = 5.0
    var abs_a = a.__abs__()
    var abs_b = b.__abs__()
    var result = abs_a + abs_b

    # Verify result
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 5.0, tolerance=1e-6
        )


fn test_double_negation() raises:
    """Test double negation: -(-a) == a."""
    var a = full([2, 2], 3.0, DType.float32)

    # Double negate
    var result = -(-a)

    # Verify result is back to original
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result._get_float32(i)), 3.0, tolerance=1e-6
        )


fn test_operators_preserve_shape() raises:
    """Test that all operators preserve tensor shape."""
    var shape: List[Int] = [3, 4, 2]
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Test reflected operators
    var add_result = a + b
    assert_equal(len(add_result.shape()), 3)

    # Test in-place operators
    var c = zeros(shape, DType.float32)
    c += b
    assert_equal(len(c.shape()), 3)

    # Test unary operators
    var neg_result = -a
    assert_equal(len(neg_result.shape()), 3)

    var pos_result = +a
    assert_equal(len(pos_result.shape()), 3)

    var abs_result = a.__abs__()
    assert_equal(len(abs_result.shape()), 3)


fn main() raises:
    """Run all tests."""
    test_radd_tensors()
    test_rsub_tensors()
    test_rmul_tensors()
    test_rtruediv_tensors()
    test_iadd_basic()
    test_isub_basic()
    test_imul_basic()
    test_itruediv_basic()
    test_inplace_operators_chain()
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
