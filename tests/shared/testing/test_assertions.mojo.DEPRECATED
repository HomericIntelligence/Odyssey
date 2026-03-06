"""Tests for shared.testing.assertions module.

Comprehensive unit tests for all assertion functions used throughout the test suite.
Tests verify correct behavior for both passing and failing assertions.
"""

from testing import assert_true, assert_equal
from shared.testing.assertions import (
    assert_true as custom_assert_true,
    assert_false,
    assert_equal as custom_assert_equal,
    assert_not_equal,
    assert_not_none,
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_shape_equal,
    assert_not_equal_tensor,
    assert_tensor_equal,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_type,
)
from shared.core import ones, zeros, full
from collections.optional import Optional


# ============================================================================
# Basic Boolean Assertion Tests
# ============================================================================


fn test_assert_true_passes() raises:
    """Test assert_true with true condition."""
    # custom_assert_true doesn't return a value, it just raises on failure
    custom_assert_true(True)
    # If we get here, the assertion passed


fn test_assert_true_fails() raises:
    """Test assert_true with false condition."""
    var failed = False
    try:
        custom_assert_true(False)
    except:
        failed = True
    assert_true(failed, "assert_true should raise error on false condition")


fn test_assert_true_custom_message() raises:
    """Test assert_true with custom error message."""
    var failed = False
    var caught_message = False
    try:
        custom_assert_true(False, "Custom error message")
    except e:
        failed = True
        var msg = String(e)
        caught_message = "Custom error message" in msg
    assert_true(failed, "assert_true should raise error")
    assert_true(caught_message, "Error message should contain custom text")


fn test_assert_false_passes() raises:
    """Test assert_false with false condition."""
    assert_false(False)


fn test_assert_false_fails() raises:
    """Test assert_false with true condition."""
    var failed = False
    try:
        assert_false(True)
    except:
        failed = True
    assert_true(failed, "assert_false should raise error on true condition")


# ============================================================================
# Generic Equality Assertion Tests
# ============================================================================


fn test_assert_equal_int_passes() raises:
    """Test assert_equal with equal integers."""
    custom_assert_equal[Int](5, 5)


fn test_assert_equal_int_fails() raises:
    """Test assert_equal with unequal integers."""
    var failed = False
    try:
        custom_assert_equal[Int](5, 3)
    except:
        failed = True
    assert_true(failed, "assert_equal should raise error on unequal values")


fn test_assert_equal_string_passes() raises:
    """Test assert_equal with equal strings."""
    custom_assert_equal[String]("hello", "hello")


fn test_assert_equal_string_fails() raises:
    """Test assert_equal with unequal strings."""
    var failed = False
    try:
        custom_assert_equal[String]("hello", "world")
    except:
        failed = True
    assert_true(failed, "assert_equal should raise error on unequal strings")


fn test_assert_not_equal_int_passes() raises:
    """Test assert_not_equal with different integers."""
    assert_not_equal[Int](5, 3)


fn test_assert_not_equal_int_fails() raises:
    """Test assert_not_equal with equal integers."""
    var failed = False
    try:
        assert_not_equal[Int](5, 5)
    except:
        failed = True
    assert_true(failed, "assert_not_equal should raise error on equal values")


fn test_assert_not_none_passes() raises:
    """Test assert_not_none with some value."""
    var opt: Optional[Int] = Optional[Int](42)
    assert_not_none[Int](opt)


fn test_assert_not_none_fails() raises:
    """Test assert_not_none with none value."""
    var opt: Optional[Int] = Optional[Int]()
    var failed = False
    try:
        assert_not_none[Int](opt)
    except:
        failed = True
    assert_true(failed, "assert_not_none should raise error on None value")


# ============================================================================
# Floating-Point Comparison Tests
# ============================================================================


fn test_assert_almost_equal_float32_passes() raises:
    """Test assert_almost_equal with close Float32 values."""
    assert_almost_equal(
        Float32(1.0), Float32(1.0000001), tolerance=Float32(1e-5)
    )


fn test_assert_almost_equal_float32_fails() raises:
    """Test assert_almost_equal with distant Float32 values."""
    var failed = False
    try:
        assert_almost_equal(Float32(1.0), Float32(2.0), tolerance=Float32(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


fn test_assert_almost_equal_float64_passes() raises:
    """Test assert_almost_equal with close Float64 values."""
    assert_almost_equal(
        Float64(1.0), Float64(1.0000001), tolerance=Float64(1e-5)
    )


fn test_assert_almost_equal_float64_fails() raises:
    """Test assert_almost_equal with distant Float64 values."""
    var failed = False
    try:
        assert_almost_equal(Float64(1.0), Float64(2.0), tolerance=Float64(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


fn test_assert_dtype_equal_passes() raises:
    """Test assert_dtype_equal with matching dtypes."""
    assert_dtype_equal(DType.float32, DType.float32)


fn test_assert_dtype_equal_fails() raises:
    """Test assert_dtype_equal with mismatched dtypes."""
    var failed = False
    try:
        assert_dtype_equal(DType.float32, DType.float64)
    except:
        failed = True
    assert_true(
        failed, "assert_dtype_equal should raise error on mismatched dtypes"
    )


fn test_assert_equal_int_specialized_passes() raises:
    """Test assert_equal_int with matching integers."""
    assert_equal_int(42, 42)


fn test_assert_equal_int_specialized_fails() raises:
    """Test assert_equal_int with mismatched integers."""
    var failed = False
    try:
        assert_equal_int(42, 43)
    except:
        failed = True
    assert_true(failed, "assert_equal_int should raise error on mismatch")


fn test_assert_equal_float_passes() raises:
    """Test assert_equal_float with exactly equal floats."""
    assert_equal_float(Float32(1.0), Float32(1.0))


fn test_assert_equal_float_fails() raises:
    """Test assert_equal_float with different floats."""
    var failed = False
    try:
        assert_equal_float(Float32(1.0), Float32(1.1))
    except:
        failed = True
    assert_true(
        failed, "assert_equal_float should raise error on different values"
    )


fn test_assert_close_float_passes() raises:
    """Test assert_close_float with numerically close values."""
    assert_close_float(1.0, 1.00001, rtol=1e-3, atol=1e-3)


fn test_assert_close_float_fails() raises:
    """Test assert_close_float with distant values."""
    var failed = False
    try:
        assert_close_float(1.0, 10.0, rtol=1e-3, atol=1e-3)
    except:
        failed = True
    assert_true(
        failed, "assert_close_float should raise error for distant values"
    )


# ============================================================================
# Comparison Assertion Tests
# ============================================================================


fn test_assert_greater_float32_passes() raises:
    """Test assert_greater with a > b (Float32)."""
    assert_greater(Float32(2.0), Float32(1.0))


fn test_assert_greater_float32_fails() raises:
    """Test assert_greater with a <= b (Float32)."""
    var failed = False
    try:
        assert_greater(Float32(1.0), Float32(2.0))
    except:
        failed = True
    assert_true(failed, "assert_greater should raise error when a <= b")


fn test_assert_greater_float64_passes() raises:
    """Test assert_greater with a > b (Float64)."""
    assert_greater(Float64(2.0), Float64(1.0))


fn test_assert_greater_float64_fails() raises:
    """Test assert_greater with a <= b (Float64)."""
    var failed = False
    try:
        assert_greater(Float64(1.0), Float64(2.0))
    except:
        failed = True
    assert_true(failed, "assert_greater should raise error when a <= b")


fn test_assert_greater_int_passes() raises:
    """Test assert_greater with a > b (Int)."""
    assert_greater(2, 1)


fn test_assert_greater_int_fails() raises:
    """Test assert_greater with a <= b (Int)."""
    var failed = False
    try:
        assert_greater(1, 2)
    except:
        failed = True
    assert_true(failed, "assert_greater should raise error when a <= b")


fn test_assert_less_float32_passes() raises:
    """Test assert_less with a < b (Float32)."""
    assert_less(Float32(1.0), Float32(2.0))


fn test_assert_less_float32_fails() raises:
    """Test assert_less with a >= b (Float32)."""
    var failed = False
    try:
        assert_less(Float32(2.0), Float32(1.0))
    except:
        failed = True
    assert_true(failed, "assert_less should raise error when a >= b")


fn test_assert_greater_or_equal_float32_passes() raises:
    """Test assert_greater_or_equal with a >= b (Float32)."""
    assert_greater_or_equal(Float32(2.0), Float32(1.0))
    assert_greater_or_equal(Float32(1.0), Float32(1.0))


fn test_assert_greater_or_equal_float32_fails() raises:
    """Test assert_greater_or_equal with a < b (Float32)."""
    var failed = False
    try:
        assert_greater_or_equal(Float32(1.0), Float32(2.0))
    except:
        failed = True
    assert_true(failed, "assert_greater_or_equal should raise error when a < b")


fn test_assert_less_or_equal_float32_passes() raises:
    """Test assert_less_or_equal with a <= b (Float32)."""
    assert_less_or_equal(Float32(1.0), Float32(2.0))
    assert_less_or_equal(Float32(1.0), Float32(1.0))


fn test_assert_less_or_equal_float32_fails() raises:
    """Test assert_less_or_equal with a > b (Float32)."""
    var failed = False
    try:
        assert_less_or_equal(Float32(2.0), Float32(1.0))
    except:
        failed = True
    assert_true(failed, "assert_less_or_equal should raise error when a > b")


# ============================================================================
# Shape and List Assertion Tests
# ============================================================================


fn test_assert_shape_equal_passes() raises:
    """Test assert_shape_equal with matching shapes."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 3, 4]
    assert_shape_equal(shape1, shape2)


fn test_assert_shape_equal_fails_dimension() raises:
    """Test assert_shape_equal with different dimension count."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 3]
    var failed = False
    try:
        assert_shape_equal(shape1, shape2)
    except:
        failed = True
    assert_true(
        failed, "assert_shape_equal should raise error on dimension mismatch"
    )


fn test_assert_shape_equal_fails_size() raises:
    """Test assert_shape_equal with different dimension sizes."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 5, 4]
    var failed = False
    try:
        assert_shape_equal(shape1, shape2)
    except:
        failed = True
    assert_true(
        failed, "assert_shape_equal should raise error on size mismatch"
    )


# ============================================================================
# Tensor Assertion Tests
# ============================================================================


fn test_assert_shape_tensor_passes() raises:
    """Test assert_shape with matching tensor shape."""
    var tensor = ones([3, 4], DType.float32)
    var expected: List[Int] = [3, 4]
    assert_shape(tensor, expected)


fn test_assert_shape_tensor_fails() raises:
    """Test assert_shape with mismatched tensor shape."""
    var tensor = ones([3, 4], DType.float32)
    var expected: List[Int] = [4, 5]
    var failed = False
    try:
        assert_shape(tensor, expected)
    except:
        failed = True
    assert_true(failed, "assert_shape should raise error on shape mismatch")


fn test_assert_dtype_tensor_passes() raises:
    """Test assert_dtype with matching dtype."""
    var tensor = ones([3, 4], DType.float32)
    assert_dtype(tensor, DType.float32)


fn test_assert_dtype_tensor_fails() raises:
    """Test assert_dtype with mismatched dtype."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dtype(tensor, DType.float64)
    except:
        failed = True
    assert_true(failed, "assert_dtype should raise error on dtype mismatch")


fn test_assert_numel_tensor_passes() raises:
    """Test assert_numel with matching element count."""
    var tensor = ones([3, 4], DType.float32)
    assert_numel(tensor, 12)  # 3 * 4 = 12


fn test_assert_numel_tensor_fails() raises:
    """Test assert_numel with mismatched element count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_numel(tensor, 10)
    except:
        failed = True
    assert_true(failed, "assert_numel should raise error on numel mismatch")


fn test_assert_dim_tensor_passes() raises:
    """Test assert_dim with matching dimension count."""
    var tensor = ones([3, 4, 5], DType.float32)
    assert_dim(tensor, 3)


fn test_assert_dim_tensor_fails() raises:
    """Test assert_dim with mismatched dimension count."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_dim(tensor, 3)
    except:
        failed = True
    assert_true(failed, "assert_dim should raise error on dimension mismatch")


fn test_assert_value_at_passes() raises:
    """Test assert_value_at with matching value."""
    var tensor = ones([3, 4], DType.float32)
    assert_value_at(tensor, 0, 1.0, tolerance=1e-6)


fn test_assert_value_at_fails() raises:
    """Test assert_value_at with non-matching value."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_value_at(tensor, 0, 2.0, tolerance=1e-6)
    except:
        failed = True
    assert_true(failed, "assert_value_at should raise error on value mismatch")


fn test_assert_all_values_passes() raises:
    """Test assert_all_values with all matching values."""
    var tensor = ones([3, 4], DType.float32)
    assert_all_values(tensor, 1.0, tolerance=1e-6)


fn test_assert_all_values_fails() raises:
    """Test assert_all_values with non-matching values."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_all_values(tensor, 2.0, tolerance=1e-6)
    except:
        failed = True
    assert_true(
        failed, "assert_all_values should raise error on value mismatch"
    )


fn test_assert_all_close_passes() raises:
    """Test assert_all_close with close tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 1.0000001, DType.float32)
    assert_all_close(tensor1, tensor2, tolerance=1e-5)


fn test_assert_all_close_fails() raises:
    """Test assert_all_close with distant tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    var failed = False
    try:
        assert_all_close(tensor1, tensor2, tolerance=1e-5)
    except:
        failed = True
    assert_true(
        failed, "assert_all_close should raise error on tensor mismatch"
    )


fn test_assert_tensor_equal_passes() raises:
    """Test assert_tensor_equal with equal tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([3, 4], DType.float32)
    assert_tensor_equal(tensor1, tensor2)


fn test_assert_tensor_equal_fails_shape() raises:
    """Test assert_tensor_equal with different shapes."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([4, 5], DType.float32)
    var failed = False
    try:
        assert_tensor_equal(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_tensor_equal should raise error on shape mismatch"
    )


fn test_assert_tensor_equal_fails_values() raises:
    """Test assert_tensor_equal with different values."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    var failed = False
    try:
        assert_tensor_equal(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_tensor_equal should raise error on value mismatch"
    )


fn test_assert_not_equal_tensor_passes() raises:
    """Test assert_not_equal_tensor with different tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    assert_not_equal_tensor(tensor1, tensor2)


fn test_assert_not_equal_tensor_fails() raises:
    """Test assert_not_equal_tensor with equal tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_not_equal_tensor(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_not_equal_tensor should raise error on equal tensors"
    )


# ============================================================================
# Type Checking Tests
# ============================================================================


fn test_assert_type_int() raises:
    """Test assert_type with int value."""
    var value: Int = 42
    assert_type[Int](value, "Int")


fn test_assert_type_float() raises:
    """Test assert_type with float value."""
    var value: Float32 = 3.14
    assert_type[Float32](value, "Float32")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all assertion tests."""
    # Boolean tests
    test_assert_true_passes()
    test_assert_true_fails()
    test_assert_true_custom_message()
    test_assert_false_passes()
    test_assert_false_fails()

    # Generic equality tests
    test_assert_equal_int_passes()
    test_assert_equal_int_fails()
    test_assert_equal_string_passes()
    test_assert_equal_string_fails()
    test_assert_not_equal_int_passes()
    test_assert_not_equal_int_fails()
    test_assert_not_none_passes()
    test_assert_not_none_fails()

    # Floating-point comparison tests
    test_assert_almost_equal_float32_passes()
    test_assert_almost_equal_float32_fails()
    test_assert_almost_equal_float64_passes()
    test_assert_almost_equal_float64_fails()
    test_assert_dtype_equal_passes()
    test_assert_dtype_equal_fails()
    test_assert_equal_int_passes()
    test_assert_equal_int_fails()
    test_assert_equal_int_specialized_passes()
    test_assert_equal_int_specialized_fails()
    test_assert_equal_float_passes()
    test_assert_equal_float_fails()
    test_assert_close_float_passes()
    test_assert_close_float_fails()

    # Comparison tests
    test_assert_greater_float32_passes()
    test_assert_greater_float32_fails()
    test_assert_greater_float64_passes()
    test_assert_greater_float64_fails()
    test_assert_greater_int_passes()
    test_assert_greater_int_fails()
    test_assert_less_float32_passes()
    test_assert_less_float32_fails()
    test_assert_greater_or_equal_float32_passes()
    test_assert_greater_or_equal_float32_fails()
    test_assert_less_or_equal_float32_passes()
    test_assert_less_or_equal_float32_fails()

    # Shape and list tests
    test_assert_shape_equal_passes()
    test_assert_shape_equal_fails_dimension()
    test_assert_shape_equal_fails_size()

    # Tensor tests
    test_assert_shape_tensor_passes()
    test_assert_shape_tensor_fails()
    test_assert_dtype_tensor_passes()
    test_assert_dtype_tensor_fails()
    test_assert_numel_tensor_passes()
    test_assert_numel_tensor_fails()
    test_assert_dim_tensor_passes()
    test_assert_dim_tensor_fails()
    test_assert_value_at_passes()
    test_assert_value_at_fails()
    test_assert_all_values_passes()
    test_assert_all_values_fails()
    test_assert_all_close_passes()
    test_assert_all_close_fails()
    test_assert_tensor_equal_passes()
    test_assert_tensor_equal_fails_shape()
    test_assert_tensor_equal_fails_values()
    test_assert_not_equal_tensor_passes()
    test_assert_not_equal_tensor_fails()

    # Type checking tests
    test_assert_type_int()
    test_assert_type_float()

    print("All assertion tests passed!")
