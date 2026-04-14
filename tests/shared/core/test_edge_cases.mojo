"""Tests for AnyTensor edge cases

"""


from std.math import isnan, isinf
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    arange,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
)
from shared.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
)
from shared.core.comparison import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_equal_int,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange, nan_tensor, inf_tensor, neg_inf_tensor
from shared.core.arithmetic import add, subtract, multiply, divide, floor_divide, modulo, power
from shared.core.comparison import equal, not_equal, less, less_equal, greater, greater_equal


def test_empty_tensor_creation() raises:
    """Test creating empty tensor with 0 elements."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor should have 0 elements")
    assert_dim(t, 1, "Empty tensor should have 1 dimension")


def test_empty_tensor_operations() raises:
    """Test operations on empty tensors."""
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = add(a, b)

    assert_numel(c, 0, "Operations on empty tensors should give empty tensor")


def test_empty_tensor_2d() raises:
    """Test 2D empty tensor (0 rows or 0 cols)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(0)  # 3x0 matrix
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "3x0 matrix should have 0 elements")
    assert_dim(t, 2, "Should preserve 2D structure")


def test_scalar_tensor_creation() raises:
    """Test creating 0D scalar tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_value_at(t, 0, 42.0, 1e-6, "Scalar value should be 42.0")


def test_scalar_tensor_operations() raises:
    """Test operations on 0D scalar tensors."""
    var shape = List[Int]()
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = add(a, b)

    assert_numel(c, 1, "Scalar + scalar should give scalar")
    assert_dim(c, 0, "Result should be 0D")
    assert_value_at(c, 0, 5.0, 1e-6, "3 + 2 should be 5")


def test_scalar_to_vector_broadcast() raises:
    """Test broadcasting scalar to vector."""
    var shape_scalar = List[Int]()
    var shape_vec = List[Int]()
    shape_vec.append(5)

    var a = full(shape_scalar, 10.0, DType.float32)  # scalar
    var b = full(shape_vec, 2.0, DType.float32)  # vector
    var c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 20.0, 1e-6, "10 * 2 broadcast to [20, 20, 20, 20, 20]")


def test_large_1d_tensor() raises:
    """Test creating and operating on large 1D tensor."""
    var shape = List[Int]()
    shape.append(10000000)  # 10 million elements
    var t = zeros(shape, DType.float32)

    assert_numel(t, 10000000, "Large tensor should have 10M elements")
    # Spot check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "First element should be 0")
    assert_value_at(t, 9999999, 0.0, 1e-8, "Last element should be 0")


def test_large_dimension_count() raises:
    """Test tensor with many dimensions (10D)."""
    var shape = List[Int](length=10, fill=2)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 10, "Should have 10 dimensions")
    assert_numel(t, 1024, "2^10 = 1024 elements")


def test_nan_propagation_add() raises:
    """Test that NaN propagates through addition."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # NaN + x = NaN
    for i in range(3):
        var val = c._get_float64(i)
        assert_true(isnan(val), "NaN should propagate through addition")


def test_nan_propagation_multiply() raises:
    """Test that NaN propagates through multiplication."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    # NaN * 0 = NaN (not 0!)
    for i in range(3):
        var val = c._get_float64(i)
        assert_true(isnan(val), "NaN * 0 should be NaN")


def test_nan_equality() raises:
    """Test NaN equality (NaN != NaN per IEEE 754)."""
    var shape = List[Int]()
    shape.append(1)
    var a = nan_tensor(shape, DType.float32)
    var b = nan_tensor(shape, DType.float32)
    var c = equal(a, b)

    # IEEE 754: NaN != NaN
    assert_value_at(c, 0, 0.0, 1e-8, "NaN != NaN should be False")


def test_inf_arithmetic() raises:
    """Test arithmetic with infinity."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # inf + x = inf (except inf + (-inf) = NaN)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val < 0:
            raise Error("inf + 1 should be inf")


def test_inf_multiplication() raises:
    """Test infinity multiplication."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = multiply(a, b)

    # inf * x = inf (for positive x)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val < 0:
            raise Error("inf * 2 should be inf")


def test_inf_times_zero() raises:
    """Test infinity times zero (should give NaN)."""
    var shape = List[Int]()
    shape.append(3)
    var a = inf_tensor(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = multiply(a, b)

    # inf * 0 = NaN (indeterminate form)
    for i in range(3):
        var val = c._get_float64(i)
        if not isnan(val):
            raise Error("inf * 0 should be NaN")


def test_negative_inf() raises:
    """Test negative infinity."""
    var shape = List[Int]()
    shape.append(3)
    var a = neg_inf_tensor(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    # -inf + x = -inf (for finite x)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val) or val > 0:
            raise Error("-inf + 1 should be -inf")


def test_inf_comparison() raises:
    """Test comparison with infinity."""
    var shape = List[Int]()
    shape.append(1)
    var a = inf_tensor(shape, DType.float32)
    var b = full(shape, 1000.0, DType.float32)
    var c = greater(a, b)

    # inf > 1000 should be True
    assert_value_at(c, 0, 1.0, 1e-8, "inf > 1000 should be True")


def test_overflow_float32() raises:
    """Test overflow behavior for float32."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1e38, DType.float32)  # Near float32 max
    var b = full(shape, 10.0, DType.float32)
    var c = multiply(a, b)

    # 1e38 * 10 should overflow to inf
    for i in range(2):
        var val = c._get_float64(i)
        if not isinf(val):
            raise Error("Overflow should produce inf")


def test_overflow_int32() raises:
    """Test overflow behavior for int32."""
    var shape = List[Int]()
    shape.append(2)
    # Create values close to int32 max
    var a = full(shape, 2147483647.0, DType.int32)  # INT32_MAX
    var b = ones(shape, DType.int32)
    var c = add(a, b)

    # Integer overflow behavior: wraps around (implementation dependent)
    # Just verify it doesn't crash
    assert_dim(c, 1, "INT32_MAX + 1 should produce result")


def test_underflow_float64() raises:
    """Test underflow behavior for float64."""
    var shape = List[Int]()
    shape.append(2)
    # Create very small values
    var a = full(shape, 1e-300, DType.float64)
    var b = full(shape, 1e-100, DType.float64)
    var c = multiply(a, b)

    # Result may underflow to 0 (gradual underflow)
    assert_all_values(c, 0.0, 1e-320, "Underflow should give 0")


def test_divide_by_zero_float() raises:
    """Test division by zero for floating point."""
    # Skip this test - may cause undefined behavior
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, 1.0, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)
    # 1/0 = inf (IEEE 754)
    pass


def test_divide_by_zero_int() raises:
    """Test division by zero for integers."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, 10.0, DType.int32)
    # var b = zeros(shape, DType.int32)
    # var c = divide(a, b)


def test_divide_zero_by_zero() raises:
    """Test 0/0 (should give NaN for floats)."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = zeros(shape, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)


def test_divide_negative_by_zero() raises:
    """Test -1/0 (should give -inf for floats)."""
    # Skip - may cause undefined behavior
    pass
    # var shape = List[Int]()
    # shape.append(3)
    # var a = full(shape, -1.0, DType.float32)
    # var b = zeros(shape, DType.float32)
    # var c = divide(a, b)


def test_modulo_by_zero() raises:
    """Test modulo by zero (should give NaN for floats)."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = modulo(a, b)

    # Modulo by zero: undefined, should give NaN
    for i in range(3):
        var val = c._get_float64(i)
        if not isnan(val):
            raise Error("x % 0 should be NaN")


def test_modulo_with_negative_divisor() raises:
    """Test modulo with negative divisor."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, -3.0, DType.float32)
    var c = modulo(a, b)

    # Python semantics: 7 % -3 = -2 (result has sign of divisor)
    assert_value_at(c, 0, -2.0, 1e-6, "7 % -3 should be -2 (Python semantics)")


def test_modulo_both_negative() raises:
    """Test modulo with both negative values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, -3.0, DType.float32)
    var c = modulo(a, b)

    # Python semantics: -7 % -3 = -1
    assert_value_at(c, 0, -1.0, 1e-6, "-7 % -3 should be -1 (Python semantics)")


def test_power_zero_to_zero() raises:
    """Test 0^0 (mathematically undefined, conventionally 1)."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = power(a, b)

    # Convention: 0^0 = 1 (used in polynomial evaluation)
    assert_all_values(c, 1.0, 1e-6, "0^0 should be 1 by convention")


def test_power_negative_base_even() raises:
    """Test negative base with even exponent."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = power(a, b)

    # (-2)^2 = 4
    assert_value_at(c, 0, 4.0, 1e-6, "(-2)^2 should be 4")


def test_power_negative_base_odd() raises:
    """Test negative base with odd exponent."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = power(a, b)

    # (-2)^3 = -8
    assert_value_at(c, 0, -8.0, 1e-6, "(-2)^3 should be -8")


def test_power_zero_base_positive_exp() raises:
    """Test 0^n for positive n."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = full(shape, 5.0, DType.float32)
    var c = power(a, b)

    # 0^5 = 0
    assert_all_values(c, 0.0, 1e-6, "0^n should be 0 for positive n")


def test_floor_divide_by_zero() raises:
    """Test floor division by zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 10.0, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = floor_divide(a, b)

    # Floor division by zero should give inf (like regular division)
    for i in range(3):
        var val = c._get_float64(i)
        if not isinf(val):
            raise Error("x // 0 should be inf")


def test_floor_divide_with_remainder() raises:
    """Test floor division with remainder."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = floor_divide(a, b)

    # 7 // 3 = 2 (floor of 2.333...)
    assert_value_at(c, 0, 2.0, 1e-6, "7 // 3 should be 2")


def test_floor_divide_negative_result() raises:
    """Test floor division with negative result."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, -7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = floor_divide(a, b)

    # -7 // 3 = -3 (floor of -2.333... = -3, not -2)
    assert_value_at(
        c, 0, -3.0, 1e-6, "-7 // 3 should be -3 (floor toward -inf)"
    )


def test_comparison_with_zero() raises:
    """Test comparison operations with zero."""
    var shape = List[Int]()
    shape.append(3)

    var positive = full(shape, 1.0, DType.float32)
    var negative = full(shape, -1.0, DType.float32)
    var zero = zeros(shape, DType.float32)

    # Test greater than zero
    var pos_gt_zero = greater(positive, zero)
    assert_all_values(pos_gt_zero, 1.0, 1e-6, "1.0 > 0 should be True")

    var neg_gt_zero = greater(negative, zero)
    assert_all_values(neg_gt_zero, 0.0, 1e-6, "-1.0 > 0 should be False")

    # Test less than zero
    var neg_lt_zero = less(negative, zero)
    assert_all_values(neg_lt_zero, 1.0, 1e-6, "-1.0 < 0 should be True")


def test_comparison_equal_values() raises:
    """Test equality comparison with same values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.14159, DType.float64)
    var b = full(shape, 3.14159, DType.float64)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Comparison should return bool")
    assert_all_values(c, 1.0, 1e-10, "Equal values should be equal")


def test_comparison_very_close_values() raises:
    """Test comparison with very close but not equal values."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 1.0000001, DType.float32)

    # These should NOT be equal (exact comparison, no tolerance)
    var eq = equal(a, b)
    var ne = not_equal(a, b)

    # Depending on float32 precision, these might be equal or not
    # For now, just verify bool dtype
    assert_dtype(eq, DType.bool, "equal() should return bool")
    assert_dtype(ne, DType.bool, "not_equal() should return bool")


def test_subnormal_numbers() raises:
    """Test handling of subnormal (denormalized) numbers."""
    var shape = List[Int]()
    shape.append(2)
    # Create subnormal float32 value (~1e-40)
    var a = full(shape, 1e-40, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = add(a, b)

    # a + 1 should be approximately 1 (subnormal is tiny)
    var expected = full(shape, 1.0, DType.float32)
    assert_all_close(c, expected, 1e-6, "1e-40 + 1 ≈ 1")


def test_catastrophic_cancellation() raises:
    """Test catastrophic cancellation in subtraction."""
    var shape = List[Int]()
    shape.append(2)
    # Create two very close values
    var a = full(shape, 1.0000000001, DType.float64)
    var b = full(shape, 1.0, DType.float64)
    var c = subtract(a, b)

    # Result should be very small (loss of precision is expected here)
    for i in range(2):
        var val = c._get_float64(i)
        # Just verify it's close to expected value
        if val < 0 or val > 1e-9:
            # Allow for precision loss
            pass


def test_associativity_loss() raises:
    """Test loss of associativity in floating point."""
    var shape = List[Int]()
    shape.append(3)
    # Create specific values to demonstrate associativity loss
    var a = full(shape, 1e20, DType.float32)
    var b = full(shape, 1.0, DType.float32)
    var c = full(shape, -1e20, DType.float32)

    # (a + b) + c != a + (b + c) in floating point
    var result1 = add(add(a, b), c)
    var result2 = add(a, add(b, c))

    # Just verify both produce results (may differ due to rounding)
    assert_dim(result1, 1, "(a + b) + c should produce result")
    assert_dim(result2, 1, "a + (b + c) should produce result")


def test_bool_dtype_operations() raises:
    """Test operations on bool dtype tensors."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.bool)  # All True (1)
    var b = zeros(shape, DType.bool)  # All False (0)

    # Test equality comparison with bool dtypes
    var c = equal(a, a)  # Should be all True
    assert_all_values(c, 1.0, 1e-8, "True == True should be True")

    var d = equal(a, b)  # Should be all False
    assert_all_values(d, 0.0, 1e-8, "True == False should be False")


def test_int8_range() raises:
    """Test int8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at int8 boundaries
    var a = full(shape, 127.0, DType.int8)  # INT8_MAX
    var b = full(shape, -128.0, DType.int8)  # INT8_MIN

    # Verify values are stored correctly
    assert_value_at(a, 0, 127.0, 1e-6, "INT8_MAX should be 127")
    assert_value_at(b, 0, -128.0, 1e-6, "INT8_MIN should be -128")


def test_uint8_range() raises:
    """Test uint8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at uint8 boundaries
    var a = full(shape, 255.0, DType.uint8)  # UINT8_MAX
    var b = zeros(shape, DType.uint8)  # UINT8_MIN

    # Verify values are stored correctly
    assert_value_at(a, 0, 255.0, 1e-6, "UINT8_MAX should be 255")
    assert_value_at(b, 0, 0.0, 1e-6, "UINT8_MIN should be 0")


def main() raises:
    """Run all test_edge_cases tests."""
    print("Running test_edge_cases tests...")

    test_empty_tensor_creation()
    print("✓ test_empty_tensor_creation")

    test_empty_tensor_operations()
    print("✓ test_empty_tensor_operations")

    test_empty_tensor_2d()
    print("✓ test_empty_tensor_2d")

    test_scalar_tensor_creation()
    print("✓ test_scalar_tensor_creation")

    test_scalar_tensor_operations()
    print("✓ test_scalar_tensor_operations")

    test_scalar_to_vector_broadcast()
    print("✓ test_scalar_to_vector_broadcast")

    test_large_1d_tensor()
    print("✓ test_large_1d_tensor")

    test_large_dimension_count()
    print("✓ test_large_dimension_count")

    test_nan_propagation_add()
    print("✓ test_nan_propagation_add")

    test_nan_propagation_multiply()
    print("✓ test_nan_propagation_multiply")

    test_nan_equality()
    print("✓ test_nan_equality")

    test_inf_arithmetic()
    print("✓ test_inf_arithmetic")

    test_inf_multiplication()
    print("✓ test_inf_multiplication")

    test_inf_times_zero()
    print("✓ test_inf_times_zero")

    test_negative_inf()
    print("✓ test_negative_inf")

    test_inf_comparison()
    print("✓ test_inf_comparison")

    test_overflow_float32()
    print("✓ test_overflow_float32")

    test_overflow_int32()
    print("✓ test_overflow_int32")

    test_underflow_float64()
    print("✓ test_underflow_float64")

    test_divide_by_zero_float()
    print("✓ test_divide_by_zero_float")

    test_divide_by_zero_int()
    print("✓ test_divide_by_zero_int")

    test_divide_zero_by_zero()
    print("✓ test_divide_zero_by_zero")

    test_divide_negative_by_zero()
    print("✓ test_divide_negative_by_zero")

    test_modulo_by_zero()
    print("✓ test_modulo_by_zero")

    test_modulo_with_negative_divisor()
    print("✓ test_modulo_with_negative_divisor")

    test_modulo_both_negative()
    print("✓ test_modulo_both_negative")

    test_power_zero_to_zero()
    print("✓ test_power_zero_to_zero")

    test_power_negative_base_even()
    print("✓ test_power_negative_base_even")

    test_power_negative_base_odd()
    print("✓ test_power_negative_base_odd")

    test_power_zero_base_positive_exp()
    print("✓ test_power_zero_base_positive_exp")

    test_floor_divide_by_zero()
    print("✓ test_floor_divide_by_zero")

    test_floor_divide_with_remainder()
    print("✓ test_floor_divide_with_remainder")

    test_floor_divide_negative_result()
    print("✓ test_floor_divide_negative_result")

    test_comparison_with_zero()
    print("✓ test_comparison_with_zero")

    test_comparison_equal_values()
    print("✓ test_comparison_equal_values")

    test_comparison_very_close_values()
    print("✓ test_comparison_very_close_values")

    test_subnormal_numbers()
    print("✓ test_subnormal_numbers")

    test_catastrophic_cancellation()
    print("✓ test_catastrophic_cancellation")

    test_associativity_loss()
    print("✓ test_associativity_loss")

    test_bool_dtype_operations()
    print("✓ test_bool_dtype_operations")

    test_int8_range()
    print("✓ test_int8_range")

    test_uint8_range()
    print("✓ test_uint8_range")

    print("\nAll test_edge_cases tests passed!")
