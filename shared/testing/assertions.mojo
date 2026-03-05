"""Assertion functions for testing neural network implementations.

This module provides a comprehensive collection of assertion functions for validating
neural network components, tensor operations, and training algorithms. These assertions
are used throughout the test suite for:

- Basic boolean and equality assertions
- Floating-point near-equality checks
- Tensor shape and data type validation
- Element-wise tensor comparisons
- Type checking helpers

Functions:
    assert_true: Assert condition is true
    assert_false: Assert condition is false
    assert_equal: Assert two values are equal (parametric)
    assert_not_equal: Assert two values are not equal (parametric)
    assert_not_none: Assert Optional value is not None
    assert_almost_equal: Assert floats are nearly equal (Float32/Float64)
    assert_dtype_equal: Assert DType values are equal
    assert_equal_int: Assert integers are equal
    assert_equal_float: Assert floats are exactly equal
    assert_close_float: Assert floats are numerically close with relative/absolute tolerance
    assert_greater: Assert a > b (parametric - Comparable & Stringable)
    assert_less: Assert a < b (parametric - Comparable & Stringable)
    assert_greater_or_equal: Assert a >= b (parametric - Comparable & Stringable)
    assert_less_or_equal: Assert a <= b (parametric - Comparable & Stringable)
    assert_shape_equal: Assert two shapes are equal
    assert_not_equal_tensor: Assert two tensors are not equal element-wise
    assert_tensor_equal: Assert two tensors are equal (shape and elements)
    assert_shape: Assert tensor has expected shape
    assert_dtype: Assert tensor has expected dtype
    assert_numel: Assert tensor has expected number of elements
    assert_dim: Assert tensor has expected number of dimensions
    assert_value_at: Assert tensor value at index matches expected
    assert_all_values: Assert all tensor values match expected constant
    assert_all_close: Assert two tensors are element-wise close
    assert_type: Assert value is of expected type (documentation)
"""

from math import isnan, isinf
from collections.optional import Optional
from shared.core.extensor import ExTensor
from shared.testing.tolerance_constants import (
    TOLERANCE_DEFAULT,
    TOLERANCE_FLOAT32,
    TOLERANCE_FLOAT64,
    TOLERANCE_GRADIENT_RTOL,
    TOLERANCE_GRADIENT_ATOL,
    TOLERANCE_CONV,
    TOLERANCE_SOFTMAX,
    TOLERANCE_CROSS_ENTROPY,
)


# ============================================================================
# Basic Boolean Assertions
# ============================================================================


fn assert_true(condition: Bool, message: String = "Assertion failed") raises:
    """Assert that condition is true.

    Args:
            condition: The boolean condition to check.
            message: Optional error message.

    Raises:
            Error: If condition is false.
    """
    if not condition:
        raise Error(message)


fn assert_false(condition: Bool, message: String = "Assertion failed") raises:
    """Assert that condition is false.

    Args:
            condition: The boolean condition to check.
            message: Optional error message.

    Raises:
            Error: If condition is true.
    """
    if condition:
        raise Error(message)


# ============================================================================
# Generic Equality Assertions
# ============================================================================


fn assert_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert exact equality of two values.

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a != b.
    """
    if a != b:
        var error_msg = message if message else "Values are not equal"
        raise Error(error_msg)


fn assert_not_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert inequality of two values.

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a == b.
    """
    if a == b:
        var error_msg = (
            message if message else "Values are equal but should not be"
        )
        raise Error(error_msg)


fn assert_not_none[
    T: Copyable & Movable
](value: Optional[T], message: String = "") raises:
    """Assert that an Optional value is not None.

    Args:
            value: The Optional value to check.
            message: Optional error message.

    Raises:
            Error: If value is None.
    """
    if not value:
        var error_msg = (
            message if message else "Value is None but should not be"
        )
        raise Error(error_msg)


# ============================================================================
# Floating-Point Comparisons
# ============================================================================


fn assert_almost_equal(
    a: Float32,
    b: Float32,
    tolerance: Float32 = Float32(TOLERANCE_DEFAULT),
    message: String = "",
) raises:
    """Assert floating point near-equality for Float32.

    Args:
            a: First value.
            b: Second value.
            tolerance: Maximum allowed difference.
            message: Optional error message.

    Raises:
            Error: If |a - b| > tolerance.
    """
    var diff = abs(a - b)
    if diff > tolerance:
        var error_msg = message if message else (
            String(a) + " !≈ " + String(b) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)


fn assert_almost_equal(
    a: Float64,
    b: Float64,
    tolerance: Float64 = TOLERANCE_DEFAULT,
    message: String = "",
) raises:
    """Assert floating point near-equality for Float64.

    Args:
            a: First value.
            b: Second value.
            tolerance: Maximum allowed difference.
            message: Optional error message.

    Raises:
            Error: If |a - b| > tolerance.
    """
    var diff = abs(a - b)
    if diff > tolerance:
        var error_msg = message if message else (
            String(a) + " !≈ " + String(b) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)


fn assert_dtype_equal(a: DType, b: DType, message: String = "") raises:
    """Assert exact equality of DType values.

    Args:
            a: First DType.
            b: Second DType.
            message: Optional error message.

    Raises:
            Error: If a != b.
    """
    if a != b:
        var error_msg = message if message else "DTypes are not equal"
        raise Error(error_msg)


fn assert_equal_int(a: Int, b: Int, message: String = "") raises:
    """Assert two integers are equal.

    Args:
            a: First integer.
            b: Second integer.
            message: Optional error message.

    Raises:
            Error: If integers are not equal.
    """
    if a != b:
        var error_msg = message if message else (
            "Expected " + String(a) + " == " + String(b)
        )
        raise Error(error_msg)


fn assert_equal_float(a: Float32, b: Float32, message: String = "") raises:
    """Assert exact equality of two Float32 values.

    Args:
            a: First float value.
            b: Second float value.
            message: Optional error message.

    Raises:
            Error: If floats are not exactly equal.
    """
    if a != b:
        var error_msg = message if message else (
            "Float values not equal: " + String(a) + " != " + String(b)
        )
        raise Error(error_msg)


fn assert_close_float(
    a: Float64,
    b: Float64,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-8,
    message: String = "",
) raises:
    """Assert two floats are numerically close.

    Uses the formula: |a - b| <= atol + rtol * |b|.

    Args:
            a: First float.
            b: Second float.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            message: Optional error message.

    Raises:
            Error: If floats differ beyond tolerance.
    """
    # Handle NaN and inf
    var a_is_nan = isnan(a)
    var b_is_nan = isnan(b)
    var a_is_inf = isinf(a)
    var b_is_inf = isinf(b)

    if a_is_nan and b_is_nan:
        return  # Both NaN, considered equal

    if a_is_nan or b_is_nan:
        var error_msg = message if message else (
            "NaN mismatch: " + String(a) + " vs " + String(b)
        )
        raise Error(error_msg)

    if a_is_inf or b_is_inf:
        if a != b:
            var error_msg = message if message else (
                "Infinity mismatch: " + String(a) + " vs " + String(b)
            )
            raise Error(error_msg)
        return

    # Check numeric closeness
    var diff = a - b if a >= b else b - a
    var threshold = atol + rtol * (b if b >= 0 else -b)

    if diff > threshold:
        var error_msg = message if message else (
            "Values differ: "
            + String(a)
            + " vs "
            + String(b)
            + " (diff="
            + String(diff)
            + ", threshold="
            + String(threshold)
            + ")"
        )
        raise Error(error_msg)


# ============================================================================
# Comparison Assertions (Greater/Less)
# ============================================================================


fn assert_greater[
    T: Comparable & Stringable
](a: T, b: T, message: String = "") raises:
    """Assert a > b using parametric type constraints.

    Works with any type supporting Comparable and Stringable traits
    (Float32, Float64, Int, etc.).

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a <= b.
    """
    if a <= b:
        var error_msg = message if message else String(a) + " <= " + String(b)
        raise Error(error_msg)


fn assert_less[
    T: Comparable & Stringable
](a: T, b: T, message: String = "") raises:
    """Assert a < b using parametric type constraints.

    Works with any type supporting Comparable and Stringable traits
    (Float32, Float64, Int, etc.).

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a >= b.
    """
    if a >= b:
        var error_msg = message if message else String(a) + " >= " + String(b)
        raise Error(error_msg)


fn assert_greater_or_equal[
    T: Comparable & Stringable
](a: T, b: T, message: String = "") raises:
    """Assert a >= b using parametric type constraints.

    Works with any type supporting Comparable and Stringable traits
    (Float32, Float64, Int, etc.).

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a < b.
    """
    if a < b:
        var error_msg = message if message else String(a) + " < " + String(b)
        raise Error(error_msg)


fn assert_less_or_equal[
    T: Comparable & Stringable
](a: T, b: T, message: String = "") raises:
    """Assert a <= b using parametric type constraints.

    Works with any type supporting Comparable and Stringable traits
    (Float32, Float64, Int, etc.).

    Args:
            a: First value.
            b: Second value.
            message: Optional error message.

    Raises:
            Error: If a > b.
    """
    if a > b:
        var error_msg = message if message else String(a) + " > " + String(b)
        raise Error(error_msg)


# ============================================================================
# Shape and List Assertions
# ============================================================================


fn assert_shape_equal(
    shape1: List[Int], shape2: List[Int], message: String = ""
) raises:
    """Assert two shapes are equal.

    Args:
            shape1: First shape.
            shape2: Second shape.
            message: Optional error message.

    Raises:
            Error: If shapes are not equal.
    """
    if len(shape1) != len(shape2):
        var error_msg = message if message else (
            "Shape dimensions differ: "
            + String(len(shape1))
            + " vs "
            + String(len(shape2))
        )
        raise Error(error_msg)

    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            var error_msg = message if message else (
                "Shape mismatch at dimension "
                + String(i)
                + ": "
                + String(shape1[i])
                + " vs "
                + String(shape2[i])
            )
            raise Error(error_msg)


# ============================================================================
# Tensor Assertions
# ============================================================================


fn assert_not_equal_tensor(
    a: ExTensor, b: ExTensor, message: String = ""
) raises:
    """Assert two tensors are not equal element-wise.

    Verifies that at least one element differs between the two tensors.
    Useful for tests verifying that weights have been updated during training.

    Args:
            a: First tensor.
            b: Second tensor.
            message: Optional error message.

    Raises:
            Error: If all elements are equal or if shapes differ.
    """
    # Check shapes match
    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("Cannot compare tensors with different dimensions")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("Cannot compare tensors with different shapes")

    # Check if all elements are equal
    var numel = a.numel()
    var all_equal = True

    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)

        if val_a != val_b:
            all_equal = False
            break

    if all_equal:
        var error_msg = (
            message if message else "Tensors are equal but should not be"
        )
        raise Error(error_msg)


fn assert_tensor_equal(a: ExTensor, b: ExTensor, message: String = "") raises:
    """Assert two ExTensors are equal (shape and all elements).

    Args:
            a: First tensor.
            b: Second tensor.
            message: Optional error message.

    Raises:
            Error: If shapes don't match or any elements differ.
    """
    # Check dimensions
    var a_shape = a.shape()
    var b_shape = b.shape()
    if len(a_shape) != len(b_shape):
        var msg = (
            "Shape mismatch: "
            + String(len(a_shape))
            + " vs "
            + String(len(b_shape))
        )
        raise Error(message + ": " + msg if message else msg)

    # Check total elements
    var a_numel = a.numel()
    var b_numel = b.numel()
    if a_numel != b_numel:
        var msg = "Size mismatch: " + String(a_numel) + " vs " + String(b_numel)
        raise Error(message + ": " + msg if message else msg)

    # Check all elements
    for i in range(a_numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b if val_a >= val_b else val_b - val_a
        if diff > 1e-10:
            var msg = (
                "Values differ at index "
                + String(i)
                + ": "
                + String(val_a)
                + " vs "
                + String(val_b)
            )
            raise Error(message + ": " + msg if message else msg)


fn assert_shape(
    tensor: ExTensor, expected: List[Int], message: String = ""
) raises:
    """Assert tensor has expected shape.

    Args:
            tensor: ExTensor to check.
            expected: Expected shape as List.
            message: Optional error message.

    Raises:
            Error: If shapes don't match.
    """
    # Get actual shape
    var actual_shape = tensor.shape()

    # Check dimensions match
    if len(actual_shape) != len(expected):
        var error_msg = message if message else (
            "Shape dimension mismatch: expected "
            + String(len(expected))
            + " dims, got "
            + String(len(actual_shape))
        )
        raise Error(error_msg)

    # Check each dimension
    for i in range(len(expected)):
        if actual_shape[i] != expected[i]:
            var error_msg = message if message else (
                "Shape mismatch at dim "
                + String(i)
                + ": expected "
                + String(expected[i])
                + ", got "
                + String(actual_shape[i])
            )
            raise Error(error_msg)


fn assert_dtype(tensor: ExTensor, expected: DType, message: String = "") raises:
    """Assert tensor has expected dtype.

    Args:
            tensor: ExTensor to check.
            expected: Expected DType.
            message: Optional error message.

    Raises:
            Error: If dtype doesn't match.
    """
    var actual = tensor.dtype()
    if actual != expected:
        var error_msg = message if message else (
            "Expected dtype " + String(expected) + ", got " + String(actual)
        )
        raise Error(error_msg)


fn assert_numel(tensor: ExTensor, expected: Int, message: String = "") raises:
    """Assert tensor has expected number of elements.

    Args:
            tensor: ExTensor to check.
            expected: Expected total element count.
            message: Optional error message.

    Raises:
            Error: If numel doesn't match.
    """
    var actual = tensor.numel()
    if actual != expected:
        var error_msg = message if message else (
            "Expected numel " + String(expected) + ", got " + String(actual)
        )
        raise Error(error_msg)


fn assert_dim(tensor: ExTensor, expected: Int, message: String = "") raises:
    """Assert tensor has expected number of dimensions.

    Args:
            tensor: ExTensor to check.
            expected: Expected dimension count.
            message: Optional error message.

    Raises:
            Error: If dim doesn't match.
    """
    var actual = len(tensor.shape())
    if actual != expected:
        var error_msg = message if message else (
            "Expected "
            + String(expected)
            + " dimensions, got "
            + String(actual)
        )
        raise Error(error_msg)


# ============================================================================
# Tensor Value Assertions
# ============================================================================


fn assert_value_at(
    tensor: ExTensor,
    index: Int,
    expected: Float64,
    tolerance: Float64 = TOLERANCE_DEFAULT,
    message: String = "",
) raises:
    """Assert tensor value at flat index matches expected value.

    Args:
            tensor: ExTensor to check.
            index: Flat index to check.
            expected: Expected value.
            tolerance: Acceptable difference (default: 1e-6).
            message: Optional error message.

    Raises:
            Error: If value doesn't match within tolerance.
    """
    if index < 0 or index >= tensor.numel():
        raise Error("Index out of bounds: " + String(index))

    var actual = tensor._get_float64(index)
    var diff = actual - expected if actual >= expected else expected - actual

    if diff > tolerance:
        var error_msg = message if message else (
            "Expected value "
            + String(expected)
            + " at index "
            + String(index)
            + ", got "
            + String(actual)
            + " (diff: "
            + String(diff)
            + ")"
        )
        raise Error(error_msg)


fn assert_all_values(
    tensor: ExTensor,
    expected: Float64,
    tolerance: Float64 = TOLERANCE_DEFAULT,
    message: String = "",
) raises:
    """Assert all tensor values match expected constant.

    Args:
            tensor: ExTensor to check.
            expected: Expected constant value.
            tolerance: Acceptable difference (default: 1e-6).
            message: Optional error message.

    Raises:
            Error: If any value doesn't match within tolerance.
    """
    var n = tensor.numel()
    for i in range(n):
        var actual = tensor._get_float64(i)
        var diff = (
            actual - expected if actual >= expected else expected - actual
        )

        if diff > tolerance:
            var error_msg = message if message else (
                "Expected all values to be "
                + String(expected)
                + ", but index "
                + String(i)
                + " is "
                + String(actual)
            )
            raise Error(error_msg)


fn assert_all_close(
    a: ExTensor,
    b: ExTensor,
    tolerance: Float64 = TOLERANCE_DEFAULT,
    message: String = "",
) raises:
    """Assert two tensors are element-wise close.

    Args:
            a: First tensor.
            b: Second tensor.
            tolerance: Acceptable difference (default: 1e-6).
            message: Optional error message.

    Raises:
            Error: If shapes don't match or values differ beyond tolerance.
    """
    # Check shapes match
    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error(
            "Shape dimension mismatch: "
            + String(len(shape_a))
            + " vs "
            + String(len(shape_b))
        )

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error(
                "Shape mismatch at dim "
                + String(i)
                + ": "
                + String(shape_a[i])
                + " vs "
                + String(shape_b[i])
            )

    # Check all values
    var n = a.numel()
    for i in range(n):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b if val_a >= val_b else val_b - val_a

        if diff > tolerance:
            var error_msg = message if message else (
                "Tensors differ at index "
                + String(i)
                + ": "
                + String(val_a)
                + " vs "
                + String(val_b)
                + " (diff: "
                + String(diff)
                + ")"
            )
            raise Error(error_msg)


# ============================================================================
# Type Checking
# ============================================================================


fn assert_type[T: AnyType](value: T, expected_type: String) raises:
    """Assert value is of expected type (for documentation purposes).

    Note: Type checking in Mojo happens at compile time, so this function
    is primarily for test documentation and clarity.

    Args:
        value: The value to check.
        expected_type: String describing the expected type (for documentation).

    Raises:
        Never raises - type checking is done at compile time.

    Note:
        This function exists for test API clarity and documentation.
        Actual type checking is performed at compile time by Mojo.
    """
    # Type checking in Mojo is compile-time
    # This function exists for test API clarity
    pass


fn assert_contiguous(tensor: ExTensor, message: String = "") raises:
    """Assert tensor has contiguous memory layout.

    Args:
            tensor: ExTensor to check.
            message: Optional error message.

    Raises:
            Error: If tensor is not contiguous.
    """
    if not tensor.is_contiguous():
        var error_msg = message if message else "Tensor should be contiguous"
        raise Error(error_msg)
