"""Tensor validation utilities for shape and dtype checking.

Provides common validation functions for tensor shape and data type compatibility
These utilities are used throughout the library to validate tensor arguments and
detect shape/dtype mismatches early.

Functions:
    validate_tensor_shape: Validate tensor has expected shape
    validate_tensor_dtype: Validate tensor has expected dtype
    validate_matching_tensors: Validate two tensors have matching shape and dtype
    validate_2d_input: Validate tensor is 2D
    validate_4d_input: Validate tensor is 4D
    validate_1d_input: Validate tensor is 1D
    validate_3d_input: Validate tensor is 3D
    validate_axis: Validate axis is in valid range for tensor
    validate_slice_range: Validate slice indices are in bounds
    validate_float_dtype: Validate tensor has float dtype
    validate_positive_shape: Validate all shape dimensions are positive
    validate_matmul_dims: Validate matmul dimension compatibility
    validate_broadcast_compatible: Validate shapes are broadcast-compatible
    validate_non_empty: Validate tensor is not empty
    validate_matching_dtype: Validate two tensors have matching dtype
"""

from .extensor import ExTensor


fn validate_tensor_shape(
    tensor: ExTensor, expected_shape: List[Int], name: String
) raises:
    """Validate that a tensor has the expected shape.

    Checks if the tensor's shape matches the expected shape. If not, raises
    an error with a descriptive message including the tensor name.

    Args:
        tensor: The tensor to validate.
        expected_shape: The expected shape as a List[Int].
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor shape does not match the expected shape

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        var expected : List[Int] = [2, 3]
        validate_tensor_shape(x, expected, "input")  # Passes
        ```
    """
    var actual_shape = tensor.shape()

    # Check number of dimensions
    if len(actual_shape) != len(expected_shape):
        raise Error(
            name
            + ": expected "
            + String(len(expected_shape))
            + "D tensor, got "
            + String(len(actual_shape))
            + "D"
        )

    # Check each dimension
    for i in range(len(expected_shape)):
        if actual_shape[i] != expected_shape[i]:
            raise Error(
                name
                + ": expected shape ["
                + _shape_to_string(expected_shape)
                + "], got ["
                + _shape_to_string(actual_shape)
                + "]"
            )


fn validate_tensor_dtype(
    tensor: ExTensor, expected_dtype: DType, name: String
) raises:
    """Validate that a tensor has the expected data type.

    Checks if the tensor's dtype matches the expected dtype. If not, raises
    an error with a descriptive message including the tensor name.

    Args:
        tensor: The tensor to validate.
        expected_dtype: The expected data type.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor dtype does not match the expected dtype

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_tensor_dtype(x, DType.float32, "input")  # Passes
        validate_tensor_dtype(x, DType.float64, "input")  # Raises error
        ```
    """
    var actual_dtype = tensor.dtype()
    if actual_dtype != expected_dtype:
        raise Error(
            name
            + ": expected dtype "
            + _dtype_to_string(expected_dtype)
            + ", got "
            + _dtype_to_string(actual_dtype)
        )


fn validate_matching_tensors(
    a: ExTensor, b: ExTensor, a_name: String, b_name: String
) raises:
    """Validate that two tensors have matching shape and dtype.

    Checks if two tensors have the same shape and data type. Useful for
    verifying that tensors can be used together in element-wise operations.

    Args:
        a: The first tensor.
        b: The second tensor.
        a_name: The name of the first tensor (for error messages).
        b_name: The name of the second tensor (for error messages).

    Raises:
        Error: If tensors have different shapes or dtypes

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        var y = ones([2, 3], DType.float32)
        validate_matching_tensors(x, y, "x", "y")  # Passes

        var z = zeros([3, 2], DType.float32)
        validate_matching_tensors(x, z, "x", "z")  # Raises error
        ```
    """
    var shape_a = a.shape()
    var shape_b = b.shape()
    var dtype_a = a.dtype()
    var dtype_b = b.dtype()

    # Check dtype match
    if dtype_a != dtype_b:
        raise Error(
            a_name
            + " and "
            + b_name
            + " have mismatched dtypes: "
            + _dtype_to_string(dtype_a)
            + " vs "
            + _dtype_to_string(dtype_b)
        )

    # Check shape match
    if len(shape_a) != len(shape_b):
        raise Error(
            a_name
            + " and "
            + b_name
            + " have mismatched number of dimensions: "
            + String(len(shape_a))
            + " vs "
            + String(len(shape_b))
        )

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error(
                a_name
                + " and "
                + b_name
                + " have mismatched shapes: ["
                + _shape_to_string(shape_a)
                + "] vs ["
                + _shape_to_string(shape_b)
                + "]"
            )


fn validate_2d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 2-dimensional.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor is not 2D

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_2d_input(x, "input")  # Passes

        var y = zeros([2, 3, 4], DType.float32)
        validate_2d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 2:
        raise Error(
            name
            + ": expected 2D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


fn validate_4d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 4-dimensional.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor is not 4D

    Example:
        ```mojo
        var x = zeros([2, 3, 4, 5], DType.float32)
        validate_4d_input(x, "input")  # Passes

        var y = zeros([2, 3, 4], DType.float32)
        validate_4d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 4:
        raise Error(
            name
            + ": expected 4D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


fn validate_1d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 1-dimensional.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor is not 1D

    Example:
        ```mojo
        var x = zeros([5], DType.float32)
        validate_1d_input(x, "input")  # Passes

        var y = zeros([2, 3], DType.float32)
        validate_1d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 1:
        raise Error(
            name
            + ": expected 1D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


fn validate_3d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 3-dimensional.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If the tensor is not 3D

    Example:
        ```mojo
        var x = zeros([2, 3, 4], DType.float32)
        validate_3d_input(x, "input")  # Passes

        var y = zeros([2, 3], DType.float32)
        validate_3d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 3:
        raise Error(
            name
            + ": expected 3D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


fn validate_axis(tensor: ExTensor, axis: Int, name: String) raises:
    """Validate that axis is within valid range for tensor dimensions.

    Supports both positive and negative indexing (e.g., -1 for last axis).

    Args:
        tensor: The tensor to validate.
        axis: The axis index to validate.
        name: The name of the parameter (for error messages).

    Raises:
        Error: If axis is out of range

    Example:
        ```mojo
        var x = zeros([2, 3, 4], DType.float32)
        validate_axis(x, 0, "axis")   # Passes
        validate_axis(x, -1, "axis")  # Passes (last axis)
        validate_axis(x, 5, "axis")   # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    # Normalize negative axis
    var normalized_axis = axis
    if axis < 0:
        normalized_axis = ndim + axis

    if normalized_axis < 0 or normalized_axis >= ndim:
        raise Error(
            name
            + ": axis "
            + String(axis)
            + " out of range for "
            + String(ndim)
            + "D tensor"
        )


fn validate_slice_range(
    tensor: ExTensor, axis: Int, start: Int, end: Int, name: String
) raises:
    """Validate that slice range is within bounds for the given axis.

    Args:
        tensor: The tensor to validate.
        axis: The axis to slice along.
        start: The start index of the slice.
        end: The end index of the slice.
        name: The name of the operation (for error messages).

    Raises:
        Error: If start or end are out of bounds, or if start >= end

    Example:
        ```mojo
        var x = zeros([10], DType.float32)
        validate_slice_range(x, 0, 2, 5, "slice")  # Passes
        validate_slice_range(x, 0, -1, 15, "slice")  # Raises error
        ```
    """
    # First validate the axis
    validate_axis(tensor, axis, "axis")

    var shape = tensor.shape()
    var normalized_axis = axis
    if axis < 0:
        normalized_axis = len(shape) + axis

    var dim_size = shape[normalized_axis]

    if start < 0:
        raise Error(
            name + ": slice start " + String(start) + " must be non-negative"
        )

    if end > dim_size:
        raise Error(
            name
            + ": slice end "
            + String(end)
            + " out of bounds for dimension size "
            + String(dim_size)
        )

    if start >= end:
        raise Error(
            name
            + ": slice range ["
            + String(start)
            + ":"
            + String(end)
            + "] is empty (start >= end)"
        )


fn validate_float_dtype(tensor: ExTensor, name: String) raises:
    """Validate that tensor has a floating-point dtype.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If tensor dtype is not float16, float32, float64, or bfloat16

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_float_dtype(x, "input")  # Passes

        var y = zeros([2, 3], DType.int32)
        validate_float_dtype(y, "input")  # Raises error
        ```
    """
    var dtype = tensor.dtype()

    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
        and dtype != DType.bfloat16
    ):
        raise Error(
            name
            + ": operation requires float dtype, got "
            + _dtype_to_string(dtype)
        )


fn validate_positive_shape(shape: List[Int], name: String) raises:
    """Validate that all dimensions in shape are positive.

    Args:
        shape: The shape to validate.
        name: The name of the parameter (for error messages).

    Raises:
        Error: If any dimension is <= 0

    Example:
        ```mojo
        var shape1 : List[Int] = [2, 3, 4]
        validate_positive_shape(shape1, "shape")  # Passes

        var shape2 : List[Int] = [2, 0, 4]
        validate_positive_shape(shape2, "shape")  # Raises error
        ```
    """
    for i in range(len(shape)):
        if shape[i] <= 0:
            raise Error(
                name
                + ": all dimensions must be positive, got dimension "
                + String(i)
                + " = "
                + String(shape[i])
            )


fn validate_matmul_dims(
    a: ExTensor, b: ExTensor, a_name: String, b_name: String
) raises:
    """Validate that two tensors have compatible dimensions for matmul.

    Both tensors must be 2D, and the inner dimensions must match:
    a.shape = [M, K], b.shape = [K, N]

    Args:
        a: The first tensor (left operand).
        b: The second tensor (right operand).
        a_name: The name of the first tensor (for error messages).
        b_name: The name of the second tensor (for error messages).

    Raises:
        Error: If tensors are not 2D or inner dimensions don't match

    Example:
        ```mojo
        var a = zeros([3, 4], DType.float32)
        var b = zeros([4, 5], DType.float32)
        validate_matmul_dims(a, b, "a", "b")  # Passes

        var c = zeros([3, 6], DType.float32)
        validate_matmul_dims(a, c, "a", "c")  # Raises error
        ```
    """
    # Validate both are 2D
    validate_2d_input(a, a_name)
    validate_2d_input(b, b_name)

    var shape_a = a.shape()
    var shape_b = b.shape()

    var a_cols = shape_a[1]
    var b_rows = shape_b[0]

    if a_cols != b_rows:
        raise Error(
            "matmul: inner dimensions must match, got "
            + a_name
            + " with shape ["
            + _shape_to_string(shape_a)
            + "] and "
            + b_name
            + " with shape ["
            + _shape_to_string(shape_b)
            + "]. Expected "
            + a_name
            + ".shape[1] == "
            + b_name
            + ".shape[0], but "
            + String(a_cols)
            + " != "
            + String(b_rows)
        )


fn validate_broadcast_compatible(
    a: ExTensor, b: ExTensor, a_name: String, b_name: String
) raises:
    """Validate that two tensors have broadcast-compatible shapes.

    Two shapes are broadcast-compatible if, for each dimension (comparing from right):
    - They are equal, OR
    - One of them is 1

    Args:
        a: The first tensor.
        b: The second tensor.
        a_name: The name of the first tensor (for error messages).
        b_name: The name of the second tensor (for error messages).

    Raises:
        Error: If shapes are not broadcast-compatible

    Example:
        ```mojo
        var a = zeros([3, 1, 4], DType.float32)
        var b = zeros([1, 5, 4], DType.float32)
        validate_broadcast_compatible(a, b, "a", "b")  # Passes

        var c = zeros([3, 2, 4], DType.float32)
        var d = zeros([3, 5, 4], DType.float32)
        validate_broadcast_compatible(c, d, "c", "d")  # Raises error
        ```
    """
    var shape_a = a.shape()
    var shape_b = b.shape()

    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)

    # Compare from right to left
    var max_ndim = ndim_a if ndim_a > ndim_b else ndim_b

    for i in range(max_ndim):
        # Get dimensions (use 1 for dimensions that don't exist)
        var dim_a = 1
        var dim_b = 1

        if i < ndim_a:
            dim_a = shape_a[ndim_a - 1 - i]
        if i < ndim_b:
            dim_b = shape_b[ndim_b - 1 - i]

        # Check if compatible
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            raise Error(
                "broadcast: shapes ["
                + _shape_to_string(shape_a)
                + "] and ["
                + _shape_to_string(shape_b)
                + "] are not compatible for "
                + a_name
                + " and "
                + b_name
                + ". At dimension "
                + String(max_ndim - 1 - i)
                + " from right: "
                + String(dim_a)
                + " vs "
                + String(dim_b)
            )


fn validate_non_empty(tensor: ExTensor, name: String) raises:
    """Validate that tensor is not empty (has at least one element).

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

    Raises:
        Error: If tensor has zero elements

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_non_empty(x, "input")  # Passes

        var y = zeros([0], DType.float32)
        validate_non_empty(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()

    # Check if any dimension is 0
    for i in range(len(shape)):
        if shape[i] == 0:
            raise Error(
                name
                + ": tensor is empty with shape ["
                + _shape_to_string(shape)
                + "]"
            )


fn validate_matching_dtype(
    a: ExTensor, b: ExTensor, a_name: String, b_name: String
) raises:
    """Validate that two tensors have matching dtypes.

    Args:
        a: The first tensor.
        b: The second tensor.
        a_name: The name of the first tensor (for error messages).
        b_name: The name of the second tensor (for error messages).

    Raises:
        Error: If tensors have different dtypes

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        var y = ones([2, 3], DType.float32)
        validate_matching_dtype(x, y, "x", "y")  # Passes

        var z = zeros([2, 3], DType.float64)
        validate_matching_dtype(x, z, "x", "z")  # Raises error
        ```
    """
    var dtype_a = a.dtype()
    var dtype_b = b.dtype()

    if dtype_a != dtype_b:
        raise Error(
            a_name
            + " and "
            + b_name
            + " have mismatched dtypes: "
            + _dtype_to_string(dtype_a)
            + " vs "
            + _dtype_to_string(dtype_b)
        )


# ============================================================================
# Helper Functions for Error Message Formatting
# ============================================================================


fn _shape_to_string(shape: List[Int]) -> String:
    """Convert a shape list to a string representation.

    Args:
        shape: The shape as a List[Int]

    Returns:
        A string like "2, 3, 4" for shape [2, 3, 4]
    """
    if len(shape) == 0:
        return ""

    var result = String(shape[0])
    for i in range(1, len(shape)):
        result += ", " + String(shape[i])

    return result


fn _dtype_to_string(dtype: DType) -> String:
    """Convert a DType to a readable string representation.

    Args:
        dtype: The data type

    Returns:
        A string like "float32", "int64", etc.
    """
    if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"
