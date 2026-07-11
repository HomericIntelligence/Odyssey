"""Generic dtype dispatch helpers for eliminating dtype branching.

This module provides compile-time specialized dispatch helpers that eliminate
the need for repeated dtype branching in tensor operations. Instead of writing
40+ lines of dtype-specific code for each operation, you can use these helpers
to dispatch to compile-time specialized versions.

Benefits:
- 80% code reduction (500 lines → 100 lines)
- Single source of truth for operations
- Compile-time specialization (zero runtime overhead)
- Easy to add new dtypes
- Better error messages with dtype context

Supported Dtypes:
- Float16 (FP16) ✓ - Fully supported for mixed precision training
- Float32 (FP32) ✓ - Default precision
- Float64 (FP64) ✓ - High precision
- Int8, Int16, Int32, Int64 ✓ - Integer types
- UInt8, UInt16, UInt32, UInt64 ✓ - Unsigned integer types
- BF16 ⚠ - Not yet available in Mojo (add when supported)

Example usage:
    # Before (40+ lines):
    if tensor.dtype() == DType.float32:
        for i in range(size):
            result[i] = Float64(op(tensor._data.bitcast[Float32]()[i]))
    elif tensor.dtype() == DType.float64:
        # ... repeat for all dtypes.

    # After (5 lines):
    return elementwise_unary[relu_op](tensor)

Error Handling:
    All dispatch functions raise Error with descriptive messages including:
    - Which function failed (dispatch_unary, dispatch_binary, etc.)
    - Which dtype was unsupported
    - Expected dtype family (all, float-only, etc.)

See notes/issues/dtype-refactoring-plan.md for complete design documentation
"""

from odyssey.tensor.any_tensor import AnyTensor
from std.collections import List
from odyssey.base.dtype_ordinal import (
    dtype_to_ordinal,
    format_dtype_name,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
)


# ============================================================================
# Helper Function: Format dtype name for error messages
# ============================================================================


def _format_dtype_name(dtype: DType) -> String:
    """Format a DType into a readable string name.

    Args:
            dtype: The dtype to format.

    Returns:
            String representation of the dtype name.

        Note: Used internally for error messages. Delegates to dtype_ordinal module.
    """
    return format_dtype_name(dtype)


# ============================================================================
# Unary Operation Dispatch
# ============================================================================


def elementwise_unary[
    dtype: DType, op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Apply unary operation with compile-time dtype specialization.

        This function is compile-time specialized for a specific dtype and operation.
        Use `dispatch_unary` for runtime dtype dispatch.

    Parameters:
        dtype: Compile-time dtype parameter.
        op: Unary operation function pointer.

    Args:
        tensor: Input tensor.

    Returns:
            New tensor with operation applied element-wise.

    Raises:
        Error: If operation fails.

    Example:
        ```mojo
        # Define operation
        def my_op[T: DType](x: Scalar[T]) -> Scalar[T]:
            return max(Scalar[T](0), x)

        # Apply with compile-time dtype
        var result = elementwise_unary[DType.float32, my_op](tensor)
        ```
    """
    var result = AnyTensor(tensor.shape(), dtype)
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(size):
        out_ptr[i] = op[dtype](in_ptr[i])

    return result^


def dispatch_unary[
    op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized unary operation.

        This function performs runtime dtype checking but dispatches to compile-time
        specialized versions of the operation, ensuring zero overhead compared to
        hand-written dtype branches.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Unary operation function pointer.

    Args:
            tensor: Input tensor.

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If tensor dtype is not supported. Error message includes the
                   unsupported dtype and list of supported dtypes.

        Example:
            ```mojo
           def relu_op[T: DType](x: Scalar[T]) -> Scalar[T]:
                return max(Scalar[T](0), x)

            var result = dispatch_unary[relu_op](tensor)  # Works for any dtype
            ```

    Note:
            Supports: float16, float32, float64, int8, int16, int32, int64,
                      uint8, uint16, uint32, uint64.
    """
    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    # Dispatch based on ordinal - compiler generates jump table for consecutive integers
    if ordinal == DTYPE_FLOAT16:
        return elementwise_unary[DType.float16, op](tensor)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_unary[DType.float32, op](tensor)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_unary[DType.float64, op](tensor)
    elif ordinal == DTYPE_INT8:
        return elementwise_unary[DType.int8, op](tensor)
    elif ordinal == DTYPE_INT16:
        return elementwise_unary[DType.int16, op](tensor)
    elif ordinal == DTYPE_INT32:
        return elementwise_unary[DType.int32, op](tensor)
    elif ordinal == DTYPE_INT64:
        return elementwise_unary[DType.int64, op](tensor)
    elif ordinal == DTYPE_UINT8:
        return elementwise_unary[DType.uint8, op](tensor)
    elif ordinal == DTYPE_UINT16:
        return elementwise_unary[DType.uint16, op](tensor)
    elif ordinal == DTYPE_UINT32:
        return elementwise_unary[DType.uint32, op](tensor)
    elif ordinal == DTYPE_UINT64:
        return elementwise_unary[DType.uint64, op](tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_unary: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Binary Operation Dispatch
# ============================================================================


def elementwise_binary[
    dtype: DType, op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](lhs: AnyTensor, rhs: AnyTensor) raises -> AnyTensor:
    """Apply binary operation with compile-time dtype specialization.

        This function is compile-time specialized for a specific dtype and operation.
        Use `dispatch_binary` for runtime dtype dispatch.
    Parameters:
            dtype: Compile-time dtype parameter.
            op: Binary operation function pointer.
    Args:
            lhs: Left-hand side tensor.
            rhs: Right-hand side tensor (must have same shape as lhs).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If shapes don't match.

        Example:
            ```mojo
            def add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
                return x + y

            var result = elementwise_binary[DType.float32, add_op](a, b)
            ```
    """
    # Validate shapes match
    if lhs._numel != rhs._numel:
        raise Error(
            "elementwise_binary: tensors must have same number of elements"
        )

    var result = AnyTensor(lhs.shape(), dtype)
    var size = lhs._numel

    var lhs_ptr = lhs._data.bitcast[Scalar[dtype]]()
    var rhs_ptr = rhs._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(size):
        out_ptr[i] = op[dtype](lhs_ptr[i], rhs_ptr[i])

    return result^


def dispatch_binary[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](lhs: AnyTensor, rhs: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized binary operation.

        This function performs runtime dtype checking but dispatches to compile-time
        specialized versions of the operation.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Binary operation function pointer.
    Args:
            lhs: Left-hand side tensor.
            rhs: Right-hand side tensor.

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If dtypes don't match or are unsupported. Error message includes
                   the actual dtypes and list of supported dtypes.

        Example:
            ```mojo
           def mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
                return x * y

            var result = dispatch_binary[mul_op](a, b)
            ```

    Note:
            Supports: float16, float32, float64, int8, int16, int32, int64,
                      uint8, uint16, uint32, uint64.
            Both tensors must have matching dtypes.
    """
    # Validate dtypes match
    if lhs._dtype != rhs._dtype:
        var lhs_dtype = _format_dtype_name(lhs._dtype)
        var rhs_dtype = _format_dtype_name(rhs._dtype)
        raise Error(
            "dispatch_binary: dtypes must match. Got lhs="
            + lhs_dtype
            + ", rhs="
            + rhs_dtype
        )

    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(lhs._dtype)

    # Dispatch based on ordinal - compiler generates jump table for consecutive integers
    if ordinal == DTYPE_FLOAT16:
        return elementwise_binary[DType.float16, op](lhs, rhs)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_binary[DType.float32, op](lhs, rhs)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_binary[DType.float64, op](lhs, rhs)
    elif ordinal == DTYPE_INT8:
        return elementwise_binary[DType.int8, op](lhs, rhs)
    elif ordinal == DTYPE_INT16:
        return elementwise_binary[DType.int16, op](lhs, rhs)
    elif ordinal == DTYPE_INT32:
        return elementwise_binary[DType.int32, op](lhs, rhs)
    elif ordinal == DTYPE_INT64:
        return elementwise_binary[DType.int64, op](lhs, rhs)
    elif ordinal == DTYPE_UINT8:
        return elementwise_binary[DType.uint8, op](lhs, rhs)
    elif ordinal == DTYPE_UINT16:
        return elementwise_binary[DType.uint16, op](lhs, rhs)
    elif ordinal == DTYPE_UINT32:
        return elementwise_binary[DType.uint32, op](lhs, rhs)
    elif ordinal == DTYPE_UINT64:
        return elementwise_binary[DType.uint64, op](lhs, rhs)
    else:
        var dtype_name = _format_dtype_name(lhs._dtype)
        raise Error(
            "dispatch_binary: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Scalar Binary Operation Dispatch (tensor op scalar)
# ============================================================================


def elementwise_scalar[
    dtype: DType, op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Apply scalar binary operation with compile-time dtype specialization.

        This function applies a binary operation between a tensor and a scalar value.
        The scalar is converted to the appropriate dtype at compile time.

    Parameters:
            dtype: Compile-time dtype parameter.
            op: Binary operation function pointer.
    Args:
            tensor: Input tensor.
            scalar: Scalar value (converted to appropriate dtype).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
           def mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
                return x * y

        var result = elementwise_scalar[DType.float32, mul_op](tensor, 2.5)
            ```
    """
    var result = AnyTensor(tensor.shape(), dtype)
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var scalar_val = Scalar[dtype](scalar)

    for i in range(size):
        out_ptr[i] = op[dtype](in_ptr[i], scalar_val)

    return result^


def dispatch_scalar[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized scalar operation.

        This function performs runtime dtype checking but dispatches to compile-time
        specialized versions of the operation. The scalar value is automatically
        converted to the tensor's dtype.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Binary operation function pointer.
    Args:
            tensor: Input tensor.
            scalar: Scalar value (converted to tensor's dtype).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If tensor dtype is not supported. Error message includes the
                   unsupported dtype and list of supported dtypes.

        Example:
            ```mojo
           def add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
                return x + y

           var result = dispatch_scalar[add_op](tensor, 1.0)
            ```

    Note:
            Supports: float16, float32, float64, int8, int16, int32, int64,
                      uint8, uint16, uint32, uint64.
            The scalar is automatically converted to match the tensor's dtype.
    """
    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    # Dispatch based on ordinal - compiler generates jump table for consecutive integers
    if ordinal == DTYPE_FLOAT16:
        return elementwise_scalar[DType.float16, op](tensor, scalar)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_scalar[DType.float32, op](tensor, scalar)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_scalar[DType.float64, op](tensor, scalar)
    elif ordinal == DTYPE_INT8:
        return elementwise_scalar[DType.int8, op](tensor, scalar)
    elif ordinal == DTYPE_INT16:
        return elementwise_scalar[DType.int16, op](tensor, scalar)
    elif ordinal == DTYPE_INT32:
        return elementwise_scalar[DType.int32, op](tensor, scalar)
    elif ordinal == DTYPE_INT64:
        return elementwise_scalar[DType.int64, op](tensor, scalar)
    elif ordinal == DTYPE_UINT8:
        return elementwise_scalar[DType.uint8, op](tensor, scalar)
    elif ordinal == DTYPE_UINT16:
        return elementwise_scalar[DType.uint16, op](tensor, scalar)
    elif ordinal == DTYPE_UINT32:
        return elementwise_scalar[DType.uint32, op](tensor, scalar)
    elif ordinal == DTYPE_UINT64:
        return elementwise_scalar[DType.uint64, op](tensor, scalar)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_scalar: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Float-only Operation Dispatch
# ============================================================================


def dispatch_float_unary[
    op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch for floating-point only unary operations.

        Use this for operations like sigmoid, tanh, exp, log that only support
        floating-point dtypes. This function validates that the input tensor is
        one of the supported float types and dispatches to the appropriate
        compile-time specialized version.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Unary operation function pointer.
    Args:
            tensor: Input tensor (must be float16/32/64).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If tensor dtype is not a float type. Error message includes the
                   actual dtype and supported float types.

        Example:
            ```mojo
           def sigmoid_op[T: DType](x: Scalar[T]) -> Scalar[T]:
                # Assuming T is float
                return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-x))

        var result = dispatch_float_unary[sigmoid_op](tensor)
            ```

    Note:
            Supports float types only: float16, float32, float64.
            For integer operations, use dispatch_unary instead.
    """
    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    # Dispatch based on ordinal - only float types
    if ordinal == DTYPE_FLOAT16:
        return elementwise_unary[DType.float16, op](tensor)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_unary[DType.float32, op](tensor)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_unary[DType.float64, op](tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_float_unary: operation only supports float16/32/64. Got "
            + dtype_name
        )


def dispatch_float_binary[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](lhs: AnyTensor, rhs: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch for floating-point only binary operations.

        Use this for operations that only support floating-point dtypes.
        This function validates that both input tensors are float types
        and dispatches to the appropriate compile-time specialized version.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Binary operation function pointer.
    Args:
            lhs: Left-hand side tensor (must be float16/32/64).
            rhs: Right-hand side tensor (must match lhs dtype).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If dtypes don't match or are not float types. Error message
                   includes the actual dtypes.

    Note:
            Supports float types only: float16, float32, float64.
            Both tensors must have matching dtypes.
    """
    # Validate dtypes match
    if lhs._dtype != rhs._dtype:
        var lhs_dtype = _format_dtype_name(lhs._dtype)
        var rhs_dtype = _format_dtype_name(rhs._dtype)
        raise Error(
            "dispatch_float_binary: dtypes must match. Got lhs="
            + lhs_dtype
            + ", rhs="
            + rhs_dtype
        )

    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(lhs._dtype)

    # Dispatch based on ordinal - only float types
    if ordinal == DTYPE_FLOAT16:
        return elementwise_binary[DType.float16, op](lhs, rhs)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_binary[DType.float32, op](lhs, rhs)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_binary[DType.float64, op](lhs, rhs)
    else:
        var dtype_name = _format_dtype_name(lhs._dtype)
        raise Error(
            "dispatch_float_binary: operation only supports float16/32/64. Got "
            + dtype_name
        )


def dispatch_float_scalar[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    """Runtime dispatch for floating-point only scalar operations.

        Use this for operations that only support floating-point dtypes.
        The scalar value is automatically converted to the tensor's float dtype.

        Uses ordinal-based dispatch for optimized lookup (compiler can generate jump table).

    Parameters:
            op: Binary operation function pointer.
    Args:
            tensor: Input tensor (must be float16/32/64).
            scalar: Scalar value (converted to tensor's dtype).

    Returns:
            New tensor with operation applied element-wise.

    Raises:
            Error: If tensor dtype is not a float type. Error message includes the
                   actual dtype and supported float types.

    Note:
            Supports float types only: float16, float32, float64.
            The scalar is automatically converted to match the tensor's float dtype.
    """
    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    # Dispatch based on ordinal - only float types
    if ordinal == DTYPE_FLOAT16:
        return elementwise_scalar[DType.float16, op](tensor, scalar)
    elif ordinal == DTYPE_FLOAT32:
        return elementwise_scalar[DType.float32, op](tensor, scalar)
    elif ordinal == DTYPE_FLOAT64:
        return elementwise_scalar[DType.float64, op](tensor, scalar)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_float_scalar: operation only supports float16/32/64. Got "
            + dtype_name
        )


# ============================================================================
# Activation-Specific Dispatch Helpers
# ============================================================================


def _softmax_impl[
    dtype: DType
](
    result: AnyTensor,
    tensor: AnyTensor,
    outer_size: Int,
    axis_size: Int,
    axis_stride: Int,
) raises:
    """Compile-time specialized softmax implementation.

    Uses log-sum-exp trick for numerical stability by subtracting max value
    before exponentiation.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output tensor (pre-allocated).
        tensor: Input tensor.
        outer_size: Product of dimensions before the softmax axis.
        axis_size: Size of the softmax axis.
        axis_stride: Product of dimensions after the softmax axis.
    """
    from std.math import exp

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    # For float16, use float32 intermediate precision
    comptime if dtype == DType.float16:
        for outer_idx in range(outer_size):
            for inner_idx in range(axis_stride):
                # Find max along axis
                var max_val = Float32(
                    in_ptr[
                        (outer_idx * axis_size + 0) * axis_stride + inner_idx
                    ]
                )
                for k in range(1, axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var val = Float32(in_ptr[idx])
                    if val > max_val:
                        max_val = val

                # Compute exp(x - max) and sum
                var sum_exp: Float32 = 0.0
                for k in range(axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var val = Float32(in_ptr[idx])
                    var exp_val = exp(val - max_val)
                    out_ptr[idx] = Scalar[dtype](exp_val)
                    sum_exp += exp_val

                # Normalize
                for k in range(axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var current = Float32(out_ptr[idx])
                    out_ptr[idx] = Scalar[dtype](current / sum_exp)
    elif dtype == DType.float32 or dtype == DType.float64:
        for outer_idx in range(outer_size):
            for inner_idx in range(axis_stride):
                # Find max along axis
                var max_val = Float64(
                    in_ptr[
                        (outer_idx * axis_size + 0) * axis_stride + inner_idx
                    ]
                )
                for k in range(1, axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var val = Float64(in_ptr[idx])
                    if val > max_val:
                        max_val = val

                # Compute exp(x - max) and sum
                var sum_exp: Float64 = 0.0
                for k in range(axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var val = Float64(in_ptr[idx])
                    var exp_val = exp(val - max_val)
                    out_ptr[idx] = Scalar[dtype](exp_val)
                    sum_exp += exp_val

                # Normalize
                for k in range(axis_size):
                    var idx = (
                        outer_idx * axis_size + k
                    ) * axis_stride + inner_idx
                    var current = Float64(out_ptr[idx])
                    out_ptr[idx] = Scalar[dtype](current / sum_exp)


def dispatch_softmax(
    tensor: AnyTensor, outer_size: Int, axis_size: Int, axis_stride: Int
) raises -> AnyTensor:
    """Runtime dispatch for softmax operation.

    Args:
        tensor: Input tensor.
        outer_size: Product of dimensions before the softmax axis.
        axis_size: Size of the softmax axis.
        axis_stride: Product of dimensions after the softmax axis.

    Returns:
        Softmax output tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float16:
        _softmax_impl[DType.float16](
            result, tensor, outer_size, axis_size, axis_stride
        )
    elif tensor._dtype == DType.float32:
        _softmax_impl[DType.float32](
            result, tensor, outer_size, axis_size, axis_stride
        )
    elif tensor._dtype == DType.float64:
        _softmax_impl[DType.float64](
            result, tensor, outer_size, axis_size, axis_stride
        )
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_softmax: only supports float16/32/64. Got " + dtype_name
        )

    return result^


def _softmax_backward_impl[
    dtype: DType
](
    result: AnyTensor,
    grad_output: AnyTensor,
    output: AnyTensor,
    outer_size: Int,
    axis_size: Int,
    axis_stride: Int,
) raises:
    """Compile-time specialized softmax backward implementation.

    Softmax gradient: grad_input[i] = output[i] * (grad_output[i] - sum_j(grad_output[j] * output[j])).

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output gradient tensor (pre-allocated).
        grad_output: Upstream gradient.
        output: Softmax forward output.
        outer_size: Product of dimensions before axis.
        axis_size: Size of softmax axis.
        axis_stride: Product of dimensions after axis.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var out_ptr = output._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        for outer in range(outer_size):
            for inner in range(axis_stride):
                # Compute dot product sum(grad * output) along axis
                var dot_sum: Float32 = 0.0
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = Float32(grad_ptr[idx])
                    var out_val = Float32(out_ptr[idx])
                    dot_sum += grad_val * out_val

                # Compute gradient: output * (grad - dot_sum)
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = Float32(grad_ptr[idx])
                    var out_val = Float32(out_ptr[idx])
                    result_ptr[idx] = Scalar[dtype](
                        out_val * (grad_val - dot_sum)
                    )
    else:
        for outer in range(outer_size):
            for inner in range(axis_stride):
                # Compute dot product sum(grad * output) along axis
                var dot_sum = Scalar[dtype](0.0)
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    dot_sum += grad_ptr[idx] * out_ptr[idx]

                # Compute gradient: output * (grad - dot_sum)
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    result_ptr[idx] = out_ptr[idx] * (grad_ptr[idx] - dot_sum)


def dispatch_softmax_backward(
    grad_output: AnyTensor,
    output: AnyTensor,
    outer_size: Int,
    axis_size: Int,
    axis_stride: Int,
) raises -> AnyTensor:
    """Runtime dispatch for softmax backward operation.

    Args:
        grad_output: Upstream gradient.
        output: Softmax forward output.
        outer_size: Product of dimensions before axis.
        axis_size: Size of softmax axis.
        axis_stride: Product of dimensions after axis.

    Returns:
        Input gradient tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(output.shape(), output._dtype)

    if output._dtype == DType.float16:
        _softmax_backward_impl[DType.float16](
            result, grad_output, output, outer_size, axis_size, axis_stride
        )
    elif output._dtype == DType.float32:
        _softmax_backward_impl[DType.float32](
            result, grad_output, output, outer_size, axis_size, axis_stride
        )
    elif output._dtype == DType.float64:
        _softmax_backward_impl[DType.float64](
            result, grad_output, output, outer_size, axis_size, axis_stride
        )
    else:
        var dtype_name = _format_dtype_name(output._dtype)
        raise Error(
            "dispatch_softmax_backward: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _gelu_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, approximate: Bool) raises:
    """Compile-time specialized GELU implementation.

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Exact: x * 0.5 * (1 + erf(x / sqrt(2))).
    Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output tensor (pre-allocated).
        tensor: Input tensor.
        approximate: Use tanh approximation if True.
    """
    from std.math import exp, erf, tanh as math_tanh

    comptime SQRT_2_OVER_PI = 0.7978845608028654
    comptime GELU_COEFF = 0.044715
    comptime SQRT_2 = 1.4142135623730951

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        if approximate:
            for i in range(tensor._numel):
                var x = Float32(in_ptr[i])
                var x_cubed = x * x * x
                var inner = Float32(SQRT_2_OVER_PI) * (
                    x + Float32(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                out_ptr[i] = Scalar[dtype](0.5 * x * (1.0 + tanh_val))
        else:
            for i in range(tensor._numel):
                var x = Float32(in_ptr[i])
                var erf_val = erf(x / Float32(SQRT_2))
                out_ptr[i] = Scalar[dtype](x * 0.5 * (1.0 + erf_val))
    elif dtype == DType.float32:
        if approximate:
            for i in range(tensor._numel):
                var x = Float32(in_ptr[i])
                var x_cubed = x * x * x
                var inner = Float32(SQRT_2_OVER_PI) * (
                    x + Float32(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                out_ptr[i] = Scalar[dtype](0.5 * x * (1.0 + tanh_val))
        else:
            for i in range(tensor._numel):
                var x = Float32(in_ptr[i])
                var erf_val = erf(x / Float32(SQRT_2))
                out_ptr[i] = Scalar[dtype](x * 0.5 * (1.0 + erf_val))
    else:  # float64
        if approximate:
            for i in range(tensor._numel):
                var x = Float64(in_ptr[i])
                var x_cubed = x * x * x
                var inner = Float64(SQRT_2_OVER_PI) * (
                    x + Float64(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                out_ptr[i] = Scalar[dtype](0.5 * x * (1.0 + tanh_val))
        else:
            for i in range(tensor._numel):
                var x = Float64(in_ptr[i])
                var erf_val = erf(x / Float64(SQRT_2))
                out_ptr[i] = Scalar[dtype](x * 0.5 * (1.0 + erf_val))


def dispatch_gelu(tensor: AnyTensor, approximate: Bool) raises -> AnyTensor:
    """Runtime dispatch for GELU activation.

    Args:
        tensor: Input tensor.
        approximate: Use tanh approximation if True.

    Returns:
        GELU output tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float16:
        _gelu_impl[DType.float16](result, tensor, approximate)
    elif tensor._dtype == DType.float32:
        _gelu_impl[DType.float32](result, tensor, approximate)
    elif tensor._dtype == DType.float64:
        _gelu_impl[DType.float64](result, tensor, approximate)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_gelu: only supports float16/32/64. Got " + dtype_name
        )

    return result^


def _gelu_backward_impl[
    dtype: DType
](
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, approximate: Bool
) raises:
    """Compile-time specialized GELU backward implementation.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output gradient tensor (pre-allocated).
        grad_output: Upstream gradient.
        x: Input from forward pass.
        approximate: Use tanh approximation derivative if True.
    """
    from std.math import exp, erf, tanh as math_tanh

    comptime SQRT_2 = 1.4142135623730951
    comptime SQRT_2_OVER_PI = 0.7978845608028654
    comptime GELU_COEFF = 0.044715
    comptime INV_SQRT_2PI = 0.3989422804014327

    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        if approximate:
            for i in range(x._numel):
                var x_val = Float32(x_ptr[i])
                var grad = Float32(grad_ptr[i])
                var x_cubed = x_val * x_val * x_val
                var inner = Float32(SQRT_2_OVER_PI) * (
                    x_val + Float32(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val
                var dtanh = Float32(SQRT_2_OVER_PI) * (
                    1.0 + 3.0 * Float32(GELU_COEFF) * x_val * x_val
                )
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh
                result_ptr[i] = Scalar[dtype](grad * dgelu)
        else:
            for i in range(x._numel):
                var x_val = Float32(x_ptr[i])
                var grad = Float32(grad_ptr[i])
                var erf_val = erf(x_val / Float32(SQRT_2))
                var pdf = Float32(INV_SQRT_2PI) * exp(-0.5 * x_val * x_val)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf
                result_ptr[i] = Scalar[dtype](grad * dgelu)
    elif dtype == DType.float32:
        if approximate:
            for i in range(x._numel):
                var x_val = Float32(x_ptr[i])
                var grad = Float32(grad_ptr[i])
                var x_cubed = x_val * x_val * x_val
                var inner = Float32(SQRT_2_OVER_PI) * (
                    x_val + Float32(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val
                var dtanh = Float32(SQRT_2_OVER_PI) * (
                    1.0 + 3.0 * Float32(GELU_COEFF) * x_val * x_val
                )
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh
                result_ptr[i] = Scalar[dtype](grad * dgelu)
        else:
            for i in range(x._numel):
                var x_val = Float32(x_ptr[i])
                var grad = Float32(grad_ptr[i])
                var erf_val = erf(x_val / Float32(SQRT_2))
                var pdf = Float32(INV_SQRT_2PI) * exp(-0.5 * x_val * x_val)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf
                result_ptr[i] = Scalar[dtype](grad * dgelu)
    else:  # float64
        if approximate:
            for i in range(x._numel):
                var x_val = Float64(x_ptr[i])
                var grad = Float64(grad_ptr[i])
                var x_cubed = x_val * x_val * x_val
                var inner = Float64(SQRT_2_OVER_PI) * (
                    x_val + Float64(GELU_COEFF) * x_cubed
                )
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val
                var dtanh = Float64(SQRT_2_OVER_PI) * (
                    1.0 + 3.0 * Float64(GELU_COEFF) * x_val * x_val
                )
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh
                result_ptr[i] = Scalar[dtype](grad * dgelu)
        else:
            for i in range(x._numel):
                var x_val = Float64(x_ptr[i])
                var grad = Float64(grad_ptr[i])
                var erf_val = erf(x_val / Float64(SQRT_2))
                var pdf = Float64(INV_SQRT_2PI) * exp(-0.5 * x_val * x_val)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf
                result_ptr[i] = Scalar[dtype](grad * dgelu)


def dispatch_gelu_backward(
    grad_output: AnyTensor, x: AnyTensor, approximate: Bool
) raises -> AnyTensor:
    """Runtime dispatch for GELU backward operation.

    Args:
        grad_output: Upstream gradient.
        x: Input from forward pass.
        approximate: Use tanh approximation derivative if True.

    Returns:
        Input gradient tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(x.shape(), x._dtype)

    if x._dtype == DType.float16:
        _gelu_backward_impl[DType.float16](result, grad_output, x, approximate)
    elif x._dtype == DType.float32:
        _gelu_backward_impl[DType.float32](result, grad_output, x, approximate)
    elif x._dtype == DType.float64:
        _gelu_backward_impl[DType.float64](result, grad_output, x, approximate)
    else:
        var dtype_name = _format_dtype_name(x._dtype)
        raise Error(
            "dispatch_gelu_backward: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _hard_sigmoid_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor) raises:
    """Compile-time specialized hard_sigmoid implementation.

    hard_sigmoid(x) = clip((x + 3) / 6, 0, 1).

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output tensor (pre-allocated).
        tensor: Input tensor.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        for i in range(tensor._numel):
            var x = Float32(in_ptr[i])
            var val = (x + 3.0) / 6.0
            val = max(Float32(0.0), min(Float32(1.0), val))
            out_ptr[i] = Scalar[dtype](val)
    else:
        var zero = Scalar[dtype](0.0)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)
        var six = Scalar[dtype](6.0)
        for i in range(tensor._numel):
            var x = in_ptr[i]
            var val = (x + three) / six
            out_ptr[i] = max(zero, min(one, val))


def dispatch_hard_sigmoid(tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch for hard_sigmoid activation.

    Args:
        tensor: Input tensor.

    Returns:
        Hard_sigmoid output tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float16:
        _hard_sigmoid_impl[DType.float16](result, tensor)
    elif tensor._dtype == DType.float32:
        _hard_sigmoid_impl[DType.float32](result, tensor)
    elif tensor._dtype == DType.float64:
        _hard_sigmoid_impl[DType.float64](result, tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_hard_sigmoid: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _hard_sigmoid_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor) raises:
    """Compile-time specialized hard_sigmoid backward implementation.

    Derivative: 1/6 if -3 < x < 3, else 0.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output gradient tensor (pre-allocated).
        grad_output: Upstream gradient.
        x: Input from forward pass.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        for i in range(x._numel):
            var x_val = Float32(x_ptr[i])
            var grad = Float32(grad_ptr[i])
            if x_val > -3.0 and x_val < 3.0:
                result_ptr[i] = Scalar[dtype](grad / 6.0)
            else:
                result_ptr[i] = Scalar[dtype](0.0)
    else:
        var zero = Scalar[dtype](0.0)
        var neg_three = Scalar[dtype](-3.0)
        var three = Scalar[dtype](3.0)
        var six = Scalar[dtype](6.0)
        for i in range(x._numel):
            var x_val = x_ptr[i]
            var grad = grad_ptr[i]
            if x_val > neg_three and x_val < three:
                result_ptr[i] = grad / six
            else:
                result_ptr[i] = zero


def dispatch_hard_sigmoid_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises -> AnyTensor:
    """Runtime dispatch for hard_sigmoid backward operation.

    Args:
        grad_output: Upstream gradient.
        x: Input from forward pass.

    Returns:
        Input gradient tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(x.shape(), x._dtype)

    if x._dtype == DType.float16:
        _hard_sigmoid_backward_impl[DType.float16](result, grad_output, x)
    elif x._dtype == DType.float32:
        _hard_sigmoid_backward_impl[DType.float32](result, grad_output, x)
    elif x._dtype == DType.float64:
        _hard_sigmoid_backward_impl[DType.float64](result, grad_output, x)
    else:
        var dtype_name = _format_dtype_name(x._dtype)
        raise Error(
            "dispatch_hard_sigmoid_backward: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _hard_swish_impl[dtype: DType](result: AnyTensor, tensor: AnyTensor) raises:
    """Compile-time specialized hard_swish implementation.

    hard_swish(x) = x * hard_sigmoid(x) = x * clip((x + 3) / 6, 0, 1).
                  = 0 if x <= -3.
                  = x if x >= 3.
                  = x * (x + 3) / 6 otherwise.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output tensor (pre-allocated).
        tensor: Input tensor.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        for i in range(tensor._numel):
            var x = Float32(in_ptr[i])
            if x <= -3.0:
                out_ptr[i] = Scalar[dtype](0.0)
            elif x >= 3.0:
                out_ptr[i] = Scalar[dtype](x)
            else:
                out_ptr[i] = Scalar[dtype](x * (x + 3.0) / 6.0)
    else:
        var zero = Scalar[dtype](0.0)
        var neg_three = Scalar[dtype](-3.0)
        var three = Scalar[dtype](3.0)
        var six = Scalar[dtype](6.0)
        for i in range(tensor._numel):
            var x = in_ptr[i]
            if x <= neg_three:
                out_ptr[i] = zero
            elif x >= three:
                out_ptr[i] = x
            else:
                out_ptr[i] = x * (x + three) / six


def dispatch_hard_swish(tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch for hard_swish activation.

    Args:
        tensor: Input tensor.

    Returns:
        Hard_swish output tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float16:
        _hard_swish_impl[DType.float16](result, tensor)
    elif tensor._dtype == DType.float32:
        _hard_swish_impl[DType.float32](result, tensor)
    elif tensor._dtype == DType.float64:
        _hard_swish_impl[DType.float64](result, tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_hard_swish: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _hard_swish_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor) raises:
    """Compile-time specialized hard_swish backward implementation.

    Derivative: 0 if x <= -3, 1 if x >= 3, (2x + 3) / 6 otherwise.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output gradient tensor (pre-allocated).
        grad_output: Upstream gradient.
        x: Input from forward pass.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    comptime if dtype == DType.float16:
        for i in range(x._numel):
            var x_val = Float32(x_ptr[i])
            var grad = Float32(grad_ptr[i])
            if x_val <= -3.0:
                result_ptr[i] = Scalar[dtype](0.0)
            elif x_val >= 3.0:
                result_ptr[i] = Scalar[dtype](grad)
            else:
                result_ptr[i] = Scalar[dtype](grad * (2.0 * x_val + 3.0) / 6.0)
    else:
        var zero = Scalar[dtype](0.0)
        var neg_three = Scalar[dtype](-3.0)
        var three = Scalar[dtype](3.0)
        var two = Scalar[dtype](2.0)
        var six = Scalar[dtype](6.0)
        for i in range(x._numel):
            var x_val = x_ptr[i]
            var grad = grad_ptr[i]
            if x_val <= neg_three:
                result_ptr[i] = zero
            elif x_val >= three:
                result_ptr[i] = grad
            else:
                result_ptr[i] = grad * (two * x_val + three) / six


def dispatch_hard_swish_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises -> AnyTensor:
    """Runtime dispatch for hard_swish backward operation.

    Args:
        grad_output: Upstream gradient.
        x: Input from forward pass.

    Returns:
        Input gradient tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(x.shape(), x._dtype)

    if x._dtype == DType.float16:
        _hard_swish_backward_impl[DType.float16](result, grad_output, x)
    elif x._dtype == DType.float32:
        _hard_swish_backward_impl[DType.float32](result, grad_output, x)
    elif x._dtype == DType.float64:
        _hard_swish_backward_impl[DType.float64](result, grad_output, x)
    else:
        var dtype_name = _format_dtype_name(x._dtype)
        raise Error(
            "dispatch_hard_swish_backward: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


def _hard_tanh_impl[
    dtype: DType
](
    result: AnyTensor, tensor: AnyTensor, min_val: Float64, max_val: Float64
) raises:
    """Compile-time specialized hard_tanh implementation.

    hard_tanh(x) = clip(x, min_val, max_val).

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output tensor (pre-allocated).
        tensor: Input tensor.
        min_val: Minimum output value.
        max_val: Maximum output value.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var min_typed = Scalar[dtype](min_val)
    var max_typed = Scalar[dtype](max_val)

    for i in range(tensor._numel):
        var x = in_ptr[i]
        out_ptr[i] = max(min_typed, min(max_typed, x))


def dispatch_hard_tanh(
    tensor: AnyTensor, min_val: Float64, max_val: Float64
) raises -> AnyTensor:
    """Runtime dispatch for hard_tanh activation.

    Args:
        tensor: Input tensor.
        min_val: Minimum output value.
        max_val: Maximum output value.

    Returns:
        Hard_tanh output tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float16:
        _hard_tanh_impl[DType.float16](result, tensor, min_val, max_val)
    elif tensor._dtype == DType.float32:
        _hard_tanh_impl[DType.float32](result, tensor, min_val, max_val)
    elif tensor._dtype == DType.float64:
        _hard_tanh_impl[DType.float64](result, tensor, min_val, max_val)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_hard_tanh: only supports float16/32/64. Got " + dtype_name
        )

    return result^


def _hard_tanh_backward_impl[
    dtype: DType
](
    result: AnyTensor,
    grad_output: AnyTensor,
    x: AnyTensor,
    min_val: Float64,
    max_val: Float64,
) raises:
    """Compile-time specialized hard_tanh backward implementation.

    Derivative: 1 if min_val < x < max_val, else 0.

    Parameters:
        dtype: Compile-time dtype parameter.
    Args:
        result: Output gradient tensor (pre-allocated).
        grad_output: Upstream gradient.
        x: Input from forward pass.
        min_val: Minimum value used in forward pass.
        max_val: Maximum value used in forward pass.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    var min_typed = Scalar[dtype](min_val)
    var max_typed = Scalar[dtype](max_val)
    var zero = Scalar[dtype](0.0)

    for i in range(x._numel):
        var x_val = x_ptr[i]
        var grad = grad_ptr[i]
        if x_val > min_typed and x_val < max_typed:
            result_ptr[i] = grad
        else:
            result_ptr[i] = zero


def dispatch_hard_tanh_backward(
    grad_output: AnyTensor, x: AnyTensor, min_val: Float64, max_val: Float64
) raises -> AnyTensor:
    """Runtime dispatch for hard_tanh backward operation.

    Args:
        grad_output: Upstream gradient.
        x: Input from forward pass.
        min_val: Minimum value used in forward pass.
        max_val: Maximum value used in forward pass.

    Returns:
        Input gradient tensor.

    Raises:
        Error: If dtype is not float16/32/64.
    """
    var result = AnyTensor(x.shape(), x._dtype)

    if x._dtype == DType.float16:
        _hard_tanh_backward_impl[DType.float16](
            result, grad_output, x, min_val, max_val
        )
    elif x._dtype == DType.float32:
        _hard_tanh_backward_impl[DType.float32](
            result, grad_output, x, min_val, max_val
        )
    elif x._dtype == DType.float64:
        _hard_tanh_backward_impl[DType.float64](
            result, grad_output, x, min_val, max_val
        )
    else:
        var dtype_name = _format_dtype_name(x._dtype)
        raise Error(
            "dispatch_hard_tanh_backward: only supports float16/32/64. Got "
            + dtype_name
        )

    return result^


# ============================================================================
# Typed-Kernel Dispatch (in-place result, integer shape args)
# ============================================================================
#
# Matrix/linear-algebra kernels do not fit the unary/binary `op` shape: they
# write into a pre-allocated `result` tensor and take integer shape arguments
# instead of returning a new tensor. These helpers provide the same 12-branch
# ordinal dispatch for that family of kernels, keyed on an explicit `dtype`
# so the caller chooses which operand drives specialization.


def dispatch_kernel_3t1i[
    kernel: def[T: DType](AnyTensor, AnyTensor, AnyTensor, Int) thin -> None
](dtype: DType, t0: AnyTensor, t1: AnyTensor, t2: AnyTensor, i0: Int) raises:
    """Dispatch a (3-tensor, 1-int) typed kernel by runtime dtype.

    Parameters:
        kernel: Compile-time specialized kernel taking three tensors and one
            integer argument.

    Args:
        dtype: Runtime dtype that selects the compile-time specialization.
        t0: First tensor argument (typically the output).
        t1: Second tensor argument.
        t2: Third tensor argument.
        i0: Integer shape argument.

    Raises:
        Error: If `dtype` is not one of the 11 supported dtypes.
    """
    var ordinal = dtype_to_ordinal(dtype)
    if ordinal == DTYPE_FLOAT16:
        kernel[DType.float16](t0, t1, t2, i0)
    elif ordinal == DTYPE_FLOAT32:
        kernel[DType.float32](t0, t1, t2, i0)
    elif ordinal == DTYPE_FLOAT64:
        kernel[DType.float64](t0, t1, t2, i0)
    elif ordinal == DTYPE_INT8:
        kernel[DType.int8](t0, t1, t2, i0)
    elif ordinal == DTYPE_INT16:
        kernel[DType.int16](t0, t1, t2, i0)
    elif ordinal == DTYPE_INT32:
        kernel[DType.int32](t0, t1, t2, i0)
    elif ordinal == DTYPE_INT64:
        kernel[DType.int64](t0, t1, t2, i0)
    elif ordinal == DTYPE_UINT8:
        kernel[DType.uint8](t0, t1, t2, i0)
    elif ordinal == DTYPE_UINT16:
        kernel[DType.uint16](t0, t1, t2, i0)
    elif ordinal == DTYPE_UINT32:
        kernel[DType.uint32](t0, t1, t2, i0)
    elif ordinal == DTYPE_UINT64:
        kernel[DType.uint64](t0, t1, t2, i0)
    else:
        raise Error(
            "dispatch_kernel_3t1i: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'"
        )


def dispatch_kernel_3t2i[
    kernel: def[T: DType](
        AnyTensor, AnyTensor, AnyTensor, Int, Int
    ) thin -> None
](
    dtype: DType,
    t0: AnyTensor,
    t1: AnyTensor,
    t2: AnyTensor,
    i0: Int,
    i1: Int,
) raises:
    """Dispatch a (3-tensor, 2-int) typed kernel by runtime dtype.

    Parameters:
        kernel: Compile-time specialized kernel taking three tensors and two
            integer arguments.

    Args:
        dtype: Runtime dtype that selects the compile-time specialization.
        t0: First tensor argument (typically the output).
        t1: Second tensor argument.
        t2: Third tensor argument.
        i0: First integer shape argument.
        i1: Second integer shape argument.

    Raises:
        Error: If `dtype` is not one of the 11 supported dtypes.
    """
    var ordinal = dtype_to_ordinal(dtype)
    if ordinal == DTYPE_FLOAT16:
        kernel[DType.float16](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_FLOAT32:
        kernel[DType.float32](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_FLOAT64:
        kernel[DType.float64](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_INT8:
        kernel[DType.int8](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_INT16:
        kernel[DType.int16](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_INT32:
        kernel[DType.int32](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_INT64:
        kernel[DType.int64](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_UINT8:
        kernel[DType.uint8](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_UINT16:
        kernel[DType.uint16](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_UINT32:
        kernel[DType.uint32](t0, t1, t2, i0, i1)
    elif ordinal == DTYPE_UINT64:
        kernel[DType.uint64](t0, t1, t2, i0, i1)
    else:
        raise Error(
            "dispatch_kernel_3t2i: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'"
        )


def dispatch_kernel_3t3i[
    kernel: def[T: DType](
        AnyTensor, AnyTensor, AnyTensor, Int, Int, Int
    ) thin -> None
](
    dtype: DType,
    t0: AnyTensor,
    t1: AnyTensor,
    t2: AnyTensor,
    i0: Int,
    i1: Int,
    i2: Int,
) raises:
    """Dispatch a (3-tensor, 3-int) typed kernel by runtime dtype.

    Parameters:
        kernel: Compile-time specialized kernel taking three tensors and three
            integer arguments.

    Args:
        dtype: Runtime dtype that selects the compile-time specialization.
        t0: First tensor argument (typically the output).
        t1: Second tensor argument.
        t2: Third tensor argument.
        i0: First integer shape argument.
        i1: Second integer shape argument.
        i2: Third integer shape argument.

    Raises:
        Error: If `dtype` is not one of the 11 supported dtypes.
    """
    var ordinal = dtype_to_ordinal(dtype)
    if ordinal == DTYPE_FLOAT16:
        kernel[DType.float16](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_FLOAT32:
        kernel[DType.float32](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_FLOAT64:
        kernel[DType.float64](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_INT8:
        kernel[DType.int8](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_INT16:
        kernel[DType.int16](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_INT32:
        kernel[DType.int32](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_INT64:
        kernel[DType.int64](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_UINT8:
        kernel[DType.uint8](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_UINT16:
        kernel[DType.uint16](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_UINT32:
        kernel[DType.uint32](t0, t1, t2, i0, i1, i2)
    elif ordinal == DTYPE_UINT64:
        kernel[DType.uint64](t0, t1, t2, i0, i1, i2)
    else:
        raise Error(
            "dispatch_kernel_3t3i: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'"
        )


# ============================================================================
# Float-Only Closure Dispatch (arity-agnostic)
# ============================================================================
#
# Some kernels (the normalization family) take wildly varied argument lists —
# mixes of UnsafePointer, AnyTensor, Int, and Float64 with arities from 7 to
# 12. A fixed-signature dispatcher cannot cover them. `dispatch_float3` instead
# takes a parametric capturing closure: the caller writes a `@parameter` body
# that closes over its locals, and this helper supplies the runtime->comptime
# dtype branch. Float16/32/64 only.


def dispatch_float3[
    body: def[T: DType]() raises capturing[_] -> None
](dtype: DType) raises:
    """Run a float-specialized closure under runtime dtype dispatch.

    Parameters:
        body: A `@parameter` closure parameterized on a compile-time dtype.
            It closes over whatever arguments the kernel needs, so this helper
            is independent of kernel arity.

    Args:
        dtype: Runtime dtype; must be float16, float32, or float64.

    Raises:
        Error: If `dtype` is not a supported float type, or if `body` raises.
    """
    if dtype == DType.float16:
        body[DType.float16]()
    elif dtype == DType.float32:
        body[DType.float32]()
    elif dtype == DType.float64:
        body[DType.float64]()
    else:
        raise Error(
            "dispatch_float3: only supports float16/32/64. Got "
            + _format_dtype_name(dtype)
        )


# ============================================================================
# Tensor-Level Binary Operation Dispatch
# ============================================================================
#
# Dispatch mechanism for binary operations that work directly with tensor
# parameters (not scalar operations). These are used for operations like
# add, subtract, multiply, divide that need to dispatch on the tensor's dtype
# at runtime, then call a parametric wrapper function at compile-time.
#
# This pattern eliminates the manual if/elif chains in modules like
# arithmetic_contiguous.mojo and simplifies the elementwise.mojo dispatch layer.


def dispatch_binary_tensor[
    impl_fn: def[dtype: DType](AnyTensor, AnyTensor) raises thin -> AnyTensor
](lhs: AnyTensor, rhs: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to parametric binary tensor operation.

    This helper eliminates manual dtype branching for binary operations that
    take tensor parameters. Instead of writing:

        if dtype == DType.float32:
            return _op_typed[DType.float32](a, b).as_any()
        elif dtype == DType.float64:
            ...

    You write:

        return dispatch_binary_tensor[_op_typed](a, b)

    Parameters:
        impl_fn: Parametric binary tensor operation with signature
                 `def[dtype: DType](AnyTensor, AnyTensor) -> AnyTensor`

    Args:
        lhs: Left-hand side tensor.
        rhs: Right-hand side tensor (must have same dtype as lhs).

    Returns:
        Result tensor from the parametric implementation.

    Raises:
        Error: If dtypes don't match or dtype is unsupported.

    Note:
        Supports all dtypes: float16, float32, float64, int8, int16, int32,
        int64, uint8, uint16, uint32, uint64.
    """
    if lhs._dtype != rhs._dtype:
        var lhs_dtype = _format_dtype_name(lhs._dtype)
        var rhs_dtype = _format_dtype_name(rhs._dtype)
        raise Error(
            "dispatch_binary_tensor: dtypes must match. Got lhs="
            + lhs_dtype
            + ", rhs="
            + rhs_dtype
        )

    var dtype = lhs._dtype
    var ordinal = dtype_to_ordinal(dtype)

    if ordinal == DTYPE_FLOAT16:
        return impl_fn[DType.float16](lhs, rhs)
    elif ordinal == DTYPE_FLOAT32:
        return impl_fn[DType.float32](lhs, rhs)
    elif ordinal == DTYPE_FLOAT64:
        return impl_fn[DType.float64](lhs, rhs)
    elif ordinal == DTYPE_INT8:
        return impl_fn[DType.int8](lhs, rhs)
    elif ordinal == DTYPE_INT16:
        return impl_fn[DType.int16](lhs, rhs)
    elif ordinal == DTYPE_INT32:
        return impl_fn[DType.int32](lhs, rhs)
    elif ordinal == DTYPE_INT64:
        return impl_fn[DType.int64](lhs, rhs)
    elif ordinal == DTYPE_UINT8:
        return impl_fn[DType.uint8](lhs, rhs)
    elif ordinal == DTYPE_UINT16:
        return impl_fn[DType.uint16](lhs, rhs)
    elif ordinal == DTYPE_UINT32:
        return impl_fn[DType.uint32](lhs, rhs)
    elif ordinal == DTYPE_UINT64:
        return impl_fn[DType.uint64](lhs, rhs)
    else:
        raise Error(
            "dispatch_binary_tensor: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


def dispatch_unary_tensor[
    impl_fn: def[dtype: DType](AnyTensor) raises thin -> AnyTensor
](tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to parametric unary tensor operation.

    This helper eliminates manual dtype branching for unary operations that
    take a single tensor parameter. Similar to dispatch_binary_tensor but
    for single-tensor operations.

    Parameters:
        impl_fn: Parametric unary tensor operation with signature
                 `def[dtype: DType](AnyTensor) -> AnyTensor`

    Args:
        tensor: Input tensor.

    Returns:
        Result tensor from the parametric implementation.

    Raises:
        Error: If tensor dtype is unsupported.

    Note:
        Supports all dtypes: float16, float32, float64, int8, int16, int32,
        int64, uint8, uint16, uint32, uint64.
    """
    var dtype = tensor._dtype
    var ordinal = dtype_to_ordinal(dtype)

    if ordinal == DTYPE_FLOAT16:
        return impl_fn[DType.float16](tensor)
    elif ordinal == DTYPE_FLOAT32:
        return impl_fn[DType.float32](tensor)
    elif ordinal == DTYPE_FLOAT64:
        return impl_fn[DType.float64](tensor)
    elif ordinal == DTYPE_INT8:
        return impl_fn[DType.int8](tensor)
    elif ordinal == DTYPE_INT16:
        return impl_fn[DType.int16](tensor)
    elif ordinal == DTYPE_INT32:
        return impl_fn[DType.int32](tensor)
    elif ordinal == DTYPE_INT64:
        return impl_fn[DType.int64](tensor)
    elif ordinal == DTYPE_UINT8:
        return impl_fn[DType.uint8](tensor)
    elif ordinal == DTYPE_UINT16:
        return impl_fn[DType.uint16](tensor)
    elif ordinal == DTYPE_UINT32:
        return impl_fn[DType.uint32](tensor)
    elif ordinal == DTYPE_UINT64:
        return impl_fn[DType.uint64](tensor)
    else:
        raise Error(
            "dispatch_unary_tensor: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


def dispatch_tensor_all_dtypes[
    body: def[T: DType]() raises capturing[_] -> None
](dtype: DType) raises:
    """Dispatch for parametric functions with arbitrary arities/parameter types.

    Use this when the kernel function has variable parameters (like elementwise
    functions that mix result tensor, input tensors, strides, shapes, etc.).
    The caller writes a `@parameter` closure that closes over its arguments,
    and this helper supplies the dtype dispatch.

    This is similar to `dispatch_float3` but supports all dtypes.

    Parameters:
        body: A `@parameter` closure parameterized on a compile-time dtype.
            It closes over whatever arguments the kernel needs.

    Args:
        dtype: Runtime dtype (any of the supported types).

    Raises:
        Error: If `dtype` is not supported, or if `body` raises.

    Note:
        Supports all dtypes: float16, float32, float64, int8, int16, int32,
        int64, uint8, uint16, uint32, uint64.

    Example:
        ```mojo
        # Dispatch to a function with many parameters
        var dtype = tensor.dtype()
        @parameter
        fn kernel() raises:
            _complex_impl[dtype](result, a, b, strides, shapes, numel)
        dispatch_tensor_all_dtypes[kernel](dtype)
        ```
    """
    var ordinal = dtype_to_ordinal(dtype)

    if ordinal == DTYPE_FLOAT16:
        body[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        body[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        body[DType.float64]()
    elif ordinal == DTYPE_INT8:
        body[DType.int8]()
    elif ordinal == DTYPE_INT16:
        body[DType.int16]()
    elif ordinal == DTYPE_INT32:
        body[DType.int32]()
    elif ordinal == DTYPE_INT64:
        body[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        body[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        body[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        body[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        body[DType.uint64]()
    else:
        raise Error(
            "dispatch_tensor_all_dtypes: unsupported dtype '"
            + _format_dtype_name(dtype)
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )
