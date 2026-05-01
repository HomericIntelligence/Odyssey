"""Typed Tensor[dtype] arithmetic dispatch cores.

Internal module -- not part of the public API.
"""

from std.collections import List
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.broadcasting import broadcast_shapes, compute_broadcast_strides
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
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
# Layer 3 (Core): Native Tensor[dtype] Broadcasting Implementation
# ============================================================================


def _broadcast_binary_typed[
    dtype: DType, op: def[T: DType](Scalar[T], Scalar[T]) -> Scalar[T]
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Apply binary operation with broadcasting on native Tensor[dtype].

    This is the core implementation -- zero dtype branches, zero bitcasts.
    Tensor[dtype]._data is already typed as UnsafePointer[Scalar[dtype], MutAnyOrigin].

    Parameters:
        dtype: Compile-time dtype parameter.
        op: Binary operation function (e.g., add, subtract, multiply, divide).

    Args:
        a: First tensor (typed).
        b: Second tensor (typed).

    Returns:
        Result Tensor[dtype] with operation applied element-wise with broadcasting.
    """
    # Ensure inputs are contiguous before flat-buffer kernel access.
    # Non-contiguous views (e.g. from transpose) have non-unit strides that
    # are not reflected in flat index arithmetic, causing silent wrong results.
    # Tensor[dtype] lacks as_contiguous(), so we round-trip via AnyTensor only
    # when needed (rare path -- most tensors are contiguous).
    # NOTE: Function-scoped import to avoid circular dependency
    # (shared.tensor.typed -> shared.core -> shared.tensor.typed).
    from shared.core.shape import as_contiguous

    var a_any = a.as_any()
    var b_any = b.as_any()
    var a_cont = a_any if a.is_contiguous() else as_contiguous(a_any)
    var b_cont = b_any if b.is_contiguous() else as_contiguous(b_any)

    # Compute broadcast shape
    var result_shape = broadcast_shapes(a_cont.shape(), b_cont.shape())
    var result = Tensor[dtype](result_shape)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a_cont.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b_cont.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides = List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final = List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Get typed pointers -- Tensor[dtype]._data is already typed, but
    # a_cont/b_cont are AnyTensor (from contiguity check), so bitcast from
    # their UInt8 storage. Result uses native typed pointer directly.
    var a_ptr = a_cont._data.bitcast[Scalar[dtype]]()
    var b_ptr = b_cont._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform operation with zero overhead (no dtype conversion!)
        result_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    return result^


# ============================================================================
# Layer 2: AnyTensor Broadcasting Helper (delegates to typed core)
# ============================================================================


def _broadcast_binary[
    dtype: DType, op: def[T: DType](Scalar[T], Scalar[T]) -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply binary operation with broadcasting via typed core.

    Converts AnyTensor inputs to Tensor[dtype], calls the typed implementation,
    and converts the result back to AnyTensor.

    Parameters:
        dtype: Compile-time dtype parameter.
        op: Binary operation function (e.g., add, subtract, multiply, divide).

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Result tensor with operation applied element-wise with broadcasting.
    """
    return _broadcast_binary_typed[dtype, op](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


# ============================================================================
# Layer 2: Runtime Dtype Dispatch (ordinal-based jump table)
# ============================================================================


def _dispatch_broadcast_binary[
    op: def[T: DType](Scalar[T], Scalar[T]) -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized Tensor[dtype] implementation.

    Performs runtime dtype checking and dispatches to the typed core via
    ordinal-based lookup (compiler can generate jump table).

    Parameters:
        op: Binary operation function pointer.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Result tensor with operation applied with broadcasting.

    Raises:
        Error: If dtypes don't match or are unsupported.
    """
    # Validate dtypes match
    if a._dtype != b._dtype:
        raise Error("Cannot operate on tensors with different dtypes")

    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(a._dtype)

    # Dispatch based on ordinal - compiler generates jump table for consecutive integers
    if ordinal == DTYPE_FLOAT16:
        return _broadcast_binary[DType.float16, op](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _broadcast_binary[DType.float32, op](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _broadcast_binary[DType.float64, op](a, b)
    elif ordinal == DTYPE_INT8:
        return _broadcast_binary[DType.int8, op](a, b)
    elif ordinal == DTYPE_INT16:
        return _broadcast_binary[DType.int16, op](a, b)
    elif ordinal == DTYPE_INT32:
        return _broadcast_binary[DType.int32, op](a, b)
    elif ordinal == DTYPE_INT64:
        return _broadcast_binary[DType.int64, op](a, b)
    elif ordinal == DTYPE_UINT8:
        return _broadcast_binary[DType.uint8, op](a, b)
    elif ordinal == DTYPE_UINT16:
        return _broadcast_binary[DType.uint16, op](a, b)
    elif ordinal == DTYPE_UINT32:
        return _broadcast_binary[DType.uint32, op](a, b)
    elif ordinal == DTYPE_UINT64:
        return _broadcast_binary[DType.uint64, op](a, b)
    else:
        raise Error("Unsupported dtype for binary operation")


def _multiply_scalar_typed[
    dt: DType
](tensor: Tensor[dt], scalar: Float32) raises -> Tensor[dt]:
    """Native typed scalar multiplication (Layer 3 core).

    Multiplies each element by a scalar without creating an intermediate
    full tensor. Zero dtype branches -- pointer is already typed.

    Args:
        tensor: Input tensor (typed).
        scalar: Scalar value to multiply by.

    Returns:
        A new Tensor[dt] with each element multiplied by the scalar.
    """
    # Ensure input is contiguous before flat-buffer kernel access.
    # Tensor[dt] lacks as_contiguous() so round-trip via AnyTensor if needed.
    # NOTE: Function-scoped import to avoid circular dependency.
    from shared.core.shape import as_contiguous

    var t_any = tensor.as_any()
    var t_cont = t_any if tensor.is_contiguous() else as_contiguous(t_any)

    var result = Tensor[dt](t_cont.shape())
    var numel = result.numel()

    var input_ptr = t_cont._data.bitcast[Scalar[dt]]()
    var result_ptr = result._data
    var scalar_cast = Scalar[dt](scalar)
    for i in range(numel):
        result_ptr[i] = input_ptr[i] * scalar_cast

    return result^


def _dispatch_multiply_scalar(
    tensor: AnyTensor, scalar: Float32
) raises -> AnyTensor:
    """Runtime dispatch for scalar multiplication."""
    var ordinal = dtype_to_ordinal(tensor.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _multiply_scalar_typed[DType.float16](
            tensor.as_tensor[DType.float16](), scalar
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _multiply_scalar_typed[DType.float32](
            tensor.as_tensor[DType.float32](), scalar
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _multiply_scalar_typed[DType.float64](
            tensor.as_tensor[DType.float64](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _multiply_scalar_typed[DType.int8](
            tensor.as_tensor[DType.int8](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT16:
        return _multiply_scalar_typed[DType.int16](
            tensor.as_tensor[DType.int16](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _multiply_scalar_typed[DType.int32](
            tensor.as_tensor[DType.int32](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _multiply_scalar_typed[DType.int64](
            tensor.as_tensor[DType.int64](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT8:
        return _multiply_scalar_typed[DType.uint8](
            tensor.as_tensor[DType.uint8](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT16:
        return _multiply_scalar_typed[DType.uint16](
            tensor.as_tensor[DType.uint16](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT32:
        return _multiply_scalar_typed[DType.uint32](
            tensor.as_tensor[DType.uint32](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT64:
        return _multiply_scalar_typed[DType.uint64](
            tensor.as_tensor[DType.uint64](), scalar
        ).as_any()
    else:
        raise Error("Unsupported dtype for multiply_scalar operation")
