"""Typed Tensor[dtype] elementwise dispatch cores.

Internal module -- not part of the public API.
"""

from odyssey.tensor.tensor import Tensor
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.base.dtype_ordinal import (
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
# Layer 3 (Core): Generic Typed Unary Operation
# ============================================================================


def _unary_typed[
    dt: DType, op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](input: Tensor[dt]) raises -> Tensor[dt]:
    """Apply unary operation on native Tensor[dtype] -- zero dtype branches.

    This is the core implementation. Tensor[dt]._data is already typed as
    UnsafePointer[Scalar[dt], MutAnyOrigin], so no bitcasts are needed.

    Parameters:
        dt: Compile-time dtype parameter.
        op: Unary scalar operation function.

    Args:
        input: Input tensor (typed).

    Returns:
        Result Tensor[dt] with operation applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    for i in range(size):
        result._data[i] = op[dt](input._data[i])
    return result^


# ============================================================================
# Layer 2: AnyTensor Dispatch Helpers (delegates to typed core)
# ============================================================================


def _dispatch_unary_typed[
    op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed unary core (all dtypes).

    Converts AnyTensor to Tensor[dt], calls the typed implementation,
    and converts the result back to AnyTensor.

    Parameters:
        op: Unary operation function pointer.

    Args:
        tensor: Input tensor.

    Returns:
        Result tensor with operation applied element-wise.
    """
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _unary_typed[DType.float16, op](
            tensor.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _unary_typed[DType.float32, op](
            tensor.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _unary_typed[DType.float64, op](
            tensor.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _unary_typed[DType.int8, op](
            tensor.as_tensor[DType.int8]()
        ).as_any()
    elif ordinal == DTYPE_INT16:
        return _unary_typed[DType.int16, op](
            tensor.as_tensor[DType.int16]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _unary_typed[DType.int32, op](
            tensor.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _unary_typed[DType.int64, op](
            tensor.as_tensor[DType.int64]()
        ).as_any()
    elif ordinal == DTYPE_UINT8:
        return _unary_typed[DType.uint8, op](
            tensor.as_tensor[DType.uint8]()
        ).as_any()
    elif ordinal == DTYPE_UINT16:
        return _unary_typed[DType.uint16, op](
            tensor.as_tensor[DType.uint16]()
        ).as_any()
    elif ordinal == DTYPE_UINT32:
        return _unary_typed[DType.uint32, op](
            tensor.as_tensor[DType.uint32]()
        ).as_any()
    elif ordinal == DTYPE_UINT64:
        return _unary_typed[DType.uint64, op](
            tensor.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("Unsupported dtype for unary operation")


def _dispatch_float_unary_typed[
    op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed unary core (float dtypes only).

    Parameters:
        op: Unary operation function pointer.

    Args:
        tensor: Input tensor (must be float16/32/64).

    Returns:
        Result tensor with operation applied element-wise.
    """
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _unary_typed[DType.float16, op](
            tensor.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _unary_typed[DType.float32, op](
            tensor.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _unary_typed[DType.float64, op](
            tensor.as_tensor[DType.float64]()
        ).as_any()
    else:
        raise Error(
            "Unsupported dtype for float unary operation (requires"
            " float16/32/64)"
        )
