"""Typed Tensor[dtype] reduction dispatch cores.

Internal module -- not part of the public API.
"""

from collections import List
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64,
)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] Typed Implementations
# ============================================================================


fn _sum_typed[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Native typed sum reduction (Layer 3 core).

    Args:
        tensor: Input typed tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with sum along specified axis.
    """
    from shared.core.reduction import sum

    var t_any = tensor.as_any()
    var result_any = sum(t_any, axis, keepdims)
    return result_any.as_tensor[dt]()


fn _mean_typed[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Native typed mean reduction (Layer 3 core).

    Args:
        tensor: Input typed tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with mean along specified axis.
    """
    from shared.core.reduction import mean

    var t_any = tensor.as_any()
    var result_any = mean(t_any, axis, keepdims)
    return result_any.as_tensor[dt]()


fn _max_reduce_typed[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Native typed max reduction (Layer 3 core).

    Args:
        tensor: Input typed tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with max along specified axis.
    """
    from shared.core.reduction import max_reduce

    var t_any = tensor.as_any()
    var result_any = max_reduce(t_any, axis, keepdims)
    return result_any.as_tensor[dt]()


fn _min_reduce_typed[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Native typed min reduction (Layer 3 core).

    Args:
        tensor: Input typed tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with min along specified axis.
    """
    from shared.core.reduction import min_reduce

    var t_any = tensor.as_any()
    var result_any = min_reduce(t_any, axis, keepdims)
    return result_any.as_tensor[dt]()


# ============================================================================
# Layer 2: Ordinal-Based Dispatch for Typed Reductions
# ============================================================================


fn _dispatch_sum_typed(
    tensor: AnyTensor, axis: Int = -1, keepdims: Bool = False
) raises -> AnyTensor:
    """Runtime dispatch to typed sum via ordinal-based lookup.

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        Sum reduction result.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _sum_typed[DType.float16](
            tensor.as_tensor[DType.float16](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _sum_typed[DType.float32](
            tensor.as_tensor[DType.float32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _sum_typed[DType.float64](
            tensor.as_tensor[DType.float64](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _sum_typed[DType.int32](
            tensor.as_tensor[DType.int32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _sum_typed[DType.int64](
            tensor.as_tensor[DType.int64](), axis, keepdims
        ).as_any()
    else:
        raise Error("sum: unsupported dtype")


fn _dispatch_mean_typed(
    tensor: AnyTensor, axis: Int = -1, keepdims: Bool = False
) raises -> AnyTensor:
    """Runtime dispatch to typed mean via ordinal-based lookup.

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        Mean reduction result.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _mean_typed[DType.float16](
            tensor.as_tensor[DType.float16](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _mean_typed[DType.float32](
            tensor.as_tensor[DType.float32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _mean_typed[DType.float64](
            tensor.as_tensor[DType.float64](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _mean_typed[DType.int32](
            tensor.as_tensor[DType.int32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _mean_typed[DType.int64](
            tensor.as_tensor[DType.int64](), axis, keepdims
        ).as_any()
    else:
        raise Error("mean: unsupported dtype")


fn _dispatch_max_reduce_typed(
    tensor: AnyTensor, axis: Int = -1, keepdims: Bool = False
) raises -> AnyTensor:
    """Runtime dispatch to typed max_reduce via ordinal-based lookup.

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        Max reduction result.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _max_reduce_typed[DType.float16](
            tensor.as_tensor[DType.float16](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _max_reduce_typed[DType.float32](
            tensor.as_tensor[DType.float32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _max_reduce_typed[DType.float64](
            tensor.as_tensor[DType.float64](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _max_reduce_typed[DType.int32](
            tensor.as_tensor[DType.int32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _max_reduce_typed[DType.int64](
            tensor.as_tensor[DType.int64](), axis, keepdims
        ).as_any()
    else:
        raise Error("max_reduce: unsupported dtype")


fn _dispatch_min_reduce_typed(
    tensor: AnyTensor, axis: Int = -1, keepdims: Bool = False
) raises -> AnyTensor:
    """Runtime dispatch to typed min_reduce via ordinal-based lookup.

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        Min reduction result.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _min_reduce_typed[DType.float16](
            tensor.as_tensor[DType.float16](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _min_reduce_typed[DType.float32](
            tensor.as_tensor[DType.float32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _min_reduce_typed[DType.float64](
            tensor.as_tensor[DType.float64](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _min_reduce_typed[DType.int32](
            tensor.as_tensor[DType.int32](), axis, keepdims
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _min_reduce_typed[DType.int64](
            tensor.as_tensor[DType.int64](), axis, keepdims
        ).as_any()
    else:
        raise Error("min_reduce: unsupported dtype")
