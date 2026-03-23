"""Comparison operations for AnyTensor.

Implements element-wise comparison operations following NumPy-style broadcasting.
Typed Tensor[dtype] implementations live in shared/tensor/typed/comparison.mojo.
This file provides the AnyTensor public API only.
"""

from collections import List
from .any_tensor import AnyTensor
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
# Layer 1: AnyTensor Public API
# ============================================================================


fn equal(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise equality comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a == b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _equal_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    # Handle DType.bool separately (not in ordinal table)
    if a.dtype() == DType.bool:
        return _equal_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _equal_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _equal_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _equal_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _equal_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _equal_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _equal_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _equal_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _equal_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _equal_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _equal_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _equal_dispatch[DType.uint64](a, b)
    else:
        raise Error("equal: unsupported dtype")


fn not_equal(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise inequality comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a != b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _not_equal_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    if a.dtype() == DType.bool:
        return _not_equal_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _not_equal_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _not_equal_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _not_equal_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _not_equal_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _not_equal_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _not_equal_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _not_equal_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _not_equal_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _not_equal_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _not_equal_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _not_equal_dispatch[DType.uint64](a, b)
    else:
        raise Error("not_equal: unsupported dtype")


fn less(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise less-than comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a < b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _less_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    if a.dtype() == DType.bool:
        return _less_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _less_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _less_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _less_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _less_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _less_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _less_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _less_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _less_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _less_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _less_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _less_dispatch[DType.uint64](a, b)
    else:
        raise Error("less: unsupported dtype")


fn less_equal(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise less-than-or-equal comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a <= b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _less_equal_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    if a.dtype() == DType.bool:
        return _less_equal_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _less_equal_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _less_equal_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _less_equal_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _less_equal_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _less_equal_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _less_equal_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _less_equal_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _less_equal_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _less_equal_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _less_equal_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _less_equal_dispatch[DType.uint64](a, b)
    else:
        raise Error("less_equal: unsupported dtype")


fn greater(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise greater-than comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a > b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _greater_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    if a.dtype() == DType.bool:
        return _greater_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _greater_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _greater_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _greater_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _greater_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _greater_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _greater_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _greater_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _greater_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _greater_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _greater_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _greater_dispatch[DType.uint64](a, b)
    else:
        raise Error("greater: unsupported dtype")


fn greater_equal(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise greater-than-or-equal comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a >= b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match
    """
    from shared.tensor.typed.comparison import _greater_equal_dispatch

    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    if a.dtype() == DType.bool:
        return _greater_equal_dispatch[DType.bool](a, b)

    var ordinal = dtype_to_ordinal(a.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _greater_equal_dispatch[DType.float16](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _greater_equal_dispatch[DType.float32](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _greater_equal_dispatch[DType.float64](a, b)
    elif ordinal == DTYPE_INT8:
        return _greater_equal_dispatch[DType.int8](a, b)
    elif ordinal == DTYPE_INT16:
        return _greater_equal_dispatch[DType.int16](a, b)
    elif ordinal == DTYPE_INT32:
        return _greater_equal_dispatch[DType.int32](a, b)
    elif ordinal == DTYPE_INT64:
        return _greater_equal_dispatch[DType.int64](a, b)
    elif ordinal == DTYPE_UINT8:
        return _greater_equal_dispatch[DType.uint8](a, b)
    elif ordinal == DTYPE_UINT16:
        return _greater_equal_dispatch[DType.uint16](a, b)
    elif ordinal == DTYPE_UINT32:
        return _greater_equal_dispatch[DType.uint32](a, b)
    elif ordinal == DTYPE_UINT64:
        return _greater_equal_dispatch[DType.uint64](a, b)
    else:
        raise Error("greater_equal: unsupported dtype")
