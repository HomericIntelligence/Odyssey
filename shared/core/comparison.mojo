"""Comparison operations with native Tensor[dtype] implementations.

Implements element-wise comparison operations following NumPy-style broadcasting.

Architecture: Tensor[dtype] typed implementations are the core (zero dtype branches).
AnyTensor versions dispatch to typed implementations via ordinal-based table.

Layer 1 (outer): AnyTensor public API (equal, less, greater, etc.)
Layer 2: dtype dispatch table (ordinal-based)
Layer 3 (core): Tensor[dtype] native implementation
"""

from collections import List
from .any_tensor import AnyTensor
from shared.tensor.tensor import Tensor
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
# Layer 3 (Core): Native Tensor[dtype] Comparison Implementations
# ============================================================================
# Each comparison op has its own typed core function that uses Tensor[dtype]._data
# directly -- zero bitcasts, zero dtype branches.


fn _equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise equality on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] == b_ptr[idx_b]

    return result^


fn _not_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise inequality on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] != b_ptr[idx_b]

    return result^


fn _less_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise less-than on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] < b_ptr[idx_b]

    return result^


fn _less_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise less-equal on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] <= b_ptr[idx_b]

    return result^


fn _greater_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise greater-than on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] > b_ptr[idx_b]

    return result^


fn _greater_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise greater-equal on native Tensor[dtype] (core).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        a: First typed tensor.
        b: Second typed tensor.

    Returns:
        Boolean result tensor.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] >= b_ptr[idx_b]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch helpers (as_tensor -> typed core -> as_any)
# ============================================================================


fn _equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


fn _not_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _not_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


fn _less_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _less_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


fn _less_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _less_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


fn _greater_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _greater_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


fn _greater_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _greater_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


# ============================================================================
# Layer 1: AnyTensor Public API
# ============================================================================


fn equal(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise equality comparison with broadcasting.

    Performs exact equality comparison on all supported dtypes. Follows IEEE 754
    semantics for floating-point comparisons:
    - NaN != NaN (per IEEE 754 standard)
    - Positive infinity == positive infinity
    - Negative infinity == negative infinity
    - Subnormal numbers are compared exactly

    Note on Precision:
    For floating-point dtypes, equality uses exact binary comparison, not
    tolerance-based comparison. This means that values that are mathematically
    equal but have different floating-point representations will compare as
    unequal. For tolerance-based comparison, users should implement
    custom logic using subtraction and comparison with a tolerance threshold.

    Example (precision loss):
        ```mojo
        # These may not be equal due to floating-point arithmetic
        var x = full([3], 0.1, DType.float32)
        var y = divide(full([3], 1.0, DType.float32),
                       full([3], 10.0, DType.float32))
        var result = equal(x, y)  # May contain False values due to rounding
        ```

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a == b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        ```mojo
        var a = full([3, 4], 2.0, DType.float32)
        var b = full([3, 4], 2.0, DType.float32)
        var c = equal(a, b)  # Shape (3, 4), all True
        ```
    """
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

    Examples:
        ```mojo
        var a = full([3, 4], 2.0, DType.float32)
        var b = full([3, 4], 3.0, DType.float32)
        var c = not_equal(a, b)  # Shape (3, 4), all True
        ```
    """
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

    Examples:
        ```mojo
        var a = full([3, 4], 2.0, DType.float32)
        var b = full([3, 4], 3.0, DType.float32)
        var c = less(a, b)  # Shape (3, 4), all True
        ```
    """
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

    Examples:
        ```mojo
        var a = full([3, 4], 2.0, DType.float32)
        var b = full([3, 4], 2.0, DType.float32)
        var c = less_equal(a, b)  # Shape (3, 4), all True
        ```
    """
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

    Examples:
        ```mojo
        var a = full([3, 4], 3.0, DType.float32)
        var b = full([3, 4], 2.0, DType.float32)
        var c = greater(a, b)  # Shape (3, 4), all True
        ```
    """
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

    Examples:
        ```mojo
        var a = full([3, 4], 3.0, DType.float32)
        var b = full([3, 4], 3.0, DType.float32)
        var c = greater_equal(a, b)  # Shape (3, 4), all True
        ```
    """
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


# ============================================================================
# Typed Tensor[dtype] overloads — delegate to typed cores directly
# ============================================================================


fn equal_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise equality comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise equality results.
    """
    return _equal_typed[dt](a, b)


fn not_equal_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise inequality comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise inequality results.
    """
    return _not_equal_typed[dt](a, b)


fn less_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise less-than comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise less-than results.
    """
    return _less_typed[dt](a, b)


fn less_equal_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise less-than-or-equal comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise less-equal results.
    """
    return _less_equal_typed[dt](a, b)


fn greater_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise greater-than comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise greater-than results.
    """
    return _greater_typed[dt](a, b)


fn greater_equal_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise greater-than-or-equal comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise greater-equal results.
    """
    return _greater_equal_typed[dt](a, b)
