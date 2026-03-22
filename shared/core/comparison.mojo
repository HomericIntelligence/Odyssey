"""Comparison operations for ExTensor with broadcasting support.

Implements element-wise comparison operations following NumPy-style broadcasting
"""

from collections import List
from .extensor import ExTensor
from shared.tensor.tensor import Tensor
from .broadcasting import broadcast_shapes, compute_broadcast_strides


# ============================================================================
# Dtype-specialized comparison helpers
# ============================================================================


@always_inline
fn _compare_equal_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized equal comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_equal(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for equal comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_equal_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_equal_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_equal_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_equal_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_equal_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_equal_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_equal_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_equal_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("equal: unsupported dtype")


@always_inline
fn _compare_not_equal_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized not_equal comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_not_equal(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for not_equal comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_not_equal_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_not_equal_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_not_equal_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_not_equal_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_not_equal_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_not_equal_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_not_equal_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_not_equal_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("not_equal: unsupported dtype")


@always_inline
fn _compare_less_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized less comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_less(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for less comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_less_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_less_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_less_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_less_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_less_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_less_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_less_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_less_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("less: unsupported dtype")


@always_inline
fn _compare_less_equal_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized less_equal comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_less_equal(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for less_equal comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_less_equal_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_less_equal_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_less_equal_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_less_equal_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_less_equal_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_less_equal_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_less_equal_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_less_equal_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("less_equal: unsupported dtype")


@always_inline
fn _compare_greater_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized greater comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_greater(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for greater comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_greater_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_greater_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_greater_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_greater_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_greater_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_greater_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_greater_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_greater_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("greater: unsupported dtype")


@always_inline
fn _compare_greater_equal_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized greater_equal comparison."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[DType.bool]]()

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


fn _dispatch_compare_greater_equal(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for greater_equal comparison."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _compare_greater_equal_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _compare_greater_equal_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _compare_greater_equal_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _compare_greater_equal_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _compare_greater_equal_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _compare_greater_equal_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _compare_greater_equal_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.bool:
        _compare_greater_equal_impl[DType.bool](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("greater_equal: unsupported dtype")


fn equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_equal(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


fn not_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_not_equal(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


fn less(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_less(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


fn less_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_less_equal(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


fn greater(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_greater(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


fn greater_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
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

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_compare_greater_equal(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


# ============================================================================
# Typed Tensor[dtype] overloads — wrap AnyTensor versions via as_any/as_tensor
# ============================================================================


fn equal[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise equality comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise equality results.
    """
    return equal(a.as_any(), b.as_any()).as_tensor[DType.bool]()


fn not_equal[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise inequality comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise inequality results.
    """
    return not_equal(a.as_any(), b.as_any()).as_tensor[DType.bool]()


fn less[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise less-than comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise less-than results.
    """
    return less(a.as_any(), b.as_any()).as_tensor[DType.bool]()


fn greater[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[DType.bool]:
    """Element-wise greater-than comparison (typed version).

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        A new Tensor[DType.bool] with element-wise greater-than results.
    """
    return greater(a.as_any(), b.as_any()).as_tensor[DType.bool]()
