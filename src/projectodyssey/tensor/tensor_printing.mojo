"""Tensor string formatting helpers for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
Cross-module private-field access (_data, _dtype, _shape, _numel) is valid
in Mojo (package-scoped privacy). See tensor_io.mojo for precedent.
"""

from std.collections import List
from .any_tensor import AnyTensor


def format_element_impl(tensor: AnyTensor, flat_idx: Int) -> String:
    """Format a single element based on dtype.

    Handles unsigned integers natively to avoid sign corruption when
    values exceed Int64 range (e.g., uint64 values > 2^63).

    Args:
        tensor: The tensor to read from.
        flat_idx: The flat index in the buffer.

    Returns:
        String representation of the element.
    """
    if tensor._dtype == DType.bool:
        return "True" if tensor._get_int64(flat_idx) != 0 else "False"
    elif tensor._dtype == DType.uint64:
        # Read as native UInt64 to avoid sign corruption via _get_int64
        var dtype_size = tensor._get_dtype_size()
        var ptr = (tensor._data + flat_idx * dtype_size).bitcast[UInt64]()
        return String(ptr[])
    elif tensor._dtype == DType.uint32:
        var dtype_size = tensor._get_dtype_size()
        var ptr = (tensor._data + flat_idx * dtype_size).bitcast[UInt32]()
        return String(ptr[])
    elif (
        tensor._dtype == DType.int8
        or tensor._dtype == DType.int16
        or tensor._dtype == DType.int32
        or tensor._dtype == DType.int64
        or tensor._dtype == DType.uint8
        or tensor._dtype == DType.uint16
    ):
        return String(tensor._get_int64(flat_idx))
    else:
        # Float types
        return String(tensor._get_float64(flat_idx))


def format_nd_slice_impl(
    tensor: AnyTensor, dim: Int, base_offset: Int
) -> String:
    """Format a slice of the N-dimensional tensor with nested brackets.

    Design: uses offset-based recursion instead of threading a mutable counter
    through calls. Each call computes its flat indices as base_offset + i * stride,
    making the function pure — its behavior is determined entirely by its arguments,
    not hidden mutable state. This mirrors the row-major index formula directly:
    element [i,j,k] lives at flat index i*(J*K) + j*K + k.

    Args:
        tensor: The tensor to format.
        dim: Current dimension level (0 = outermost).
        base_offset: Flat index offset for the start of this slice.

    Returns:
        String with nested brackets representing the N-D structure.
    """
    var ndim = len(tensor._shape)

    # Base case: innermost dimension (last dim)
    if dim == ndim - 1:
        var result = String("[")
        for i in range(tensor._shape[dim]):
            if i > 0:
                result += ", "
            result += format_element_impl(tensor, base_offset + i)
        result += "]"
        return result

    # Compute stride for current dimension (product of all inner dims)
    var stride = 1
    for d in range(dim + 1, ndim):
        stride *= tensor._shape[d]

    # Recursive case: format sub-array
    var result = String("[")
    for i in range(tensor._shape[dim]):
        if i > 0:
            result += ", "
        result += format_nd_slice_impl(
            tensor, dim + 1, base_offset + i * stride
        )

    result += "]"
    return result


def write_to_str_impl(tensor: AnyTensor) -> String:
    """Format tensor to string for write_to (Writable trait implementation).

    Converts the tensor to human-readable string representation with NumPy-style
    truncation. For tensors with more than 1000 elements, shows only the first 3
    and last 3 elements with '...' in between.

    Formats values by dtype:
    - Float types: display as decimals (1.0, 2.5, etc.)
    - Integer types: display without decimals (1, 42, -100, etc.)
    - Bool type: display as True/False

    Args:
        tensor: The tensor to format.

    Returns:
        Formatted string representation.
        1D: `AnyTensor([v0, v1, ...], dtype=<dtype>)`
        2D+: `AnyTensor([[v0, v1, ...], ...], shape=[d0, d1, ...], dtype=<dtype>)`
    """
    comptime TRUNCATE_THRESHOLD = 1000
    comptime SHOW_ELEMENTS = 3

    var ndim = len(tensor._shape)

    # Special case: empty tensor
    if ndim == 0 or tensor._numel == 0:
        return "AnyTensor([], dtype=" + String(tensor._dtype) + ")"

    # For 1D tensors: use flat format
    if ndim == 1:
        var result = String("AnyTensor([")
        if tensor._numel > TRUNCATE_THRESHOLD:
            for i in range(SHOW_ELEMENTS):
                if i > 0:
                    result += ", "
                result += format_element_impl(tensor, i)
            result += ", ..."
            for i in range(tensor._numel - SHOW_ELEMENTS, tensor._numel):
                result += ", " + format_element_impl(tensor, i)
        else:
            for i in range(tensor._numel):
                if i > 0:
                    result += ", "
                result += format_element_impl(tensor, i)
        result += "], dtype=" + String(tensor._dtype) + ")"
        return result

    # For multi-dimensional tensors (2D+): build nested brackets.
    # Truncate if total elements exceed threshold to prevent
    # massive string output for large tensors (e.g., [100, 100]).
    if tensor._numel > TRUNCATE_THRESHOLD:
        # Show first and last sub-arrays along outermost dimension
        var stride = 1
        for d in range(1, ndim):
            stride *= tensor._shape[d]

        var data_str = String("[")
        for i in range(SHOW_ELEMENTS):
            if i > 0:
                data_str += ", "
            data_str += format_nd_slice_impl(tensor, 1, i * stride)
        data_str += ", ..."
        for i in range(tensor._shape[0] - SHOW_ELEMENTS, tensor._shape[0]):
            data_str += ", " + format_nd_slice_impl(tensor, 1, i * stride)
        data_str += "]"

        var result = String("AnyTensor(") + data_str + ", shape=["
        for i in range(len(tensor._shape)):
            if i > 0:
                result += ", "
            result += String(tensor._shape[i])
        result += "], dtype=" + String(tensor._dtype) + ")"
        return result

    var data_str = format_nd_slice_impl(tensor, 0, 0)

    var result = String("AnyTensor(") + data_str + ", shape=["
    for i in range(len(tensor._shape)):
        if i > 0:
            result += ", "
        result += String(tensor._shape[i])
    result += "], dtype=" + String(tensor._dtype) + ")"
    return result


def write_repr_impl(tensor: AnyTensor) -> String:
    """Format tensor to detailed repr string for debugging.

    Returns the format: AnyTensor(shape=[...], dtype=<dtype>, numel=N, data=[...]).
    For large tensors (>1000 elements), shows first 3 and last 3 elements with '...'.

    Args:
        tensor: The tensor to format.

    Returns:
        Detailed representation string including shape, dtype, numel, and sample data.
    """
    comptime TRUNCATE_THRESHOLD = 1000
    comptime SHOW_ELEMENTS = 3

    var shape_str = String("[")
    for i in range(len(tensor._shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(tensor._shape[i])
    shape_str += "]"
    var result = String("AnyTensor(shape=") + shape_str
    result += ", dtype=" + String(tensor._dtype)
    result += ", numel=" + String(tensor._numel)
    result += ", data=["
    if tensor._numel > TRUNCATE_THRESHOLD:
        for i in range(SHOW_ELEMENTS):
            if i > 0:
                result += ", "
            result += String(tensor._get_float64(i))
        result += ", ..."
        for i in range(tensor._numel - SHOW_ELEMENTS, tensor._numel):
            result += ", " + String(tensor._get_float64(i))
    else:
        for i in range(tensor._numel):
            if i > 0:
                result += ", "
            result += String(tensor._get_float64(i))
    result += "])"
    return result
