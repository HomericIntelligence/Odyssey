"""Tensor view and copy operations for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
Cross-module private-field access is valid in Mojo (package-scoped privacy).
See tensor_io.mojo for precedent.
"""

from std.collections import List
from .any_tensor import AnyTensor


def slice_impl(
    tensor: AnyTensor, start: Int, end: Int, axis: Int = 0
) raises -> AnyTensor:
    """Extract a slice along the specified axis, returning a view.

    Args:
        tensor: Source tensor to slice.
        start: Starting index (inclusive).
        end: Ending index (exclusive).
        axis: Axis to slice along (default: 0).

    Returns:
        View tensor referencing same memory.

    Raises:
        Error: If indices are out of bounds or axis is invalid.
    """
    # Validate axis
    if axis < 0 or axis >= len(tensor._shape):
        raise Error(
            "Axis "
            + String(axis)
            + " out of range for tensor with "
            + String(len(tensor._shape))
            + " dimensions"
        )

    # Validate indices
    var dim_size = tensor._shape[axis]
    if start < 0 or start > dim_size:
        raise Error(
            "Start index "
            + String(start)
            + " out of range [0, "
            + String(dim_size)
            + "]"
        )
    if end < start or end > dim_size:
        raise Error(
            "End index "
            + String(end)
            + " out of range ["
            + String(start)
            + ", "
            + String(dim_size)
            + "]"
        )

    # Calculate offset to start of slice
    var offset_elements = start * tensor._strides[axis]
    var dtype_size = tensor._get_dtype_size()
    var offset_bytes = offset_elements * dtype_size

    # Create view by copying (increments refcount)
    var result = tensor.copy()
    result._is_view = True

    # Update the sliced dimension in place
    result._shape[axis] = end - start

    # Update data pointer to point to sliced data
    result._data = tensor._data + offset_bytes

    # Strides remain the same (already copied by copy constructor)

    # Recalculate numel after shape change
    result._numel = 1
    for i in range(len(result._shape)):
        result._numel *= result._shape[i]

    return result^


def transpose_impl(tensor: AnyTensor, dim0: Int, dim1: Int) raises -> AnyTensor:
    """Return a non-contiguous view with dim0 and dim1 swapped.

    Args:
        tensor: Source tensor.
        dim0: First dimension to swap.
        dim1: Second dimension to swap.

    Returns:
        View tensor with permuted shape and strides.

    Raises:
        Error: If tensor has fewer than 2 dimensions or dims are out of bounds.
    """
    var ndim = tensor.dim()
    if ndim < 2:
        raise Error("transpose requires at least 2 dimensions")
    if dim0 < 0 or dim0 >= ndim:
        raise Error("transpose: dim0 out of range")
    if dim1 < 0 or dim1 >= ndim:
        raise Error("transpose: dim1 out of range")

    var result = tensor.copy()
    result._is_view = True

    var tmp_shape = result._shape[dim0]
    result._shape[dim0] = result._shape[dim1]
    result._shape[dim1] = tmp_shape

    var tmp_stride = result._strides[dim0]
    result._strides[dim0] = result._strides[dim1]
    result._strides[dim1] = tmp_stride

    return result^


def clone_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Create a deep copy of the tensor with independent memory.

    Args:
        tensor: Source tensor to clone.

    Returns:
        New AnyTensor with same shape, dtype, and values.

    Raises:
        Error: If memory allocation fails.
    """
    var shape_copy = tensor._shape.copy()
    var result = AnyTensor(shape_copy, tensor._dtype)

    # Iterate through all elements using multi-dimensional indexing
    # to correctly handle non-contiguous source tensors with stride-aware access
    var nd_idx = List[Int]()
    for _ in range(len(tensor._shape)):
        nd_idx.append(0)

    var dtype_size = tensor._get_dtype_size()

    for out_idx in range(tensor._numel):
        # Compute flat offset in source tensor using strides
        var src_offset = 0
        for d in range(len(tensor._shape)):
            src_offset += nd_idx[d] * tensor._strides[d]

        # Read from source using stride-aware byte offset
        var offset_bytes = src_offset * dtype_size
        var val: Float64

        if tensor._dtype == DType.float16:
            var ptr = (tensor._data + offset_bytes).bitcast[Float16]()
            val = ptr[].cast[DType.float64]()
        elif tensor._dtype == DType.bfloat16:
            var ptr = (tensor._data + offset_bytes).bitcast[BFloat16]()
            val = Float64(Float32(ptr[]))
        elif tensor._dtype == DType.float32:
            var ptr = (tensor._data + offset_bytes).bitcast[Float32]()
            val = ptr[].cast[DType.float64]()
        elif tensor._dtype == DType.float64:
            var ptr = (tensor._data + offset_bytes).bitcast[Float64]()
            val = ptr[]
        else:
            # For integer types, use _get_int64 via byte offset
            if tensor._dtype == DType.int8:
                var ptr = (tensor._data + offset_bytes).bitcast[Int8]()
                val = Float64(ptr[])
            elif tensor._dtype == DType.int16:
                var ptr = (tensor._data + offset_bytes).bitcast[Int16]()
                val = Float64(ptr[])
            elif tensor._dtype == DType.int32:
                var ptr = (tensor._data + offset_bytes).bitcast[Int32]()
                val = Float64(ptr[])
            elif tensor._dtype == DType.int64:
                var ptr = (tensor._data + offset_bytes).bitcast[Int64]()
                val = Float64(ptr[])
            elif tensor._dtype == DType.uint8:
                var ptr = (tensor._data + offset_bytes).bitcast[UInt8]()
                val = Float64(Int(ptr[]))
            elif tensor._dtype == DType.uint16:
                var ptr = (tensor._data + offset_bytes).bitcast[UInt16]()
                val = Float64(Int(ptr[]))
            elif tensor._dtype == DType.uint32:
                var ptr = (tensor._data + offset_bytes).bitcast[UInt32]()
                val = Float64(Int(ptr[]))
            elif tensor._dtype == DType.uint64:
                var ptr = (tensor._data + offset_bytes).bitcast[UInt64]()
                val = Float64(Int(ptr[]))
            else:
                val = 0.0

        # Write to output tensor at flat index
        result._set_float64(out_idx, val)

        # Increment multi-dimensional index
        var d = len(tensor._shape) - 1
        while d >= 0:
            nd_idx[d] += 1
            if nd_idx[d] < tensor._shape[d]:
                break
            nd_idx[d] = 0
            d -= 1

    return result^


def diff_impl(tensor: AnyTensor, n: Int = 1) raises -> AnyTensor:
    """Calculate consecutive differences.

    Args:
        tensor: Source tensor.
        n: Order of differences (default: 1).

    Returns:
        New AnyTensor with differences.

    Raises:
        Error: If n <= 0 or n >= tensor size.
    """
    if n <= 0:
        raise Error("diff order n must be positive, got " + String(n))
    if n >= tensor._numel:
        raise Error(
            "diff order n="
            + String(n)
            + " exceeds tensor size "
            + String(tensor._numel)
        )

    var current = tensor
    for _ in range(n):
        var new_size = current._numel - 1
        var new_shape = List[Int]()
        new_shape.append(new_size)
        var result = AnyTensor(new_shape, current._dtype)

        for i in range(new_size):
            var val = current._get_float64(i + 1) - current._get_float64(i)
            result._set_float64(i, val)

        current = result^

    return current^


def reshape_impl(tensor: AnyTensor, new_shape: List[Int]) raises -> AnyTensor:
    """Reshape tensor to new shape (must have same total elements).

    Returns a zero-copy view (shallow pointer copy) sharing data with the
    original tensor. The result has `is_view() == True` and `is_contiguous() == True`
    (because reshape only changes the shape/stride metadata, not the flat layout).
    Uses reference counting to ensure data remains valid while any view is alive.

    Note: This mirrors the view semantics of `slice()` — no data is copied.
    Compare with operations that return independent copies (e.g. `as_contiguous()`).

    Args:
        tensor: Source tensor to reshape.
        new_shape: The new shape for the tensor.

    Returns:
        A zero-copy view with the requested shape, sharing the same flat data buffer.

    Raises:
        Error: If the total number of elements doesn't match.

    Example:
    ```mojo
        var t = zeros([2, 3], DType.float32)
        var reshaped = t.reshape([6])  # (2, 3) -> (6,), zero-copy view
    ```
    """
    # Verify total elements match
    var new_numel = 1
    for i in range(len(new_shape)):
        new_numel *= new_shape[i]

    if new_numel != tensor._numel:
        raise Error("Cannot reshape: element count mismatch")

    # Create view by explicitly copying (increments refcount via copy constructor)
    var result = tensor.copy()
    result._is_view = True  # Mark as view since it shares data with original

    # Update shape
    result._shape = List[Int]()
    for i in range(len(new_shape)):
        result._shape.append(new_shape[i])

    # Recalculate strides for new shape
    result._strides = List[Int]()
    var stride = 1
    # Pre-allocate strides list with correct forward iteration
    for _ in range(len(new_shape)):
        result._strides.append(0)
    # Now fill strides in backward order
    for i in range(len(new_shape) - 1, -1, -1):
        result._strides[i] = stride
        stride *= new_shape[i]

    return result^


def array_equal_impl(
    tensor: AnyTensor, other: AnyTensor, equal_nan: Bool = True
) raises -> Bool:
    """Whole-tensor equality returning a single `Bool`.

    Unlike element-wise equality, this returns one `Bool`: True iff `tensor`
    and `other` have the same shape and dtype and every element compares equal.

    With `equal_nan=True` (the default), NaN matches NaN position-wise —
    mirroring `__hash__`, which canonicalizes NaN so NaN-containing
    tensors hash equally. This is what makes a NaN-containing tensor
    usable as a dict/set key: `__hash__` and `array_equal` agree, so
    lookup succeeds. With `equal_nan=False`, IEEE 754 semantics apply
    and any NaN makes the tensors unequal (see issue #4061).

    Args:
        tensor: Source tensor.
        other: The tensor to compare against.
        equal_nan: If True, NaN equals NaN at the same position.

    Returns:
        True iff the tensors are equal under the chosen NaN policy.
    """
    from std.math import isnan

    if tensor._dtype != other._dtype:
        return False
    if len(tensor._shape) != len(other._shape):
        return False
    for i in range(len(tensor._shape)):
        if tensor._shape[i] != other._shape[i]:
            return False
    if tensor._numel != other._numel:
        return False

    for i in range(tensor._numel):
        var a = tensor._get_float64(i)
        var b = other._get_float64(i)
        var a_nan = isnan(a)
        var b_nan = isnan(b)
        if a_nan or b_nan:
            # Equal only if both are NaN and equal_nan is set.
            if not (equal_nan and a_nan and b_nan):
                return False
        elif a != b:
            return False
    return True
