"""Tensor indexing helpers for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
"""

from std.collections import List
from std.math import ceildiv
from .any_tensor import AnyTensor


def normalize_slice_indices_impl(
    size: Int, start: Int, end: Int, step: Int
) -> Tuple[Int, Int, Int, Int]:
    """Normalize slice indices to valid ranges.

    Handles negative indices, clamping, and returns normalized
    (start, end, step, result_size) for valid iteration.

    Args:
        size: Size of the dimension being sliced.
        start: Start index (may be negative).
        end: End index (may be negative).
        step: Step value (can be negative for reverse).

    Returns:
        Tuple of (normalized_start, normalized_end, normalized_step, result_size).
        result_size is the number of elements in the slice result.
    """
    var norm_start = start
    var norm_end = end
    var norm_step = step
    var result_size: Int

    if step < 0:
        # Negative step: reverse iteration
        var neg_step = -step
        # Clamp start to [0, size-1], end to [-1, size-1]
        norm_start = max(0, min(norm_start, size - 1))
        norm_end = max(-1, min(norm_end, size - 1))
        result_size = max(0, ceildiv(norm_start - norm_end, neg_step))
    else:
        # Positive step: forward iteration
        # Normalize negative indices first
        if norm_start < 0:
            norm_start = size + norm_start
        if norm_end < 0:
            norm_end = size + norm_end
        # Clamp forward slice to [0, size]
        norm_start = max(0, min(norm_start, size))
        norm_end = max(0, min(norm_end, size))
        result_size = max(0, ceildiv(norm_end - norm_start, step))

    return (norm_start, norm_end, norm_step, result_size)


def _resolve_index_impl(tensor: AnyTensor, index: Int) raises -> Int:
    """Resolve flat index to memory offset, with bounds check.

    For non-contiguous tensors, converts flat index to memory offset
    via nd-coordinates and strides.

    Args:
        tensor: The tensor to resolve the index in.
        index: Flat logical index.

    Returns:
        Memory offset for the element.

    Raises:
        Error: If index is out of bounds.
    """
    if index < 0 or index >= tensor._numel:
        raise Error("Index out of bounds")
    if not tensor.is_contiguous():
        var remaining = index
        var mem_offset = 0
        for i in range(len(tensor._shape)):
            var dim_size = 1
            for j in range(i + 1, len(tensor._shape)):
                dim_size *= tensor._shape[j]
            var coord = remaining // dim_size
            remaining = remaining % dim_size
            mem_offset += coord * tensor._strides[i]
        return mem_offset
    return index


def _getitem_int_impl(tensor: AnyTensor, index: Int) raises -> Float32:
    """Get element at flat index as Float32.

    For contiguous tensors, the flat index maps directly to a memory offset.
    For non-contiguous tensors (e.g., after transpose or axis>0 slice), the
    flat index is first converted to multi-dimensional coordinates using the
    tensor's shape, then mapped to a memory offset using strides.

    Args:
        tensor: The tensor to access.
        index: The flat index to access (logical element index in
            row-major order of the tensor's shape).

    Returns:
        The value at the given index as Float32.

    Raises:
        Error: If index is out of bounds.
    """
    if index < 0 or index >= tensor._numel:
        raise Error("Index out of bounds")

    # For non-contiguous tensors, convert flat index to nd-coordinates
    # then use strides to compute the real memory offset.
    if not tensor.is_contiguous():
        var remaining = index
        var mem_offset = 0
        for i in range(len(tensor._shape)):
            # Compute the product of dimensions after axis i
            var dim_size = 1
            for j in range(i + 1, len(tensor._shape)):
                dim_size *= tensor._shape[j]
            var coord = remaining // dim_size
            remaining = remaining % dim_size
            mem_offset += coord * tensor._strides[i]
        return tensor._get_float32(mem_offset)

    # Return value based on dtype
    return tensor._get_float32(index)


def _getitem_multi_impl(
    tensor: AnyTensor, indices: List[Int]
) raises -> Float32:
    """Get element at multi-dimensional index.

    Args:
        tensor: The tensor to access.
        indices: Per-dimension indices (one per axis).

    Returns:
        The value at the given indices as Float32.

    Raises:
        Error: If number of indices doesn't match tensor rank,
               or any index is out of bounds.
    """
    if len(indices) != len(tensor._shape):
        raise Error(
            "Number of indices ("
            + String(len(indices))
            + ") must match tensor rank ("
            + String(len(tensor._shape))
            + ")"
        )
    var mem_offset = 0
    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= tensor._shape[i]:
            raise Error("Index out of bounds at dimension " + String(i))
        mem_offset += indices[i] * tensor._strides[i]
    return tensor._get_float32(mem_offset)


def _getitem_slice_impl(tensor: AnyTensor, slice: Slice) raises -> AnyTensor:
    """Get slice of 1D tensor [start:end] or [start:end:step].

    Args:
        tensor: The tensor to slice (must be 1D).
        slice: Slice object specifying start, end, and optional step.

    Returns:
        New tensor containing a **copy** of the sliced data. The result
        does not share memory with the original tensor.

    Raises:
        Error: If tensor is not 1D or indices are invalid.

    Notes:
        This function always returns a copy (`_is_view = False`), regardless
        of the step value. This is by design: materializing a strided copy
        keeps the implementation simple and avoids lifetime management
        complexity.
    """
    if len(tensor._shape) != 1:
        raise Error("Single slice only supported for 1D tensors")

    # Handle slice parameters — extract step first so defaults depend on sign
    var size = tensor._shape[0]
    var step = slice.step.or_else(1)

    var start: Int
    var end: Int
    if step < 0:
        # Negative step: default start=last element, default end=before index 0
        start = slice.start.or_else(size - 1)
        end = slice.end.or_else(-size - 1)
    else:
        start = slice.start.or_else(0)
        end = slice.end.or_else(size)

    # Normalize negative indices
    if start < 0:
        start = size + start
    if end < 0:
        end = size + end

    # Handle negative step (reverse)
    var result_size: Int
    if step < 0:
        var neg_step = -step
        # Clamp start to [0, size-1], end to [-1, size-1]
        start = max(0, min(start, size - 1))
        end = max(-1, min(end, size - 1))
        # No swap: iterate src_idx = start - i * neg_step while src_idx > end
        result_size = max(0, ceildiv(start - end, neg_step))

        # Create result tensor with shape
        var shape = List[Int]()
        shape.append(result_size)
        var result = AnyTensor(shape, tensor._dtype)
        result._is_view = False

        # Copy in reverse
        var dtype_size = tensor._get_dtype_size()
        var src_ptr = tensor._data
        var dst_ptr = result._data

        for i in range(result_size):
            var src_idx = start - i * neg_step
            var src_offset = src_idx * dtype_size
            var dst_offset = i * dtype_size
            for b in range(dtype_size):
                dst_ptr[dst_offset + b] = src_ptr[src_offset + b]

        return result^
    else:
        # Clamp forward slice to [0, size]
        start = max(0, min(start, size))
        end = max(0, min(end, size))
        # Normal forward slice
        result_size = max(0, ceildiv(end - start, step))

    # Create result tensor with shape
    var shape = List[Int]()
    shape.append(result_size)
    var result = AnyTensor(shape, tensor._dtype)
    result._is_view = False  # Strided slice creates copy, not view

    # Copy strided data
    var dtype_size = tensor._get_dtype_size()
    var src_ptr = tensor._data
    var dst_ptr = result._data

    for i in range(result_size):
        var src_idx = start + i * step
        var src_offset = src_idx * dtype_size
        var dst_offset = i * dtype_size

        # Copy element (byte-wise)
        for b in range(dtype_size):
            dst_ptr[dst_offset + b] = src_ptr[src_offset + b]

    return result^
