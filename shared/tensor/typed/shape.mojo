"""Typed Tensor[dtype] shape dispatch cores.

Internal module -- not part of the public API.
"""

from std.collections import List, Optional
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.broadcasting import (
    are_shapes_broadcastable,
    compute_broadcast_strides,
)
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
# Layer 3 (Core): Native Tensor[dtype] as_contiguous implementation
# ============================================================================


def _as_contiguous_typed[
    dtype: DType
](tensor: Tensor[dtype]) raises -> Tensor[dtype]:
    """Convert tensor to contiguous memory layout (native Tensor[dtype] core).

    Zero dtype branches, zero bitcasts.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        A new contiguous Tensor[dtype] with the same data.
    """
    var shape = tensor.shape()
    var numel = tensor.numel()
    var result = Tensor[dtype](shape)
    var src_ptr = tensor._data
    var dst_ptr = result._data

    if tensor.is_contiguous():
        # Fast path: direct typed copy
        for i in range(numel):
            dst_ptr[i] = src_ptr[i]
    else:
        # Slow path: stride-based indexing
        var ndim = len(shape)
        for i in range(numel):
            var src_elem_offset = 0
            var remaining = i
            for d in range(ndim - 1, -1, -1):
                var coord = remaining % shape[d]
                remaining //= shape[d]
                src_elem_offset += coord * tensor._strides[d]
            dst_ptr[i] = src_ptr[src_elem_offset]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch for as_contiguous
# ============================================================================


def _as_contiguous_dispatch[
    dtype: DType
](tensor: AnyTensor) raises -> AnyTensor:
    """Dispatch as_contiguous to typed core."""
    return _as_contiguous_typed[dtype](tensor.as_tensor[dtype]()).as_any()


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] reshape implementation
# ============================================================================


def _reshape_typed[
    dtype: DType
](tensor: Tensor[dtype], new_shape: List[Int]) raises -> Tensor[dtype]:
    """Reshape tensor to new shape (native Tensor[dtype] core).

    This is the core implementation -- zero dtype branches, zero bitcasts.
    Tensor[dtype]._data is already typed as UnsafePointer[Scalar[dtype], MutAnyOrigin].

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        new_shape: Target shape (must have same total number of elements).

    Returns:
        A new Tensor[dtype] with the specified shape.

    Raises:
        Error: If new shape has different number of elements.
    """
    from shared.base.shape_utils import _resolve_shape

    var total_elements = tensor.numel()
    var final_shape = _resolve_shape(new_shape, total_elements)

    # Create new tensor with new shape
    var result = Tensor[dtype](final_shape)

    # Copy data using typed pointers -- zero bitcasts
    if tensor.is_contiguous():
        # Fast path: direct typed copy
        var src_ptr = tensor._data
        var dst_ptr = result._data
        for i in range(total_elements):
            dst_ptr[i] = src_ptr[i]
    else:
        # Slow path: compute stride-based offset for each element
        var src_shape = tensor.shape()
        var ndim = len(src_shape)
        var src_ptr = tensor._data
        var dst_ptr = result._data
        for i in range(total_elements):
            var remaining = i
            var src_elem_offset = 0
            for d in range(ndim - 1, -1, -1):
                var coord = remaining % src_shape[d]
                remaining //= src_shape[d]
                src_elem_offset += coord * tensor._strides[d]
            dst_ptr[i] = src_ptr[src_elem_offset]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch helpers (as_tensor -> typed core -> as_any)
# ============================================================================


def _reshape_dispatch[
    dtype: DType
](tensor: AnyTensor, new_shape: List[Int]) raises -> AnyTensor:
    """Dispatch reshape to typed core."""
    return _reshape_typed[dtype](tensor.as_tensor[dtype](), new_shape).as_any()


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] broadcast_to implementation
# ============================================================================


def _broadcast_to_typed[
    dtype: DType
](tensor: Tensor[dtype], target_shape: List[Int]) raises -> Tensor[dtype]:
    """Broadcast tensor to target shape (native Tensor[dtype] core).

    Zero dtype branches, zero bitcasts.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        target_shape: Target shape to broadcast to.

    Returns:
        Broadcasted Tensor[dtype].
    """
    var shape = tensor.shape()

    if len(target_shape) < len(shape):
        raise Error("broadcast_to: cannot broadcast to fewer dimensions")

    if not are_shapes_broadcastable(shape, target_shape):
        raise Error("broadcast_to: shapes are not broadcast-compatible")

    var broadcast_strides = compute_broadcast_strides(shape, target_shape)

    var result = Tensor[dtype](target_shape)
    var result_numel = result.numel()

    var src_ptr = tensor._data
    var dst_ptr = result._data

    for i in range(result_numel):
        var coords = List[Int]()
        var temp_i = i
        for j in range(len(target_shape)):
            var stride = 1
            for k in range(j + 1, len(target_shape)):
                stride *= target_shape[k]
            var coord = temp_i // stride
            coords.append(coord)
            temp_i = temp_i % stride

        var src_idx = 0
        for j in range(len(target_shape)):
            src_idx += coords[j] * broadcast_strides[j]

        # Copy value using typed pointer -- zero bitcasts
        dst_ptr[i] = src_ptr[src_idx]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch for broadcast_to
# ============================================================================


def _broadcast_to_dispatch[
    dtype: DType
](tensor: AnyTensor, target_shape: List[Int]) raises -> AnyTensor:
    """Dispatch broadcast_to to typed core."""
    return _broadcast_to_typed[dtype](
        tensor.as_tensor[dtype](), target_shape
    ).as_any()


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] permute implementation
# ============================================================================


def _validate_permute_dims(dims: List[Int], ndim: Int) raises:
    """Validate dims is a valid permutation of [0..ndim-1].

    Args:
        dims: Permutation to validate.
        ndim: Number of dimensions expected.

    Raises:
        Error: If dims is not a valid permutation.
    """
    if len(dims) != ndim:
        raise Error(
            "permute: dims length "
            + String(len(dims))
            + " must equal tensor ndim "
            + String(ndim)
        )

    var seen = List[Bool]()
    for _ in range(ndim):
        seen.append(False)

    for i in range(ndim):
        var d = dims[i]
        if d < 0 or d >= ndim:
            raise Error(
                "permute: dimension "
                + String(d)
                + " out of range [0, "
                + String(ndim)
                + ")"
            )
        if seen[d]:
            raise Error(
                "permute: dimension " + String(d) + " appears multiple times"
            )
        seen[d] = True


def _permute_typed[
    dtype: DType
](tensor: Tensor[dtype], dims: List[Int]) raises -> Tensor[dtype]:
    """Permute tensor dimensions (native Tensor[dtype] core).

    Zero dtype branches, zero bitcasts.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        dims: Permutation of dimensions.

    Returns:
        Tensor with permuted dimensions.
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    _validate_permute_dims(dims, ndim)

    # Compute new shape
    var new_shape = List[Int]()
    for i in range(ndim):
        new_shape.append(shape[dims[i]])

    # Create result tensor
    var result = Tensor[dtype](new_shape)
    var result_numel = result.numel()

    var src_ptr = tensor._data
    var dst_ptr = result._data

    # Fill result by permuting coordinates
    for i in range(result_numel):
        # Compute coordinates in result tensor
        var result_coords = List[Int]()
        var temp_i = i
        for j in range(ndim):
            var stride = 1
            for k in range(j + 1, ndim):
                stride *= new_shape[k]
            var coord = temp_i // stride
            result_coords.append(coord)
            temp_i = temp_i % stride

        # Compute source coordinates by inverse permutation
        var src_coords = List[Int]()
        for _ in range(ndim):
            src_coords.append(0)

        for j in range(ndim):
            src_coords[dims[j]] = result_coords[j]

        # Compute source index using actual tensor strides (not derived from
        # shape) so non-contiguous inputs (e.g. transpose_view) read the right
        # element. Reading shape-derived contiguous strides here was the
        # historical bug fixed for #4086 cascade.
        var src_idx = 0
        for j in range(ndim):
            src_idx += src_coords[j] * tensor._strides[j]

        # Copy value using typed pointer -- zero bitcasts
        dst_ptr[i] = src_ptr[src_idx]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch for permute
# ============================================================================


def _permute_dispatch[
    dtype: DType
](tensor: AnyTensor, dims: List[Int]) raises -> AnyTensor:
    """Dispatch permute to typed core."""
    return _permute_typed[dtype](tensor.as_tensor[dtype](), dims).as_any()
