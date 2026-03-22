"""Shape manipulation operations with native Tensor[dtype] implementations.

Implements shape operations like reshape, squeeze, unsqueeze, flatten, concatenate, stack, split
Following the Python Array API Standard 2023.12.

Architecture: Tensor[dtype] typed implementations are the core (zero dtype branches).
AnyTensor versions dispatch to typed implementations via ordinal-based table.
Collection ops (concatenate, stack, split) stay on AnyTensor (they use List[AnyTensor]).

Layer 1 (outer): AnyTensor public API (reshape, flatten, etc.)
Layer 2: dtype dispatch table (ordinal-based)
Layer 3 (core): Tensor[dtype] native implementation (_reshape_typed, etc.)

Optimizations:
- Zero-copy views with stride-based indexing
- memcpy for bulk copying of contiguous memory blocks
- Automatic contiguity detection and conversion
"""

from collections import List
from memory import memcpy, UnsafePointer
from .any_tensor import AnyTensor
from shared.tensor.tensor import Tensor
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
# Zero-Copy Views and Memory Optimization Helpers
# ============================================================================


fn is_contiguous(tensor: AnyTensor) -> Bool:
    """Check if tensor data is contiguous in memory (row-major C order).

        A tensor is contiguous if elements are laid out sequentially in memory
        with no gaps. This is true when strides match C-order (row-major) layout.

    Args:
            tensor: The tensor to check.

    Returns:
            True if tensor is contiguous in memory, False otherwise.

    Note:
            Contiguous tensors can be efficiently copied with memcpy instead of
            element-by-element copying.
    """
    # Delegate to AnyTensor.is_contiguous() method to avoid code duplication
    # (Issue #3844: consolidated implementations)
    return tensor.is_contiguous()


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] as_contiguous implementation
# ============================================================================


fn _as_contiguous_typed[
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


fn _as_contiguous_dispatch[
    dtype: DType
](tensor: AnyTensor) raises -> AnyTensor:
    """Dispatch as_contiguous to typed core."""
    return _as_contiguous_typed[dtype](
        tensor.as_tensor[dtype]()
    ).as_any()


fn as_contiguous(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor to contiguous memory layout if needed.

        If the tensor is already contiguous, returns a copy. If it's a view with
        non-contiguous strides, creates a new contiguous copy.

    Args:
            tensor: The tensor to make contiguous.

    Returns:
            A new contiguous tensor with the same data.

    Raises:
            Error: If operation fails.

    Note:
            This function always copies data. For zero-copy operations, check
            is_contiguous() first.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _as_contiguous_dispatch[DType.float16](tensor)
    elif ordinal == DTYPE_FLOAT32:
        return _as_contiguous_dispatch[DType.float32](tensor)
    elif ordinal == DTYPE_FLOAT64:
        return _as_contiguous_dispatch[DType.float64](tensor)
    elif ordinal == DTYPE_INT8:
        return _as_contiguous_dispatch[DType.int8](tensor)
    elif ordinal == DTYPE_INT16:
        return _as_contiguous_dispatch[DType.int16](tensor)
    elif ordinal == DTYPE_INT32:
        return _as_contiguous_dispatch[DType.int32](tensor)
    elif ordinal == DTYPE_INT64:
        return _as_contiguous_dispatch[DType.int64](tensor)
    elif ordinal == DTYPE_UINT8:
        return _as_contiguous_dispatch[DType.uint8](tensor)
    elif ordinal == DTYPE_UINT16:
        return _as_contiguous_dispatch[DType.uint16](tensor)
    elif ordinal == DTYPE_UINT32:
        return _as_contiguous_dispatch[DType.uint32](tensor)
    elif ordinal == DTYPE_UINT64:
        return _as_contiguous_dispatch[DType.uint64](tensor)
    else:
        raise Error("as_contiguous: unsupported dtype")


fn view(tensor: AnyTensor, new_shape: List[Int]) raises -> AnyTensor:
    """Create a zero-copy view of tensor with new shape (if compatible).

        Attempts to create a view with different shape while preserving the
        underlying data and strides. Returns a view if possible, otherwise raises
        an error.

        This is more strict than reshape() which always copies. view() only succeeds
        if the new shape is compatible with the current stride pattern.

    Args:
            tensor: Input tensor.
            new_shape: Target shape.

    Returns:
            A new AnyTensor sharing the same data with different shape/strides.

    Raises:
            Error: If reshape cannot be done as a view (would require data movement).

    Note:
            This is an advanced function. Most code should use reshape() which
            handles all cases by copying if necessary.

    Examples:
    ```
            # View works for compatible reshapes
            var a = ones([2, 3], DType.float32)  # Contiguous (2, 3)
            var b = view(a, [6])  # Creates view with shape (6,)

            # View fails for non-trivial reshapes
            # var c = view(a, [3, 2])  # Would fail - need to transpose memory layout
    ```
    """
    var old_numel = tensor.numel()
    var new_numel = 1
    var new_len = len(new_shape)

    # Validate new shape has same total elements
    for i in range(new_len):
        new_numel *= new_shape[i]

    if new_numel != old_numel:
        raise Error("view: new shape must have same number of elements")

    # Use AnyTensor's built-in reshape which creates views via __copyinit__
    # This leverages reference counting for safe shared ownership
    return tensor.reshape(new_shape)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] reshape implementation
# ============================================================================


fn _resolve_shape(
    new_shape: List[Int], total_elements: Int
) raises -> List[Int]:
    """Resolve -1 dimension and validate shape.

    Args:
        new_shape: Target shape (may contain -1).
        total_elements: Total elements in source tensor.

    Returns:
        Resolved shape with no -1 values.

    Raises:
        Error: If shape is invalid.
    """
    var inferred_dim = -1
    var known_product: Int = 1
    var new_len = len(new_shape)

    for i in range(new_len):
        if new_shape[i] == -1:
            if inferred_dim != -1:
                raise Error(
                    "reshape: can only specify one unknown dimension (-1)"
                )
            inferred_dim = i
        elif new_shape[i] < 0:
            raise Error("reshape: shape dimensions must be positive or -1")
        else:
            known_product *= new_shape[i]

    var final_shape = List[Int]()

    if inferred_dim != -1:
        if total_elements % known_product != 0:
            raise Error("reshape: cannot infer dimension, incompatible size")
        var inferred_size = total_elements // known_product

        for i in range(new_len):
            if i == inferred_dim:
                final_shape.append(inferred_size)
            else:
                final_shape.append(new_shape[i])
    else:
        for i in range(new_len):
            final_shape.append(new_shape[i])

    var new_total: Int = 1
    for i in range(new_len):
        new_total *= final_shape[i]

    if new_total != total_elements:
        raise Error("reshape: new shape must have same number of elements")

    return final_shape^


fn _reshape_typed[
    dtype: DType
](tensor: Tensor[dtype], new_shape: List[Int]) raises -> Tensor[dtype]:
    """Reshape tensor to new shape (native Tensor[dtype] core).

    This is the core implementation -- zero dtype branches, zero bitcasts.
    Tensor[dtype]._data is already typed as UnsafePointer[Scalar[dtype]].

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


fn _reshape_dispatch[
    dtype: DType
](tensor: AnyTensor, new_shape: List[Int]) raises -> AnyTensor:
    """Dispatch reshape to typed core."""
    return _reshape_typed[dtype](
        tensor.as_tensor[dtype](), new_shape
    ).as_any()


fn reshape(tensor: AnyTensor, new_shape: List[Int]) raises -> AnyTensor:
    """Reshape tensor to new shape.

    Args:
            tensor: Input tensor.
            new_shape: Target shape (must have same total number of elements).

    Returns:
            A new tensor with the specified shape.

    Raises:
            Error: If new shape has different number of elements.

    Examples:
    ```
            # Reshape 1D to 2D
            var a = arange(0.0, 12.0, 1.0, DType.float32)  # Shape (12,)
            var b = reshape(a, [3, 4])  # Shape (3, 4)

            # With -1 for inferred dimension
            var c = reshape(a, [3, -1])  # Shape (3, 4) - infers 4
    ```
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _reshape_dispatch[DType.float16](tensor, new_shape)
    elif ordinal == DTYPE_FLOAT32:
        return _reshape_dispatch[DType.float32](tensor, new_shape)
    elif ordinal == DTYPE_FLOAT64:
        return _reshape_dispatch[DType.float64](tensor, new_shape)
    elif ordinal == DTYPE_INT8:
        return _reshape_dispatch[DType.int8](tensor, new_shape)
    elif ordinal == DTYPE_INT16:
        return _reshape_dispatch[DType.int16](tensor, new_shape)
    elif ordinal == DTYPE_INT32:
        return _reshape_dispatch[DType.int32](tensor, new_shape)
    elif ordinal == DTYPE_INT64:
        return _reshape_dispatch[DType.int64](tensor, new_shape)
    elif ordinal == DTYPE_UINT8:
        return _reshape_dispatch[DType.uint8](tensor, new_shape)
    elif ordinal == DTYPE_UINT16:
        return _reshape_dispatch[DType.uint16](tensor, new_shape)
    elif ordinal == DTYPE_UINT32:
        return _reshape_dispatch[DType.uint32](tensor, new_shape)
    elif ordinal == DTYPE_UINT64:
        return _reshape_dispatch[DType.uint64](tensor, new_shape)
    else:
        raise Error("reshape: unsupported dtype")

fn reshape[dt: DType](tensor: Tensor[dt], new_shape: List[Int]) raises -> Tensor[dt]:
    """Reshape tensor to new shape (typed overload).

    Parameters:
        dt: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        new_shape: Target shape (must have same total elements).

    Returns:
        A new Tensor[dt] with the given shape.
    """
    return _reshape_typed[dt](tensor, new_shape)


fn squeeze(tensor: AnyTensor, axis: Int = -999) raises -> AnyTensor:
    """Remove size-1 dimensions.

    Args:
            tensor: Input tensor.
            axis: Specific dimension to squeeze (optional, default squeezes all size-1 dims).

    Returns:
            Tensor with size-1 dimensions removed.

    Raises:
            Error: If specified axis is not size 1.

    Examples:
    ```
            # Squeeze all size-1 dims
            var a = ones([1, 3, 1, 4], DType.float32)  # Shape (1, 3, 1, 4)
            var b = squeeze(a)  # Shape (3, 4)

            # Squeeze specific axis
            var c = squeeze(a, axis=0)  # Shape (3, 1, 4)
    ```
    """
    var old_shape = tensor.shape()
    var ndim = len(old_shape)

    if axis != -999:
        # Squeeze specific dimension
        var actual_axis = axis if axis >= 0 else ndim + axis

        if actual_axis < 0 or actual_axis >= ndim:
            raise Error("squeeze: dimension out of range")

        if old_shape[actual_axis] != 1:
            raise Error("squeeze: cannot squeeze dimension that is not size 1")

        # Create new shape without this dimension
        var new_shape = List[Int]()
        for i in range(ndim):
            if i != actual_axis:
                new_shape.append(old_shape[i])

        return reshape(tensor, new_shape)
    else:
        # Squeeze all size-1 dimensions
        var new_dims = 0
        for i in range(ndim):
            if old_shape[i] != 1:
                new_dims += 1

        if new_dims == ndim:
            # No size-1 dims, return copy
            return reshape(tensor, old_shape)

        # Build new shape
        var new_shape = List[Int]()
        for i in range(ndim):
            if old_shape[i] != 1:
                new_shape.append(old_shape[i])

        return reshape(tensor, new_shape)


fn unsqueeze(tensor: AnyTensor, axis: Int) raises -> AnyTensor:
    """Add a size-1 dimension at specified position.

    Args:
            tensor: Input tensor.
            axis: Position to insert new dimension (supports negative indexing).

    Returns:
            Tensor with additional size-1 dimension.

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var a = ones([3, 4], DType.float32)  # Shape (3, 4)
            var b = unsqueeze(a, axis=0)  # Shape (1, 3, 4)
            var c = unsqueeze(a, axis=-1)  # Shape (3, 4, 1)
    ```
    """
    var old_shape = tensor.shape()
    var ndim = len(old_shape)
    var new_ndim = ndim + 1

    # Handle negative indexing (allow axis in range [-ndim-1, ndim])
    var actual_axis = axis if axis >= 0 else new_ndim + axis

    if actual_axis < 0 or actual_axis > ndim:
        raise Error("unsqueeze: dimension out of range")

    # Create new shape with size-1 dimension inserted
    var new_shape = List[Int]()
    var j = 0
    for i in range(new_ndim):
        if i == actual_axis:
            new_shape.append(1)
        else:
            new_shape.append(old_shape[j])
            j += 1

    return reshape(tensor, new_shape)


@always_inline
fn expand_dims(tensor: AnyTensor, axis: Int) raises -> AnyTensor:
    """Alias for unsqueeze(). Add a size-1 dimension at specified position.

    Args:
            tensor: Input tensor.
            axis: Position to insert new dimension.

    Returns:
            Tensor with additional size-1 dimension.

    Raises:
            Error: If operation fails.
    """
    return unsqueeze(tensor, axis)


fn flatten(tensor: AnyTensor) raises -> AnyTensor:
    """Flatten tensor to 1D.

    Args:
            tensor: Input tensor.

    Returns:
            1D tensor with all elements in row-major (C) order.

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var a = ones([3, 4], DType.float32)  # Shape (3, 4)
            var b = flatten(a)  # Shape (12,)
    ```
    """
    var numel = tensor.numel()
    var shape_1d = List[Int]()
    shape_1d.append(numel)

    return reshape(tensor, shape_1d)


@always_inline
fn ravel(tensor: AnyTensor) raises -> AnyTensor:
    """Flatten tensor to 1D (comptime for flatten).

        Note: Our implementation now uses zero-copy views for contiguous tensors.
        If the tensor is contiguous, ravel() returns a view. Otherwise, it copies.

    Args:
            tensor: Input tensor.

    Returns:
            1D tensor with all elements (may be a view if contiguous).

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var a = ones([2, 3], DType.float32)  # Contiguous
            var b = ravel(a)  # Returns view of shape (6,)
    ```
    """
    # For contiguous tensors, we can safely flatten as a view
    # For non-contiguous tensors, we need to copy
    if is_contiguous(tensor):
        var new_shape = List[Int]()
        new_shape.append(tensor.numel())
        return view(tensor, new_shape)
    else:
        return flatten(tensor)


fn concatenate(tensors: List[AnyTensor], axis: Int = 0) raises -> AnyTensor:
    """Concatenate tensors along an existing axis.

    Args:
            tensors: Vector of tensors to concatenate.
            axis: Axis along which to concatenate (default 0).

    Returns:
            Concatenated tensor.

    Raises:
            Error: If tensors have incompatible shapes.

    Examples:
    ```
            var a = ones([2, 3], DType.float32)  # 2x3
            var b = ones([3, 3], DType.float32)  # 3x3
            var tensors : List[AnyTensor] = []
            tensors.append(a)
            tensors.append(b)
            var c = concatenate(tensors, axis=0)  # Shape (5, 3)
    ```
    """
    var num_tensors = len(tensors)
    if num_tensors == 0:
        raise Error("concatenate: need at least one tensor")

    if num_tensors == 1:
        # Single tensor, just return copy
        return reshape(tensors[0], tensors[0].shape())

    # Get reference shape and dtype from first tensor
    var ref_shape = tensors[0].shape()
    var ndim = len(ref_shape)
    var dtype = tensors[0].dtype()

    # Handle negative axis
    var actual_axis = axis if axis >= 0 else ndim + axis
    if actual_axis < 0 or actual_axis >= ndim:
        raise Error("concatenate: axis out of range")

    # Validate all tensors have same shape except along concat axis
    var concat_size = 0
    for i in range(num_tensors):
        var shape = tensors[i].shape()

        if len(shape) != ndim:
            raise Error(
                "concatenate: all tensors must have same number of dimensions"
            )

        if tensors[i].dtype() != dtype:
            raise Error("concatenate: all tensors must have same dtype")

        for j in range(ndim):
            if j != actual_axis and shape[j] != ref_shape[j]:
                raise Error("concatenate: incompatible shapes")

        concat_size += shape[actual_axis]

    # Create result shape
    var result_shape = List[Int]()
    for i in range(ndim):
        if i == actual_axis:
            result_shape.append(concat_size)
        else:
            result_shape.append(ref_shape[i])

    # Create result tensor
    var result = AnyTensor(result_shape, dtype)

    # Copy data from each tensor into the result
    var dtype_size = result._get_dtype_size()

    if actual_axis == 0:
        # Axis-0 concatenation: tensors are simply stacked in memory order
        var offset_bytes = 0
        for tensor_idx in range(num_tensors):
            var t = tensors[tensor_idx]
            var t_numel = t.numel()
            var t_bytes = t_numel * dtype_size

            if is_contiguous(t):
                memcpy(
                    dest=(result._data + offset_bytes).bitcast[UInt8](),
                    src=t._data,
                    count=t_bytes,
                )
            else:
                var t_shape = t.shape()
                var t_ndim = len(t_shape)
                var result_elem_offset = offset_bytes // dtype_size
                for i in range(t_numel):
                    var remaining = i
                    var src_elem_offset = 0
                    for d in range(t_ndim - 1, -1, -1):
                        var coord = remaining % t_shape[d]
                        remaining //= t_shape[d]
                        src_elem_offset += coord * t._strides[d]
                    var src_byte_offset = src_elem_offset * dtype_size
                    var dst_byte_offset = (result_elem_offset + i) * dtype_size
                    var src_ptr = t._data.bitcast[UInt8]()
                    var dst_ptr = result._data.bitcast[UInt8]()
                    for b in range(dtype_size):
                        dst_ptr[dst_byte_offset + b] = src_ptr[src_byte_offset + b]

            offset_bytes += t_bytes
    else:
        # General case: copy slices along the concat axis
        # For each "outer" index (dimensions before concat axis) and each
        # "inner" block (dimensions after concat axis), copy contiguous chunks.

        # Compute the number of outer iterations (product of dims before axis)
        var outer_size = 1
        for i in range(actual_axis):
            outer_size *= result_shape[i]

        # Compute the inner block size (product of dims after axis)
        var inner_size = 1
        for i in range(actual_axis + 1, ndim):
            inner_size *= result_shape[i]

        # Result row stride along concat axis = concat_size * inner_size
        var result_row_width = concat_size * inner_size

        for outer in range(outer_size):
            # For this outer index, copy each tensor's slice
            var col_offset = 0  # offset within the concat axis
            for tensor_idx in range(num_tensors):
                var t = tensors[tensor_idx]
                var t_shape = t.shape()
                var t_axis_size = t_shape[actual_axis]
                var t_row_width = t_axis_size * inner_size

                # Source offset: outer * t_row_width elements
                var src_offset = outer * t_row_width * dtype_size
                # Dest offset: outer * result_row_width + col_offset * inner_size
                var dst_offset = (
                    outer * result_row_width + col_offset * inner_size
                ) * dtype_size

                var chunk_bytes = t_row_width * dtype_size
                memcpy(
                    dest=(result._data + dst_offset).bitcast[UInt8](),
                    src=(t._data + src_offset).bitcast[UInt8](),
                    count=chunk_bytes,
                )

                col_offset += t_axis_size

    return result^


fn stack(tensors: List[AnyTensor], axis: Int = 0) raises -> AnyTensor:
    """Stack tensors along a new axis.

    Args:
            tensors: Vector of tensors to stack (must have identical shapes).
            axis: Position of new axis (default 0).

    Returns:
            Stacked tensor with one additional dimension.

    Raises:
            Error: If tensors have different shapes.

    Examples:
    ```
            var a = ones([2, 3], DType.float32)  # 2x3
            var b = ones([2, 3], DType.float32)  # 2x3
            var tensors : List[AnyTensor] = []
            tensors.append(a)
            tensors.append(b)
            var c = stack(tensors, axis=0)  # Shape (2, 2, 3)
    ```
    """
    var num_tensors = len(tensors)
    if num_tensors == 0:
        raise Error("stack: need at least one tensor")

    # All tensors must have identical shapes
    var ref_shape = tensors[0].shape()
    var ndim = len(ref_shape)
    var dtype = tensors[0].dtype()

    for i in range(1, num_tensors):
        var shape = tensors[i].shape()

        if len(shape) != ndim:
            raise Error(
                "stack: all tensors must have same number of dimensions"
            )

        if tensors[i].dtype() != dtype:
            raise Error("stack: all tensors must have same dtype")

        for j in range(ndim):
            if shape[j] != ref_shape[j]:
                raise Error("stack: all tensors must have identical shapes")

    # Add unsqueeze dimension to each tensor
    var new_ndim = ndim + 1
    var actual_axis = axis if axis >= 0 else new_ndim + axis

    if actual_axis < 0 or actual_axis > ndim:
        raise Error("stack: axis out of range")

    # Unsqueeze each tensor and concatenate
    var unsqueezed: List[AnyTensor] = []
    for i in range(num_tensors):
        unsqueezed.append(unsqueeze(tensors[i], actual_axis))

    return concatenate(unsqueezed, actual_axis)


# ============================================================================
# Split Operations
# ============================================================================


fn split(
    tensor: AnyTensor, num_splits: Int, axis: Int = 0
) raises -> List[AnyTensor]:
    """Split tensor into equal parts along an axis.

    Divides tensor into num_splits equal parts along the specified axis.
    The tensor size along the axis must be divisible by num_splits.

    Args:
        tensor: Input tensor to split.
        num_splits: Number of equal parts to split into.
        axis: Axis along which to split (default: 0).

    Returns:
        List of AnyTensor objects, each with same shape except along split axis.

    Raises:
        Error: If axis is invalid, num_splits <= 0, or tensor size not divisible.

    Examples:
    ```mojo
        var a = arange(0.0, 12.0, 1.0, DType.float32)  # Shape: (12,)
        var parts = split(a, 3)  # 3 parts of size (4,) each

        # For 2D tensor along axis 0:
        var b = ones([6, 4], DType.float32)
        var parts = split(b, 2)  # 2 parts of shape (3, 4)

        # For 2D tensor along axis 1:
        var c = ones([4, 6], DType.float32)
        var parts = split(c, 3, axis=1)  # 3 parts of shape (4, 2)
    ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    # Validate axis
    var actual_axis = axis if axis >= 0 else ndim + axis
    if actual_axis < 0 or actual_axis >= ndim:
        raise Error("split: axis out of range")

    # Validate num_splits
    if num_splits <= 0:
        raise Error("split: num_splits must be positive")

    var split_size = shape[actual_axis]
    if split_size % num_splits != 0:
        raise Error(
            "split: tensor size along axis must be divisible by num_splits"
        )

    var chunk_size = split_size // num_splits

    # Create slices for each split
    var results: List[AnyTensor] = []

    for i in range(num_splits):
        var start_idx = i * chunk_size
        var end_idx = start_idx + chunk_size

        # Use slice() to extract the chunk
        var chunk = tensor.slice(start_idx, end_idx, actual_axis)
        results.append(chunk)

    return results^


fn split_with_indices(
    tensor: AnyTensor, split_indices: List[Int], axis: Int = 0
) raises -> List[AnyTensor]:
    """Split tensor at specified indices along an axis.

    Divides tensor at specified indices along the given axis.
    Indices specify the starting position of each split section.

    Args:
        tensor: Input tensor to split.
        split_indices: List of indices where to split (e.g., [3, 7] splits into 3 sections).
        axis: Axis along which to split (default: 0).

    Returns:
        List of AnyTensor objects resulting from splits.

    Raises:
        Error: If axis is invalid or indices are out of bounds/unordered.

    Examples:
    ```mojo
        # Split [0,1,2,3,4,5,6,7,8,9] at indices [3, 7]
        # Results in: [0-2], [3-6], [7-9]
        var a = arange(0.0, 10.0, 1.0, DType.float32)
        var parts = split_with_indices(a, [3, 7])
        # parts[0].shape() = (3,)  # indices 0-2
        # parts[1].shape() = (4,)  # indices 3-6
        # parts[2].shape() = (3,)  # indices 7-9

        # For 2D tensor:
        var b = ones([10, 5], DType.float32)
        var parts = split_with_indices(b, [3, 7], axis=0)
        # parts[0].shape() = (3, 5)
        # parts[1].shape() = (4, 5)
        # parts[2].shape() = (3, 5)
    ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    # Validate axis
    var actual_axis = axis if axis >= 0 else ndim + axis
    if actual_axis < 0 or actual_axis >= ndim:
        raise Error("split_with_indices: axis out of range")

    var size_along_axis = shape[actual_axis]

    # Validate indices
    var num_indices = len(split_indices)
    if num_indices == 0:
        raise Error("split_with_indices: split_indices cannot be empty")

    # Check that indices are ordered and within bounds
    for i in range(num_indices):
        if split_indices[i] <= 0:
            raise Error("split_with_indices: indices must be positive")
        if split_indices[i] >= size_along_axis:
            raise Error("split_with_indices: index out of bounds")
        if i > 0 and split_indices[i] <= split_indices[i - 1]:
            raise Error(
                "split_with_indices: indices must be strictly increasing"
            )

    # Create slices based on indices
    var results: List[AnyTensor] = []
    var prev_idx = 0

    for i in range(num_indices):
        var curr_idx = split_indices[i]
        if curr_idx > prev_idx:
            var chunk = tensor.slice(prev_idx, curr_idx, actual_axis)
            results.append(chunk)
        prev_idx = curr_idx

    # Add final chunk
    if prev_idx < size_along_axis:
        var final_chunk = tensor.slice(prev_idx, size_along_axis, actual_axis)
        results.append(final_chunk)

    return results^


# ============================================================================
# Shape Computation Functions for Neural Network Layers
# ============================================================================


fn conv2d_output_shape(
    input_h: Int,
    input_w: Int,
    kernel_h: Int,
    kernel_w: Int,
    stride: Int,
    padding: Int,
    dilation: Int = 1,
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D convolution.

        Calculates the spatial output dimensions (height, width) of a 2D convolution
        operation given input dimensions, kernel size, stride, padding, and dilation.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_h: Kernel height in pixels.
            kernel_w: Kernel width in pixels.
            stride: Convolution stride (same for both dimensions).
            padding: Zero-padding added to input (same for all sides).
            dilation: Dilation factor for kernel (default: 1 for standard convolution).

    Returns:
            Tuple of (output_height, output_width).

        Formula:
            output_h = (input_h + 2*padding - dilation*(kernel_h - 1) - 1) // stride + 1
            output_w = (input_w + 2*padding - dilation*(kernel_w - 1) - 1) // stride + 1

    Examples:
    ```
            # Standard 3x3 convolution with stride=1, padding=1
            var out_h, out_w = conv2d_output_shape(224, 224, 3, 3, 1, 1)  # (224, 224)

            # 5x5 convolution with stride=2, padding=2
            var out_h, out_w = conv2d_output_shape(224, 224, 5, 5, 2, 2)  # (112, 112)

            # Dilated convolution (dilation=2)
            var out_h, out_w = conv2d_output_shape(224, 224, 3, 3, 1, 1, dilation=2)  # (222, 222)
    ```
    """
    var out_h = (
        input_h + 2 * padding - dilation * (kernel_h - 1) - 1
    ) // stride + 1
    var out_w = (
        input_w + 2 * padding - dilation * (kernel_w - 1) - 1
    ) // stride + 1
    return Tuple[Int, Int](out_h, out_w)


fn pool_output_shape(
    input_h: Int, input_w: Int, kernel_size: Int, stride: Int, padding: Int
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D pooling.

        Calculates the spatial output dimensions (height, width) of a 2D pooling
        operation given input dimensions, kernel size, stride, and padding.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_size: Pooling window size (square, same for both dimensions).
            stride: Pooling stride (same for both dimensions).
            padding: Zero-padding added to input (same for all sides).

    Returns:
            Tuple of (output_height, output_width).

        Formula:
            output_h = (input_h + 2*padding - kernel_size) // stride + 1
            output_w = (input_w + 2*padding - kernel_size) // stride + 1

    Examples:
    ```
            # 2x2 max pooling with stride=2, no padding
            var out_h, out_w = pool_output_shape(224, 224, 2, 2, 0)  # (112, 112)

            # 3x3 pooling with stride=1, padding=1 (same spatial dims)
            var out_h, out_w = pool_output_shape(224, 224, 3, 1, 1)  # (224, 224)
    ```
    """
    var out_h = (input_h + 2 * padding - kernel_size) // stride + 1
    var out_w = (input_w + 2 * padding - kernel_size) // stride + 1
    return Tuple[Int, Int](out_h, out_w)


fn flatten_size(height: Int, width: Int, channels: Int) -> Int:
    """Compute flattened size for fully connected layer input.

        Calculates the total number of elements in a flattened tensor from
        4D spatial dimensions. Used to determine input size for dense/linear layers
        following convolutional or pooling layers.

    Args:
            height: Spatial height dimension.
            width: Spatial width dimension.
            channels: Number of channels.

    Returns:
            Total number of elements: height * width * channels.

    Examples:
    ```
            # After final pooling layer in CNN
            var fc_input_size = flatten_size(7, 7, 512)  # 25088 for 7x7x512 feature map
            var fc_weight_shape = [4096, 25088]  # Common dense layer size

            # After initial conv layer
            var fc_input_size = flatten_size(112, 112, 64)  # 802816 elements
    ```
    """
    return height * width * channels


fn flatten_to_2d(tensor: AnyTensor) raises -> AnyTensor:
    """Flatten a 4D tensor to 2D, preserving the batch dimension.

        Commonly used before fully connected layers in CNNs to reshape
        (batch, channels, height, width) to (batch, channels * height * width).

    Args:
            tensor: Input tensor of shape (batch, channels, height, width).

    Returns:
            Tensor of shape (batch, channels * height * width).

    Raises:
            Error: If input tensor is not 4D.

    Examples:
    ```
            # After pooling layer, flatten before FC layer
            var pool_out = maxpool2d(x, kernel_size=2, stride=2)  # (32, 64, 7, 7)
            var flattened = flatten_to_2d(pool_out)  # (32, 3136)

            # Use in forward pass
            var fc_input = flatten_to_2d(conv_output)
            var fc_output = linear(fc_input, weights, bias)
    ```

    Note:
            Raises error if input is not 4D.
    """
    var shape = tensor.shape()

    if len(shape) != 4:
        raise Error(
            "flatten_to_2d requires 4D input (batch, channels, height, width),"
            " got "
            + String(len(shape))
            + "D"
        )

    var batch_size = shape[0]
    var channels = shape[1]
    var height = shape[2]
    var width = shape[3]
    var flattened_size = channels * height * width

    var new_shape = List[Int]()
    new_shape.append(batch_size)
    new_shape.append(flattened_size)
    return reshape(tensor, new_shape)


fn transposed_conv2d_output_shape(
    input_h: Int,
    input_w: Int,
    kernel_h: Int,
    kernel_w: Int,
    stride: Int,
    padding: Int,
    output_padding: Int = 0,
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D transposed convolution.

        Calculates the spatial output dimensions (height, width) of a 2D transposed
        convolution (deconvolution) operation. Transposed convolution upsamples the
        input and is commonly used in decoder networks and generative models.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_h: Kernel height in pixels.
            kernel_w: Kernel width in pixels.
            stride: Convolution stride (same for both dimensions).
            padding: Padding applied to input (same for all sides).
            output_padding: Additional padding added to output (default: 0).

    Returns:
            Tuple of (output_height, output_width).

        Formula:
            output_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding
            output_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding

    Examples:
    ```
            # Upsample 7x7 to 14x14 with stride=2
            var out_h, out_w = transposed_conv2d_output_shape(7, 7, 4, 4, 2, 1)  # (14, 14)

            # Upsample 14x14 to 28x28 with stride=2
            var out_h, out_w = transposed_conv2d_output_shape(14, 14, 4, 4, 2, 1)  # (28, 28)
    ```
    """
    var out_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding
    var out_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding
    return Tuple[Int, Int](out_h, out_w)


fn global_avgpool_output_shape(
    batch: Int, channels: Int
) -> Tuple[Int, Int, Int, Int]:
    """Compute output shape for global average pooling.

        Global average pooling reduces each channel to a single value by averaging
        all spatial dimensions. The output has shape (batch, channels, 1, 1).

    Args:
            batch: Batch size.
            channels: Number of channels.

    Returns:
            Tuple of (batch, channels, 1, 1).

    Examples:
    ```
            # Global average pooling on feature map
            var shape = global_avgpool_output_shape(32, 512)  # (32, 512, 1, 1)

            # Common in classification networks (replaces flatten + FC)
            var shape = global_avgpool_output_shape(16, 2048)  # (16, 2048, 1, 1)
    ```
    """
    return Tuple[Int, Int, Int, Int](batch, channels, 1, 1)


fn linear_output_shape(batch_size: Int, out_features: Int) -> Tuple[Int, Int]:
    """Compute output shape for linear/dense layer.

        Linear layers transform input features to output features. The output
        shape is (batch_size, out_features).

    Args:
            batch_size: Number of samples in the batch.
            out_features: Number of output features (neurons).

    Returns:
            Tuple of (batch_size, out_features).

    Examples:
    ```
            # Classification head: 512 features -> 10 classes
            var shape = linear_output_shape(32, 10)  # (32, 10)

            # Hidden layer: 784 features -> 256 hidden units
            var shape = linear_output_shape(64, 256)  # (64, 256)
    ```
    """
    return Tuple[Int, Int](batch_size, out_features)


fn tile(tensor: AnyTensor, reps: List[Int]) raises -> AnyTensor:
    """Tile tensor by repeating along each dimension.

    Args:
            tensor: Input tensor.
            reps: Number of repetitions along each dimension.

    Returns:
            Tiled tensor with shape[i] = input_shape[i] * reps[i].

    Raises:
            Error: If reps is empty.

    Examples:
    ```
            var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
            var reps = List[Int]()
            reps.append(3)
            var b = tile(a, reps)  # [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ```
    """
    if len(reps) == 0:
        raise Error("tile: reps must have at least one element")

    var shape = tensor.shape()
    var ndim = len(shape)
    var nreps = len(reps)

    # Determine output dimensions
    var out_ndim = max(ndim, nreps)

    # Pad shapes with 1s if needed
    var padded_shape = List[Int]()
    var padded_reps = List[Int]()

    # Pad input shape on the left
    for _ in range(out_ndim - ndim):
        padded_shape.append(1)
    for i in range(ndim):
        padded_shape.append(shape[i])

    # Pad reps on the left
    for _ in range(out_ndim - nreps):
        padded_reps.append(1)
    for i in range(nreps):
        padded_reps.append(reps[i])

    # Compute result shape
    var result_shape = List[Int]()
    for i in range(out_ndim):
        result_shape.append(padded_shape[i] * padded_reps[i])

    # Create result tensor
    var result = AnyTensor(result_shape, tensor.dtype())
    var result_numel = result.numel()

    # Fill result by repeating input
    for i in range(result_numel):
        # Compute coordinates in result tensor
        var coords = List[Int]()
        var temp_i = i
        for j in range(out_ndim):
            var stride = 1
            for k in range(j + 1, out_ndim):
                stride *= result_shape[k]
            var coord = temp_i // stride
            coords.append(coord)
            temp_i = temp_i % stride

        # Map to source coordinates using modulo
        var src_idx = 0
        for j in range(out_ndim):
            var src_coord = coords[j] % padded_shape[j]
            var src_stride = 1
            for k in range(j + 1, out_ndim):
                src_stride *= padded_shape[k]
            src_idx += src_coord * src_stride

        # Adjust for original tensor dimensions
        var adjusted_idx: Int
        if out_ndim > ndim:
            # Remove padding from source index calculation
            var temp_idx = src_idx
            for j in range(out_ndim - ndim):
                var stride = 1
                for k in range(j + 1, out_ndim):
                    stride *= padded_shape[k]
                temp_idx = temp_idx % stride
            adjusted_idx = temp_idx
        else:
            adjusted_idx = src_idx

        # Copy value
        var val = tensor._get_float64(adjusted_idx)
        result._set_float64(i, val)

    return result^


fn repeat(tensor: AnyTensor, n: Int, axis: Int = -1) raises -> AnyTensor:
    """Repeat each element n times along axis.

    Args:
            tensor: Input tensor.
            n: Number of times to repeat each element.
            axis: Axis along which to repeat (default -1 flattens first).

    Returns:
            Tensor with elements repeated.

    Raises:
            Error: If axis is out of range or n < 1.

    Examples:
    ```
            var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
            var b = repeat(a, 2)  # [0, 0, 1, 1, 2, 2] (flatten then repeat)
    ```
    """
    if n < 1:
        raise Error("repeat: n must be >= 1")

    if axis == -1:
        # Flatten then repeat each element
        var flat = flatten(tensor)
        var numel = flat.numel()
        var result_shape = List[Int]()
        result_shape.append(numel * n)
        var result = AnyTensor(result_shape, tensor.dtype())

        for i in range(numel):
            var val = flat._get_float64(i)
            for j in range(n):
                result._set_float64(i * n + j, val)

        return result^
    else:
        # Repeat along specific axis
        var shape = tensor.shape()
        var ndim = len(shape)

        # Handle negative axis
        var actual_axis = axis if axis >= 0 else ndim + axis
        if actual_axis < 0 or actual_axis >= ndim:
            raise Error("repeat: axis out of range")

        # Compute result shape
        var result_shape = List[Int]()
        for i in range(ndim):
            if i == actual_axis:
                result_shape.append(shape[i] * n)
            else:
                result_shape.append(shape[i])

        # Create result tensor
        var result = AnyTensor(result_shape, tensor.dtype())
        var result_numel = result.numel()

        # Fill result by repeating elements
        for i in range(result_numel):
            # Compute coordinates in result tensor
            var coords = List[Int]()
            var temp_i = i
            for j in range(ndim):
                var stride = 1
                for k in range(j + 1, ndim):
                    stride *= result_shape[k]
                var coord = temp_i // stride
                coords.append(coord)
                temp_i = temp_i % stride

            # Map to source coordinates
            var src_coords = List[Int]()
            for j in range(ndim):
                if j == actual_axis:
                    src_coords.append(coords[j] // n)
                else:
                    src_coords.append(coords[j])

            # Compute source index
            var src_idx = 0
            for j in range(ndim):
                var src_stride = 1
                for k in range(j + 1, ndim):
                    src_stride *= shape[k]
                src_idx += src_coords[j] * src_stride

            # Copy value
            var val = tensor._get_float64(src_idx)
            result._set_float64(i, val)

        return result^


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] broadcast_to implementation
# ============================================================================


fn _broadcast_to_typed[
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
    from shared.base.broadcasting import (
        are_shapes_broadcastable,
        compute_broadcast_strides,
    )

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


fn _broadcast_to_dispatch[
    dtype: DType
](tensor: AnyTensor, target_shape: List[Int]) raises -> AnyTensor:
    """Dispatch broadcast_to to typed core."""
    return _broadcast_to_typed[dtype](
        tensor.as_tensor[dtype](), target_shape
    ).as_any()


fn broadcast_to(tensor: AnyTensor, target_shape: List[Int]) raises -> AnyTensor:
    """Broadcast tensor to target shape.

    Args:
            tensor: Input tensor.
            target_shape: Target shape to broadcast to.

    Returns:
            Broadcasted tensor.

    Raises:
            Error: If shapes are not broadcast-compatible.
            Error: If `target_shape` has fewer dimensions than the input tensor
                (broadcasting cannot reduce the number of dimensions; it can only
                expand dimensions of size 1 or prepend new dimensions of any size).

    Examples:
    ```
            var a = arange(0.0, 3.0, 1.0, DType.float32)  # Shape (3,)
            var target = List[Int]()
            target.append(4)
            target.append(3)
            var b = broadcast_to(a, target)  # Shape (4, 3)
    ```
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _broadcast_to_dispatch[DType.float16](tensor, target_shape)
    elif ordinal == DTYPE_FLOAT32:
        return _broadcast_to_dispatch[DType.float32](tensor, target_shape)
    elif ordinal == DTYPE_FLOAT64:
        return _broadcast_to_dispatch[DType.float64](tensor, target_shape)
    elif ordinal == DTYPE_INT8:
        return _broadcast_to_dispatch[DType.int8](tensor, target_shape)
    elif ordinal == DTYPE_INT16:
        return _broadcast_to_dispatch[DType.int16](tensor, target_shape)
    elif ordinal == DTYPE_INT32:
        return _broadcast_to_dispatch[DType.int32](tensor, target_shape)
    elif ordinal == DTYPE_INT64:
        return _broadcast_to_dispatch[DType.int64](tensor, target_shape)
    elif ordinal == DTYPE_UINT8:
        return _broadcast_to_dispatch[DType.uint8](tensor, target_shape)
    elif ordinal == DTYPE_UINT16:
        return _broadcast_to_dispatch[DType.uint16](tensor, target_shape)
    elif ordinal == DTYPE_UINT32:
        return _broadcast_to_dispatch[DType.uint32](tensor, target_shape)
    elif ordinal == DTYPE_UINT64:
        return _broadcast_to_dispatch[DType.uint64](tensor, target_shape)
    else:
        raise Error("broadcast_to: unsupported dtype")

fn broadcast_to[dt: DType](
    tensor: Tensor[dt], target_shape: List[Int]
) raises -> Tensor[dt]:
    """Broadcast tensor to target shape (typed overload).

    Parameters:
        dt: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        target_shape: Shape to broadcast to.

    Returns:
        A new Tensor[dt] broadcast to the target shape.
    """
    return _broadcast_to_typed[dt](tensor, target_shape)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] permute implementation
# ============================================================================


fn _validate_permute_dims(dims: List[Int], ndim: Int) raises:
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


fn _permute_typed[
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

        # Compute source index
        var src_idx = 0
        for j in range(ndim):
            var src_stride = 1
            for k in range(j + 1, ndim):
                src_stride *= shape[k]
            src_idx += src_coords[j] * src_stride

        # Copy value using typed pointer -- zero bitcasts
        dst_ptr[i] = src_ptr[src_idx]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch for permute
# ============================================================================


fn _permute_dispatch[
    dtype: DType
](tensor: AnyTensor, dims: List[Int]) raises -> AnyTensor:
    """Dispatch permute to typed core."""
    return _permute_typed[dtype](
        tensor.as_tensor[dtype](), dims
    ).as_any()


fn permute(tensor: AnyTensor, dims: List[Int]) raises -> AnyTensor:
    """Permute tensor dimensions (generalized transpose).

    Args:
            tensor: Input tensor.
            dims: Permutation of dimensions (must be valid permutation of [0..ndim-1]).

    Returns:
            Tensor with permuted dimensions.

    Raises:
            Error: If dims is not a valid permutation.

    Examples:
    ```
            var a = ones([2, 3, 4], DType.float32)  # Shape (2, 3, 4)
            var perm = List[Int]()
            perm.append(2)
            perm.append(0)
            perm.append(1)
            var b = permute(a, perm)  # Shape (4, 2, 3)
    ```
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())

    if ordinal == DTYPE_FLOAT16:
        return _permute_dispatch[DType.float16](tensor, dims)
    elif ordinal == DTYPE_FLOAT32:
        return _permute_dispatch[DType.float32](tensor, dims)
    elif ordinal == DTYPE_FLOAT64:
        return _permute_dispatch[DType.float64](tensor, dims)
    elif ordinal == DTYPE_INT8:
        return _permute_dispatch[DType.int8](tensor, dims)
    elif ordinal == DTYPE_INT16:
        return _permute_dispatch[DType.int16](tensor, dims)
    elif ordinal == DTYPE_INT32:
        return _permute_dispatch[DType.int32](tensor, dims)
    elif ordinal == DTYPE_INT64:
        return _permute_dispatch[DType.int64](tensor, dims)
    elif ordinal == DTYPE_UINT8:
        return _permute_dispatch[DType.uint8](tensor, dims)
    elif ordinal == DTYPE_UINT16:
        return _permute_dispatch[DType.uint16](tensor, dims)
    elif ordinal == DTYPE_UINT32:
        return _permute_dispatch[DType.uint32](tensor, dims)
    elif ordinal == DTYPE_UINT64:
        return _permute_dispatch[DType.uint64](tensor, dims)
    else:
        raise Error("permute: unsupported dtype")

fn permute[dt: DType](tensor: Tensor[dt], dims: List[Int]) raises -> Tensor[dt]:
    """Permute tensor dimensions (typed overload).

    Parameters:
        dt: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.
        dims: Permutation of dimensions.

    Returns:
        A new Tensor[dt] with permuted dimensions.
    """
    return _permute_typed[dt](tensor, dims)


# ============================================================================
# Typed Tensor[dtype] overloads — delegate to typed cores directly
# ============================================================================


fn reshape_typed[dt: DType](
    tensor: Tensor[dt], new_shape: List[Int]
) raises -> Tensor[dt]:
    """Reshape tensor to new shape (typed version).

    Args:
        tensor: Input typed tensor.
        new_shape: Target shape (must have same total elements).

    Returns:
        A new Tensor[dt] with the given shape.
    """
    return _reshape_typed[dt](tensor, new_shape)


fn squeeze_typed[dt: DType](
    tensor: Tensor[dt], axis: Int = -999
) raises -> Tensor[dt]:
    """Remove size-1 dimensions (typed version).

    Args:
        tensor: Input typed tensor.
        axis: Specific axis to squeeze, or -999 for all size-1 dims.

    Returns:
        A new Tensor[dt] with size-1 dimensions removed.
    """
    # Compute the squeezed shape, then delegate to typed reshape
    var old_shape = tensor.shape()
    var ndim = len(old_shape)

    if axis != -999:
        var actual_axis = axis if axis >= 0 else ndim + axis
        if actual_axis < 0 or actual_axis >= ndim:
            raise Error("squeeze: dimension out of range")
        if old_shape[actual_axis] != 1:
            raise Error("squeeze: cannot squeeze dimension that is not size 1")
        var new_shape = List[Int]()
        for i in range(ndim):
            if i != actual_axis:
                new_shape.append(old_shape[i])
        return _reshape_typed[dt](tensor, new_shape)
    else:
        var new_shape = List[Int]()
        for i in range(ndim):
            if old_shape[i] != 1:
                new_shape.append(old_shape[i])
        if len(new_shape) == ndim:
            return _reshape_typed[dt](tensor, old_shape)
        return _reshape_typed[dt](tensor, new_shape)


fn unsqueeze_typed[dt: DType](tensor: Tensor[dt], axis: Int) raises -> Tensor[dt]:
    """Insert a size-1 dimension at the given axis (typed version).

    Args:
        tensor: Input typed tensor.
        axis: Position to insert the new dimension.

    Returns:
        A new Tensor[dt] with an added size-1 dimension.
    """
    var old_shape = tensor.shape()
    var ndim = len(old_shape)
    var new_ndim = ndim + 1
    var actual_axis = axis if axis >= 0 else new_ndim + axis
    if actual_axis < 0 or actual_axis > ndim:
        raise Error("unsqueeze: dimension out of range")
    var new_shape = List[Int]()
    var j = 0
    for i in range(new_ndim):
        if i == actual_axis:
            new_shape.append(1)
        else:
            new_shape.append(old_shape[j])
            j += 1
    return _reshape_typed[dt](tensor, new_shape)


fn expand_dims_typed[dt: DType](
    tensor: Tensor[dt], axis: Int
) raises -> Tensor[dt]:
    """Insert a size-1 dimension at the given axis (typed version).

    Alias for unsqueeze, following NumPy naming convention.

    Args:
        tensor: Input typed tensor.
        axis: Position to insert the new dimension.

    Returns:
        A new Tensor[dt] with an added size-1 dimension.
    """
    return unsqueeze_typed[dt](tensor, axis)


fn flatten_typed[dt: DType](tensor: Tensor[dt]) raises -> Tensor[dt]:
    """Flatten tensor to 1D (typed version).

    Args:
        tensor: Input typed tensor.

    Returns:
        A new 1D Tensor[dt] with all elements in row-major order.
    """
    var shape_1d = List[Int]()
    shape_1d.append(tensor.numel())
    return _reshape_typed[dt](tensor, shape_1d)


fn broadcast_to_typed[dt: DType](
    tensor: Tensor[dt], target_shape: List[Int]
) raises -> Tensor[dt]:
    """Broadcast tensor to target shape (typed version).

    Args:
        tensor: Input typed tensor.
        target_shape: Shape to broadcast to.

    Returns:
        A new Tensor[dt] broadcast to the target shape.
    """
    return _broadcast_to_typed[dt](tensor, target_shape)


fn permute_typed[dt: DType](
    tensor: Tensor[dt], dims: List[Int]
) raises -> Tensor[dt]:
    """Permute tensor dimensions (typed version).

    Args:
        tensor: Input typed tensor.
        dims: Permutation of dimensions.

    Returns:
        A new Tensor[dt] with permuted dimensions.
    """
    return _permute_typed[dt](tensor, dims)
