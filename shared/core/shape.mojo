"""Shape manipulation operations for ExTensor.

Implements shape operations like reshape, squeeze, unsqueeze, flatten, concatenate, stack, split
Following the Python Array API Standard 2023.12.

Optimizations:
- Zero-copy views with stride-based indexing
- memcpy for bulk copying of contiguous memory blocks
- Automatic contiguity detection and conversion
"""

from collections import List
from memory import memcpy, UnsafePointer
from shared.core.extensor import ExTensor


# ============================================================================
# Zero-Copy Views and Memory Optimization Helpers
# ============================================================================


fn is_contiguous(tensor: ExTensor) -> Bool:
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
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim == 0:
        return True  # Scalar is trivially contiguous

    if ndim == 1:
        # 1D tensor is contiguous if stride is 1
        return tensor._strides[0] == 1

    # For multi-dimensional tensors, check if strides match C-order
    # In C-order (row-major), stride[i] = product(shape[i+1:])
    var expected_stride = 1
    for i in range(ndim - 1, -1, -1):
        if tensor._strides[i] != expected_stride:
            return False
        expected_stride *= shape[i]

    return True


fn as_contiguous(tensor: ExTensor) raises -> ExTensor:
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
    if is_contiguous(tensor):
        # Already contiguous - just copy
        var shape = tensor.shape()
        var result = ExTensor(shape, tensor.dtype())
        var numel = tensor.numel()

        # Use memcpy for efficient bulk copy
        var dtype_size = tensor._get_dtype_size()
        var total_bytes = numel * dtype_size
        memcpy(dest=result._data, src=tensor._data, count=total_bytes)

        return result^
    else:
        # Non-contiguous - use stride-based indexing to read source elements
        var shape = tensor.shape()
        var ndim = len(shape)
        var result = ExTensor(shape, tensor.dtype())
        var numel = tensor.numel()
        var dtype_size = tensor._get_dtype_size()
        var src_ptr = tensor._data
        var dst_ptr = result._data

        for i in range(numel):
            # Convert flat output index i to multi-dimensional coords (C-order),
            # then compute source byte offset using the tensor's actual strides.
            var src_elem_offset = 0
            var remaining = i
            for d in range(ndim - 1, -1, -1):
                var coord = remaining % shape[d]
                remaining //= shape[d]
                src_elem_offset += coord * tensor._strides[d]

            var src_byte_offset = src_elem_offset * dtype_size
            var dst_byte_offset = i * dtype_size
            for b in range(dtype_size):
                dst_ptr[dst_byte_offset + b] = src_ptr[src_byte_offset + b]

        return result^


fn view(tensor: ExTensor, new_shape: List[Int]) raises -> ExTensor:
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
            A new ExTensor sharing the same data with different shape/strides.

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

    # Use ExTensor's built-in reshape which creates views via __copyinit__
    # This leverages reference counting for safe shared ownership
    return tensor.reshape(new_shape)


fn reshape(tensor: ExTensor, new_shape: List[Int]) raises -> ExTensor:
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
    # Calculate total elements in new shape (handling -1 for inference)
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

    # If we have -1, infer that dimension
    var final_shape = List[Int]()
    var total_elements = tensor.numel()

    if inferred_dim != -1:
        # Infer the -1 dimension
        if total_elements % known_product != 0:
            raise Error("reshape: cannot infer dimension, incompatible size")
        var inferred_size = total_elements // known_product

        for i in range(new_len):
            if i == inferred_dim:
                final_shape.append(inferred_size)
            else:
                final_shape.append(new_shape[i])
    else:
        # No -1, just copy
        for i in range(new_len):
            final_shape.append(new_shape[i])

    # Verify total elements match
    var new_total: Int = 1
    for i in range(new_len):
        new_total *= final_shape[i]

    if new_total != total_elements:
        raise Error("reshape: new shape must have same number of elements")

    # Create new tensor with new shape (data copy)
    var result = ExTensor(final_shape, tensor.dtype())

    # Copy data element by element (row-major order preserved)
    for i in range(total_elements):
        var val = tensor._get_float64(i)
        result._set_float64(i, val)

    return result^


fn squeeze(tensor: ExTensor, axis: Int = -999) raises -> ExTensor:
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


fn unsqueeze(tensor: ExTensor, axis: Int) raises -> ExTensor:
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
fn expand_dims(tensor: ExTensor, axis: Int) raises -> ExTensor:
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


fn flatten(tensor: ExTensor) raises -> ExTensor:
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
fn ravel(tensor: ExTensor) raises -> ExTensor:
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


fn concatenate(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
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
            var tensors : List[ExTensor] = []
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
    var result = ExTensor(result_shape, dtype)

    # Copy data from each tensor with memcpy optimization for contiguous tensors
    var dtype_size = result._get_dtype_size()
    var offset_bytes = 0

    for tensor_idx in range(num_tensors):
        var t = tensors[tensor_idx]
        var t_numel = t.numel()
        var t_bytes = t_numel * dtype_size

        # Use memcpy for efficient bulk copy if source is contiguous
        if is_contiguous(t):
            memcpy(
                dest=(result._data + offset_bytes).bitcast[UInt8](),
                src=t._data,
                count=t_bytes,
            )
        else:
            # Fall back to element-by-element copy for non-contiguous tensors
            for i in range(t_numel):
                var val = t._get_float64(i)
                result._set_float64(offset_bytes // dtype_size + i, val)

        offset_bytes += t_bytes

    return result^


fn stack(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
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
            var tensors : List[ExTensor] = []
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
    var unsqueezed: List[ExTensor] = []
    for i in range(num_tensors):
        unsqueezed.append(unsqueeze(tensors[i], actual_axis))

    return concatenate(unsqueezed, actual_axis)


# ============================================================================
# Split Operations
# ============================================================================


fn split(
    tensor: ExTensor, num_splits: Int, axis: Int = 0
) raises -> List[ExTensor]:
    """Split tensor into equal parts along an axis.

    Divides tensor into num_splits equal parts along the specified axis.
    The tensor size along the axis must be divisible by num_splits.

    Args:
        tensor: Input tensor to split.
        num_splits: Number of equal parts to split into.
        axis: Axis along which to split (default: 0).

    Returns:
        List of ExTensor objects, each with same shape except along split axis.

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
    var results: List[ExTensor] = []

    for i in range(num_splits):
        var start_idx = i * chunk_size
        var end_idx = start_idx + chunk_size

        # Use slice() to extract the chunk
        var chunk = tensor.slice(start_idx, end_idx, actual_axis)
        results.append(chunk)

    return results^


fn split_with_indices(
    tensor: ExTensor, split_indices: List[Int], axis: Int = 0
) raises -> List[ExTensor]:
    """Split tensor at specified indices along an axis.

    Divides tensor at specified indices along the given axis.
    Indices specify the starting position of each split section.

    Args:
        tensor: Input tensor to split.
        split_indices: List of indices where to split (e.g., [3, 7] splits into 3 sections).
        axis: Axis along which to split (default: 0).

    Returns:
        List of ExTensor objects resulting from splits.

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
    var results: List[ExTensor] = []
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


fn flatten_to_2d(tensor: ExTensor) raises -> ExTensor:
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


fn tile(tensor: ExTensor, reps: List[Int]) raises -> ExTensor:
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
    var result = ExTensor(result_shape, tensor.dtype())
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


fn repeat(tensor: ExTensor, n: Int, axis: Int = -1) raises -> ExTensor:
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
        var result = ExTensor(result_shape, tensor.dtype())

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
        var result = ExTensor(result_shape, tensor.dtype())
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


fn broadcast_to(tensor: ExTensor, target_shape: List[Int]) raises -> ExTensor:
    """Broadcast tensor to target shape.

    Args:
            tensor: Input tensor.
            target_shape: Target shape to broadcast to.

    Returns:
            Broadcasted tensor.

    Raises:
            Error: If shapes are not broadcast-compatible.

    Examples:
    ```
            var a = arange(0.0, 3.0, 1.0, DType.float32)  # Shape (3,)
            var target = List[Int]()
            target.append(4)
            target.append(3)
            var b = broadcast_to(a, target)  # Shape (4, 3)
    ```
    """
    from shared.core.broadcasting import (
        are_shapes_broadcastable,
        compute_broadcast_strides,
    )

    var shape = tensor.shape()

    # Cannot reduce number of dimensions (target must have >= dims than source)
    if len(target_shape) < len(shape):
        raise Error("broadcast_to: cannot broadcast to fewer dimensions")

    # Check if broadcast is valid
    if not are_shapes_broadcastable(shape, target_shape):
        raise Error("broadcast_to: shapes are not broadcast-compatible")

    # Compute broadcast strides
    var broadcast_strides = compute_broadcast_strides(shape, target_shape)

    # Create result tensor
    var result = ExTensor(target_shape, tensor.dtype())
    var result_numel = result.numel()

    # Fill result using broadcast strides
    for i in range(result_numel):
        # Compute coordinates in result tensor
        var coords = List[Int]()
        var temp_i = i
        for j in range(len(target_shape)):
            var stride = 1
            for k in range(j + 1, len(target_shape)):
                stride *= target_shape[k]
            var coord = temp_i // stride
            coords.append(coord)
            temp_i = temp_i % stride

        # Compute source index using broadcast strides
        var src_idx = 0
        for j in range(len(target_shape)):
            src_idx += coords[j] * broadcast_strides[j]

        # Copy value
        var val = tensor._get_float64(src_idx)
        result._set_float64(i, val)

    return result^


fn permute(tensor: ExTensor, dims: List[Int]) raises -> ExTensor:
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
    var shape = tensor.shape()
    var ndim = len(shape)

    # Validate dims is correct length
    if len(dims) != ndim:
        raise Error(
            "permute: dims length "
            + String(len(dims))
            + " must equal tensor ndim "
            + String(ndim)
        )

    # Validate dims is a permutation of [0..ndim-1]
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

    # Compute new shape
    var new_shape = List[Int]()
    for i in range(ndim):
        new_shape.append(shape[dims[i]])

    # Create result tensor
    var result = ExTensor(new_shape, tensor.dtype())
    var result_numel = result.numel()

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
            src_coords.append(0)  # Initialize

        for j in range(ndim):
            src_coords[dims[j]] = result_coords[j]

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
