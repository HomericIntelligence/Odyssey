"""Broadcasting utilities for ExTensor.

Implements NumPy-style broadcasting rules for tensor operations
"""


fn broadcast_shapes(shape1: List[Int], shape2: List[Int]) raises -> List[Int]:
    """Compute the broadcast shape of two tensor shapes.

    Args:
        shape1: First tensor shape.
        shape2: Second tensor shape.

    Returns:
        The broadcast result shape.

    Raises:
        Error if shapes are not broadcast-compatible.

    Broadcasting rules:
        1. Compare shapes element-wise from right to left
        2. Dimensions are compatible if they are equal or one is 1
        3. Missing dimensions are treated as 1
        4. Output shape is element-wise maximum of input shapes

    Examples:
    ```
        broadcast_shapes([3, 4, 5], [4, 5]) -> [3, 4, 5]
        broadcast_shapes([3, 1, 5], [3, 4, 5]) -> [3, 4, 5]
        broadcast_shapes([3, 4], [5, 4]) -> Error (incompatible)
    ```
    """
    var ndim1 = len(shape1)
    var ndim2 = len(shape2)
    var max_ndim = max(ndim1, ndim2)

    var result_shape = List[Int]()

    # Process dimensions from right to left
    for i in range(max_ndim):
        # Get dimension from each shape (1 if dimension doesn't exist)
        var dim1_idx = ndim1 - 1 - i
        var dim2_idx = ndim2 - 1 - i

        var dim1 = shape1[dim1_idx] if dim1_idx >= 0 else 1
        var dim2 = shape2[dim2_idx] if dim2_idx >= 0 else 1

        # Check compatibility
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise Error(
                "Shapes are not broadcast-compatible: dimension "
                + String(i)
                + " has sizes "
                + String(dim1)
                + " and "
                + String(dim2)
            )

        # Result dimension is the maximum
        var result_dim = max(dim1, dim2)
        result_shape.append(result_dim)

    # Reverse to get correct order (we built it backwards)
    var final_shape = List[Int]()
    for i in range(len(result_shape) - 1, -1, -1):
        final_shape.append(result_shape[i])

    return final_shape^


fn are_shapes_broadcastable(shape1: List[Int], shape2: List[Int]) -> Bool:
    """Check if two shapes are broadcast-compatible.

    Args:
        shape1: First tensor shape.
        shape2: Second tensor shape.

    Returns:
        True if shapes are broadcast-compatible, False otherwise.

    Examples:
    ```
        are_shapes_broadcastable([3, 4, 5], [4, 5]) -> True
        are_shapes_broadcastable([3, 4], [5, 4]) -> False
    ```
    """
    var ndim1 = len(shape1)
    var ndim2 = len(shape2)
    var max_ndim = max(ndim1, ndim2)

    for i in range(max_ndim):
        var dim1_idx = ndim1 - 1 - i
        var dim2_idx = ndim2 - 1 - i

        var dim1 = shape1[dim1_idx] if dim1_idx >= 0 else 1
        var dim2 = shape2[dim2_idx] if dim2_idx >= 0 else 1

        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    return True


fn compute_broadcast_strides(
    original_shape: List[Int],
    broadcast_shape: List[Int],
) -> List[Int]:
    """Compute strides for broadcasting a tensor to a new shape.

    Args:
        original_shape: The original tensor shape.
        broadcast_shape: The target broadcast shape.

    Returns:
        Strides for the broadcast tensor (0 for broadcasted dimensions).

    Note:
        Dimensions that are 1 in the original shape get stride 0 in the broadcast.
        This allows efficient broadcasting without materializing extra copies.

    Examples:
        ```
        original_shape = [3, 1, 5]
        broadcast_shape = [3, 4, 5]
        result = [stride_for_3, 0, stride_for_5]  # Middle dimension has stride 0
        ```
    """
    var ndim_orig = len(original_shape)
    var ndim_broad = len(broadcast_shape)

    var broadcast_strides = List[Int]()

    # Calculate original row-major strides
    # For shape [D0, D1, D2], strides are [D1*D2, D2, 1]
    var orig_strides_final = List[Int]()
    for i in range(ndim_orig):
        var stride = 1
        for j in range(i + 1, ndim_orig):
            stride *= original_shape[j]
        orig_strides_final.append(stride)

    # Compute broadcast strides
    for i in range(ndim_broad):
        var orig_idx = i - (ndim_broad - ndim_orig)

        if orig_idx < 0:
            # Dimension doesn't exist in original -> stride 0
            broadcast_strides.append(0)
        elif original_shape[orig_idx] == 1 and broadcast_shape[i] > 1:
            # Dimension is 1 and being broadcast -> stride 0
            broadcast_strides.append(0)
        else:
            # Normal dimension -> use original stride
            broadcast_strides.append(orig_strides_final[orig_idx])

    return broadcast_strides^


struct BroadcastIterator:
    """Iterator for efficiently iterating over broadcast tensor elements.

    This allows element-wise operations to work efficiently with broadcasting
    without materializing the full broadcast tensor.

    Note:
        __iter__() is intentionally not implemented because List[Int] fields
        are not Copyable, and __iter__ would need to return Self which requires
        copying. Use __next__ directly with a while loop instead:

        ```mojo
        var iterator = BroadcastIterator(shape, strides1, strides2)
        while iterator.has_next():
            var (idx1, idx2) = iterator.__next__()
            # Use idx1 and idx2 to access elements
        ```
    """

    var shape: List[Int]
    var strides1: List[Int]
    var strides2: List[Int]
    var size: Int
    var position: Int

    fn __init__(
        out self,
        var shape: List[Int],
        var strides1: List[Int],
        var strides2: List[Int],
    ):
        """Initialize broadcast iterator.

        Args:
            shape: The broadcast output shape.
            strides1: Broadcast strides for first tensor.
            strides2: Broadcast strides for second tensor.
        """
        self.shape = shape^
        self.strides1 = strides1^
        self.strides2 = strides2^
        self.position = 0

        # Calculate total size
        self.size = 1
        for i in range(len(self.shape)):
            self.size *= self.shape[i]

    fn __next__(mut self) raises -> Tuple[Int, Int]:
        """Get next pair of indices for the two tensors.

        Returns:
            Tuple of (index1, index2) for accessing elements.

        Raises:
            Error when iteration is complete.

        Algorithm:
            Converts flat position to multi-dimensional coordinates using the
            broadcast shape and row-major (C-style) layout, then applies
            broadcast strides to get indices for both tensors. Broadcasting
            strides are 0 for dimensions that are being broadcast, allowing
            efficient iteration without materializing the full broadcast tensor.
        """
        if self.position >= self.size:
            raise Error("Iterator exhausted")

        # Compute multi-dimensional coordinates from flat position (row-major)
        # For shape [D0, D1, D2], position p -> coords where:
        #   coord[0] = p // (D1 * D2)
        #   coord[1] = (p % (D1 * D2)) // D2
        #   coord[2] = p % D2
        var pos = self.position
        var idx1 = 0
        var idx2 = 0
        var ndim = len(self.shape)

        # Process each dimension from left to right
        # We need to compute strides for coordinate extraction
        for i in range(ndim):
            # Compute the product of dimensions to the right
            var stride = 1
            for j in range(i + 1, ndim):
                stride *= self.shape[j]

            # Extract coordinate for this dimension
            var coord = pos // stride
            pos = pos % stride

            # Apply broadcast strides (handles stride 0 for broadcast dims)
            idx1 += coord * self.strides1[i]
            idx2 += coord * self.strides2[i]

        self.position += 1
        return (idx1, idx2)

    fn has_next(self) -> Bool:
        """Check if more elements remain."""
        return self.position < self.size
