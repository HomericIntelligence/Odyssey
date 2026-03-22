"""Utility functions for tensor operations.

This module provides common utility functions for tensor manipulation including:
- argmax: Find index of maximum value
- top_k_indices: Find indices of top k maximum values
- top_k: Find top k values and their indices
- argsort: Sort indices by values

All functions work with AnyTensor and follow the pure functional design pattern
"""

from collections import List
from .any_tensor import AnyTensor


fn argmax(tensor: AnyTensor) raises -> Int:
    """Find the index of the maximum value in a flattened tensor.

    Args:
        tensor: Input tensor.

    Returns:
        The linear index of the maximum element.

    Raises:
        Error: If tensor is empty.

    Examples:
        ```mojo
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        var idx = argmax(t)  # Returns 9
        ```
    """
    if tensor.numel() == 0:
        raise Error("argmax: tensor is empty")

    var max_val = tensor._get_float64(0)
    var max_idx = 0

    for i in range(1, tensor.numel()):
        var val = tensor._get_float64(i)
        if val > max_val:
            max_val = val
            max_idx = i

    return max_idx


fn argmax(tensor: AnyTensor, axis: Int) raises -> AnyTensor:
    """Find indices of maximum values along an axis.

    Args:
        tensor: Input tensor.
        axis: Axis along which to find argmax.

    Returns:
        A tensor of indices (dtype: int64) with reduced dimensions

    Raises:
        Error: If axis is out of bounds

    Examples:
        ```mojo
        var t = ones([3, 4], DType.float32)
        var indices = argmax(t, axis=1)  # Shape: [3]
        ```
    """
    if axis < 0 or axis >= tensor.dim():
        raise Error(
            "argmax: axis "
            + String(axis)
            + " is out of bounds for tensor with "
            + String(tensor.dim())
            + " dimensions"
        )

    # Build result shape
    var result_shape = List[Int]()
    for i in range(tensor.dim()):
        if i != axis:
            result_shape.append(tensor.shape()[i])

    var result = AnyTensor(result_shape, DType.int64)

    # Compute strides for indexing
    var input_shape = tensor.shape()
    var ndim = tensor.dim()
    var strides = List[Int]()
    for _ in range(ndim):
        strides.append(0)
    var stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = stride
        stride *= input_shape[i]

    var axis_size = input_shape[axis]

    # For each position in the result
    for result_idx in range(result.numel()):
        # Convert result index to coordinates
        var result_dim = result.dim()
        var result_coords = List[Int]()
        for _ in range(result_dim):
            result_coords.append(0)
        var temp_idx = result_idx
        for i in range(result_dim - 1, -1, -1):
            result_coords[i] = temp_idx % result.shape()[i]
            temp_idx //= result.shape()[i]

        # Map result coordinates to input coordinates (accounting for reduced axis)
        var tensor_dim = tensor.dim()
        var input_coords = List[Int]()
        for _ in range(tensor_dim):
            input_coords.append(0)
        var result_coord_idx = 0
        for i in range(tensor_dim):
            if i != axis:
                input_coords[i] = result_coords[result_coord_idx]
                result_coord_idx += 1
            else:
                input_coords[i] = 0  # Will iterate over this

        # Find argmax along the reduction axis
        input_coords[axis] = 0
        var linear_idx = 0
        for i in range(tensor.dim()):
            linear_idx += input_coords[i] * strides[i]
        var max_val = tensor._get_float64(linear_idx)
        var max_idx = 0

        # Compare with remaining values
        for k in range(1, axis_size):
            input_coords[axis] = k

            # Convert coordinates to linear index
            linear_idx = 0
            for i in range(tensor.dim()):
                linear_idx += input_coords[i] * strides[i]

            var val = tensor._get_float64(linear_idx)
            if val > max_val:
                max_val = val
                max_idx = k

        result._set_int64(result_idx, Int64(max_idx))

    return result^


fn top_k_indices(tensor: AnyTensor, k: Int) raises -> List[Int]:
    """Find indices of the k largest values in a flattened tensor.

    Args:
        tensor: Input tensor.
        k: Number of top values to find.

    Returns:
        List of indices sorted by their values (descending)

    Raises:
        Error: If k is invalid or tensor is empty

    Examples:
        ```mojo
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        var indices = top_k_indices(t, 3)  # Returns [9, 8, 7]
        ```
    """
    if k < 0:
        raise Error("top_k_indices: k must be non-negative, got " + String(k))
    if k > tensor.numel():
        raise Error(
            "top_k_indices: k ("
            + String(k)
            + ") cannot exceed tensor size ("
            + String(tensor.numel())
            + ")"
        )
    if tensor.numel() == 0:
        raise Error("top_k_indices: tensor is empty")

    var numel = tensor.numel()

    # Create list of (value, index) pairs
    var pairs = List[Tuple[Float64, Int]]()
    for i in range(numel):
        var val = tensor._get_float64(i)
        pairs.append((val, i))

    # Simple selection sort to find top k (not the most efficient, but correct)
    for i in range(k):
        var max_idx = i
        for j in range(i + 1, numel):
            var max_val = pairs[max_idx][0]
            var curr_val = pairs[j][0]
            if curr_val > max_val:
                max_idx = j

        # Swap
        if max_idx != i:
            var temp = pairs[i]
            pairs[i] = pairs[max_idx]
            pairs[max_idx] = temp

    # Extract indices of top k
    var result = List[Int]()
    for i in range(k):
        result.append(pairs[i][1])

    return result^


fn top_k(tensor: AnyTensor, k: Int) raises -> Tuple[AnyTensor, List[Int]]:
    """Find the k largest values and their indices in a flattened tensor.

    Args:
        tensor: Input tensor.
        k: Number of top values to find.

    Returns:
        A tuple containing:
        - A tensor of top k values (shape: [k])
        - A list of their corresponding indices

    Raises:
        Error: If k is invalid or tensor is empty

    Examples:
        ```mojo
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        var (values, indices) = top_k(t, 3)  # Values: [10.0, 9.0, 8.0], Indices: [9, 8, 7]
        ```
    """
    var indices = top_k_indices(tensor, k)

    # Create result tensor for values
    var values_shape = List[Int]()
    values_shape.append(k)
    var values = AnyTensor(values_shape, tensor.dtype())

    for i in range(k):
        var idx = indices[i]
        var val = tensor._get_float64(idx)
        values._set_float64(i, val)

    return (values, indices^)


fn argsort(tensor: AnyTensor, descending: Bool = False) raises -> List[Int]:
    """Return indices that would sort the tensor.

    Args:
        tensor: Input tensor.
        descending: If True, sort in descending order. If False, ascending.

    Returns:
        List of indices that sort the tensor

    Raises:
        Error: If tensor is empty

    Examples:
        ```mojo
        var t = arange(5.0, 0.0, -1.0, DType.float32)  # [5, 4, 3, 2, 1]
        var idx = argsort(t, descending=False)  # Returns [4, 3, 2, 1, 0] for ascending
        ```
    """
    if tensor.numel() == 0:
        raise Error("argsort: tensor is empty")

    var numel = tensor.numel()

    # Create list of (value, index) pairs
    var pairs = List[Tuple[Float64, Int]]()
    for i in range(numel):
        var val = tensor._get_float64(i)
        pairs.append((val, i))

    # Bubble sort (simple and correct, not the most efficient)
    for i in range(numel):
        for j in range(0, numel - i - 1):
            var should_swap = False
            if descending:
                if pairs[j][0] < pairs[j + 1][0]:
                    should_swap = True
            else:
                if pairs[j][0] > pairs[j + 1][0]:
                    should_swap = True

            if should_swap:
                var temp = pairs[j]
                pairs[j] = pairs[j + 1]
                pairs[j + 1] = temp

    # Extract indices
    var result = List[Int]()
    for i in range(numel):
        result.append(pairs[i][1])

    return result^
