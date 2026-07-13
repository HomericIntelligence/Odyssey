"""Tensor split operations for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
Cross-module private-field access (_shape) is valid in Mojo (package-scoped privacy).
See tensor_io.mojo for precedent.
"""

from std.collections import List
from .any_tensor import AnyTensor


def split_impl(
    tensor: AnyTensor, num_splits: Int, axis: Int = 0
) raises -> List[AnyTensor]:
    """Split tensor into equal-sized parts along an axis.

    Args:
        tensor: The tensor to split.
        num_splits: Number of equal parts.
        axis: Axis along which to split (default: 0).

    Returns:
        List of AnyTensor objects.

    Raises:
        Error: If axis is invalid, num_splits <= 0, or not evenly divisible.
    """
    if num_splits <= 0:
        raise Error(
            "split: num_splits must be positive, got " + String(num_splits)
        )
    if axis < 0 or axis >= len(tensor._shape):
        raise Error(
            "split: axis "
            + String(axis)
            + " out of range for "
            + String(len(tensor._shape))
            + "-D tensor"
        )
    var dim_size = tensor._shape[axis]
    if dim_size % num_splits != 0:
        raise Error(
            "split: dimension "
            + String(dim_size)
            + " not divisible by "
            + String(num_splits)
        )
    var chunk_size = dim_size // num_splits
    var parts = List[AnyTensor]()
    for i in range(num_splits):
        var start = i * chunk_size
        var end = start + chunk_size
        # slice returns a view; clone to get independent memory
        var part = tensor.slice(start, end, axis).clone()
        parts.append(part^)
    return parts^


def split_with_indices_impl(
    tensor: AnyTensor, split_indices: List[Int], axis: Int = 0
) raises -> List[AnyTensor]:
    """Split tensor at specified indices along an axis.

    Args:
        tensor: The tensor to split.
        split_indices: List of split indices.
        axis: Axis along which to split (default: 0).

    Returns:
        List of AnyTensor objects.

    Raises:
        Error: If axis is invalid or indices are out of bounds/unordered.
    """
    if axis < 0 or axis >= len(tensor._shape):
        raise Error(
            "split_with_indices: axis "
            + String(axis)
            + " out of range for "
            + String(len(tensor._shape))
            + "-D tensor"
        )
    var dim_size = tensor._shape[axis]
    var parts = List[AnyTensor]()
    var prev = 0
    for i in range(len(split_indices)):
        var idx = split_indices[i]
        if idx < prev or idx > dim_size:
            raise Error(
                "split_with_indices: index "
                + String(idx)
                + " out of bounds or unordered"
            )
        if idx > prev:
            var part = tensor.slice(prev, idx, axis).clone()
            parts.append(part^)
        prev = idx
    # Final segment from last index to end
    if prev < dim_size:
        var part = tensor.slice(prev, dim_size, axis).clone()
        parts.append(part^)
    return parts^
