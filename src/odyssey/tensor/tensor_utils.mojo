"""Tensor utility functions for AnyTensor.

Provides convenience wrappers: copy, clone, item, diff, calculate_max_batch_size.

These were extracted from any_tensor.mojo to improve SRP compliance.
Each function delegates to the corresponding AnyTensor method.
"""

from std.collections import List
from odyssey.tensor.any_tensor import AnyTensor


def calculate_max_batch_size(
    sample_shape: List[Int],
    dtype: DType,
    max_memory_bytes: Int = 500_000_000,  # 500 MB default
) raises -> Int:
    """Calculate maximum safe batch size for given sample shape.

    Args:
            sample_shape: Shape of a single sample (e.g., [1, 28, 28] for MNIST).
            dtype: Data type of the tensor.
            max_memory_bytes: Maximum memory to use for a batch (default: 500 MB).

    Returns:
            Maximum batch size that fits in memory.

    Raises:
            Error: If sample shape is invalid or no batch size can fit in memory.

    Example:
            ```mojo
            # For MNIST: (1, 28, 28) images
            var sample_shape = List[Int]()
            sample_shape.append(1)
            sample_shape.append(28)
            sample_shape.append(28)
            var max_batch = calculate_max_batch_size(sample_shape, DType.float32)
            print("Max batch size:", max_batch)  # ~640,000 samples
            ```
    """
    var sample_elements = 1
    for i in range(len(sample_shape)):
        sample_elements *= sample_shape[i]

    var dtype_size = AnyTensor._get_dtype_size_static(dtype)
    var bytes_per_sample = sample_elements * dtype_size

    if bytes_per_sample <= 0:
        raise Error("Invalid sample shape or dtype")

    var max_batch = max_memory_bytes // bytes_per_sample

    if max_batch < 1:
        raise Error(
            "Single sample ("
            + String(bytes_per_sample)
            + " bytes) exceeds memory limit ("
            + String(max_memory_bytes)
            + " bytes)"
        )

    return max_batch


# ============================================================================
# Utility Function Wrappers
# ============================================================================


def copy(tensor: AnyTensor) raises -> AnyTensor:
    """Create an independent deep copy of the tensor.

    This is a convenience wrapper around the AnyTensor.clone() method,
    following NumPy naming conventions. The returned tensor has its own
    independent memory; modifications to it do not affect the original.

    Args:
        tensor: The tensor to copy.

    Returns:
        A new AnyTensor that is a deep copy of the input.

    Raises:
        Error: If memory allocation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = copy(x)  # Independent deep copy
        ```
    """
    return tensor.clone()


def clone(tensor: AnyTensor) raises -> AnyTensor:
    """Create a clone of the tensor.

    This is a convenience wrapper around the AnyTensor.clone() method.

    Args:
        tensor: The tensor to clone.

    Returns:
        A new AnyTensor that is a deep copy of the input.

    Raises:
        Error: If memory allocation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = clone(x)  # Independent copy
        ```
    """
    return tensor.clone()


def item(tensor: AnyTensor) raises -> Float64:
    """Extract the value from a single-element tensor.

    This is a convenience wrapper around the AnyTensor.item() method.

    Args:
        tensor: A tensor with exactly one element.

    Returns:
        The scalar value as Float64.

    Raises:
        Error: If tensor has more than one element.

    Example:
        ```mojo
        var x = full([], 42.0, DType.float32)
        var val = item(x)  # Returns 42.0
        ```
    """
    return tensor.item()


def diff(tensor: AnyTensor, n: Int = 1) raises -> AnyTensor:
    """Calculate consecutive differences along an axis.

    This is a convenience wrapper around the AnyTensor.diff() method.

    Args:
        tensor: The input tensor.
        n: Order of differences (default: 1).

    Returns:
        A new AnyTensor with differences computed.

    Raises:
        Error: If operation fails.

    Example:
        ```mojo
        var x = arange(0.0, 5.0, 1.0, DType.float32)
        var d = diff(x)  # [1.0, 1.0, 1.0, 1.0]
        ```
    """
    return tensor.diff(n)


def tolist(tensor: AnyTensor) raises -> List[Float64]:
    """Convert tensor to a flat list of Float64 values.

    This is a convenience wrapper around the AnyTensor.tolist() method.

    Args:
        tensor: The tensor to convert.

    Returns:
        A flat list containing all tensor values as Float64.

    Example:
        ```mojo
        var x = arange(0.0, 5.0, 1.0, DType.float32)
        var lst = tolist(x)  # [0.0, 1.0, 2.0, 3.0, 4.0]
        ```
    """
    return tensor.tolist()


def contiguous(tensor: AnyTensor) raises -> AnyTensor:
    """Return a contiguous copy of the tensor.

    This is a convenience wrapper around the AnyTensor.contiguous() method.
    If the tensor is already contiguous, returns a clone.
    Otherwise, creates a new contiguous tensor with the same data.

    Args:
        tensor: The tensor to make contiguous.

    Returns:
        A contiguous AnyTensor with the same shape, dtype, and values.

    Raises:
        Error: If memory allocation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var c = contiguous(x)  # Already contiguous, returns clone
        ```
    """
    return tensor.contiguous()
