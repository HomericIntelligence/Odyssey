"""Test fixtures for AnyTensor testing.

Provides common tensor creation utilities for tests, including
random tensors, sequential tensors, and special value tensors.

These fixtures wrap the comprehensive infrastructure in projectodyssey.testing
with convenient test-specific APIs.
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.tensor.tensor_creation import (
    nan_tensor as shared_nan_tensor,
)
from projectodyssey.tensor.tensor_creation import (
    inf_tensor as shared_inf_tensor,
)
from projectodyssey.testing.data_generators import (
    random_tensor as shared_random_tensor,
)
from projectodyssey.testing.data_generators import random_uniform


def random_tensor(
    shape: List[Int], dtype: DType = DType.float32
) raises -> AnyTensor:
    """Create a tensor with random values from uniform distribution [0, 1).

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).

    Returns:
        AnyTensor with random values uniformly distributed in [0, 1).

    Example:
        ```mojo
        var weights = random_tensor([10, 5], DType.float32)
        # Creates 10x5 tensor with random values in [0, 1)
        ```

    Raises:
        Error: If operation fails.
    """
    return shared_random_tensor(shape, dtype)


def sequential_tensor(
    shape: List[Int], dtype: DType = DType.float32
) raises -> AnyTensor:
    """Create tensor with sequential values 0, 1, 2, 3, ...

    Tensor is filled with sequential values in row-major order, then reshaped
    to the requested shape.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).

    Returns:
        AnyTensor with values 0, 1, 2, ... in flattened order.

    Example:
        ```mojo
        var tensor = sequential_tensor([2, 3], DType.float32)
        # Returns tensor [[0, 1, 2], [3, 4, 5]]
        ```

    Raises:
        Error: If operation fails.
    """
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with sequential values
    for i in range(numel):
        tensor._set_float64(i, Float64(i))

    return tensor


def nan_tensor(shape: List[Int]) raises -> AnyTensor:
    """Create tensor filled with NaN values.

    Args:
        shape: Shape of the output tensor as a list of dimensions.

    Returns:
        AnyTensor with all elements set to NaN.

    Example:
        ```mojo
        var tensor = nan_tensor([3, 3])
        # Returns 3x3 tensor with all NaN values
        ```

    Note:
        Creates float32 tensors with NaN values. NaN is represented as
        0x7fc00000 in float32 bit representation.

    Raises:
        Error: If operation fails.
    """
    # Use shared implementation that safely creates NaN tensors
    return shared_nan_tensor(shape, DType.float32)


def inf_tensor(shape: List[Int]) raises -> AnyTensor:
    """Create tensor filled with infinity values.

    Args:
        shape: Shape of the output tensor as a list of dimensions.

    Returns:
        AnyTensor with all elements set to positive infinity.

    Example:
        ```mojo
        var tensor = inf_tensor([3, 3])
        # Returns 3x3 tensor with all infinity values
        ```

    Note:
        Creates float32 tensors with positive infinity values.
        Positive infinity is represented as 0x7f800000 in float32 bit representation.

    Raises:
        Error: If operation fails.
    """
    # Use shared implementation that safely creates inf tensors
    return shared_inf_tensor(shape, DType.float32)


def ones_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of ones matching input shape and dtype.

    Args:
        tensor: Template tensor to match shape and dtype from.

    Returns:
        AnyTensor of ones with same shape and dtype as input.

    Example:
        ```mojo
        var t1 = random_tensor([3, 4], DType.float32)
        var t2 = ones_like(t1)
        # t2 has shape [3, 4] and dtype float32, all values are 1.0
        ```

    Raises:
        Error: If operation fails.
    """
    return ones(tensor.shape(), tensor.dtype())


def zeros_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of zeros matching input shape and dtype.

    Args:
        tensor: Template tensor to match shape and dtype from.

    Returns:
        AnyTensor of zeros with same shape and dtype as input.

    Example:
        ```mojo
        var t1 = random_tensor([3, 4], DType.float32)
        var t2 = zeros_like(t1)
        # t2 has shape [3, 4] and dtype float32, all values are 0.0
        ```

    Raises:
        Error: If operation fails.
    """
    return zeros(tensor.shape(), tensor.dtype())
