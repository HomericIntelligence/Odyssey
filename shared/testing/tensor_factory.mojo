"""Unified Tensor Factory for Test Tensor Creation

Provides high-level factory functions for creating test tensors with consistent
dtype handling and value initialization patterns.

This module consolidates tensor creation utilities to:
- Reduce boilerplate in tests
- Ensure consistent value type handling across dtypes
- Provide convenient initialization patterns (zeros, ones, random, etc.)
- Support both fixed seeds and random generation

Example:
    from shared.testing import tensor_factory
    from shared.core import AnyTensor

    # Create tensors with unified factory
    var zeros = tensor_factory.zeros_tensor([10, 5], DType.float32)
    var ones = tensor_factory.ones_tensor([10, 5], DType.float32)
    var random = tensor_factory.random_tensor([10, 5], DType.float32, 0.0, 1.0)
    var normal = tensor_factory.random_normal_tensor(
        [10, 5], DType.float32, mean=0.0, std=1.0
    )

    # Set specific values with automatic dtype conversion
    tensor_factory.set_tensor_value(random, 0, 42.0, DType.float32)
"""

from random import random_float64
from math import sqrt, log, cos, sin, pi
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full


# ============================================================================
# Tensor Creation Factories
# ============================================================================


fn zeros_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements.

    Returns:
        AnyTensor with all elements initialized to zero.

    Example:
    ```
        var weights = zeros_tensor([10, 5], DType.float32)
        # Creates 10x5 tensor filled with zeros
    ```

    Raises:
        Error: If operation fails.
    """
    return zeros(shape, dtype)


fn ones_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with ones.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements.

    Returns:
        AnyTensor with all elements initialized to one.

    Example:
    ```
        var weights = ones_tensor([10, 5], DType.float32)
        # Creates 10x5 tensor filled with ones
    ```

    Raises:
        Error: If operation fails.
    """
    return ones(shape, dtype)


fn full_tensor(
    shape: List[Int], fill_value: Float64, dtype: DType
) raises -> AnyTensor:
    """Create a tensor filled with a specific value.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        fill_value: Value to fill the tensor with (as Float64).
        dtype: Data type of tensor elements.

    Returns:
        AnyTensor with all elements initialized to fill_value (converted to dtype).

    Example:
    ```
        var weights = full_tensor([10, 5], 3.14, DType.float32)
        # Creates 10x5 tensor filled with 3.14
    ```

    Raises:
        Error: If operation fails.
    """
    return full(shape, fill_value, dtype)


fn random_tensor(
    shape: List[Int],
    dtype: DType = DType.float32,
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    seed: Int = -1,
) raises -> AnyTensor:
    """Create a tensor with random values from uniform distribution [low, high).

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).
        low: Lower bound of uniform distribution (inclusive, default: 0.0).
        high: Upper bound of uniform distribution (exclusive, default: 1.0).
        seed: Random seed for reproducibility (default: -1 for random seed).
              Note: Current implementation does not use seed parameter.

    Returns:
        AnyTensor with random values uniformly distributed in [low, high).

    Example:
        var weights = random_tensor(
            [10, 5], DType.float32, low=-1.0, high=1.0
        )
        # Creates 10x5 tensor with random values in [-1.0, 1.0)

    Note:
        Values are uniformly distributed in [low, high) regardless of dtype.
        For integer dtypes, values are truncated to Int.

    Raises:
        Error: If operation fails.
    """
    # Create empty tensor with the specified shape
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Calculate value range
    var value_range = high - low

    # Fill with random values
    for i in range(numel):
        var rand_val = random_float64()
        var scaled_val = low + rand_val * value_range

        # Convert to appropriate dtype
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            tensor._set_float64(i, scaled_val)
        elif (
            dtype == DType.int8
            or dtype == DType.int16
            or dtype == DType.int32
            or dtype == DType.int64
        ):
            tensor._set_int64(i, Int(scaled_val))
        elif (
            dtype == DType.uint8
            or dtype == DType.uint16
            or dtype == DType.uint32
            or dtype == DType.uint64
        ):
            tensor._set_int64(i, Int(scaled_val))

    return tensor^


fn random_normal_tensor(
    shape: List[Int],
    dtype: DType = DType.float32,
    mean: Float64 = 0.0,
    std: Float64 = 1.0,
    seed: Int = -1,
) raises -> AnyTensor:
    """Create a tensor with random values from normal distribution N(mean, std^2).

    Uses Box-Muller transform to convert uniform random values to normal distribution.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).
        mean: Mean of the normal distribution (default: 0.0).
        std: Standard deviation of the normal distribution (default: 1.0).
        seed: Random seed for reproducibility (default: -1 for random seed).
              Note: Current implementation does not use seed parameter.

    Returns:
        AnyTensor with random values from normal distribution N(mean, std^2).

    Example:
        var weights = random_normal_tensor(
            [784, 256], DType.float32, mean=0.0, std=0.01
        )
        # Creates 784x256 tensor with normally distributed values

    Note:
        Uses Box-Muller transform for efficiency.
        For integer dtypes, values are truncated to Int after sampling.

    Raises:
        Error: If operation fails.
    """
    # Create empty tensor with the specified shape
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Generate pairs of normal values using Box-Muller transform
    var i = 0
    while i < numel:
        # Generate two uniform random values
        var u1 = random_float64()
        var u2 = random_float64()

        # Avoid log(0)
        if u1 < 1e-10:
            u1 = 1e-10

        # Box-Muller transform
        var r = sqrt(-2.0 * log(u1))
        var theta = 2.0 * pi * u2

        # Convert to appropriate dtype
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            # First value
            var z1 = r * cos(theta)
            tensor._set_float64(i, mean + z1 * std)
            i += 1

            # Second value (if there's room)
            if i < numel:
                var z2 = r * sin(theta)
                tensor._set_float64(i, mean + z2 * std)
                i += 1
        else:
            # For integer dtypes, still use Box-Muller but truncate
            var z1 = r * cos(theta)
            tensor._set_int64(i, Int(mean + z1 * std))
            i += 1

            if i < numel:
                var z2 = r * sin(theta)
                tensor._set_int64(i, Int(mean + z2 * std))
                i += 1

    return tensor^


# ============================================================================
# Tensor Value Setting Helper
# ============================================================================


fn set_tensor_value(
    mut tensor: AnyTensor, index: Int, value: Float64, dtype: DType
) raises:
    """Set a single tensor element with automatic dtype conversion.

    This helper function provides consistent value setting across different dtypes,
    handling the necessary type conversions internally.

    Args:
        tensor: The tensor to modify (must be mutable).
        index: Linear index of the element to set (flattened array indexing).
        value: The value to set (as Float64).
        dtype: The dtype of the tensor (for correct type conversion).

    Raises:
        Error if index is out of bounds or dtype is unsupported.

    Example:
        var tensor = zeros_tensor([10], DType.float32)
        set_tensor_value(tensor, 0, 3.14, DType.float32)
        set_tensor_value(tensor, 5, 2.71, DType.float32)

    Note:
        Integer values will be truncated/cast from the Float64 input.
        The function works with both flat and multi-dimensional tensors.
    """
    # Convert to appropriate dtype
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        tensor._set_float64(index, value)
    elif (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        tensor._set_int64(index, Int(value))
    elif (
        dtype == DType.uint8
        or dtype == DType.uint16
        or dtype == DType.uint32
        or dtype == DType.uint64
    ):
        tensor._set_int64(index, Int(value))
