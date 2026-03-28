"""Tensor creation / factory functions for AnyTensor.

Provides: zeros, ones, full, empty, arange, eye, linspace, ones_like,
zeros_like, full_like, nan_tensor, inf_tensor, neg_inf_tensor, randn.

These were extracted from any_tensor.mojo to improve SRP compliance.
All functions create and return new AnyTensor instances.
"""

from collections import List
from math import sqrt, log, cos, sin
from utils.numerics import inf as numeric_inf, neg_inf as numeric_neg_inf
from random import random_float64, seed as random_seed
from .any_tensor import AnyTensor
from .tensor_constants import MAX_TENSOR_BYTES


fn zeros(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with zeros.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with zeros.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = zeros([3, 4], DType.float32)
            # Creates a 3x4 tensor of float32 zeros.
    ```

    Performance:
        O(n) time where n is the number of elements.
    """
    var tensor = AnyTensor(shape, dtype)
    tensor._fill_zero()  # Efficiently zero out all bytes
    return tensor^


fn ones(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with ones.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with ones.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = ones([3, 4], DType.float32)
            # Creates a 3x4 tensor of float32 ones.
    ```
    """
    var tensor = AnyTensor(shape, dtype)

    # Fill with ones based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        tensor._fill_value_float(1.0)
    else:
        tensor._fill_value_int(1)

    return tensor^


fn full(shape: List[Int], fill_value: Float64, dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with a specific value.

    Args:
            shape: The shape of the output tensor.
            fill_value: The value to fill the tensor with.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with fill_value.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
            ```var t = full([3, 4], 42.0, DType.float32)
            # Creates a 3x4 tensor filled with 42.0
            ```
    """
    var tensor = AnyTensor(shape, dtype)

    # Fill with value based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        tensor._fill_value_float(fill_value)
    else:
        tensor._fill_value_int(Int(fill_value))

    return tensor^


fn empty(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create an uninitialized tensor (fast allocation).

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor with uninitialized memory.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Warning:
            The tensor contains uninitialized memory. Values are undefined until written.
            Use this for performance when you will immediately write to all elements.

    Examples:
    ```
            var t = empty([3, 4], DType.float32)
            # Creates a 3x4 tensor with undefined values.
    ```
    """
    # Just allocate without initialization
    var tensor = AnyTensor(shape, dtype)
    return tensor^


fn arange(
    start: Float64, stop: Float64, step: Float64, dtype: DType
) raises -> AnyTensor:
    """Create 1D tensor with evenly spaced values.

    Args:
            start: Start value (inclusive).
            stop: End value (exclusive).
            step: Spacing between values.
            dtype: The data type of tensor elements.

    Returns:
            A new 1D AnyTensor with values in range [start, stop) with given step.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
        ```
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        # Creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        var t2 = arange(0.0, 10.0, 2.0, DType.int32)
        # Creates [0, 2, 4, 6, 8]
        ```
    """
    # Calculate number of elements
    var num_elements = Int((stop - start) / step)
    var shape = List[Int]()
    shape.append(num_elements)

    var tensor = AnyTensor(shape, dtype)

    # Fill with sequence
    var value = start
    for i in range(num_elements):
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(i, value)
        else:
            tensor._set_int64(i, Int(value))
        value += step

    return tensor^


fn eye(n: Int, m: Int, k: Int, dtype: DType) raises -> AnyTensor:
    """Create 2D tensor with ones on diagonal.

    Args:
            n: Number of rows.
            m: Number of columns.
            k: Diagonal offset (0 for main diagonal, >0 for upper, <0 for lower).
            dtype: The data type of tensor elements.

    Returns:
            A new 2D AnyTensor with ones on the k-th diagonal.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = eye(3, 3, 0, DType.float32)
            # Creates 3x3 identity matrix.

            var t2 = eye(3, 4, 1, DType.float32)
            # Creates 3x4 matrix with ones on diagonal above main.
    ```
    """
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)

    var tensor = AnyTensor(shape, dtype)
    tensor._fill_zero()

    # Set diagonal to one
    for i in range(n):
        var j = i + k
        if j >= 0 and j < m:
            var index = i * m + j
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(index, 1.0)
            else:
                tensor._set_int64(index, 1)

    return tensor^


fn linspace(
    start: Float64, stop: Float64, num: Int, dtype: DType
) raises -> AnyTensor:
    """Create 1D tensor with evenly spaced values (inclusive).

    Args:
            start: Start value (inclusive).
            stop: End value (inclusive).
            num: Number of values.
            dtype: The data type of tensor elements.

    Returns:
            A new 1D AnyTensor with num evenly spaced values.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
        ```var t = linspace(0.0, 10.0, 11, DType.float32)
        # Creates [0.0, 1.0, 2.0, ..., 10.0]

        var t2 = linspace(0.0, 1.0, 5, DType.float64)
        # Creates [0.0, 0.25, 0.5, 0.75, 1.0]
        ```
    """
    var shape = List[Int]()
    shape.append(num)

    var tensor = AnyTensor(shape, dtype)

    if num == 1:
        # Special case: single value
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(0, start)
        else:
            tensor._set_int64(0, Int(start))
    else:
        # Calculate step size
        var step = (stop - start) / (num - 1)

        # Fill with sequence
        for i in range(num):
            var value = start + step * i
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(i, value)
            else:
                tensor._set_int64(i, Int(value))

    return tensor^


fn ones_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of ones with same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.

    Returns:
            A new AnyTensor filled with ones, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = zeros([3, 4], DType.float32)
        var y = ones_like(x)  # (3, 4) tensor of ones, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return ones(shape, dtype)


fn zeros_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of zeros with same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.

    Returns:
            A new AnyTensor filled with zeros, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = zeros_like(x)  # (3, 4) tensor of zeros, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return zeros(shape, dtype)


fn full_like(tensor: AnyTensor, fill_value: Float64) raises -> AnyTensor:
    """Create tensor filled with a value, same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.
            fill_value: Value to fill the tensor with.

    Returns:
            A new AnyTensor filled with fill_value, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = full_like(x, 3.14)  # (3, 4) tensor of 3.14, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return full(shape, fill_value, dtype)


fn nan_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with NaN values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with NaN values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = nan_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with NaN
        ```
    """
    # Only floating-point types support NaN
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("nan_tensor: only floating-point dtypes support NaN")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with NaN values
    # IEEE 754 NaN is represented as 0.0 / 0.0
    var nan_value = 0.0 / 0.0

    for i in range(tensor.numel()):
        tensor._set_float64(i, nan_value)

    return tensor^


fn inf_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with positive infinity values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with positive infinity values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = inf_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with +inf
        ```
    """
    # Only floating-point types support Inf
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("inf_tensor: only floating-point dtypes support Inf")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with +inf values using proper IEEE 754 infinity constant
    var inf_value: Float64 = numeric_inf[DType.float64]()

    for i in range(tensor.numel()):
        tensor._set_float64(i, inf_value)

    return tensor^


fn neg_inf_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with negative infinity values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with negative infinity values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = neg_inf_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with -inf
        ```
    """
    # Only floating-point types support Inf
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("neg_inf_tensor: only floating-point dtypes support Inf")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with -inf values using proper IEEE 754 infinity constant
    var neg_inf_value: Float64 = numeric_neg_inf[DType.float64]()

    for i in range(tensor.numel()):
        tensor._set_float64(i, neg_inf_value)

    return tensor^


fn _dtype_to_string(dtype: DType) -> String:
    """Convert a DType to a readable string representation.

    Args:
        dtype: The data type.

    Returns:
        A string like "float32", "int64", etc.
    """
    if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"


fn randn(shape: List[Int], dtype: DType, seed: Int = 0) raises -> AnyTensor:
    """Create tensor filled with random values from standard normal distribution.

        Uses Box-Muller transform to generate normally distributed random values
        from uniform random values. Generates values with mean=0 and std=1.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (should be floating-point).
            seed: Random seed for reproducibility (default: 0 uses system randomness).

    Returns:
            A new AnyTensor filled with random values from N(0, 1).

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
            ```var t = randn([3, 4], DType.float32)
            # Creates 3x4 tensor with values from N(0, 1)

            var t2 = randn([100, 100], DType.float32, seed=42)
            # Reproducible random tensor with seed=42```

    Note:
            For integer dtypes, values are generated as floats then truncated.
            Box-Muller transform generates pairs of independent normal values.
    """
    # Verify floating-point dtype (best practice)
    if not (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        print(
            "Warning: randn() is designed for floating-point types, got",
            _dtype_to_string(dtype),
        )

    # Set random seed if provided (0 uses system randomness)
    if seed > 0:
        random_seed(seed)

    var tensor = AnyTensor(shape, dtype)

    # Box-Muller transform: generates pairs of independent N(0,1) values
    # from pairs of uniform random values
    var i = 0
    while i < tensor.numel():
        # Generate two uniform random values in (0, 1]
        var u1 = random_float64()
        var u2 = random_float64()

        # Ensure u1 is not zero (would cause log(0))
        if u1 < 1e-10:
            u1 = 1e-10

        # Box-Muller transform
        var magnitude = sqrt(-2.0 * log(u1))
        var angle = 2.0 * 3.14159265358979323846 * u2

        # Generate two independent normal values
        var z0 = magnitude * cos(angle)
        var z1 = magnitude * sin(angle)

        # Store first value
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(i, z0)
        else:
            tensor._set_int64(i, Int(z0))

        i += 1

        # Store second value if there's room
        if i < tensor.numel():
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(i, z1)
            else:
                tensor._set_int64(i, Int(z1))
            i += 1

    return tensor^
