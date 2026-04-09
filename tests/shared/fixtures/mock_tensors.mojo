"""Mock tensor utilities for testing.

This module provides utilities for creating test tensors with known patterns
and comparing tensors with appropriate tolerance for floating-point operations.

Key functions:
- create_random_tensor(): Generate random tensors with deterministic seed
- create_zeros_tensor(): Create zero-filled tensors
- create_ones_tensor(): Create one-filled tensors
- create_sequential_tensor(): Create tensors with sequential values
- assert_tensors_equal(): Compare tensors with epsilon tolerance
- assert_shape_equal(): Validate tensor shapes

All functions use deterministic seeds for reproducible tests.
"""

from std.random import seed, randn


# ============================================================================
# Tensor Creation Utilities
# ============================================================================


def create_random_tensor(
    shape: List[Int], random_seed: Int = 42
) -> List[Float32]:
    """Create random tensor with deterministic seed.

    Generates a flat list of random Float32 values representing a tensor.
    Uses deterministic seeding for reproducible tests.

    Args:
        shape: Tensor dimensions (e.g., [2, 3, 4] for 2x3x4 tensor).
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        Flat list of random Float32 values.

    Example:
        ```mojo
        # Create 2x3 random tensor
        var tensor = create_random_tensor([2, 3], seed=42)
        # tensor contains 6 random values (2 * 3)
        ```

    Note:
        The tensor is returned as a flat list. Shape information must be
        tracked separately (e.g., in a Tensor struct wrapper).
    """
    # Set deterministic seed
    seed(random_seed)

    # Calculate total size
    var total_size = 1
    for dim in shape:
        total_size *= dim[]

    # Generate random values
    var data = List[Float32]()
    for i in range(total_size):
        # randn() generates standard normal distribution
        var value = Float32(randn())
        data.append(value)

    return data


def create_zeros_tensor(shape: List[Int]) -> List[Float32]:
    """Create zero-filled tensor.

    Args:
        shape: Tensor dimensions.

    Returns:
        Flat list of zeros.

    Example:
        ```mojo
        var zeros = create_zeros_tensor([3, 3])
        # zeros contains [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ```
    """
    var total_size = 1
    for dim in shape:
        total_size *= dim[]

    var data = List[Float32]()
    for _ in range(total_size):
        data.append(0.0)

    return data


def create_ones_tensor(shape: List[Int]) -> List[Float32]:
    """Create one-filled tensor.

    Args:
        shape: Tensor dimensions.

    Returns:
        Flat list of ones.

    Example:
        ```mojo
        var ones = create_ones_tensor([2, 2])
        # ones contains [1.0, 1.0, 1.0, 1.0]
        ```
    """
    var total_size = 1
    for dim in shape:
        total_size *= dim[]

    var data = List[Float32]()
    for _ in range(total_size):
        data.append(1.0)

    return data


def create_sequential_tensor(
    shape: List[Int], start: Float32 = 0.0
) -> List[Float32]:
    """Create tensor with sequential values [start, start+1, start+2, ...].

    Args:
        shape: Tensor dimensions.
        start: Starting value (default: 0.0).

    Returns:
        Flat list of sequential values.

    Example:
        ```mojo
        var seq = create_sequential_tensor([2, 3], start=1.0)
        # seq contains [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ```

    Note:
        Useful for testing indexing and reshape operations where
        you need to track which values end up where.
    """
    var total_size = 1
    for dim in shape:
        total_size *= dim[]

    var data = List[Float32]()
    for i in range(total_size):
        data.append(start + Float32(i))

    return data


def create_constant_tensor(shape: List[Int], value: Float32) -> List[Float32]:
    """Create tensor filled with constant value.

    Args:
        shape: Tensor dimensions.
        value: Fill value.

    Returns:
        Flat list filled with constant value.

    Example:
        ```mojo
        var fives = create_constant_tensor([2, 2], 5.0)
        # fives contains [5.0, 5.0, 5.0, 5.0]
        ```
    """
    var total_size = 1
    for dim in shape:
        total_size *= dim[]

    var data = List[Float32]()
    for _ in range(total_size):
        data.append(value)

    return data


# ============================================================================
# Tensor Comparison Utilities
# ============================================================================


def assert_tensors_equal(
    a: List[Float32],
    b: List[Float32],
    epsilon: Float64 = 1e-6,
    message: String = "",
) raises:
    """Assert two tensors are approximately equal.

    Compares tensors element-wise with epsilon tolerance for floating-point
    precision. Tensors must have the same size.

    Args:
        a: First tensor (flat list).
        b: Second tensor (flat list).
        epsilon: Maximum allowed element-wise difference (default: 1e-6).
        message: Optional error message prefix.

    Raises:
        Error if tensors have different sizes or any element differs by > epsilon.

    Example:
        ```mojo
        var a = create_ones_tensor([2, 2])
        var b = create_constant_tensor([2, 2], 1.0000001)
        assert_tensors_equal(a, b, epsilon=1e-5)  # Passes
        ```

    Note:
        For exact equality (e.g., integer tensors), use epsilon=0.0
    """
    # Check sizes match
    if len(a) != len(b):
        var error_msg = message if message else "Tensor sizes don't match"
        error_msg = (
            error_msg
            + " (a: "
            + String(len(a))
            + ", b: "
            + String(len(b))
            + ")"
        )
        raise Error(error_msg)

    # Compare elements
    for i in range(len(a)):
        var diff = abs(a[i] - b[i])
        if Float64(diff) > epsilon:
            var error_msg = message if message else "Tensors not equal"
            error_msg = (
                error_msg
                + " at index "
                + String(i)
                + " (a: "
                + String(a[i])
                + ", b: "
                + String(b[i])
                + ", diff: "
                + String(diff)
                + ")"
            )
            raise Error(error_msg)


def assert_shape_equal(
    actual_shape: List[Int], expected_shape: List[Int], message: String = ""
) raises:
    """Assert tensor shape matches expected shape.

    Args:
        actual_shape: Actual tensor shape.
        expected_shape: Expected tensor shape.
        message: Optional error message prefix.

    Raises:
        Error if shapes don't match.

    Example:
        ```mojo
        var shape : List[Int] = [2, 3, 4]
        var expected : List[Int] = [2, 3, 4]
        assert_shape_equal(shape, expected)  # Passes
        ```
    """
    # Check dimensions match
    if len(actual_shape) != len(expected_shape):
        var error_msg = message if message else "Shape dimensions don't match"
        error_msg = (
            error_msg
            + " (actual: "
            + String(len(actual_shape))
            + ", expected: "
            + String(len(expected_shape))
            + ")"
        )
        raise Error(error_msg)

    # Compare each dimension
    for i in range(len(actual_shape)):
        if actual_shape[i] != expected_shape[i]:
            var error_msg = message if message else "Shape mismatch"
            error_msg = (
                error_msg
                + " at dimension "
                + String(i)
                + " (actual: "
                + String(actual_shape[i])
                + ", expected: "
                + String(expected_shape[i])
                + ")"
            )
            raise Error(error_msg)


def calculate_tensor_size(shape: List[Int]) -> Int:
    """Calculate total tensor size from shape.

    Args:
        shape: Tensor dimensions.

    Returns:
        Total number of elements.

    Example:
        ```mojo
        var size = calculate_tensor_size([2, 3, 4])
        # size = 24 (2 * 3 * 4)
        ```
    """
    var total = 1
    for dim in shape:
        total *= dim[]
    return total


# ============================================================================
# Tensor Statistics Utilities (for validation)
# ============================================================================


def tensor_mean(data: List[Float32]) -> Float32:
    """Calculate tensor mean.

    Args:
        data: Flat tensor data.

    Returns:
        Mean value.

    Example:
        ```mojo
        var data = create_sequential_tensor([2, 2], start=0.0)
        var mean = tensor_mean(data)  # (0+1+2+3)/4 = 1.5
        ```
    """
    if len(data) == 0:
        return 0.0

    var sum = Float32(0.0)
    for i in range(len(data)):
        sum += data[i]

    return sum / Float32(len(data))


def tensor_min(data: List[Float32]) -> Float32:
    """Find minimum tensor value.

    Args:
        data: Flat tensor data.

    Returns:
        Minimum value.

    Example:
        ```mojo
        var data = create_sequential_tensor([2, 2], start=1.0)
        var min_val = tensor_min(data)  # 1.0
        ```
    """
    if len(data) == 0:
        return 0.0

    var min_val = data[0]
    for i in range(1, len(data)):
        if data[i] < min_val:
            min_val = data[i]

    return min_val


def tensor_max(data: List[Float32]) -> Float32:
    """Find maximum tensor value.

    Args:
        data: Flat tensor data.

    Returns:
        Maximum value.

    Example:
        ```mojo
        var data = create_sequential_tensor([2, 2], start=1.0)
        var max_val = tensor_max(data)  # 4.0
        ```
    """
    if len(data) == 0:
        return 0.0

    var max_val = data[0]
    for i in range(1, len(data)):
        if data[i] > max_val:
            max_val = data[i]

    return max_val
