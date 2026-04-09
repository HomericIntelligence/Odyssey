"""Special FP- Test Values

Provides test values that are exactly representable in all float dtypes:
- FP4, FP8, FP16, FP32, BF16

These values are chosen for:
1. Exact representation (no rounding errors)
2. Simple arithmetic properties
3. Predictable behavior across dtypes

Special Values (Positive):
- 0.0: Zero (identity for addition)
- 0.5: Half (simple fraction, 2^-1)
- 1.0: One (identity for multiplication)
- 1.5: One and a half (1 + 2^-1)

Special Values (Negative):
- -1.0: Negative one (for ReLU gradient testing)
- -0.5: Negative half (for ReLU gradient testing)

All 6 values have exact binary representations in IEEE 754 formats.
For Int8 quantization, positive values map to: 0, 0, 1, 2 respectively.

Numerical Edge Cases:
- NaN: Created via 0.0/0.0, for testing gradient overflow/underflow
- Inf: Created via 1.0/0.0 or -1.0/0.0, for testing numerical stability

Usage:
    from shared.testing.special_values import (
        create_special_value_tensor,
        create_seeded_random_tensor,
        create_nan_tensor,
        create_inf_tensor,
        SPECIAL_VALUE_ZERO,
        SPECIAL_VALUE_ONE,
        SPECIAL_VALUE_NEG_ONE
    )

    # Create tensor filled with ones
    var ones = create_special_value_tensor([3, 3], DType.float32, SPECIAL_VALUE_ONE)

    # Create tensor with alternating pattern
    var pattern = create_alternating_pattern_tensor([4, 4], DType.float16)

    # Create seeded random tensor for reproducible gradient checking
    var random = create_seeded_random_tensor([3, 3], DType.float32, 42, -1.0, 1.0)

    # Create NaN tensor for testing gradient overflow detection
    var nan_tensor = create_nan_tensor([3, 3], DType.float32)

    # Create Inf tensors for testing numerical stability
    var pos_inf = create_inf_tensor([3, 3], DType.float32, positive=True)
    var neg_inf = create_inf_tensor([3, 3], DType.float32, positive=False)
"""

from shared.tensor.any_tensor import AnyTensor
from shared.testing.tensor_factory import zeros
from random import seed as random_seed, random_float64


# ============================================================================
# Special Test Value Constants
# ============================================================================

comptime SPECIAL_VALUE_ZERO: Float64 = 0.0
comptime SPECIAL_VALUE_HALF: Float64 = 0.5
comptime SPECIAL_VALUE_ONE: Float64 = 1.0
comptime SPECIAL_VALUE_ONE_HALF: Float64 = 1.5
comptime SPECIAL_VALUE_NEG_HALF: Float64 = -0.5
comptime SPECIAL_VALUE_NEG_ONE: Float64 = -1.0


# ============================================================================
# Tensor Creation Functions
# ============================================================================


fn create_special_value_tensor(
    shape: List[Int], dtype: DType, value: Float64 = SPECIAL_VALUE_ONE
) raises -> AnyTensor:
    """Create tensor filled with a special value.

    Args:
        shape: Tensor dimensions.
        dtype: Data type (FP4, FP8, FP16, FP32, BF16, or Int8).
        value: Special value to fill (must be -1.0, -0.5, 0.0, 0.5, 1.0, or 1.5).

    Returns:
        AnyTensor filled with the special value.

    Raises:
        Error if value is not a special value (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5).

    Example:
        ```mojo
        # Create 3x3 tensor filled with ones (FP32)
        var ones = create_special_value_tensor([3, 3], DType.float32, 1.0)

        # Create 2x2 tensor filled with halves (FP16)
        var halves = create_special_value_tensor([2, 2], DType.float16, 0.5)

        # Create 4x4 tensor filled with negative ones (for ReLU testing)
        var neg_ones = create_special_value_tensor([4, 4], DType.float32, -1.0)

        # Create 2x2 tensor filled with zeros (BF16)
        var zeros_bf16 = create_special_value_tensor([2, 2], DType.bfloat16, 0.0)
        ```
    """
    # Validate value is special
    if (
        value != -1.0
        and value != -0.5
        and value != 0.0
        and value != 0.5
        and value != 1.0
        and value != 1.5
    ):
        raise Error(
            "Value must be a special value: -1.0, -0.5, 0.0, 0.5, 1.0, or 1.5."
            " Got: "
            + String(value)
        )

    # Create zero-initialized tensor
    var tensor = zeros(shape, dtype)

    # If value is zero, we're done
    if value == 0.0:
        return tensor^

    # Fill with special value
    var numel = 1
    for dim in shape:
        numel *= dim

    # Set all elements to the special value
    for i in range(numel):
        tensor._set_float64(i, value)

    return tensor^


fn create_alternating_pattern_tensor(
    shape: List[Int], dtype: DType
) raises -> AnyTensor:
    """Create tensor with alternating special values pattern.

    Pattern repeats: -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.0, -0.5, 0.0, ...

    Useful for testing layer behavior with varied inputs including negative values.
    All values are exactly representable across all dtypes.

    Args:
        shape: Tensor dimensions.
        dtype: Data type (FP4, FP8, FP16, FP32, BF16, or Int8).

    Returns:
        AnyTensor with alternating special value pattern.

    Example:
        ```mojo
        # Create 3x2 tensor with alternating pattern
        var pattern = create_alternating_pattern_tensor([3, 2], DType.float32)
        # pattern[0] = -1.0, pattern[1] = -0.5, pattern[2] = 0.0
        # pattern[3] = 0.5, pattern[4] = 1.0, pattern[5] = 1.5
        ```

    Raises:
        Error: If operation fails.
    """
    # Create zero-initialized tensor
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Special values in alternating pattern (6 values for positive/negative coverage)
    var values = List[Float64]()
    values.append(SPECIAL_VALUE_NEG_ONE)
    values.append(SPECIAL_VALUE_NEG_HALF)
    values.append(SPECIAL_VALUE_ZERO)
    values.append(SPECIAL_VALUE_HALF)
    values.append(SPECIAL_VALUE_ONE)
    values.append(SPECIAL_VALUE_ONE_HALF)

    # Fill with alternating pattern
    for i in range(numel):
        var val = values[i % 6]
        tensor._set_float64(i, val)

    return tensor^


fn verify_special_value_invariants(
    tensor: AnyTensor, expected_value: Float64
) raises:
    """Verify all elements match expected special value.

    This function performs exact equality checks (zero tolerance) because
    special values are exactly representable in all dtypes.

    Args:
        tensor: Tensor to check.
        expected_value: Expected special value (-1.0, -0.5, 0.0, 0.5, 1.0, or 1.5).

    Raises:
        Error if any element doesn't match expected value exactly.
        Error if expected_value is not a special value.

    Example:
        ```mojo
        var ones = create_special_value_tensor([3, 3], DType.float32, 1.0)
        verify_special_value_invariants(ones, 1.0)  # Passes

        var neg_ones = create_special_value_tensor([3, 3], DType.float32, -1.0)
        verify_special_value_invariants(neg_ones, -1.0)  # Passes

        var mixed = create_alternating_pattern_tensor([2, 2], DType.float16)
        verify_special_value_invariants(mixed, 1.0)  # Fails - not all 1.0
        ```
    """
    # Validate expected_value is special
    if (
        expected_value != -1.0
        and expected_value != -0.5
        and expected_value != 0.0
        and expected_value != 0.5
        and expected_value != 1.0
        and expected_value != 1.5
    ):
        raise Error(
            "Expected value must be a special value: -1.0, -0.5, 0.0, 0.5, 1.0,"
            " or 1.5. Got: "
            + String(expected_value)
        )

    var numel = tensor.numel()
    for i in range(numel):
        var actual = tensor._get_float64(i)
        if actual != expected_value:
            raise Error(
                "Element "
                + String(i)
                + " = "
                + String(actual)
                + ", expected "
                + String(expected_value)
                + " (exact match required for special values)"
            )


fn create_seeded_random_tensor(
    shape: List[Int],
    dtype: DType,
    seed: Int = 42,
    low: Float64 = -1.0,
    high: Float64 = 1.0,
) raises -> AnyTensor:
    """Create random tensor with fixed seed for gradient checking.

    Generates a tensor filled with random values in the range [low, high] using
    a fixed seed for reproducibility. The seed ensures that calling this function
    multiple times with the same seed produces identical results, which is essential
    for testing and gradient checking.

    Args:
        shape: Tensor dimensions.
        dtype: Data type (FP4, FP8, FP16, FP32, BF16, or Int8).
        seed: Random seed for reproducibility (default: 42).
        low: Minimum value for random range (default: -1.0).
        high: Maximum value for random range (default: 1.0).

    Returns:
        AnyTensor with seeded random values in [low, high].

    Raises:
        Error if low >= high (invalid range).

    Example:
        ```mojo
        # Create reproducible random tensor for gradient checking
        var random1 = create_seeded_random_tensor([3, 3], DType.float32, 42, -1.0, 1.0)
        var random2 = create_seeded_random_tensor([3, 3], DType.float32, 42, -1.0, 1.0)
        # random1 and random2 are identical

        # Different seed produces different values
        var random3 = create_seeded_random_tensor([3, 3], DType.float32, 123, -1.0, 1.0)
        # random3 is different from random1

        # For small random values (gradients)
        var small_random = create_seeded_random_tensor([2, 2], DType.float64, 42, -0.01, 0.01)
        ```
    """
    if low >= high:
        raise Error(
            "Invalid range: low ("
            + String(low)
            + ") must be less than high ("
            + String(high)
            + ")"
        )

    # Set seed for reproducible random generation
    random_seed(seed)

    # Create zero-initialized tensor
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with seeded random values
    var range_size = high - low
    for i in range(numel):
        # Generate uniform random in [0, 1) and scale to [low, high)
        var random_val = random_float64()
        var scaled_val = low + random_val * range_size
        tensor._set_float64(i, scaled_val)

    return tensor^


fn create_zeros_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create tensor filled with zeros (special value).

    Convenience wrapper around create_special_value_tensor for zero initialization.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.

    Returns:
        AnyTensor filled with zeros.

    Example:
        ```mojo
        var z = create_zeros_tensor([3, 3], DType.float32)
        verify_special_value_invariants(z, 0.0)  # Passes
        ```

    Raises:
        Error: If operation fails.
    """
    return create_special_value_tensor(shape, dtype, SPECIAL_VALUE_ZERO)


fn create_ones_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create tensor filled with ones (special value).

    Convenience wrapper around create_special_value_tensor for one initialization.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.

    Returns:
        AnyTensor filled with ones.

    Example:
        ```mojo
        var o = create_ones_tensor([3, 3], DType.float32)
        verify_special_value_invariants(o, 1.0)  # Passes
        ```

    Raises:
        Error: If operation fails.
    """
    return create_special_value_tensor(shape, dtype, SPECIAL_VALUE_ONE)


fn create_halves_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create tensor filled with 0.5 values (special value).

    Convenience wrapper around create_special_value_tensor for half initialization.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.

    Returns:
        AnyTensor filled with 0.5.

    Example:
        ```mojo
        var h = create_halves_tensor([3, 3], DType.float32)
        verify_special_value_invariants(h, 0.5)  # Passes
        ```

    Raises:
        Error: If operation fails.
    """
    return create_special_value_tensor(shape, dtype, SPECIAL_VALUE_HALF)


fn create_one_and_half_tensor(
    shape: List[Int], dtype: DType
) raises -> AnyTensor:
    """Create tensor filled with 1.5 values (special value).

    Convenience wrapper around create_special_value_tensor for 1.5 initialization.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.

    Returns:
        AnyTensor filled with 1.5.

    Example:
        ```mojo
        var t = create_one_and_half_tensor([3, 3], DType.float32)
        verify_special_value_invariants(t, 1.5)  # Passes
        ```

    Raises:
        Error: If operation fails.
    """
    return create_special_value_tensor(shape, dtype, SPECIAL_VALUE_ONE_HALF)


# ============================================================================
# NaN and Inf Tensor Creation (For Numerical Testing)
# ============================================================================


fn create_nan_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create tensor filled with NaN (Not a Number) values.

    Creates NaN values by dividing zero by zero (0.0 / 0.0).
    This works across all float dtypes.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.

    Returns:
        AnyTensor filled with NaN values.

    Raises:
        Error: If operation fails.

    Example:
        ```mojo
        # Create 3x3 tensor filled with NaN (for testing gradient overflow)
        var nan_tensor = create_nan_tensor([3, 3], DType.float32)
        # verify has_nan(nan_tensor) == True
        ```
    """
    var result = zeros(shape, dtype)

    # Create NaN by dividing zero by zero
    # This works across all dtypes
    for i in range(result._numel):
        result._set_float64(i, Float64(0.0) / Float64(0.0))

    return result


fn create_inf_tensor(
    shape: List[Int], dtype: DType, positive: Bool = True
) raises -> AnyTensor:
    """Create tensor filled with Infinity values.

    Creates Infinity by dividing by zero (1.0/0.0 or -1.0/0.0).
    This works across all float dtypes.

    Args:
        shape: Tensor dimensions.
        dtype: Data type.
        positive: If True, create +Infinity; if False, create -Infinity.

    Returns:
        AnyTensor filled with Infinity values.

    Raises:
        Error: If operation fails.

    Example:
        ```mojo
        # Create 3x3 tensor filled with +Inf
        var pos_inf = create_inf_tensor([3, 3], DType.float32, positive=True)

        # Create 2x2 tensor filled with -Inf
        var neg_inf = create_inf_tensor([2, 2], DType.float32, positive=False)
        ```
    """
    var result = zeros(shape, dtype)

    # Create Inf by dividing by zero
    # This works across all dtypes
    if positive:
        for i in range(result._numel):
            result._set_float64(i, Float64(1.0) / Float64(0.0))
    else:
        for i in range(result._numel):
            result._set_float64(i, Float64(-1.0) / Float64(0.0))

    return result
