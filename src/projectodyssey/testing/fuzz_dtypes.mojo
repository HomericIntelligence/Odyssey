"""DType Fuzzing Utilities for ML Odyssey.

Provides utilities for generating random data types and edge case values
for each dtype. Enables discovering dtype-related bugs in tensor operations
through systematic dtype exploration.

Key Features:
    - Random dtype selection with constraints
    - Edge case values for each dtype (min, max, special)
    - Dtype-aware value generation
    - Type compatibility checking

Supported DTypes:
    - Floating point: float16, float32, float64, bfloat16
    - Signed integer: int8, int16, int32, int64
    - Unsigned integer: uint8, uint16, uint32, uint64
    - Boolean: bool

Usage:
    from projectodyssey.testing.fuzz_dtypes import (
        DTypeFuzzer,
        get_edge_values,
        get_dtype_range,
        random_value_for_dtype,
    )

    # Generate random dtypes
    var fuzzer = DTypeFuzzer(seed=42)
    var dtype = fuzzer.random_float_dtype()

    # Get edge case values
    var edge_vals = get_edge_values(DType.float32)

Example:
    ```mojo
    var fuzzer = DTypeFuzzer(seed=42)

    # Test with various dtypes
    for _ in range(100):
        var dtype = fuzzer.random_dtype()
        var tensor = zeros(shape, dtype)
        # Test operations...
    ```
"""

from std.random import random_float64, seed as random_seed
from projectodyssey.testing.fuzz_core import SeededRNG


# ============================================================================
# DType Fuzzer
# ============================================================================


struct DTypeFuzzer(Copyable, Movable):
    """Generator for random data types and dtype-specific values.

    Provides methods for randomly selecting data types and generating
    appropriate values for testing.

    Attributes:
        rng: Seeded random number generator.

    Example:
        ```mojo
        var fuzzer = DTypeFuzzer(seed=42)
        var dtype = fuzzer.random_dtype()
        var val = fuzzer.random_value(dtype)
        ```
    """

    var rng: SeededRNG

    def __init__(out self, seed: Int = 42):
        """Initialize dtype fuzzer.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = SeededRNG(seed)

    def next_iteration(mut self):
        """Advance to next iteration for fresh randomness."""
        self.rng.next_iteration()

    def random_dtype(mut self) -> DType:
        """Select a random data type from all supported types.

        Returns:
            Random DType.

        Example:
            ```mojo
            var dtype = fuzzer.random_dtype()
            # Might return float32, int8, uint16, etc.
            ```
        """
        var dtypes = get_all_dtypes()
        var idx = self.rng.random_int(0, len(dtypes))
        return dtypes[idx]

    def random_float_dtype(mut self) -> DType:
        """Select a random floating-point data type.

        Returns:
            Random floating-point DType.

        Example:
            ```mojo
            var dtype = fuzzer.random_float_dtype()
            # Returns one of: float16, float32, float64, bfloat16
            ```
        """
        var dtypes = get_float_dtypes()
        var idx = self.rng.random_int(0, len(dtypes))
        return dtypes[idx]

    def random_int_dtype(mut self) -> DType:
        """Select a random signed integer data type.

        Returns:
            Random signed integer DType.

        Example:
            ```mojo
            var dtype = fuzzer.random_int_dtype()
            # Returns one of: int8, int16, int32, int64
            ```
        """
        var dtypes = get_int_dtypes()
        var idx = self.rng.random_int(0, len(dtypes))
        return dtypes[idx]

    def random_uint_dtype(mut self) -> DType:
        """Select a random unsigned integer data type.

        Returns:
            Random unsigned integer DType.

        Example:
            ```mojo
            var dtype = fuzzer.random_uint_dtype()
            # Returns one of: uint8, uint16, uint32, uint64
            ```
        """
        var dtypes = get_uint_dtypes()
        var idx = self.rng.random_int(0, len(dtypes))
        return dtypes[idx]

    def random_numeric_dtype(mut self) -> DType:
        """Select a random numeric (non-bool) data type.

        Returns:
            Random numeric DType (float, int, or uint).

        Example:
            ```mojo
            var dtype = fuzzer.random_numeric_dtype()
            # Returns any numeric dtype
            ```
        """
        var dtypes = get_numeric_dtypes()
        var idx = self.rng.random_int(0, len(dtypes))
        return dtypes[idx]

    def random_value(mut self, dtype: DType) -> Float64:
        """Generate a random value appropriate for the given dtype.

        Args:
            dtype: Target data type.

        Returns:
            Random value suitable for the dtype.

        Example:
            ```mojo
            var val = fuzzer.random_value(DType.int8)
            # Returns value in [-128, 127]
            ```
        """
        var range_tuple = get_dtype_range(dtype)
        var low = range_tuple[0]
        var high = range_tuple[1]
        return self.rng.random_float(low, high)

    def random_edge_value(mut self, dtype: DType) -> Float64:
        """Select a random edge case value for the given dtype.

        Args:
            dtype: Target data type.

        Returns:
            Random edge case value (min, max, zero, etc.).

        Example:
            ```mojo
            var val = fuzzer.random_edge_value(DType.float32)
            # Might return 0.0, Inf, -Inf, NaN, max, min, etc.
            ```
        """
        var edge_vals = get_edge_values(dtype)
        var idx = self.rng.random_int(0, len(edge_vals))
        return edge_vals[idx]


# ============================================================================
# DType Categories
# ============================================================================


def get_all_dtypes() -> List[DType]:
    """Get list of all supported data types.

    Returns:
        List of all DTypes including float, int, uint, and bool.

    Example:
        ```mojo
        var dtypes = get_all_dtypes()
        for dtype in dtypes:
            print(dtype)
        ```
    """
    var dtypes = List[DType]()
    # Floating point
    dtypes.append(DType.float16)
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.bfloat16)
    # Signed integers
    dtypes.append(DType.int8)
    dtypes.append(DType.int16)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    # Unsigned integers
    dtypes.append(DType.uint8)
    dtypes.append(DType.uint16)
    dtypes.append(DType.uint32)
    dtypes.append(DType.uint64)
    # Boolean
    dtypes.append(DType.bool)
    return dtypes^


def get_float_dtypes() -> List[DType]:
    """Get list of floating-point data types.

    Returns:
        List of float DTypes: float16, float32, float64, bfloat16.

    Example:
        ```mojo
        var float_dtypes = get_float_dtypes()
        # [float16, float32, float64, bfloat16]
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float16)
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.bfloat16)
    return dtypes^


def get_int_dtypes() -> List[DType]:
    """Get list of signed integer data types.

    Returns:
        List of signed int DTypes: int8, int16, int32, int64.

    Example:
        ```mojo
        var int_dtypes = get_int_dtypes()
        # [int8, int16, int32, int64]
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.int8)
    dtypes.append(DType.int16)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    return dtypes^


def get_uint_dtypes() -> List[DType]:
    """Get list of unsigned integer data types.

    Returns:
        List of unsigned int DTypes: uint8, uint16, uint32, uint64.

    Example:
        ```mojo
        var uint_dtypes = get_uint_dtypes()
        # [uint8, uint16, uint32, uint64]
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.uint8)
    dtypes.append(DType.uint16)
    dtypes.append(DType.uint32)
    dtypes.append(DType.uint64)
    return dtypes^


def get_numeric_dtypes() -> List[DType]:
    """Get list of all numeric (non-bool) data types.

    Returns:
        List of all numeric DTypes.

    Example:
        ```mojo
        var numeric_dtypes = get_numeric_dtypes()
        # All float, int, and uint types
        ```
    """
    var dtypes = List[DType]()
    # Floating point
    dtypes.append(DType.float16)
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.bfloat16)
    # Signed integers
    dtypes.append(DType.int8)
    dtypes.append(DType.int16)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    # Unsigned integers
    dtypes.append(DType.uint8)
    dtypes.append(DType.uint16)
    dtypes.append(DType.uint32)
    dtypes.append(DType.uint64)
    return dtypes^


def get_common_ml_dtypes() -> List[DType]:
    """Get list of data types commonly used in ML.

    Returns:
        List of ML-relevant DTypes: float16, float32, bfloat16.

    Example:
        ```mojo
        var ml_dtypes = get_common_ml_dtypes()
        # [float16, float32, bfloat16]
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float16)
    dtypes.append(DType.float32)
    dtypes.append(DType.bfloat16)
    return dtypes^


# ============================================================================
# DType Ranges
# ============================================================================


def get_dtype_range(dtype: DType) -> Tuple[Float64, Float64]:
    """Get the valid value range for a data type.

    Args:
        dtype: Data type to get range for.

    Returns:
        Tuple of (min_value, max_value).

    Example:
        ```mojo
        var range = get_dtype_range(DType.int8)
        # (-128.0, 127.0)
        ```
    """
    if dtype == DType.float16:
        return (-65504.0, 65504.0)
    elif dtype == DType.float32:
        return (-3.4e38, 3.4e38)
    elif dtype == DType.float64:
        return (-1.7e308, 1.7e308)
    elif dtype == DType.bfloat16:
        return (-3.4e38, 3.4e38)  # Same range as float32
    elif dtype == DType.int8:
        return (-128.0, 127.0)
    elif dtype == DType.int16:
        return (-32768.0, 32767.0)
    elif dtype == DType.int32:
        return (-2147483648.0, 2147483647.0)
    elif dtype == DType.int64:
        return (-9223372036854775808.0, 9223372036854775807.0)
    elif dtype == DType.uint8:
        return (0.0, 255.0)
    elif dtype == DType.uint16:
        return (0.0, 65535.0)
    elif dtype == DType.uint32:
        return (0.0, 4294967295.0)
    elif dtype == DType.uint64:
        return (0.0, 18446744073709551615.0)
    elif dtype == DType.bool:
        return (0.0, 1.0)
    else:
        # Default to float32 range
        return (-3.4e38, 3.4e38)


def get_dtype_safe_range(dtype: DType) -> Tuple[Float64, Float64]:
    """Get a safe value range for a data type (avoiding overflow).

    Returns a conservative range that avoids potential overflow/underflow
    issues in arithmetic operations.

    Args:
        dtype: Data type to get safe range for.

    Returns:
        Tuple of (min_value, max_value).

    Example:
        ```mojo
        var safe_range = get_dtype_safe_range(DType.float32)
        # (-1e6, 1e6) - conservative range for safe arithmetic
        ```
    """
    if dtype == DType.float16:
        return (-1000.0, 1000.0)
    elif dtype == DType.float32:
        return (-1e6, 1e6)
    elif dtype == DType.float64:
        return (-1e15, 1e15)
    elif dtype == DType.bfloat16:
        return (-1000.0, 1000.0)
    elif dtype == DType.int8:
        return (-100.0, 100.0)
    elif dtype == DType.int16:
        return (-30000.0, 30000.0)
    elif dtype == DType.int32:
        return (-2000000000.0, 2000000000.0)
    elif dtype == DType.int64:
        return (-9e18, 9e18)
    elif dtype == DType.uint8:
        return (0.0, 200.0)
    elif dtype == DType.uint16:
        return (0.0, 60000.0)
    elif dtype == DType.uint32:
        return (0.0, 4000000000.0)
    elif dtype == DType.uint64:
        return (0.0, 1e18)
    elif dtype == DType.bool:
        return (0.0, 1.0)
    else:
        return (-1e6, 1e6)


# ============================================================================
# Edge Case Values
# ============================================================================


def get_edge_values(dtype: DType) -> List[Float64]:
    """Get list of edge case values for a data type.

    Returns values that are commonly problematic or boundary cases.

    Args:
        dtype: Data type to get edge values for.

    Returns:
        List of edge case values as Float64.

    Example:
        ```mojo
        var edges = get_edge_values(DType.float32)
        # [0.0, -0.0, 1.0, -1.0, Inf, -Inf, NaN, max, min, epsilon]
        ```
    """
    var values = List[Float64]()

    if is_float_dtype(dtype):
        # Zeros
        values.append(0.0)
        values.append(-0.0)
        # Ones
        values.append(1.0)
        values.append(-1.0)
        # Infinity
        values.append(Float64(1.0) / Float64(0.0))  # +Inf
        values.append(Float64(-1.0) / Float64(0.0))  # -Inf
        # NaN
        values.append(Float64(0.0) / Float64(0.0))  # NaN
        # Approximate bounds based on dtype
        if dtype == DType.float16:
            values.append(65504.0)  # Max float16
            values.append(-65504.0)  # Min float16
            values.append(6.1e-5)  # Epsilon-ish
        elif dtype == DType.float32:
            values.append(3.4e38)  # Approx max float32
            values.append(-3.4e38)  # Approx min float32
            values.append(1.2e-7)  # Epsilon-ish
        else:  # float64
            values.append(1.7e308)  # Approx max float64
            values.append(-1.7e308)  # Approx min float64
            values.append(2.2e-16)  # Epsilon-ish
        # Subnormal-ish
        values.append(1e-40)

    elif is_signed_int_dtype(dtype):
        # Common values
        values.append(0.0)
        values.append(1.0)
        values.append(-1.0)
        # Bounds
        var range_tuple = get_dtype_range(dtype)
        values.append(range_tuple[0])  # Min
        values.append(range_tuple[1])  # Max
        # Near bounds
        values.append(range_tuple[0] + 1)
        values.append(range_tuple[1] - 1)

    elif is_unsigned_int_dtype(dtype):
        # Common values
        values.append(0.0)
        values.append(1.0)
        # Bounds
        var range_tuple = get_dtype_range(dtype)
        values.append(range_tuple[0])  # Min (0)
        values.append(range_tuple[1])  # Max
        # Near max
        values.append(range_tuple[1] - 1)

    elif dtype == DType.bool:
        values.append(0.0)  # False
        values.append(1.0)  # True

    else:
        # Fallback
        values.append(0.0)
        values.append(1.0)

    return values^


def get_special_float_values() -> List[Float64]:
    """Get list of special floating-point values.

    Returns:
        List containing 0, -0, Inf, -Inf, NaN.

    Example:
        ```mojo
        var specials = get_special_float_values()
        for val in specials:
            var tensor = full(shape, val, DType.float32)
            # Test with special value...
        ```
    """
    var values = List[Float64]()
    values.append(0.0)
    values.append(-0.0)
    values.append(Float64(1.0) / Float64(0.0))  # +Inf
    values.append(Float64(-1.0) / Float64(0.0))  # -Inf
    values.append(Float64(0.0) / Float64(0.0))  # NaN
    return values^


# ============================================================================
# DType Classification
# ============================================================================


def is_float_dtype(dtype: DType) -> Bool:
    """Check if dtype is a floating-point type.

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is float16, float32, float64, or bfloat16.

    Example:
        ```mojo
        var is_float = is_float_dtype(DType.float32)  # True
        var is_not_float = is_float_dtype(DType.int32)  # False
        ```
    """
    return (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    )


def is_signed_int_dtype(dtype: DType) -> Bool:
    """Check if dtype is a signed integer type.

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is int8, int16, int32, or int64.

    Example:
        ```mojo
        var is_int = is_signed_int_dtype(DType.int32)  # True
        ```
    """
    return (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    )


def is_unsigned_int_dtype(dtype: DType) -> Bool:
    """Check if dtype is an unsigned integer type.

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is uint8, uint16, uint32, or uint64.

    Example:
        ```mojo
        var is_uint = is_unsigned_int_dtype(DType.uint32)  # True
        ```
    """
    return (
        dtype == DType.uint8
        or dtype == DType.uint16
        or dtype == DType.uint32
        or dtype == DType.uint64
    )


def is_integer_dtype(dtype: DType) -> Bool:
    """Check if dtype is any integer type (signed or unsigned).

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is any int or uint type.

    Example:
        ```mojo
        var is_int = is_integer_dtype(DType.int32)  # True
        var is_not_int = is_integer_dtype(DType.float32)  # False
        ```
    """
    return is_signed_int_dtype(dtype) or is_unsigned_int_dtype(dtype)


def is_numeric_dtype(dtype: DType) -> Bool:
    """Check if dtype is numeric (not bool).

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is float or integer.

    Example:
        ```mojo
        var is_numeric = is_numeric_dtype(DType.float32)  # True
        var is_not_numeric = is_numeric_dtype(DType.bool)  # False
        ```
    """
    return is_float_dtype(dtype) or is_integer_dtype(dtype)


def supports_negative(dtype: DType) -> Bool:
    """Check if dtype supports negative values.

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype can represent negative values.

    Example:
        ```mojo
        var can_neg = supports_negative(DType.int32)  # True
        var no_neg = supports_negative(DType.uint32)  # False
        ```
    """
    return is_float_dtype(dtype) or is_signed_int_dtype(dtype)


def supports_nan_inf(dtype: DType) -> Bool:
    """Check if dtype supports NaN and Inf values.

    Args:
        dtype: Data type to check.

    Returns:
        True if dtype is floating-point (supports NaN/Inf).

    Example:
        ```mojo
        var supports = supports_nan_inf(DType.float32)  # True
        var no_support = supports_nan_inf(DType.int32)  # False
        ```
    """
    return is_float_dtype(dtype)


# ============================================================================
# DType Size and Precision
# ============================================================================


def get_dtype_size_bytes(dtype: DType) -> Int:
    """Get the size of a dtype in bytes.

    Args:
        dtype: Data type to get size for.

    Returns:
        Size in bytes.

    Example:
        ```mojo
        var size = get_dtype_size_bytes(DType.float32)  # 4
        ```
    """
    if dtype == DType.float16 or dtype == DType.bfloat16:
        return 2
    elif dtype == DType.float32:
        return 4
    elif dtype == DType.float64:
        return 8
    elif dtype == DType.int8 or dtype == DType.uint8:
        return 1
    elif dtype == DType.int16 or dtype == DType.uint16:
        return 2
    elif dtype == DType.int32 or dtype == DType.uint32:
        return 4
    elif dtype == DType.int64 or dtype == DType.uint64:
        return 8
    elif dtype == DType.bool:
        return 1
    else:
        return 4  # Default


def get_dtype_precision_bits(dtype: DType) -> Int:
    """Get the precision (mantissa bits) for floating-point dtypes.

    Args:
        dtype: Data type to get precision for.

    Returns:
        Number of mantissa bits (0 for non-float types).

    Example:
        ```mojo
        var bits = get_dtype_precision_bits(DType.float32)  # 23
        ```
    """
    if dtype == DType.float16:
        return 10
    elif dtype == DType.float32:
        return 23
    elif dtype == DType.float64:
        return 52
    elif dtype == DType.bfloat16:
        return 7
    else:
        return 0


def dtype_to_string(dtype: DType) -> String:
    """Convert dtype to human-readable string.

    Args:
        dtype: Data type to convert.

    Returns:
        String representation.

    Example:
        ```mojo
        var s = dtype_to_string(DType.float32)  # "float32"
        ```
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.bfloat16:
        return "bfloat16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"
