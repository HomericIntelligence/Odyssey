"""Data Type Testing Utilities

Provides functions for iterating over data types during comprehensive testing.

This module helps tests run across multiple numeric types to ensure correctness
across different precision levels:
- FP16 (float16): Lower precision for edge cases
- FP32 (float32): Standard precision (default)
- BFloat16 (bfloat16): Alternate lower precision
- Int8 (int8): Integer quantization

Custom types (FP4, FP8, BF8) are not available in Mojo stdlib v0.26.1 and are
not included. When support is added, extend get_test_dtypes() to include them.

Usage:
    from shared.testing.dtype_utils import get_test_dtypes

    # Get all dtypes to test
    var dtypes = get_test_dtypes()

    # Use in test loops
    for dtype in dtypes:
        # Test with current dtype
        test_layer(dtype)

Typical test patterns:
    1. Simple tests: Use FP32 only (fast)
    2. Comprehensive tests: Use all dtypes (thorough)
    3. Edge case tests: Use FP16 + BFloat16 (precision issues)
"""


# ============================================================================
# Public API
# ============================================================================


def get_test_dtypes() -> List[DType]:
    """Return all dtypes to test for comprehensive coverage.

    Returns a list of DType values suitable for testing neural network
    implementations across different precision levels:
    - FP16 (float16): Lower precision, tests edge cases
    - FP32 (float32): Standard precision, baseline correctness
    - BFloat16 (bfloat16): Alternate lower precision format
    - Int8 (int8): Integer quantization, tests discrete values

    Returns:
        List of DType values ordered from lower to higher precision:
        [FP16, BFloat16, Int8, FP32]

    Notes:
        - These are the dtypes available in Mojo stdlib v0.26.1
        - FP4, FP8, BF8 are documented in requirements but not available yet
        - Custom dtypes should be added here once implemented
        - See shared/types/ for future custom dtype implementations

    Example:
        ```mojo
        from shared.testing.dtype_utils import get_test_dtypes

        var dtypes = get_test_dtypes()
        for dtype in dtypes:
            print(f"Testing with dtype: {dtype}")
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float16)  # Lower precision
    dtypes.append(DType.bfloat16)  # Alternate format
    dtypes.append(DType.int8)  # Integer quantization
    dtypes.append(DType.float32)  # Standard precision
    return dtypes^


def get_float_dtypes() -> List[DType]:
    """Return only floating-point dtypes for testing.

    Excludes integer types (Int8) which may not be suitable for all
    floating-point operations.

    Returns:
        List of floating-point DType values:
        [FP16, BFloat16, FP32]

    Usage:
        ```mojo
        from shared.testing.dtype_utils import get_float_dtypes

        # For operations that require floating-point arithmetic
        var float_dtypes = get_float_dtypes()
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float16)
    dtypes.append(DType.bfloat16)
    dtypes.append(DType.float32)
    return dtypes^


def get_precision_dtypes() -> List[DType]:
    """Return dtypes in order of increasing precision.

    Useful for testing behavior as precision increases from lower to higher.
    Ordered: [Int8, FP16, BFloat16, FP32]

    Returns:
        List of DType values ordered by precision level.

    Usage:
        ```mojo
        from shared.testing.dtype_utils import get_precision_dtypes

        # Test precision-dependent behavior
        var dtypes = get_precision_dtypes()
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.int8)  # Lowest precision
    dtypes.append(DType.float16)  # Lower float precision
    dtypes.append(DType.bfloat16)  # Alternate format
    dtypes.append(DType.float32)  # Highest standard precision
    return dtypes^


def get_float32_only() -> List[DType]:
    """Return only FP32 for quick testing.

    Useful for:
    - Development: Fast iteration without multi-dtype overhead
    - CI sanity checks: Verify basic correctness
    - Debugging: Isolate issues with single dtype

    Returns:
        List containing only FP32: [FP32]

    Usage:
        ```mojo
        from shared.testing.dtype_utils import get_float32_only

        # For quick tests during development
        var dtypes = get_float32_only()
        ```
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    return dtypes^


def dtype_to_string(dtype: DType) -> String:
    """Convert DType to human-readable string.

    Args:
        dtype: The DType to convert.

    Returns:
        String representation of the dtype (e.g., "float32", "int8").

    Example:
        ```mojo
        var name = dtype_to_string(DType.float32)  # Returns "float32"
        ```
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.bfloat16:
        return "bfloat16"
    elif dtype == DType.int8:
        return "int8"
    else:
        return "unknown"
