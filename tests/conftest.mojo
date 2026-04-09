"""Test configuration for AnyTensor test suite.

This file contains shared configuration and setup for all tests.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones


# ============================================================================
# Test Constants
# ============================================================================

# Default tolerance for floating-point comparisons
comptime DEFAULT_RTOL = 1e-5
comptime DEFAULT_ATOL = 1e-8
comptime DEFAULT_FLOAT_DTYPE = DType.float32

# Per-dtype relative tolerances
comptime RTOL_FP32 = 1e-5
comptime RTOL_FP16 = 1e-2
comptime RTOL_BF16 = 1e-1
comptime RTOL_FP8 = 1e-1

# Per-dtype absolute tolerances
comptime ATOL_FP32 = 1e-8
comptime ATOL_FP16 = 1e-3
comptime ATOL_BF16 = 1e-2
comptime ATOL_FP8 = 1e-2


# ============================================================================
# Multi-Precision Test Constants
# ============================================================================


def get_test_dtypes() -> List[DType]:
    """Get list of data types to test across.

    Returns:
        List of DType values including float32, float16, etc.
    """
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    dtypes.append(DType.float16)
    return dtypes^


def get_rtol(dtype: DType) -> Float64:
    """Get relative tolerance for a given data type.

    Args:
        dtype: The data type.

    Returns:
        Recommended relative tolerance for the dtype.
    """
    if dtype == DType.float32:
        return RTOL_FP32
    elif dtype == DType.float16:
        return RTOL_FP16
    elif dtype == DType.bfloat16:
        return RTOL_BF16
    else:
        # Default to conservative tolerance for unknown types
        return RTOL_FP8


def get_atol(dtype: DType) -> Float64:
    """Get absolute tolerance for a given data type.

    Args:
        dtype: The data type.

    Returns:
        Recommended absolute tolerance for the dtype.
    """
    if dtype == DType.float32:
        return ATOL_FP32
    elif dtype == DType.float16:
        return ATOL_FP16
    elif dtype == DType.bfloat16:
        return ATOL_BF16
    else:
        # Default to conservative tolerance for unknown types
        return ATOL_FP8


# ============================================================================
# Test Timing Utilities
# ============================================================================


def measure_time[func: def () raises -> None]() raises -> Float64:
    """Measure execution time of a function in milliseconds.

    Returns:
        Execution time in milliseconds. Returns 0.0 if timing not available.

    Example:
        ```mojo
        def my_test() raises:
            var t = random_tensor([1000, 1000], DType.float32)
            _ = t + t

        var time_ms = measure_time[my_test]()
        ```

    Note:
        This is a placeholder implementation. Real timing would use
        time.perf_counter_ns() or similar when available in Mojo.
    """
    try:
        func()
    except:
        pass
    # Placeholder: return 0.0 since timing APIs may not be available
    return 0.0


def measure_throughput[
    func: def () raises -> None
](n_iterations: Int) raises -> Float64:
    """Measure throughput (operations per second) of a function.

    Args:
        n_iterations: Number of times to run the function.

    Returns:
        Operations per second. Returns 0.0 if timing not available.

    Example:
        ```mojo
        def my_op() raises:
            var t = random_tensor([100, 100], DType.float32)
            _ = t + t

        var ops_per_sec = measure_throughput[my_op](1000)
        ```
    """
    var total_time = 0.0
    for _ in range(n_iterations):
        var iter_time = measure_time[func]()
        total_time += iter_time

    if total_time > 0.0:
        # Convert milliseconds to seconds and calculate ops/sec
        var total_seconds = total_time / 1000.0
        return Float64(n_iterations) / total_seconds
    return 0.0


# ============================================================================
# Test Fixtures Helper
# ============================================================================


struct TestFixtures:
    """Helper struct for creating common test tensors.

    Provides convenient methods for creating fixed-value tensors
    for use in tests.

    Example:
        ```mojo
        var fixtures = TestFixtures()
        var small = fixtures.small_tensor()
        var random = fixtures.random_tensor(10, 10)
        ```
    """

    def small_tensor(self) raises unified {read} -> AnyTensor:
        """Create a small 3x3 tensor with known values.

        Returns:
            AnyTensor with shape [3, 3] and deterministic values.

        Example:
            ```mojo
            var t = TestFixtures().small_tensor()
            # Returns [[1, 2, 3], [4, 5, 6], [7, 8, 9]] as float32
            ```
        """
        var tensor = zeros([3, 3], DType.float32)
        # Fill with values 1-9
        for i in range(9):
            tensor._set_float64(i, Float64(i + 1))
        return tensor

    def random_tensor(self, rows: Int, cols: Int) raises unified {read} -> AnyTensor:
        """Create a random tensor with deterministic seed.

        Args:
            rows: Number of rows.
            cols: Number of columns.

        Returns:
            AnyTensor with shape [rows, cols] containing random values.

        Note:
            Uses seed=42 for reproducibility.

        Example:
            ```mojo
            var t = TestFixtures().random_tensor(10, 10)
            # Returns 10x10 tensor with random values
            ```
        """
        from tests.helpers.fixtures import (
            random_tensor as fixture_random_tensor,
        )

        var shape = List[Int]()
        shape.append(rows)
        shape.append(cols)
        return fixture_random_tensor(shape, DType.float32)
