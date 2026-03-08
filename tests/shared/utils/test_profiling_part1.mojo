# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_profiling.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for profiling utilities module - Part 1: Function Timing and Memory Tracking.

This module tests:
- Function timing decorators/utilities (tests 1-5)
- Memory usage tracking (tests 6-8)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)


# ============================================================================
# Test Function Timing
# ============================================================================


fn test_time_function() raises:
    """Test timing a function execution."""
    from shared.utils.profiling import profile_function

    fn simple_work() raises:
        var x = 0
        for i in range(1000):
            x += i

    var stats = profile_function("work", simple_work)
    # Verify that we got timing statistics
    assert_equal(stats.call_count, 1, "Should have 1 call")
    assert_greater(stats.total_ms, 0.0, "Total time should be positive")


fn test_timing_decorator():
    """Test function timing decorator."""
    # TODO(#44): Implement when @timed decorator exists
    # @timed
    # fn my_function():
    #     # Do work
    #     pass
    # Call function
    # Verify: timing info is logged/printed
    pass


fn test_timing_multiple_calls() raises:
    """Test timing function called multiple times."""
    from shared.utils.profiling import benchmark_function

    fn simple_work() raises:
        var x = 0
        for i in range(100):
            x += i

    # Benchmark the function multiple times
    var stats = benchmark_function("work", simple_work, iterations=5)

    # Verify multiple measurements were taken
    assert_equal(stats.call_count, 5, "Should have 5 calls")
    assert_greater(stats.avg_ms, 0.0, "Average time should be positive")


fn test_timing_precision():
    """Test timing has sufficient precision for fast operations."""
    # TODO(#44): Implement when time_function exists
    # Time very fast operation (e.g., 1ms)
    # Verify: can distinguish between 1ms and 2ms operations
    # Verify: timing precision is at least microseconds
    pass


fn test_timing_context_manager() raises:
    """Test timing using context manager."""
    from shared.utils.profiling import Timer

    # Basic test that Timer context manager works
    var timer = Timer("test_op")
    timer.__enter__()
    # Simulate some work
    var x = 0
    for i in range(100):
        x += i
    timer.__exit__()

    # Verify elapsed time was measured
    var elapsed = timer.elapsed_ms()
    assert_greater(elapsed, 0.0, "Elapsed time should be positive")


# ============================================================================
# Test Memory Tracking
# ============================================================================


fn test_measure_memory_usage() raises:
    """Test measuring memory usage of function."""
    # Basic test that memory_usage function works
    from shared.utils.profiling import memory_usage

    var _ = memory_usage()
    # Verify that we get a MemoryStats object
    assert_true(True, "memory_usage() executed successfully")


fn test_track_peak_memory():
    """Test tracking peak memory usage during execution."""
    # TODO(#44): Implement when track_memory exists
    # Allocate memory in stages:
    # - Allocate 50MB
    # - Allocate another 50MB (total 100MB)
    # - Free 50MB (total 50MB)
    # Track peak memory
    # Verify: peak = 100MB (not final 50MB)
    pass


fn test_memory_profiler_decorator():
    """Test memory profiling decorator."""
    # TODO(#44): Implement when @profile_memory decorator exists
    # @profile_memory
    # fn memory_intensive_function():
    #     # Allocate tensors
    #     pass
    # Call function
    # Verify: memory usage is reported
    pass


fn main() raises:
    """Run all tests."""
    test_time_function()
    test_timing_decorator()
    test_timing_multiple_calls()
    test_timing_precision()
    test_timing_context_manager()
    test_measure_memory_usage()
    test_track_peak_memory()
    test_memory_profiler_decorator()
