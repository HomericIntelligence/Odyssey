# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_profiling.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for profiling utilities module - Part 3: Report Format, Nested/Line/CPU-GPU Profiling.

This module tests:
- Report format (JSON) (test 1)
- Nested profiling (tests 2-4)
- Line-by-line profiling (tests 5-6)
- CPU vs GPU profiling (tests 7-8)
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
# Test Performance Reports (continued)
# ============================================================================


fn test_report_format_json() raises:
    """Test report output in JSON format."""
    from shared.utils.profiling import (
        ProfilingReport,
        TimingStats,
    )

    var report = ProfilingReport()
    var stats = TimingStats()
    stats.name = "test_func"
    stats.total_ms = 10.0
    stats.call_count = 1
    stats.avg_ms = 10.0

    report.add_timing("test_func", stats)
    report.total_time_ms = 10.0

    var json_output = report.to_json()
    # Verify JSON contains expected content
    assert_true(len(json_output) > 0, "JSON output should not be empty")


# ============================================================================
# Test Nested Profiling
# ============================================================================


fn test_profile_nested_functions():
    """Test profiling nested function calls."""
    # TODO(#44): Implement when nested profiling exists
    # function_a calls function_b calls function_c
    # Profile all three functions
    # Verify: can distinguish time spent in each function
    # Verify: total time adds up correctly
    pass


fn test_profile_call_stack():
    """Test capturing call stack in profile."""
    # TODO(#44): Implement when call stack profiling exists
    # Profile function with nested calls
    # Verify: profile shows call hierarchy
    # Example:
    # function_a (50ms)
    #   -> function_b (30ms)
    #     -> function_c (20ms)
    pass


fn test_profile_recursive_functions():
    """Test profiling recursive function calls."""
    # TODO(#44): Implement when profiling exists
    # Define recursive function (e.g., factorial)
    # Profile recursive calls
    # Verify: each recursive level is tracked
    # Verify: total time is correct
    pass


# ============================================================================
# Test Line-by-Line Profiling
# ============================================================================


fn test_line_profiler():
    """Test profiling execution time per line."""
    # TODO(#44): Implement when line profiler exists
    # @profile_lines
    # fn my_function():
    #     line1: do_work()  # 10ms
    #     line2: do_work()  # 20ms
    #     line3: do_work()  # 30ms
    # Profile function
    # Verify: time reported for each line
    pass


fn test_find_bottleneck_lines():
    """Test identifying bottleneck lines in function."""
    # TODO(#44): Implement when line profiler exists
    # Profile function with one slow line
    # Verify: bottleneck line is highlighted
    # Verify: percentage of total time is shown
    pass


# ============================================================================
# Test CPU vs GPU Profiling
# ============================================================================


fn test_profile_cpu_operations():
    """Test profiling CPU-bound operations."""
    # TODO(#44): Implement when CPU profiling exists
    # Profile matrix multiplication on CPU
    # Verify: CPU time is recorded
    # Verify: can distinguish compute vs memory operations
    pass


fn test_profile_gpu_operations():
    """Test profiling GPU-bound operations (future)."""
    # TODO(#44): Implement when GPU support exists
    # Profile matrix multiplication on GPU
    # Verify: GPU kernel time is recorded
    # Verify: can measure kernel launch overhead
    pass


fn main() raises:
    """Run all tests."""
    test_report_format_json()
    test_profile_nested_functions()
    test_profile_call_stack()
    test_profile_recursive_functions()
    test_line_profiler()
    test_find_bottleneck_lines()
    test_profile_cpu_operations()
    test_profile_gpu_operations()
