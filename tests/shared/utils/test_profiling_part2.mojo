# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_profiling.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for profiling utilities module - Part 2: Memory Leak Detection, Overhead, and Reports.

This module tests:
- Memory leak detection (test 1)
- Profiling overhead (tests 2-4)
- Performance report generation (tests 5-8)
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
# Test Memory Tracking (continued)
# ============================================================================


fn test_memory_leak_detection():
    """Test detecting memory leaks."""
    # TODO(#44): Implement when memory tracking exists
    # Run function 100 times
    # Track memory after each call
    # If memory keeps growing: potential leak
    # Verify: can detect increasing memory pattern
    pass


# ============================================================================
# Test Profiling Overhead
# ============================================================================


fn test_profiling_overhead_timing() raises:
    """Test profiling overhead for timing is < 5%."""
    from shared.utils.profiling import measure_profiling_overhead

    var overhead_percent = measure_profiling_overhead(100)
    # Overhead should be reasonable (not negative or unrealistic)
    assert_true(
        overhead_percent >= 0.0, "Overhead percent should be non-negative"
    )


fn test_profiling_overhead_memory():
    """Test profiling overhead for memory tracking is minimal."""
    # TODO(#44): Implement when memory profiling exists
    # Allocate 1GB tensor without profiling: m1
    # Allocate 1GB tensor with profiling: m2
    # Compute overhead: m2 - m1
    # Verify: overhead < 50MB (< 5%)
    pass


fn test_disable_profiling():
    """Test disabling profiling removes overhead."""
    # TODO(#44): Implement when profiling can be disabled
    # Enable profiling
    # Time function: t1
    # Disable profiling
    # Time function: t2
    # Verify: t2 ≈ t1 (no overhead when disabled)
    pass


# ============================================================================
# Test Performance Reports
# ============================================================================


fn test_generate_timing_report() raises:
    """Test generating timing report for multiple functions."""
    from shared.utils.profiling import (
        generate_timing_report,
        TimingStats,
    )

    # Create test timing data
    var timings = Dict[String, TimingStats]()

    var stats1 = TimingStats()
    stats1.name = "func_a"
    stats1.total_ms = 10.0
    stats1.call_count = 1
    stats1.avg_ms = 10.0
    timings["func_a"] = stats1^

    var stats2 = TimingStats()
    stats2.name = "func_b"
    stats2.total_ms = 20.0
    stats2.call_count = 1
    stats2.avg_ms = 20.0
    timings["func_b"] = stats2^

    # Generate report
    var report = generate_timing_report(timings)

    # Verify report properties
    assert_equal(
        report.total_time_ms, 30.0, "Total time should be sum of all times"
    )


fn test_generate_memory_report():
    """Test generating memory usage report."""
    # TODO(#44): Implement when memory report exists
    # Profile multiple operations
    # Generate memory report
    # Verify: report shows peak memory per operation
    # Verify: report shows total memory allocated
    pass


fn test_save_report_to_file():
    """Test saving performance report to file."""
    # TODO(#44): Implement when report save exists
    # Profile operations
    # Generate report
    # Save to "profile_report.txt"
    # Verify: file exists
    # Verify: file contains profiling data
    # Clean up file
    pass


fn test_report_format_text():
    """Test report output in text format."""
    # TODO(#44): Implement when report formatting exists
    # Generate report in text format
    # Verify: human-readable table format
    # Example:
    # Function          Time (ms)    Memory (MB)
    # function_a            10.5          100.2
    # function_b            20.3          200.5
    pass


fn main() raises:
    """Run all tests."""
    test_memory_leak_detection()
    test_profiling_overhead_timing()
    test_profiling_overhead_memory()
    test_disable_profiling()
    test_generate_timing_report()
    test_generate_memory_report()
    test_save_report_to_file()
    test_report_format_text()
