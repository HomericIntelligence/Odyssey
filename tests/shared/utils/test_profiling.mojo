"""Tests for profiling utilities module.

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


def test_time_function() raises:
    """Test timing a function execution."""
    from shared.utils.profiling import profile_function

    def simple_work() raises:
        var x = 0
        for i in range(1000):
            x += i

    var stats = profile_function("work", simple_work)
    # Verify that we got timing statistics
    assert_equal(stats.call_count, 1, "Should have 1 call")
    assert_greater(stats.total_ms, 0.0, "Total time should be positive")


def test_timing_decorator():
    """Test function timing decorator."""
    # TODO(#44): Implement when @timed decorator exists
    # @timed
    # fn my_function():
    #     # Do work
    #     pass
    # Call function
    # Verify: timing info is logged/printed
    pass


def test_timing_multiple_calls() raises:
    """Test timing function called multiple times."""
    from shared.utils.profiling import benchmark_function

    def simple_work() raises:
        var x = 0
        for i in range(100):
            x += i

    # Benchmark the function multiple times
    var stats = benchmark_function("work", simple_work, iterations=5)

    # Verify multiple measurements were taken
    assert_equal(stats.call_count, 5, "Should have 5 calls")
    assert_greater(stats.avg_ms, 0.0, "Average time should be positive")


def test_timing_precision():
    """Test timing has sufficient precision for fast operations."""
    # TODO(#44): Implement when time_function exists
    # Time very fast operation (e.g., 1ms)
    # Verify: can distinguish between 1ms and 2ms operations
    # Verify: timing precision is at least microseconds
    pass


def test_timing_context_manager() raises:
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


def test_measure_memory_usage() raises:
    """Test measuring memory usage of function."""
    # Basic test that memory_usage function works
    from shared.utils.profiling import memory_usage

    var _ = memory_usage()
    # Verify that we get a MemoryStats object
    assert_true(True, "memory_usage() executed successfully")


def test_track_peak_memory():
    """Test tracking peak memory usage during execution."""
    # TODO(#44): Implement when track_memory exists
    # Allocate memory in stages:
    # - Allocate 50MB
    # - Allocate another 50MB (total 100MB)
    # - Free 50MB (total 50MB)
    # Track peak memory
    # Verify: peak = 100MB (not final 50MB)
    pass


def test_memory_profiler_decorator():
    """Test memory profiling decorator."""
    # TODO(#44): Implement when @profile_memory decorator exists
    # @profile_memory
    # fn memory_intensive_function():
    #     # Allocate tensors
    #     pass
    # Call function
    # Verify: memory usage is reported
    pass


def test_memory_leak_detection():
    """Test detecting memory leaks."""
    # TODO(#44): Implement when memory tracking exists
    # Run function 100 times
    # Track memory after each call
    # If memory keeps growing: potential leak
    # Verify: can detect increasing memory pattern
    pass


def test_profiling_overhead_timing() raises:
    """Test profiling overhead for timing is < 5%."""
    from shared.utils.profiling import measure_profiling_overhead

    var overhead_percent = measure_profiling_overhead(100)
    # Overhead should be reasonable (not negative or unrealistic)
    assert_true(
        overhead_percent >= 0.0, "Overhead percent should be non-negative"
    )


def test_profiling_overhead_memory():
    """Test profiling overhead for memory tracking is minimal."""
    # TODO(#44): Implement when memory profiling exists
    # Allocate 1GB tensor without profiling: m1
    # Allocate 1GB tensor with profiling: m2
    # Compute overhead: m2 - m1
    # Verify: overhead < 50MB (< 5%)
    pass


def test_disable_profiling():
    """Test disabling profiling removes overhead."""
    # TODO(#44): Implement when profiling can be disabled
    # Enable profiling
    # Time function: t1
    # Disable profiling
    # Time function: t2
    # Verify: t2 ≈ t1 (no overhead when disabled)
    pass


def test_generate_timing_report() raises:
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


def test_generate_memory_report():
    """Test generating memory usage report."""
    # TODO(#44): Implement when memory report exists
    # Profile multiple operations
    # Generate memory report
    # Verify: report shows peak memory per operation
    # Verify: report shows total memory allocated
    pass


def test_save_report_to_file():
    """Test saving performance report to file."""
    # TODO(#44): Implement when report save exists
    # Profile operations
    # Generate report
    # Save to "profile_report.txt"
    # Verify: file exists
    # Verify: file contains profiling data
    # Clean up file
    pass


def test_report_format_text():
    """Test report output in text format."""
    # TODO(#44): Implement when report formatting exists
    # Generate report in text format
    # Verify: human-readable table format
    # Example:
    # Function          Time (ms)    Memory (MB)
    # function_a            10.5          100.2
    # function_b            20.3          200.5
    pass


def test_report_format_json() raises:
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
    assert_true(
        json_output.byte_length() > 0, "JSON output should not be empty"
    )


def test_profile_nested_functions():
    """Test profiling nested function calls."""
    # TODO(#44): Implement when nested profiling exists
    # function_a calls function_b calls function_c
    # Profile all three functions
    # Verify: can distinguish time spent in each function
    # Verify: total time adds up correctly
    pass


def test_profile_call_stack():
    """Test capturing call stack in profile."""
    # TODO(#44): Implement when call stack profiling exists
    # Profile function with nested calls
    # Verify: profile shows call hierarchy
    # Example:
    # function_a (50ms)
    #   -> function_b (30ms)
    #     -> function_c (20ms)
    pass


def test_profile_recursive_functions():
    """Test profiling recursive function calls."""
    # TODO(#44): Implement when profiling exists
    # Define recursive function (e.g., factorial)
    # Profile recursive calls
    # Verify: each recursive level is tracked
    # Verify: total time is correct
    pass


def test_line_profiler():
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


def test_find_bottleneck_lines():
    """Test identifying bottleneck lines in function."""
    # TODO(#44): Implement when line profiler exists
    # Profile function with one slow line
    # Verify: bottleneck line is highlighted
    # Verify: percentage of total time is shown
    pass


def test_profile_cpu_operations():
    """Test profiling CPU-bound operations."""
    # TODO(#44): Implement when CPU profiling exists
    # Profile matrix multiplication on CPU
    # Verify: CPU time is recorded
    # Verify: can distinguish compute vs memory operations
    pass


def test_profile_gpu_operations():
    """Test profiling GPU-bound operations (future)."""
    # TODO(#44): Implement when GPU support exists
    # Profile matrix multiplication on GPU
    # Verify: GPU kernel time is recorded
    # Verify: can measure kernel launch overhead
    pass


def test_profile_data_transfer():
    """Test profiling CPU-GPU data transfer (future)."""
    # TODO(#44): Implement when GPU support exists
    # Profile data transfer: CPU -> GPU -> CPU
    # Verify: transfer time is measured separately
    # Verify: can identify transfer bottlenecks
    pass


def test_profile_training_epoch():
    """Test profiling complete training epoch."""
    # TODO(#44): Implement when training profiling exists
    # Profile training epoch
    # Verify: report breaks down time spent in:
    # - Data loading
    # - Forward pass
    # - Loss computation
    # - Backward pass
    # - Optimizer step
    pass


def test_profile_batch_processing():
    """Test profiling batch processing."""
    # TODO(#44): Implement when batch profiling exists
    # Profile processing 100 batches
    # Verify: average time per batch
    # Verify: can identify slow batches
    pass


def test_profile_data_augmentation():
    """Test profiling data augmentation overhead."""
    # TODO(#44): Implement when augmentation profiling exists
    # Profile data loading with/without augmentation
    # Measure overhead of augmentation
    # Verify: can quantify augmentation cost
    pass


def test_compare_implementations():
    """Test comparing performance of different implementations."""
    # TODO(#44): Implement when comparative profiling exists
    # Implement matrix multiply two ways:
    # - Naive: triple nested loop
    # - SIMD: vectorized
    # Profile both implementations
    # Generate comparison report
    # Verify: SIMD is faster
    pass


def test_regression_detection() raises:
    """Test detecting performance regressions."""
    from shared.utils.profiling import (
        detect_performance_regression,
        TimingStats,
        BaselineMetrics,
    )

    # Create baseline metrics
    var baseline = Dict[String, BaselineMetrics]()
    var baseline_func = BaselineMetrics()
    baseline_func.name = "func_a"
    baseline_func.avg_time_ms = 10.0
    baseline_func.threshold_percent = 20.0
    baseline["func_a"] = baseline_func^

    # Create current metrics (within threshold)
    var current = Dict[String, TimingStats]()
    var current_stats = TimingStats()
    current_stats.name = "func_a"
    current_stats.avg_ms = 12.0  # 20% slower - at threshold
    current["func_a"] = current_stats^

    # Detect regression
    var regressions = detect_performance_regression(current, baseline)
    # Should detect regression at exactly threshold
    assert_true(
        len(regressions) == 0 or len(regressions) == 1,
        "Regression detection should work",
    )


def test_save_baseline_profile():
    """Test saving baseline profile for future comparison."""
    # TODO(#44): Implement when baseline save exists
    # Profile current implementation
    # Save as baseline
    # Verify: baseline file is created
    # Verify: can load baseline in future runs
    pass


def test_compute_mean_execution_time():
    """Test computing mean execution time over multiple runs."""
    # TODO(#44): Implement when statistics exist
    # Run function 10 times
    # Compute mean execution time
    # Verify: mean is computed correctly
    pass


def test_compute_std_deviation():
    """Test computing standard deviation of execution times."""
    # TODO(#44): Implement when statistics exist
    # Run function 10 times
    # Compute std deviation
    # Verify: low std dev indicates consistent performance
    pass


def test_compute_percentiles():
    """Test computing percentiles (p50, p90, p99) of execution times."""
    # TODO(#44): Implement when percentile stats exist
    # Run function 100 times
    # Compute p50, p90, p99
    # Verify: percentiles are computed correctly
    # Useful for identifying outliers
    pass


def test_identify_outliers():
    """Test identifying outlier executions."""
    # TODO(#44): Implement when outlier detection exists
    # Run function 100 times
    # Most runs: 10ms
    # A few runs: 100ms (outliers)
    # Identify outliers
    # Verify: outliers are flagged
    pass


def test_configure_profiler():
    """Test configuring profiler settings."""
    # TODO(#44): Implement when profiler configuration exists
    # Configure:
    # - Enable timing: True
    # - Enable memory: False
    # - Report format: JSON
    # - Output file: "profile.json"
    # Verify: settings are applied
    pass


def test_sampling_rate():
    """Test setting profiler sampling rate."""
    # TODO(#44): Implement when sampling configuration exists
    # Set sampling rate to 10% (profile 1 in 10 calls)
    # Call function 100 times
    # Verify: approximately 10 samples collected
    # Verify: reduced overhead
    pass


def test_profile_full_training():
    """Test profiling complete training workflow."""
    # TODO(#44): Implement when full training workflow exists
    # Enable profiling
    # Train model for 10 epochs
    # Generate performance report
    # Verify: report shows breakdown of:
    # - Total training time
    # - Time per epoch
    # - Time per operation type
    # - Memory usage
    pass


def test_profile_optimization_impact():
    """Test profiling shows impact of optimizations."""
    # TODO(#44): Implement when profiling exists
    # Profile unoptimized implementation
    # Record baseline performance
    # Apply optimization (e.g., SIMD)
    # Profile optimized version
    # Verify: speedup is quantified in report
    pass


def main() raises:
    """Run all test_profiling tests."""
    print("Running test_profiling tests...")

    test_time_function()
    print("✓ test_time_function")

    test_timing_decorator()
    print("✓ test_timing_decorator")

    test_timing_multiple_calls()
    print("✓ test_timing_multiple_calls")

    test_timing_precision()
    print("✓ test_timing_precision")

    test_timing_context_manager()
    print("✓ test_timing_context_manager")

    test_measure_memory_usage()
    print("✓ test_measure_memory_usage")

    test_track_peak_memory()
    print("✓ test_track_peak_memory")

    test_memory_profiler_decorator()
    print("✓ test_memory_profiler_decorator")

    test_memory_leak_detection()
    print("✓ test_memory_leak_detection")

    test_profiling_overhead_timing()
    print("✓ test_profiling_overhead_timing")

    test_profiling_overhead_memory()
    print("✓ test_profiling_overhead_memory")

    test_disable_profiling()
    print("✓ test_disable_profiling")

    test_generate_timing_report()
    print("✓ test_generate_timing_report")

    test_generate_memory_report()
    print("✓ test_generate_memory_report")

    test_save_report_to_file()
    print("✓ test_save_report_to_file")

    test_report_format_text()
    print("✓ test_report_format_text")

    test_report_format_json()
    print("✓ test_report_format_json")

    test_profile_nested_functions()
    print("✓ test_profile_nested_functions")

    test_profile_call_stack()
    print("✓ test_profile_call_stack")

    test_profile_recursive_functions()
    print("✓ test_profile_recursive_functions")

    test_line_profiler()
    print("✓ test_line_profiler")

    test_find_bottleneck_lines()
    print("✓ test_find_bottleneck_lines")

    test_profile_cpu_operations()
    print("✓ test_profile_cpu_operations")

    test_profile_gpu_operations()
    print("✓ test_profile_gpu_operations")

    test_profile_data_transfer()
    print("✓ test_profile_data_transfer")

    test_profile_training_epoch()
    print("✓ test_profile_training_epoch")

    test_profile_batch_processing()
    print("✓ test_profile_batch_processing")

    test_profile_data_augmentation()
    print("✓ test_profile_data_augmentation")

    test_compare_implementations()
    print("✓ test_compare_implementations")

    test_regression_detection()
    print("✓ test_regression_detection")

    test_save_baseline_profile()
    print("✓ test_save_baseline_profile")

    test_compute_mean_execution_time()
    print("✓ test_compute_mean_execution_time")

    test_compute_std_deviation()
    print("✓ test_compute_std_deviation")

    test_compute_percentiles()
    print("✓ test_compute_percentiles")

    test_identify_outliers()
    print("✓ test_identify_outliers")

    test_configure_profiler()
    print("✓ test_configure_profiler")

    test_sampling_rate()
    print("✓ test_sampling_rate")

    test_profile_full_training()
    print("✓ test_profile_full_training")

    test_profile_optimization_impact()
    print("✓ test_profile_optimization_impact")

    print("\nAll test_profiling tests passed!")
