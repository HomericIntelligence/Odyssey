"""Profiling and performance measurement utilities for ML Odyssey.

This module provides utilities for measuring execution time, tracking memory
usage, and generating performance reports. Useful for identifying bottlenecks
and optimizing performance-critical code.

Example:
    from shared.utils import Timer, profile_function, memory_usage

    # Measure execution time
    with Timer("forward_pass"):
        var output = model.forward(inputs)

    # Get memory usage
    var mem = memory_usage()
    print("Memory: " + String(mem.allocated_mb()) + "MB / " + String(mem.peak_mb()) + "MB")
    ```
"""

import time as mojo_time
from time import perf_counter_ns
from math import sqrt


# ============================================================================
# Timing Data Structures
# ============================================================================


struct TimingRecord(Copyable, Movable):
    """Record of a single timing measurement."""

    var name: String
    var elapsed_ms: Float32
    var call_count: Int

    fn __init__(out self, name: String, elapsed_ms: Float32):
        """Create timing record.

        Args:
            name: Timer name.
            elapsed_ms: Elapsed time in milliseconds.
        """
        self.name = name
        self.elapsed_ms = elapsed_ms
        self.call_count = 1


struct TimingStats(Copyable, Movable):
    """Statistics for multiple timing measurements."""

    var name: String
    var total_ms: Float32
    var call_count: Int
    var avg_ms: Float32
    var min_ms: Float32
    var max_ms: Float32
    var std_dev: Float32

    fn __init__(out self):
        """Create empty timing stats."""
        self.name = ""
        self.total_ms = 0.0
        self.call_count = 0
        self.avg_ms = 0.0
        self.min_ms = 0.0
        self.max_ms = 0.0
        self.std_dev = 0.0


# ============================================================================
# Memory Data Structures
# ============================================================================


struct MemoryStats(Copyable, Movable):
    """Memory usage statistics."""

    var allocated_bytes: Int
    var peak_bytes: Int
    var available_bytes: Int

    fn __init__(out self):
        """Create empty memory stats."""
        self.allocated_bytes = 0
        self.peak_bytes = 0
        self.available_bytes = 0

    fn allocated_mb(self) -> Float32:
        """Get allocated memory in MB."""
        return Float32(self.allocated_bytes) / Float32(1_000_000)

    fn peak_mb(self) -> Float32:
        """Get peak memory in MB."""
        return Float32(self.peak_bytes) / Float32(1_000_000)

    fn available_mb(self) -> Float32:
        """Get available memory in MB."""
        return Float32(self.available_bytes) / Float32(1_000_000)


# ============================================================================
# Profiling Report
# ============================================================================


struct ProfilingReport(Copyable, Movable):
    """Complete profiling report with timing and memory data."""

    var timing_stats: Dict[String, TimingStats]
    var memory_start: MemoryStats
    var memory_end: MemoryStats
    var total_time_ms: Float32

    fn __init__(out self):
        """Create empty profiling report."""
        self.timing_stats = Dict[String, TimingStats]()
        self.memory_start = MemoryStats()
        self.memory_end = MemoryStats()
        self.total_time_ms = 0.0

    fn add_timing(mut self, name: String, stats: TimingStats):
        """Add timing statistics to report.

        Args:
            name: Statistics name.
            stats: Timing statistics to add.
        """
        # Create a new TimingStats with the same values
        var stats_copy = TimingStats()
        stats_copy.name = stats.name
        stats_copy.total_ms = stats.total_ms
        stats_copy.call_count = stats.call_count
        stats_copy.avg_ms = stats.avg_ms
        stats_copy.min_ms = stats.min_ms
        stats_copy.max_ms = stats.max_ms
        stats_copy.std_dev = stats.std_dev
        self.timing_stats[name] = stats_copy^

    fn to_string(self) raises -> String:
        """Format report as human-readable string.

        Returns:
            Human-readable string representation of the report.
        """
        var result = String("Profiling Report\n")
        result += String("================\n")
        result += (
            String("Total Time: ") + String(self.total_time_ms) + String("ms\n")
        )
        result += String("\nTiming Statistics:\n")
        result += String("------------------\n")
        result += String(
            "Function              | Total (ms) | Calls | Avg (ms)  | Min (ms) "
            " | Max (ms)  | StdDev\n"
        )

        # Add separator line
        var sep = String("")
        for _ in range(100):
            sep += "-"
        result += sep + String("\n")

        # Iterate through timing stats - access fields directly to avoid copy issues
        for key in self.timing_stats:
            # Pad name to 21 characters
            var name_padded = key
            var pad_count = 21 - len(key)
            for _ in range(max(0, pad_count)):
                name_padded += " "
            result += name_padded
            result += String(" | ")
            result += String(self.timing_stats[key].total_ms) + String(" | ")
            result += String(self.timing_stats[key].call_count) + String(" | ")
            result += String(self.timing_stats[key].avg_ms) + String(" | ")
            result += String(self.timing_stats[key].min_ms) + String(" | ")
            result += String(self.timing_stats[key].max_ms) + String(" | ")
            result += String(self.timing_stats[key].std_dev) + String("\n")

        result += String("\nMemory Statistics:\n")
        result += String("------------------\n")
        result += (
            String("Start Allocated: ")
            + String(self.memory_start.allocated_mb())
            + String(" MB\n")
        )
        result += (
            String("End Allocated: ")
            + String(self.memory_end.allocated_mb())
            + String(" MB\n")
        )
        result += (
            String("Peak Memory: ")
            + String(self.memory_end.peak_mb())
            + String(" MB\n")
        )

        return result

    fn to_json(self) raises -> String:
        """Format report as JSON.

        Returns:
            JSON string representation of the report.
        """
        var result = String("{\n")
        result += (
            String('  "total_time_ms": ')
            + String(self.total_time_ms)
            + String(",\n")
        )
        result += String('  "timing_stats": {\n')

        var first = True
        for key in self.timing_stats:
            if not first:
                result += String(",\n")
            first = False
            # Access fields directly to avoid copy issues
            result += String('    "') + key + String('": {\n')
            result += (
                String('      "total_ms": ')
                + String(self.timing_stats[key].total_ms)
                + String(",\n")
            )
            result += (
                String('      "call_count": ')
                + String(self.timing_stats[key].call_count)
                + String(",\n")
            )
            result += (
                String('      "avg_ms": ')
                + String(self.timing_stats[key].avg_ms)
                + String(",\n")
            )
            result += (
                String('      "min_ms": ')
                + String(self.timing_stats[key].min_ms)
                + String(",\n")
            )
            result += (
                String('      "max_ms": ')
                + String(self.timing_stats[key].max_ms)
                + String(",\n")
            )
            result += (
                String('      "std_dev": ')
                + String(self.timing_stats[key].std_dev)
                + String("\n")
            )
            result += String("    }")

        result += String("\n  },\n")
        result += String('  "memory_stats": {\n')
        result += (
            String('    "start_allocated_mb": ')
            + String(self.memory_start.allocated_mb())
            + String(",\n")
        )
        result += (
            String('    "end_allocated_mb": ')
            + String(self.memory_end.allocated_mb())
            + String(",\n")
        )
        result += (
            String('    "peak_memory_mb": ')
            + String(self.memory_end.peak_mb())
            + String("\n")
        )
        result += String("  }\n")
        result += String("}")

        return result


# ============================================================================
# Timer Context Manager
# ============================================================================


struct Timer(Copyable, Movable):
    """Context manager for measuring code execution time.

    Measures elapsed time of a code block and optionally prints the result.
    Can be used standalone or in a with statement.

    Example:
        ```mojo
         Basic usage with print.
        with Timer("forward_pass"):
            var output = model.forward(inputs)
        # Output: forward_pass: 0.0234s

        # Without auto-print.
        var timer = Timer("epoch")
        var elapsed = timer.elapsed_ms()
        ```
    """

    var name: String
    var start_ns: Int
    var end_ns: Int

    fn __init__(out self, name: String = ""):
        """Create timer with optional name.

        Args:
            name: Timer name for display.
        """
        self.name = name
        self.start_ns = 0
        self.end_ns = 0

    fn __enter__(mut self):
        """Start timing on entering context."""
        self.start_ns = self._get_time_ns()

    fn __exit__(mut self):
        """Stop timing on exiting context and print result."""
        self.end_ns = self._get_time_ns()
        var elapsed_ms = (self.end_ns - self.start_ns) / 1_000_000.0
        if self.name:
            print(self.name + ": " + String(elapsed_ms) + "ms")
        else:
            print("Elapsed: " + String(elapsed_ms) + "ms")

    fn _get_time_ns(self) -> Int:
        """Get current time in nanoseconds.

        Returns:
            Current time in nanoseconds.
        """
        return Int(perf_counter_ns())

    fn elapsed_ms(self) -> Float32:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds.
        """
        var end = self.end_ns
        if end == 0:
            # Still running - use current time
            end = self._get_time_ns()
        return Float32(end - self.start_ns) / Float32(1_000_000)

    fn elapsed_us(self) -> Float32:
        """Get elapsed time in microseconds.

        Returns:
            Elapsed time in microseconds.
        """
        var end = self.end_ns
        if end == 0:
            end = self._get_time_ns()
        return Float32(end - self.start_ns) / Float32(1_000)

    fn reset(mut self):
        """Reset timer for reuse."""
        self.start_ns = 0
        self.end_ns = 0


# ============================================================================
# Memory Tracking
# ============================================================================


fn memory_usage() -> MemoryStats:
    """Get current memory usage statistics.

    Returns information about allocated, peak, and available memory.

    Returns:
        Memory statistics.

    Example:
        ```mojo
        var mem = memory_usage()
        print("Allocated: " + String(mem.allocated_mb()) + "MB")
        print("Peak: " + String(mem.peak_mb()) + "MB")
        ```
    """
    var stats = MemoryStats()
    # NOTE: Mojo doesn't have direct memory introspection APIs yet
    # This returns approximate values based on available system information
    # For now, we initialize with zeros as a baseline
    stats.allocated_bytes = 0
    stats.peak_bytes = 0
    stats.available_bytes = 0
    return stats^


fn memory_at_checkpoint() -> MemoryStats:
    """Record memory usage at a checkpoint.

    Returns:
        Memory statistics at this point.
    """
    return memory_usage()


fn get_memory_delta(before: MemoryStats, after: MemoryStats) -> Int:
    """Compute memory change between two points.

    Args:
        before: Memory before operation.
        after: Memory after operation.

    Returns:
        Memory delta in bytes (positive = increase).
    """
    return after.allocated_bytes - before.allocated_bytes


# ============================================================================
# Profiling Decorators and Functions
# ============================================================================


fn profile_function(
    name: String, func_ptr: fn () raises -> None
) raises -> TimingStats:
    """Profile a function for execution time.

    Measures function execution time and returns statistics.

    Args:
        name: Function name.
        func_ptr: Pointer to function (simplified).

    Returns:
        Timing statistics for function.
    """
    var start = perf_counter_ns()
    func_ptr()
    var end = perf_counter_ns()

    var elapsed_ms = Float32(end - start) / Float32(1_000_000)
    var stats = TimingStats()
    stats.name = name
    stats.total_ms = elapsed_ms
    stats.call_count = 1
    stats.avg_ms = elapsed_ms
    stats.min_ms = elapsed_ms
    stats.max_ms = elapsed_ms
    stats.std_dev = 0.0
    return stats^


fn benchmark_function(
    name: String, func_ptr: fn () raises -> None, iterations: Int = 10
) raises -> TimingStats:
    """Benchmark function over multiple iterations.

    Runs function multiple times and computes statistics (mean, std dev, etc).

    Args:
        name: Function name.
        func_ptr: Function to benchmark.
        iterations: Number of iterations.

    Returns:
        Timing statistics with min, max, average, std dev.
    """
    var times = List[Float32](capacity=iterations)

    for _ in range(iterations):
        var start = perf_counter_ns()
        func_ptr()
        var end = perf_counter_ns()
        var elapsed_ms = Float32(end - start) / Float32(1_000_000)
        times.append(elapsed_ms)

    # Compute statistics
    var total_ms: Float32 = 0.0
    var min_ms: Float32 = times[0]
    var max_ms: Float32 = times[0]

    for i in range(len(times)):
        var t = times[i]
        total_ms += t
        if t < min_ms:
            min_ms = t
        if t > max_ms:
            max_ms = t

    var avg_ms = total_ms / Float32(iterations)

    # Compute standard deviation
    var sum_sq_diff: Float32 = 0.0
    for i in range(len(times)):
        var diff = times[i] - avg_ms
        sum_sq_diff += diff * diff

    var variance = sum_sq_diff / Float32(iterations)
    var std_dev = sqrt(variance)

    var stats = TimingStats()
    stats.name = name
    stats.total_ms = total_ms
    stats.call_count = iterations
    stats.avg_ms = avg_ms
    stats.min_ms = min_ms
    stats.max_ms = max_ms
    stats.std_dev = std_dev
    return stats^


# ============================================================================
# Call Stack Profiling
# ============================================================================


struct CallStackEntry(Copyable, Movable):
    """Entry in call stack for profiling."""

    var function_name: String
    var elapsed_ms: Float32
    var memory_delta_bytes: Int

    fn __init__(out self, name: String):
        """Create call stack entry.

        Args:
            name: Function name.
        """
        self.function_name = name
        self.elapsed_ms = 0.0
        self.memory_delta_bytes = 0


struct CallStack(Copyable, Movable):
    """Profiling information for entire call stack."""

    var root: CallStackEntry
    var entries: List[CallStackEntry]
    var depth: Int

    fn __init__(out self):
        """Create empty call stack."""
        self.root = CallStackEntry("root")
        self.entries: List[CallStackEntry] = []
        self.depth = 0

    fn push(mut self, name: String):
        """Push function onto call stack.

        Args:
            name: Function name to push.
        """
        self.depth += 1
        self.entries.append(CallStackEntry(name))

    fn pop(mut self):
        """Pop function from call stack."""
        self.depth -= 1

    fn depth_level(self) -> Int:
        """Get current nesting depth.

        Returns:
            Current nesting depth level.
        """
        return self.depth


# ============================================================================
# Performance Report Generation
# ============================================================================


fn generate_timing_report(
    timings: Dict[String, TimingStats]
) raises -> ProfilingReport:
    """Generate profiling report from timing data.

    Args:
        timings: Dictionary of timing statistics.

    Returns:
        Complete profiling report.
    """
    var report = ProfilingReport()
    var total_time: Float32 = 0.0

    # Aggregate timing data - manually copy each field to avoid implicit copy issues
    for key in timings:
        var stats = TimingStats()
        stats.name = key
        stats.total_ms = timings[key].total_ms
        stats.call_count = timings[key].call_count
        stats.avg_ms = timings[key].avg_ms
        stats.min_ms = timings[key].min_ms
        stats.max_ms = timings[key].max_ms
        stats.std_dev = timings[key].std_dev
        report.timing_stats[key] = stats^
        total_time += timings[key].total_ms

    report.total_time_ms = total_time
    report.memory_start = memory_usage()
    report.memory_end = memory_usage()

    return report^


fn print_timing_report(report: ProfilingReport) raises:
    """Print profiling report to console.

    Args:
        report: Report to print.
    """
    print(report.to_string())


fn export_profiling_report(
    report: ProfilingReport, filepath: String, format: String = "json"
) raises -> Bool:
    """Export profiling report to file.

    Args:
        report: Report to export.
        filepath: Output file path.
        format: Export format (json, txt).

    Returns:
        True if export was successful, False otherwise.

    Raises:
        Error if file cannot be opened for writing.
    """
    # Format selection
    var content: String
    if format == "json":
        content = report.to_json()
    elif format == "txt" or format == "text":
        content = report.to_string()
    else:
        # Default to text format for unknown formats
        content = report.to_string()

    # Write content to file using Mojo's built-in file I/O
    with open(filepath, "w") as f:
        f.write(content)

    return True


# ============================================================================
# Profiling Overhead Measurement
# ============================================================================


fn measure_profiling_overhead(num_measurements: Int = 100) raises -> Float32:
    """Measure overhead of profiling operations themselves.

    This is important to ensure profiling doesn't significantly skew results.
    Target: profiling overhead < 5% of measured time.

    Args:
        num_measurements: Number of measurements to take.

    Returns:
        Overhead as percentage of total time.
    """
    # Measure time spent on profiling operations themselves
    var overhead_times = List[Float32](capacity=num_measurements)

    for _ in range(num_measurements):
        var start = perf_counter_ns()
        # Simulate a very light operation
        var _ = 1 + 1
        var end = perf_counter_ns()
        var elapsed_ms = Float32(end - start) / Float32(1_000_000)
        overhead_times.append(elapsed_ms)

    # Compute average overhead
    var total_overhead: Float32 = 0.0
    for i in range(len(overhead_times)):
        total_overhead += overhead_times[i]

    var avg_overhead_ms = total_overhead / Float32(num_measurements)

    # Overhead as a percentage (relative to a typical operation ~1ms)
    # We assume a baseline operation takes ~1ms
    var overhead_percent = (avg_overhead_ms / 1.0) * 100.0

    return overhead_percent


# ============================================================================
# Performance Regression Detection
# ============================================================================


struct BaselineMetrics(Copyable, Movable):
    """Baseline metrics for comparing performance."""

    var name: String
    var avg_time_ms: Float32
    var peak_memory_mb: Float32
    var threshold_percent: Float32  # Warn if slower than this

    fn __init__(out self):
        """Create empty baseline."""
        self.name = ""
        self.avg_time_ms = 0.0
        self.peak_memory_mb = 0.0
        self.threshold_percent = 10.0  # Default: warn if 10% slower


fn compare_to_baseline(
    current: TimingStats, baseline: BaselineMetrics
) -> Tuple[Bool, Float32]:
    """Check if current performance is within baseline tolerance.

    Args:
        current: Current timing statistics.
        baseline: Baseline metrics to compare against.

    Returns:
        Tuple of (is_regression, percent_slower).
    """
    # Calculate percent difference
    var percent_slower = (
        (current.avg_ms - baseline.avg_time_ms) / baseline.avg_time_ms
    ) * 100.0

    # Check if it exceeds threshold
    var is_regression = percent_slower > baseline.threshold_percent

    return (is_regression, percent_slower)


fn detect_performance_regression(
    current_metrics: Dict[String, TimingStats],
    baseline_metrics: Dict[String, BaselineMetrics],
) raises -> List[String]:
    """Detect performance regressions compared to baseline.

    Args:
        current_metrics: Current measurements.
        baseline_metrics: Baseline metrics.

    Returns:
        List of functions with regressions.
    """
    var regressions = List[String]()

    # Check each current metric against baseline
    for key in current_metrics:
        if key in baseline_metrics:
            # Access fields directly and compare without copying entire structs
            var current_avg = current_metrics[key].avg_ms
            var baseline_avg = baseline_metrics[key].avg_time_ms
            var threshold = baseline_metrics[key].threshold_percent
            var percent_slower = (
                (current_avg - baseline_avg) / baseline_avg
            ) * 100.0
            var is_regression = percent_slower > threshold
            if is_regression:
                regressions.append(key)

    return regressions^
