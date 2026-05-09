"""Benchmarking runner with performance measurement and statistical analysis.

This module provides tools for benchmarking function execution with accurate
timing measurements, warmup iterations, and statistical analysis (mean, std dev,
percentiles, throughput).

Example:
   ```mojo
    from shared.benchmarking import benchmark_function, print_benchmark_report

    def compute_operation():
        var result = expensive_operation()

    var config = BenchmarkConfig(warmup_iters=10, measure_iters=100)
    var result = benchmark_function(compute_operation, config)
    print_benchmark_report(result, "Expensive Operation")
    ```
"""

from std.time import perf_counter_ns
from std.math import sqrt
from std.collections import List
from shared.benchmarking.result import (
    BenchmarkResult as LowLevelBenchmarkResult,
)


# ============================================================================
# Benchmark Configuration
# ============================================================================


@fieldwise_init
struct BenchmarkConfig(Copyable, Movable):
    """Configuration for benchmarking operations.

    Attributes:
        warmup_iters: Number of warmup iterations before measurement.
        measure_iters: Number of measurement iterations.
        compute_percentiles: Whether to compute p50, p95, p99.
        report_throughput: Whether to report items per second.
    """

    var warmup_iters: Int
    """Number of warmup iterations before measurement."""
    var measure_iters: Int
    """Number of measurement iterations."""
    var compute_percentiles: Bool
    """Whether to compute percentiles (p50, p95, p99)."""
    var report_throughput: Bool
    """Whether to report items per second."""


# ============================================================================
# Benchmark Results - High-Level API
# ============================================================================


@fieldwise_init
struct BenchmarkStatistics(Copyable, ImplicitlyCopyable, Movable):
    """Results from a benchmark run with statistical analysis (High-Level API).

    Contains timing statistics, percentiles, and throughput metrics for a
    benchmarked operation. Suitable for detailed performance reporting.

    Note: This was renamed from BenchmarkResult to avoid namespace collision
    with the low-level BenchmarkResult in result.mojo. See Issue #2457.

    Attributes:
        mean_latency_ms: Mean execution time in milliseconds.
        std_dev_ms: Standard deviation in milliseconds.
        p50_ms: 50th percentile (median) latency in milliseconds.
        p95_ms: 95th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        min_latency_ms: Minimum latency in milliseconds.
        max_latency_ms: Maximum latency in milliseconds.
        throughput: Operations per second (ops/sec).
        iterations: Total measurement iterations.
        warmup_iterations: Warmup iterations performed.
    """

    var mean_latency_ms: Float64
    """Mean execution time in milliseconds."""
    var std_dev_ms: Float64
    """Standard deviation in milliseconds."""
    var p50_ms: Float64
    """50th percentile (median) latency in milliseconds."""
    var p95_ms: Float64
    """95th percentile latency in milliseconds."""
    var p99_ms: Float64
    """99th percentile latency in milliseconds."""
    var min_latency_ms: Float64
    """Minimum latency in milliseconds."""
    var max_latency_ms: Float64
    """Maximum latency in milliseconds."""
    var throughput: Float64
    """Operations per second (ops/sec)."""
    var iterations: Int
    """Total measurement iterations."""
    var warmup_iterations: Int
    """Warmup iterations performed."""


# ============================================================================
# Benchmarking Implementation
# ============================================================================


def _compute_percentile(data: List[Float64], percentile: Float64) -> Float64:
    """Compute percentile from sorted data.

    Args:
            data: Sorted list of values.
            percentile: Percentile to compute (0-100).

    Returns:
            Percentile value.
    """
    if len(data) == 0:
        return 0.0

    if percentile <= 0.0:
        return data[0]
    if percentile >= 100.0:
        return data[len(data) - 1]

    var idx_float = (percentile / 100.0) * Float64(len(data) - 1)
    var idx = Int(idx_float)
    var frac = idx_float - Float64(idx)

    if idx >= len(data) - 1:
        return data[len(data) - 1]

    # Linear interpolation between points
    var lower = data[idx]
    var upper = data[idx + 1]
    return lower + frac * (upper - lower)


def _sort_ascending(mut data: List[Float64]):
    """Simple bubble sort for small lists.

    Args:
            data: List to sort in-place.
    """
    var n = len(data)
    for i in range(n):
        for j in range(n - i - 1):
            if data[j] > data[j + 1]:
                # Swap
                var temp = data[j]
                data[j] = data[j + 1]
                data[j + 1] = temp


def _get_time_ns() -> Int:
    """Get current time in nanoseconds using platform-specific timer.

        Uses high-resolution timer from Mojo's time module:
        - Linux: clock_gettime(CLOCK_MONOTONIC)
        - macOS: mach_absolute_time()
        - Windows: QueryPerformanceCounter()

    Returns:
            Time in nanoseconds as Int.
    """
    return Int(perf_counter_ns())


def _ns_to_ms(ns: Int) -> Float64:
    """Convert nanoseconds to milliseconds.

    Args:
            ns: Time in nanoseconds.

    Returns:
            Time in milliseconds as Float64.
    """
    return Float64(ns) / 1_000_000.0


# ============================================================================
# Main Benchmarking Function
# ============================================================================


def benchmark_function[
    FuncType: def() raises -> None
](
    func: FuncType,
    warmup_iters: Int = 10,
    measure_iters: Int = 100,
    compute_percentiles: Bool = True,
) raises -> BenchmarkStatistics:
    """Benchmark a function with statistical analysis.

        Executes a function multiple times with warmup iterations, then
        measures timing statistics across measurement iterations using high-
        resolution timers. Computes mean, standard deviation, percentiles,
        and throughput.

    Args:
            func: Function to benchmark (takes no args, returns nothing).
            warmup_iters: Number of warmup iterations (default 10).
            measure_iters: Number of measurement iterations (default 100).
            compute_percentiles: Whether to compute percentiles (default True).

    Returns:
            BenchmarkStatistics with timing statistics (latencies in milliseconds).

    Raises:
            Error: If benchmarking fails.

        Example:
           ```mojo
            def expensive_op():
                _ = compute_something()

            var result = benchmark_function(expensive_op, warmup_iters=10, measure_iters=100)
            print("Mean latency:", result.mean_latency_ms, "ms")
            ```
    """
    # Warmup iterations - warm up CPU cache, JIT compilation, etc.
    for _ in range(warmup_iters):
        func()

    # Measurement iterations - collect latencies in nanoseconds
    var latencies_ns = List[Int]()

    for _ in range(measure_iters):
        var start_time_ns = _get_time_ns()
        func()
        var end_time_ns = _get_time_ns()

        var elapsed_ns = end_time_ns - start_time_ns
        latencies_ns.append(elapsed_ns)

    # Convert to milliseconds for statistics computation
    var latencies_ms: List[Float64] = []
    for latency_ns in latencies_ns:
        latencies_ms.append(_ns_to_ms(latency_ns))

    # Compute statistics
    var total_latency_ms = 0.0
    var min_latency_ms = latencies_ms[0]
    var max_latency_ms = latencies_ms[0]

    for latency_ms in latencies_ms:
        total_latency_ms += latency_ms
        if latency_ms < min_latency_ms:
            min_latency_ms = latency_ms
        if latency_ms > max_latency_ms:
            max_latency_ms = latency_ms

    var mean_latency_ms = total_latency_ms / Float64(measure_iters)

    # Compute standard deviation using sample variance (N denominator)
    var variance = 0.0
    for latency_ms in latencies_ms:
        var diff = latency_ms - mean_latency_ms
        variance += diff * diff
    variance = variance / Float64(measure_iters)
    var std_dev_ms = sqrt(variance)

    # Compute percentiles if requested
    var p50_ms = 0.0
    var p95_ms = 0.0
    var p99_ms = 0.0

    if compute_percentiles:
        _sort_ascending(latencies_ms)
        p50_ms = _compute_percentile(latencies_ms, 50.0)
        p95_ms = _compute_percentile(latencies_ms, 95.0)
        p99_ms = _compute_percentile(latencies_ms, 99.0)

    # Compute throughput (operations per second)
    # If mean latency is in ms, then ops/sec = 1000 / mean_latency_ms
    var throughput_ops_per_sec: Float64
    if mean_latency_ms > 0.0:
        throughput_ops_per_sec = 1000.0 / mean_latency_ms
    else:
        throughput_ops_per_sec = 0.0

    return BenchmarkStatistics(
        mean_latency_ms=mean_latency_ms,
        std_dev_ms=std_dev_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        min_latency_ms=min_latency_ms,
        max_latency_ms=max_latency_ms,
        throughput=throughput_ops_per_sec,
        iterations=measure_iters,
        warmup_iterations=warmup_iters,
    )


# ============================================================================
# BenchmarkRunner Class - Advanced benchmarking with low-level tracking
# ============================================================================


struct BenchmarkRunner(Movable):
    """Advanced benchmarking runner using low-level result tracking.

    Provides fine-grained iteration timing collection via the low-level
    BenchmarkResult struct from the result module. Useful for detailed
    performance analysis and percentile computation.

    Example:
       ```mojo
        var runner = BenchmarkRunner("operation", warmup_iters=10)
        for _ in range(100):
            var start = now()
            some_operation()
            var end = now()
            runner.record_iteration(end - start)

        var stats = runner.compute_stats()
        ```
    """

    var name: String
    """Descriptive name for the benchmarked operation."""
    var warmup_iters: Int
    """Number of warmup iterations."""
    var result: LowLevelBenchmarkResult
    """Low-level result tracker for timing data."""

    def __init__(out self, name: String, warmup_iters: Int = 10):
        """Initialize a benchmark runner.

        Args:
            name: Descriptive name for the benchmarked operation.
            warmup_iters: Number of warmup iterations (default 10).
        """
        self.name = name
        self.warmup_iters = warmup_iters
        self.result = LowLevelBenchmarkResult(name, iterations=0)

    def run_warmup[
        FuncType: def() raises -> None
    ](mut self, func: FuncType) raises:
        """Run warmup iterations.

        Args:
            func: Function to run during warmup phase.

        Raises:
            Error: If func raises during warmup.
        """
        for _ in range(self.warmup_iters):
            func()

    def record_iteration(mut self, time_ns: Int):
        """Record a single iteration's execution time.

        Args:
            time_ns: Execution time in nanoseconds.
        """
        self.result.record(time_ns)

    def get_mean_ms(self) -> Float64:
        """Get mean execution time in milliseconds.

        Returns:
            Mean time in milliseconds.
        """
        return self.result.mean() / 1_000_000.0

    def get_std_ms(self) -> Float64:
        """Get standard deviation of execution times in milliseconds.

        Returns:
            Standard deviation in milliseconds.
        """
        return self.result.std() / 1_000_000.0

    def get_min_ms(self) -> Float64:
        """Get minimum execution time in milliseconds.

        Returns:
            Minimum time in milliseconds.
        """
        return self.result.min_time() / 1_000_000.0

    def get_max_ms(self) -> Float64:
        """Get maximum execution time in milliseconds.

        Returns:
            Maximum time in milliseconds.
        """
        return self.result.max_time() / 1_000_000.0

    def get_iterations(self) -> Int:
        """Get total number of iterations recorded.

        Returns:
            Number of iterations.
        """
        return self.result.iterations


# ============================================================================
# Reporting and Display
# ============================================================================


def print_benchmark_report(
    result: BenchmarkStatistics, name: String = "Benchmark"
):
    """Print formatted benchmark report.

    Args:
            result: BenchmarkStatistics from benchmark_function().
            name: Name of benchmarked operation (default "Benchmark").
    """
    print("")
    print("=" * 70)
    print("Benchmark Report: " + name)
    print("=" * 70)
    print("")

    # Configuration
    print("Configuration:")
    print("  Warmup iterations: ", result.warmup_iterations)
    print("  Measurement iterations: ", result.iterations)
    print("")

    # Latency statistics
    print("Latency Statistics (milliseconds):")
    print("  Mean: ", result.mean_latency_ms)
    print("  Std Dev: ", result.std_dev_ms)
    print("  Min: ", result.min_latency_ms)
    print("  Max: ", result.max_latency_ms)
    print("  Median (p50): ", result.p50_ms)
    print("  p95: ", result.p95_ms)
    print("  p99: ", result.p99_ms)
    print("")

    # Throughput
    print("Throughput:")
    print("  Operations/sec: ", result.throughput)
    print("")

    print("=" * 70)
    print("")


def print_benchmark_summary(
    results: List[BenchmarkStatistics], names: List[String] = List[String]()
):
    """Print summary table of multiple benchmark results.

        Useful for comparing performance across multiple functions.

    Args:
            results: List of BenchmarkStatistics objects.
            names: Optional list of operation names (defaults to "Op 1", "Op 2", etc.).
    """
    print("")
    print("=" * 100)
    print("Benchmark Summary")
    print("=" * 100)
    print("")

    # Helper function for manual ljust replacement
    def _pad(s: String, width: Int) -> String:
        var result = s
        var padding = width - len(s)
        if padding > 0:
            for _ in range(padding):
                result += " "
        return result

    # Print header
    print(
        _pad("Operation", 20),
        _pad("Mean (ms)", 15),
        _pad("Std Dev (ms)", 15),
        _pad("P50 (ms)", 15),
        _pad("P95 (ms)", 15),
        _pad("P99 (ms)", 15),
        _pad("Ops/sec", 15),
    )
    print("-" * 100)

    # Print results
    for i in range(len(results)):
        var result = results[i]
        var name: String

        if i < len(names):
            name = names[i]
        else:
            name = "Operation " + String(i + 1)

        print(
            _pad(name, 20),
            _pad(String(result.mean_latency_ms), 15),
            _pad(String(result.std_dev_ms), 15),
            _pad(String(result.p50_ms), 15),
            _pad(String(result.p95_ms), 15),
            _pad(String(result.p99_ms), 15),
            _pad(String(result.throughput), 15),
        )

    print("=" * 100)
    print("")


# ============================================================================
# Helper Functions
# ============================================================================


def create_benchmark_config(
    warmup_iters: Int = 10,
    measure_iters: Int = 100,
    compute_percentiles: Bool = True,
    report_throughput: Bool = True,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
            warmup_iters: Warmup iterations (default 10).
            measure_iters: Measurement iterations (default 100).
            compute_percentiles: Compute percentiles (default True).
            report_throughput: Report throughput (default True).

    Returns:
            BenchmarkConfig with specified settings.
    """
    return BenchmarkConfig(
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        compute_percentiles=compute_percentiles,
        report_throughput=report_throughput,
    )


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

# Alias for backward compatibility (Issue #2457)
# The high-level BenchmarkResult was renamed to BenchmarkStatistics to avoid
# namespace collision with the low-level BenchmarkResult in result.mojo.
comptime BenchmarkResult = BenchmarkStatistics


# ============================================================================
# Legacy API (Backwards Compatibility with benchmarks/framework.mojo)
# ============================================================================


struct LegacyBenchmarkConfig(Copyable, Movable):
    """Legacy benchmark configuration (backwards compatible).

    Matches the old benchmarks/framework.mojo API.
    """

    var warmup_iterations: Int
    """Number of warmup iterations."""
    var measure_iterations: Int
    """Number of measurement iterations."""

    def __init__(
        out self,
        warmup: Int = 100,
        iterations: Int = 1000,
    ):
        """Initialize benchmark configuration.

        Args:
            warmup: Warmup iterations (default: 100).
            iterations: Measurement iterations (default: 1000).
        """
        self.warmup_iterations = warmup
        self.measure_iterations = iterations


struct LegacyBenchmarkResult(Copyable, Movable):
    """Legacy benchmark result with microsecond units (backwards compatible).

    Matches the old benchmarks/framework.mojo API with field names in microseconds.
    """

    var name: String
    """Name of the benchmarked operation."""
    var mean_time_us: Float64
    """Mean execution time in microseconds."""
    var std_dev_us: Float64
    """Standard deviation in microseconds."""
    var min_time_us: Float64
    """Minimum execution time in microseconds."""
    var max_time_us: Float64
    """Maximum execution time in microseconds."""
    var p50_us: Float64
    """50th percentile (median) time in microseconds."""
    var p95_us: Float64
    """95th percentile time in microseconds."""
    var p99_us: Float64
    """99th percentile time in microseconds."""
    var throughput_ops_per_sec: Float64
    """Operations per second (ops/sec)."""
    var memory_mb: Float64
    """Memory used in megabytes (optional)."""
    var input_shape: String
    """Input shape description (optional)."""
    var dtype: String
    """Data type description (optional)."""

    def __init__(
        out self,
        name: String,
        mean_time_us: Float64,
        std_dev_us: Float64,
        min_time_us: Float64,
        max_time_us: Float64,
        p50_us: Float64,
        p95_us: Float64,
        p99_us: Float64,
        throughput_ops_per_sec: Float64,
        memory_mb: Float64 = 0.0,
        input_shape: String = "",
        dtype: String = "",
    ):
        """Initialize legacy benchmark result."""
        self.name = name
        self.mean_time_us = mean_time_us
        self.std_dev_us = std_dev_us
        self.min_time_us = min_time_us
        self.max_time_us = max_time_us
        self.p50_us = p50_us
        self.p95_us = p95_us
        self.p99_us = p99_us
        self.throughput_ops_per_sec = throughput_ops_per_sec
        self.memory_mb = memory_mb
        self.input_shape = input_shape
        self.dtype = dtype


def benchmark_operation[
    FuncType: def() raises -> None
](
    name: String,
    operation: FuncType,
    config: LegacyBenchmarkConfig = LegacyBenchmarkConfig(
        warmup=100, iterations=1000
    ),
) raises -> LegacyBenchmarkResult:
    """Run operation multiple times and collect statistics (legacy API).

        Performs warmup phase for JIT compilation, then measures operation
        execution time across multiple iterations. Computes mean, standard
        deviation, percentiles, and throughput.

        This function provides backwards compatibility with the old
        benchmarks/framework.mojo API.

    Args:
            name: Descriptive name for the operation.
            operation: Function to benchmark (should be self-contained).
            config: Benchmark configuration (warmup, iterations).

    Returns:
            LegacyBenchmarkResult with timing statistics in microseconds.

    Raises:
            Error: If benchmarking operation fails.
    """
    # Use the new benchmark_function internally
    var result = benchmark_function(
        operation,
        warmup_iters=config.warmup_iterations,
        measure_iters=config.measure_iterations,
        compute_percentiles=True,
    )

    # Convert milliseconds to microseconds for backwards compatibility
    return LegacyBenchmarkResult(
        name=name,
        mean_time_us=result.mean_latency_ms * 1000.0,
        std_dev_us=result.std_dev_ms * 1000.0,
        min_time_us=result.min_latency_ms * 1000.0,
        max_time_us=result.max_latency_ms * 1000.0,
        p50_us=result.p50_ms * 1000.0,
        p95_us=result.p95_ms * 1000.0,
        p99_us=result.p99_ms * 1000.0,
        throughput_ops_per_sec=result.throughput,
        memory_mb=0.0,
    )
