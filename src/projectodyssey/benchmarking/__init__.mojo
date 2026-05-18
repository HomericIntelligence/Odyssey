"""Benchmarking Framework for ML Odyssey.

Comprehensive performance measurement utilities with two APIs:

1. **High-Level API** (runner.py) - Simple benchmarking with statistics
   - benchmark_function(): Single function with automatic warmup/measurement
   - BenchmarkStatistics: Statistics including percentiles (p50, p95, p99)
   - Automatic throughput calculation

2. **Low-Level API** (result.py) - Granular iteration tracking
   - BenchmarkResult: Record individual iteration times, compute online stats

Namespace Resolution (Issue #2457):
   - High-level API uses BenchmarkStatistics (renamed from BenchmarkResult)
   - Low-level API uses BenchmarkResult (unchanged)
   - BenchmarkResult comptime exists in runner.mojo for backward compatibility
   - Uses Welford's algorithm for numerically stable mean/variance
   - Efficient memory usage for large iteration counts

Modules:
    result: Low-level iteration-level timing and statistics (Issue #2282)
    runner: High-level function benchmarking and reporting (Issue #2283)

Features:
    - High-resolution timers (platform-specific nanosecond precision)
    - Warmup iterations for JIT compilation and cache warming
    - Percentile calculations (p50, p95, p99)
    - Throughput measurement (operations per second)
    - Standard deviation and min/max tracking
    - Formatted benchmark reports

Example - High-level quick benchmarking:
    from projectodyssey.benchmarking import benchmark_function, print_benchmark_report

    def expensive_operation():
        # Your operation here
        pass

    var result = benchmark_function(expensive_operation, warmup_iters=10, measure_iters=100)
    print_benchmark_report(result, "Expensive Operation")

Example - Low-level granular tracking:
    from projectodyssey.benchmarking.result import BenchmarkResult
    from time import now

    var result = BenchmarkResult("custom_benchmark", iterations=0)

    for _ in range(1000):
        var start = now()
        operation()
        var end = now()
        result.record(end - start)

    print("Mean:", result.mean() / 1_000_000.0, "ms")
    print("Std Dev:", result.std() / 1_000_000.0, "ms")

Example - Advanced runner with manual control:
    from projectodyssey.benchmarking import BenchmarkRunner
    from time import now

    var runner = BenchmarkRunner("custom_operation", warmup_iters=10)
    runner.run_warmup(lambda: operation())

    for _ in range(100):
        var start = now()
        operation()
        var end = now()
        runner.record_iteration(end - start)

    print("Mean latency:", runner.get_mean_ms(), "ms")
    print("Std dev:", runner.get_std_ms(), "ms")
"""

# Package version
from projectodyssey.version import VERSION

# ============================================================================
# Exports - Implemented modules
# ============================================================================

from projectodyssey.benchmarking.runner import (
    BenchmarkStatistics,  # High-level benchmark results with percentiles (primary)
    BenchmarkResult,  # Alias for backward compatibility (points to BenchmarkStatistics)
    benchmark_function,  # Main benchmarking function
    print_benchmark_report,  # Print formatted results
    print_benchmark_summary,  # Print summary table
    BenchmarkConfig,  # Configuration for benchmarking
    create_benchmark_config,  # Create default config
    BenchmarkRunner,  # Advanced runner with low-level tracking
    # Legacy API (backwards compatibility with benchmarks/framework.mojo)
    LegacyBenchmarkConfig,  # Legacy config with warmup/iterations params
    LegacyBenchmarkResult,  # Legacy result with microsecond units
    benchmark_operation,  # Legacy benchmark_operation function
)
