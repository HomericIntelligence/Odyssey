# Performance Benchmarking Guide

## Overview

This guide shows how to benchmark and profile ML Odyssey implementations for performance
analysis. ML Odyssey provides a mature benchmarking framework with both high-level and
low-level APIs, along with best practices for measuring, analyzing, and optimizing
performance-critical code.

### When to Benchmark

Benchmark your code when you need to:

- **Measure performance**: Get precise timing for specific operations
- **Compare optimizations**: Validate that changes actually improve performance
- **Track regressions**: Detect performance degradation in CI/CD pipelines
- **Analyze bottlenecks**: Identify which operations consume the most time
- **Profile scalability**: Understand how performance changes with input size

## Quick Start (5 Minutes)

The high-level API makes it simple to benchmark a function:

```mojo
from shared.benchmarking import benchmark_function, print_benchmark_report

fn expensive_operation() raises:
    # Your operation here
    var result = 0.0
    for i in range(1000):
        result += Float64(i) * Float64(i)

var stats = benchmark_function(
    expensive_operation,
    warmup_iters=10,
    measure_iters=100
)
print_benchmark_report(stats, "Expensive Operation")
```

Output:

```text
======================================================================
Benchmark Report: Expensive Operation
======================================================================

Configuration:
  Warmup iterations:  10
  Measurement iterations:  100

Latency Statistics (milliseconds):
  Mean:  0.523
  Std Dev:  0.045
  Min:  0.412
  Max:  0.698
  Median (p50):  0.518
  p95:  0.612
  p99:  0.691

Throughput:
  Operations/sec:  1910.91

======================================================================
```

## Benchmarking Framework

### High-Level API

The high-level API handles benchmarking with automatic warmup, measurement, and statistics
computation. Use this for most benchmarking tasks.

#### benchmark_function()

Benchmark a function with a single call:

```mojo
from shared.benchmarking import benchmark_function
from time import perf_counter_ns

fn matrix_multiply() raises:
    # Your matrix operation here
    pass

var stats = benchmark_function(
    matrix_multiply,
    warmup_iters=5,      # Warm up CPU cache and JIT
    measure_iters=50,    # Collect 50 timing measurements
    compute_percentiles=True  # Compute p50, p95, p99
)
```

**Parameters**:

- `func`: Function to benchmark (takes no arguments, returns nothing)
- `warmup_iters`: Number of warmup iterations (default: 10)
- `measure_iters`: Number of measurement iterations (default: 100)
- `compute_percentiles`: Whether to compute percentiles (default: True)

**Returns**: `BenchmarkStatistics` struct with timing data

#### BenchmarkStatistics

The result struct contains comprehensive timing information:

```mojo
struct BenchmarkStatistics:
    var mean_latency_ms: Float64      # Mean execution time
    var std_dev_ms: Float64           # Standard deviation
    var p50_ms: Float64               # 50th percentile (median)
    var p95_ms: Float64               # 95th percentile latency
    var p99_ms: Float64               # 99th percentile latency
    var min_latency_ms: Float64       # Minimum execution time
    var max_latency_ms: Float64       # Maximum execution time
    var throughput: Float64           # Operations per second
    var iterations: Int               # Total measurement iterations
    var warmup_iterations: Int        # Warmup iterations performed
```

**Key Metrics**:

- **Mean**: Average execution time (primary metric)
- **Std Dev**: Variability in execution time (lower is better)
- **Percentiles**: Distribution tail (p95/p99 for tail latency)
- **Throughput**: Operations per second (ops/sec)

#### print_benchmark_report()

Print a formatted benchmark report:

```mojo
from shared.benchmarking import print_benchmark_report

var stats = benchmark_function(my_operation, measure_iters=100)
print_benchmark_report(stats, "My Operation")
```

#### print_benchmark_summary()

Compare multiple operations side-by-side:

```mojo
from shared.benchmarking import print_benchmark_summary

var results = List[BenchmarkStatistics]()
results.append(benchmark_function(op1, measure_iters=100))
results.append(benchmark_function(op2, measure_iters=100))
results.append(benchmark_function(op3, measure_iters=100))

var names = List[String]()
names.append("Operation 1")
names.append("Operation 2")
names.append("Operation 3")

print_benchmark_summary(results, names)
```

Output:

```text
====================================================================================================
Benchmark Summary
====================================================================================================

Operation           Mean (ms)      Std Dev (ms)   P50 (ms)       P95 (ms)       P99 (ms)       Ops/sec
----------------------------------------------------------------------------------------------------
Operation 1         0.523          0.045          0.518          0.612          0.691          1910.91
Operation 2         0.412          0.038          0.408          0.485          0.543          2427.18
Operation 3         0.698          0.062          0.692          0.821          0.891          1432.66
====================================================================================================
```

### Low-Level API

For advanced use cases where you need fine-grained control, use the low-level API to
record individual iteration times.

#### BenchmarkResult

The low-level result tracker records individual iteration times:

```mojo
from shared.benchmarking.result import BenchmarkResult
from time import perf_counter_ns

var result = BenchmarkResult("custom_benchmark", iterations=0)

for _ in range(100):
    var start_ns = Int(perf_counter_ns())
    expensive_operation()
    var end_ns = Int(perf_counter_ns())
    result.record(end_ns - start_ns)

# Query statistics
var mean_ms = result.mean() / 1_000_000.0
var std_ms = result.std() / 1_000_000.0
print("Mean:", mean_ms, "ms")
print("Std Dev:", std_ms, "ms")
```

**Methods**:

- `record(time_ns: Int)` - Record a single iteration time in nanoseconds
- `mean() -> Float64` - Compute mean execution time (nanoseconds)
- `std() -> Float64` - Compute standard deviation (nanoseconds)
- `min_time() -> Float64` - Get minimum iteration time (nanoseconds)
- `max_time() -> Float64` - Get maximum iteration time (nanoseconds)

**Key Implementation Details**:

Uses Welford's algorithm for numerically stable online computation of mean and variance.
This allows efficient computation without storing all measurements, even for millions of
iterations.

### BenchmarkRunner

Advanced runner with manual measurement control:

```mojo
from shared.benchmarking import BenchmarkRunner
from time import perf_counter_ns

var runner = BenchmarkRunner("custom_operation", warmup_iters=10)
runner.run_warmup(lambda: expensive_operation())

for _ in range(100):
    var start_ns = Int(perf_counter_ns())
    expensive_operation()
    var end_ns = Int(perf_counter_ns())
    runner.record_iteration(end_ns - start_ns)

# Query results
print("Mean:", runner.get_mean_ms(), "ms")
print("Std Dev:", runner.get_std_ms(), "ms")
```

## Benchmarking ML Operations

### Basic Timing Pattern

For simple operations, use the high-level API:

```mojo
fn benchmark_relu() raises:
    from shared.benchmarking import benchmark_function

    fn compute_relu() raises:
        var tensor = ExTensor([1024, 1024], DType.float32)
        relu(tensor)

    var stats = benchmark_function(compute_relu, measure_iters=50)
    print_benchmark_report(stats, "ReLU Forward")
```

### Matrix Multiplication with GFLOPS

When benchmarking computationally intensive operations, compute GFLOPS (billions of
floating-point operations per second):

```mojo
fn benchmark_matmul_with_gflops() raises:
    from shared.benchmarking import benchmark_function

    fn compute_matmul() raises:
        var a = ExTensor([512, 512], DType.float32)
        var b = ExTensor([512, 512], DType.float32)
        var c = zeros[DType.float32]([512, 512])
        matmul(a, b, c)

    var stats = benchmark_function(compute_matmul, measure_iters=20)

    # Calculate GFLOPS
    # Matrix multiplication: C = A * B where both are NxN
    # Floating-point operations: 2 * N^3 (N^3 multiply + N^3 add)
    var N = 512.0
    var flops = 2.0 * N * N * N  # Total flops per iteration
    var mean_seconds = stats.mean_latency_ms / 1000.0
    var gflops = (flops / 1e9) / mean_seconds

    print("Matrix Multiplication Performance")
    print("Size: 512x512")
    print("Mean latency:", stats.mean_latency_ms, "ms")
    print("GFLOPS:", gflops)
```

### Comparison Benchmarks

Compare two implementations to measure speedup:

```mojo
fn compare_optimizations() raises:
    from shared.benchmarking import benchmark_function, print_benchmark_summary

    fn baseline_op() raises:
        # Baseline implementation
        pass

    fn optimized_op() raises:
        # Optimized implementation
        pass

    var baseline = benchmark_function(baseline_op, measure_iters=100)
    var optimized = benchmark_function(optimized_op, measure_iters=100)

    # Calculate speedup
    var speedup = baseline.mean_latency_ms / optimized.mean_latency_ms
    print("Speedup:", speedup, "x")

    # Show side-by-side comparison
    var results = List[BenchmarkStatistics]()
    results.append(baseline)
    results.append(optimized)

    var names = List[String]()
    names.append("Baseline")
    names.append("Optimized")

    print_benchmark_summary(results, names)
```

## External Profiling Tools

While the benchmarking framework provides high-resolution timing, external tools can give
additional insights into cache behavior, CPU utilization, and system performance.

### Linux perf

The `perf` tool provides detailed performance analysis:

```bash
# Run your benchmark under perf
perf record -o perf.data mojo run benchmarks/bench_matmul.mojo

# Generate report
perf report
```

For cache statistics:

```bash
# Measure cache hit rate
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
    mojo run benchmarks/bench_matmul.mojo

# Output example:
# Performance counter stats for 'mojo run benchmarks/bench_matmul.mojo':
#
#    15,234,891 cache-references   #   45.2% of all cache refs
#     2,123,456 cache-misses       #   13.9% of cache references
#     8,234,123 LLC-loads
#       456,789 LLC-load-misses    #    5.5% of LL-cache accesses
```

### System Timing

Use the `time` command for quick overall timing:

```bash
# Simple timing
time mojo run benchmarks/bench_matmul.mojo

# Detailed metrics (Linux)
/usr/bin/time -v mojo run benchmarks/bench_matmul.mojo

# Output includes:
# User time (seconds): 5.23
# System time (seconds): 0.12
# Elapsed (wall clock) time: 5.45
# Maximum resident set size (kbytes): 256000
# Page reclaims: 150000
```

## Benchmark Suites

For comprehensive performance testing, create benchmark suites that measure multiple
operations.

### Example: Tensor Operations Suite

```mojo
from shared.benchmarking import benchmark_function, print_benchmark_summary

fn benchmark_tensor_operations() raises:
    var results = List[BenchmarkStatistics]()
    var names = List[String]()

    # Add operation
    fn bench_add() raises:
        var a = ExTensor([1024, 1024], DType.float32)
        var b = ExTensor([1024, 1024], DType.float32)
        var c = zeros[DType.float32]([1024, 1024])
        tensor_add(a, b, c)

    results.append(benchmark_function(bench_add, measure_iters=50))
    names.append("Tensor Add")

    # Multiply operation
    fn bench_mul() raises:
        var a = ExTensor([1024, 1024], DType.float32)
        var b = ExTensor([1024, 1024], DType.float32)
        var c = zeros[DType.float32]([1024, 1024])
        tensor_multiply(a, b, c)

    results.append(benchmark_function(bench_mul, measure_iters=50))
    names.append("Tensor Multiply")

    # ReLU operation
    fn bench_relu() raises:
        var a = ExTensor([1024, 1024], DType.float32)
        relu(a)

    results.append(benchmark_function(bench_relu, measure_iters=50))
    names.append("ReLU")

    # Print comparison
    print_benchmark_summary(results, names)
```

## Memory Profiling

Track memory usage alongside performance metrics:

```mojo
from shared.benchmarking import benchmark_function
from memory import memset_pattern

fn benchmark_with_memory_tracking() raises:
    fn memory_intensive_op() raises:
        var tensor = ExTensor([10000, 10000], DType.float32)
        # Operations on large tensor
        pass

    var stats = benchmark_function(memory_intensive_op, measure_iters=5)

    # Estimate memory usage
    var elements = 10000 * 10000
    var bytes_per_element = 4  # float32
    var memory_mb = Float64(elements * bytes_per_element) / (1024.0 * 1024.0)

    print("Operation:", memory_mb, "MB")
    print("Mean latency:", stats.mean_latency_ms, "ms")
```

To track peak memory:

```bash
# Use system tools
/usr/bin/time -v mojo run your_benchmark.mojo

# Check memory usage during execution
watch -n 0.1 'ps aux | grep mojo'
```

## SIMD Performance Measurement

Verify that SIMD optimizations actually improve performance:

```mojo
fn benchmark_simd_speedup() raises:
    from shared.benchmarking import benchmark_function

    fn scalar_operation() raises:
        var result = 0.0
        for i in range(10000):
            result += Float64(i) * 1.5

    fn simd_operation() raises:
        # SIMD implementation using vector operations
        pass

    var scalar_stats = benchmark_function(scalar_operation, measure_iters=100)
    var simd_stats = benchmark_function(simd_operation, measure_iters=100)

    var speedup = scalar_stats.mean_latency_ms / simd_stats.mean_latency_ms
    print("SIMD Speedup:", speedup, "x")

    # Typical speedups:
    # - 4x for 128-bit SIMD (4 floats)
    # - 8x for 256-bit SIMD (8 floats)
    # - 16x for 512-bit SIMD (16 floats)
```

## Cache Performance

Understanding cache behavior is critical for performance optimization.

### Memory Access Patterns

```mojo
fn benchmark_cache_efficiency() raises:
    from shared.benchmarking import benchmark_function

    fn row_major_access() raises:
        # Efficient: sequential memory access
        var matrix = ExTensor([1024, 1024], DType.float32)
        var sum = 0.0
        for i in range(1024):
            for j in range(1024):
                sum += matrix._get_float64(i * 1024 + j)

    fn column_major_access() raises:
        # Inefficient: scattered memory access
        var matrix = ExTensor([1024, 1024], DType.float32)
        var sum = 0.0
        for j in range(1024):
            for i in range(1024):
                sum += matrix._get_float64(i * 1024 + j)

    var row_major = benchmark_function(row_major_access, measure_iters=20)
    var col_major = benchmark_function(column_major_access, measure_iters=20)

    var cache_impact = col_major.mean_latency_ms / row_major.mean_latency_ms
    print("Cache impact (column/row):", cache_impact, "x slower")
```

### Cache-Aware Blocking

For matrix operations, blocking improves cache reuse:

```mojo
fn benchmark_blocked_matmul() raises:
    from shared.benchmarking import benchmark_function

    fn naive_matmul() raises:
        # Naive: poor cache reuse
        pass

    fn blocked_matmul() raises:
        # Blocked: 64x64 tiles fit in L1 cache
        pass

    var naive = benchmark_function(naive_matmul, measure_iters=5)
    var blocked = benchmark_function(blocked_matmul, measure_iters=5)

    print("Blocking speedup:",
          naive.mean_latency_ms / blocked.mean_latency_ms, "x")
```

## CI/CD Integration

Performance regression detection ensures optimizations persist over time. See
[Issue #2646](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2646) for automated
regression testing setup.

### Baseline Management

Store baseline results for comparison:

```json
{
  "timestamp": "2025-01-13T10:00:00Z",
  "environment": {
    "os": "linux",
    "cpu": "x86_64",
    "mojo_version": "0.26.1"
  },
  "benchmarks": [
    {
      "name": "matmul_512x512",
      "mean_latency_ms": 12.5,
      "std_dev_ms": 0.8,
      "throughput": 8000000.0,
      "iterations": 50
    }
  ]
}
```

### Regression Detection

Compare new results against baselines:

```bash
# Threshold: >10% slowdown triggers alert
new_mean = 13.8  # ms
baseline_mean = 12.5  # ms
regression = (new_mean - baseline_mean) / baseline_mean * 100

# If regression > 10%, alert and block merge
```

## Interpreting Results

### Performance Indicators

**Good Performance Signs**:

- Mean latency consistent across runs (low std dev)
- P95/P99 close to mean (no outliers)
- Speedups match theoretical expectations
- Throughput increasing with optimization level

**Expected Speedups** (from Issue #2588):

| Optimization | Speedup |
| --- | --- |
| Float64 → dtype-specific | 3-5x |
| Dtype-specific → SIMD | 4-8x |
| SIMD → cache-tiled | 2-3x |
| **Total (Naive → Optimized)** | **30-120x** |

### Red Flags

**Warning Signs**:

- High variance (std dev > 20% of mean) → interference, throttling
- Consistent slowdown across all operations → regression
- Speedup < 10% → optimization may not be worth complexity
- Increase in std dev after optimization → added non-determinism

### Troubleshooting

| Issue | Cause | Solution |
| --- | --- | --- |
| High variance | System load | Run in isolation, use affinity |
| No speedup | Optimization ineffective | Profile with perf, check assembly |
| Slowdown | Regression | Bisect commits, check diff |
| Outliers (p99 >> mean) | GC, page faults | Warmup longer, increase iterations |

## Best Practices

### Benchmarking Discipline

1. **Warmup First**: Always include warmup iterations
2. **Enough Samples**: 50+ measurement iterations for statistics
3. **Isolate Operations**: Benchmark single operations, not pipelines
4. **Fix Seeds**: Use seeded randomness for reproducibility
5. **Multiple Runs**: Run benchmarks multiple times to verify stability

### Code Quality

1. **Keep Benchmarks Updated**: Update benchmarks when changing APIs
2. **Document Expectations**: Include expected speedups as comments
3. **Version Tracking**: Note Mojo version and hardware when benchmarking
4. **Commit Results**: Store baselines in version control
5. **Automate Comparison**: Use CI/CD for regression detection

## Examples

See the following for complete benchmarking examples:

- **[benchmarks/bench_matmul.mojo][bench-matmul]** - Progressive optimization
  with GFLOPS tracking
- **[benchmarks/bench_simd.mojo][bench-simd]** - SIMD vs scalar comparison

[bench-matmul]: https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/benchmarks/bench_matmul.mojo
[bench-simd]: https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/benchmarks/bench_simd.mojo

## Related Issues

Performance optimization ongoing work:

- [Issue #2588](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2588) - Matrix
  multiplication optimization (naive → 120x)
- [Issue #2589](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2589) - SIMD
  vectorization opportunities
- [Issue #2590](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2590) - Float64
  conversion overhead (1.5-3x slowdown)
- [Issue #2646](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2646) - CI/CD
  performance regression testing

## References

- [Benchmarking Infrastructure][bench-infra] - Framework
- [Mojo Manual](https://docs.modular.com/mojo/manual/) - Official documentation
- [Welford's Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
  - Online variance computation

[bench-infra]: https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/benchmarks/README.md
