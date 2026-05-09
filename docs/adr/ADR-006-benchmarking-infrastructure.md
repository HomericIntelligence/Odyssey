# ADR-006: Benchmarking Infrastructure

**Status**: Accepted

**Date**: 2025-12-28

**Decision Owner**: Chief Architect

## Executive Summary

This ADR documents ML Odyssey's benchmarking infrastructure for measuring performance of ML
operations. The system provides accurate timing, statistical analysis, percentile computation,
and throughput metrics for performance-critical code paths.

## Context

### Problem Statement

ML frameworks require reliable performance measurement:

1. **Accurate Timing**: Nanosecond precision for fast operations
2. **Statistical Rigor**: Mean, standard deviation, percentiles
3. **Reproducibility**: Warmup to eliminate JIT and cache effects
4. **Comparison**: Side-by-side operation comparison
5. **Regression Detection**: Track performance over time

### Requirements

1. **High-Resolution Timing**: Use platform's best timer (nanosecond precision)
2. **Warmup Phase**: Eliminate cold start variance
3. **Statistical Analysis**: Mean, std dev, min, max, percentiles (p50, p95, p99)
4. **Throughput Metrics**: Operations per second
5. **Reporting**: Formatted output for human review
6. **Backward Compatibility**: Support existing benchmark patterns

## Decision

### Core Benchmarking API

**High-Level API** (`shared/benchmarking/runner.mojo`):

```mojo
fn benchmark_function(
    func: fn() raises -> None,
    warmup_iters: Int = 10,
    measure_iters: Int = 100,
    compute_percentiles: Bool = True,
) raises -> BenchmarkStatistics

struct BenchmarkStatistics:
    var mean_latency_ms: Float64
    var std_dev_ms: Float64
    var p50_ms: Float64
    var p95_ms: Float64
    var p99_ms: Float64
    var min_latency_ms: Float64
    var max_latency_ms: Float64
    var throughput: Float64  # ops/sec
    var iterations: Int
    var warmup_iterations: Int
```

**Low-Level API** (`shared/benchmarking/result.mojo`):

```mojo
struct BenchmarkResult:
    var name: String
    var iterations: Int
    var times: List[Int]  # Nanoseconds per iteration

    fn record(mut self, time_ns: Int)
    fn mean(self) -> Float64
    fn std(self) -> Float64
    fn min_time(self) -> Float64
    fn max_time(self) -> Float64
```

### Timing Implementation

High-resolution timing using platform-specific timers:

```mojo
from time import perf_counter_ns

fn _get_time_ns() -> Int:
    """Platform-specific high-resolution timer.

    - Linux: clock_gettime(CLOCK_MONOTONIC)
    - macOS: mach_absolute_time()
    - Windows: QueryPerformanceCounter()
    """
    return Int(perf_counter_ns())
```

### Warmup Strategy

Warmup iterations eliminate startup variance:

1. **JIT Compilation**: Mojo compiles on first execution
2. **Cache Warming**: CPU caches populated with working set
3. **Branch Prediction**: CPU learns branch patterns
4. **Memory Allocation**: Initial allocation overhead

Default: 10 warmup iterations (configurable)

### Statistical Analysis

**Mean and Standard Deviation**:

```mojo
# Sample variance with N denominator (population variance)
var mean = total / n
var variance = sum((x - mean)^2) / n
var std_dev = sqrt(variance)
```

**Percentile Computation**:

```mojo
# Linear interpolation between sorted values
fn _compute_percentile(data: List[Float64], percentile: Float64) -> Float64:
    var idx = (percentile / 100.0) * (len(data) - 1)
    var lower = data[Int(idx)]
    var upper = data[Int(idx) + 1]
    var frac = idx - Float64(Int(idx))
    return lower + frac * (upper - lower)
```

**Throughput Calculation**:

```mojo
# Operations per second from mean latency in milliseconds
var throughput = 1000.0 / mean_latency_ms
```

### BenchmarkRunner Class

For advanced use cases with fine-grained control:

```mojo
struct BenchmarkRunner:
    var name: String
    var warmup_iters: Int
    var result: BenchmarkResult

    fn run_warmup(mut self, func: fn() raises -> None) raises
    fn record_iteration(mut self, time_ns: Int)
    fn get_mean_ms(self) -> Float64
    fn get_std_ms(self) -> Float64
```

### Reporting Functions

**Single Benchmark Report**:

```mojo
fn print_benchmark_report(result: BenchmarkStatistics, name: String = "Benchmark")

# Output:
# ======================================================================
# Benchmark Report: Matrix Multiplication
# ======================================================================
#
# Configuration:
#   Warmup iterations:  10
#   Measurement iterations:  100
#
# Latency Statistics (milliseconds):
#   Mean:  1.234
#   Std Dev:  0.056
#   Min:  1.150
#   Max:  1.380
#   Median (p50):  1.230
#   p95:  1.320
#   p99:  1.365
#
# Throughput:
#   Operations/sec:  810.37
```

**Multi-Benchmark Summary**:

```mojo
fn print_benchmark_summary(results: List[BenchmarkStatistics], names: List[String])

# Output:
# Operation     Mean (ms)    Std Dev    P50      P95      P99      Ops/sec
# ---------------------------------------------------------------------------
# MatMul        1.234        0.056      1.230    1.320    1.365    810.37
# Conv2D        2.567        0.123      2.550    2.780    2.890    389.56
```

### Legacy API Compatibility

For backward compatibility with existing benchmarks:

```mojo
struct LegacyBenchmarkResult:
    var mean_time_us: Float64  # Microseconds for legacy code
    var std_dev_us: Float64
    # ... other fields in microseconds

fn benchmark_operation(
    name: String,
    operation: fn() raises -> None,
    config: LegacyBenchmarkConfig,
) raises -> LegacyBenchmarkResult
```

## Rationale

### Why Warmup Iterations?

First executions are not representative:

- JIT compilation overhead
- Cold CPU caches
- Memory allocation initialization

10 warmup iterations is typically sufficient for stable measurements.

### Why Percentiles?

Mean and standard deviation don't capture tail latency:

- **p50 (median)**: Typical user experience
- **p95**: Worst 5% of requests
- **p99**: Worst 1% of requests

ML systems often have latency requirements at p95 or p99, not just mean.

### Why Nanosecond Precision?

Many ML operations complete in microseconds:

- SIMD vector operations: < 1 us
- Small tensor operations: 1-100 us
- Layer forward passes: 100 us - 10 ms

Nanosecond precision ensures accurate measurement of fast operations.

### Why Two APIs?

**High-Level API** (`benchmark_function`):

- Simple one-liner benchmarking
- Automatic warmup and statistics
- Best for most use cases

**Low-Level API** (`BenchmarkRunner`, `BenchmarkResult`):

- Fine-grained control over measurement
- Custom timing loops
- Integration with existing code

## Consequences

### Positive

- **Accurate Measurement**: Nanosecond precision timers
- **Statistical Rigor**: Complete statistical analysis
- **Easy to Use**: One-function benchmarking
- **Flexible**: Low-level API for advanced needs
- **Readable Reports**: Formatted output for review

### Negative

- **Two APIs**: Maintenance of both high-level and low-level
- **Memory Overhead**: Stores all timing data for percentiles
- **Sorting Cost**: Percentile computation requires sorting

### Neutral

- **Platform Dependency**: Uses platform-specific timers (handled by Mojo)
- **Legacy Support**: Maintains backward compatibility with microsecond units

## Alternatives Considered

### Alternative 1: External Benchmarking Tool

**Description**: Use external tool like hyperfine for benchmarking.

**Pros**:

- Mature tool with advanced features
- No implementation work

**Cons**:

- Can't benchmark internal functions
- Process startup overhead
- Less control over warmup

**Why Rejected**: Need to benchmark internal Mojo functions.

### Alternative 2: Simple Timer Only

**Description**: Just provide timing, no statistics.

**Pros**:

- Minimal implementation
- Low overhead

**Cons**:

- Users must implement statistics
- No percentiles
- No warmup handling

**Why Rejected**: Statistical analysis is essential for ML benchmarks.

### Alternative 3: Only High-Level API

**Description**: Single `benchmark_function` only.

**Pros**:

- Simpler API surface
- Less code to maintain

**Cons**:

- Can't handle custom measurement loops
- No integration with existing patterns

**Why Rejected**: Need flexibility for various use cases.

## Implementation Details

### File Locations

```text
shared/benchmarking/
├── __init__.mojo       # Package exports
├── runner.mojo         # High-level API, BenchmarkStatistics
└── result.mojo         # Low-level BenchmarkResult

benchmarks/
├── bench_matmul.mojo   # Matrix multiplication benchmarks
├── bench_simd.mojo     # SIMD operation benchmarks
├── reporter.mojo       # Report generation
└── stats.mojo          # Additional statistics

tests/shared/benchmarking/
├── test_runner.mojo    # Runner tests
└── test_result.mojo    # Result tests
```

### Usage Examples

**Basic Benchmarking**:

```mojo
fn expensive_operation() raises:
    var a = AnyTensor.randn([1000, 1000])
    var b = AnyTensor.randn([1000, 1000])
    var c = a.matmul(b)

var result = benchmark_function(expensive_operation, warmup_iters=10, measure_iters=100)
print_benchmark_report(result, "1000x1000 MatMul")
```

**Comparing Operations**:

```mojo
var results = List[BenchmarkStatistics]()
var names = List[String]()

results.append(benchmark_function(matmul_naive))
names.append("Naive MatMul")

results.append(benchmark_function(matmul_simd))
names.append("SIMD MatMul")

print_benchmark_summary(results, names)
```

**Custom Timing Loop**:

```mojo
var runner = BenchmarkRunner("custom_op", warmup_iters=10)
runner.run_warmup(my_operation)

for _ in range(100):
    var start = perf_counter_ns()
    my_operation()
    var end = perf_counter_ns()
    runner.record_iteration(end - start)

print("Mean:", runner.get_mean_ms(), "ms")
```

## References

### Related Files

- `shared/benchmarking/runner.mojo`: High-level API
- `shared/benchmarking/result.mojo`: Low-level API
- `benchmarks/*.mojo`: Benchmark implementations
- `tests/shared/benchmarking/`: Benchmark tests

### Related ADRs

- [ADR-003](ADR-003-memory-pool-architecture.md): Memory pool benchmarks

### External Documentation

- [Mojo Time Module](https://mojolang.org/docs/stdlib/time/)
- [Latency Percentiles](https://bravenewgeek.com/everything-you-know-about-latency-is-wrong/)

## Revision History

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 2025-12-28 | Chief Architect | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-006-benchmarking-infrastructure.md`
- **Status**: Accepted
- **Review Frequency**: As-needed
- **Next Review**: On performance infrastructure changes
- **Supersedes**: None
- **Superseded By**: None
