"""Tests for benchmark runner utilities (Issue #2201).

Tests benchmarking functionality including:
- BenchmarkConfig creation and configuration
- BenchmarkResult structure and statistics
- benchmark_function() with various warmup and iteration counts
- print_benchmark_report() for formatted output
- Percentile computation
- Statistical calculations (mean, std dev)
"""

from testing import assert_true, assert_equal
from shared.benchmarking import (
    BenchmarkResult,
    BenchmarkConfig,
    benchmark_function,
    create_benchmark_config,
    print_benchmark_report,
)


def test_benchmark_config_creation() raises:
    """Test creating benchmark configurations."""
    var config = BenchmarkConfig(
        warmup_iters=5,
        measure_iters=50,
        compute_percentiles=True,
        report_throughput=True,
    )
    assert_equal(config.warmup_iters, 5)
    assert_equal(config.measure_iters, 50)
    assert_true(config.compute_percentiles)
    assert_true(config.report_throughput)
    print("PASS: test_benchmark_config_creation")


def test_create_benchmark_config() raises:
    """Test helper function to create config."""
    var config = create_benchmark_config(warmup_iters=10, measure_iters=100)
    assert_equal(config.warmup_iters, 10)
    assert_equal(config.measure_iters, 100)
    assert_true(config.compute_percentiles)
    assert_true(config.report_throughput)
    print("PASS: test_create_benchmark_config")


def test_create_benchmark_config_defaults() raises:
    """Test default values in config creation."""
    var config = create_benchmark_config()
    assert_equal(config.warmup_iters, 10)
    assert_equal(config.measure_iters, 100)
    assert_true(config.compute_percentiles)
    assert_true(config.report_throughput)
    print("PASS: test_create_benchmark_config_defaults")


def test_benchmark_result_creation() raises:
    """Test creating benchmark result."""
    var result = BenchmarkResult(
        mean_latency_ms=10.5,
        std_dev_ms=2.3,
        p50_ms=10.0,
        p95_ms=15.0,
        p99_ms=18.0,
        min_latency_ms=8.0,
        max_latency_ms=20.0,
        throughput=95.24,
        iterations=100,
        warmup_iterations=10,
    )

    assert_true(result.mean_latency_ms > 10.0 and result.mean_latency_ms < 11.0)
    assert_true(result.std_dev_ms > 2.0 and result.std_dev_ms < 2.5)
    assert_equal(result.p50_ms, 10.0)
    assert_equal(result.p95_ms, 15.0)
    assert_equal(result.p99_ms, 18.0)
    assert_equal(result.min_latency_ms, 8.0)
    assert_equal(result.max_latency_ms, 20.0)
    assert_true(result.throughput > 95.0 and result.throughput < 96.0)
    assert_equal(result.iterations, 100)
    assert_equal(result.warmup_iterations, 10)
    print("PASS: test_benchmark_result_creation")


def test_benchmark_simple_operation() raises:
    """Test benchmarking a simple operation."""

    def simple_operation() raises:
        # Just do some basic computation
        var x = 0
        for i in range(100):
            x += i

    # Benchmark with small iterations for test
    var result = benchmark_function(
        simple_operation, warmup_iters=2, measure_iters=5
    )

    assert_equal(result.iterations, 5)
    assert_equal(result.warmup_iterations, 2)
    assert_true(result.mean_latency_ms >= 0.0)
    assert_true(result.std_dev_ms >= 0.0)
    assert_true(result.min_latency_ms >= 0.0)
    assert_true(result.max_latency_ms >= 0.0)
    assert_true(result.throughput >= 0.0)
    print("PASS: test_benchmark_simple_operation")


def test_benchmark_with_percentiles() raises:
    """Test benchmarking with percentile computation."""

    def simple_operation() raises:
        var x = 0
        for i in range(50):
            x += i

    var result = benchmark_function(
        simple_operation,
        warmup_iters=2,
        measure_iters=5,
        compute_percentiles=True,
    )

    assert_true(result.p50_ms >= 0.0)
    assert_true(result.p95_ms >= 0.0)
    assert_true(result.p99_ms >= 0.0)
    # Percentiles should be ordered
    assert_true(result.p50_ms <= result.p95_ms)
    assert_true(result.p95_ms <= result.p99_ms)
    print("PASS: test_benchmark_with_percentiles")


def test_print_benchmark_report() raises:
    """Test printing benchmark report."""
    var result = BenchmarkResult(
        mean_latency_ms=10.5,
        std_dev_ms=2.3,
        p50_ms=10.0,
        p95_ms=15.0,
        p99_ms=18.0,
        min_latency_ms=8.0,
        max_latency_ms=20.0,
        throughput=95.24,
        iterations=100,
        warmup_iterations=10,
    )

    # Just ensure it doesn't crash
    print_benchmark_report(result, "Test Operation")
    print("PASS: test_print_benchmark_report")


def test_benchmark_config_custom_values() raises:
    """Test creating config with custom values."""
    var config = BenchmarkConfig(
        warmup_iters=20,
        measure_iters=200,
        compute_percentiles=False,
        report_throughput=False,
    )
    assert_equal(config.warmup_iters, 20)
    assert_equal(config.measure_iters, 200)
    assert_true(not config.compute_percentiles)
    assert_true(not config.report_throughput)
    print("PASS: test_benchmark_config_custom_values")


def test_benchmark_result_statistics() raises:
    """Test that benchmark results contain reasonable statistics."""
    # Create a result with realistic values
    var result = BenchmarkResult(
        mean_latency_ms=50.0,
        std_dev_ms=5.0,
        p50_ms=48.0,
        p95_ms=62.0,
        p99_ms=70.0,
        min_latency_ms=40.0,
        max_latency_ms=85.0,
        throughput=20.0,  # 1000 / 50 = 20 ops/sec
        iterations=50,
        warmup_iterations=5,
    )

    # Verify statistical properties
    assert_true(result.min_latency_ms <= result.p50_ms)
    assert_true(result.p50_ms <= result.mean_latency_ms)
    assert_true(result.mean_latency_ms <= result.max_latency_ms)
    assert_true(result.p50_ms <= result.p95_ms)
    assert_true(result.p95_ms <= result.p99_ms)
    print("PASS: test_benchmark_result_statistics")


def test_benchmark_multiple_iterations() raises:
    """Test benchmarking with different iteration counts."""

    def operation() raises:
        var sum = 0
        for i in range(10):
            sum += i

    # Test with different iteration counts
    var result1 = benchmark_function(operation, warmup_iters=1, measure_iters=3)
    var result2 = benchmark_function(
        operation, warmup_iters=5, measure_iters=10
    )

    assert_equal(result1.iterations, 3)
    assert_equal(result1.warmup_iterations, 1)
    assert_equal(result2.iterations, 10)
    assert_equal(result2.warmup_iterations, 5)
    print("PASS: test_benchmark_multiple_iterations")


def main() raises:
    """Run all benchmark tests."""
    print("")
    print("=" * 70)
    print("Benchmarking Framework Unit Tests")
    print("=" * 70)
    print("")

    test_benchmark_config_creation()
    test_create_benchmark_config()
    test_create_benchmark_config_defaults()
    test_benchmark_result_creation()
    test_benchmark_simple_operation()
    test_benchmark_with_percentiles()
    test_print_benchmark_report()
    test_benchmark_config_custom_values()
    test_benchmark_result_statistics()
    test_benchmark_multiple_iterations()

    print("")
    print("=" * 70)
    print("All benchmarking tests passed!")
    print("=" * 70)
    print("")
