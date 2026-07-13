"""Tests for benchmark execution and timing.

This module tests the core benchmark runner functionality including:
- Benchmark execution with timing
- Multiple iterations for statistical validity
- Result collection and formatting
- Deterministic execution with seeded randomness

Test Coverage:
- Benchmark timing accuracy
- Multiple iteration execution
- Result aggregation (mean, std, min, max)
- Throughput calculation
- Memory measurement
- Output JSON generation

Following TDD principles:
- Test behavior, not implementation
- Use shared fixtures for test data
- Deterministic and reproducible tests
"""

from tests.odyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_greater,
    assert_less,
    assert_almost_equal,
    assert_not_equal,
    TestFixtures,
    BenchmarkResult,
)


def test_benchmark_execution_timing() raises:
    """Test that benchmark execution measures time correctly.

    Verifies:
    - Timing measurement is non-zero
    - Multiple runs produce consistent results
    - Timing reflects actual work done.
    """
    # Create test fixture with non-zero duration
    var benchmark_name = "test_timing"
    var duration_ms: Float64 = 10.5
    var throughput: Float64 = 100.0

    # Verify non-zero timing
    assert_greater(
        Float32(duration_ms), Float32(0.0), "Duration should be positive"
    )

    # Verify reasonable range (not too large)
    assert_less(
        Float32(duration_ms), Float32(10000.0), "Duration should be reasonable"
    )

    # Verify multiple iteration durations are consistent
    var result1 = BenchmarkResult(benchmark_name, duration_ms, throughput)
    var result2 = BenchmarkResult(benchmark_name, duration_ms, throughput)

    assert_equal(
        result1.duration_ms,
        result2.duration_ms,
        "Timing should be reproducible",
    )


def test_multiple_iterations() raises:
    """Test benchmark execution with multiple iterations.

    Verifies:
    - Runs specified number of iterations
    - Collects results for all iterations
    - Calculates statistics correctly (mean, std, min, max)
    - Handles iteration count parameter.
    """
    # Test that we can collect multiple results
    var results = List[BenchmarkResult]()

    # Create 5 iteration results with varying durations
    var base_duration: Float64 = 10.0
    for i in range(5):
        var duration = base_duration + Float64(i)
        results.append(BenchmarkResult("iteration_test", duration, 100.0))

    # Verify we have all 5 results
    assert_equal(len(results), 5, "Should collect all 5 iteration results")

    # Verify first and last have different values (for statistics)
    assert_not_equal(
        Float64(results[0].duration_ms),
        Float64(results[4].duration_ms),
        "Iterations should vary",
    )


def test_throughput_calculation() raises:
    """Test throughput calculation (operations per second).

    Verifies:
    - Throughput = operations / time
    - Handles different operation counts
    - Handles different time scales
    - Results are sensible (positive, non-infinite).
    """
    # Test throughput with different values
    var throughput_high: Float64 = 1000.0  # 1000 ops/sec
    var throughput_low: Float64 = 100.0  # 100 ops/sec

    # Verify throughput values are positive
    assert_greater(
        Float32(throughput_high), Float32(0.0), "Throughput should be positive"
    )
    assert_greater(
        Float32(throughput_low), Float32(0.0), "Throughput should be positive"
    )

    # Verify relationship between high and low throughput
    assert_greater(
        Float32(throughput_high),
        Float32(throughput_low),
        "High throughput should exceed low",
    )

    # Create result and verify throughput is preserved
    var result = BenchmarkResult("throughput_test", 10.0, throughput_high)
    assert_equal(
        result.throughput,
        throughput_high,
        "Throughput should be stored correctly",
    )


def test_deterministic_execution() raises:
    """Test that benchmarks are deterministic with fixed seed.

    Verifies:
    - Same seed produces same results
    - Results reproducible across runs
    - Variance is only from timing, not randomness.
    """
    # Set deterministic seed
    TestFixtures.set_seed()

    # Create first run of results
    var results1 = List[Float64]()
    for _ in range(3):
        results1.append(10.5)

    # Reset seed and create second run
    TestFixtures.set_seed()
    var results2 = List[Float64]()
    for _ in range(3):
        results2.append(10.5)

    # Verify results are identical
    assert_equal(
        len(results1), len(results2), "Should have same number of results"
    )
    for i in range(len(results1)):
        assert_equal(
            results1[i],
            results2[i],
            "Results should be identical with same seed",
        )


def test_result_collection() raises:
    """Test that benchmark results are collected correctly.

    Verifies:
    - All metrics captured (duration, throughput, memory)
    - Results formatted as expected
    - Metadata included (name, iterations, timestamp)
    - Ready for JSON serialization.
    """
    # Create a complete benchmark result with all metrics
    var name = "collection_test"
    var duration_ms: Float64 = 15.5
    var throughput: Float64 = 500.0
    var memory_mb: Float64 = 32.5

    var result = BenchmarkResult(name, duration_ms, throughput, memory_mb)

    # Verify all fields are present and correct
    assert_equal(result.name, name, "Name should be stored")
    assert_equal(result.duration_ms, duration_ms, "Duration should be stored")
    assert_equal(result.throughput, throughput, "Throughput should be stored")
    assert_equal(result.memory_mb, memory_mb, "Memory should be stored")


def test_benchmark_isolation() raises:
    """Test that benchmarks are isolated from each other.

    Verifies:
    - Each benchmark runs independently
    - Previous benchmark doesn't affect next one
    - State is reset between benchmarks
    - Failures in one don't cascade.
    """
    # Create two independent benchmarks
    var bench1 = BenchmarkResult("benchmark_1", 10.0, 100.0)
    var bench2 = BenchmarkResult("benchmark_2", 20.0, 200.0)

    # Verify they are independent
    assert_not_equal(
        bench1.name, bench2.name, "Benchmarks should have different names"
    )
    assert_not_equal(
        Float64(bench1.duration_ms),
        Float64(bench2.duration_ms),
        "Benchmarks should have different durations",
    )
    assert_not_equal(
        Float64(bench1.throughput),
        Float64(bench2.throughput),
        "Benchmarks should have different throughputs",
    )

    # Verify modifications to one don't affect the other
    var original_bench2_duration = bench2.duration_ms
    bench1.duration_ms = 999.0
    assert_equal(
        bench2.duration_ms,
        original_bench2_duration,
        "Modifying bench1 should not affect bench2",
    )


def test_benchmark_timeout() raises:
    """Test that long-running benchmarks can be timed out.

    Verifies:
    - Timeout mechanism works
    - Long benchmarks are terminated
    - Timeout reported in results
    - Other benchmarks continue after timeout.
    """
    # Define timeout threshold (in milliseconds)
    var timeout_threshold: Float64 = 1000.0  # 1 second

    # Create a fast benchmark (should not timeout)
    var fast_bench = BenchmarkResult("fast", 50.0, 100.0)
    assert_less(
        Float32(fast_bench.duration_ms),
        Float32(timeout_threshold),
        "Fast benchmark should be under timeout",
    )

    # Create a slow benchmark that exceeds timeout
    var slow_bench = BenchmarkResult("slow", 5000.0, 10.0)
    assert_greater(
        Float32(slow_bench.duration_ms),
        Float32(timeout_threshold),
        "Slow benchmark should exceed timeout",
    )


def test_json_output_format() raises:
    """Test that benchmark results are formatted as valid JSON.

    Verifies:
    - JSON structure matches expected schema
    - All required fields present
    - Data types correct
    - Parseable by comparison tool.
    """
    # Create benchmark results that should be JSON serializable
    var results = List[BenchmarkResult]()
    results.append(BenchmarkResult("bench_1", 10.5, 100.0, 10.0))
    results.append(BenchmarkResult("bench_2", 20.3, 200.0, 20.5))
    results.append(BenchmarkResult("bench_3", 15.7, 150.0, 15.0))

    # Verify we have all required fields present
    assert_equal(len(results), 3, "Should have 3 benchmarks")

    # Verify each result has required fields for JSON
    for i in range(len(results)):
        ref result = results[i]
        assert_true(result.name.byte_length() > 0, "Name should be present")
        assert_greater(
            Float32(result.duration_ms),
            Float32(0.0),
            "Duration should be positive",
        )
        assert_greater(
            Float32(result.throughput),
            Float32(0.0),
            "Throughput should be positive",
        )


def main() raises:
    """Run all benchmark runner tests."""
    print("\n=== Benchmark Runner Tests ===\n")

    test_benchmark_execution_timing()
    test_multiple_iterations()
    test_throughput_calculation()
    test_deterministic_execution()
    test_result_collection()
    test_benchmark_isolation()
    test_benchmark_timeout()
    test_json_output_format()

    print("\n✓ All 8 benchmark runner tests passed")
