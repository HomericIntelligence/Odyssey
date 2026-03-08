# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_result.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for consolidated BenchmarkResult struct (Part 1 of 2).

Tests cover:
- Initialization and basic setup
- Recording individual iteration times
- Mean computation with Welford's algorithm
- Standard deviation calculation
- Min/max tracking
- String formatting
"""

from testing import assert_true, assert_almost_equal
from shared.benchmarking.result import BenchmarkResult


fn test_initialization() raises:
    """Test BenchmarkResult initialization."""
    var result = BenchmarkResult("test_op", iterations=0)

    assert_true(result.name == "test_op", "Name should be set")
    assert_true(result.iterations == 0, "Iterations should start at 0")
    assert_true(result.total_time_ns == 0, "Total time should start at 0")
    assert_true(result.min_time_ns == 0, "Min time should start at 0")
    assert_true(result.max_time_ns == 0, "Max time should start at 0")


fn test_record_single_iteration() raises:
    """Test recording a single iteration."""
    var result = BenchmarkResult("single_test", iterations=0)
    result.record(5000)

    assert_true(result.iterations == 1, "Should have 1 iteration")
    assert_true(result.total_time_ns == 5000, "Total should be 5000")
    assert_true(result.min_time_ns == 5000, "Min should be 5000")
    assert_true(result.max_time_ns == 5000, "Max should be 5000")
    assert_almost_equal(Float64(result.mean()), 5000.0, atol=1e-6)


fn test_record_multiple_iterations() raises:
    """Test recording multiple iterations with constant times."""
    var result = BenchmarkResult("constant_test", iterations=0)

    # Record 10 iterations of 1000 ns each
    for _ in range(10):
        result.record(1000)

    assert_true(result.iterations == 10, "Should have 10 iterations")
    assert_true(result.total_time_ns == 10000, "Total should be 10000")
    assert_true(result.min_time_ns == 1000, "Min should be 1000")
    assert_true(result.max_time_ns == 1000, "Max should be 1000")
    assert_almost_equal(Float64(result.mean()), 1000.0, atol=1e-6)
    assert_almost_equal(Float64(result.std()), 0.0, atol=1e-6)


fn test_mean_calculation() raises:
    """Test mean calculation with varying times."""
    var result = BenchmarkResult("mean_test", iterations=0)

    # Record: 1000, 2000, 3000
    result.record(1000)
    result.record(2000)
    result.record(3000)

    assert_true(result.iterations == 3, "Should have 3 iterations")
    # Mean = (1000 + 2000 + 3000) / 3 = 2000
    assert_almost_equal(Float64(result.mean()), 2000.0, atol=1e-6)


fn test_std_dev_calculation() raises:
    """Test standard deviation with known variance.

    Data: 1000, 2000, 3000
    Mean: 2000
    Deviations: -1000, 0, 1000
    Squared deviations: 1000000, 0, 1000000
    Sample variance (N-1): 2000000 / 2 = 1000000
    Sample std dev: sqrt(1000000) = 1000
    """
    var result = BenchmarkResult("std_test", iterations=0)

    result.record(1000)
    result.record(2000)
    result.record(3000)

    # Expected std dev = 1000.0
    var std = result.std()
    assert_almost_equal(std, 1000.0, atol=1e-5)


fn test_std_dev_zero_iterations() raises:
    """Test std dev returns 0 with no iterations."""
    var result = BenchmarkResult("empty_test", iterations=0)
    assert_almost_equal(Float64(result.std()), 0.0, atol=1e-6)


fn test_std_dev_single_iteration() raises:
    """Test std dev returns 0 with single iteration."""
    var result = BenchmarkResult("single_iter_test", iterations=0)
    result.record(5000)
    assert_almost_equal(Float64(result.std()), 0.0, atol=1e-6)


fn test_min_max_tracking() raises:
    """Test min and max time tracking."""
    var result = BenchmarkResult("minmax_test", iterations=0)

    result.record(5000)
    result.record(1000)
    result.record(8000)
    result.record(3000)

    assert_true(result.min_time_ns == 1000, "Min should be 1000")
    assert_true(result.max_time_ns == 8000, "Max should be 8000")
    assert_almost_equal(Float64(result.min_time()), 1000.0, atol=1e-6)
    assert_almost_equal(Float64(result.max_time()), 8000.0, atol=1e-6)


fn main() raises:
    """Run all tests."""
    test_initialization()
    test_record_single_iteration()
    test_record_multiple_iterations()
    test_mean_calculation()
    test_std_dev_calculation()
    test_std_dev_zero_iterations()
    test_std_dev_single_iteration()
    test_min_max_tracking()

    print("All tests passed!")
