"""Tests for consolidated BenchmarkResult struct.

Tests cover:
- Initialization and basic setup
- Recording individual iteration times
- Mean computation with Welford's algorithm
- Standard deviation calculation
- Min/max tracking
- String formatting
"""


from std.testing import assert_true, assert_almost_equal
from odyssey.benchmarking.result import BenchmarkResult


def test_initialization() raises:
    """Test BenchmarkResult initialization."""
    var result = BenchmarkResult("test_op", iterations=0)

    assert_true(result.name == "test_op", "Name should be set")
    assert_true(result.iterations == 0, "Iterations should start at 0")
    assert_true(result.total_time_ns == 0, "Total time should start at 0")
    assert_true(result.min_time_ns == 0, "Min time should start at 0")
    assert_true(result.max_time_ns == 0, "Max time should start at 0")


def test_record_single_iteration() raises:
    """Test recording a single iteration."""
    var result = BenchmarkResult("single_test", iterations=0)
    result.record(5000)

    assert_true(result.iterations == 1, "Should have 1 iteration")
    assert_true(result.total_time_ns == 5000, "Total should be 5000")
    assert_true(result.min_time_ns == 5000, "Min should be 5000")
    assert_true(result.max_time_ns == 5000, "Max should be 5000")
    assert_almost_equal(Float64(result.mean()), 5000.0, atol=1e-6)


def test_record_multiple_iterations() raises:
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


def test_mean_calculation() raises:
    """Test mean calculation with varying times."""
    var result = BenchmarkResult("mean_test", iterations=0)

    # Record: 1000, 2000, 3000
    result.record(1000)
    result.record(2000)
    result.record(3000)

    assert_true(result.iterations == 3, "Should have 3 iterations")
    # Mean = (1000 + 2000 + 3000) / 3 = 2000
    assert_almost_equal(Float64(result.mean()), 2000.0, atol=1e-6)


def test_std_dev_calculation() raises:
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


def test_std_dev_zero_iterations() raises:
    """Test std dev returns 0 with no iterations."""
    var result = BenchmarkResult("empty_test", iterations=0)
    assert_almost_equal(Float64(result.std()), 0.0, atol=1e-6)


def test_std_dev_single_iteration() raises:
    """Test std dev returns 0 with single iteration."""
    var result = BenchmarkResult("single_iter_test", iterations=0)
    result.record(5000)
    assert_almost_equal(Float64(result.std()), 0.0, atol=1e-6)


def test_min_max_tracking() raises:
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


def test_large_iteration_count() raises:
    """Test with many iterations (stress test for numerical stability)."""
    var result = BenchmarkResult("large_test", iterations=0)

    # Record 1000 iterations with slight variation
    for i in range(1000):
        var time = 10000 + (i % 100)
        result.record(time)

    assert_true(result.iterations == 1000, "Should have 1000 iterations")

    # Mean should be around 10049.5
    var mean = result.mean()
    assert_almost_equal(mean, 10049.5, atol=1.0)

    # Min should be 10000, max should be 10099
    assert_true(result.min_time_ns == 10000, "Min should be 10000")
    assert_true(result.max_time_ns == 10099, "Max should be 10099")


def test_string_representation() raises:
    """Test string formatting of results."""
    var result = BenchmarkResult("format_test", iterations=0)

    result.record(1000)
    result.record(2000)
    result.record(3000)

    var result_str = result.__str__()

    # Check that key information is in the string
    assert_true(
        "format_test" in result_str, "String should contain operation name"
    )
    assert_true(
        "Iterations:" in result_str, "String should contain iterations label"
    )
    assert_true("Mean:" in result_str, "String should contain mean label")
    assert_true("Std Dev:" in result_str, "String should contain std dev label")
    assert_true("Min:" in result_str, "String should contain min label")
    assert_true("Max:" in result_str, "String should contain max label")


def test_zero_times() raises:
    """Test recording zero nanosecond times."""
    var result = BenchmarkResult("zero_test", iterations=0)

    result.record(0)
    result.record(0)
    result.record(0)

    assert_true(result.mean() == 0.0, "Mean should be 0")
    assert_true(result.min_time() == 0.0, "Min should be 0")
    assert_true(result.max_time() == 0.0, "Max should be 0")


def test_welford_numerical_stability() raises:
    """Test Welford's algorithm numerical stability with different magnitude changes.

    This tests that the algorithm maintains precision even when adding
    very different magnitude values sequentially.
    """
    var result = BenchmarkResult("stability_test", iterations=0)

    # Add small values first
    for _ in range(100):
        result.record(100)

    # Then add a much larger value
    result.record(10000)

    # Then more small values
    for _ in range(100):
        result.record(100)

    # Mean should be approximately (100*100 + 10000 + 100*100) / 201
    # = (10000 + 10000 + 10000) / 201 = 30000 / 201 ≈ 149.25
    var expected_mean = Float64(30000) / 201.0
    var actual_mean = result.mean()

    assert_almost_equal(actual_mean, expected_mean, atol=0.1)


def test_sequential_recording() raises:
    """Test that sequential recording produces correct statistics."""
    var result = BenchmarkResult("sequential_test", iterations=0)

    # Record 1-10 in sequence: 1, 2, 3, ..., 10
    for i in range(1, 11):
        result.record(i * 1000)

    # Mean of 1000..10000 = 5500
    assert_almost_equal(Float64(result.mean()), 5500.0, atol=1e-5)

    # Sample std dev calculation:
    # Deviations from mean: -4500, -3500, -2500, -1500, -500, 500, 1500, 2500, 3500, 4500
    # Sum of squared deviations: 2*4500^2 + 2*3500^2 + 2*2500^2 + 2*1500^2 + 2*500^2
    # = 2*(20250000 + 12250000 + 6250000 + 2250000 + 250000)
    # = 2*41250000 = 82500000
    # Sample variance = 82500000 / 9 ≈ 9166666.67
    # Sample std = sqrt(9166666.67) ≈ 3027.65
    var expected_std = 3027.65
    var actual_std = result.std()
    assert_almost_equal(actual_std, expected_std, atol=50.0)


def test_mean_getter_with_no_records() raises:
    """Test mean() returns 0 when no iterations recorded."""
    var result = BenchmarkResult("empty_mean_test", iterations=0)
    assert_almost_equal(Float64(result.mean()), 0.0, atol=1e-6)


def test_min_getter_with_no_records() raises:
    """Test min_time() returns 0 when no iterations recorded."""
    var result = BenchmarkResult("empty_min_test", iterations=0)
    assert_almost_equal(Float64(result.min_time()), 0.0, atol=1e-6)


def test_max_getter_with_no_records() raises:
    """Test max_time() returns 0 when no iterations recorded."""
    var result = BenchmarkResult("empty_max_test", iterations=0)
    assert_almost_equal(Float64(result.max_time()), 0.0, atol=1e-6)


def main() raises:
    """Run all test_result tests."""
    print("Running test_result tests...")

    test_initialization()
    print("✓ test_initialization")

    test_record_single_iteration()
    print("✓ test_record_single_iteration")

    test_record_multiple_iterations()
    print("✓ test_record_multiple_iterations")

    test_mean_calculation()
    print("✓ test_mean_calculation")

    test_std_dev_calculation()
    print("✓ test_std_dev_calculation")

    test_std_dev_zero_iterations()
    print("✓ test_std_dev_zero_iterations")

    test_std_dev_single_iteration()
    print("✓ test_std_dev_single_iteration")

    test_min_max_tracking()
    print("✓ test_min_max_tracking")

    test_large_iteration_count()
    print("✓ test_large_iteration_count")

    test_string_representation()
    print("✓ test_string_representation")

    test_zero_times()
    print("✓ test_zero_times")

    test_welford_numerical_stability()
    print("✓ test_welford_numerical_stability")

    test_sequential_recording()
    print("✓ test_sequential_recording")

    test_mean_getter_with_no_records()
    print("✓ test_mean_getter_with_no_records")

    test_min_getter_with_no_records()
    print("✓ test_min_getter_with_no_records")

    test_max_getter_with_no_records()
    print("✓ test_max_getter_with_no_records")

    print("\nAll test_result tests passed!")
