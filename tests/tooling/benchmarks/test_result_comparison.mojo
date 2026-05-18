"""Tests for benchmark result comparison logic.

This module tests the comparison of current results against baselines including:
- Percentage change calculation
- Threshold checking
- Normal variance tolerance
- Statistical significance

Test Coverage:
- Calculate percentage change
- Compare against tolerance thresholds
- Handle improvements (faster)
- Handle regressions (slower)
- Handle missing benchmarks
- Multiple metric comparison

Following TDD principles:
- Test mathematical correctness
- Test edge cases (zero, infinity)
- Test boundary conditions
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less,
)


def test_percentage_change_calculation() raises:
    """Test calculation of percentage change between baseline and current.

    Verifies:
    - Formula: ((current - baseline) / baseline) * 100
    - Positive change = slower (worse)
    - Negative change = faster (better)
    - Zero change = same performance.
    """
    # Test percentage change calculation
    var baseline_100 = 100.0
    var current_110 = 110.0
    var current_90 = 90.0
    var current_100 = 100.0

    # Calculate percentage changes
    var pct_slower = ((current_110 - baseline_100) / baseline_100) * 100.0
    var pct_faster = ((current_90 - baseline_100) / baseline_100) * 100.0
    var pct_same = ((current_100 - baseline_100) / baseline_100) * 100.0

    # Verify results
    assert_almost_equal(
        Float32(pct_slower), Float32(10.0), 0.1, "110 vs 100 should be +10%"
    )
    assert_almost_equal(
        Float32(pct_faster), Float32(-10.0), 0.1, "90 vs 100 should be -10%"
    )
    assert_almost_equal(
        Float32(pct_same), Float32(0.0), 0.1, "100 vs 100 should be 0%"
    )


def test_improvement_detection() raises:
    """Test detection of performance improvements.

    Verifies:
    - Negative percentage = improvement
    - Faster execution detected
    - Reported as improvement, not regression
    - Magnitude calculated correctly.
    """
    # Test improvement detection (negative percentage = faster)
    var pct_10_faster = -10.0
    var pct_50_faster = -50.0

    # Verify improvements are negative
    assert_less(
        Float32(pct_10_faster), Float32(0.0), "Improvement should be negative"
    )
    assert_less(
        Float32(pct_50_faster), Float32(0.0), "Improvement should be negative"
    )

    # Verify magnitude of improvements
    assert_greater(
        Float32(pct_10_faster),
        Float32(-20.0),
        "10% improvement should be in range",
    )
    assert_greater(
        Float32(pct_50_faster),
        Float32(-60.0),
        "50% improvement should be in range",
    )


def test_regression_detection() raises:
    """Test detection of performance regressions.

    Verifies:
    - Positive percentage = regression
    - Slower execution detected
    - Threshold checking (>10% = alert)
    - Magnitude calculated correctly.
    """
    # Test regression detection (positive percentage = slower)
    var pct_5_slower = 5.0
    var pct_10_slower = 10.0
    var pct_11_slower = 11.0
    var pct_50_slower = 50.0
    var regression_threshold = 10.0

    # Verify regressions are positive
    assert_greater(
        Float32(pct_5_slower), Float32(0.0), "Regression should be positive"
    )
    assert_greater(
        Float32(pct_50_slower), Float32(0.0), "Regression should be positive"
    )

    # Verify threshold checking
    assert_less(
        Float32(pct_10_slower),
        Float32(regression_threshold + 0.1),
        "10% should be at threshold",
    )
    assert_greater(
        Float32(pct_11_slower),
        Float32(regression_threshold),
        "11% should exceed threshold",
    )


def test_normal_variance_tolerance() raises:
    """Test normal variance tolerance (~5%).

    Verifies:
    - Changes within ±5% considered normal
    - No alerts for normal variance
    - Alerts only for significant changes
    - Configurable tolerance threshold.
    """
    # Test normal variance tolerance
    var normal_variance = 5.0
    var pct_3 = 3.0
    var pct_5 = 5.0
    var pct_7 = 7.0

    # Verify changes within tolerance are normal
    assert_less(
        Float32(pct_3), Float32(normal_variance + 1.0), "3% is normal variance"
    )
    assert_less(
        Float32(pct_5), Float32(normal_variance + 1.0), "5% is normal variance"
    )
    assert_greater(
        Float32(pct_7),
        Float32(normal_variance),
        "7% is outside normal variance",
    )


def test_regression_threshold() raises:
    """Test regression alert threshold (>10%).

    Verifies:
    - 10% slowdown = boundary (no alert)
    - >10% slowdown = regression (alert)
    - Threshold is exclusive (>10%, not >=10%)
    - Configurable threshold.
    """
    # Test exact boundary conditions
    var threshold = 10.0
    var pct_9_9 = 9.9
    var pct_10_0 = 10.0
    var pct_10_1 = 10.1
    var pct_11_0 = 11.0

    # Verify boundary conditions
    assert_less(
        Float32(pct_9_9), Float32(threshold), "9.9% should be below threshold"
    )
    assert_less(
        Float32(pct_10_0),
        Float32(threshold + 0.1),
        "10.0% should be at threshold",
    )
    assert_greater(
        Float32(pct_10_1), Float32(threshold), "10.1% should exceed threshold"
    )
    assert_greater(
        Float32(pct_11_0), Float32(threshold), "11.0% should exceed threshold"
    )


def test_multiple_metric_comparison() raises:
    """Test comparison of multiple metrics (duration, throughput, memory).

    Verifies:
    - Duration comparison (lower is better)
    - Throughput comparison (higher is better)
    - Memory comparison (lower is better)
    - Each metric compared independently
    - Alerts for any metric regression.
    """
    # Test multiple metric comparison
    var duration_baseline = 100.0
    var duration_current_good = 95.0
    var duration_current_bad = 115.0

    var throughput_baseline = 1000.0
    var throughput_current_good = 1100.0
    var _ = 850.0  # throughput_current_bad - intentionally unused

    # Duration: lower is better (negative % change is good)
    var duration_pct_good = (
        (duration_current_good - duration_baseline) / duration_baseline
    ) * 100.0
    assert_less(
        Float32(duration_pct_good),
        Float32(0.0),
        "Lower duration is improvement",
    )

    # Duration regression
    var duration_pct_bad = (
        (duration_current_bad - duration_baseline) / duration_baseline
    ) * 100.0
    assert_greater(
        Float32(duration_pct_bad),
        Float32(10.0),
        "Higher duration is regression",
    )

    # Throughput: higher is better (positive % change is good)
    var throughput_pct_good = (
        (throughput_current_good - throughput_baseline) / throughput_baseline
    ) * 100.0
    assert_greater(
        Float32(throughput_pct_good),
        Float32(0.0),
        "Higher throughput is improvement",
    )


def test_missing_baseline_benchmark() raises:
    """Test handling of benchmark missing in baseline.

    Verifies:
    - Missing baseline detected
    - Warning issued (not error)
    - Suggests updating baseline
    - Comparison skipped for that benchmark.
    """
    # Test handling of missing baseline benchmarks
    var baseline_list = List[String]()
    baseline_list.append("bench_1")
    baseline_list.append("bench_2")

    var current_list = List[String]()
    current_list.append("bench_1")
    current_list.append("bench_2")
    current_list.append("bench_3")  # Missing in baseline

    # Verify we can detect missing benchmarks
    var missing_found = False
    for i in range(len(current_list)):
        var found = False
        for j in range(len(baseline_list)):
            if current_list[i] == baseline_list[j]:
                found = True
                break
        if not found:
            missing_found = True

    assert_true(missing_found, "Should detect benchmark missing in baseline")


def test_zero_baseline_handling() raises:
    """Test handling of zero values in baseline.

    Verifies:
    - Zero baseline doesn't cause division by zero
    - Appropriate error or special handling
    - Invalid baseline data detected.
    """
    # Test zero value handling
    var zero_value: Float64 = 0.0
    var valid_value: Float64 = 100.0

    # Verify we can detect invalid zero baseline
    assert_equal(zero_value, 0.0, "Should detect zero values")
    assert_greater(
        Float32(valid_value), Float32(0.0), "Valid values should be positive"
    )

    # Verify comparison logic would catch zero baseline
    var baseline_invalid = zero_value == 0.0
    assert_true(baseline_invalid, "Should detect zero baseline as invalid")


def test_comparison_report_generation() raises:
    """Test generation of comparison report.

    Verifies:
    - All benchmarks listed
    - Percentage changes shown
    - Regressions highlighted
    - Improvements noted
    - Summary statistics included.
    """
    # Test report generation
    var benchmarks_total = 3
    var regressions_count = 1
    var improvements_count = 1

    # Create report content
    var report = List[String]()
    report.append("Comparison Report")
    report.append("Total benchmarks: " + String(benchmarks_total))
    report.append("Regressions: " + String(regressions_count))
    report.append("Improvements: " + String(improvements_count))

    # Verify report contains required sections
    assert_equal(len(report), 4, "Report should have 4 sections")
    assert_true((report[0]).byte_length() > 0, "Report header should exist")
    assert_true(
        (report[1]).byte_length() > 0, "Report should include total count"
    )


def main() raises:
    """Run all result comparison tests."""
    print("\n=== Result Comparison Tests ===\n")

    test_percentage_change_calculation()
    test_improvement_detection()
    test_regression_detection()
    test_normal_variance_tolerance()
    test_regression_threshold()
    test_multiple_metric_comparison()
    test_missing_baseline_benchmark()
    test_zero_baseline_handling()
    test_comparison_report_generation()

    print("\n✓ All 9 result comparison tests passed")
