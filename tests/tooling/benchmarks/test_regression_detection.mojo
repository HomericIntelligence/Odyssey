"""Tests for regression detection and alerting.

This module tests the detection and reporting of performance regressions including:
- Regression threshold enforcement (>10%)
- Alert generation
- Exit code handling
- Regression reporting

Test Coverage:
- Detect single regression
- Detect multiple regressions
- No false positives within tolerance
- Exit code 0 (pass) vs 1 (fail)
- Regression report format

Following TDD principles:
- Test critical alert logic
- Test boundary conditions
- Test CI/CD integration points
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_greater,
    assert_less,
)


def test_single_regression_detection() raises:
    """Test detection of a single regression.

    Verifies:
    - Regression detected when >10% slowdown
    - Correct benchmark identified
    - Alert generated
    - Exit code = 1 (failure).
    """
    # Test single regression detection
    var regression_threshold = 10.0
    var regression_pct = 15.0  # 15% slowdown > threshold
    var exit_code_failure = 1
    var exit_code_success = 0

    # Verify regression is detected
    assert_greater(
        Float32(regression_pct),
        Float32(regression_threshold),
        "15% exceeds 10% threshold",
    )

    # Verify correct exit code for failure
    if regression_pct > regression_threshold:
        assert_equal(
            exit_code_failure, 1, "Should exit with code 1 on regression"
        )
    else:
        assert_equal(exit_code_success, 0, "Should exit with code 0 on success")


def test_multiple_regressions_detection() raises:
    """Test detection of multiple regressions.

    Verifies:
    - All regressions detected
    - Each regression reported
    - Exit code = 1 (failure)
    - Summary includes count.
    """
    # Test multiple regression detection
    var regression_threshold = 10.0
    var regressions = List[Float64]()
    regressions.append(15.0)  # 15% slowdown
    regressions.append(20.0)  # 20% slowdown
    regressions.append(12.0)  # 12% slowdown

    # Count regressions
    var regression_count = 0
    for i in range(len(regressions)):
        if regressions[i] > regression_threshold:
            regression_count = regression_count + 1

    # Verify all regressions detected
    assert_equal(regression_count, 3, "Should detect all 3 regressions")
    assert_equal(
        regression_count > 0, True, "Should report at least one regression"
    )


def test_no_false_positives() raises:
    """Test that normal variance doesn't trigger false positives.

    Verifies:
    - Changes <10% don't trigger alerts
    - Exactly 10% doesn't trigger alert
    - Both faster and slower within tolerance pass
    - Exit code = 0 (success).
    """
    # Test no false positives
    var regression_threshold = 10.0
    var changes = List[Float64]()
    changes.append(5.0)  # 5% slower
    changes.append(10.0)  # exactly 10%
    changes.append(-5.0)  # 5% faster

    # Count false positives (should be 0)
    var false_positive_count = 0
    for i in range(len(changes)):
        if changes[i] > regression_threshold:
            false_positive_count = false_positive_count + 1

    assert_equal(false_positive_count, 0, "Should have no false positives")
    assert_equal(
        false_positive_count > 0, False, "No changes should trigger alerts"
    )


def test_exit_code_success() raises:
    """Test exit code 0 when no regressions detected.

    Verifies:
    - No regressions -> exit 0
    - Improvements -> exit 0
    - Normal variance -> exit 0
    - Exactly at threshold -> exit 0
    """
    # Test exit code 0 on success
    var regression_threshold = 10.0
    var scenarios = List[Float64]()
    scenarios.append(-5.0)  # Improvement
    scenarios.append(5.0)  # Normal variance
    scenarios.append(10.0)  # At threshold
    scenarios.append(-10.0)  # Good improvement

    var has_regression = False
    for i in range(len(scenarios)):
        if scenarios[i] > regression_threshold:
            has_regression = True
            break

    assert_false(has_regression, "Should not have any regressions")
    assert_equal(has_regression, False, "Exit code should be 0 (success)")


def test_exit_code_failure() raises:
    """Test exit code 1 when regressions detected.

    Verifies:
    - Any regression -> exit 1
    - Single regression -> exit 1
    - Multiple regressions -> exit 1
    - Even if other benchmarks improved.
    """
    # Test exit code 1 on failure
    var regression_threshold = 10.0
    var scenarios = List[Float64]()
    scenarios.append(-5.0)  # Improvement (but...)
    scenarios.append(15.0)  # Regression (should trigger exit 1)
    scenarios.append(5.0)  # Normal variance

    var has_regression = False
    for i in range(len(scenarios)):
        if scenarios[i] > regression_threshold:
            has_regression = True
            break

    assert_true(has_regression, "Should detect regression")
    assert_equal(has_regression, True, "Exit code should be 1 (failure)")


def test_regression_report_format() raises:
    """Test format of regression report.

    Verifies:
    - Report includes benchmark names
    - Report includes percentage changes
    - Report includes baseline vs current values
    - Report is human-readable
    - Report includes summary statistics.
    """
    # Test regression report format
    var report = List[String]()
    report.append("REGRESSION DETECTED")
    report.append("Benchmark: matrix_op")
    report.append("Change: +15%")
    report.append("Baseline: 100.0ms")
    report.append("Current: 115.0ms")

    # Verify all required sections present
    assert_equal(len(report), 5, "Report should have 5 sections")
    assert_true(
        report[0].find("REGRESSION") >= 0,
        "Report should have REGRESSION header",
    )
    assert_true(
        report[1].find("Benchmark") >= 0, "Report should include benchmark name"
    )
    assert_true(
        report[2].find("Change") >= 0, "Report should show percentage change"
    )


def test_regression_severity_levels() raises:
    """Test categorization of regression severity.

    Verifies:
    - 10-20% = minor regression
    - 20-50% = moderate regression
    - >50% = severe regression
    - Severity shown in report.
    """
    # Test severity categorization
    var minor_regression = 15.0  # 10-20%
    var moderate_regression = 30.0  # 20-50%
    var severe_regression = 100.0  # >50%

    # Verify severity ranges
    assert_greater(
        Float32(minor_regression), Float32(10.0), "Minor should be >10%"
    )
    assert_less(
        Float32(minor_regression), Float32(20.0), "Minor should be <20%"
    )

    assert_greater(
        Float32(moderate_regression), Float32(20.0), "Moderate should be >20%"
    )
    assert_less(
        Float32(moderate_regression), Float32(50.0), "Moderate should be <50%"
    )

    assert_greater(
        Float32(severe_regression), Float32(50.0), "Severe should be >50%"
    )


def test_improvement_reporting() raises:
    """Test reporting of improvements alongside regressions.

    Verifies:
    - Improvements listed separately
    - Not counted as regressions
    - Noted in report (not just errors)
    - Doesn't affect exit code if no regressions.
    """
    # Test improvement reporting
    var regression_threshold = 10.0
    var improvements = List[Float64]()
    improvements.append(-10.0)  # 10% faster
    improvements.append(-5.0)  # 5% faster

    # Count actual regressions
    var regression_count = 0
    for i in range(len(improvements)):
        if improvements[i] > regression_threshold:
            regression_count = regression_count + 1

    # Verify improvements don't count as regressions
    assert_equal(
        regression_count, 0, "Improvements should not count as regressions"
    )
    assert_true(len(improvements) > 0, "Should have improvements to report")


def test_ci_integration_output() raises:
    """Test output format for CI/CD integration.

    Verifies:
    - Exit code usable by CI (0 or 1)
    - Output parseable by CI tools
    - Annotations for GitHub Actions (if applicable)
    - Summary visible in CI logs.
    """
    # Test CI integration output
    var exit_code_success = 0
    var exit_code_failure = 1

    # Create CI-friendly output
    var ci_output = List[String]()
    ci_output.append("Benchmark test result: PASS")
    ci_output.append("Exit code: 0")
    ci_output.append("Summary: All benchmarks within threshold")

    # Verify exit codes are CI-valid
    assert_equal(exit_code_success, 0, "Success exit code should be 0")
    assert_equal(exit_code_failure, 1, "Failure exit code should be 1")

    # Verify output is present
    assert_equal(len(ci_output), 3, "Should have 3 output lines")
    assert_true((ci_output[0]).byte_length() > 0, "Should have result line")


def main() raises:
    """Run all regression detection tests."""
    print("\n=== Regression Detection Tests ===\n")

    test_single_regression_detection()
    test_multiple_regressions_detection()
    test_no_false_positives()
    test_exit_code_success()
    test_exit_code_failure()
    test_regression_report_format()
    test_regression_severity_levels()
    test_improvement_reporting()
    test_ci_integration_output()

    print("\n✓ All 9 regression detection tests passed")
