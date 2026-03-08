"""Integration test suite.

Tests for cross-module workflows and component interactions:
- test_training_workflow.mojo  - End-to-end training loops
- test_data_pipeline.mojo      - Data loading workflows
- test_end_to_end.mojo         - Complete model training scenarios
- test_training_e2e.mojo       - Training end-to-end scenarios
- test_multi_precision_training_part1.mojo - Multi-precision training tests (part 1: FP32/FP16/BF16/FP8, overflow, config, scaling)
- test_multi_precision_training_part2.mojo - Multi-precision training tests (part 2: master weights, accuracy, memory, TOML config)
- test_packaging.mojo          - Package and distribution tests
"""

from collections import List


struct TestResult(Copyable):
    """Result of a single test execution.

    Attributes:
        name: Name of the test.
        passed: Whether the test passed.
        duration_ms: Execution time in milliseconds.
        error_message: Error message if test failed (empty if passed).
    """

    var name: String
    var passed: Bool
    var duration_ms: Float64
    var error_message: String

    fn __init__(
        out self,
        name: String,
        passed: Bool,
        duration_ms: Float64 = 0.0,
        error_message: String = "",
    ):
        """Initialize a test result.

        Args:
            name: Name of the test.
            passed: Whether the test passed.
            duration_ms: Execution time in milliseconds.
            error_message: Error message if test failed.
        """
        self.name = name
        self.passed = passed
        self.duration_ms = duration_ms
        self.error_message = error_message


fn run_test_safely[func: fn () raises -> None](name: String) -> TestResult:
    """Run a test function safely, catching any errors.

    Args:
        name: Name of the test for reporting.

    Returns:
        TestResult with success status and any error messages.

    Example:
        ```mojo
        fn test_something() raises:
            var x = 1 + 1
            if x != 2:
                raise Error("Math is broken")

        var result = run_test_safely[test_something]("test_something")
        print(result.name + ": " + ("PASSED" if result.passed else "FAILED"))
        ```
    """
    try:
        func()
        return TestResult(name, True, 0.0, "")
    except e:
        return TestResult(name, False, 0.0, String(e))


fn run_integration_tests() -> Int:
    """Run all integration tests and report results.

    Returns:
        Number of failed tests. Returns 0 if all tests pass.

    Note:
        This is a placeholder that provides the infrastructure for
        running integration tests. Individual test modules (test_*.mojo)
        have their own main() functions and can be run directly.
    """
    var failed_count = 0

    # Note: Integration tests are typically run individually via:
    # mojo test tests/shared/integration/test_*.mojo
    # This function provides the framework for a unified test runner.

    return failed_count


fn main() raises:
    """Test suite initialization for integration tests.

    This serves as the entry point for running integration tests.
    Individual test modules have their own main() functions.
    """
    var failed = run_integration_tests()
    if failed > 0:
        print("Integration tests FAILED: " + String(failed) + " test(s) failed")
    else:
        print("Integration tests PASSED")
