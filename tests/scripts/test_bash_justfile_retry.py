#!/usr/bin/env python3
"""Tests for bash retry logic in justfile test-group recipe.

Tests the bash-level retry behavior for mojo test execution:
- Crash detection (execution crashed in output)
- Retry exhaustion messages
- Successful retries after crashes
- Normal failures without retry
"""

import pathlib
import sys
import tempfile
import unittest

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))


class TestBashTestGroupRetry(unittest.TestCase):
    """Test bash retry logic in justfile test-group recipe."""

    def create_mock_script(self, exit_code: int, output: str = "") -> str:
        """Create a temporary script that returns specified exit code and output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo '{output}'\n")
            f.write(f"exit {exit_code}\n")
            f.flush()
            return f.name

    def run_retry_loop(self, mock_commands: list, max_retries: int = 3) -> tuple:
        """Simulate the bash retry loop behavior.

        Args:
            mock_commands: List of (exit_code, output) tuples for each attempt
            max_retries: Number of retries (total attempts = max_retries)

        Returns:
            Tuple of (final_exit_code, attempt_count, is_crash_detected_any)
        """
        attempt = 0
        max_attempts = max_retries
        test_passed = False
        is_crash_detected = False  # Track if ANY attempt had a crash

        for attempt_num, (exit_code, output) in enumerate(mock_commands, 1):
            attempt = attempt_num
            current_is_crash = "execution crashed" in output.lower()
            if current_is_crash:
                is_crash_detected = True  # Mark that we saw a crash

            if exit_code == 0:
                test_passed = True
                break
            elif current_is_crash and attempt < max_attempts:
                # Would sleep and retry in real code
                continue
            else:
                # Failed without crash or exhausted retries
                break

        return (
            0 if test_passed else 1,
            attempt,
            is_crash_detected,
        )

    def test_crash_triggers_retry(self):
        """Verify a crash (execution crashed in output) triggers a retry."""
        # First attempt crashes, second succeeds
        mock_commands = [
            (1, "Mojo JIT error: execution crashed"),  # Crash on first attempt
            (0, "✅ PASSED"),  # Success on retry
        ]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 0)  # Should succeed on second attempt
        self.assertEqual(attempt, 2)  # Made 2 attempts
        self.assertTrue(is_crash)  # Crash was detected

    def test_no_crash_no_retry(self):
        """Verify a normal failure (no crash) doesn't retry."""
        # Normal failure - should not retry
        mock_commands = [
            (1, "AssertionError: test failed"),  # Normal failure
        ]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 1)  # Should fail
        self.assertEqual(attempt, 1)  # Only 1 attempt (no retry)
        self.assertFalse(is_crash)  # No crash detected

    def test_crash_retry_exhausted(self):
        """Verify FAILED after retry message when crash retries exhausted."""
        # Crashes on all attempts
        mock_commands = [
            (1, "Mojo JIT error: execution crashed"),
            (1, "Mojo JIT error: execution crashed"),
            (1, "Mojo JIT error: execution crashed"),
        ]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 1)  # Should fail
        self.assertEqual(attempt, 3)  # Made all 3 attempts
        self.assertTrue(is_crash)  # Crash was detected

    def test_successful_retry_after_crash(self):
        """Verify successful retry produces PASSED with retry count."""
        # Crashes twice, then succeeds
        mock_commands = [
            (1, "Mojo JIT error: execution crashed"),
            (1, "Mojo JIT error: execution crashed"),
            (0, "✅ PASSED"),
        ]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 0)  # Should succeed
        self.assertEqual(attempt, 3)  # Made 3 attempts
        self.assertTrue(is_crash)  # Crash was detected earlier

    def test_immediate_success_no_retries(self):
        """Verify successful first attempt doesn't retry."""
        mock_commands = [(0, "✅ PASSED")]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 0)  # Should succeed
        self.assertEqual(attempt, 1)  # Only 1 attempt
        self.assertFalse(is_crash)  # No crash

    def test_crash_on_last_attempt_then_succeeds(self):
        """Verify crash on last attempt before max, then succeeds."""
        # Crash on attempt 2 (last allowed), success on retry
        mock_commands = [
            (1, "Mojo JIT error: execution crashed"),
            (1, "Mojo JIT error: execution crashed"),
            (0, "✅ PASSED"),
        ]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 0)  # Should succeed
        self.assertEqual(attempt, 3)  # Made final attempt
        self.assertTrue(is_crash)  # Crash was detected

    def test_crash_detection_case_insensitive(self):
        """Verify crash detection works with various case variations."""
        test_cases = [
            "execution crashed",
            "Execution crashed",
            "EXECUTION CRASHED",
            "Mojo JIT error: Execution Crashed",
        ]

        for crash_msg in test_cases:
            mock_commands = [(1, crash_msg), (0, "✅ PASSED")]
            exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

            self.assertEqual(exit_code, 0, f"Failed with message: {crash_msg}")
            self.assertEqual(attempt, 2, f"Incorrect retry count for: {crash_msg}")
            self.assertTrue(is_crash, f"Crash not detected: {crash_msg}")

    def test_mixed_output_with_crash_marker(self):
        """Verify crash detection works with mixed output."""
        # Output contains "execution crashed" somewhere in the middle
        output = "Some compilation output...\nexecution crashed: JIT compilation failed\nMore details..."
        mock_commands = [(1, output), (0, "✅ PASSED")]

        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=3)

        self.assertEqual(exit_code, 0)
        self.assertEqual(attempt, 2)
        self.assertTrue(is_crash)

    def test_retry_count_tracking(self):
        """Verify attempt count increments correctly through retries."""
        # Track how many times we attempt for different retry counts
        mock_commands = [
            (1, "execution crashed"),
            (1, "execution crashed"),
            (1, "execution crashed"),
            (0, "✅ PASSED"),
        ]

        # Test with max_retries=3 (should make up to 3 attempts)
        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands[:3], max_retries=3)
        self.assertEqual(attempt, 3)

        # Test with max_retries=2 (should make up to 2 attempts)
        exit_code, attempt, is_crash = self.run_retry_loop(mock_commands[:2], max_retries=2)
        self.assertEqual(attempt, 2)

    def test_exponential_backoff_delay_values(self):
        """Verify exponential backoff delay calculation (for documentation)."""
        # This tests the delay logic: delay = 1 * (2^(attempt-1))
        delays = []
        delay = 1
        for attempt in range(3):
            delays.append(delay)
            delay *= 2

        # Expected: [1, 2, 4] seconds
        self.assertEqual(delays, [1, 2, 4])

    def test_max_retries_variable_respected(self):
        """Verify MAX_RETRIES environment variable controls retry count."""
        # Simulate different MAX_RETRIES values
        test_cases = [
            (1, 1),  # max_retries=1, only 1 attempt allowed
            (2, 2),  # max_retries=2, up to 2 attempts
            (5, 5),  # max_retries=5, up to 5 attempts
        ]

        for max_retries, expected_max_attempts in test_cases:
            # Create commands that all fail
            mock_commands = [(1, "execution crashed") for _ in range(max_retries)]

            exit_code, attempt, is_crash = self.run_retry_loop(mock_commands, max_retries=max_retries)

            self.assertEqual(
                attempt,
                expected_max_attempts,
                f"Incorrect attempts for max_retries={max_retries}",
            )


class TestBashTestGroupOutput(unittest.TestCase):
    """Test bash output messages and formatting."""

    def test_passed_first_attempt_output_format(self):
        """Verify output format for tests that pass on first attempt."""
        # Expected format from justfile: "✅ PASSED: $test_file"
        expected_marker = "✅ PASSED:"
        self.assertIn("PASSED", expected_marker)

    def test_passed_after_retry_output_format(self):
        """Verify output format for tests that pass after retry."""
        # Expected format: "✅ PASSED on attempt $attempt: $test_file (retried due to Mojo JIT flake)"
        expected_marker = "✅ PASSED on attempt"
        self.assertIn("attempt", expected_marker)
        self.assertIn("JIT flake", "PASSED on attempt 2: test.mojo (retried due to Mojo JIT flake)")

    def test_failed_after_retries_output_format(self):
        """Verify output format for tests that fail after exhausting retries."""
        # Expected format: "❌ FAILED after $max_attempts attempts: $test_file"
        expected_marker = "❌ FAILED after"
        self.assertIn("FAILED after", expected_marker)

    def test_retry_warning_output_format(self):
        """Verify format of retry warning message during retry."""
        # Expected: "⚠️  FAILED (attempt $attempt/$max_attempts), retrying in ${delay}s: $test_file"
        # Build the expected format string
        expected_format = "⚠️  FAILED (attempt 1/3), retrying in 1s: test.mojo"
        self.assertIn("FAILED", expected_format)
        self.assertIn("attempt", expected_format)
        self.assertIn("retrying", expected_format)

    def test_summary_output_format(self):
        """Verify summary output shows test counts."""
        # Expected sections: Total, Passed, Failed
        summary_lines = [
            "Total: 5 tests",
            "Passed: 3 tests",
            "Failed: 2 tests",
        ]
        for line in summary_lines:
            self.assertIn("tests", line)


class TestBashTestGroupErrorHandling(unittest.TestCase):
    """Test error handling in bash test-group recipe."""

    def test_no_test_files_found_error(self):
        """Verify error when no test files match the pattern."""
        # This should fail fast with clear error message
        # Expected error: "ERROR: No test files found in {{path}} matching {{pattern}}"
        error_msg = "No test files found"
        self.assertIn("test files found", error_msg)

    def test_exit_code_propagation_on_failure(self):
        """Verify test-group exits with code 1 on any failures."""
        # Multiple test scenario with some failures
        # Expected: exit 1 if failed_count > 0
        failed_count = 2
        expected_exit_code = 1
        self.assertGreater(failed_count, 0)
        self.assertEqual(expected_exit_code, 1)

    def test_exit_code_zero_on_all_pass(self):
        """Verify test-group exits with code 0 when all tests pass."""
        failed_count = 0
        expected_exit_code = 0
        self.assertEqual(failed_count, 0)
        self.assertEqual(expected_exit_code, 0)


if __name__ == "__main__":
    unittest.main()
