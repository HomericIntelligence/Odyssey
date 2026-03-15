#!/usr/bin/env python3
"""Smoke tests for comprehensive-tests workflow properties.

Validates that .github/workflows/comprehensive-tests.yml has correct trigger coverage
and matrix configuration, preventing regression of critical CI properties:
  1. pull_request trigger is present (so all PRs run the full test suite)
  2. fail-fast: false is set (so all test groups run even if one fails)
  3. Key matrix test groups are present: Core Tensors, Core Gradient, Models
  4. test-mojo-comprehensive job depends on both mojo-compilation and validate-test-coverage
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
COMPREHENSIVE_TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "comprehensive-tests.yml"


@pytest.fixture(scope="module")
def comprehensive_tests_workflow_content() -> str:
    """Read the comprehensive-tests workflow file once for all tests in this module."""
    assert COMPREHENSIVE_TESTS_WORKFLOW.exists(), f"comprehensive-tests.yml not found at {COMPREHENSIVE_TESTS_WORKFLOW}"
    return COMPREHENSIVE_TESTS_WORKFLOW.read_text(encoding="utf-8")


class TestComprehensiveTestsTriggers:
    """Verify trigger coverage for comprehensive-tests.yml."""

    def test_pull_request_trigger_present(self, comprehensive_tests_workflow_content: str) -> None:
        """comprehensive-tests.yml must trigger on pull_request events.

        Without this trigger, the full test suite is skipped on PRs,
        allowing regressions to merge undetected.
        """
        assert re.search(r"^\s+pull_request\b", comprehensive_tests_workflow_content, re.MULTILINE), (
            "comprehensive-tests.yml is missing a pull_request trigger. "
            "Add 'pull_request:' under the 'on:' block to run tests on all PRs."
        )

    def test_push_trigger_present(self, comprehensive_tests_workflow_content: str) -> None:
        """comprehensive-tests.yml must trigger on push to main."""
        assert re.search(r"^\s+push\b", comprehensive_tests_workflow_content, re.MULTILINE), (
            "comprehensive-tests.yml is missing a push trigger."
        )


class TestMatrixStrategy:
    """Verify matrix strategy configuration for the test suite."""

    def test_fail_fast_is_false(self, comprehensive_tests_workflow_content: str) -> None:
        """The matrix strategy must have fail-fast: false.

        Without fail-fast: false, a single failing test group cancels all
        other running groups. This hides which groups pass and which fail,
        making it harder to diagnose and fix CI failures. The flag ensures
        all test groups always run to completion.
        """
        assert re.search(r"fail-fast:\s*false", comprehensive_tests_workflow_content), (
            "comprehensive-tests.yml is missing 'fail-fast: false' in the matrix strategy. "
            "Without this, a single failing test group will cancel all other groups, "
            "hiding which tests pass and making failures harder to diagnose."
        )

    def test_core_tensors_group_present(self, comprehensive_tests_workflow_content: str) -> None:
        """The matrix must include the 'Core Tensors' test group.

        Core Tensors is the foundational test group covering tensor operations,
        arithmetic, broadcasting, and matrix operations. Removing it would leave
        the most critical shared library components untested on every PR.
        """
        assert re.search(r'"Core Tensors"', comprehensive_tests_workflow_content), (
            "comprehensive-tests.yml matrix is missing the 'Core Tensors' test group. "
            "This group covers tensor operations and is required for regression protection."
        )

    def test_core_gradient_group_present(self, comprehensive_tests_workflow_content: str) -> None:
        """The matrix must include the 'Core Gradient' test group.

        Core Gradient tests backward passes and gradient checking. Removing this
        group would leave the autograd engine untested on every PR.
        """
        assert re.search(r'"Core Gradient"', comprehensive_tests_workflow_content), (
            "comprehensive-tests.yml matrix is missing the 'Core Gradient' test group. "
            "This group covers backward passes and gradient checking."
        )

    def test_models_group_present(self, comprehensive_tests_workflow_content: str) -> None:
        """The matrix must include the 'Models' test group.

        Models is the primary PR-blocking test suite for model layer tests.
        Removing this group would leave model implementations untested on every PR.
        """
        assert re.search(r'"Models"', comprehensive_tests_workflow_content), (
            "comprehensive-tests.yml matrix is missing the 'Models' test group. "
            "This is the primary PR-blocking test suite for model layer tests."
        )


class TestJobDependencies:
    """Verify job dependency chain for test-mojo-comprehensive."""

    def test_mojo_tests_needs_compilation(self, comprehensive_tests_workflow_content: str) -> None:
        """test-mojo-comprehensive must depend on mojo-compilation.

        Without this dependency, tests could run against a broken package,
        producing confusing failures that are actually compilation errors.
        The compilation gate ensures tests only run when the package builds cleanly.
        """
        # Find the test-mojo-comprehensive job's needs declaration
        job_pattern = re.compile(
            r"test-mojo-comprehensive:.*?(?=\n\w|\Z)",
            re.DOTALL,
        )
        job_match = job_pattern.search(comprehensive_tests_workflow_content)
        assert job_match is not None, "Could not find 'test-mojo-comprehensive' job in comprehensive-tests.yml."

        job_block = job_match.group(0)
        assert "mojo-compilation" in job_block, (
            "test-mojo-comprehensive job does not depend on 'mojo-compilation'. "
            "Without this dependency, tests may run against a broken package."
        )

    def test_mojo_tests_needs_coverage_validation(self, comprehensive_tests_workflow_content: str) -> None:
        """test-mojo-comprehensive must depend on validate-test-coverage.

        Without this dependency, tests run even when the test coverage
        validation script reports missing or misconfigured test groups,
        making it impossible to detect when test files are orphaned from the matrix.
        """
        job_pattern = re.compile(
            r"test-mojo-comprehensive:.*?(?=\n\w|\Z)",
            re.DOTALL,
        )
        job_match = job_pattern.search(comprehensive_tests_workflow_content)
        assert job_match is not None, "Could not find 'test-mojo-comprehensive' job in comprehensive-tests.yml."

        job_block = job_match.group(0)
        assert "validate-test-coverage" in job_block, (
            "test-mojo-comprehensive job does not depend on 'validate-test-coverage'. "
            "Without this dependency, test coverage gaps will not block the test run."
        )
