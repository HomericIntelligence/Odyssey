#!/usr/bin/env python3
"""Smoke tests for pre-commit workflow properties.

Validates that .github/workflows/pre-commit.yml has correct trigger coverage and
step configuration, preventing regression of the patterns established in the
pre-commit CI setup:
  1. pull_request trigger is present (so all PRs are checked)
  2. mojo-format step is fail-fast (no continue-on-error, no ::warning:: wrapper)
     per HomericIntelligence/Odysseus#282 (Bucket F).
  3. Main pre-commit run skips mojo-format (SKIP=mojo-format) to avoid duplicate
     runs (the dedicated fail-fast step runs it separately).
  4. __matmul__ enforcement step is present and blocking (no continue-on-error).
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
PRE_COMMIT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pre-commit.yml"


@pytest.fixture(scope="module")
def pre_commit_workflow_content() -> str:
    """Read the pre-commit workflow file once for all tests in this module."""
    assert PRE_COMMIT_WORKFLOW.exists(), f"pre-commit.yml not found at {PRE_COMMIT_WORKFLOW}"
    return PRE_COMMIT_WORKFLOW.read_text(encoding="utf-8")


class TestPreCommitTriggers:
    """Verify trigger coverage for pre-commit.yml."""

    def test_pull_request_trigger_present(self, pre_commit_workflow_content: str) -> None:
        """pre-commit.yml must trigger on pull_request events.

        Without this trigger, pre-commit checks are skipped on PRs,
        allowing formatting violations and linting errors to merge undetected.
        """
        assert re.search(r"^\s+pull_request\b", pre_commit_workflow_content, re.MULTILINE), (
            "pre-commit.yml is missing a pull_request trigger. "
            "Add 'pull_request:' under the 'on:' block to check all PRs."
        )

    def test_push_trigger_present(self, pre_commit_workflow_content: str) -> None:
        """pre-commit.yml must trigger on push to main."""
        assert re.search(r"^\s+push\b", pre_commit_workflow_content, re.MULTILINE), (
            "pre-commit.yml is missing a push trigger."
        )


class TestMojoFormatHandling:
    """Verify mojo-format step is fail-fast per Bucket F (Odysseus#282)."""

    def test_mojo_format_step_is_fail_fast(self, pre_commit_workflow_content: str) -> None:
        """The mojo-format step must be fail-fast.

        Per HomericIntelligence/Odysseus#282 (Bucket F), the prior "advisory"
        forms are forbidden:

          1. `continue-on-error: true` (forbidden Bucket A)
          2. `if ! <cmd>; then echo "::warning::..."; fi` wrapper (forbidden
             Bucket F — morally identical to continue-on-error: true)

        If mojo format reports a diff, run the formatter locally in a
        GLIBC-compatible container (see docs/dev/mojo-glibc-compatibility.md)
        and commit the formatted files. The CI step must surface real diffs.
        """
        mojo_format_step_pattern = re.compile(
            r"-\s+name:\s+Run mojo format.*?(?=\n\s*-\s+name:|\Z)",
            re.DOTALL,
        )
        mojo_format_match = mojo_format_step_pattern.search(pre_commit_workflow_content)
        assert mojo_format_match is not None, "Could not find 'Run mojo format' step in pre-commit.yml."

        mojo_format_block = mojo_format_match.group(0)
        assert "continue-on-error: true" not in mojo_format_block, (
            "The mojo-format step has `continue-on-error: true` (forbidden Bucket A). "
            "Remove the flag so formatter diffs block the PR."
        )
        assert "::warning::" not in mojo_format_block, (
            "The mojo-format step uses the `::warning::` advisory pattern "
            "(forbidden Bucket F per HomericIntelligence/Odysseus#282). "
            "Run the command bare so CI fails on diffs."
        )
        assert re.search(r"pre-commit\s+run\s+mojo-format", mojo_format_block), (
            "The mojo-format step does not invoke `pre-commit run mojo-format`."
        )

    def test_main_hook_run_skips_mojo_format(self, pre_commit_workflow_content: str) -> None:
        """The main pre-commit run step must skip mojo-format via SKIP=mojo-format.

        The main hook run (which covers all other hooks) must exclude
        mojo-format to avoid duplicate runs — the dedicated fail-fast step
        further down in the workflow runs mojo-format on its own.
        """
        assert re.search(r"SKIP=mojo-format", pre_commit_workflow_content), (
            "pre-commit.yml is missing 'SKIP=mojo-format' on the main pre-commit run step. "
            "The main hook run must exclude mojo-format; it is run separately as a dedicated fail-fast step."
        )


class TestMatmulEnforcement:
    """Verify __matmul__ enforcement step is present and blocking."""

    def test_matmul_enforcement_step_present(self, pre_commit_workflow_content: str) -> None:
        """The __matmul__ enforcement step must be present.

        This step prevents .__matmul__() call sites from being introduced.
        Removing this step would allow the forbidden pattern to merge silently,
        breaking the matmul(A, B) convention enforced across the codebase.
        """
        assert re.search(r"__matmul__", pre_commit_workflow_content), (
            "pre-commit.yml is missing the __matmul__ enforcement step. "
            "This step must be present to prevent .__matmul__() call sites from merging."
        )

    def test_matmul_enforcement_is_blocking(self, pre_commit_workflow_content: str) -> None:
        """The __matmul__ enforcement step must not have continue-on-error: true.

        Unlike mojo-format (which is advisory), the __matmul__ enforcement is a
        hard requirement. If this step had continue-on-error: true, violations
        would not block the PR and the enforcement would be silently bypassed.
        """
        matmul_step_pattern = re.compile(
            r"-\s+name:\s+Enforce no \.__matmul__\(\) call sites.*?(?=\n\s*-\s+name:|\Z)",
            re.DOTALL,
        )
        matmul_match = matmul_step_pattern.search(pre_commit_workflow_content)
        assert matmul_match is not None, "Could not find 'Enforce no .__matmul__() call sites' step in pre-commit.yml."

        matmul_block = matmul_match.group(0)
        assert "continue-on-error: true" not in matmul_block, (
            "The __matmul__ enforcement step has 'continue-on-error: true'. "
            "This step must be blocking to prevent .__matmul__() call sites from merging."
        )
