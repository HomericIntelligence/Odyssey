#!/usr/bin/env python3
"""Smoke tests for security workflow properties.

Validates that .github/workflows/security.yml has correct trigger coverage and
security properties, preventing regression of gaps fixed in #3143:
  1. pull_request trigger is present
  2. Semgrep step has no continue-on-error: true
  3. Gitleaks does not use --no-git
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"


@pytest.fixture(scope="module")
def security_workflow_content() -> str:
    """Read the security workflow file once for all tests in this module."""
    assert SECURITY_WORKFLOW.exists(), f"security.yml not found at {SECURITY_WORKFLOW}"
    return SECURITY_WORKFLOW.read_text(encoding="utf-8")


class TestSecurityWorkflowTriggers:
    """Verify trigger coverage for security.yml."""

    def test_pull_request_trigger_present(self, security_workflow_content: str) -> None:
        """security.yml must trigger on pull_request events.

        Without this trigger, security scanning is skipped on PRs,
        allowing vulnerabilities to merge undetected.
        """
        # Match 'pull_request:' or 'pull_request' (with optional trailing colon/whitespace)
        # under the 'on:' block
        assert re.search(r"^\s+pull_request\b", security_workflow_content, re.MULTILINE), (
            "security.yml is missing a pull_request trigger. Add 'pull_request:' under the 'on:' block to scan all PRs."
        )

    def test_push_trigger_present(self, security_workflow_content: str) -> None:
        """security.yml must trigger on push to main."""
        assert re.search(r"^\s+push\b", security_workflow_content, re.MULTILINE), (
            "security.yml is missing a push trigger."
        )


class TestSemgrepStep:
    """Verify Semgrep SAST step configuration."""

    def test_semgrep_has_no_continue_on_error(self, security_workflow_content: str) -> None:
        """Semgrep step must not have continue-on-error: true.

        Setting continue-on-error: true silences SAST failures, allowing
        vulnerabilities to pass CI undetected.
        """
        # Find the Semgrep action block and verify continue-on-error is absent
        semgrep_pattern = re.compile(
            r"(returntocorp/semgrep-action|semgrep/semgrep-action).*?(?=\n\s*-\s+name:|\Z)",
            re.DOTALL,
        )
        semgrep_match = semgrep_pattern.search(security_workflow_content)
        assert semgrep_match is not None, (
            "Could not find Semgrep action step in security.yml. "
            "Expected a step using 'returntocorp/semgrep-action' or 'semgrep/semgrep-action'."
        )

        semgrep_block = semgrep_match.group(0)
        assert "continue-on-error: true" not in semgrep_block, (
            "Semgrep step has 'continue-on-error: true'. Remove it so SAST failures block the PR."
        )

    def test_semgrep_action_used(self, security_workflow_content: str) -> None:
        """Semgrep action must be present in the workflow."""
        assert re.search(r"uses:\s+(?:returntocorp|semgrep)/semgrep-action", security_workflow_content), (
            "No Semgrep action found in security.yml. "
            "Expected 'uses: returntocorp/semgrep-action' or 'uses: semgrep/semgrep-action'."
        )


class TestGitleaksStep:
    """Verify Gitleaks secret scanning step configuration."""

    def test_gitleaks_has_no_no_git_flag(self, security_workflow_content: str) -> None:
        """Gitleaks must not use the --no-git flag.

        --no-git scans only the working directory as a flat filesystem,
        bypassing git history. This misses secrets committed in past commits
        and then removed from the working tree.
        """
        assert "--no-git" not in security_workflow_content, (
            "Gitleaks is invoked with '--no-git' in security.yml. "
            "Remove '--no-git' so Gitleaks scans the full git history for secrets."
        )

    def test_gitleaks_present(self, security_workflow_content: str) -> None:
        """Gitleaks must be present in the workflow for secret scanning."""
        assert re.search(r"gitleaks", security_workflow_content, re.IGNORECASE), (
            "No Gitleaks step found in security.yml. Secret scanning via Gitleaks is required."
        )

    def test_gitleaks_has_exit_code(self, security_workflow_content: str) -> None:
        """Gitleaks must use --exit-code=1 to fail CI on secret detection."""
        assert "--exit-code=1" in security_workflow_content, (
            "Gitleaks is not configured with '--exit-code=1'. Without this, secret detection does not fail CI."
        )
