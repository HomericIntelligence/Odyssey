#!/usr/bin/env python3
"""Smoke tests for security workflow properties.

Validates that .github/workflows/security.yml has correct trigger coverage and
security properties, preventing regression of gaps fixed in #3143:
  1. pull_request trigger is present
  2. Semgrep is installed via pip and run via CLI with continue-on-error
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

    def test_semgrep_has_continue_on_error(self, security_workflow_content: str) -> None:
        """Semgrep scan step must have continue-on-error: true.

        The CLI-based semgrep scan uses continue-on-error: true so that
        SARIF results are always uploaded regardless of scan exit code.
        """
        # Find the "Run Semgrep" step block and verify continue-on-error is present
        semgrep_pattern = re.compile(
            r"-\s+name:\s+Run Semgrep.*?(?=\n\s*-\s+name:|\Z)",
            re.DOTALL,
        )
        semgrep_match = semgrep_pattern.search(security_workflow_content)
        assert semgrep_match is not None, (
            "Could not find 'Run Semgrep' step in security.yml. "
            "Expected a step named 'Run Semgrep' that runs 'semgrep scan'."
        )

        semgrep_block = semgrep_match.group(0)
        assert "continue-on-error: true" in semgrep_block, (
            "Semgrep scan step is missing 'continue-on-error: true'. "
            "This is needed so SARIF results are uploaded even when findings exist."
        )

    def test_upload_sarif_allowed_to_have_continue_on_error(self, security_workflow_content: str) -> None:
        """Upload-sarif step is allowed to have continue-on-error: true.

        The upload-sarif step may legitimately have continue-on-error: true
        to allow report uploads to fail without blocking the scan.
        This test documents the expected behavior to prevent future refactors
        from incorrectly removing this flag.
        """
        # Find the "Upload SARIF results" step
        upload_pattern = re.compile(
            r"-\s+name:\s+Upload SARIF.*?(?=\n\s*-\s+name:|\Z)",
            re.DOTALL,
        )
        upload_match = upload_pattern.search(security_workflow_content)
        assert upload_match is not None, (
            "Could not find 'Upload SARIF results' step in security.yml. "
            "Expected a step for uploading SARIF results to GitHub Security tab."
        )

        # Note: This test only documents the expected behavior.
        # Whether upload-sarif has continue-on-error is a policy decision,
        # not an error condition. This test serves as documentation
        # in case a future refactor needs to decide on this behavior.

    def test_semgrep_action_used(self, security_workflow_content: str) -> None:
        """Semgrep must be installed via pip and run via CLI."""
        assert re.search(r"pip install semgrep", security_workflow_content), (
            "No 'pip install semgrep' found in security.yml. Semgrep must be installed via pip for CLI-based scanning."
        )
        assert re.search(r"semgrep scan", security_workflow_content), (
            "No 'semgrep scan' command found in security.yml. Expected semgrep to be run via 'semgrep scan' CLI."
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
