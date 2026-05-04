#!/usr/bin/env python3
"""Regression tests for paper-validation.yml workflow properties.

Validates that .github/workflows/paper-validation.yml is well-formed and
contains required configuration for paper implementation validation:
  1. Workflow file is valid YAML with required top-level keys
  2. Correct trigger paths (papers/**)
  3. All 'uses:' actions are SHA-pinned (not tag-pinned)
  4. Structure, implementation, and report jobs are defined
  5. Permissions are restrictive (contents: read, pull-requests: write)
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "paper-validation.yml"

SHA_RE = re.compile(r"uses:\s+\S+@([0-9a-f]{40})")
TAG_RE = re.compile(r"uses:\s+\S+@v[0-9]")


@pytest.fixture(scope="module")
def workflow_content() -> str:
    """Read the workflow file once for all tests in this module."""
    assert WORKFLOW.exists(), f"paper-validation.yml not found at {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow_yaml(workflow_content: str) -> dict:
    """Parse and return the workflow YAML as a dict."""
    try:
        parsed = yaml.safe_load(workflow_content)
        assert isinstance(parsed, dict), f"Workflow must be a YAML mapping, got {type(parsed)}"
        return parsed
    except yaml.YAMLError as e:
        pytest.fail(f"paper-validation.yml is invalid YAML: {e}")


class TestWorkflowExists:
    """Verify the workflow file is present and parseable."""

    def test_workflow_file_exists(self) -> None:
        """paper-validation.yml must exist in .github/workflows/."""
        assert WORKFLOW.exists(), (
            f"paper-validation.yml not found at {WORKFLOW}. "
            "This file is required to validate paper implementations in CI."
        )

    def test_workflow_is_valid_yaml(self, workflow_yaml: dict) -> None:
        """Workflow must be valid YAML (fixture already validates this)."""
        assert isinstance(workflow_yaml, dict)


class TestWorkflowTriggers:
    """Verify trigger configuration targets papers/ directory changes."""

    def test_workflow_triggers_on_papers_path(self, workflow_content: str) -> None:
        """Workflow must trigger when papers/** changes.

        Without this path filter, paper validation would run on every push
        regardless of whether paper content changed, wasting CI minutes.
        Without the filter, paper regressions on PRs would go undetected.
        """
        assert "papers/**" in workflow_content, (
            "paper-validation.yml must include 'papers/**' in path triggers. "
            "This ensures the workflow runs when paper implementations are changed."
        )

    def test_workflow_has_pull_request_trigger(self, workflow_content: str) -> None:
        """Workflow must trigger on pull_request events."""
        assert re.search(r"^\s+pull_request\b", workflow_content, re.MULTILINE), (
            "paper-validation.yml is missing a pull_request trigger. "
            "Without it, paper structure violations can merge undetected."
        )

    def test_workflow_has_push_to_main_trigger(self, workflow_content: str) -> None:
        """Workflow must trigger on pushes to main."""
        assert re.search(r"^\s+push\b", workflow_content, re.MULTILINE), (
            "paper-validation.yml is missing a push trigger for the main branch."
        )

    def test_workflow_has_workflow_dispatch(self, workflow_content: str) -> None:
        """Workflow must support manual dispatch via workflow_dispatch.

        Manual dispatch allows on-demand validation of the papers/ directory
        without requiring a code change to trigger the workflow.
        """
        assert "workflow_dispatch" in workflow_content, (
            "paper-validation.yml is missing workflow_dispatch trigger. "
            "Manual dispatch is required to run paper validation on demand."
        )


class TestWorkflowPermissions:
    """Verify permissions follow principle of least privilege."""

    def test_permissions_key_present(self, workflow_yaml: dict) -> None:
        """Workflow must declare explicit permissions."""
        assert "permissions" in workflow_yaml, (
            "paper-validation.yml is missing a top-level 'permissions' key. "
            "Explicit permissions are required to follow least-privilege principle."
        )

    def test_contents_permission_is_read(self, workflow_yaml: dict) -> None:
        """Workflow must request read-only contents access."""
        permissions = workflow_yaml.get("permissions", {})
        assert isinstance(permissions, dict), "permissions must be a mapping"
        assert permissions.get("contents") == "read", (
            "permissions.contents must be 'read'. "
            "Paper validation only reads files and should not write to the repository."
        )

    def test_pull_requests_permission_is_write(self, workflow_yaml: dict) -> None:
        """Workflow must request write access to pull-requests (for PR comments)."""
        permissions = workflow_yaml.get("permissions", {})
        assert isinstance(permissions, dict), "permissions must be a mapping"
        assert permissions.get("pull-requests") == "write", (
            "permissions.pull-requests must be 'write' to allow posting validation results "
            "as PR comments via the paper-report job."
        )


class TestWorkflowJobs:
    """Verify required jobs are defined."""

    def test_jobs_key_present(self, workflow_yaml: dict) -> None:
        """Workflow must have a 'jobs' key."""
        assert "jobs" in workflow_yaml, "paper-validation.yml missing 'jobs' key"

    def test_validate_structure_job_exists(self, workflow_yaml: dict) -> None:
        """validate-structure job must be defined.

        This job validates that each paper directory has required files
        (README.md, metadata.yml) before any implementation is checked.
        """
        jobs = workflow_yaml.get("jobs", {})
        assert "validate-structure" in jobs, (
            "paper-validation.yml is missing the 'validate-structure' job. "
            "This job is required to check that each paper directory has the required structure."
        )

    def test_validate_implementation_job_exists(self, workflow_yaml: dict) -> None:
        """validate-implementation job must be defined."""
        jobs = workflow_yaml.get("jobs", {})
        assert "validate-implementation" in jobs, (
            "paper-validation.yml is missing the 'validate-implementation' job. "
            "This job checks that Mojo implementation files and tests are present."
        )

    def test_paper_report_job_exists(self, workflow_yaml: dict) -> None:
        """paper-report job must be defined to aggregate and surface results."""
        jobs = workflow_yaml.get("jobs", {})
        assert "paper-report" in jobs, (
            "paper-validation.yml is missing the 'paper-report' job. "
            "This job aggregates validation results and posts them to PRs."
        )

    def test_paper_report_job_depends_on_validate_jobs(self, workflow_yaml: dict) -> None:
        """paper-report job must depend on both validation jobs."""
        jobs = workflow_yaml.get("jobs", {})
        report_job = jobs.get("paper-report", {})
        needs = report_job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "validate-structure" in needs, (
            "paper-report job must list 'validate-structure' in its 'needs' to ensure "
            "structure validation completes before reporting."
        )
        assert "validate-implementation" in needs, (
            "paper-report job must list 'validate-implementation' in its 'needs' to ensure "
            "implementation validation completes before reporting."
        )


class TestWorkflowActionPins:
    """Verify all 'uses:' references are SHA-pinned, not tag-pinned."""

    def test_no_tag_pinned_actions(self, workflow_content: str) -> None:
        """No 'uses:' line may reference an action by version tag (e.g. @v3).

        Tag-pinned actions can change silently when a maintainer force-pushes the tag,
        creating a supply-chain risk. SHA pins are immutable.
        """
        violations = []
        for i, line in enumerate(workflow_content.splitlines(), 1):
            if TAG_RE.search(line):
                violations.append(f"  line {i}: {line.strip()}")
        assert not violations, (
            "paper-validation.yml contains tag-pinned actions (must use SHA instead):\n"
            + "\n".join(violations)
        )

    def test_all_external_actions_are_sha_pinned(self, workflow_content: str) -> None:
        """Every external 'uses:' line must reference a 40-char hex SHA.

        Local action references (uses: ./.github/actions/...) are exempt.
        """
        violations = []
        for i, line in enumerate(workflow_content.splitlines(), 1):
            stripped = line.strip()
            if not re.match(r"uses:", stripped):
                continue
            # Skip local action references
            if re.search(r"uses:\s+\./", stripped):
                continue
            if not SHA_RE.search(line):
                violations.append(f"  line {i}: {line.strip()}")
        assert not violations, (
            "paper-validation.yml has external actions not pinned to a SHA:\n"
            + "\n".join(violations)
        )
