#!/usr/bin/env python3
"""Integration tests for precommit-benchmark.yml workflow validity.

Validates that .github/workflows/precommit-benchmark.yml is well-formed and
contains required configuration for the pre-commit hook performance monitoring:
  1. Workflow file is valid YAML
  2. Required top-level keys exist (on, jobs, permissions)
  3. Permissions are restrictive (contents: read)
  4. Benchmark step captures elapsed time as an output
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "precommit-benchmark.yml"


@pytest.fixture(scope="module")
def workflow_yaml() -> dict:
    """Parse and return the workflow YAML as a dict."""
    assert WORKFLOW.exists(), f"Workflow file not found at {WORKFLOW}"
    content = WORKFLOW.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), f"Workflow must be a YAML mapping, got {type(parsed)}"
        return parsed
    except yaml.YAMLError as e:
        pytest.fail(f"Workflow YAML is invalid: {e}")


class TestWorkflowStructure:
    """Test that the workflow has the required structure."""

    def test_workflow_file_exists(self) -> None:
        """Verify the workflow file exists at the expected path."""
        assert WORKFLOW.exists(), f"Workflow file not found at {WORKFLOW}"

    def test_workflow_has_on_trigger(self, workflow_yaml: dict) -> None:
        """Workflow must have an 'on' key defining triggers.

        Note: YAML parses 'on' as a boolean True, not as string 'on'.
        """
        # Check for YAML's representation of 'on' keyword (Python True)
        assert True in workflow_yaml or "on" in workflow_yaml, (
            "Workflow missing 'on' key (triggers). YAML parses 'on:' as boolean True."
        )

    def test_workflow_has_jobs(self, workflow_yaml: dict) -> None:
        """Workflow must have a 'jobs' key."""
        assert "jobs" in workflow_yaml, "Workflow missing 'jobs' key"

    def test_workflow_has_permissions(self, workflow_yaml: dict) -> None:
        """Workflow must have a 'permissions' key."""
        assert "permissions" in workflow_yaml, "Workflow missing 'permissions' key"

    def test_jobs_is_mapping(self, workflow_yaml: dict) -> None:
        """Jobs must be a mapping (dict) of job definitions."""
        assert isinstance(workflow_yaml["jobs"], dict), "jobs must be a mapping, not empty or scalar"

    def test_precommit_benchmark_job_exists(self, workflow_yaml: dict) -> None:
        """The 'precommit-benchmark' job must be defined."""
        assert "precommit-benchmark" in workflow_yaml["jobs"], "Job 'precommit-benchmark' not found in jobs"


class TestPermissions:
    """Test workflow permissions configuration."""

    def test_permissions_structure(self, workflow_yaml: dict) -> None:
        """Permissions must be a mapping."""
        permissions = workflow_yaml.get("permissions")
        assert isinstance(permissions, dict), f"permissions must be a mapping, got {type(permissions)}"

    def test_permissions_contents_is_read(self, workflow_yaml: dict) -> None:
        """Permissions.contents must be set to 'read' (principle of least privilege)."""
        permissions = workflow_yaml.get("permissions", {})
        assert permissions.get("contents") == "read", (
            "permissions.contents must be 'read'. Benchmark workflow should not write to repository."
        )


class TestPrecommitBenchmarkJob:
    """Test the precommit-benchmark job configuration."""

    @pytest.fixture
    def job_config(self, workflow_yaml: dict) -> dict:
        """Extract the precommit-benchmark job configuration."""
        return workflow_yaml["jobs"]["precommit-benchmark"]

    def test_job_has_steps(self, job_config: dict) -> None:
        """Job must have a 'steps' key."""
        assert "steps" in job_config, "precommit-benchmark job missing 'steps' key"

    def test_job_steps_is_list(self, job_config: dict) -> None:
        """Steps must be a list."""
        assert isinstance(job_config["steps"], list), "steps must be a list of step definitions"

    def test_job_has_time_hooks_step(self, job_config: dict) -> None:
        """Job must have a 'Time pre-commit hooks' step."""
        step_names = [step.get("name", "") for step in job_config.get("steps", [])]
        assert "Time pre-commit hooks" in step_names, (
            "Job must have a 'Time pre-commit hooks' step that measures hook execution time"
        )

    def test_time_hooks_step_has_id(self, job_config: dict) -> None:
        """The 'Time pre-commit hooks' step must have an 'id' for output reference."""
        steps = job_config.get("steps", [])
        time_hooks_step = next(
            (s for s in steps if s.get("name") == "Time pre-commit hooks"),
            None,
        )
        assert time_hooks_step is not None, "'Time pre-commit hooks' step not found"
        assert "id" in time_hooks_step, (
            "'Time pre-commit hooks' step must have an 'id' to capture and reference step outputs"
        )

    def test_time_hooks_step_has_elapsed_output(self, job_config: dict) -> None:
        """The 'Time pre-commit hooks' step must capture 'elapsed' as an output."""
        steps = job_config.get("steps", [])
        time_hooks_step = next(
            (s for s in steps if s.get("name") == "Time pre-commit hooks"),
            None,
        )
        assert time_hooks_step is not None, "'Time pre-commit hooks' step not found"

        # Check if the step's 'run' command contains the GITHUB_OUTPUT assignment
        run_command = time_hooks_step.get("run", "")
        assert "elapsed=" in run_command, (
            "'Time pre-commit hooks' step must capture 'elapsed' output (e.g., 'echo \"elapsed=$ELAPSED\" >> \"$GITHUB_OUTPUT\"')"
        )

    def test_benchmark_summary_step_exists(self, job_config: dict) -> None:
        """Job must have a 'Write benchmark summary' step."""
        step_names = [step.get("name", "") for step in job_config.get("steps", [])]
        assert "Write benchmark summary" in step_names, (
            "Job must have a 'Write benchmark summary' step that processes the benchmark results"
        )

    def test_benchmark_summary_uses_elapsed_output(self, job_config: dict) -> None:
        """The 'Write benchmark summary' step must reference the elapsed output."""
        steps = job_config.get("steps", [])
        summary_step = next(
            (s for s in steps if s.get("name") == "Write benchmark summary"),
            None,
        )
        assert summary_step is not None, "'Write benchmark summary' step not found"

        # Check that the step references the elapsed output
        step_yaml = str(summary_step)
        assert "elapsed" in step_yaml, (
            "'Write benchmark summary' step must use the 'elapsed' output from 'Time pre-commit hooks' step"
        )
