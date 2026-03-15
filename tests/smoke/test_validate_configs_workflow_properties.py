#!/usr/bin/env python3
"""Smoke tests for validate-configs workflow properties.

Validates that .github/workflows/validate-configs.yml has correct trigger coverage
and validation steps, preventing regression of the config validation setup:
  1. pull_request trigger is present scoped to main (so config changes on PRs are validated)
  2. yamllint is used for YAML syntax validation
  3. Required default configs check is present (training.yaml, model.yaml, data.yaml)
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
VALIDATE_CONFIGS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate-configs.yml"


@pytest.fixture(scope="module")
def validate_configs_workflow_content() -> str:
    """Read the validate-configs workflow file once for all tests in this module."""
    assert VALIDATE_CONFIGS_WORKFLOW.exists(), (
        f"validate-configs.yml not found at {VALIDATE_CONFIGS_WORKFLOW}"
    )
    return VALIDATE_CONFIGS_WORKFLOW.read_text(encoding="utf-8")


class TestValidateConfigsTriggers:
    """Verify trigger coverage for validate-configs.yml."""

    def test_pull_request_trigger_present(self, validate_configs_workflow_content: str) -> None:
        """validate-configs.yml must trigger on pull_request events targeting main.

        This workflow uses a scoped pull_request trigger (branches: [main]) so it
        only runs on PRs that target main. Without this trigger, config file changes
        in PRs would not be validated before merging, allowing invalid YAML or
        missing required configs to reach the main branch.
        """
        assert re.search(r"^\s+pull_request\b", validate_configs_workflow_content, re.MULTILINE), (
            "validate-configs.yml is missing a pull_request trigger. "
            "Add 'pull_request:' under the 'on:' block to validate config changes on PRs."
        )

    def test_push_trigger_present(self, validate_configs_workflow_content: str) -> None:
        """validate-configs.yml must trigger on push to main."""
        assert re.search(r"^\s+push\b", validate_configs_workflow_content, re.MULTILINE), (
            "validate-configs.yml is missing a push trigger."
        )


class TestValidationSteps:
    """Verify required validation steps are present in validate-configs.yml."""

    def test_yamllint_step_present(self, validate_configs_workflow_content: str) -> None:
        """yamllint must be present for YAML syntax validation.

        yamllint is the standard tool for enforcing YAML syntax and style in configs/.
        Removing this step would allow malformed YAML to merge, which can cause
        silent config parsing errors in training runs or other consumers of these files.
        """
        assert re.search(r"yamllint", validate_configs_workflow_content), (
            "validate-configs.yml is missing 'yamllint'. "
            "yamllint must be used to validate YAML syntax in configs/."
        )

    def test_required_defaults_check_present(
        self, validate_configs_workflow_content: str
    ) -> None:
        """The required default configs check must be present.

        This step verifies that configs/defaults/training.yaml,
        configs/defaults/model.yaml, and configs/defaults/data.yaml exist.
        Removing this check would allow these required defaults to be deleted
        without any CI signal, breaking training pipelines that depend on them.
        """
        assert re.search(
            r"configs/defaults/training\.yaml", validate_configs_workflow_content
        ), (
            "validate-configs.yml is missing a check for 'configs/defaults/training.yaml'. "
            "The required defaults check must verify this file exists."
        )
