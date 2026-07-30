#!/usr/bin/env python3
"""Regression tests for the Gradient Checker Soak workflow."""

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
GRADIENT_SOAK_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gradient-soak.yml"
WORKFLOW_SMOKE = REPO_ROOT / ".github" / "workflows" / "workflow-smoke-test.yml"
PROPERTY_TEST_PATH = "tests/smoke/test_gradient_soak_workflow_properties.py"
SETUP_JUST_ACTION = "extractions/setup-just@53165ef7e734c5c07cb06b3c8e7b647c5aa16db3"
SETUP_JUST_VERSION = "1.36.0"


def test_pinned_setup_just_precedes_gradient_soak_command() -> None:
    """The soak must install the pinned Just version before invoking `just`."""
    workflow: dict[str, Any] = yaml.safe_load(GRADIENT_SOAK_WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["gradient-soak"]["steps"]

    setup_indices = [index for index, step in enumerate(steps) if step.get("uses") == SETUP_JUST_ACTION]
    just_indices = [index for index, step in enumerate(steps) if re.search(r"\bjust\s+", str(step.get("run", "")))]

    assert just_indices, "gradient-soak.yml must invoke Just to run the soak"
    assert setup_indices, "gradient-soak.yml must install Just with the repository's exact SHA-pinned setup-just action"
    assert setup_indices[0] < min(just_indices), "The pinned setup-just step must run before the first Just command"
    assert steps[setup_indices[0]].get("with", {}).get("just-version") == SETUP_JUST_VERSION


def test_gradient_soak_changes_trigger_workflow_property_checks() -> None:
    """Changes to the soak or its regression test must run workflow smoke CI."""
    workflow: dict[Any, Any] = yaml.safe_load(WORKFLOW_SMOKE.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    watched_paths = set(triggers["push"]["paths"])

    assert {
        ".github/workflows/gradient-soak.yml",
        PROPERTY_TEST_PATH,
    } <= watched_paths


def test_gradient_soak_property_test_runs_in_workflow_smoke() -> None:
    """Workflow smoke CI must execute the Gradient Soak regression test."""
    workflow: dict[str, Any] = yaml.safe_load(WORKFLOW_SMOKE.read_text(encoding="utf-8"))
    run_blocks = [str(step.get("run", "")) for job in workflow["jobs"].values() for step in job.get("steps", [])]

    assert any(PROPERTY_TEST_PATH in run_block for run_block in run_blocks)
