#!/usr/bin/env python3
"""Regression tests for staged merge-queue readiness.

The live rulesets remain operator-owned. These tests lock the repository-side
contract that activation and the queued smoke check will consume.
"""

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "github" / "merge-queue-policy.json"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

REQUIRED_WORKFLOWS = (
    "_required.yml",
    "comprehensive-tests.yml",
    "pre-commit.yml",
    "workflow-smoke-test.yml",
)

EXPECTED_REQUIRED_CONTEXTS = [
    "Audit Shared Links",
    "Build Validation",
    "Code Quality Analysis",
    "Core Layers",
    "Data Utilities Test Suite",
    "Gradient Checking Tests",
    "Gradient Coverage Report",
    "Mojo Package Compilation",
    "Mojo Syntax Validation",
    "Other Workflow Property Checks",
    "Python Tests",
    "Security Workflow Property Checks",
    "Test Coverage Validation",
    "Test Metrics",
    "build",
    "deps/version-sync",
    "install",
    "integration-tests",
    "lint",
    "lint-notebooks",
    "mypy",
    "package",
    "pre-commit",
    "python-syntax",
    "release",
    "schema-validation",
    "security/dependency-scan",
    "security/secrets-scan",
    "test",
    "unit-tests",
    "validate-notebooks",
]

EXPECTED_MERGE_QUEUE_RULE = {
    "type": "merge_queue",
    "parameters": {
        "check_response_timeout_minutes": 60,
        "grouping_strategy": "ALLGREEN",
        "max_entries_to_build": 10,
        "max_entries_to_merge": 5,
        "merge_method": "SQUASH",
        "min_entries_to_merge": 1,
        "min_entries_to_merge_wait_minutes": 5,
    },
}


def _load_yaml(path: Path) -> dict[Any, Any]:
    """Load one workflow and normalize PyYAML's YAML 1.1 ``on`` key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _on_block(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return a workflow trigger mapping despite PyYAML treating ``on`` as bool."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), "workflow on block must be a mapping"
    return triggers


def test_policy_matches_live_required_contexts_and_approved_queue_rule() -> None:
    """The policy artifact must be the exact reviewed activation contract."""
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy == {
        "repository": "HomericIntelligence/Odyssey",
        "target_branch": "main",
        "activation_ruleset": "homeric-main-baseline",
        "required_contexts": EXPECTED_REQUIRED_CONTEXTS,
        "merge_queue_rule": EXPECTED_MERGE_QUEUE_RULE,
    }
    assert policy["required_contexts"] == sorted(policy["required_contexts"])


def test_every_required_context_workflow_handles_merge_group() -> None:
    """Every workflow posting a required context must run for queue candidates."""
    for workflow_name in REQUIRED_WORKFLOWS:
        workflow = _load_yaml(WORKFLOW_DIR / workflow_name)
        assert _on_block(workflow).get("merge_group") == {"types": ["checks_requested"]}, (
            f"{workflow_name} must handle merge_group/checks_requested"
        )


def test_existing_pull_request_and_push_triggers_are_preserved() -> None:
    """Queue readiness must not narrow the existing PR or main-push coverage."""
    required_triggers = _on_block(_load_yaml(WORKFLOW_DIR / "_required.yml"))
    assert set(required_triggers) == {"pull_request", "merge_group", "push"}
    assert required_triggers["pull_request"] is None
    assert required_triggers["push"] == {"branches": ["main"]}

    for workflow_name in REQUIRED_WORKFLOWS[1:3]:
        triggers = _on_block(_load_yaml(WORKFLOW_DIR / workflow_name))
        assert set(triggers) == {
            "pull_request",
            "merge_group",
            "push",
            "workflow_dispatch",
        }
        assert triggers["pull_request"] is None
        assert triggers["push"] == {"branches": ["main"]}

    smoke_triggers = _on_block(_load_yaml(WORKFLOW_DIR / "workflow-smoke-test.yml"))
    assert set(smoke_triggers) == {
        "pull_request",
        "merge_group",
        "push",
        "workflow_dispatch",
    }
    assert smoke_triggers["pull_request"] is None
    assert smoke_triggers["push"]["branches"] == ["main"]
    assert set(smoke_triggers["push"]["paths"]) == {
        ".github/workflows/_required.yml",
        ".github/workflows/comprehensive-tests.yml",
        ".github/workflows/pre-commit.yml",
        ".github/workflows/security.yml",
        ".github/workflows/validate-configs.yml",
        ".github/workflows/workflow-smoke-test.yml",
        "configs/github/merge-queue-policy.json",
        "tests/smoke/test_comprehensive_tests_workflow_properties.py",
        "tests/smoke/test_merge_queue_workflow_properties.py",
        "tests/smoke/test_pre_commit_workflow_properties.py",
        "tests/smoke/test_security_workflow_properties.py",
        "tests/smoke/test_validate_configs_workflow_properties.py",
    }


def test_required_contexts_are_emitted_once_across_queue_workflows() -> None:
    """The queue candidate must receive every pinned context exactly once."""
    emitted: list[str] = []
    for workflow_name in REQUIRED_WORKFLOWS:
        jobs = _load_yaml(WORKFLOW_DIR / workflow_name).get("jobs")
        assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
        for job_id, job in jobs.items():
            assert isinstance(job, dict), f"{workflow_name}:{job_id} must be a mapping"
            job_name = str(job.get("name", job_id))
            emitted.append(job_name)
            if job_name in EXPECTED_REQUIRED_CONTEXTS:
                assert job.get("if") in (None, "always()"), f"{workflow_name}:{job_id} must not skip merge_group runs"

    required_emitted = [name for name in emitted if name in EXPECTED_REQUIRED_CONTEXTS]
    assert sorted(required_emitted) == EXPECTED_REQUIRED_CONTEXTS
    assert len(required_emitted) == len(set(required_emitted))


def test_queue_workflows_preserve_minimum_permissions() -> None:
    """Adding queue triggers must not broaden the workflows' token permissions."""
    expected_permissions = {
        "_required.yml": {"contents": "read"},
        "comprehensive-tests.yml": {
            "contents": "read",
            "pull-requests": "write",
        },
        "pre-commit.yml": {"contents": "read"},
        "workflow-smoke-test.yml": {"contents": "read"},
    }

    for workflow_name, permissions in expected_permissions.items():
        workflow = _load_yaml(WORKFLOW_DIR / workflow_name)
        assert workflow.get("permissions") == permissions


def test_release_workflow_remains_tag_or_manual_only() -> None:
    """Queue readiness must not execute the real publishing workflow."""
    triggers = _on_block(_load_yaml(WORKFLOW_DIR / "release.yml"))

    assert triggers["push"] == {"tags": ["v*"]}
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
    assert "merge_group" not in triggers


def test_merge_queue_regression_runs_in_required_smoke_workflow() -> None:
    """The focused regression itself must run in a required queue context."""
    smoke_workflow = (WORKFLOW_DIR / "workflow-smoke-test.yml").read_text(encoding="utf-8")
    assert "tests/smoke/test_merge_queue_workflow_properties.py" in smoke_workflow
