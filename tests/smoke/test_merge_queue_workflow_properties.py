#!/usr/bin/env python3
"""Regression tests for staged merge-queue readiness.

The live rulesets remain operator-owned. These tests lock the repository-side
contract that activation and the queued smoke check will consume.

Contract (fast merge-queue gate): merge_group events run EXACTLY ONE job —
`merge-queue-smoke` in merge-queue-smoke.yml — finishing well under five
minutes on a single runner slot. The former per-queue-entry re-run of the
full required matrix serialized on runner slots (70-90 min per merge).
PR-side CI is completely untouched: the four PR workflows still emit every
pinned context on pull_request events. The activation ruleset will pin
`merge-queue-smoke` as the ONLY required context (PR-side checks keep
running but become non-blocking after the flip).
"""

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "github" / "merge-queue-policy.json"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PR_COMMENT_WORKFLOW = "comprehensive-test-pr-comments.yml"
QUEUE_SMOKE_WORKFLOW = "merge-queue-smoke.yml"

# Workflows that emit the PR-side pinned contexts. They must NOT run on
# merge_group any more — the queue is served solely by merge-queue-smoke.yml.
PR_CONTEXT_WORKFLOWS = (
    "_required.yml",
    "comprehensive-tests.yml",
    "pre-commit.yml",
    "workflow-smoke-test.yml",
)

# The single context the activation ruleset pins (produced exclusively on
# merge_group events by merge-queue-smoke.yml).
EXPECTED_REQUIRED_CONTEXTS = [
    "merge-queue-smoke",
]

# Workflows whose `pull_request:` trigger is allowed to add a specific
# `paths-ignore` filter for `.github/workflows/**`. This exemption lets the
# heavy Mojo matrix + Semgrep SAST job exit early on workflow-only PRs
# (preventing spurious JIT/timeout flakes from blocking CI orchestrator
# fixes that propose new workflow files — dependabot-uv-lock-regen) without altering the
# broader PR-push coverage contract. Adding any other filter shape (extra
# paths, a different key, wildcards, etc.) still fails the assertion below.
PULL_REQUEST_PATH_FILTER_EXEMPTIONS: dict[str, dict[str, list[str]]] = {
    "comprehensive-tests.yml": {"paths-ignore": [".github/workflows/**"]},
}

# PR-side contexts that must keep being emitted exactly once per PR head.
EXPECTED_PR_CONTEXTS = [
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
        "min_entries_to_merge_wait_minutes": 0,
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


def test_policy_matches_single_context_and_approved_queue_rule() -> None:
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


def test_queue_smoke_is_the_only_merge_group_workflow() -> None:
    """merge_group must run exactly one workflow with one fast smoke job."""
    merge_group_workflows = [
        path.name for path in sorted(WORKFLOW_DIR.glob("*.yml")) if "merge_group" in _on_block(_load_yaml(path))
    ]
    assert merge_group_workflows == [QUEUE_SMOKE_WORKFLOW]

    smoke = _load_yaml(WORKFLOW_DIR / QUEUE_SMOKE_WORKFLOW)
    assert _on_block(smoke) == {"merge_group": {"types": ["checks_requested"]}}

    jobs = smoke.get("jobs")
    assert isinstance(jobs, dict) and list(jobs) == ["merge-queue-smoke"]
    gate = jobs["merge-queue-smoke"]
    assert gate.get("name") == "merge-queue-smoke"
    assert int(gate.get("timeout-minutes", 0)) <= 5


def test_pr_context_workflows_do_not_run_for_merge_group() -> None:
    """The heavy PR workflows must not re-run per queue entry."""
    for workflow_name in PR_CONTEXT_WORKFLOWS:
        workflow = _load_yaml(WORKFLOW_DIR / workflow_name)
        assert "merge_group" not in _on_block(workflow), (
            f"{workflow_name} must not handle merge_group; the queue is served solely by {QUEUE_SMOKE_WORKFLOW}"
        )


def test_existing_pull_request_and_push_triggers_are_preserved() -> None:
    """Queue changes must not narrow the existing PR or main-push coverage."""
    required_triggers = _on_block(_load_yaml(WORKFLOW_DIR / "_required.yml"))
    assert set(required_triggers) == {"pull_request", "push"}
    assert required_triggers["pull_request"] is None
    assert required_triggers["push"] == {"branches": ["main"]}

    for workflow_name in PR_CONTEXT_WORKFLOWS[1:3]:
        triggers = _on_block(_load_yaml(WORKFLOW_DIR / workflow_name))
        assert set(triggers) == {
            "pull_request",
            "push",
            "workflow_dispatch",
        }
        # `pull_request` MUST be either `None` (the contract default) OR the
        # exact `paths-ignore: ['.github/workflows/**']` exemption declared
        # in `PULL_REQUEST_PATH_FILTER_EXEMPTIONS` for this workflow. Any
        # other shape (e.g. `paths:`, `paths-ignore` with extra patterns)
        # narrows PR coverage and fails the contract by definition.
        expected_pr_filter = PULL_REQUEST_PATH_FILTER_EXEMPTIONS.get(workflow_name)
        assert triggers["pull_request"] == expected_pr_filter
        assert triggers["push"] == {"branches": ["main"]}

    smoke_triggers = _on_block(_load_yaml(WORKFLOW_DIR / "workflow-smoke-test.yml"))
    assert set(smoke_triggers) == {
        "pull_request",
        "push",
        "workflow_dispatch",
    }
    assert smoke_triggers["pull_request"] is None
    assert smoke_triggers["push"]["branches"] == ["main"]
    assert set(smoke_triggers["push"]["paths"]) == {
        ".github/workflows/_required.yml",
        ".github/workflows/comprehensive-test-pr-comments.yml",
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


def test_pr_contexts_are_emitted_once_across_pr_workflows() -> None:
    """Every PR-side pinned context must be emitted exactly once per head."""
    emitted: list[str] = []
    for workflow_name in PR_CONTEXT_WORKFLOWS:
        jobs = _load_yaml(WORKFLOW_DIR / workflow_name).get("jobs")
        assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
        for job_id, job in jobs.items():
            assert isinstance(job, dict), f"{workflow_name}:{job_id} must be a mapping"
            job_name = str(job.get("name", job_id))
            emitted.append(job_name)
            if job_name in EXPECTED_PR_CONTEXTS:
                assert job.get("if") in (None, "always()"), f"{workflow_name}:{job_id} must not skip pull_request runs"

    pr_emitted = [name for name in emitted if name in EXPECTED_PR_CONTEXTS]
    assert sorted(pr_emitted) == EXPECTED_PR_CONTEXTS
    assert len(pr_emitted) == len(set(pr_emitted))


def test_queue_workflows_preserve_minimum_permissions() -> None:
    """Queue-facing workflows must keep read-only token permissions."""
    expected_permissions = {
        "_required.yml": {"contents": "read"},
        "comprehensive-tests.yml": {"contents": "read"},
        "pre-commit.yml": {"contents": "read"},
        "workflow-smoke-test.yml": {"contents": "read"},
        QUEUE_SMOKE_WORKFLOW: {"contents": "read"},
    }

    for workflow_name, permissions in expected_permissions.items():
        workflow = _load_yaml(WORKFLOW_DIR / workflow_name)
        assert workflow.get("permissions") == permissions


def test_merge_group_cannot_receive_write_scope() -> None:
    """Merge-group jobs stay read-only; only trusted PR comments can write."""
    workflow = _load_yaml(WORKFLOW_DIR / "comprehensive-tests.yml")
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)

    assert not {
        job_id
        for job_id, job in jobs.items()
        if isinstance(job, dict) and "write" in job.get("permissions", {}).values()
    }

    comment_workflow = _load_yaml(WORKFLOW_DIR / PR_COMMENT_WORKFLOW)
    assert _on_block(comment_workflow) == {
        "workflow_run": {
            "workflows": ["Comprehensive Tests"],
            "types": ["completed"],
        }
    }
    assert comment_workflow.get("permissions") == {"contents": "read"}

    comment_jobs = comment_workflow.get("jobs")
    assert isinstance(comment_jobs, dict)
    assert set(comment_jobs) == {"post-pr-comments"}
    comment_job = comment_jobs["post-pr-comments"]
    assert comment_job.get("if") == (
        "github.event.workflow_run.event == 'pull_request' && github.event.workflow_run.pull_requests[0]"
    )
    assert comment_job.get("permissions") == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }

    uses = [str(step.get("uses", "")) for step in comment_job.get("steps", [])]
    assert all(not action.startswith("actions/checkout@") for action in uses)
    assert all(not action.startswith("./") for action in uses)


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
