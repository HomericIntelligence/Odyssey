#!/usr/bin/env python3
"""Regression tests for real required-context merge-group parity.

Pull-request and merge-group commits must run the same real producer
workflows. A smoke-only workflow cannot replace a protected context.
"""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PR_COMMENT_WORKFLOW = "comprehensive-test-pr-comments.yml"
OBSOLETE_MERGE_QUEUE_POLICY = REPO_ROOT / "configs" / "github" / "merge-queue-policy.json"

PRODUCER_WORKFLOWS = (
    "_required.yml",
    "comprehensive-tests.yml",
    "pre-commit.yml",
    "workflow-smoke-test.yml",
)

EXPECTED_REQUIRED_CONTEXTS = frozenset(
    {
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
        "Test Report",
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
    }
)

EXPECTED_CONCURRENCY_GROUP = (
    "${{ github.workflow }}-${{ github.event_name }}-${{ github.event.pull_request.number || github.sha }}"
)

# The report explicitly accepts the optional workflow-dispatch-only SIMD lane.
# Its separate report-contract tests prove that every other skipped result fails.
OPTIONAL_DEPENDENCY_CONDITIONS = {
    (
        "comprehensive-tests.yml",
        "test-report",
        "simd-analysis",
    ): "github.event_name == 'workflow_dispatch' && inputs.run_extended == true",
}

# Event-specific steps are forbidden unless they are an exact, non-validator
# side effect. This PR-only report does not affect the job verdict.
OPTIONAL_STEP_EVENT_CONDITIONS = {
    (
        "comprehensive-tests.yml",
        "validate-test-coverage",
        "Post validation report to PR",
    ): "github.event_name == 'pull_request' && steps.validation.outputs.exit_code != '0'",
}


def _workflow_paths(workflow_dir: Path = WORKFLOW_DIR) -> list[Path]:
    """Return all GitHub workflow files for both accepted YAML suffixes."""
    return sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))


def _load_yaml(path: Path) -> dict[Any, Any]:
    """Load one workflow and normalize PyYAML's YAML 1.1 ``on`` key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _load_workflows(workflow_dir: Path = WORKFLOW_DIR) -> dict[str, dict[Any, Any]]:
    """Load every workflow without omitting the .yaml extension."""
    return {path.name: _load_yaml(path) for path in _workflow_paths(workflow_dir)}


def _on_block(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return a trigger mapping despite PyYAML treating ``on`` as Boolean."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), "workflow on block must be a mapping"
    return triggers


def _needs(job: dict[str, Any]) -> list[str]:
    """Normalize a GitHub Actions job dependency to a list."""
    value = job.get("needs", [])
    if isinstance(value, str):
        return [value]
    assert isinstance(value, list), "job needs must be a string or list"
    return [str(item) for item in value]


def _effective_job_permissions(
    workflow: dict[Any, Any],
    job: dict[str, Any],
) -> dict[str, str]:
    """Resolve the job-level override that GitHub applies to one job."""
    permissions = job.get("permissions", workflow.get("permissions"))
    assert isinstance(permissions, dict), "effective job permissions must be a mapping"
    return {str(scope): str(access) for scope, access in permissions.items()}


def _assert_no_smoke_carrier(workflow_dir: Path = WORKFLOW_DIR) -> None:
    """Reject a smoke-only carrier under either accepted YAML suffix."""
    for path in _workflow_paths(workflow_dir):
        workflow = _load_yaml(path)
        jobs = workflow.get("jobs", {})
        assert isinstance(jobs, dict), f"{path.name} must define a jobs mapping"
        job_names = {str(job.get("name", job_id)) for job_id, job in jobs.items() if isinstance(job, dict)}
        assert path.stem != "merge-queue-smoke", f"obsolete smoke carrier: {path.name}"
        assert "merge-queue-smoke" not in jobs, f"obsolete smoke job id: {path.name}"
        assert "merge-queue-smoke" not in job_names, f"obsolete smoke context: {path.name}"


def _assert_real_job(workflow_name: str, job_id: str, job: dict[str, Any]) -> None:
    """Require one executable, fail-closed job body."""
    assert job.get("continue-on-error") in (None, False), f"{workflow_name}:{job_id} must not suppress failures"
    if job.get("uses"):
        return

    assert job.get("runs-on"), f"{workflow_name}:{job_id} must select a runner"
    steps = job.get("steps")
    assert isinstance(steps, list) and steps, f"{workflow_name}:{job_id} must execute real steps"
    for index, step in enumerate(steps):
        assert isinstance(step, dict), f"{workflow_name}:{job_id}:step-{index} must be a mapping"
        assert step.get("continue-on-error") in (None, False), (
            f"{workflow_name}:{job_id}:step-{index} must not suppress failures"
        )
        assert step.get("uses") or step.get("run"), (
            f"{workflow_name}:{job_id}:step-{index} must execute an action or command"
        )
        condition = step.get("if")
        if isinstance(condition, str) and any(
            token in condition
            for token in (
                "github.event_name",
                "github.event.pull_request",
                "github.event.merge_group",
            )
        ):
            step_name = str(step.get("name", f"step-{index}"))
            optional_key = (workflow_name, job_id, step_name)
            assert condition == OPTIONAL_STEP_EVENT_CONDITIONS.get(optional_key), (
                f"{workflow_name}:{job_id}:{step_name} has an unapproved event-specific condition"
            )


def _assert_dependency_reachability(
    workflow_name: str,
    jobs: dict[str, dict[str, Any]],
    required_job_id: str,
    dependency_id: str,
    visited: set[str],
) -> None:
    """Reject event-suppressed or missing dependencies of a required context."""
    assert dependency_id in jobs, f"{workflow_name}:{required_job_id} needs missing job {dependency_id}"
    if dependency_id in visited:
        return
    visited.add(dependency_id)

    dependency = jobs[dependency_id]
    condition = dependency.get("if")
    optional_key = (workflow_name, required_job_id, dependency_id)
    if optional_key in OPTIONAL_DEPENDENCY_CONDITIONS:
        assert condition == OPTIONAL_DEPENDENCY_CONDITIONS[optional_key]
    else:
        assert condition in (None, "always()"), (
            f"{workflow_name}:{required_job_id} depends on event-suppressed {dependency_id}"
        )

    for transitive_id in _needs(dependency):
        _assert_dependency_reachability(
            workflow_name,
            jobs,
            required_job_id,
            transitive_id,
            visited,
        )


def _required_context_owners(
    workflows: dict[str, dict[Any, Any]],
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Index protected-context owners for each protected event."""
    owners = {"pull_request": {}, "merge_group": {}}
    for workflow_name, workflow in workflows.items():
        triggers = _on_block(workflow)
        protected_events = owners.keys() & triggers.keys()
        if not protected_events:
            continue

        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
        for raw_job_id, raw_job in jobs.items():
            job_id = str(raw_job_id)
            assert isinstance(raw_job, dict), f"{workflow_name}:{job_id} must be a mapping"
            context = str(raw_job.get("name", job_id))
            if context not in EXPECTED_REQUIRED_CONTEXTS:
                continue
            for event_name in protected_events:
                owners[event_name].setdefault(context, []).append((workflow_name, job_id))
    return owners


def _validate_producer_contract(workflows: dict[str, dict[Any, Any]]) -> None:
    """Validate real producer parity and fail-closed graph invariants."""
    owners_by_event = _required_context_owners(workflows)
    for event_name, owners in owners_by_event.items():
        assert set(owners) == EXPECTED_REQUIRED_CONTEXTS, f"{event_name} required-context set differs from live policy"
        duplicates = {context: entries for context, entries in owners.items() if len(entries) != 1}
        assert not duplicates, f"{event_name} contexts must have one producer: {duplicates}"

    pr_owners = owners_by_event["pull_request"]
    merge_group_owners = owners_by_event["merge_group"]
    assert pr_owners == merge_group_owners, "pull-request and merge-group context owners differ"

    jobs_by_workflow: dict[str, dict[str, dict[str, Any]]] = {}

    for workflow_name in PRODUCER_WORKFLOWS:
        assert workflow_name in workflows, f"missing producer workflow {workflow_name}"
        workflow = workflows[workflow_name]
        triggers = _on_block(workflow)
        assert set(triggers) == {
            "pull_request",
            "merge_group",
            "push",
            "workflow_dispatch",
        }
        assert triggers["pull_request"] is None
        assert triggers["merge_group"] == {"types": ["checks_requested"]}
        assert triggers["push"]["branches"] == ["main"]

        assert workflow.get("permissions") == {"contents": "read"}
        assert workflow.get("concurrency") == {
            "group": EXPECTED_CONCURRENCY_GROUP,
            "cancel-in-progress": True,
        }

        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict) and jobs, f"{workflow_name} must define jobs"
        typed_jobs: dict[str, dict[str, Any]] = {}
        for raw_job_id, raw_job in jobs.items():
            job_id = str(raw_job_id)
            assert isinstance(raw_job, dict), f"{workflow_name}:{job_id} must be a mapping"
            typed_jobs[job_id] = raw_job
            _assert_real_job(workflow_name, job_id, raw_job)
            for dependency_id in _needs(raw_job):
                assert dependency_id in jobs, f"{workflow_name}:{job_id} needs missing job {dependency_id}"
        jobs_by_workflow[workflow_name] = typed_jobs

    for context_owners in pr_owners.values():
        workflow_name, required_job_id = context_owners[0]
        assert workflow_name in jobs_by_workflow, (
            f"required context is owned outside the reviewed producer set: {workflow_name}"
        )
        typed_jobs = jobs_by_workflow[workflow_name]
        required_job = typed_jobs[required_job_id]
        assert required_job.get("if") in (None, "always()"), (
            f"{workflow_name}:{required_job_id} suppresses a required event"
        )
        for dependency_id in _needs(required_job):
            _assert_dependency_reachability(
                workflow_name,
                typed_jobs,
                required_job_id,
                dependency_id,
                set(),
            )


def test_real_producers_emit_the_exact_required_set_for_pr_and_merge_group() -> None:
    """All 32 protected contexts have one event-neutral real producer."""
    _validate_producer_contract(_load_workflows())


def test_smoke_only_carrier_is_absent_for_both_yaml_suffixes() -> None:
    """No .yml or .yaml workflow can restore the obsolete smoke context."""
    _assert_no_smoke_carrier()


def test_smoke_carrier_detection_includes_yaml(tmp_path: Path) -> None:
    """The workflow inventory must not miss an alternate-extension carrier."""
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "hidden-smoke.yaml").write_text(
        "name: Hidden smoke\non:\n  merge_group:\n    types: [checks_requested]\n"
        "jobs:\n  merge-queue-smoke:\n    name: merge-queue-smoke\n"
        "    runs-on: ubuntu-latest\n    steps:\n      - run: 'true'\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="obsolete smoke"):
        _assert_no_smoke_carrier(workflow_dir)


def test_obsolete_merge_queue_policy_is_absent() -> None:
    """The staged producer work must not retain an impossible smoke policy."""
    assert not OBSOLETE_MERGE_QUEUE_POLICY.exists()


def _set_required_job_condition(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["_required.yml"]["jobs"]["lint"]["if"] = "github.event_name == 'pull_request'"


def _set_event_suppressed_need(workflows: dict[str, dict[Any, Any]]) -> None:
    jobs = workflows["_required.yml"]["jobs"]
    jobs["pr-only"] = {
        "if": "github.event_name == 'pull_request'",
        "runs-on": "ubuntu-latest",
        "steps": [{"run": "true"}],
    }
    jobs["lint"]["needs"] = "pr-only"


def _set_pr_only_validator_step(workflows: dict[str, dict[Any, Any]]) -> None:
    steps = workflows["comprehensive-tests.yml"]["jobs"]["validate-test-coverage"]["steps"]
    validator = next(step for step in steps if step.get("name") == "Validate test coverage")
    validator["if"] = "github.event_name == 'pull_request'"


def _set_missing_need(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["_required.yml"]["jobs"]["lint"]["needs"] = "missing-job"


def _set_continue_on_error(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["pre-commit.yml"]["jobs"]["pre-commit"]["steps"][0]["continue-on-error"] = True


def _set_expression_continue_on_error(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["pre-commit.yml"]["jobs"]["pre-commit"]["continue-on-error"] = "${{ github.event_name == 'merge_group' }}"


def _set_hollow_job(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["pre-commit.yml"]["jobs"]["pre-commit"]["steps"] = []


def _set_fork_colliding_concurrency(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["pre-commit.yml"]["concurrency"]["group"] = (
        "${{ github.workflow }}-${{ github.event_name }}-${{ github.head_ref || github.sha }}"
    )


def _add_yaml_duplicate_context(workflows: dict[str, dict[Any, Any]]) -> None:
    workflows["duplicate.yaml"] = {
        True: {
            "pull_request": None,
            "merge_group": {"types": ["checks_requested"]},
        },
        "jobs": {
            "duplicate-lint": {
                "name": "lint",
                "runs-on": "ubuntu-latest",
                "steps": [{"run": "true"}],
            }
        },
    }


@pytest.mark.parametrize(
    "mutate",
    [
        _set_required_job_condition,
        _set_event_suppressed_need,
        _set_pr_only_validator_step,
        _set_missing_need,
        _set_continue_on_error,
        _set_expression_continue_on_error,
        _set_hollow_job,
        _set_fork_colliding_concurrency,
        _add_yaml_duplicate_context,
    ],
    ids=[
        "required-job-event-condition",
        "event-suppressed-need",
        "pr-only-validator-step",
        "missing-need",
        "step-continue-on-error",
        "job-expression-continue-on-error",
        "hollow-job",
        "fork-pr-concurrency-collision",
        "yaml-duplicate-context-owner",
    ],
)
def test_contract_rejects_false_green_mutations(
    mutate: Callable[[dict[str, dict[Any, Any]]], None],
) -> None:
    """The guard must fail for common parity and false-green regressions."""
    workflows = deepcopy(_load_workflows())
    mutate(workflows)

    with pytest.raises(AssertionError):
        _validate_producer_contract(workflows)


def test_existing_push_and_manual_triggers_are_preserved() -> None:
    """Merge-group parity must not narrow main-push or manual behavior."""
    workflows = _load_workflows()
    for workflow_name in PRODUCER_WORKFLOWS[:3]:
        triggers = _on_block(workflows[workflow_name])
        assert triggers["push"] == {"branches": ["main"]}
        assert "workflow_dispatch" in triggers

    smoke_triggers = _on_block(workflows["workflow-smoke-test.yml"])
    assert smoke_triggers["push"]["branches"] == ["main"]
    assert set(smoke_triggers["push"]["paths"]) == {
        ".github/workflows/_required.yml",
        ".github/actions/setup-container/action.yml",
        ".github/actions/setup-uv/action.yml",
        ".github/workflows/comprehensive-test-pr-comments.yml",
        ".github/workflows/comprehensive-tests.yml",
        ".github/workflows/container-publish.yml",
        ".github/workflows/gradient-soak.yml",
        ".github/workflows/dependabot-uv-lock.yml",
        ".github/workflows/dependabot-uv-lock-writer.yml",
        ".github/workflows/pre-commit.yml",
        ".github/workflows/release.yml",
        ".github/workflows/security.yml",
        ".github/workflows/validate-configs.yml",
        ".github/workflows/workflow-smoke-test.yml",
        "Dockerfile",
        "docker-compose.yml",
        "justfile",
        "scripts/ci/commit_generated_dependencies.py",
        "scripts/ci/comprehensive_pr_comment.py",
        "scripts/ci/dependency_audit_contract.py",
        "scripts/ci/ensure-podman-runtime.sh",
        "scripts/ci/parse_pip_audit.py",
        "scripts/sync_requirements.py",
        "tests/scripts/test_commit_generated_dependencies.py",
        "tests/scripts/test_comprehensive_pr_comment.py",
        "tests/scripts/test_comprehensive_pr_comment_publisher.py",
        "tests/scripts/test_dependency_audit_contract.py",
        "tests/scripts/test_parse_pip_audit.py",
        "tests/smoke/test_comprehensive_pr_comments_workflow_properties.py",
        "tests/smoke/test_comprehensive_report_workflow_properties.py",
        "tests/smoke/test_comprehensive_tests_workflow_properties.py",
        "tests/smoke/test_container_runtime_workflow_properties.py",
        "tests/smoke/test_dependabot_uv_lock_workflow_properties.py",
        "tests/smoke/test_gradient_soak_workflow_properties.py",
        "tests/smoke/test_merge_queue_workflow_properties.py",
        "tests/smoke/test_pre_commit_workflow_properties.py",
        "tests/smoke/test_security_workflow_properties.py",
        "tests/smoke/test_validate_configs_workflow_properties.py",
        "tests/workflows/test_security_dependency_audit.py",
    }
    assert "workflow_dispatch" in smoke_triggers


def test_merge_group_cannot_receive_write_scope() -> None:
    """Merge-group jobs stay read-only; only trusted PR comments can write."""
    workflows = _load_workflows()
    for workflow_name in PRODUCER_WORKFLOWS:
        workflow = workflows[workflow_name]
        assert workflow.get("permissions") == {"contents": "read"}
        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict)
        for job_id, job in jobs.items():
            assert isinstance(job, dict)
            assert _effective_job_permissions(workflow, job) == {"contents": "read"}, (
                f"{workflow_name}:{job_id} grants write access on merge_group"
            )

    comment_workflow = workflows[PR_COMMENT_WORKFLOW]
    assert _on_block(comment_workflow) == {
        "workflow_run": {
            "workflows": ["Comprehensive Tests"],
            "types": ["completed"],
        },
        "repository_dispatch": {"types": ["dependabot-comprehensive-test-comment"]},
    }
    assert comment_workflow.get("permissions") == {"contents": "read"}

    comment_jobs = comment_workflow.get("jobs")
    assert isinstance(comment_jobs, dict)
    assert set(comment_jobs) == {"resolve-pr-context", "post-pr-comments"}
    resolver_job = comment_jobs["resolve-pr-context"]
    resolver_condition = str(resolver_job.get("if"))
    assert "github.event_name == 'repository_dispatch'" in resolver_condition
    assert "github.event.workflow_run.event == 'pull_request'" in resolver_condition
    assert resolver_job.get("permissions") == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "read",
    }
    comment_job = comment_jobs["post-pr-comments"]
    comment_condition = str(comment_job.get("if"))
    assert "needs.resolve-pr-context.result == 'success'" in comment_condition
    assert "needs.resolve-pr-context.outputs.pr_number != ''" in comment_condition
    assert "needs.resolve-pr-context.outputs.writer_ready == 'true'" in comment_condition
    assert comment_job.get("needs") == "resolve-pr-context"
    assert comment_job.get("concurrency", {}).get("group") == (
        "comprehensive-test-pr-comment-pr-${{ needs.resolve-pr-context.outputs.pr_number }}"
    )
    assert comment_job.get("concurrency", {}).get("queue") == "max"
    assert comment_job.get("concurrency", {}).get("cancel-in-progress") is False
    assert comment_job.get("permissions") == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }

    uses = [str(step.get("uses", "")) for step in comment_job.get("steps", [])]
    checkouts = [
        step for step in comment_job.get("steps", []) if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkouts) == 1
    assert checkouts[0].get("with") == {
        "repository": "${{ github.repository }}",
        "ref": "${{ github.sha }}",
        "fetch-depth": 1,
        "persist-credentials": False,
        "sparse-checkout": (
            ".github/workflows/comprehensive-tests.yml\n"
            "scripts/ci/comprehensive_pr_comment.py\n"
            "scripts/ci/comprehensive_report.py\n"
        ),
        "sparse-checkout-cone-mode": False,
    }
    assert all(not action.startswith("./") for action in uses)


def test_release_workflow_remains_tag_or_manual_only() -> None:
    """Queue readiness must not execute the real publishing workflow."""
    triggers = _on_block(_load_yaml(WORKFLOW_DIR / "release.yml"))

    assert triggers["push"] == {"tags": ["v*"]}
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
    assert "merge_group" not in triggers


def test_merge_queue_regression_runs_in_a_real_required_context() -> None:
    """The regression itself must run in the workflow-property producer."""
    workflow = (WORKFLOW_DIR / "workflow-smoke-test.yml").read_text(encoding="utf-8")
    assert "tests/smoke/test_merge_queue_workflow_properties.py" in workflow
