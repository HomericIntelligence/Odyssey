#!/usr/bin/env python3
"""Trust-boundary properties for deterministic Dependabot generation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
SETUP_UV_ACTION = REPO_ROOT / ".github" / "actions" / "setup-uv" / "action.yml"
PRODUCER_WORKFLOW = WORKFLOW_DIR / "dependabot-uv-lock.yml"
WRITER_WORKFLOW = WORKFLOW_DIR / "dependabot-uv-lock-writer.yml"
PUBLISHER_SCRIPT = REPO_ROOT / "scripts" / "ci" / "commit_generated_dependencies.py"
SETUP_UV_V9 = "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
UPLOAD_ARTIFACT_V7 = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
CHECKOUT_V7 = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
UV_VERSION = "0.12.0"
SYNC_COMMAND = "python scripts/sync_requirements.py --repo-root ."
CHECK_COMMAND = "python scripts/sync_requirements.py --check --repo-root ."
ALLOWLIST = (
    "_required.yml",
    "comprehensive-tests.yml",
    "pre-commit.yml",
    "workflow-smoke-test.yml",
)


def _load(path: Path) -> dict[Any, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _on(workflow: dict[Any, Any]) -> dict[str, Any]:
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _steps(workflow: dict[Any, Any], job_id: str) -> list[dict[str, Any]]:
    steps = workflow["jobs"][job_id]["steps"]
    assert isinstance(steps, list)
    return steps


def _run(step: dict[str, Any]) -> str:
    return str(step.get("run", "")).strip()


def _script_allowlist() -> tuple[str, ...]:
    tree = ast.parse(PUBLISHER_SCRIPT.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REQUIRED_WORKFLOWS" for target in node.targets)
    )
    value = ast.literal_eval(assignment.value)
    assert isinstance(value, tuple)
    return value


def test_pr_side_generator_is_strictly_read_only_and_bot_only() -> None:
    workflow = _load(PRODUCER_WORKFLOW)
    job = workflow["jobs"]["regenerate-uv-lock"]
    condition = str(job["if"])

    assert workflow["permissions"] == {"contents": "read"}
    assert "github.actor == 'dependabot[bot]'" in condition
    assert "github.event.pull_request.user.login == 'dependabot[bot]'" in condition
    assert "github.event.pull_request.head.repo.full_name == github.repository" in condition
    assert "contains(" not in condition

    checkout = _steps(workflow, "regenerate-uv-lock")[0]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"

    source = PRODUCER_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "contents: write",
        "actions: write",
        "GITHUB_TOKEN",
        "git commit",
        "git push",
        "createCommitOnBranch",
        "workflow_dispatch",
    ):
        assert forbidden not in source


def test_generator_preserves_deterministic_lock_and_export_contract() -> None:
    steps = _steps(_load(PRODUCER_WORKFLOW), "regenerate-uv-lock")
    all_runs = "\n".join(_run(step) for step in steps)

    assert "uv lock --upgrade" not in all_runs
    lock_regeneration = next(step for step in steps if step.get("name") == "Regenerate uv.lock deterministically")
    assert _run(lock_regeneration) == "uv lock"
    assert any(_run(step) == SYNC_COMMAND for step in steps)
    environment_sync = next(step for step in steps if step.get("name") == "Synchronize locked environment")
    assert _run(environment_sync) == "uv sync --locked"
    verification = next(step for step in steps if step.get("name") == "Verify generated dependency files")
    assert "uv lock --check" in _run(verification)
    assert CHECK_COMMAND in _run(verification)
    assert " snapshot " in f" {_run(verification)} "
    assert verification["env"]["DEPENDENCY_SNAPSHOT"] == ("${{ runner.temp }}/dependabot-dependency-snapshot.json")

    step_names = [str(step.get("name", "")) for step in steps]
    assert step_names.index("Regenerate uv.lock deterministically") < step_names.index("Synchronize locked environment")
    assert step_names.index("Synchronize locked environment") < step_names.index("Regenerate requirements exports")
    assert step_names.index("Regenerate requirements exports") < step_names.index("Verify generated dependency files")
    assert step_names.index("Verify generated dependency files") < step_names.index(
        "Package exact untrusted result artifact"
    )


def test_generator_always_uploads_exact_named_artifact_and_manifest() -> None:
    steps = _steps(_load(PRODUCER_WORKFLOW), "regenerate-uv-lock")
    package = next(step for step in steps if step.get("name") == "Package exact untrusted result artifact")
    upload = next(step for step in steps if step.get("name") == "Upload exact dependency result artifact")

    assert package["if"] == "always()"
    assert " package " in f" {_run(package)} "
    assert '--snapshot-path "$DEPENDENCY_SNAPSHOT"' in _run(package)
    assert package["env"]["DEPENDENCY_SNAPSHOT"] == ("${{ runner.temp }}/dependabot-dependency-snapshot.json")
    assert package["env"]["PRODUCER_STATUS"] == "${{ job.status }}"
    assert upload["if"] == "always()"
    assert upload["uses"] == UPLOAD_ARTIFACT_V7
    assert upload["with"] == {
        "name": "dependabot-generated-dependencies",
        "path": "${{ runner.temp }}/dependabot-generated-dependencies",
        "if-no-files-found": "error",
        "retention-days": 1,
    }


def test_default_branch_writer_is_the_only_write_scope_and_never_uses_pr_code() -> None:
    producer = _load(PRODUCER_WORKFLOW)
    writer = _load(WRITER_WORKFLOW)
    assert producer["permissions"] == {"contents": "read"}
    assert _on(writer) == {
        "workflow_run": {
            "workflows": ["Regenerate dependency files on Dependabot PRs"],
            "types": ["completed"],
        }
    }
    assert writer["permissions"] == {
        "actions": "write",
        "contents": "write",
        "pull-requests": "read",
    }
    assert set(writer["jobs"]) == {"publish-generated-dependencies"}
    job = writer["jobs"]["publish-generated-dependencies"]
    condition = str(job["if"])
    assert "github.event.workflow_run.event == 'pull_request'" in condition
    assert "github.event.workflow_run.conclusion == 'success'" in condition
    assert "github.event.workflow_run.actor.login == 'dependabot[bot]'" in condition
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in condition
    assert "github.event.workflow_run.path == '.github/workflows/dependabot-uv-lock.yml'" in condition
    concurrency = writer["concurrency"]
    assert "workflow_run.head_sha" in concurrency["group"]
    assert "workflow_run.id" not in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False

    steps = _steps(writer, "publish-generated-dependencies")
    assert [step["name"] for step in steps] == [
        "Checkout trusted default-branch publisher",
        "Install trusted pinned uv",
        "Authenticate source, publish signed child, and dispatch read-only checks",
    ]
    checkout = steps[0]
    assert checkout["uses"] == CHECKOUT_V7
    assert checkout["with"] == {
        "repository": "${{ github.repository }}",
        "ref": "${{ github.sha }}",
        "fetch-depth": 1,
        "persist-credentials": False,
    }
    assert "pull_requests" not in str(checkout)
    assert "head_sha" not in str(checkout)

    trusted_uv = steps[1]
    assert trusted_uv["uses"] == SETUP_UV_V9
    assert trusted_uv["with"] == {
        "version": UV_VERSION,
        "enable-cache": False,
    }

    publish = steps[2]
    assert _run(publish) == (
        'python scripts/ci/commit_generated_dependencies.py publish --event-path "$GITHUB_EVENT_PATH"'
    )
    assert publish["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert publish["env"]["TRUSTED_DEFAULT_SHA"] == "${{ github.sha }}"
    assert "ARTIFACT_ROOT" not in publish["env"]

    approved_run_commands = {_run(publish)}
    for privileged_job in writer["jobs"].values():
        for step in privileged_job["steps"]:
            action = str(step.get("uses", ""))
            command = _run(step)
            assert not action.startswith("./")
            assert action in {"", CHECKOUT_V7, SETUP_UV_V9}
            assert command in {"", *approved_run_commands}
            assert "workflow_run.head_sha" not in str(step.get("with", {}))

    source = WRITER_WORKFLOW.read_text(encoding="utf-8")
    assert "actions/download-artifact@" not in source


def test_dispatch_allowlist_is_literal_exact_and_every_target_is_read_only() -> None:
    assert _script_allowlist() == ALLOWLIST
    source = PUBLISHER_SCRIPT.read_text(encoding="utf-8")
    assert "createCommitOnBranch" in source
    assert "expectedHeadOid" in source
    assert "wasSignedByGitHub" in source
    assert "DCO_BODY" in source

    for workflow_name in ALLOWLIST:
        workflow = _load(WORKFLOW_DIR / workflow_name)
        assert "workflow_dispatch" in _on(workflow)
        assert workflow["permissions"] == {"contents": "read"}
        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict)
        for job in jobs.values():
            assert isinstance(job, dict)
            assert "write" not in job.get("permissions", {}).values()


def test_writer_dispatches_the_comment_monitor_beside_required_checks() -> None:
    source = PUBLISHER_SCRIPT.read_text(encoding="utf-8")

    assert 'COMMENT_WORKFLOW = "comprehensive-test-pr-comments.yml"' in source
    assert "def dispatch_comment_workflow(" in source
    assert '"source_head_sha": commit_oid' in source
    assert '"source_head_branch": head_ref' in source
    assert source.index("dispatched = dispatch_required_workflows(") < source.index(
        "dispatch_comment_workflow(\n        repository=context.repository,"
    )
    assert "comprehensive-test-pr-comments.yml" not in ALLOWLIST


def test_dependency_generation_uses_one_pinned_uv_version() -> None:
    setup_action = _load(SETUP_UV_ACTION)
    assert setup_action["inputs"]["uv-version"]["default"] == UV_VERSION
    setup_step = setup_action["runs"]["steps"][0]
    assert setup_step["uses"] == SETUP_UV_V9
    assert setup_step["with"]["version"] == "${{ inputs.uv-version }}"

    expected_jobs = {
        "dependabot-uv-lock.yml": "regenerate-uv-lock",
        "dependabot-uv-lock-writer.yml": "publish-generated-dependencies",
        "comprehensive-tests.yml": "validate-dep-sync",
        "_required.yml": "deps-version-sync",
    }
    for workflow_name, job_id in expected_jobs.items():
        install = next(
            step
            for step in _steps(_load(WORKFLOW_DIR / workflow_name), job_id)
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
        )
        assert install["uses"] == SETUP_UV_V9
        assert install["with"]["version"] == UV_VERSION


def test_required_dependency_sync_jobs_check_lock_and_exports() -> None:
    for workflow_name, job_id in {
        "comprehensive-tests.yml": "validate-dep-sync",
        "_required.yml": "deps-version-sync",
    }.items():
        all_runs = "\n".join(_run(step) for step in _steps(_load(WORKFLOW_DIR / workflow_name), job_id))
        assert "uv lock --check" in all_runs
        assert CHECK_COMMAND in all_runs


def test_python_producer_installs_pinned_uv_before_running_full_suite() -> None:
    """The full Python suite exercises the real frozen requirements exporter."""
    steps = _steps(_load(WORKFLOW_DIR / "comprehensive-tests.yml"), "test-python")
    install_index, install = next(
        (index, step) for index, step in enumerate(steps) if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    )
    test_index = next(index for index, step in enumerate(steps) if step.get("name") == "Run Python tests")

    assert install_index < test_index
    assert install["uses"] == SETUP_UV_V9
    assert install["with"] == {"version": UV_VERSION, "enable-cache": False}


def test_workflow_smoke_tracks_both_halves_of_the_trust_boundary() -> None:
    workflow_path = WORKFLOW_DIR / "workflow-smoke-test.yml"
    source = workflow_path.read_text(encoding="utf-8")
    workflow = _load(workflow_path)
    for path in (
        ".github/workflows/dependabot-uv-lock.yml",
        ".github/workflows/dependabot-uv-lock-writer.yml",
        ".github/workflows/comprehensive-test-pr-comments.yml",
        "scripts/ci/commit_generated_dependencies.py",
        "scripts/sync_requirements.py",
        "tests/scripts/test_commit_generated_dependencies.py",
        "tests/smoke/test_dependabot_uv_lock_workflow_properties.py",
        "tests/smoke/test_comprehensive_pr_comments_workflow_properties.py",
    ):
        assert path in source

    all_runs = "\n".join(_run(step) for step in _steps(workflow, "smoke-test-other-workflows"))
    assert "tests/scripts/test_commit_generated_dependencies.py" in all_runs
    assert "tests/smoke/test_dependabot_uv_lock_workflow_properties.py" in all_runs
    assert "tests/smoke/test_comprehensive_pr_comments_workflow_properties.py" in all_runs
