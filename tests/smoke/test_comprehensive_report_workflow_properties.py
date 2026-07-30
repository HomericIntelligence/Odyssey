#!/usr/bin/env python3
"""Workflow contracts for the fail-closed comprehensive Test Report.

These tests intentionally inspect behavior-bearing workflow structure rather
than report prose.  They prevent a newly added job or a setup-time artifact
failure from silently escaping the aggregate gate.
"""

import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "comprehensive-tests.yml"
WORKFLOW_SMOKE_PATH = REPO_ROOT / ".github" / "workflows" / "workflow-smoke-test.yml"

EXPECTED_UPSTREAM_JOB_IDS = {
    "mojo-syntax-check",
    "mojo-compilation",
    "example-backward-tests",
    "training-smoke",
    "validate-test-coverage",
    "audit-shared-links",
    "validate-dep-sync",
    "test-mojo-comprehensive",
    "test-configs",
    "test-benchmarks",
    "test-core-layers",
    "test-python",
    "code-quality",
    "simd-analysis",
    "build-validation",
    "gradient-tests",
    "gradient-coverage",
    "test-data-utilities",
    "test-metrics",
}

EXPECTED_PRODUCERS_BY_JOB = {
    "test-mojo-comprehensive": {
        "comprehensive-core-tensors-loss",
        "comprehensive-core-gradient-utils",
        "comprehensive-data",
        "comprehensive-autograd-tensor-base",
        "comprehensive-models-misc",
        "comprehensive-integration-bench",
    },
    "test-configs": {"configs"},
    "test-benchmarks": {"benchmarks"},
    "test-core-layers": {"core-layers"},
    "test-python": {"python-tests"},
    "gradient-tests": {"gradient-checking"},
    "test-data-utilities": {"data-utilities"},
}


def _load_workflow() -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    assert isinstance(workflow.get("jobs"), dict)
    return workflow


def _steps_for(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _serialized(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def test_test_report_needs_exactly_every_other_top_level_job() -> None:
    jobs = _load_workflow()["jobs"]
    assert set(jobs) - {"test-report"} == EXPECTED_UPSTREAM_JOB_IDS

    needs = jobs["test-report"].get("needs")
    assert isinstance(needs, list)
    assert len(needs) == len(set(needs)), "test-report.needs must not contain duplicate job IDs"
    assert set(needs) == EXPECTED_UPSTREAM_JOB_IDS
    assert set(needs) == set(jobs) - {"test-report"}


def test_each_result_producer_writes_an_always_run_outcome_manifest() -> None:
    jobs = _load_workflow()["jobs"]

    for job_id, producers in EXPECTED_PRODUCERS_BY_JOB.items():
        manifest_steps = [step for step in _steps_for(jobs[job_id]) if "outcome-manifest.json" in _serialized(step)]
        assert len(manifest_steps) == 1, f"{job_id} must have exactly one step that writes outcome-manifest.json"

        manifest_step = manifest_steps[0]
        serialized = _serialized(manifest_step)
        assert manifest_step.get("if") == "always()"
        assert "${{ job.status }}" in serialized
        assert "mkdir -p test-results" in serialized
        assert '"producer"' in serialized
        assert '"job_id"' in serialized
        assert '"status"' in serialized

        if job_id == "test-mojo-comprehensive":
            assert "${{ matrix.group }}" in serialized
            assert "comprehensive-" in serialized
        else:
            producer = next(iter(producers))
            assert producer in serialized
            assert job_id in serialized


def test_each_result_producer_upload_is_fail_closed_and_always_runs() -> None:
    jobs = _load_workflow()["jobs"]

    for job_id in EXPECTED_PRODUCERS_BY_JOB:
        uploads = [
            step
            for step in _steps_for(jobs[job_id])
            if "actions/upload-artifact@" in str(step.get("uses", ""))
            and str(step.get("with", {}).get("name", "")).startswith("test-results-")
        ]
        assert len(uploads) == 1, f"{job_id} must upload exactly one test-results artifact"

        upload = uploads[0]
        assert upload.get("if") == "always()"
        assert upload.get("with", {}).get("if-no-files-found") == "error"
        assert "test-results" in str(upload.get("with", {}).get("path", ""))


def test_report_job_passes_needs_json_to_the_stdlib_validator() -> None:
    report_job = _load_workflow()["jobs"]["test-report"]
    serialized = _serialized(report_job)

    assert "${{ toJSON(needs) }}" in serialized
    assert "scripts/ci/comprehensive_report.py" in serialized
    validator_step = next(
        step for step in _steps_for(report_job) if "scripts/ci/comprehensive_report.py" in str(step.get("run", ""))
    )
    validator_command = validator_step["run"]
    assert "--artifacts-dir all-results" in validator_command
    assert "--output test-report.md" in validator_command
    assert '--event-name "$EVENT_NAME"' in validator_command
    assert '--run-extended "$RUN_EXTENDED"' in validator_command

    report_uploads = [
        step
        for step in _steps_for(report_job)
        if "actions/upload-artifact@" in str(step.get("uses", ""))
        and step.get("with", {}).get("name") == "comprehensive-test-report"
    ]
    assert len(report_uploads) == 1
    upload = report_uploads[0]
    assert upload.get("if") == "always()"
    assert upload.get("with", {}).get("path") == "test-report.md"
    assert upload.get("with", {}).get("if-no-files-found") == "error"


def test_result_download_preserves_artifact_directories_for_recursive_discovery() -> None:
    report_job = _load_workflow()["jobs"]["test-report"]
    downloads = [step for step in _steps_for(report_job) if "actions/download-artifact@" in str(step.get("uses", ""))]
    assert len(downloads) == 1
    assert downloads[0].get("with", {}).get("merge-multiple") is False


def test_workflow_smoke_job_runs_report_property_suites() -> None:
    smoke_workflow = WORKFLOW_SMOKE_PATH.read_text(encoding="utf-8")
    assert "test_comprehensive_report_workflow_properties.py" in smoke_workflow
    assert "test_comprehensive_pr_comments_workflow_properties.py" in smoke_workflow
