#!/usr/bin/env python3
"""Fail-closed properties for the trusted comprehensive PR commenter."""

from pathlib import Path
import re
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "comprehensive-test-pr-comments.yml"
COMPREHENSIVE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "comprehensive-tests.yml"


def _load_workflow() -> dict[Any, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _on_block(workflow: dict[Any, Any]) -> dict[str, Any]:
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _load_job() -> dict[str, Any]:
    workflow = _load_workflow()
    job = workflow["jobs"]["post-pr-comments"]
    assert isinstance(job, dict)
    return job


def test_missing_report_replaces_stale_green_comment_with_failure_fallback() -> None:
    job = _load_job()
    steps = job["steps"]

    downloads = {
        str(step.get("id", "")): step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    }
    assert set(downloads) == {"download-metrics", "download-report"}
    assert downloads["download-metrics"].get("continue-on-error") is not True
    assert downloads["download-report"].get("continue-on-error") is not True
    assert "discover_artifacts.outputs.has_metrics" in downloads["download-metrics"]["if"]
    assert "always()" in downloads["download-report"]["if"]
    assert "discover_artifacts.outputs.has_report" in downloads["download-report"]["if"]

    discovery_step = next(step for step in steps if step.get("id") == "discover_artifacts")
    assert "actions/github-script@" in discovery_step["uses"]
    discovery_script = discovery_step["with"]["script"]
    assert "listWorkflowRunArtifacts" in discovery_script
    assert "has_metrics" in discovery_script
    assert "has_report" in discovery_script

    post_step = next(step for step in steps if step.get("name") == "Post or update PR comments")
    assert post_step.get("if") == "always() && steps.resolve-context.outcome == 'success'"

    environment = post_step.get("env", {})
    assert environment["REPORT_AVAILABLE"] == ("${{ steps.discover_artifacts.outputs.has_report }}")
    assert environment["METRICS_AVAILABLE"] == ("${{ steps.discover_artifacts.outputs.has_metrics }}")
    assert environment["REPORT_DOWNLOAD_OUTCOME"] == ("${{ steps.download-report.outcome }}")
    assert environment["METRICS_DOWNLOAD_OUTCOME"] == ("${{ steps.download-metrics.outcome }}")
    assert environment["WORKFLOW_RUN_URL"] == ("${{ steps.resolve-context.outputs.source_run_url }}")
    assert environment["WORKFLOW_CONCLUSION"] == ("${{ steps.resolve-context.outputs.source_run_conclusion }}")
    assert environment["SOURCE_HEAD_SHA"] == ("${{ steps.resolve-context.outputs.source_head_sha }}")

    script = post_step["with"]["script"]
    assert "REPORT_DOWNLOAD_OUTCOME" in script
    assert "WORKFLOW_RUN_URL" in script
    assert "WORKFLOW_CONCLUSION" in script
    assert "🧪 Comprehensive Test Results" in script
    assert "report unavailable" in script.lower()


def test_commenter_rejects_stale_runs_and_verdict_mismatches() -> None:
    job = _load_job()
    concurrency = job.get("concurrency", {})
    assert "workflow_run.head_sha" in str(concurrency.get("group", ""))
    assert "pull_requests[0].number" not in str(concurrency.get("group", ""))
    assert concurrency.get("cancel-in-progress") is True

    post_step = next(step for step in job["steps"] if step.get("name") == "Post or update PR comments")
    script = post_step["with"]["script"]

    assert "github.rest.pulls.get" in script
    assert "SOURCE_HEAD_SHA" in script
    assert "workflow_run.pull_requests[0]" not in script
    assert "pullRequest.head.sha" in script
    assert "Stale workflow run" in script

    assert "WORKFLOW_CONCLUSION" in script
    assert "✅ PASS —" in script
    assert "❌ FAIL —" in script
    assert "does not match the workflow conclusion" in script
    assert "try {" in script
    assert "catch" in script


def test_writer_dispatches_commenter_as_a_sibling_bound_to_the_exact_comprehensive_run() -> None:
    workflow = _load_workflow()
    triggers = _on_block(workflow)
    dispatch = triggers.get("workflow_dispatch")
    assert dispatch == {
        "inputs": {
            "source_head_sha": {
                "description": "Exact Dependabot head whose Comprehensive run must be reported",
                "required": True,
                "type": "string",
            },
            "source_head_branch": {
                "description": "Exact same-repository Dependabot branch",
                "required": True,
                "type": "string",
            },
        }
    }

    job = _load_job()
    condition = str(job["if"])
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "github.event.workflow_run.event == 'pull_request'" in condition
    assert "github.event.workflow_run.event == 'workflow_dispatch'" not in condition

    concurrency = str(job["concurrency"]["group"])
    assert "inputs.source_head_sha" in concurrency
    assert "github.event.workflow_run.head_sha" in concurrency

    resolve = next(step for step in job["steps"] if step.get("id") == "resolve-context")
    assert resolve["env"] == {
        "SOURCE_HEAD_SHA": "${{ inputs.source_head_sha }}",
        "SOURCE_HEAD_BRANCH": "${{ inputs.source_head_branch }}",
    }
    script = resolve["with"]["script"]
    assert "context.eventName === 'workflow_dispatch'" in script
    assert "listWorkflowRuns" in script
    assert "workflow_id: 'comprehensive-tests.yml'" in script
    assert "candidateRun.head_sha === sourceHeadSha" in script
    assert "candidateRun.head_branch === sourceHeadBranch" in script
    assert "matches.length !== 1" in script
    assert "github.rest.actions.getWorkflowRun" in script
    assert "run.status !== 'completed'" in script
    assert "Comprehensive run did not complete" in script
    assert "run.path !== '.github/workflows/comprehensive-tests.yml'" in script
    assert "run.event !== 'workflow_dispatch'" in script
    assert "setSourceFailure" in script
    assert "core.setOutput('source_ready', 'false')" in script
    assert "core.setOutput('source_error', message)" in script

    assert "listPullRequestsAssociatedWithCommit" in script
    assert "commit_sha: sourceHeadSha" in script
    assert "pullRequest.state === 'open'" in script
    assert "pullRequest.head.sha === sourceHeadSha" in script
    assert "candidates.length !== 1" in script
    assert "github.rest.pulls.get" in script
    assert "pullRequest.head.sha !== sourceHeadSha" in script
    assert "context.eventName === 'workflow_dispatch'" in script
    assert "pullRequest.user.login !== 'dependabot[bot]'" in script
    assert "pullRequest.head.ref !== sourceHeadBranch" in script
    assert "pullRequest.head.repo.full_name !== expectedRepository" in script
    assert "core.setFailed" in script
    assert "core.setOutput('pr_number'" in script
    assert "core.setOutput('source_run_id'" in script
    assert "core.setOutput('source_run_url'" in script
    assert "core.setOutput('source_run_conclusion'" in script

    for step in job["steps"][1:]:
        assert "steps.resolve-context.outcome == 'success'" in str(step.get("if", ""))

    discovery = next(step for step in job["steps"] if step.get("id") == "discover_artifacts")
    assert "steps.resolve-context.outputs.source_run_id" in discovery["env"]["SOURCE_RUN_ID"]
    assert "run_id: Number(process.env.SOURCE_RUN_ID)" in discovery["with"]["script"]

    post = next(step for step in job["steps"] if step.get("name") == "Post or update PR comments")
    assert post["env"]["PR_NUMBER"] == "${{ steps.resolve-context.outputs.pr_number }}"
    assert post["env"]["WORKFLOW_RUN_URL"] == "${{ steps.resolve-context.outputs.source_run_url }}"
    assert post["env"]["WORKFLOW_CONCLUSION"] == ("${{ steps.resolve-context.outputs.source_run_conclusion }}")
    assert post["env"]["SOURCE_RESOLUTION_ERROR"] == ("${{ steps.resolve-context.outputs.source_error }}")
    assert "source_ready" not in str(post["if"])
    assert "SOURCE_RESOLUTION_ERROR" in post["with"]["script"]

    enforcement = next(step for step in job["steps"] if step.get("name") == "Enforce trusted source resolution")
    assert "always()" in enforcement["if"]
    assert "outputs.source_ready != 'true'" in enforcement["if"]
    assert "exit 1" in enforcement["run"]


def test_comment_monitor_budget_exceeds_the_declared_comprehensive_needs_dag() -> None:
    job = _load_job()
    resolve = next(step for step in job["steps"] if step.get("id") == "resolve-context")
    script = resolve["with"]["script"]
    attempt_match = re.search(r"attempt < (\d+)", script)
    interval_match = re.search(r"setTimeout\(resolve, (\d+)\)", script)
    assert attempt_match is not None
    assert interval_match is not None

    poll_minutes = int(attempt_match.group(1)) * int(interval_match.group(1)) / 1000 / 60
    comprehensive = yaml.safe_load(COMPREHENSIVE_WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = comprehensive["jobs"]
    assert all("timeout-minutes" in candidate for candidate in jobs.values())

    bounds: dict[str, int] = {}

    def declared_path_minutes(job_id: str) -> int:
        if job_id in bounds:
            return bounds[job_id]
        candidate = jobs[job_id]
        raw_needs = candidate.get("needs", [])
        needs = [raw_needs] if isinstance(raw_needs, str) else raw_needs
        upstream = max(
            (declared_path_minutes(dependency) for dependency in needs),
            default=0,
        )
        bounds[job_id] = upstream + candidate["timeout-minutes"]
        return bounds[job_id]

    comprehensive_bound = declared_path_minutes("test-report")
    assert poll_minutes >= comprehensive_bound + 10
    assert job["timeout-minutes"] >= poll_minutes + 10


def test_trusted_commenter_never_checks_out_or_executes_pr_code() -> None:
    job = _load_job()
    uses = [str(step.get("uses", "")) for step in job["steps"]]
    assert all(not action.startswith("actions/checkout@") for action in uses)
    assert all(not action.startswith("./") for action in uses)
