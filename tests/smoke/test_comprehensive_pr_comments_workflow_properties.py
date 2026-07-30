#!/usr/bin/env python3
"""Fail-closed properties for the trusted comprehensive PR commenter."""

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "comprehensive-test-pr-comments.yml"


def _load_job() -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
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
    assert post_step.get("if") == "always()"

    environment = post_step.get("env", {})
    assert environment["REPORT_AVAILABLE"] == ("${{ steps.discover_artifacts.outputs.has_report }}")
    assert environment["METRICS_AVAILABLE"] == ("${{ steps.discover_artifacts.outputs.has_metrics }}")
    assert environment["REPORT_DOWNLOAD_OUTCOME"] == ("${{ steps.download-report.outcome }}")
    assert environment["METRICS_DOWNLOAD_OUTCOME"] == ("${{ steps.download-metrics.outcome }}")
    assert environment["WORKFLOW_RUN_URL"] == ("${{ github.event.workflow_run.html_url }}")
    assert environment["WORKFLOW_CONCLUSION"] == ("${{ github.event.workflow_run.conclusion }}")

    script = post_step["with"]["script"]
    assert "REPORT_DOWNLOAD_OUTCOME" in script
    assert "WORKFLOW_RUN_URL" in script
    assert "WORKFLOW_CONCLUSION" in script
    assert "🧪 Comprehensive Test Results" in script
    assert "report unavailable" in script.lower()


def test_commenter_rejects_stale_runs_and_verdict_mismatches() -> None:
    job = _load_job()
    concurrency = job.get("concurrency", {})
    assert "pull_requests[0].number" in str(concurrency.get("group", ""))
    assert concurrency.get("cancel-in-progress") is True

    post_step = next(step for step in job["steps"] if step.get("name") == "Post or update PR comments")
    script = post_step["with"]["script"]

    assert "github.rest.pulls.get" in script
    assert "workflow_run.pull_requests[0]" in script
    assert "head?.sha" in script
    assert "pullRequest.head.sha" in script
    assert "Stale workflow run" in script

    assert "WORKFLOW_CONCLUSION" in script
    assert "✅ PASS —" in script
    assert "❌ FAIL —" in script
    assert "does not match the workflow conclusion" in script
    assert "try {" in script
    assert "catch" in script
