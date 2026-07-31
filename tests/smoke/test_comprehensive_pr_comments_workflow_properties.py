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


def _load_resolver_job() -> dict[str, Any]:
    workflow = _load_workflow()
    job = workflow["jobs"]["resolve-pr-context"]
    assert isinstance(job, dict)
    return job


def test_write_scoped_commenter_never_consumes_pr_artifacts() -> None:
    job = _load_job()
    steps = job["steps"]

    serialized = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "actions/download-artifact@" not in serialized
    assert "listWorkflowRunArtifacts" not in serialized
    assert "downloadArtifact" not in serialized
    assert "/actions/artifacts" not in serialized
    assert "metrics-pr-comment.md" not in serialized
    assert "test-report.md" not in serialized
    assert "reports/metrics" not in serialized
    assert "reports/comprehensive" not in serialized

    collect_step = next(step for step in steps if step.get("id") == "collect-jobs")
    assert "actions/github-script@" in collect_step["uses"]
    assert collect_step["env"]["SOURCE_RUN_ID"] == ("${{ needs.resolve-pr-context.outputs.source_run_id }}")
    assert collect_step["env"]["SOURCE_HEAD_SHA"] == ("${{ needs.resolve-pr-context.outputs.source_head_sha }}")
    assert collect_step["env"]["SOURCE_RUN_CONCLUSION"] == (
        "${{ needs.resolve-pr-context.outputs.source_run_conclusion }}"
    )
    assert collect_step["env"]["SOURCE_RUN_ATTEMPT"] == ("${{ needs.resolve-pr-context.outputs.source_run_attempt }}")
    assert collect_step["env"]["SOURCE_RUN_EVENT"] == ("${{ needs.resolve-pr-context.outputs.source_run_event }}")
    assert collect_step["env"]["SOURCE_RUN_NUMBER"] == ("${{ needs.resolve-pr-context.outputs.source_run_number }}")
    assert collect_step["env"]["SOURCE_WORKFLOW_ID"] == ("${{ needs.resolve-pr-context.outputs.source_workflow_id }}")
    assert collect_step["env"]["COMMENT_CONTEXT_PATH"] == ("${{ runner.temp }}/comprehensive-pr-comment-context.json")
    collect_script = collect_step["with"]["script"]
    assert "listJobsForWorkflowRun" in collect_script
    assert "filter: 'latest'" in collect_script
    assert "run_id: sourceRunId" in collect_script
    assert "github.rest.actions.getWorkflowRun" in collect_script
    assert collect_script.count("await getFreshRun()") == 2
    assert "run_attempt" in collect_script
    assert "workflow_id" in collect_script
    assert "job.run_id" in collect_script
    assert "job.head_sha" in collect_script
    assert "job.name === 'Test Report'" in collect_script
    assert "reportJobs.length !== 1" in collect_script
    assert "reportJobs[0].run_attempt !== sourceRunAttempt" in collect_script
    assert "TRUSTED_POLICY_PATHS" in collect_script
    assert ".github/workflows/comprehensive-tests.yml" in collect_script
    assert "scripts/ci/comprehensive_report.py" in collect_script
    assert "github.rest.repos.getContent" in collect_script
    assert "ref: sourceHeadSha" in collect_script
    assert "trustedContent.equals(sourceContent)" in collect_script
    assert "jobs.length" in collect_script
    assert "MAX_JOBS" in collect_script
    assert "SOURCE_HEAD_SHA" in collect_script
    assert "SOURCE_RUN_CONCLUSION" in collect_script
    assert "schema_version: 1" in collect_script
    assert "writeFileSync" in collect_script

    action_methods = set(re.findall(r"github\.rest\.actions\.([A-Za-z0-9_]+)", serialized))
    assert action_methods == {
        "getWorkflow",
        "getWorkflowRun",
        "listJobsForWorkflowRun",
        "listWorkflowRuns",
    }

    render_step = next(step for step in steps if step.get("id") == "render-comment")
    assert "steps.collect-jobs.outcome == 'success'" in render_step["if"]
    assert render_step["env"] == {
        "COMMENT_CONTEXT_PATH": "${{ runner.temp }}/comprehensive-pr-comment-context.json",
        "COMMENT_OUTPUT_PATH": "${{ runner.temp }}/comprehensive-pr-comment.md",
        "EXPECTED_REPOSITORY": "${{ github.repository }}",
    }
    assert "scripts/ci/comprehensive_pr_comment.py" in render_step["run"]
    assert '"$COMMENT_CONTEXT_PATH"' in render_step["run"]
    assert '"$COMMENT_OUTPUT_PATH"' in render_step["run"]
    assert '"$EXPECTED_REPOSITORY"' in render_step["run"]
    assert "${{" not in render_step["run"]

    post_step = next(step for step in steps if step.get("name") == "Post or update trusted PR comment")
    assert post_step.get("if") == "always() && !cancelled()"

    environment = post_step.get("env", {})
    assert environment["COLLECT_JOBS_OUTCOME"] == "${{ steps.collect-jobs.outcome }}"
    assert environment["RENDER_COMMENT_OUTCOME"] == "${{ steps.render-comment.outcome }}"
    assert environment["COMMENT_OUTPUT_PATH"] == ("${{ runner.temp }}/comprehensive-pr-comment.md")
    assert environment["WORKFLOW_RUN_URL"] == ("${{ needs.resolve-pr-context.outputs.source_run_url }}")
    assert environment["WORKFLOW_CONCLUSION"] == ("${{ needs.resolve-pr-context.outputs.source_run_conclusion }}")
    assert environment["SOURCE_HEAD_SHA"] == ("${{ needs.resolve-pr-context.outputs.source_head_sha }}")
    assert environment["SOURCE_RUN_ATTEMPT"] == ("${{ needs.resolve-pr-context.outputs.source_run_attempt }}")
    assert environment["SOURCE_RUN_EVENT"] == ("${{ needs.resolve-pr-context.outputs.source_run_event }}")
    assert environment["SOURCE_RUN_ID"] == ("${{ needs.resolve-pr-context.outputs.source_run_id }}")
    assert environment["SOURCE_RUN_NUMBER"] == ("${{ needs.resolve-pr-context.outputs.source_run_number }}")
    assert environment["SOURCE_WORKFLOW_ID"] == ("${{ needs.resolve-pr-context.outputs.source_workflow_id }}")

    script = post_step["with"]["script"]
    assert "RENDER_COMMENT_OUTCOME" in script
    assert "COLLECT_JOBS_OUTCOME" in script
    assert "COMMENT_OUTPUT_PATH" in script
    assert "WORKFLOW_RUN_URL" in script
    assert "WORKFLOW_CONCLUSION" in script
    assert "SOURCE_READY" in script
    assert "sourceReady" in script
    assert "comprehensiveWorkflowUrl" in script
    assert "hasResolvedRunId ? runUrl : comprehensiveWorkflowUrl" in script
    assert "<!-- odyssey:comprehensive-test-report:v1 -->" in script
    assert "github-actions[bot]" in script
    assert "41898282" in script
    assert "15368" in script
    assert "performed_via_github_app" in script
    assert "comment.user.type === 'Bot'" not in script
    assert ".includes(marker)" not in script
    assert ".includes(MARKER)" not in script
    assert ".startsWith(`${MARKER}\\n`)" in script
    assert "readFileSync" in script
    assert "MAX_COMMENT_BYTES" in script
    assert "const MAX_DISCOVERED_COMMENT_BYTES = 65_536;" in script
    assert "MAX_COMMENT_PAGES" in script
    assert "COMMENT_PAGE_SIZE" in script
    assert "Trusted Comprehensive report unavailable" in script
    assert "core.setFailed" in script
    assert "Source run identity no longer matches" in script
    assert "sourceIdentityValid = false" in script
    assert script.count("github.rest.pulls.get") >= 2
    assert script.count("github.rest.actions.getWorkflowRun") >= 2
    assert "postTimePullRequest" in script
    assert "postTimeRun" in script
    assert "mutationTimePullRequest" in script
    assert "mutationTimeRun" in script
    assert "getSourceRunOrder" in script
    assert script.count("await getSourceRunOrder()") >= 3
    assert "SOURCE_RUN_EVENTS.has(candidate.event)" in script
    assert "event: sourceRunEvent" not in script
    assert "newer source run" in script
    assert "head changed before comment mutation" in script
    assert "run changed before comment mutation" in script
    assert script.index("github.rest.issues.listComments") > script.index("postTimePullRequest")
    assert script.index("github.rest.issues.listComments") > script.index("postTimeRun")
    assert script.index("mutationTimePullRequest") > script.index("github.rest.issues.listComments")
    assert script.index("mutationTimeRun") > script.index("github.rest.issues.listComments")
    assert script.index("github.rest.issues.createComment") > script.index("mutationTimeRun")
    assert script.index("github.rest.issues.updateComment") > script.index("mutationTimeRun")
    assert "await github.paginate(\n              github.rest.issues.listComments" not in script
    assert "commentPage <= MAX_COMMENT_PAGES + 1" in script
    assert "seenCommentIds.has(comment.id)" in script
    assert "Comment discovery failed; posted a red fallback." in script
    assert "comment.body?.includes('🧪 Comprehensive Test Results')" in script
    assert "comment.body?.includes('Test Metrics Report')" in script
    assert "hasRetiredMetricsMarker" in script


def test_commenter_rejects_stale_runs_and_verdict_mismatches() -> None:
    job = _load_job()
    concurrency = job.get("concurrency", {})
    assert concurrency.get("group") == (
        "comprehensive-test-pr-comment-pr-${{ needs.resolve-pr-context.outputs.pr_number }}"
    )
    assert concurrency.get("queue") == "max"
    assert concurrency.get("cancel-in-progress") is False
    assert job.get("needs") == "resolve-pr-context"
    assert "needs.resolve-pr-context.outputs.pr_number != ''" in str(job.get("if"))
    assert "needs.resolve-pr-context.outputs.writer_ready == 'true'" in str(job.get("if"))

    post_step = next(step for step in job["steps"] if step.get("name") == "Post or update trusted PR comment")
    script = post_step["with"]["script"]

    assert "github.rest.pulls.get" in script
    assert "SOURCE_HEAD_SHA" in script
    assert "workflow_run.pull_requests[0]" not in script
    assert "pullRequest.head.sha" in script
    assert "Stale workflow run" in script

    assert "WORKFLOW_CONCLUSION" in script
    assert "RENDER_COMMENT_OUTCOME" in script
    assert "COLLECT_JOBS_OUTCOME" in script
    assert "Trusted Comprehensive report unavailable" in script
    assert "try {" in script
    assert "catch" in script


def test_writer_dispatches_commenter_as_a_sibling_bound_to_the_exact_comprehensive_run() -> None:
    workflow = _load_workflow()
    triggers = _on_block(workflow)
    assert "workflow_dispatch" not in triggers
    assert triggers.get("repository_dispatch") == {"types": ["dependabot-comprehensive-test-comment"]}

    resolver_job = _load_resolver_job()
    condition = str(resolver_job["if"])
    assert "github.event_name == 'repository_dispatch'" in condition
    assert "github.event.workflow_run.event == 'pull_request'" in condition
    assert "github.event.workflow_run.event == 'workflow_dispatch'" in condition

    resolve = next(step for step in resolver_job["steps"] if step.get("id") == "resolve-context")
    assert resolve["env"] == {
        "SOURCE_HEAD_SHA": "${{ github.event.client_payload.source_head_sha }}",
        "SOURCE_HEAD_BRANCH": "${{ github.event.client_payload.source_head_branch }}",
    }
    script = resolve["with"]["script"]
    assert "context.eventName === 'repository_dispatch'" in script
    assert "context.payload.action" in script
    assert "dependabot-comprehensive-test-comment" in script
    assert "github.rest.actions.getWorkflow" in script
    assert "expectedWorkflow.id" in script
    assert "run.workflow_id" in script
    assert "run.repository?.full_name" in script
    assert "sourceHeadBranch = run.head_branch" in script
    assert "sourceHeadRepository = run.head_repository.full_name" in script
    assert "sourceWorkflowRunEvents.has(run.event)" in script
    assert "listWorkflowRuns" in script
    assert "workflow_id: 'comprehensive-tests.yml'" in script
    assert "candidateRun.head_sha === sourceHeadSha" in script
    assert "candidateRun.head_branch === sourceHeadBranch" in script
    assert "matches.length < 1" in script
    assert "matches.reduce" in script
    assert "github.rest.actions.getWorkflowRun" in script
    assert "run.status !== 'completed'" in script
    assert "Comprehensive run did not complete" in script
    assert "run.path !== '.github/workflows/comprehensive-tests.yml'" in script
    assert "run.event !== 'workflow_dispatch'" in script
    assert "setSourceFailure" in script
    assert "core.setOutput('source_ready', 'false')" in script
    assert "core.setOutput('source_error', message)" in script
    assert "authorizeWriterForCurrentPullRequest" in script
    assert script.count("github.rest.pulls.get") >= 2
    assert "core.setOutput('writer_ready', 'true')" in script

    assert "github.rest.pulls.list" in script
    assert "state: 'open'" in script
    assert "head: `${sourceHeadOwner}:${sourceHeadBranch}`" in script
    assert "listPullRequestsAssociatedWithCommit" not in script
    assert "pullRequest.state === 'open'" in script
    assert "pullRequest.head.sha === sourceHeadSha" in script
    assert "pullRequest.head.ref === sourceHeadBranch" in script
    assert "pullRequest.head.repo?.full_name === sourceHeadRepository" in script
    assert "candidates.length === 0" in script
    assert "No current open PR remains" in script
    assert "candidates.length > 1" in script
    assert "github.rest.pulls.get" in script
    assert "pullRequest.head.sha !== sourceHeadSha" in script
    assert "pullRequest.head.ref !== sourceHeadBranch" in script
    assert "pullRequest.head.repo?.full_name !== sourceHeadRepository" in script
    assert "context.eventName === 'repository_dispatch'" in script
    assert "pullRequest.user.login !== 'dependabot[bot]'" in script
    assert "pullRequest.head.ref !== sourceHeadBranch" in script
    assert "sourceHeadRepository !== expectedRepository" in script
    assert "core.setFailed" in script
    assert "core.setOutput('pr_number'" in script
    assert "core.setOutput('source_run_id'" in script
    assert "core.setOutput('source_run_url'" in script
    assert "core.setOutput('source_run_conclusion'" in script
    assert "source_run_attempt" in script
    assert "source_workflow_id" in script
    assert "source_head_repository" in script
    assert "source_head_branch" in script
    for output_name in (
        "source_run_attempt",
        "source_run_event",
        "source_run_id",
        "source_run_number",
        "source_workflow_id",
    ):
        assert resolver_job["outputs"][output_name] == (f"${{{{ steps.resolve-context.outputs.{output_name} }}}}")
    assert resolver_job["outputs"]["writer_ready"] == ("${{ steps.resolve-context.outputs.writer_ready }}")

    job = _load_job()
    collect = next(step for step in job["steps"] if step.get("id") == "collect-jobs")
    assert "needs.resolve-pr-context.outputs.source_run_id" in collect["env"]["SOURCE_RUN_ID"]
    assert collect["env"]["SOURCE_HEAD_BRANCH"] == ("${{ needs.resolve-pr-context.outputs.source_head_branch }}")
    assert "run_id: sourceRunId" in collect["with"]["script"]
    assert "run.head_branch === sourceHeadBranch" in collect["with"]["script"]
    assert "run.head_repository?.full_name === sourceHeadRepository" in collect["with"]["script"]

    post = next(step for step in job["steps"] if step.get("name") == "Post or update trusted PR comment")
    assert post["env"]["PR_NUMBER"] == ("${{ needs.resolve-pr-context.outputs.pr_number }}")
    assert post["env"]["WORKFLOW_RUN_URL"] == ("${{ needs.resolve-pr-context.outputs.source_run_url }}")
    assert post["env"]["WORKFLOW_CONCLUSION"] == ("${{ needs.resolve-pr-context.outputs.source_run_conclusion }}")
    assert post["env"]["SOURCE_HEAD_BRANCH"] == ("${{ needs.resolve-pr-context.outputs.source_head_branch }}")
    assert post["env"]["SOURCE_HEAD_REPOSITORY"] == ("${{ needs.resolve-pr-context.outputs.source_head_repository }}")
    assert post["env"]["SOURCE_RUN_ATTEMPT"] == ("${{ needs.resolve-pr-context.outputs.source_run_attempt }}")
    assert post["env"]["SOURCE_RUN_EVENT"] == ("${{ needs.resolve-pr-context.outputs.source_run_event }}")
    assert post["env"]["SOURCE_RUN_ID"] == ("${{ needs.resolve-pr-context.outputs.source_run_id }}")
    assert post["env"]["SOURCE_RUN_NUMBER"] == ("${{ needs.resolve-pr-context.outputs.source_run_number }}")
    assert post["env"]["SOURCE_WORKFLOW_ID"] == ("${{ needs.resolve-pr-context.outputs.source_workflow_id }}")
    assert "pullRequest.head.ref !== sourceHeadBranch" in post["with"]["script"]
    assert "pullRequest.head.repo?.full_name !== sourceHeadRepository" in post["with"]["script"]
    assert "run.head_branch === sourceHeadBranch" in post["with"]["script"]
    assert "run.head_repository?.full_name === sourceHeadRepository" in post["with"]["script"]
    assert post["if"] == "always() && !cancelled()"
    assert "SOURCE_RESOLUTION_ERROR" not in post["env"]
    assert "SOURCE_RESOLUTION_ERROR" not in post["with"]["script"]

    enforcement = next(step for step in job["steps"] if step.get("name") == "Enforce trusted source resolution")
    assert "always()" in enforcement["if"]
    assert "needs.resolve-pr-context.outputs.source_ready != 'true'" in enforcement["if"]
    assert "exit 1" in enforcement["run"]


def test_comment_monitor_budget_exceeds_the_declared_comprehensive_needs_dag() -> None:
    resolver_job = _load_resolver_job()
    resolve = next(step for step in resolver_job["steps"] if step.get("id") == "resolve-context")
    script = resolve["with"]["script"]
    attempt_match = re.search(
        r"let attempt = 0;\s+attempt < (\d+)",
        script,
    )
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
    assert resolver_job["timeout-minutes"] >= poll_minutes + 10


def test_trusted_commenter_checks_out_only_its_immutable_trusted_renderer() -> None:
    job = _load_job()
    checkouts = [step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@")]
    assert len(checkouts) == 1
    assert checkouts[0]["with"] == {
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

    resolver_job = _load_resolver_job()
    resolve = next(step for step in resolver_job["steps"] if step.get("id") == "resolve-context")
    script = resolve["with"]["script"]
    assert "context.ref" in script
    assert "context.payload.repository.default_branch" in script
    assert "must run from the default branch" in script

    uses = [str(step.get("uses", "")) for step in job["steps"]]
    assert all(not action.startswith("./") for action in uses)
    serialized = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pullRequest.head.sha }}" not in serialized
