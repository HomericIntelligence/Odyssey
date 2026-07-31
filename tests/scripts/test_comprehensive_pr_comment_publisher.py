#!/usr/bin/env python3
"""Execute the trusted inline publisher against mocked GitHub APIs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/comprehensive-test-pr-comments.yml"
MARKER = "<!-- odyssey:comprehensive-test-report:v1 -->"
HEAD_SHA = "a" * 40
HEAD_BRANCH = "feature/trusted-comment"
DEPENDABOT_BRANCH = "dependabot/pip/uv-1.2.3"
REPOSITORY = "HomericIntelligence/Odyssey"
REPOSITORY_ID = 12345
RUN_ID = 123456
RUN_ATTEMPT = 1
RUN_NUMBER = 789
WORKFLOW_ID = 209513613
RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"


def _publisher_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["post-pr-comments"]["steps"]
    post = next(step for step in steps if step.get("name") == "Post or update trusted PR comment")
    script = post["with"]["script"]
    injected = script.replace(
        "const fs = require('fs');",
        "const fs = mockFs;",
        1,
    )
    assert injected != script
    return injected


def _collector_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["post-pr-comments"]["steps"]
    collect = next(step for step in steps if step.get("id") == "collect-jobs")
    script = collect["with"]["script"]
    injected = script.replace(
        "const fs = require('fs');",
        "const fs = mockFs;",
        1,
    )
    assert injected != script
    return injected


def _resolver_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["resolve-pr-context"]["steps"]
    resolve = next(step for step in steps if step.get("id") == "resolve-context")
    return resolve["with"]["script"]


def _github_scripts() -> list[str]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return [
        step["with"]["script"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/github-script@")
    ]


def _actions_identity() -> dict[str, Any]:
    return {
        "user": {
            "login": "github-actions[bot]",
            "id": 41898282,
            "type": "Bot",
        },
        "performed_via_github_app": {
            "id": 15368,
            "slug": "github-actions",
        },
    }


def _comment(comment_id: int, body: str, *, expected_bot: bool = True) -> dict[str, Any]:
    if expected_bot:
        identity = _actions_identity()
    else:
        identity = {
            "user": {"login": "attacker", "id": 999, "type": "User"},
            "performed_via_github_app": None,
        }
    return {"id": comment_id, "body": body, **identity}


def _run(
    *,
    run_id: int = RUN_ID,
    attempt: int = RUN_ATTEMPT,
    run_number: int = RUN_NUMBER,
    conclusion: str = "success",
    event: str = "pull_request",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": "Comprehensive Tests",
        "run_attempt": attempt,
        "run_number": run_number,
        "workflow_id": WORKFLOW_ID,
        "head_sha": HEAD_SHA,
        "head_branch": HEAD_BRANCH,
        "head_repository": {"full_name": REPOSITORY},
        "event": event,
        "status": "completed",
        "conclusion": conclusion,
        "path": ".github/workflows/comprehensive-tests.yml",
        "repository": {"full_name": REPOSITORY},
        "html_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
    }


def _fixture() -> dict[str, Any]:
    trusted_body = f"{MARKER}\n## Comprehensive Test Results\n\n✅ **PASS — authoritative results agree.**\n"
    return {
        "env": {
            "COLLECT_JOBS_OUTCOME": "success",
            "COMMENT_OUTPUT_PATH": "/runner/temp/comment.md",
            "PR_NUMBER": "42",
            "RENDER_COMMENT_OUTCOME": "success",
            "SOURCE_HEAD_BRANCH": HEAD_BRANCH,
            "SOURCE_HEAD_REPOSITORY": REPOSITORY,
            "SOURCE_HEAD_SHA": HEAD_SHA,
            "SOURCE_READY": "true",
            "SOURCE_RUN_ATTEMPT": str(RUN_ATTEMPT),
            "SOURCE_RUN_EVENT": "pull_request",
            "SOURCE_RUN_ID": str(RUN_ID),
            "SOURCE_RUN_NUMBER": str(RUN_NUMBER),
            "SOURCE_WORKFLOW_ID": str(WORKFLOW_ID),
            "WORKFLOW_CONCLUSION": "success",
            "WORKFLOW_RUN_URL": RUN_URL,
        },
        "pulls": [
            {
                "state": "open",
                "head": {
                    "sha": HEAD_SHA,
                    "ref": HEAD_BRANCH,
                    "repo": {"full_name": REPOSITORY},
                },
            },
            {
                "state": "open",
                "head": {
                    "sha": HEAD_SHA,
                    "ref": HEAD_BRANCH,
                    "repo": {"full_name": REPOSITORY},
                },
            },
            {
                "state": "open",
                "head": {
                    "sha": HEAD_SHA,
                    "ref": HEAD_BRANCH,
                    "repo": {"full_name": REPOSITORY},
                },
            },
        ],
        "runs": [_run(), _run(), _run()],
        "latest_runs": [[_run()], [_run()], [_run()]],
        "comments": [
            _comment(10, f"{MARKER}\nold green"),
            _comment(11, f"{MARKER}\nattacker marker", expected_bot=False),
            _comment(12, "prefix\n## Test Metrics Report\nraw artifact"),
        ],
        "trusted_body": trusted_body,
        "trusted_file_available": True,
    }


def _collector_fixture() -> dict[str, Any]:
    trusted_policy = {
        ".github/workflows/comprehensive-tests.yml": "trusted workflow\n",
        "scripts/ci/comprehensive_report.py": "trusted report\n",
    }
    jobs = [
        {
            "id": 101,
            "run_id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "head_sha": HEAD_SHA,
            "name": "Mojo Syntax Validation",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 102,
            "run_id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "head_sha": HEAD_SHA,
            "name": "Test Report",
            "status": "completed",
            "conclusion": "success",
        },
    ]
    return {
        "env": {
            "COMMENT_CONTEXT_PATH": "/runner/temp/context.json",
            "RUNNER_TEMP": "/runner/temp",
            "SOURCE_HEAD_BRANCH": HEAD_BRANCH,
            "SOURCE_HEAD_REPOSITORY": REPOSITORY,
            "SOURCE_HEAD_SHA": HEAD_SHA,
            "SOURCE_RUN_ATTEMPT": str(RUN_ATTEMPT),
            "SOURCE_RUN_CONCLUSION": "success",
            "SOURCE_RUN_EVENT": "pull_request",
            "SOURCE_RUN_ID": str(RUN_ID),
            "SOURCE_RUN_NUMBER": str(RUN_NUMBER),
            "SOURCE_WORKFLOW_ID": str(WORKFLOW_ID),
        },
        "trusted_policy": trusted_policy,
        "source_policy": deepcopy(trusted_policy),
        "run": _run(),
        "jobs": jobs,
    }


def _resolver_run(
    *,
    run_id: int = RUN_ID,
    attempt: int = RUN_ATTEMPT,
    run_number: int = RUN_NUMBER,
    event: str = "workflow_dispatch",
) -> dict[str, Any]:
    return {
        **_run(
            run_id=run_id,
            attempt=attempt,
            run_number=run_number,
            event=event,
        ),
        "name": "Comprehensive Tests",
        "head_branch": DEPENDABOT_BRANCH,
        "repository": {
            "full_name": REPOSITORY,
            "id": REPOSITORY_ID,
        },
    }


def _resolver_pull(
    *,
    sha: str = HEAD_SHA,
    branch: str = DEPENDABOT_BRANCH,
) -> dict[str, Any]:
    return {
        "number": 42,
        "state": "open",
        "user": {"login": "dependabot[bot]"},
        "head": {
            "sha": sha,
            "ref": branch,
            "repo": {"full_name": REPOSITORY},
        },
    }


def _resolver_fixture() -> dict[str, Any]:
    pull_request = _resolver_pull()
    return {
        "context": {
            "eventName": "repository_dispatch",
            "ref": "refs/heads/main",
            "repo": {
                "owner": "HomericIntelligence",
                "repo": "Odyssey",
            },
            "payload": {
                "action": "dependabot-comprehensive-test-comment",
                "repository": {
                    "id": REPOSITORY_ID,
                    "default_branch": "main",
                },
                "workflow_run": None,
            },
        },
        "env": {
            "SOURCE_HEAD_BRANCH": DEPENDABOT_BRANCH,
            "SOURCE_HEAD_SHA": HEAD_SHA,
        },
        "workflow": {
            "id": WORKFLOW_ID,
            "name": "Comprehensive Tests",
            "path": ".github/workflows/comprehensive-tests.yml",
        },
        "associated": [pull_request],
        "pulls": [deepcopy(pull_request), deepcopy(pull_request)],
        "runs": [_resolver_run()],
        "fresh_runs": [],
    }


def _workflow_run_resolver_fixture(event: str) -> dict[str, Any]:
    fixture = _resolver_fixture()
    source_run = _resolver_run(event=event)
    fixture["context"] = {
        "eventName": "workflow_run",
        "ref": "refs/heads/main",
        "repo": {
            "owner": "HomericIntelligence",
            "repo": "Odyssey",
        },
        "payload": {
            "action": "completed",
            "repository": {
                "id": REPOSITORY_ID,
                "default_branch": "main",
            },
            "workflow_run": source_run,
        },
    }
    fixture["env"] = {
        "SOURCE_HEAD_BRANCH": "",
        "SOURCE_HEAD_SHA": "",
    }
    return fixture


def _execute(fixture: dict[str, Any]) -> dict[str, Any]:
    if shutil.which("node") is None:
        pytest.skip("Node.js is unavailable")

    preamble = """
const fixture = FIXTURE;
Object.assign(process.env, fixture.env);
const state = { created: [], updated: [], failed: [], info: [] };
const pullResponses = [...fixture.pulls];
const runResponses = [...fixture.runs];
const latestRunResponses = [...fixture.latest_runs];
const mockFs = {
  lstatSync: () => {
    if (!fixture.trusted_file_available) {
      throw new Error('missing');
    }
    return {
      isFile: () => true,
      isSymbolicLink: () => false,
      size: Buffer.byteLength(fixture.trusted_body, 'utf8'),
    };
  },
  readFileSync: () => fixture.trusted_body,
};
const context = { repo: { owner: 'HomericIntelligence', repo: 'Odyssey' } };
const core = {
  info: message => state.info.push(message),
  setFailed: message => state.failed.push(message),
};
const listComments = async () => ({ data: fixture.comments });
const listWorkflowRuns = async () => ({ data: { workflow_runs: [] } });
const github = {
  paginate: async method => {
    if (method === listComments) return fixture.comments;
    if (method === listWorkflowRuns) {
      if (latestRunResponses.length === 0) {
        throw new Error('no latest-run response');
      }
      return latestRunResponses.shift();
    }
    throw new Error('unexpected pagination method');
  },
  rest: {
    pulls: {
      get: async () => {
        if (pullResponses.length === 0) throw new Error('no pull response');
        return { data: pullResponses.shift() };
      },
    },
    actions: {
      listWorkflowRuns,
      getWorkflowRun: async () => {
        if (runResponses.length === 0) throw new Error('no run response');
        const response = runResponses.shift();
        if (response === null) throw new Error('run unavailable');
        return { data: response };
      },
    },
    issues: {
      listComments,
      createComment: async args => state.created.push(args),
      updateComment: async args => state.updated.push(args),
    },
  },
};
"""
    harness = (
        preamble.replace("FIXTURE", json.dumps(fixture))
        + "\n(async () => {\n"
        + _publisher_script()
        + "\n})().then(() => console.log(JSON.stringify(state))).catch(error => {"
        + " console.error(error); process.exit(1); });\n"
    )
    completed = subprocess.run(
        ["node", "-e", harness],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _execute_collector(fixture: dict[str, Any]) -> dict[str, Any]:
    if shutil.which("node") is None:
        pytest.skip("Node.js is unavailable")

    preamble = """
const fixture = FIXTURE;
Object.assign(process.env, fixture.env);
const state = { failed: [], info: [], writes: [] };
const mockFs = {
  lstatSync: filePath => {
    const value = fixture.trusted_policy[filePath];
    if (value === undefined) throw new Error('missing trusted policy');
    return {
      isFile: () => true,
      isSymbolicLink: () => false,
      size: Buffer.byteLength(value, 'utf8'),
    };
  },
  readFileSync: filePath => Buffer.from(fixture.trusted_policy[filePath], 'utf8'),
  writeFileSync: (filePath, content, options) => {
    state.writes.push({ filePath, content, options });
  },
};
const context = { repo: { owner: 'HomericIntelligence', repo: 'Odyssey' } };
const core = {
  info: message => state.info.push(message),
  setFailed: message => state.failed.push(message),
};
const github = {
  rest: {
    repos: {
      getContent: async args => {
        const value = fixture.source_policy[args.path];
        if (value === undefined) throw new Error('missing source policy');
        return {
          data: {
            type: 'file',
            path: args.path,
            encoding: 'base64',
            size: Buffer.byteLength(value, 'utf8'),
            content: Buffer.from(value, 'utf8').toString('base64'),
          },
        };
      },
    },
    actions: {
      getWorkflowRun: async () => ({ data: fixture.run }),
      listJobsForWorkflowRun: async () => ({
        data: { total_count: fixture.jobs.length, jobs: fixture.jobs },
      }),
    },
  },
};
"""
    harness = (
        preamble.replace("FIXTURE", json.dumps(fixture))
        + "\n(async () => {\n"
        + _collector_script()
        + "\n})().then(() => console.log(JSON.stringify(state))).catch(error => {"
        + " console.error(error); process.exit(1); });\n"
    )
    completed = subprocess.run(
        ["node", "-e", harness],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _execute_resolver(fixture: dict[str, Any]) -> dict[str, Any]:
    if shutil.which("node") is None:
        pytest.skip("Node.js is unavailable")

    preamble = """
const fixture = FIXTURE;
Object.assign(process.env, fixture.env);
const state = { failed: [], info: [], outputs: {} };
const pullResponses = [...fixture.pulls];
const freshRunResponses = [...fixture.fresh_runs];
const listAssociated = async () => ({ data: fixture.associated });
const listRuns = async () => ({ data: { workflow_runs: fixture.runs } });
const context = fixture.context;
const core = {
  info: message => state.info.push(message),
  setFailed: message => state.failed.push(message),
  setOutput: (name, value) => { state.outputs[name] = String(value); },
};
const github = {
  paginate: async method => {
    if (method === listAssociated) return fixture.associated;
    if (method === listRuns) return fixture.runs;
    throw new Error('unexpected pagination method');
  },
  rest: {
    actions: {
      getWorkflow: async () => ({ data: fixture.workflow }),
      listWorkflowRuns: listRuns,
      getWorkflowRun: async () => {
        if (freshRunResponses.length === 0) {
          throw new Error('no fresh run response');
        }
        return { data: freshRunResponses.shift() };
      },
    },
    repos: {
      listPullRequestsAssociatedWithCommit: listAssociated,
    },
    pulls: {
      get: async () => {
        if (pullResponses.length === 0) throw new Error('no pull response');
        return { data: pullResponses.shift() };
      },
    },
  },
};
"""
    harness = (
        preamble.replace("FIXTURE", json.dumps(fixture)).replace(
            "REPOSITORY_ID",
            str(REPOSITORY_ID),
        )
        + "\n(async () => {\n"
        + _resolver_script()
        + "\n})().then(() => console.log(JSON.stringify(state))).catch(error => {"
        + " console.error(error); process.exit(1); });\n"
    )
    completed = subprocess.run(
        ["node", "-e", harness],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_every_inline_github_script_compiles_as_an_async_function() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is unavailable")

    scripts = _github_scripts()
    assert len(scripts) == 3
    compiler = (
        "const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;"
        "for (const source of JSON.parse(process.argv[1])) new AsyncFunction(source);"
    )
    subprocess.run(
        ["node", "-e", compiler, json.dumps(scripts)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_resolver_authorizes_current_dependabot_pr_for_writer() -> None:
    state = _execute_resolver(_resolver_fixture())

    assert state["failed"] == []
    assert state["outputs"]["pr_number"] == "42"
    assert state["outputs"]["source_head_branch"] == DEPENDABOT_BRANCH
    assert state["outputs"]["source_head_repository"] == REPOSITORY
    assert state["outputs"]["source_head_sha"] == HEAD_SHA
    assert state["outputs"]["source_ready"] == "true"
    assert state["outputs"]["source_run_attempt"] == str(RUN_ATTEMPT)
    assert state["outputs"]["source_run_event"] == "workflow_dispatch"
    assert state["outputs"]["source_run_id"] == str(RUN_ID)
    assert state["outputs"]["source_run_number"] == str(RUN_NUMBER)
    assert state["outputs"]["source_workflow_id"] == str(WORKFLOW_ID)
    assert state["outputs"]["writer_ready"] == "true"


@pytest.mark.parametrize("event", ["pull_request", "workflow_dispatch"])
def test_resolver_authorizes_each_supported_workflow_run_event(
    event: str,
) -> None:
    state = _execute_resolver(_workflow_run_resolver_fixture(event))

    assert state["failed"] == []
    assert state["outputs"]["pr_number"] == "42"
    assert state["outputs"]["source_head_branch"] == DEPENDABOT_BRANCH
    assert state["outputs"]["source_head_repository"] == REPOSITORY
    assert state["outputs"]["source_head_sha"] == HEAD_SHA
    assert state["outputs"]["source_ready"] == "true"
    assert state["outputs"]["source_run_event"] == event
    assert state["outputs"]["source_run_id"] == str(RUN_ID)
    assert state["outputs"]["writer_ready"] == "true"


def test_resolver_selects_newest_same_head_dispatch_run() -> None:
    fixture = _resolver_fixture()
    fixture["runs"] = [
        _resolver_run(),
        _resolver_run(
            run_id=RUN_ID + 1,
            run_number=RUN_NUMBER + 1,
        ),
    ]

    state = _execute_resolver(fixture)

    assert state["failed"] == []
    assert state["outputs"]["source_ready"] == "true"
    assert state["outputs"]["source_run_id"] == str(RUN_ID + 1)
    assert state["outputs"]["source_run_number"] == str(RUN_NUMBER + 1)
    assert state["outputs"]["writer_ready"] == "true"


def test_resolver_does_not_enqueue_stale_dependabot_writer() -> None:
    fixture = _resolver_fixture()
    fixture["pulls"][1] = _resolver_pull(sha="b" * 40)

    state = _execute_resolver(fixture)

    assert state["failed"] == []
    assert state["outputs"]["pr_number"] == "42"
    assert state["outputs"].get("source_ready") != "true"
    assert state["outputs"].get("writer_ready") != "true"
    assert any("changed before writer scheduling" in item for item in state["info"])


def test_resolver_schedules_fail_closed_writer_for_current_source_failure() -> None:
    fixture = _resolver_fixture()
    fixture["runs"] = []

    state = _execute_resolver(fixture)

    assert state["failed"] == []
    assert state["outputs"]["source_ready"] == "false"
    assert state["outputs"]["writer_ready"] == "true"


def test_collector_authenticates_policy_and_writes_bounded_context() -> None:
    state = _execute_collector(_collector_fixture())

    assert state["failed"] == []
    assert len(state["writes"]) == 1
    payload = json.loads(state["writes"][0]["content"])
    assert payload["run"]["id"] == RUN_ID
    assert [job["name"] for job in payload["jobs"]] == [
        "Mojo Syntax Validation",
        "Test Report",
    ]


@pytest.mark.parametrize(
    "policy_path",
    [
        ".github/workflows/comprehensive-tests.yml",
        "scripts/ci/comprehensive_report.py",
    ],
)
def test_collector_rejects_source_policy_drift(policy_path: str) -> None:
    fixture = _collector_fixture()
    fixture["source_policy"][policy_path] = "attacker policy\n"

    state = _execute_collector(fixture)

    assert state["writes"] == []
    assert state["failed"] == ["Source Comprehensive policy differs from trusted main."]


def test_collector_requires_current_attempt_test_report() -> None:
    fixture = _collector_fixture()
    fixture["env"]["SOURCE_RUN_ATTEMPT"] = "2"
    fixture["run"]["run_attempt"] = 2
    fixture["jobs"][0]["run_attempt"] = 1
    fixture["jobs"][1]["run_attempt"] = 1

    state = _execute_collector(fixture)

    assert state["writes"] == []
    assert state["failed"] == ["The authoritative Test Report is not from the current run attempt."]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", "workflow_dispatch"),
        ("run_number", RUN_NUMBER + 1),
    ],
)
def test_collector_rejects_run_order_identity_drift(
    field: str,
    value: object,
) -> None:
    fixture = _collector_fixture()
    fixture["run"][field] = value

    state = _execute_collector(fixture)

    assert state["writes"] == []
    assert state["failed"] == ["Source run changed before job collection."]


def test_valid_body_updates_only_owned_comments_and_retires_metrics() -> None:
    state = _execute(_fixture())

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert state["created"] == []
    assert state["failed"] == []
    assert updated[10].startswith(f"{MARKER}\n")
    assert "PASS" in updated[10]
    assert 11 not in updated
    assert "diagnostic only" in updated[12]
    assert RUN_URL in updated[12]


def test_missing_renderer_output_replaces_stale_green_with_red_fallback() -> None:
    fixture = _fixture()
    fixture["trusted_file_available"] = False

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert "FAIL" in updated[10]
    assert "PASS" not in updated[10]
    assert RUN_URL in updated[10]
    assert state["failed"] == ["Trusted comment rendering failed; posted a red fallback."]


@pytest.mark.parametrize(
    "runs",
    [
        [
            _run(conclusion="failure"),
            _run(conclusion="failure"),
        ],
        [
            _run(),
            _run(conclusion="failure"),
            _run(conclusion="failure"),
        ],
        [None, _run()],
    ],
)
def test_source_run_drift_or_unavailability_posts_red(
    runs: list[dict[str, Any] | None],
) -> None:
    fixture = _fixture()
    fixture["runs"] = runs

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert "FAIL" in updated[10]
    assert "PASS" not in updated[10]
    assert RUN_URL in updated[10]
    assert state["failed"]


def test_post_time_head_change_never_mutates_comments() -> None:
    fixture = _fixture()
    fixture["pulls"][1] = {
        "state": "open",
        "head": {"sha": "b" * 40},
    }

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("head changed before comment mutation" in item for item in state["info"])


@pytest.mark.parametrize(
    "head",
    [
        {
            "sha": HEAD_SHA,
            "ref": "feature/same-commit-other-branch",
            "repo": {"full_name": REPOSITORY},
        },
        {
            "sha": HEAD_SHA,
            "ref": HEAD_BRANCH,
            "repo": {"full_name": "attacker/Odyssey"},
        },
    ],
)
def test_post_time_source_tuple_change_never_mutates_comments(
    head: dict[str, Any],
) -> None:
    fixture = _fixture()
    fixture["pulls"][1] = {"state": "open", "head": head}

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("head changed before comment mutation" in item for item in state["info"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_branch", "feature/same-commit-other-branch"),
        ("head_repository", {"full_name": "attacker/Odyssey"}),
    ],
)
def test_source_run_branch_or_repository_drift_posts_red(
    field: str,
    value: object,
) -> None:
    fixture = _fixture()
    fixture["runs"][0][field] = value

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert "FAIL" in updated[10]
    assert "PASS" not in updated[10]
    assert state["failed"]


def test_obsolete_run_attempt_never_overwrites_newer_same_head_comment() -> None:
    fixture = _fixture()
    newer = _run(attempt=RUN_ATTEMPT + 1)
    fixture["latest_runs"] = [[_run()], [newer]]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run" in item for item in state["info"])


@pytest.mark.parametrize(
    "newer",
    [
        _run(attempt=RUN_ATTEMPT + 1),
        _run(run_id=RUN_ID + 1, run_number=RUN_NUMBER + 1),
    ],
    ids=["newer-attempt", "newer-distinct-run"],
)
def test_initially_obsolete_source_run_never_mutates_comment(
    newer: dict[str, Any],
) -> None:
    fixture = _fixture()
    fixture["latest_runs"] = [[newer]]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run" in item for item in state["info"])


def test_newer_attempt_observed_after_order_check_never_mutates_comment() -> None:
    fixture = _fixture()
    fixture["runs"] = [_run(attempt=RUN_ATTEMPT + 1)]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run attempt" in item for item in state["info"])


def test_newer_attempt_observed_after_final_order_check_never_mutates_comment() -> None:
    fixture = _fixture()
    fixture["runs"] = [_run(), _run(attempt=RUN_ATTEMPT + 1)]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run attempt" in item for item in state["info"])


def test_obsolete_distinct_run_never_overwrites_newer_same_head_comment() -> None:
    fixture = _fixture()
    newer = _run(run_id=RUN_ID + 1, run_number=RUN_NUMBER + 1)
    fixture["latest_runs"] = [[_run()], [newer]]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run" in item for item in state["info"])


def test_newer_run_appearing_during_comment_discovery_never_gets_overwritten() -> None:
    fixture = _fixture()
    newer = _run(run_id=RUN_ID + 1, run_number=RUN_NUMBER + 1)
    fixture["latest_runs"] = [[_run()], [_run()], [_run(), newer]]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run" in item for item in state["info"])


@pytest.mark.parametrize(
    ("source_event", "newer_event"),
    [
        ("pull_request", "workflow_dispatch"),
        ("workflow_dispatch", "pull_request"),
    ],
)
def test_cross_event_newer_run_never_gets_overwritten(
    source_event: str,
    newer_event: str,
) -> None:
    fixture = _fixture()
    source = _run(event=source_event)
    newer = _run(
        run_id=RUN_ID + 1,
        run_number=RUN_NUMBER + 1,
        event=newer_event,
    )
    fixture["env"]["SOURCE_RUN_EVENT"] = source_event
    fixture["runs"] = [source, source]
    fixture["latest_runs"] = [[source], [source, newer]]

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert any("newer source run" in item for item in state["info"])


@pytest.mark.parametrize(
    "latest_runs",
    [
        [],
        [[]],
        [[_run(run_id=0)]],
        [[_run(run_number=0)]],
        [[_run(attempt=0)]],
        [[_run(), _run()]],
    ],
    ids=[
        "api-failure",
        "empty",
        "invalid-id",
        "invalid-run-number",
        "invalid-attempt",
        "duplicate",
    ],
)
def test_unknown_initial_source_run_order_fails_without_mutating_comment(
    latest_runs: list[list[dict[str, Any]]],
) -> None:
    fixture = _fixture()
    fixture["latest_runs"] = latest_runs

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert state["failed"] == ["Unable to prove the source run is the newest exact PR-head run."]


@pytest.mark.parametrize(
    "latest_runs",
    [
        [[_run()]],
        [[_run()], []],
        [[_run()], [_run(run_number=0)]],
        [[_run()], [_run(), _run()]],
    ],
    ids=["api-failure", "empty", "malformed", "duplicate"],
)
def test_unknown_final_source_run_order_fails_without_mutating_comment(
    latest_runs: list[list[dict[str, Any]]],
) -> None:
    fixture = _fixture()
    fixture["latest_runs"] = latest_runs

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert state["failed"] == ["Unable to prove source-run ordering before comment mutation."]


@pytest.mark.parametrize(
    "latest_runs",
    [
        [[_run()], [_run()]],
        [[_run()], [_run()], []],
        [[_run()], [_run()], [_run(run_number=0)]],
        [[_run()], [_run()], [_run(), _run()]],
    ],
    ids=["api-failure", "empty", "malformed", "duplicate"],
)
def test_unknown_mutation_time_source_order_fails_without_mutating_comment(
    latest_runs: list[list[dict[str, Any]]],
) -> None:
    fixture = _fixture()
    fixture["latest_runs"] = latest_runs

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert state["failed"] == ["Unable to prove source-run ordering after comment discovery."]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SOURCE_RUN_NUMBER", ""),
        ("SOURCE_RUN_NUMBER", "0"),
        ("SOURCE_RUN_NUMBER", "not-a-number"),
        ("SOURCE_RUN_EVENT", "push"),
    ],
)
def test_invalid_ready_source_ordering_context_never_mutates_comment(
    field: str,
    value: str,
) -> None:
    fixture = _fixture()
    fixture["env"][field] = value

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert state["failed"] == ["Resolved source-run ordering context is invalid."]


def test_overlapping_legacy_report_and_metrics_comment_keeps_report() -> None:
    fixture = _fixture()
    fixture["comments"] = [
        _comment(
            25,
            "# 🧪 Comprehensive Test Results\nPASS\n## Test Metrics Report\nraw",
        ),
    ]

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert set(updated) == {25}
    assert updated[25].startswith(f"{MARKER}\n")
    assert "PASS" in updated[25]
    assert "diagnostic only and are no longer" not in updated[25]


def test_collection_failure_replaces_stale_green_with_red_fallback() -> None:
    fixture = _fixture()
    fixture["env"]["COLLECT_JOBS_OUTCOME"] = "failure"
    fixture["env"]["RENDER_COMMENT_OUTCOME"] = "skipped"
    fixture["trusted_file_available"] = False

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert "FAIL" in updated[10]
    assert "PASS" not in updated[10]
    assert state["failed"]


def test_wrong_identity_marker_is_ignored_and_safe_comment_is_created() -> None:
    fixture = _fixture()
    fixture["comments"] = [
        _comment(20, f"{MARKER}\nattacker green", expected_bot=False),
    ]

    state = _execute(fixture)

    assert state["updated"] == []
    assert len(state["created"]) == 1
    assert state["created"][0]["body"].startswith(f"{MARKER}\n")
    assert "PASS" in state["created"][0]["body"]


def test_all_owned_legacy_duplicates_are_neutralized() -> None:
    fixture = _fixture()
    fixture["trusted_file_available"] = False
    fixture["comments"] = [
        _comment(30, "attacker prefix\n# 🧪 Comprehensive Test Results\nPASS"),
        _comment(31, "other prefix\n# 🧪 Comprehensive Test Results\nPASS"),
        _comment(32, "attacker prefix\n## Test Metrics Report\n[bad](javascript:x)"),
    ]

    state = _execute(fixture)

    updated = {item["comment_id"]: item["body"] for item in state["updated"]}
    assert set(updated) == {30, 31, 32}
    assert "FAIL" in updated[30]
    assert "FAIL" in updated[31]
    assert "javascript:" not in updated[32]


def test_unresolved_source_without_ordering_identity_never_mutates_comment() -> None:
    fixture = _fixture()
    fixture["env"].update(
        {
            "COLLECT_JOBS_OUTCOME": "skipped",
            "RENDER_COMMENT_OUTCOME": "skipped",
            "SOURCE_READY": "false",
            "SOURCE_RUN_ATTEMPT": "",
            "SOURCE_RUN_ID": "",
            "SOURCE_WORKFLOW_ID": "",
            "WORKFLOW_CONCLUSION": "unknown",
            "WORKFLOW_RUN_URL": (f"https://github.com/{REPOSITORY}/actions/workflows/comprehensive-tests.yml"),
        }
    )
    fixture["runs"] = []

    state = _execute(fixture)

    assert state["created"] == []
    assert state["updated"] == []
    assert state["failed"] == ["No authoritative source-run ordering identity is available."]
