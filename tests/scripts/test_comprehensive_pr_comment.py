#!/usr/bin/env python3
"""Behavior tests for the trusted Comprehensive PR-comment renderer."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(SCRIPT_DIR))

from comprehensive_pr_comment import (  # noqa: E402
    COMMENT_MARKER,
    MAX_INPUT_BYTES,
    MAX_JOBS,
    ValidationError,
    load_context,
    main,
    render_comment,
    render_comment_file,
)


REPOSITORY = "HomericIntelligence/Odyssey"
RUN_ID = 123456
RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"


def _job(job_id: int, name: str, conclusion: str = "success") -> dict[str, object]:
    return {
        "id": job_id,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "url": f"{RUN_URL}/job/{job_id}",
    }


def _context(conclusion: str = "success") -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "run": {
            "id": RUN_ID,
            "status": "completed",
            "conclusion": conclusion,
            "head_sha": "a" * 40,
            "url": RUN_URL,
        },
        "jobs": [
            _job(101, "Mojo Syntax Validation"),
            _job(102, "Test Report", conclusion),
        ],
    }


def test_complete_api_context_is_rendered_by_trusted_code() -> None:
    body = render_comment(_context(), expected_repository=REPOSITORY)

    assert body.startswith(f"{COMMENT_MARKER}\n")
    assert body.count(COMMENT_MARKER) == 1
    assert "## Comprehensive Test Results" in body
    assert "PASS" in body
    assert "Mojo Syntax Validation" in body
    assert "Test Report" in body
    assert f"[Workflow run]({RUN_URL})" in body
    assert "global test" not in body.lower()
    assert "pass rate" not in body.lower()


def test_failed_authoritative_run_renders_a_red_verdict() -> None:
    payload = _context("failure")
    jobs = payload["jobs"]
    assert isinstance(jobs, list)
    jobs[0] = _job(101, "Mojo Syntax Validation", "failure")

    body = render_comment(payload, expected_repository=REPOSITORY)

    assert "FAIL" in body
    assert "failure" in body
    assert "PASS" not in body


def test_markup_links_comment_markers_and_mentions_are_neutralized() -> None:
    payload = _context()
    jobs = payload["jobs"]
    assert isinstance(jobs, list)
    jobs[0] = _job(
        101,
        "<script>alert(1)</script> @everyone @org/team "
        "[click](javascript:alert(1)) | `code` "
        "<!-- odyssey:comprehensive-test-report:v1 --> \u202eevil",
    )

    body = render_comment(payload, expected_repository=REPOSITORY)

    assert "<script>" not in body
    assert "[click](javascript:" not in body
    assert "@everyone" not in body
    assert "@org/team" not in body
    assert "\u202e" not in body
    assert body.count(COMMENT_MARKER) == 1
    assert "&lt;script&gt;" in body
    assert "＠everyone" in body


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("schema_version",), 1.0),
        (("schema_version",), True),
        (("repository",), "attacker/example"),
        (("run", "status"), "in_progress"),
        (("run", "conclusion"), "unknown"),
        (("run", "head_sha"), "not-a-sha"),
        (("run", "url"), "javascript:alert(1)"),
        (("jobs", 0, "status"), "queued"),
        (("jobs", 0, "conclusion"), "unknown"),
        (("jobs", 0, "url"), "https://evil.example/job/101"),
    ],
)
def test_closed_schema_rejects_invalid_identity_and_enums(
    path: tuple[str | int, ...],
    value: object,
) -> None:
    payload = _context()
    target: object = payload
    for component in path[:-1]:
        assert isinstance(target, (dict, list))
        target = target[component]  # type: ignore[index]
    assert isinstance(target, (dict, list))
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        render_comment(payload, expected_repository=REPOSITORY)


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("top", "unexpected"),
        ("run", "unexpected"),
        ("job", "unexpected"),
    ],
)
def test_closed_schema_rejects_unknown_fields(container: str, field: str) -> None:
    payload = _context()
    if container == "top":
        payload[field] = True
    elif container == "run":
        run = payload["run"]
        assert isinstance(run, dict)
        run[field] = True
    else:
        jobs = payload["jobs"]
        assert isinstance(jobs, list)
        job = jobs[0]
        assert isinstance(job, dict)
        job[field] = True

    with pytest.raises(ValidationError):
        render_comment(payload, expected_repository=REPOSITORY)


def test_requires_exactly_one_test_report_job() -> None:
    missing = _context()
    missing_jobs = missing["jobs"]
    assert isinstance(missing_jobs, list)
    missing_jobs[1] = _job(102, "Not The Report")

    duplicate = _context()
    duplicate_jobs = duplicate["jobs"]
    assert isinstance(duplicate_jobs, list)
    duplicate_jobs.append(_job(103, "Test Report"))

    for payload in (missing, duplicate):
        with pytest.raises(ValidationError, match="Test Report"):
            render_comment(payload, expected_repository=REPOSITORY)


def test_rejects_duplicate_job_ids_and_empty_job_lists() -> None:
    duplicate = _context()
    duplicate_jobs = duplicate["jobs"]
    assert isinstance(duplicate_jobs, list)
    second = duplicate_jobs[1]
    assert isinstance(second, dict)
    second["id"] = 101

    empty = _context()
    empty["jobs"] = []

    for payload in (duplicate, empty):
        with pytest.raises(ValidationError):
            render_comment(payload, expected_repository=REPOSITORY)


@pytest.mark.parametrize(
    ("workflow_conclusion", "report_conclusion"),
    [
        ("success", "failure"),
        ("failure", "success"),
        ("cancelled", "failure"),
    ],
)
def test_test_report_conclusion_must_match_workflow_conclusion(
    workflow_conclusion: str,
    report_conclusion: str,
) -> None:
    payload = _context(workflow_conclusion)
    jobs = payload["jobs"]
    assert isinstance(jobs, list)
    jobs[1] = _job(102, "Test Report", report_conclusion)

    with pytest.raises(ValidationError, match="conclusion"):
        render_comment(payload, expected_repository=REPOSITORY)


def test_job_count_and_string_lengths_are_bounded() -> None:
    too_many = _context()
    too_many["jobs"] = [
        _job(index + 1, "Test Report" if index == 0 else f"Job {index}") for index in range(MAX_JOBS + 1)
    ]

    overlong = _context()
    overlong_jobs = overlong["jobs"]
    assert isinstance(overlong_jobs, list)
    overlong_jobs[0] = _job(101, "x" * 257)

    for payload in (too_many, overlong):
        with pytest.raises(ValidationError):
            render_comment(payload, expected_repository=REPOSITORY)


def test_file_loader_rejects_malformed_empty_and_oversized_json(tmp_path: Path) -> None:
    context_path = tmp_path / "context.json"

    for raw in (b"", b"{", b"[]", b"x" * (MAX_INPUT_BYTES + 1)):
        context_path.write_bytes(raw)
        with pytest.raises(ValidationError):
            load_context(context_path)


def test_file_loader_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    context_path = tmp_path / "context.json"
    context_path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="duplicate"):
        load_context(context_path)


def test_file_loader_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_context()), encoding="utf-8")
    context_path = tmp_path / "context.json"
    try:
        context_path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValidationError, match="regular file"):
        load_context(context_path)


def test_maximum_valid_input_that_expands_past_comment_limit_is_rejected() -> None:
    payload = _context()
    payload["jobs"] = [_job(1, "Test Report")]
    jobs = payload["jobs"]
    assert isinstance(jobs, list)
    jobs.extend(_job(index + 2, f"{index:03d}" + "&" * 253) for index in range(MAX_JOBS - 1))

    with pytest.raises(ValidationError, match="exceeds"):
        render_comment(payload, expected_repository=REPOSITORY)


def test_render_failure_removes_any_stale_output(tmp_path: Path) -> None:
    input_path = tmp_path / "context.json"
    output_path = tmp_path / "comment.md"
    invalid = deepcopy(_context())
    invalid["schema_version"] = 999
    input_path.write_text(json.dumps(invalid), encoding="utf-8")
    output_path.write_text(f"{COMMENT_MARKER}\nPASS\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        render_comment_file(
            input_path,
            output_path,
            expected_repository=REPOSITORY,
        )

    assert not output_path.exists()


def test_cli_failure_leaves_no_postable_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "context.json"
    output_path = tmp_path / "comment.md"
    input_path.write_text("{}", encoding="utf-8")
    output_path.write_text(f"{COMMENT_MARKER}\nPASS\n", encoding="utf-8")

    result = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-repository",
            REPOSITORY,
        ]
    )

    assert result == 1
    assert not output_path.exists()
    assert "ERROR:" in capsys.readouterr().err


def test_render_file_writes_only_a_validated_bounded_comment(tmp_path: Path) -> None:
    input_path = tmp_path / "context.json"
    output_path = tmp_path / "comment.md"
    input_path.write_text(json.dumps(_context()), encoding="utf-8")

    render_comment_file(
        input_path,
        output_path,
        expected_repository=REPOSITORY,
    )

    body = output_path.read_text(encoding="utf-8")
    assert body.startswith(COMMENT_MARKER)
    assert len(body.encode("utf-8")) < 65_536
