#!/usr/bin/env python3
"""Validate API-derived Comprehensive results and render a safe PR comment.

The write-scoped ``workflow_run`` consumer must not trust artifacts produced by
pull-request code.  This module accepts only a small, closed JSON document built
from GitHub's API by the trusted default-branch workflow.  It validates that
document before rendering bounded Markdown and neutralizes all untrusted job
names.

Python is used because this is JSON-heavy CI automation, as permitted by
ADR-001 (``docs/adr/ADR-001-language-selection-tooling.md``). The
implementation is standard-library-only so the trusted consumer does not
install or execute pull-request dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata


SCHEMA_VERSION = 1
COMMENT_MARKER = "<!-- odyssey:comprehensive-test-report:v1 -->"
MAX_INPUT_BYTES = 262_144
MAX_COMMENT_BYTES = 60_000
MAX_JOBS = 128
MAX_JOB_NAME_BYTES = 256
MAX_REPOSITORY_BYTES = 200
MAX_URL_BYTES = 512

_TOP_LEVEL_KEYS = {"schema_version", "repository", "run", "jobs"}
_RUN_KEYS = {"id", "status", "conclusion", "head_sha", "url"}
_JOB_KEYS = {"id", "name", "status", "conclusion", "url"}
_COMPLETED_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "neutral",
    "skipped",
    "stale",
    "startup_failure",
    "success",
    "timed_out",
}
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")

# Table and inline-Markdown metacharacters are encoded.  Mentions use a
# full-width at sign so GitHub cannot notify users or teams.
_MARKDOWN_TRANSLATION = str.maketrans(
    {
        "\\": "&#92;",
        "`": "&#96;",
        "*": "&#42;",
        "_": "&#95;",
        "{": "&#123;",
        "}": "&#125;",
        "[": "&#91;",
        "]": "&#93;",
        "<": "&lt;",
        ">": "&gt;",
        "(": "&#40;",
        ")": "&#41;",
        "#": "&#35;",
        "+": "&#43;",
        "-": "&#45;",
        ".": "&#46;",
        "!": "&#33;",
        "|": "&#124;",
        "&": "&amp;",
        "@": "＠",
    }
)


class ValidationError(ValueError):
    """Raised when API-derived comment context violates the closed contract."""


def _fail(message: str) -> ValidationError:
    """Create a validation error containing only trusted static prose."""
    return ValidationError(message)


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise _fail(f"{label} keys must be strings")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise _fail(f"{label} fields do not match schema version {SCHEMA_VERSION}")


def _require_positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{label} must be an integer")
    if value <= 0 or value > 9_007_199_254_740_991:
        raise _fail(f"{label} is outside the supported range")
    return value


def _require_string(
    value: object,
    label: str,
    *,
    maximum_bytes: int,
    allow_multiline: bool = False,
) -> str:
    if not isinstance(value, str):
        raise _fail(f"{label} must be a string")
    size = len(value.encode("utf-8"))
    if size == 0 or size > maximum_bytes:
        raise _fail(f"{label} has an invalid size")
    if "\x00" in value:
        raise _fail(f"{label} contains a forbidden control character")
    if not allow_multiline and any(character in value for character in "\r\n"):
        raise _fail(f"{label} must be one line")
    return value


def _require_conclusion(value: object, label: str) -> str:
    conclusion = _require_string(value, label, maximum_bytes=32)
    if conclusion not in _COMPLETED_CONCLUSIONS:
        raise _fail(f"{label} is not a completed GitHub conclusion")
    return conclusion


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _fail("JSON contains a duplicate object field")
        result[key] = value
    return result


def load_context(path: Path) -> dict[str, Any]:
    """Load one bounded, duplicate-key-free JSON object from a regular file."""
    try:
        stat = path.lstat()
    except OSError as error:
        raise _fail("comment context is unavailable") from error
    if path.is_symlink() or not path.is_file():
        raise _fail("comment context must be a regular file")
    if stat.st_size <= 0 or stat.st_size > MAX_INPUT_BYTES:
        raise _fail("comment context has an invalid size")

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail("comment context is not valid UTF-8 JSON") from error
    return _require_mapping(value, "comment context")


def _validated_repository(value: object, expected_repository: str) -> str:
    expected = _require_string(
        expected_repository,
        "expected repository",
        maximum_bytes=MAX_REPOSITORY_BYTES,
    )
    if _REPOSITORY_RE.fullmatch(expected) is None:
        raise _fail("expected repository has an invalid form")

    repository = _require_string(
        value,
        "repository",
        maximum_bytes=MAX_REPOSITORY_BYTES,
    )
    if repository != expected:
        raise _fail("repository does not match the trusted workflow repository")
    return repository


def _validated_run(
    value: object,
    repository: str,
) -> tuple[int, str, str]:
    run = _require_mapping(value, "run")
    _require_exact_keys(run, _RUN_KEYS, "run")

    run_id = _require_positive_integer(run["id"], "run.id")
    status = _require_string(run["status"], "run.status", maximum_bytes=32)
    if status != "completed":
        raise _fail("run.status must be completed")
    conclusion = _require_conclusion(run["conclusion"], "run.conclusion")
    head_sha = _require_string(run["head_sha"], "run.head_sha", maximum_bytes=40)
    if _SHA_RE.fullmatch(head_sha) is None:
        raise _fail("run.head_sha must be a lowercase full commit SHA")

    run_url = _require_string(
        run["url"],
        "run.url",
        maximum_bytes=MAX_URL_BYTES,
    )
    expected_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    if run_url != expected_url:
        raise _fail("run.url does not match the trusted run identity")
    return run_id, conclusion, run_url


def _validated_jobs(
    value: object,
    *,
    repository: str,
    run_id: int,
    workflow_conclusion: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise _fail("jobs must be an array")
    if not value or len(value) > MAX_JOBS:
        raise _fail("jobs has an invalid item count")

    validated: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    report_conclusions: list[str] = []
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"

    for index, raw_job in enumerate(value):
        job = _require_mapping(raw_job, f"jobs[{index}]")
        _require_exact_keys(job, _JOB_KEYS, f"jobs[{index}]")

        job_id = _require_positive_integer(job["id"], f"jobs[{index}].id")
        if job_id in seen_ids:
            raise _fail("jobs contains a duplicate job id")
        seen_ids.add(job_id)

        name = _require_string(
            job["name"],
            f"jobs[{index}].name",
            maximum_bytes=MAX_JOB_NAME_BYTES,
            allow_multiline=True,
        )
        if name == "Test Report" and name in seen_names:
            raise _fail("jobs must contain exactly one Test Report")
        if name in seen_names:
            raise _fail("jobs contains a duplicate job name")
        seen_names.add(name)

        status = _require_string(
            job["status"],
            f"jobs[{index}].status",
            maximum_bytes=32,
        )
        if status != "completed":
            raise _fail("every source job must be completed")
        conclusion = _require_conclusion(
            job["conclusion"],
            f"jobs[{index}].conclusion",
        )

        url = _require_string(
            job["url"],
            f"jobs[{index}].url",
            maximum_bytes=MAX_URL_BYTES,
        )
        if url != f"{run_url}/job/{job_id}":
            raise _fail("job URL does not match its trusted run and job ids")

        if name == "Test Report":
            report_conclusions.append(conclusion)
        validated.append(
            {
                "id": job_id,
                "name": name,
                "status": status,
                "conclusion": conclusion,
                "url": url,
            }
        )

    if len(report_conclusions) != 1:
        raise _fail("jobs must contain exactly one Test Report")
    if report_conclusions[0] != workflow_conclusion:
        raise _fail("Test Report conclusion contradicts the workflow conclusion")
    return validated


def _safe_markdown_text(value: str) -> str:
    """Collapse layout controls and neutralize Markdown, HTML, and mentions."""
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character for character in value
    )
    collapsed = " ".join(without_controls.split())
    return collapsed.translate(_MARKDOWN_TRANSLATION)


def render_comment(
    context: object,
    *,
    expected_repository: str,
) -> str:
    """Validate one API context and return bounded trusted Markdown."""
    payload = _require_mapping(context, "comment context")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "comment context")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise _fail(f"schema_version must equal {SCHEMA_VERSION}")

    repository = _validated_repository(payload["repository"], expected_repository)
    run_id, workflow_conclusion, run_url = _validated_run(
        payload["run"],
        repository,
    )
    jobs = _validated_jobs(
        payload["jobs"],
        repository=repository,
        run_id=run_id,
        workflow_conclusion=workflow_conclusion,
    )

    if workflow_conclusion == "success":
        verdict = "✅ **PASS — authoritative Test Report and workflow conclusion agree.**"
    else:
        verdict = "❌ **FAIL — authoritative Test Report and workflow conclusion agree.**"

    lines = [
        COMMENT_MARKER,
        "## Comprehensive Test Results",
        "",
        verdict,
        "",
        f"- Workflow conclusion: `{workflow_conclusion}`",
        f"- [Workflow run]({run_url})",
        "",
        "| Job | Status | Conclusion |",
        "| --- | --- | --- |",
    ]
    for job in jobs:
        safe_name = _safe_markdown_text(str(job["name"]))
        lines.append(f"| [{safe_name}]({job['url']}) | `{job['status']}` | `{job['conclusion']}` |")
    lines.extend(
        [
            "",
            "_This comment is rendered by trusted default-branch code from "
            "GitHub API job results. Test artifacts remain diagnostic only._",
            "",
        ]
    )
    body = "\n".join(lines)
    if body.count(COMMENT_MARKER) != 1:
        raise _fail("rendered comment marker is not unique")
    if len(body.encode("utf-8")) > MAX_COMMENT_BYTES:
        raise _fail("rendered comment exceeds the supported size")
    return body


def render_comment_file(
    input_path: Path,
    output_path: Path,
    *,
    expected_repository: str,
) -> None:
    """Validate an input file and atomically create a fresh trusted body file."""
    try:
        output_path.unlink(missing_ok=True)
    except OSError as error:
        raise _fail("stale comment output could not be removed") from error

    context = load_context(input_path)
    body = render_comment(context, expected_repository=expected_repository)
    encoded = body.encode("utf-8")
    if not output_path.parent.is_dir():
        raise _fail("comment output parent is unavailable")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
    except OSError as error:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise _fail("trusted comment output could not be written") from error


def main(argv: list[str] | None = None) -> int:
    """Render a trusted comment from bounded API-derived JSON."""
    parser = argparse.ArgumentParser(
        description="Render a trusted Comprehensive PR comment",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-repository", required=True)
    arguments = parser.parse_args(argv)

    try:
        render_comment_file(
            arguments.input,
            arguments.output,
            expected_repository=arguments.expected_repository,
        )
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
