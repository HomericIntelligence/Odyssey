#!/usr/bin/env python3
"""Unit tests for fail-closed parsing of pip-audit JSON output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "parse_pip_audit.py"


def _dependency(name: str, version: str, vulns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "version": version, "vulns": vulns}


def _vulnerability(vulnerability_id: str) -> dict[str, Any]:
    return {
        "id": vulnerability_id,
        "fix_versions": ["9.9.9"],
        "aliases": [f"CVE-{vulnerability_id[-4:]}"],
        "description": "A test advisory",
    }


def _run_parser(
    tmp_path: Path,
    payload: object,
    *,
    audit_exit_code: int,
    requirements: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    input_path = tmp_path / "pip-audit.json"
    requirements_path = tmp_path / "requirements.txt"
    report_path = tmp_path / "pip-audit-report.md"
    output_path = tmp_path / "github-output.txt"

    if isinstance(payload, bytes):
        input_path.write_bytes(payload)
    elif isinstance(payload, str):
        input_path.write_text(payload, encoding="utf-8")
    else:
        input_path.write_text(json.dumps(payload), encoding="utf-8")

    if requirements is None:
        requirement_lines = []
        if isinstance(payload, dict) and isinstance(payload.get("dependencies"), list):
            for dependency in payload["dependencies"]:
                if (
                    isinstance(dependency, dict)
                    and isinstance(dependency.get("name"), str)
                    and isinstance(dependency.get("version"), str)
                ):
                    requirement_lines.append(f"{dependency['name']}=={dependency['version']}")
        requirements = "\n".join(requirement_lines or ["safe==1.0"]) + "\n"
    requirements_path.write_text(requirements, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--requirements",
            str(requirements_path),
            "--audit-exit-code",
            str(audit_exit_code),
            "--report",
            str(report_path),
            "--github-output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, report_path, output_path


def test_modern_manifest_counts_vulnerabilities_and_writes_findings(tmp_path: Path) -> None:
    payload = {
        "dependencies": [
            _dependency("safe", "1.0", []),
            _dependency("affected", "2.0", [_vulnerability("PYSEC-1000"), _vulnerability("GHSA-2000")]),
        ],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=1)

    assert result.returncode == 1, result.stderr
    assert output_path.read_text(encoding="utf-8") == (
        "count=2\nverdict=findings\ndiagnostic=2 known vulnerabilities found\n"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "PYSEC-1000" in report
    assert "GHSA-2000" in report
    assert "affected" in report


def test_modern_manifest_without_findings_is_successful(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("safe", "1.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=0)

    assert result.returncode == 0, result.stderr
    assert output_path.read_text(encoding="utf-8") == (
        "count=0\nverdict=success\ndiagnostic=No known vulnerabilities found\n"
    )
    assert "No known vulnerabilities found" in report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"dependencies": [_dependency("safe", "1.0", [])]}, "missing required key"),
        ([], "top level must be an object"),
        ("", "empty"),
        ("not-json", "valid JSON"),
    ],
)
def test_empty_or_malformed_output_fails_closed(
    tmp_path: Path,
    payload: object,
    expected_error: str,
) -> None:
    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=0)

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    output = output_path.read_text(encoding="utf-8")
    assert "count=unknown\n" in output
    assert "verdict=malformed\n" in output
    assert expected_error in output


@pytest.mark.parametrize(
    ("payload", "audit_exit_code", "expected_error", "expected_code", "expected_verdict"),
    [
        (
            {"dependencies": [_dependency("safe", "1.0", [])], "fixes": []},
            1,
            "reported failure without vulnerability findings",
            4,
            "contradictory",
        ),
        (
            {"dependencies": [_dependency("affected", "2.0", [_vulnerability("PYSEC-1000")])], "fixes": []},
            0,
            "reported success despite vulnerability findings",
            4,
            "contradictory",
        ),
        (
            {"dependencies": [_dependency("affected", "2.0", [_vulnerability("PYSEC-1000")])], "fixes": []},
            2,
            "operational failure",
            3,
            "operational",
        ),
    ],
)
def test_exit_code_and_manifest_must_describe_the_same_outcome(
    tmp_path: Path,
    payload: object,
    audit_exit_code: int,
    expected_error: str,
    expected_code: int,
    expected_verdict: str,
) -> None:
    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=audit_exit_code)

    assert result.returncode == expected_code
    assert expected_error in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    output = output_path.read_text(encoding="utf-8")
    assert "count=unknown\n" in output
    assert f"verdict={expected_verdict}\n" in output
    assert "diagnostic=" in output
    assert expected_error in output


def test_skipped_dependency_is_an_incomplete_audit(tmp_path: Path) -> None:
    payload = {
        "dependencies": [{"name": "unknown", "skip_reason": "could not resolve"}],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=0)

    assert result.returncode == 2
    assert "skipped dependency" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    output = output_path.read_text(encoding="utf-8")
    assert "count=unknown\n" in output
    assert "verdict=malformed\n" in output


def test_non_fix_audit_rejects_nonempty_fixes_as_contradictory(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("safe", "1.0", [])],
        "fixes": [{"name": "safe", "version": "2.0"}],
    }

    result, report_path, output_path = _run_parser(tmp_path, payload, audit_exit_code=0)

    assert result.returncode == 4
    assert "non-empty fixes" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    output = output_path.read_text(encoding="utf-8")
    assert "verdict=contradictory\n" in output
    assert "count=unknown\n" in output


def test_operational_exit_code_is_authoritative_when_output_is_malformed(tmp_path: Path) -> None:
    result, report_path, output_path = _run_parser(tmp_path, "not-json", audit_exit_code=2)

    assert result.returncode == 3
    assert "operational failure" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    output = output_path.read_text(encoding="utf-8")
    assert "count=unknown\n" in output
    assert "verdict=operational\n" in output


def test_unrelated_clean_dependency_cannot_replace_the_requested_audit_set(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("unrelated-clean", "9.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements="expected-vulnerable==1.0\n",
    )

    assert result.returncode == 4
    assert "missing audited dependency expected-vulnerable==1.0" in result.stderr
    assert "unexpected audited dependency unrelated-clean==9.0" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=contradictory\n" in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("requirements", "payload", "expected_error"),
    [
        (
            "expected==1.0\n",
            {"dependencies": [], "fixes": []},
            "missing audited dependency expected==1.0",
        ),
        (
            "expected==1.0\n",
            {
                "dependencies": [
                    _dependency("expected", "1.0", []),
                    _dependency("unexpected", "2.0", []),
                ],
                "fixes": [],
            },
            "unexpected audited dependency unexpected==2.0",
        ),
        (
            "expected==1.0\n",
            {"dependencies": [_dependency("expected", "2.0", [])], "fixes": []},
            "version mismatch for expected: requirements=1.0, audit=2.0",
        ),
    ],
)
def test_requirements_and_json_must_have_exact_name_version_equality(
    tmp_path: Path,
    requirements: str,
    payload: object,
    expected_error: str,
) -> None:
    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements=requirements,
    )

    assert result.returncode == 4
    assert expected_error in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=contradictory\n" in output_path.read_text(encoding="utf-8")


def test_names_versions_and_applicable_markers_are_normalized(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("my-package", "1.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements=('My_Package==1.0.0\nnot-applicable==7.0 ; python_version < "1"\n'),
    )

    assert result.returncode == 0, result.stderr
    assert "Dependencies audited: 1" in report_path.read_text(encoding="utf-8")
    assert "verdict=success\n" in output_path.read_text(encoding="utf-8")


def test_duplicate_normalized_requirement_is_malformed(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("my-package", "1.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements="My_Package==1.0\nmy-package==1.0.0\n",
    )

    assert result.returncode == 2
    assert "duplicate applicable requirement: my-package" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=malformed\n" in output_path.read_text(encoding="utf-8")


def test_duplicate_normalized_json_dependency_is_malformed(tmp_path: Path) -> None:
    payload = {
        "dependencies": [
            _dependency("My_Package", "1.0", []),
            _dependency("my-package", "1.0.0", []),
        ],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements="my-package==1.0\n",
    )

    assert result.returncode == 2
    assert "duplicate dependency in pip-audit JSON: my-package" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=malformed\n" in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "requirements",
    [
        "not-pinned>=1.0\n",
        "wildcard==1.*\n",
        "-r other-requirements.txt\n",
        "not a valid requirement\n",
    ],
)
def test_unverifiable_requirement_input_is_malformed(tmp_path: Path, requirements: str) -> None:
    payload = {
        "dependencies": [_dependency("safe", "1.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements=requirements,
    )

    assert result.returncode == 2
    assert "requirements input" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=malformed\n" in output_path.read_text(encoding="utf-8")


def test_invalid_json_dependency_name_is_malformed(tmp_path: Path) -> None:
    payload = {
        "dependencies": [_dependency("invalid name!", "1.0", [])],
        "fixes": [],
    }

    result, report_path, output_path = _run_parser(
        tmp_path,
        payload,
        audit_exit_code=0,
        requirements="safe==1.0\n",
    )

    assert result.returncode == 2
    assert "dependencies[0].name must be a valid package name" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=malformed\n" in output_path.read_text(encoding="utf-8")


def test_binary_malformed_audit_json_writes_diagnostics_without_traceback(tmp_path: Path) -> None:
    result, report_path, output_path = _run_parser(
        tmp_path,
        b"\xff\xfe\x00",
        audit_exit_code=0,
        requirements="safe==1.0\n",
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "pip-audit output is not valid UTF-8" in result.stderr
    assert "Audit failed" in report_path.read_text(encoding="utf-8")
    assert "verdict=malformed\n" in output_path.read_text(encoding="utf-8")
