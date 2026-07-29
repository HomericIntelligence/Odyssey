#!/usr/bin/env python3
"""Behavioral tests for the fail-closed dependency-audit artifact contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "dependency_audit_contract.py"

PRODUCERS = {
    "python-audit": ("python-audit-manifest.json", "pip-audit-report.md"),
    "pixi-audit": ("pixi-audit-manifest.json", "pixi-audit-report.md"),
    "license-audit": ("license-audit-manifest.json", "license-report.md"),
}


def _write_artifact(
    root: Path,
    producer: str,
    *,
    status: str = "success",
    verdict: str = "success",
    findings: int | None = 0,
    nested: bool = True,
) -> None:
    manifest_name, report_name = PRODUCERS[producer]
    destination = root / producer / "nested" if nested else root
    destination.mkdir(parents=True, exist_ok=True)
    (destination / report_name).write_text(f"# {producer} report\n", encoding="utf-8")
    (destination / manifest_name).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": producer,
                "status": status,
                "verdict": verdict,
                "findings": findings,
                "report": report_name,
            }
        ),
        encoding="utf-8",
    )


def _run_aggregate(
    tmp_path: Path,
    *,
    needs: dict[str, dict[str, str]],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report = tmp_path / "combined-report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "aggregate",
            "--artifacts",
            str(artifacts),
            "--needs-json",
            json.dumps(needs),
            "--report",
            str(report),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, report


def _successful_needs() -> dict[str, dict[str, str]]:
    return {producer: {"result": "success"} for producer in PRODUCERS}


def test_complete_successful_artifacts_are_green_even_when_nested(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode == 0, result.stderr
    content = report.read_text(encoding="utf-8")
    assert "✅ PASS" in content
    for producer in PRODUCERS:
        assert f"| `{producer}` | success |" in content


@pytest.mark.parametrize("missing", list(PRODUCERS))
def test_missing_artifact_fails_and_names_the_producer(tmp_path: Path, missing: str) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        if producer != missing:
            _write_artifact(artifacts, producer)

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    content = report.read_text(encoding="utf-8")
    assert "❌ FAIL" in content
    assert f"missing manifest for {missing}" in content


def test_duplicate_manifest_fails_closed(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)
    duplicate = artifacts / "duplicate"
    duplicate.mkdir(parents=True)
    source = next(artifacts.rglob("python-audit-manifest.json"))
    (duplicate / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    assert "duplicate manifest for python-audit" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("payload", ["", "{not json", "[]"])
def test_malformed_manifest_fails_closed(tmp_path: Path, payload: str) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)
    manifest = next(artifacts.rglob("pixi-audit-manifest.json"))
    manifest.write_text(payload, encoding="utf-8")

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    content = report.read_text(encoding="utf-8")
    assert "invalid manifest for pixi-audit" in content


def test_binary_malformed_manifest_is_reported_without_crashing_aggregate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)
    manifest = next(artifacts.rglob("pixi-audit-manifest.json"))
    manifest.write_bytes(b"\xff\xfe\x00")

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    content = report.read_text(encoding="utf-8")
    assert "❌ FAIL" in content
    assert "invalid manifest for pixi-audit: manifest is not valid UTF-8" in content


def test_binary_malformed_report_is_reported_without_crashing_aggregate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)
    report_artifact = next(artifacts.rglob("pixi-audit-report.md"))
    report_artifact.write_bytes(b"\xff\xfe\x00")

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    content = report.read_text(encoding="utf-8")
    assert "❌ FAIL" in content
    assert "invalid manifest for pixi-audit: report is not valid UTF-8" in content


@pytest.mark.parametrize("result_name", ["failure", "cancelled", "skipped"])
def test_non_successful_upstream_result_is_red(tmp_path: Path, result_name: str) -> None:
    artifacts = tmp_path / "artifacts"
    for producer in PRODUCERS:
        _write_artifact(artifacts, producer)
    needs = _successful_needs()
    needs["python-audit"]["result"] = result_name

    result, report = _run_aggregate(tmp_path, needs=needs)

    assert result.returncode != 0
    content = report.read_text(encoding="utf-8")
    assert f"| `python-audit` | {result_name} |" in content
    assert f"upstream python-audit result is {result_name}" in content


def test_finding_or_operational_manifest_cannot_be_green(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_artifact(artifacts, "python-audit", status="failure", verdict="findings", findings=2)
    _write_artifact(artifacts, "pixi-audit", status="failure", verdict="operational", findings=None)
    _write_artifact(artifacts, "license-audit")
    needs = _successful_needs()
    needs["python-audit"]["result"] = "failure"
    needs["pixi-audit"]["result"] = "failure"

    result, report = _run_aggregate(tmp_path, needs=needs)

    assert result.returncode != 0
    content = report.read_text(encoding="utf-8")
    assert "python-audit manifest verdict is findings" in content
    assert "pixi-audit manifest verdict is operational" in content


def test_contradictory_success_manifest_with_findings_fails(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_artifact(artifacts, "python-audit", findings=3)
    _write_artifact(artifacts, "pixi-audit")
    _write_artifact(artifacts, "license-audit")

    result, report = _run_aggregate(tmp_path, needs=_successful_needs())

    assert result.returncode != 0
    assert "success manifest has nonzero findings" in report.read_text(encoding="utf-8")


def test_manifest_writer_creates_strict_json_contract(tmp_path: Path) -> None:
    report = tmp_path / "pip-audit-report.md"
    report.write_text("# report\n", encoding="utf-8")
    manifest = tmp_path / "python-audit-manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "manifest",
            "--producer",
            "python-audit",
            "--status",
            "success",
            "--verdict",
            "success",
            "--findings",
            "0",
            "--report",
            str(report),
            "--output",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "producer": "python-audit",
        "status": "success",
        "verdict": "success",
        "findings": 0,
        "report": "pip-audit-report.md",
    }
