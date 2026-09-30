#!/usr/bin/env python3
"""Fail-closed regression tests for the comprehensive CI report.

Issue #5731: the historical report could claim success after discovering zero
test groups, even when upstream jobs had failed during setup.  These tests
define the stdlib-only report evaluator contract before its implementation.
"""

import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_SCRIPT = REPO_ROOT / "scripts" / "ci" / "comprehensive_report.py"

REQUIRED_JOB_IDS = (
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
)

EXPECTED_PRODUCERS = {
    "comprehensive-core-tensors-loss": "test-mojo-comprehensive",
    "comprehensive-core-gradient-utils": "test-mojo-comprehensive",
    "comprehensive-data": "test-mojo-comprehensive",
    "comprehensive-autograd-tensor-base": "test-mojo-comprehensive",
    "comprehensive-models-misc": "test-mojo-comprehensive",
    "comprehensive-integration-bench": "test-mojo-comprehensive",
    "configs": "test-configs",
    "benchmarks": "test-benchmarks",
    "core-layers": "test-core-layers",
    "python-tests": "test-python",
    "gradient-checking": "gradient-tests",
    "data-utilities": "test-data-utilities",
}


@lru_cache(maxsize=1)
def _load_report_module() -> ModuleType:
    """Load the planned script lazily so a missing script is a RED assertion."""
    assert REPORT_SCRIPT.exists(), (
        "Issue #5731 requires scripts/ci/comprehensive_report.py; "
        "the fail-closed report evaluator has not been implemented yet."
    )
    spec = importlib.util.spec_from_file_location("comprehensive_report", REPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _successful_needs(**overrides: str) -> dict[str, dict[str, str]]:
    statuses = {job_id: "success" for job_id in REQUIRED_JOB_IDS}
    statuses.update(overrides)
    return {job_id: {"result": status} for job_id, status in statuses.items()}


def _write_manifest(
    artifacts_dir: Path,
    producer: str,
    *,
    status: str = "success",
    location: str | None = None,
) -> Path:
    manifest_dir = artifacts_dir / (location or producer)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "outcome-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "producer": producer,
                "job_id": EXPECTED_PRODUCERS[producer],
                "status": status,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_all_successful_manifests(artifacts_dir: Path) -> None:
    for producer in EXPECTED_PRODUCERS:
        _write_manifest(artifacts_dir, producer)


def _evaluate(
    *,
    needs: dict[str, dict[str, str]],
    artifacts_dir: Path,
    event_name: str = "pull_request",
    run_extended: bool = False,
) -> Any:
    report = _load_report_module()
    assert tuple(report.REQUIRED_JOB_IDS) == REQUIRED_JOB_IDS
    assert dict(report.EXPECTED_PRODUCERS) == EXPECTED_PRODUCERS
    return report.evaluate_report(
        needs=needs,
        artifacts_dir=artifacts_dir,
        event_name=event_name,
        run_extended=run_extended,
    )


def test_zero_artifacts_is_a_failure(tmp_path: Path) -> None:
    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    diagnostics = "\n".join(result.errors)
    for producer in EXPECTED_PRODUCERS:
        assert producer in diagnostics


def test_upstream_failure_cannot_be_hidden_by_passing_manifests(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)

    result = _evaluate(
        needs=_successful_needs(**{"mojo-compilation": "failure"}),
        artifacts_dir=tmp_path,
    )

    assert result.passed is False
    assert any("mojo-compilation" in diagnostic for diagnostic in result.errors)
    assert "mojo-compilation" in result.markdown


def test_unexpected_upstream_job_and_result_are_reported(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)
    needs = _successful_needs()
    needs["unexpected-setup"] = {"result": "failure"}

    result = _evaluate(needs=needs, artifacts_dir=tmp_path)

    assert result.passed is False
    assert "unexpected-setup" in result.markdown
    assert "failure" in result.markdown


def test_missing_manifest_is_a_failure(tmp_path: Path) -> None:
    missing_producer = "python-tests"
    for producer in EXPECTED_PRODUCERS:
        if producer != missing_producer:
            _write_manifest(tmp_path, producer)

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    assert any(missing_producer in diagnostic for diagnostic in result.errors)


def test_duplicate_manifest_is_a_failure(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)
    duplicated_producer = "configs"
    _write_manifest(tmp_path, duplicated_producer, location="duplicate-configs")

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    assert any(duplicated_producer in diagnostic and "duplicate" in diagnostic.lower() for diagnostic in result.errors)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "not JSON",
        "[]",
        '{"producer": "configs"}',
    ],
    ids=["empty", "invalid-json", "wrong-json-type", "missing-fields"],
)
def test_malformed_manifest_is_a_failure(tmp_path: Path, contents: str) -> None:
    _write_all_successful_manifests(tmp_path)
    malformed = tmp_path / "configs" / "outcome-manifest.json"
    malformed.write_text(contents, encoding="utf-8")

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    assert any("manifest" in diagnostic.lower() for diagnostic in result.errors)


def test_unexpected_manifest_is_a_failure(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)
    unexpected_dir = tmp_path / "unexpected"
    unexpected_dir.mkdir()
    (unexpected_dir / "outcome-manifest.json").write_text(
        json.dumps(
            {
                "producer": "unexpected-producer",
                "job_id": "test-python",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    assert any("unexpected-producer" in diagnostic for diagnostic in result.errors)


def test_manifest_status_cannot_contradict_non_matrix_job(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)
    _write_manifest(tmp_path, "configs", status="failure")

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    diagnostics = "\n".join(result.errors)
    assert "configs" in diagnostics
    assert "test-configs" in diagnostics


def test_manifest_job_mapping_cannot_contradict_contract(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)
    configs_manifest = tmp_path / "configs" / "outcome-manifest.json"
    configs_manifest.write_text(
        json.dumps(
            {
                "producer": "configs",
                "job_id": "test-python",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is False
    diagnostics = "\n".join(result.errors)
    assert "configs" in diagnostics
    assert "test-python" in diagnostics
    assert "test-configs" in diagnostics


def test_manifest_at_download_root_is_discovered(tmp_path: Path) -> None:
    root_level_producer = "configs"
    for producer in EXPECTED_PRODUCERS:
        _write_manifest(
            tmp_path,
            producer,
            location="." if producer == root_level_producer else producer,
        )

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is True
    assert result.errors == () or result.errors == []
    assert root_level_producer in result.markdown


@pytest.mark.parametrize("status", ["skipped", "cancelled"])
def test_required_job_that_did_not_succeed_is_a_failure(tmp_path: Path, status: str) -> None:
    _write_all_successful_manifests(tmp_path)

    result = _evaluate(
        needs=_successful_needs(**{"mojo-compilation": status}),
        artifacts_dir=tmp_path,
    )

    assert result.passed is False
    assert any("mojo-compilation" in diagnostic and status in diagnostic for diagnostic in result.errors)


@pytest.mark.parametrize(
    ("event_name", "run_extended"),
    [
        ("pull_request", False),
        ("merge_group", False),
        ("push", False),
        ("workflow_dispatch", False),
    ],
)
def test_simd_skip_is_allowed_when_extended_analysis_is_not_requested(
    tmp_path: Path,
    event_name: str,
    run_extended: bool,
) -> None:
    _write_all_successful_manifests(tmp_path)

    result = _evaluate(
        needs=_successful_needs(**{"simd-analysis": "skipped"}),
        artifacts_dir=tmp_path,
        event_name=event_name,
        run_extended=run_extended,
    )

    assert result.passed is True
    assert result.errors == () or result.errors == []


def test_extended_dispatch_requires_simd_success(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)

    result = _evaluate(
        needs=_successful_needs(**{"simd-analysis": "skipped"}),
        artifacts_dir=tmp_path,
        event_name="workflow_dispatch",
        run_extended=True,
    )

    assert result.passed is False
    assert any("simd-analysis" in diagnostic and "skipped" in diagnostic for diagnostic in result.errors)


def test_one_failed_matrix_shard_fails_the_aggregate(tmp_path: Path) -> None:
    failed_producer = "comprehensive-autograd-tensor-base"
    _write_all_successful_manifests(tmp_path)
    _write_manifest(tmp_path, failed_producer, status="failure")

    result = _evaluate(
        needs=_successful_needs(**{"test-mojo-comprehensive": "failure"}),
        artifacts_dir=tmp_path,
    )

    assert result.passed is False
    evidence = "\n".join((*result.errors, result.markdown))
    assert failed_producer in evidence
    assert "test-mojo-comprehensive" in evidence


def test_complete_green_contract_passes_and_reports_every_input(tmp_path: Path) -> None:
    _write_all_successful_manifests(tmp_path)

    result = _evaluate(needs=_successful_needs(), artifacts_dir=tmp_path)

    assert result.passed is True
    assert result.errors == () or result.errors == []
    for job_id in REQUIRED_JOB_IDS:
        assert job_id in result.markdown
    for producer in EXPECTED_PRODUCERS:
        assert producer in result.markdown
    assert "pass rate" not in result.markdown.lower()
    assert "total tests" not in result.markdown.lower()


def test_cli_malformed_input_writes_red_report_before_failing(tmp_path: Path) -> None:
    report = _load_report_module()
    output = tmp_path / "report.md"

    return_code = report.main(
        [
            "--needs-json",
            "not-json",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    assert return_code == 1
    assert output.is_file()
    contents = output.read_text(encoding="utf-8")
    assert "❌ FAIL —" in contents
    assert "Unable to load report inputs" in contents


def test_cli_aggregation_crash_writes_red_report_before_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _load_report_module()
    output = tmp_path / "report.md"

    def crash(**_kwargs: object) -> None:
        raise RuntimeError("simulated aggregation crash")

    monkeypatch.setattr(report, "evaluate_report", crash)
    return_code = report.main(
        [
            "--needs-json",
            "{}",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    assert return_code == 1
    assert output.is_file()
    contents = output.read_text(encoding="utf-8")
    assert "❌ FAIL —" in contents
    assert "simulated aggregation crash" in contents


def test_isolated_report_cli_ignores_adjacent_stdlib_shadow(
    tmp_path: Path,
) -> None:
    script_dir = tmp_path / "scripts" / "ci"
    script_dir.mkdir(parents=True)
    copied_report = script_dir / REPORT_SCRIPT.name
    copied_report.write_bytes(REPORT_SCRIPT.read_bytes())
    shadow_marker = tmp_path / "shadow-json-imported"
    (script_dir / "json.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(shadow_marker)!r}).write_text('executed', encoding='utf-8')\n"
        "raise RuntimeError('shadow json imported')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-I", str(copied_report), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not shadow_marker.exists()
