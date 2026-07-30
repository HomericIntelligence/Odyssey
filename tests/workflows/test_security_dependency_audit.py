#!/usr/bin/env python3
"""Behavioral properties for deterministic dependency auditing in security.yml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "security.yml"
WORKFLOW_SMOKE = Path(__file__).parents[2] / ".github" / "workflows" / "workflow-smoke-test.yml"


@pytest.fixture(scope="module")
def workflow_content() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _step(workflow_content: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - name: |\Z)",
        workflow_content,
    )
    assert match is not None, f"Could not find workflow step {name!r}"
    return match.group("body")


def test_python_audit_targets_requirements_dev_from_locked_environment(workflow_content: str) -> None:
    step = _step(workflow_content, "Audit requirements-dev.txt with pip-audit")

    assert re.search(r"uv run --frozen pip-audit\b.*--requirement requirements-dev\.txt", step)
    assert "--strict" in step
    assert "--no-deps" in step
    assert re.search(r"(?m)^\s*pip-audit\s+--format json", step) is None
    assert "pip install requirements-dev.txt" not in step


def test_python_audit_preserves_exit_code_for_validated_parsing(workflow_content: str) -> None:
    step = _step(workflow_content, "Audit requirements-dev.txt with pip-audit")

    assert re.search(r"(?m)^\s*audit_rc=\$\?", step)
    assert "scripts/ci/parse_pip_audit.py" in step
    assert "--requirements requirements-dev.txt" in step
    assert '--audit-exit-code "$audit_rc"' in step
    assert '--github-output "$GITHUB_OUTPUT"' in step


def test_uv_lock_export_cannot_fall_back_to_an_empty_audit(workflow_content: str) -> None:
    step = _step(workflow_content, "Audit PyPI packages from the uv lockfile")

    assert "uv export --frozen" in step
    assert ": > uv-requirements.txt" not in step
    assert "No PyPI packages found in uv.lock — skipping pip-audit." not in step
    assert "test -s uv-requirements.txt" in step
    assert "grep -viE" not in step
    assert "pypi-from-uv.txt" not in step
    assert "--no-emit-project --all-groups > uv-requirements.txt" in step


def test_uv_lock_audit_uses_same_validated_parser(workflow_content: str) -> None:
    step = _step(workflow_content, "Audit PyPI packages from the uv lockfile")

    assert re.search(r"uv run --frozen pip-audit\b.*--requirement uv-requirements\.txt", step)
    assert "--strict" in step
    assert "--no-deps" in step
    assert "scripts/ci/parse_pip_audit.py" in step
    assert "--requirements uv-requirements.txt" in step
    assert '--audit-exit-code "$audit_rc"' in step
    assert '|| echo "?"' not in step
    assert '|| echo "0"' not in step


def test_redundant_safety_scan_and_mutable_tool_installs_are_removed(workflow_content: str) -> None:
    assert "Audit with Safety" not in workflow_content
    assert re.search(r"(?m)^\s+(?:python -m )?pip install .*pip-audit", workflow_content) is None
    assert re.search(r"(?m)^\s+(?:python -m )?pip install .*pip-licenses", workflow_content) is None
    assert "uses: ./.github/actions/setup-uv" in workflow_content
    assert "uv run --frozen pip-licenses" in workflow_content


def test_audit_jobs_emit_always_uploaded_manifests(workflow_content: str) -> None:
    for producer, manifest in {
        "Python": "python-audit-manifest.json",
        "Pixi": "pixi-audit-manifest.json",
        "License": "license-audit-manifest.json",
    }.items():
        manifest_step = _step(workflow_content, f"Write {producer} audit manifest")
        assert "if: always()" in manifest_step
        assert "scripts/ci/dependency_audit_contract.py manifest" in manifest_step
        assert manifest in manifest_step

    for upload_step in (
        "Upload Python audit results",
        "Upload Pixi audit results",
        "Upload license audit",
    ):
        step = _step(workflow_content, upload_step)
        assert "if: always()" in step
        assert "if-no-files-found: error" in step


def test_license_audit_fails_on_findings_or_tool_failure(workflow_content: str) -> None:
    step = _step(workflow_content, "Check licenses")

    assert "uv run --frozen pip-licenses" in step
    assert '--fail-on="GPL-3.0;AGPL-3.0"' in step
    assert 'exit "$license_rc"' in step
    assert "Failed to generate license report" not in step


def test_aggregate_uses_exact_named_downloads_and_fail_closed_validator(workflow_content: str) -> None:
    for artifact in (
        "python-audit-results",
        "pixi-audit-results",
        "license-audit-results",
    ):
        assert f"name: {artifact}" in workflow_content
        assert f"path: audit-results/{artifact}" in workflow_content

    aggregate = _step(workflow_content, "Generate and validate combined report")
    assert "scripts/ci/dependency_audit_contract.py aggregate" in aggregate
    assert "NEEDS_JSON" in aggregate
    assert "${{ toJSON(needs) }}" in aggregate
    assert "audit-results/**/*.md" not in workflow_content
    assert "${VULNS:-0}" not in workflow_content


def test_permissions_are_minimal_and_pr_code_has_no_write_token(workflow_content: str) -> None:
    assert re.search(r"(?m)^permissions:\n  contents: read\n\nconcurrency:", workflow_content)
    assert "uses: ./.github/actions/pr-comment" not in workflow_content
    assert sorted(re.findall(r"(?m)^      ([a-z-]+): write$", workflow_content)) == [
        "issues",
        "security-events",
    ]

    sast_job = re.search(r"(?ms)^  sast-scan:\n(?P<body>.*?)(?=^  [a-z].*:\n)", workflow_content)
    assert sast_job is not None
    assert "permissions:\n      contents: read\n      security-events: write" in sast_job.group("body")

    report_job = re.search(r"(?ms)^  audit-report:\n(?P<body>.*?)(?=^  [a-z].*:\n|\Z)", workflow_content)
    assert report_job is not None
    assert "permissions:\n      contents: read\n      issues: write" in report_job.group("body")
    assert "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)" in report_job.group("body")
    assert "ref: ${{ github.event.repository.default_branch }}" in report_job.group("body")
    assert "uses: ./" not in report_job.group("body")


@pytest.mark.parametrize("step_name", ["Upload Python audit results", "Upload Pixi audit results"])
def test_audit_artifacts_upload_even_when_validation_fails(workflow_content: str, step_name: str) -> None:
    step = _step(workflow_content, step_name)

    assert "if: always()" in step
    assert "if-no-files-found: error" in step


def test_workflow_smoke_runs_dependency_audit_regressions() -> None:
    smoke = WORKFLOW_SMOKE.read_text(encoding="utf-8")
    test_step = _step(smoke, "Run security workflow smoke tests")

    assert "tests/workflows/test_security_dependency_audit.py" in smoke
    assert "scripts/ci/parse_pip_audit.py" in smoke
    assert "scripts/ci/dependency_audit_contract.py" in smoke
    assert "tests/workflows/test_security_dependency_audit.py" in test_step
    assert "tests/scripts/test_parse_pip_audit.py" in test_step
    assert "tests/scripts/test_dependency_audit_contract.py" in test_step
