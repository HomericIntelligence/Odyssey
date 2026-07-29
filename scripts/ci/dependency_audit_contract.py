#!/usr/bin/env python3
"""Write and validate fail-closed dependency-audit outcome artifacts.

ADR-001 justification: Python is used because this CI helper requires recursive
artifact discovery and strict JSON validation that Mojo's automation APIs do
not currently provide.

See docs/adr/ADR-001-language-selection-tooling.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_PRODUCERS = {
    "python-audit": ("python-audit-manifest.json", "pip-audit-report.md"),
    "pixi-audit": ("pixi-audit-manifest.json", "pixi-audit-report.md"),
    "license-audit": ("license-audit-manifest.json", "license-report.md"),
}
EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "producer",
    "status",
    "verdict",
    "findings",
    "report",
}
ALLOWED_JOB_RESULTS = {"success", "failure", "cancelled", "skipped"}
ALLOWED_VERDICTS = {"success", "findings", "malformed", "operational", "contradictory"}


class ContractError(ValueError):
    """Raised when a manifest cannot be written safely."""


@dataclass(frozen=True)
class Manifest:
    """One validated producer outcome."""

    producer: str
    status: str
    verdict: str
    findings: int | None
    report_path: Path


def _parse_findings(raw: str) -> int | None:
    if raw == "unknown":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError("findings must be a non-negative integer or 'unknown'") from exc
    if value < 0:
        raise ContractError("findings must be a non-negative integer or 'unknown'")
    return value


def write_manifest(args: argparse.Namespace) -> int:
    """Write a producer manifest after checking its diagnostic report exists."""

    if args.producer not in EXPECTED_PRODUCERS:
        raise ContractError(f"unexpected producer: {args.producer}")
    if args.status not in ALLOWED_JOB_RESULTS:
        raise ContractError(f"invalid job status: {args.status}")
    if args.verdict not in ALLOWED_VERDICTS:
        raise ContractError(f"invalid audit verdict: {args.verdict}")
    if not args.report.is_file() or not args.report.read_text(encoding="utf-8").strip():
        raise ContractError(f"report is missing or empty: {args.report}")

    findings = _parse_findings(args.findings)
    payload = {
        "schema_version": 1,
        "producer": args.producer,
        "status": args.status,
        "verdict": args.verdict,
        "findings": findings,
        "report": args.report.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _read_needs(raw: str, diagnostics: list[str]) -> dict[str, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        diagnostics.append(f"needs JSON is malformed: {exc}")
        return {}
    if not isinstance(payload, dict):
        diagnostics.append("needs JSON must be an object")
        return {}

    results: dict[str, str] = {}
    expected = set(EXPECTED_PRODUCERS)
    actual = set(payload)
    for missing in sorted(expected - actual):
        diagnostics.append(f"missing upstream result for {missing}")
    for unexpected in sorted(actual - expected):
        diagnostics.append(f"unexpected upstream result for {unexpected}")

    for producer, value in sorted(payload.items()):
        if not isinstance(value, dict):
            diagnostics.append(f"upstream {producer} metadata must be an object")
            results[producer] = "invalid"
            continue
        result = value.get("result")
        if result not in ALLOWED_JOB_RESULTS:
            diagnostics.append(f"upstream {producer} has invalid result {result!r}")
            results[producer] = "invalid"
            continue
        results[producer] = result
        if producer in expected and result != "success":
            diagnostics.append(f"upstream {producer} result is {result}")
    return results


def _load_manifest(path: Path, producer: str) -> Manifest:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("manifest is not valid UTF-8") from exc
    except OSError as exc:
        raise ContractError(f"could not read manifest: {exc}") from exc
    if not raw.strip():
        raise ContractError("manifest is empty")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("manifest top level must be an object")
    if set(payload) != EXPECTED_MANIFEST_KEYS:
        missing = sorted(EXPECTED_MANIFEST_KEYS - set(payload))
        unexpected = sorted(set(payload) - EXPECTED_MANIFEST_KEYS)
        details = []
        if missing:
            details.append(f"missing keys {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected keys {', '.join(unexpected)}")
        raise ContractError("; ".join(details))
    if payload["schema_version"] != 1:
        raise ContractError("schema_version must be 1")
    if payload["producer"] != producer:
        raise ContractError(f"producer must be {producer!r}")

    status = payload["status"]
    verdict = payload["verdict"]
    findings = payload["findings"]
    expected_report = EXPECTED_PRODUCERS[producer][1]
    if status not in ALLOWED_JOB_RESULTS:
        raise ContractError(f"invalid status {status!r}")
    if verdict not in ALLOWED_VERDICTS:
        raise ContractError(f"invalid verdict {verdict!r}")
    if findings is not None and (isinstance(findings, bool) or not isinstance(findings, int) or findings < 0):
        raise ContractError("findings must be a non-negative integer or null")
    if payload["report"] != expected_report:
        raise ContractError(f"report must be {expected_report!r}")

    report_path = path.parent / expected_report
    try:
        report_text = report_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("report is not valid UTF-8") from exc
    except OSError as exc:
        raise ContractError(f"report is missing: {exc}") from exc
    if not report_text.strip():
        raise ContractError("report is empty")
    return Manifest(producer, status, verdict, findings, report_path)


def _discover_manifests(root: Path, diagnostics: list[str]) -> dict[str, Manifest]:
    manifests: dict[str, Manifest] = {}
    if not root.is_dir():
        diagnostics.append(f"artifact root is missing: {root}")
        return manifests

    expected_names = {details[0]: producer for producer, details in EXPECTED_PRODUCERS.items()}
    candidates = sorted(path for path in root.rglob("*-audit-manifest.json") if path.is_file())
    grouped: dict[str, list[Path]] = {}
    for path in candidates:
        producer = expected_names.get(path.name)
        if producer is None:
            diagnostics.append(f"unexpected manifest: {path.relative_to(root)}")
            continue
        grouped.setdefault(producer, []).append(path)

    for producer in EXPECTED_PRODUCERS:
        paths = grouped.get(producer, [])
        if not paths:
            diagnostics.append(f"missing manifest for {producer}")
            continue
        if len(paths) > 1:
            diagnostics.append(f"duplicate manifest for {producer}: {len(paths)} copies")
            continue
        try:
            manifests[producer] = _load_manifest(paths[0], producer)
        except ContractError as exc:
            diagnostics.append(f"invalid manifest for {producer}: {exc}")
    return manifests


def _validate_consistency(
    needs: dict[str, str],
    manifests: dict[str, Manifest],
    diagnostics: list[str],
) -> None:
    for producer, manifest in manifests.items():
        result = needs.get(producer)
        if result in ALLOWED_JOB_RESULTS and manifest.status != result:
            diagnostics.append(f"{producer} manifest status {manifest.status} contradicts upstream result {result}")
        if manifest.status != "success":
            diagnostics.append(f"{producer} manifest status is {manifest.status}")
        if manifest.verdict != "success":
            diagnostics.append(f"{producer} manifest verdict is {manifest.verdict}")
        if manifest.verdict == "success" and manifest.findings != 0:
            diagnostics.append(f"{producer} success manifest has nonzero findings")


def _render_report(
    needs: dict[str, str],
    manifests: dict[str, Manifest],
    diagnostics: list[str],
) -> str:
    passing = not diagnostics
    lines = [
        "# Dependency Audit Report",
        "",
        "✅ PASS — all dependency audits and artifacts are complete and successful."
        if passing
        else "❌ FAIL — dependency audit evidence is incomplete or unsuccessful.",
        "",
        "## Upstream jobs",
        "",
        "| Job | Result |",
        "| --- | --- |",
    ]
    for producer in sorted(set(EXPECTED_PRODUCERS) | set(needs)):
        lines.append(f"| `{producer}` | {needs.get(producer, 'missing')} |")

    lines.extend(
        [
            "",
            "## Producer artifacts",
            "",
            "| Producer | Manifest status | Verdict | Findings |",
            "| --- | --- | --- | --- |",
        ]
    )
    for producer in EXPECTED_PRODUCERS:
        manifest = manifests.get(producer)
        if manifest is None:
            lines.append(f"| `{producer}` | missing/invalid | unavailable | unavailable |")
        else:
            findings = manifest.findings if manifest.findings is not None else "unknown"
            lines.append(f"| `{producer}` | {manifest.status} | {manifest.verdict} | {findings} |")

    lines.extend(["", "## Diagnostics", ""])
    if diagnostics:
        lines.extend(f"- ❌ {diagnostic}" for diagnostic in diagnostics)
    else:
        lines.append("- No contract violations.")

    for producer in EXPECTED_PRODUCERS:
        manifest = manifests.get(producer)
        if manifest is None:
            continue
        lines.extend(
            [
                "",
                "---",
                "",
                f"## {producer} diagnostic report",
                "",
                manifest.report_path.read_text(encoding="utf-8").rstrip(),
            ]
        )
    return "\n".join(lines) + "\n"


def aggregate(args: argparse.Namespace) -> int:
    """Generate the combined report, then enforce the complete contract."""

    diagnostics: list[str] = []
    needs = _read_needs(args.needs_json, diagnostics)
    manifests = _discover_manifests(args.artifacts, diagnostics)
    _validate_consistency(needs, manifests, diagnostics)
    args.report.write_text(_render_report(needs, manifests, diagnostics), encoding="utf-8")
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="write one producer outcome manifest")
    manifest.add_argument("--producer", required=True)
    manifest.add_argument("--status", required=True)
    manifest.add_argument("--verdict", required=True)
    manifest.add_argument("--findings", required=True)
    manifest.add_argument("--report", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("aggregate", help="validate artifacts and write the combined report")
    report.add_argument("--artifacts", type=Path, required=True)
    report.add_argument("--needs-json", required=True)
    report.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "manifest":
            return write_manifest(args)
        return aggregate(args)
    except (ContractError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
