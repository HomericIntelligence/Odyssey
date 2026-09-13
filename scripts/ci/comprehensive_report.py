#!/usr/bin/env python3
"""Build the fail-closed comprehensive CI report.

ADR-001 Justification: Python is required for CI automation that parses JSON,
walks downloaded artifact trees, and returns a reliable process exit status.
Mojo's subprocess support cannot provide the exit-code guarantees needed for
this gate. See: docs/adr/ADR-001-language-selection-tooling.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


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

_VALID_STATUSES = frozenset({"success", "failure", "cancelled", "skipped"})
_SIMD_SKIP_EVENTS = frozenset({"pull_request", "merge_group", "push", "workflow_dispatch"})


@dataclass(frozen=True)
class EvaluationResult:
    """The report text and its fail-closed verdict."""

    passed: bool
    errors: tuple[str, ...]
    markdown: str


@dataclass(frozen=True)
class _Manifest:
    producer: str
    job_id: str
    status: str
    path: Path


def _display(value: object) -> str:
    """Render untrusted values safely inside a Markdown table cell."""
    return str(value).replace("|", r"\|").replace("\n", " ")


def _simd_skip_allowed(event_name: str, run_extended: bool) -> bool:
    return event_name in _SIMD_SKIP_EVENTS and not (event_name == "workflow_dispatch" and run_extended)


def _validate_needs(
    needs: Mapping[str, object],
    *,
    event_name: str,
    run_extended: bool,
) -> tuple[dict[str, str], list[str]]:
    statuses: dict[str, str] = {}
    errors: list[str] = []

    missing_jobs = [job_id for job_id in REQUIRED_JOB_IDS if job_id not in needs]
    for job_id in missing_jobs:
        errors.append(f"Required upstream job {job_id!r} is missing from needs.")

    unexpected_jobs = sorted(set(needs) - set(REQUIRED_JOB_IDS))
    for job_id in unexpected_jobs:
        record = needs[job_id]
        result = record.get("result") if isinstance(record, Mapping) else None
        status = result if isinstance(result, str) and result else "invalid"
        statuses[job_id] = status
        errors.append(f"Unexpected upstream job {job_id!r} is present in needs with result {status!r}.")

    for job_id in REQUIRED_JOB_IDS:
        record = needs.get(job_id)
        if not isinstance(record, Mapping):
            statuses[job_id] = "missing" if record is None else "invalid"
            if job_id in needs:
                errors.append(f"Upstream job {job_id!r} has an invalid needs record.")
            continue

        result = record.get("result")
        if not isinstance(result, str) or not result:
            statuses[job_id] = "invalid"
            errors.append(f"Upstream job {job_id!r} has no valid result in needs.")
            continue

        statuses[job_id] = result
        if result == "success":
            continue
        if job_id == "simd-analysis" and result == "skipped" and _simd_skip_allowed(event_name, run_extended):
            continue
        errors.append(f"Required upstream job {job_id!r} concluded with {result!r}, not success.")

    return statuses, errors


def _load_manifest(path: Path) -> tuple[_Manifest | None, str | None]:
    try:
        contents = path.read_text(encoding="utf-8")
        payload: Any = json.loads(contents)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"Manifest {path} could not be read as JSON: {exc}."

    if not isinstance(payload, dict):
        return None, f"Manifest {path} must contain a JSON object."

    required_fields = ("producer", "job_id", "status")
    invalid_fields = [
        field for field in required_fields if not isinstance(payload.get(field), str) or not payload[field]
    ]
    if invalid_fields:
        fields = ", ".join(invalid_fields)
        return None, f"Manifest {path} has missing or invalid fields: {fields}."

    status = payload["status"]
    if status not in _VALID_STATUSES:
        return None, f"Manifest {path} has unsupported status {status!r}."

    return (
        _Manifest(
            producer=payload["producer"],
            job_id=payload["job_id"],
            status=status,
            path=path,
        ),
        None,
    )


def _validate_manifests(
    artifacts_dir: Path,
    job_statuses: Mapping[str, str],
) -> tuple[dict[str, list[_Manifest]], list[str]]:
    manifests: dict[str, list[_Manifest]] = {producer: [] for producer in EXPECTED_PRODUCERS}
    errors: list[str] = []

    if not artifacts_dir.is_dir():
        errors.append(f"Artifacts directory {artifacts_dir} does not exist or is not a directory.")
        paths: Sequence[Path] = ()
    else:
        paths = sorted(artifacts_dir.rglob("outcome-manifest.json"))

    for path in paths:
        manifest, error = _load_manifest(path)
        if error is not None:
            errors.append(error)
            continue
        assert manifest is not None

        if manifest.producer not in EXPECTED_PRODUCERS:
            errors.append(f"Manifest {path} names unexpected producer {manifest.producer!r}.")
            continue

        manifests[manifest.producer].append(manifest)

    for producer, expected_job_id in EXPECTED_PRODUCERS.items():
        producer_manifests = manifests[producer]
        if not producer_manifests:
            errors.append(f"Expected producer {producer!r} has no outcome manifest.")
            continue
        if len(producer_manifests) > 1:
            locations = ", ".join(str(item.path) for item in producer_manifests)
            errors.append(f"Expected producer {producer!r} has duplicate outcome manifests: {locations}.")
            continue

        manifest = producer_manifests[0]
        if manifest.job_id != expected_job_id:
            errors.append(
                f"Producer {producer!r} names job {manifest.job_id!r}; the contract requires {expected_job_id!r}."
            )

        if manifest.status != "success":
            errors.append(
                f"Producer {producer!r} for job {expected_job_id!r} concluded with {manifest.status!r}, not success."
            )

        upstream_status = job_statuses.get(expected_job_id)
        if (
            expected_job_id != "test-mojo-comprehensive"
            and upstream_status in _VALID_STATUSES
            and manifest.status != upstream_status
        ):
            errors.append(
                f"Producer {producer!r} status {manifest.status!r} "
                f"contradicts upstream job {expected_job_id!r} status "
                f"{upstream_status!r}."
            )

    return manifests, errors


def _build_markdown(
    *,
    passed: bool,
    errors: Sequence[str],
    job_statuses: Mapping[str, str],
    manifests: Mapping[str, Sequence[_Manifest]],
) -> str:
    verdict = (
        "✅ PASS — upstream jobs and producer manifests are complete and successful."
        if passed
        else "❌ FAIL — the comprehensive CI contract is incomplete or unsuccessful."
    )
    lines = [
        "# 🧪 Comprehensive Test Results",
        "",
        verdict,
        "",
        "## Upstream jobs",
        "",
        "| Job | Result |",
        "| --- | --- |",
    ]
    reported_job_ids = (
        *REQUIRED_JOB_IDS,
        *sorted(set(job_statuses) - set(REQUIRED_JOB_IDS)),
    )
    for job_id in reported_job_ids:
        lines.append(f"| `{_display(job_id)}` | `{_display(job_statuses.get(job_id, 'missing'))}` |")

    lines.extend(
        [
            "",
            "## Result producers",
            "",
            "| Producer | Expected job | Manifest status |",
            "| --- | --- | --- |",
        ]
    )
    for producer, job_id in EXPECTED_PRODUCERS.items():
        producer_manifests = manifests.get(producer, ())
        if len(producer_manifests) == 1:
            status = producer_manifests[0].status
        elif len(producer_manifests) > 1:
            status = "duplicate"
        else:
            status = "missing"
        lines.append(f"| `{_display(producer)}` | `{_display(job_id)}` | `{_display(status)}` |")

    lines.extend(["", "## Diagnostics", ""])
    if errors:
        lines.extend(f"- {_display(error)}" for error in errors)
    else:
        lines.append("- No contract violations detected.")
    lines.append("")
    return "\n".join(lines)


def evaluate_report(
    *,
    needs: Mapping[str, object],
    artifacts_dir: Path | str,
    event_name: str,
    run_extended: bool,
) -> EvaluationResult:
    """Validate upstream job results and recursively discovered manifests."""
    if not isinstance(needs, Mapping):
        needs = {}
        initial_errors: list[str] = ["The needs payload must be a JSON object."]
    else:
        initial_errors = []

    job_statuses, needs_errors = _validate_needs(
        needs,
        event_name=event_name,
        run_extended=run_extended,
    )
    manifests, manifest_errors = _validate_manifests(
        Path(artifacts_dir),
        job_statuses,
    )
    errors = tuple((*initial_errors, *needs_errors, *manifest_errors))
    passed = not errors
    markdown = _build_markdown(
        passed=passed,
        errors=errors,
        job_statuses=job_statuses,
        manifests=manifests,
    )
    return EvaluationResult(passed=passed, errors=errors, markdown=markdown)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the fail-closed comprehensive CI report.")
    needs_group = parser.add_mutually_exclusive_group()
    needs_group.add_argument(
        "--needs-json",
        help="GitHub Actions needs context as inline JSON.",
    )
    needs_group.add_argument(
        "--needs-file",
        type=Path,
        help="Path containing the GitHub Actions needs context as JSON.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "test-results")),
        help="Directory containing downloaded result artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "REPORT_OUTPUT",
                "comprehensive-test-report.md",
            )
        ),
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME", "pull_request"),
        help="GitHub event name.",
    )
    parser.add_argument(
        "--run-extended",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=_parse_bool(os.environ.get("RUN_EXTENDED", "false")),
        help="Whether extended SIMD analysis was requested.",
    )
    return parser


def _read_needs(args: argparse.Namespace) -> Mapping[str, object]:
    if args.needs_file is not None:
        payload = json.loads(args.needs_file.read_text(encoding="utf-8"))
    else:
        raw_needs = (
            args.needs_json
            if args.needs_json is not None
            else os.environ.get("CI_NEEDS_JSON", os.environ.get("NEEDS_JSON"))
        )
        if raw_needs is None:
            raise ValueError("needs JSON is required via --needs-json, --needs-file, CI_NEEDS_JSON, or NEEDS_JSON")
        payload = json.loads(raw_needs)

    if not isinstance(payload, dict):
        raise ValueError("needs JSON must contain an object")
    return payload


def _input_failure(message: str) -> EvaluationResult:
    errors = (message,)
    markdown = _build_markdown(
        passed=False,
        errors=errors,
        job_statuses={},
        manifests={},
    )
    return EvaluationResult(passed=False, errors=errors, markdown=markdown)


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        needs = _read_needs(args)
        result = evaluate_report(
            needs=needs,
            artifacts_dir=args.artifacts_dir,
            event_name=args.event_name,
            run_extended=args.run_extended,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        result = _input_failure(f"Unable to load report inputs: {exc}.")
    except Exception as exc:  # pragma: no cover - last-resort CI fail-closed path
        result = _input_failure(f"Report aggregation crashed with {type(exc).__name__}: {exc}.")

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.markdown, encoding="utf-8")
    except OSError as exc:
        print(f"Unable to write report to {args.output}: {exc}", file=sys.stderr)
        return 1

    print(result.markdown)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
