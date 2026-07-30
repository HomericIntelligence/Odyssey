#!/usr/bin/env python3
"""Validate and summarize modern pip-audit JSON without false-green fallbacks.

ADR-001 justification: Python is used because this CI helper requires JSON
schema validation and structured subprocess-output handling that Mojo's
automation APIs do not currently provide.

See docs/adr/ADR-001-language-selection-tooling.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version


class AuditError(ValueError):
    """Raised when pip-audit output cannot establish a trustworthy result."""


class MalformedAuditError(AuditError):
    """Raised when the JSON payload does not satisfy the audit schema."""


class OperationalAuditError(AuditError):
    """Raised when pip-audit or its output transport did not complete."""


class ContradictoryAuditError(AuditError):
    """Raised when independently reported audit outcomes disagree."""


@dataclass(frozen=True)
class Finding:
    """One vulnerability reported for a resolved dependency."""

    package: str
    version: str
    vulnerability_id: str
    fix_versions: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedDependency:
    """One normalized name/version pair covered by the audit."""

    name: str
    version: Version


@dataclass(frozen=True)
class AuditResult:
    """Validated audit contents used by the report and workflow outputs."""

    dependencies: tuple[ResolvedDependency, ...]
    findings: tuple[Finding, ...]


def _require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedAuditError(f"{path} must be a non-empty string")
    return value


def _require_string_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MalformedAuditError(f"{path} must be a list of strings")
    return tuple(value)


def parse_requirements(path: Path) -> dict[str, Version]:
    """Load the exact applicable pinned dependency set passed to pip-audit."""

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedAuditError("requirements input is not valid UTF-8") from exc
    except OSError as exc:
        raise OperationalAuditError(f"could not read requirements input: {exc}") from exc

    if not raw.strip():
        raise MalformedAuditError("requirements input is empty")

    dependencies: dict[str, Version] = {}
    ignored_option_prefixes = (
        "--index-url ",
        "--extra-index-url ",
        "--find-links ",
        "--trusted-host ",
    )
    for line_number, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith(ignored_option_prefixes):
            continue
        if text.startswith("-"):
            raise MalformedAuditError(
                f"requirements input line {line_number} uses unsupported include or option syntax"
            )
        try:
            requirement = Requirement(text)
        except InvalidRequirement as exc:
            raise MalformedAuditError(f"requirements input line {line_number} is invalid: {exc}") from exc
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if requirement.url is not None:
            raise MalformedAuditError(f"requirements input line {line_number} is not an exactly pinned name/version")

        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            raise MalformedAuditError(f"requirements input line {line_number} is not an exactly pinned name/version")
        try:
            version = Version(specifiers[0].version)
        except InvalidVersion as exc:
            raise MalformedAuditError(f"requirements input line {line_number} has an invalid pinned version") from exc

        name = canonicalize_name(requirement.name)
        if name in dependencies:
            raise MalformedAuditError(f"duplicate applicable requirement: {name}")
        dependencies[name] = version

    if not dependencies:
        raise MalformedAuditError("requirements input contains no applicable pinned dependencies")
    return dependencies


def parse_manifest(path: Path) -> AuditResult:
    """Load and validate pip-audit's modern ``dependencies``/``fixes`` schema."""

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedAuditError("pip-audit output is not valid UTF-8") from exc
    except OSError as exc:
        raise OperationalAuditError(f"could not read pip-audit output: {exc}") from exc

    if not raw.strip():
        raise MalformedAuditError("pip-audit output is empty")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedAuditError(f"pip-audit output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise MalformedAuditError("pip-audit JSON top level must be an object")

    missing_keys = {"dependencies", "fixes"} - payload.keys()
    if missing_keys:
        raise MalformedAuditError(f"pip-audit JSON is missing required key(s): {', '.join(sorted(missing_keys))}")

    dependencies = payload["dependencies"]
    fixes = payload["fixes"]
    if not isinstance(dependencies, list):
        raise MalformedAuditError("pip-audit JSON dependencies must be a list")
    if not isinstance(fixes, list):
        raise MalformedAuditError("pip-audit JSON fixes must be a list")
    if fixes:
        raise ContradictoryAuditError("non-fix pip-audit run returned non-empty fixes")

    findings: list[Finding] = []
    resolved_dependencies: list[ResolvedDependency] = []
    package_names: set[str] = set()
    for dependency_index, dependency in enumerate(dependencies):
        path_prefix = f"dependencies[{dependency_index}]"
        if not isinstance(dependency, dict):
            raise MalformedAuditError(f"{path_prefix} must be an object")

        package_value = _require_nonempty_string(dependency.get("name"), f"{path_prefix}.name")
        try:
            package = canonicalize_name(package_value, validate=True)
        except InvalidName as exc:
            raise MalformedAuditError(f"{path_prefix}.name must be a valid package name") from exc
        if package in package_names:
            raise MalformedAuditError(f"duplicate dependency in pip-audit JSON: {package}")
        package_names.add(package)

        if "skip_reason" in dependency:
            reason = _require_nonempty_string(dependency["skip_reason"], f"{path_prefix}.skip_reason")
            raise MalformedAuditError(f"pip-audit reported skipped dependency {package}: {reason}")

        version_value = _require_nonempty_string(dependency.get("version"), f"{path_prefix}.version")
        try:
            version = Version(version_value)
        except InvalidVersion as exc:
            raise MalformedAuditError(f"{path_prefix}.version must be a valid version") from exc
        resolved_dependencies.append(ResolvedDependency(package, version))
        vulnerabilities = dependency.get("vulns")
        if not isinstance(vulnerabilities, list):
            raise MalformedAuditError(f"{path_prefix}.vulns must be a list")

        for vulnerability_index, vulnerability in enumerate(vulnerabilities):
            vulnerability_path = f"{path_prefix}.vulns[{vulnerability_index}]"
            if not isinstance(vulnerability, dict):
                raise MalformedAuditError(f"{vulnerability_path} must be an object")
            vulnerability_id = _require_nonempty_string(vulnerability.get("id"), f"{vulnerability_path}.id")
            fix_versions = _require_string_list(vulnerability.get("fix_versions"), f"{vulnerability_path}.fix_versions")
            if "aliases" in vulnerability:
                _require_string_list(vulnerability["aliases"], f"{vulnerability_path}.aliases")
            if "description" in vulnerability and not isinstance(vulnerability["description"], str):
                raise MalformedAuditError(f"{vulnerability_path}.description must be a string")

            findings.append(Finding(package, str(version), vulnerability_id, fix_versions))

    return AuditResult(tuple(resolved_dependencies), tuple(findings))


def validate_dependency_set(requirements: dict[str, Version], result: AuditResult) -> None:
    """Require pip-audit JSON to cover exactly the requested applicable pins."""

    audited = {dependency.name: dependency.version for dependency in result.dependencies}
    diagnostics: list[str] = []
    for name in sorted(requirements.keys() - audited.keys()):
        diagnostics.append(f"missing audited dependency {name}=={requirements[name]}")
    for name in sorted(audited.keys() - requirements.keys()):
        diagnostics.append(f"unexpected audited dependency {name}=={audited[name]}")
    for name in sorted(requirements.keys() & audited.keys()):
        if requirements[name] != audited[name]:
            diagnostics.append(f"version mismatch for {name}: requirements={requirements[name]}, audit={audited[name]}")
    if diagnostics:
        raise ContradictoryAuditError("; ".join(diagnostics))


def validate_exit_code(result: AuditResult, audit_exit_code: int) -> None:
    """Require the subprocess exit code to agree with the validated manifest."""

    if audit_exit_code not in (0, 1):
        raise OperationalAuditError(f"pip-audit operational failure (exit code {audit_exit_code})")
    if audit_exit_code == 0 and result.findings:
        raise ContradictoryAuditError("pip-audit reported success despite vulnerability findings")
    if audit_exit_code == 1 and not result.findings:
        raise ContradictoryAuditError("pip-audit reported failure without vulnerability findings")


def render_report(result: AuditResult) -> str:
    """Render a diagnostic report from a validated audit result."""

    lines = [
        "## pip-audit Scan Results",
        "",
        f"Dependencies audited: {len(result.dependencies)}",
        "",
    ]
    if not result.findings:
        lines.append("✅ No known vulnerabilities found")
    else:
        lines.extend(
            [
                f"⚠️ Found {len(result.findings)} known vulnerabilities",
                "",
                "| Package | Version | Advisory | Fix versions |",
                "| --- | --- | --- | --- |",
            ]
        )
        for finding in result.findings:
            fixes = ", ".join(finding.fix_versions) or "None published"
            lines.append(f"| {finding.package} | {finding.version} | {finding.vulnerability_id} | {fixes} |")
    return "\n".join(lines) + "\n"


def render_error(message: str) -> str:
    """Render a report that clearly distinguishes an incomplete audit."""

    return "\n".join(
        [
            "## pip-audit Scan Results",
            "",
            f"❌ Audit failed: {message}",
            "",
            "No vulnerability verdict is available for this audit.",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="pip-audit JSON manifest")
    parser.add_argument("--requirements", type=Path, required=True, help="exact requirements input audited")
    parser.add_argument("--audit-exit-code", type=int, required=True, help="pip-audit subprocess exit code")
    parser.add_argument("--report", type=Path, required=True, help="Markdown report output")
    parser.add_argument("--github-output", type=Path, help="Optional GitHub Actions output file")
    return parser.parse_args(argv)


def write_outputs(path: Path | None, count: int | None, verdict: str, diagnostic: str) -> None:
    """Append a complete, single-line-safe GitHub Actions result contract."""

    if path is None:
        return
    safe_diagnostic = " ".join(diagnostic.splitlines()).strip()
    with path.open("a", encoding="utf-8") as output:
        output.write(f"count={count if count is not None else 'unknown'}\n")
        output.write(f"verdict={verdict}\n")
        output.write(f"diagnostic={safe_diagnostic}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.audit_exit_code not in (0, 1):
            raise OperationalAuditError(f"pip-audit operational failure (exit code {args.audit_exit_code})")
        requirements = parse_requirements(args.requirements)
        result = parse_manifest(args.input)
        validate_dependency_set(requirements, result)
        validate_exit_code(result, args.audit_exit_code)
    except AuditError as exc:
        args.report.write_text(render_error(str(exc)), encoding="utf-8")
        if isinstance(exc, OperationalAuditError):
            exit_code, verdict = 3, "operational"
        elif isinstance(exc, ContradictoryAuditError):
            exit_code, verdict = 4, "contradictory"
        else:
            exit_code, verdict = 2, "malformed"
        write_outputs(args.github_output, None, verdict, str(exc))
        print(str(exc), file=sys.stderr)
        return exit_code

    args.report.write_text(render_report(result), encoding="utf-8")
    finding_count = len(result.findings)
    if finding_count:
        diagnostic = f"{finding_count} known vulnerabilities found"
        write_outputs(args.github_output, finding_count, "findings", diagnostic)
        return 1
    write_outputs(args.github_output, 0, "success", "No known vulnerabilities found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
