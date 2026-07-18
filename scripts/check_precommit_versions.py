#!/usr/bin/env python3
"""Check pre-commit version consistency against pyproject.toml.

Post-ADR-018 (uv migration): dependencies live in pyproject.toml as PEP 508
requirement strings under [project.dependencies] and the [dependency-groups]
tables, not in pixi.toml's TOML-table form. This script parses those strings
and compares the declared lower bounds against the pinned pre-commit hook revs.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from hephaestus.ci.precommit import (
    DEFAULT_HOOK_TO_PIXI_MAP,
    check_version_drift,
    extract_external_hooks,
    load_precommit_config,
)
from hephaestus.utils.helpers import get_repo_root

# name (>=|==|~=) X.Y[.Z...]  — capture the package name and the lower bound.
_REQ_RE = re.compile(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?:>=|==|~=)\s*([0-9]+(?:\.[0-9]+)*)")


def _parse_requirement(spec: str) -> tuple[str, str] | None:
    """Return (package_lower, lower_bound) from a PEP 508 requirement string."""
    m = _REQ_RE.match(spec.strip())
    if not m:
        return None
    return m.group(1).lower(), m.group(2)


def load_all_pyproject_versions(pyproject_path: Path) -> dict[str, str]:
    """Load pyproject.toml → package→lower-bound from ALL dependency sections.

    Covers [project.dependencies] plus every list under [dependency-groups]
    (dev, notebook, ...) so dev-only packages (mypy, nbstripout,
    pre-commit-hooks) are found — the same coverage the old pixi.toml reader had.
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)

    versions: dict[str, str] = {}

    def _ingest(specs: list) -> None:
        for spec in specs:
            parsed = _parse_requirement(str(spec))
            if parsed:
                pkg, ver = parsed
                versions[pkg] = ver

    _ingest(data.get("project", {}).get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        if isinstance(group, list):
            _ingest(group)

    return versions


def main() -> int:
    root = get_repo_root()
    precommit_path = root / ".pre-commit-config.yaml"
    pyproject_path = root / "pyproject.toml"

    repos = load_precommit_config(precommit_path)
    external_hooks = extract_external_hooks(repos)
    declared_versions = load_all_pyproject_versions(pyproject_path)
    issues = check_version_drift(external_hooks, declared_versions, DEFAULT_HOOK_TO_PIXI_MAP)

    if issues:
        print("Pre-commit version drift detected:")
        for issue in issues:
            print(f"  - {issue}")
        print(
            "\nFix: update the rev in .pre-commit-config.yaml or the version "
            "constraint in pyproject.toml so they match."
        )
        return 1

    print("OK: all pre-commit hook versions are consistent with pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
