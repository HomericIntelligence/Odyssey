#!/usr/bin/env python3
"""Validate dependency consistency across project configuration files.

Checks:
1. Every package in requirements*.txt has a corresponding entry in pixi.toml.
2. Pinned versions in requirements*.txt fall within pixi.toml range constraints.
3. pyproject.toml does not declare [project.dependencies] or
   [project.optional-dependencies].

Usage:
    python scripts/check_dep_sync.py
    python scripts/check_dep_sync.py --repo-root /path/to/repo
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


class VersionRange(NamedTuple):
    """A single version constraint (operator + version)."""

    op: str
    version: Tuple[int, ...]


def parse_version(v: str) -> Tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints."""
    return tuple(int(x) for x in v.split("."))


def parse_pixi_constraints(spec: str) -> List[VersionRange]:
    """Parse a pixi.toml version spec like ``>=1.2.0,<2`` into constraints."""
    constraints: List[VersionRange] = []
    # Remove surrounding quotes
    spec = spec.strip().strip('"').strip("'")
    for part in spec.split(","):
        part = part.strip()
        match = re.match(r"(>=|<=|>|<|==|!=|~=)(.+)", part)
        if match:
            op, ver = match.group(1), match.group(2)
            constraints.append(VersionRange(op=op, version=parse_version(ver)))
    return constraints


def version_satisfies(version: Tuple[int, ...], constraints: List[VersionRange]) -> bool:
    """Check if *version* satisfies all *constraints*."""
    for c in constraints:
        # Pad shorter tuple with zeros for comparison
        v = version + (0,) * max(0, len(c.version) - len(version))
        cv = c.version + (0,) * max(0, len(version) - len(c.version))
        if c.op == ">=" and not (v >= cv):
            return False
        if c.op == "<=" and not (v <= cv):
            return False
        if c.op == ">" and not (v > cv):
            return False
        if c.op == "<" and not (v < cv):
            return False
        if c.op == "==" and not (v == cv):
            return False
        if c.op == "!=" and (v == cv):
            return False
    return True


def parse_pixi_toml(path: Path) -> Dict[str, str]:
    """Extract package name → version spec from pixi.toml [dependencies]."""
    deps: Dict[str, str] = {}
    in_deps = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[dependencies]"):
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
            continue
        if in_deps and "=" in stripped and not stripped.startswith("#"):
            # Handle inline comments
            line_no_comment = stripped.split("#")[0].strip()
            match = re.match(r'(\S+)\s*=\s*"([^"]+)"', line_no_comment)
            if match:
                deps[match.group(1)] = match.group(2)
    return deps


def parse_requirements(path: Path) -> Dict[str, str]:
    """Extract package name → pinned version from a requirements file."""
    pins: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-r"):
            continue
        # Strip inline comments
        stripped = stripped.split("#")[0].strip()
        match = re.match(r"([a-zA-Z0-9_-]+)==(.+)", stripped)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def check_pyproject_no_deps(path: Path) -> List[str]:
    """Return errors if pyproject.toml still declares dependencies."""
    errors: List[str] = []
    text = path.read_text()
    if re.search(r"^\[project\.dependencies\]", text, re.MULTILINE):
        errors.append("pyproject.toml contains [project.dependencies] — remove it")
    if re.search(r"^\[project\].*?^dependencies\s*=", text, re.MULTILINE | re.DOTALL):
        # Check for inline dependencies in [project] section
        in_project = False
        for line in text.splitlines():
            if line.strip() == "[project]":
                in_project = True
            elif line.strip().startswith("[") and line.strip() != "[project]":
                in_project = False
            elif in_project and line.strip().startswith("dependencies"):
                errors.append(
                    "pyproject.toml [project] section contains 'dependencies' — "
                    "remove it (deps are managed in pixi.toml)"
                )
                break
    if "[project.optional-dependencies]" in text:
        errors.append("pyproject.toml contains [project.optional-dependencies] — remove it")
    return errors


def check_dep_sync(repo_root: Optional[Path] = None) -> List[str]:
    """Run all dependency consistency checks.

    Returns:
        List of error messages (empty means all checks passed).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    errors: List[str] = []

    pixi_path = repo_root / "pixi.toml"
    pyproject_path = repo_root / "pyproject.toml"

    if not pixi_path.exists():
        errors.append("pixi.toml not found")
        return errors

    pixi_deps = parse_pixi_toml(pixi_path)
    # Normalize pixi package names to lowercase
    pixi_deps_lower = {k.lower(): v for k, v in pixi_deps.items()}

    # Check requirements files
    for req_file in ["requirements.txt", "requirements-dev.txt"]:
        req_path = repo_root / req_file
        if not req_path.exists():
            continue
        pins = parse_requirements(req_path)
        for pkg, pinned_ver in pins.items():
            if pkg not in pixi_deps_lower:
                errors.append(f"{req_file}: {pkg}=={pinned_ver} has no matching entry in pixi.toml")
                continue
            constraints = parse_pixi_constraints(pixi_deps_lower[pkg])
            if constraints:
                ver_tuple = parse_version(pinned_ver)
                if not version_satisfies(ver_tuple, constraints):
                    errors.append(
                        f"{req_file}: {pkg}=={pinned_ver} falls outside pixi.toml constraint {pixi_deps_lower[pkg]}"
                    )

    # Check pyproject.toml
    if pyproject_path.exists():
        errors.extend(check_pyproject_no_deps(pyproject_path))

    return errors


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detect).",
    )
    args = parser.parse_args()

    errors = check_dep_sync(args.repo_root)
    if errors:
        print("Dependency sync check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK: all dependency declarations are consistent")


if __name__ == "__main__":
    main()
