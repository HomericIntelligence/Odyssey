#!/usr/bin/env python3
"""Synchronize requirements*.txt from pixi.toml resolved versions.

Reads pixi.lock (via ``pixi list --json``) and regenerates
``requirements.txt`` and ``requirements-dev.txt`` with exact pins
matching the pixi-resolved versions.

Usage:
    python scripts/sync_requirements.py
    python scripts/sync_requirements.py --check  # verify files are up-to-date
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Header written at the top of every generated requirements file.
GENERATED_HEADER = """\
# AUTO-GENERATED from pixi.toml — do not edit manually.
# Regenerate with: python scripts/sync_requirements.py
# These files exist for pip-only contexts (Docker, CI fallback).
"""

# Packages that belong in requirements.txt (core runtime/testing).
CORE_PACKAGES: List[str] = [
    "pytest",
    "pytest-cov",
    "pytest-timeout",
    "pytest-xdist",
    "ruff",
    "mypy",
    "jinja2",
    "pyyaml",
    "click",
]

# Additional packages in requirements-dev.txt (dev-only tooling).
DEV_PACKAGES: List[str] = [
    "pre-commit",
    "safety",
    "bandit",
    "mkdocs",
    "mkdocs-material",
    "pytest-benchmark",
]


def get_pixi_packages() -> Dict[str, str]:
    """Return a mapping of {package_name: version} from ``pixi list --json``."""
    result = subprocess.run(
        ["pixi", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running pixi list --json: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(result.stdout)
    return {pkg["name"]: pkg["version"] for pkg in data}


def generate_requirements(
    packages: List[str],
    resolved: Dict[str, str],
    *,
    include_base: Optional[str] = None,
    section_comments: Optional[Dict[str, str]] = None,
) -> str:
    """Build a requirements file string with exact pins from resolved versions.

    Args:
        packages: Package names to include.
        resolved: Mapping of package name to resolved version.
        include_base: Optional ``-r <file>`` line to include at the top.
        section_comments: Optional mapping of package name to inline comment.

    Returns:
        The full file content as a string.
    """
    if section_comments is None:
        section_comments = {}

    lines: List[str] = [GENERATED_HEADER]
    if include_base:
        lines.append(include_base)
        lines.append("")

    for pkg in packages:
        version = resolved.get(pkg)
        if version is None:
            print(
                f"Warning: {pkg} not found in pixi environment, skipping",
                file=sys.stderr,
            )
            continue
        comment = section_comments.get(pkg, "")
        suffix = f"  {comment}" if comment else ""
        lines.append(f"{pkg}=={version}{suffix}")

    # Ensure trailing newline
    lines.append("")
    return "\n".join(lines)


def write_requirements(repo_root: Path, resolved: Dict[str, str]) -> List[Path]:
    """Write requirements.txt and requirements-dev.txt under *repo_root*.

    Returns:
        List of paths written.
    """
    core_comments: Dict[str, str] = {
        "jinja2": "# Template engine (paper scaffolding, code generation)",
        "pyyaml": "# YAML parsing (configuration)",
        "click": "# CLI framework (command-line tools)",
    }
    dev_comments: Dict[str, str] = {
        "pre-commit": "# Pre-commit hooks",
        "safety": "# Security scanning",
        "bandit": "# Security scanning",
        "mkdocs": "# Documentation",
        "mkdocs-material": "# Documentation",
        "pytest-benchmark": "# Benchmarking",
    }

    req_txt = generate_requirements(
        CORE_PACKAGES,
        resolved,
        section_comments=core_comments,
    )
    req_dev_txt = generate_requirements(
        DEV_PACKAGES,
        resolved,
        include_base="-r requirements.txt",
        section_comments=dev_comments,
    )

    paths: List[Path] = []
    for name, content in [
        ("requirements.txt", req_txt),
        ("requirements-dev.txt", req_dev_txt),
    ]:
        path = repo_root / name
        path.write_text(content)
        paths.append(path)
        print(f"Wrote {path}")
    return paths


def check_requirements(repo_root: Path, resolved: Dict[str, str]) -> bool:
    """Return True if existing requirements files match what would be generated."""
    core_comments: Dict[str, str] = {
        "jinja2": "# Template engine (paper scaffolding, code generation)",
        "pyyaml": "# YAML parsing (configuration)",
        "click": "# CLI framework (command-line tools)",
    }
    dev_comments: Dict[str, str] = {
        "pre-commit": "# Pre-commit hooks",
        "safety": "# Security scanning",
        "bandit": "# Security scanning",
        "mkdocs": "# Documentation",
        "mkdocs-material": "# Documentation",
        "pytest-benchmark": "# Benchmarking",
    }

    expected = {
        "requirements.txt": generate_requirements(
            CORE_PACKAGES,
            resolved,
            section_comments=core_comments,
        ),
        "requirements-dev.txt": generate_requirements(
            DEV_PACKAGES,
            resolved,
            include_base="-r requirements.txt",
            section_comments=dev_comments,
        ),
    }

    ok = True
    for name, expected_content in expected.items():
        path = repo_root / name
        if not path.exists():
            print(f"FAIL: {name} does not exist", file=sys.stderr)
            ok = False
            continue
        actual = path.read_text()
        if actual != expected_content:
            print(
                f"FAIL: {name} is out of date — regenerate with: python scripts/sync_requirements.py",
                file=sys.stderr,
            )
            ok = False
    return ok


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify requirements files are up-to-date (exit 1 if not).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detect).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    resolved = get_pixi_packages()

    if args.check:
        if check_requirements(repo_root, resolved):
            print("OK: requirements files are up-to-date")
        else:
            sys.exit(1)
    else:
        write_requirements(repo_root, resolved)


if __name__ == "__main__":
    main()
