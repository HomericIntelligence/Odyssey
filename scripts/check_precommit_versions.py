#!/usr/bin/env python3
"""
Check pre-commit version consistency against pixi.toml

Validates that external pre-commit hook revs in .pre-commit-config.yaml
match (or are compatible with) the corresponding pixi.toml package versions.
This prevents the version-drift issue described in #3369 / #4030.

Usage:
    python scripts/check_precommit_versions.py [--config PATH] [--pixi PATH]

Exit codes:
    0: All versions are consistent
    1: Version drift detected (or file not found)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from common import get_repo_root

# Mapping from external pre-commit repo URL to pixi.toml package name.
#
# Only repos that have a pixi/conda-forge counterpart with matching versioning
# are listed here.  markdownlint-cli2 is intentionally excluded: it is a
# Node.js tool whose conda-forge package uses a different version series than
# the npm package referenced by the pre-commit rev, so version comparison
# would produce false positives.  It is still covered by the pre-commit hook
# itself and its npm version is pinned in .pre-commit-config.yaml.
HOOK_TO_PIXI_MAP: Dict[str, str] = {
    "https://github.com/pre-commit/mirrors-mypy": "mypy",
    "https://github.com/kynan/nbstripout": "nbstripout",
    "https://github.com/pre-commit/pre-commit-hooks": "pre-commit-hooks",
}


def normalize_version(rev: str) -> str:
    """
    Strip a leading 'v' from a git tag so it can be compared numerically.

    Args:
        rev: A git tag such as "v1.19.1" or "0.7.1"

    Returns:
        Version string without leading 'v', e.g. "1.19.1"
    """
    return rev.lstrip("v")


def load_precommit_config(config_path: Path) -> List[Dict]:
    """
    Parse .pre-commit-config.yaml and return the list of repo entries.

    Args:
        config_path: Path to .pre-commit-config.yaml

    Returns:
        List of repo dicts from the ``repos`` key

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the YAML is missing the ``repos`` key
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Pre-commit config not found: {config_path}")

    with config_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "repos" not in data:
        raise ValueError(f"No 'repos' key in {config_path}")

    return data["repos"]  # type: ignore[return-value]


def extract_external_hooks(repos: List[Dict]) -> Dict[str, str]:
    """
    Extract external (non-local) repo URLs and their ``rev`` values.

    Args:
        repos: List of repo dicts from .pre-commit-config.yaml

    Returns:
        Dict mapping repo URL → rev string, e.g.
        {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
    """
    result: Dict[str, str] = {}
    for repo in repos:
        url = repo.get("repo", "")
        rev = repo.get("rev", "")
        if url and url != "local" and rev:
            result[url] = rev
    return result


def parse_pixi_constraint(constraint: str) -> Optional[str]:
    """
    Extract the lower-bound version from a pixi/conda version constraint.

    Handles patterns like:
    - ``>=1.19.1,<2``  → ``"1.19.1"``
    - ``==0.26.1``     → ``"0.26.1"``
    - ``>=0.7.1``      → ``"0.7.1"``
    - ``0.12.1``       → ``"0.12.1"`` (bare version)

    Args:
        constraint: A pixi version constraint string

    Returns:
        Lower-bound version string, or None if unparseable
    """
    # Try >=X.Y.Z or ==X.Y.Z
    match = re.search(r"[><=]=?\s*(\d+\.\d+[\.\d]*)", constraint)
    if match:
        return match.group(1)
    # Bare version string
    bare = re.match(r"^(\d+\.\d+[\.\d]*)$", constraint.strip())
    if bare:
        return bare.group(1)
    return None


def load_pixi_versions(pixi_path: Path) -> Dict[str, str]:
    """
    Parse pixi.toml and return a dict mapping package name → lower-bound version.

    Args:
        pixi_path: Path to pixi.toml

    Returns:
        Dict mapping package name (lower-case) → lower-bound version string

    Raises:
        FileNotFoundError: if the file does not exist
    """
    if not pixi_path.exists():
        raise FileNotFoundError(f"pixi.toml not found: {pixi_path}")

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Fallback: manual TOML parsing for [dependencies] section only
            return _parse_pixi_dependencies_fallback(pixi_path)

    with pixi_path.open("rb") as fh:
        data = tomllib.load(fh)

    deps: Dict[str, str] = {}
    # Collect from [dependencies] (production deps)
    for pkg, constraint in data.get("dependencies", {}).items():
        version = parse_pixi_constraint(str(constraint))
        if version:
            deps[pkg.lower()] = version
    # Also collect from [feature.*.dependencies] (e.g. [feature.dev.dependencies])
    for feature_data in data.get("feature", {}).values():
        for pkg, constraint in feature_data.get("dependencies", {}).items():
            version = parse_pixi_constraint(str(constraint))
            if version:
                deps[pkg.lower()] = version
    return deps


def _parse_pixi_dependencies_fallback(pixi_path: Path) -> Dict[str, str]:
    """
    Minimal TOML parser for dependency sections (no tomllib/tomli).

    Parses both [dependencies] and [feature.*.dependencies] sections so that
    packages moved to feature-specific sections (e.g. [feature.dev.dependencies])
    are still detected by the version-drift checker.

    Args:
        pixi_path: Path to pixi.toml

    Returns:
        Dict mapping package name → lower-bound version string
    """
    deps: Dict[str, str] = {}
    in_deps = False
    for line in pixi_path.read_text().splitlines():
        stripped = line.strip()
        # Match [dependencies] or [feature.*.dependencies]
        if stripped == "[dependencies]" or re.match(r"^\[feature\.[^]]+\.dependencies\]$", stripped):
            in_deps = True
            continue
        if stripped.startswith("["):
            in_deps = False
            continue
        if in_deps and "=" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition("=")
            pkg = key.strip().lower()
            constraint = value.strip().strip('"')
            version = parse_pixi_constraint(constraint)
            if version:
                deps[pkg] = version
    return deps


def version_tuple(version: str) -> Tuple[int, ...]:
    """
    Convert a version string to a comparable tuple of integers.

    Args:
        version: Version string, e.g. "1.19.1"

    Returns:
        Tuple of ints, e.g. (1, 19, 1)
    """
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_version_drift(
    external_hooks: Dict[str, str],
    pixi_versions: Dict[str, str],
) -> List[str]:
    """
    Compare external hook revs against pixi.toml lower-bound versions.

    A drift is reported when the hook rev (normalized) does not match the
    pixi lower-bound exactly.  An exact match is required so that the two
    files always point to the same release.

    Args:
        external_hooks: Dict mapping repo URL → rev (from .pre-commit-config.yaml)
        pixi_versions: Dict mapping package name → lower-bound version (from pixi.toml)

    Returns:
        List of human-readable drift messages (empty if everything is consistent)
    """
    issues: List[str] = []
    for repo_url, rev in external_hooks.items():
        pkg_name = HOOK_TO_PIXI_MAP.get(repo_url)
        if pkg_name is None:
            # Not in map — not tracked, skip
            continue
        pixi_version = pixi_versions.get(pkg_name.lower())
        if pixi_version is None:
            issues.append(
                f"MISSING: '{pkg_name}' is used in .pre-commit-config.yaml "
                f"(rev={rev!r}) but has no entry in pixi.toml. "
                f"Add '{pkg_name} = \">={normalize_version(rev)}\"' to pixi.toml."
            )
            continue
        hook_version = normalize_version(rev)
        if version_tuple(hook_version) != version_tuple(pixi_version):
            issues.append(
                f"DRIFT: '{pkg_name}' — .pre-commit-config.yaml rev is "
                f"{hook_version!r} but pixi.toml lower bound is {pixi_version!r}. "
                f"They must match."
            )
    return issues


def check_version_consistency(
    precommit_path: Optional[Path] = None,
    pixi_path: Optional[Path] = None,
) -> List[str]:
    """
    Top-level check: load both config files and return a list of drift issues.

    Args:
        precommit_path: Path to .pre-commit-config.yaml (defaults to repo root)
        pixi_path: Path to pixi.toml (defaults to repo root)

    Returns:
        List of drift/missing messages (empty means consistent)
    """
    root = get_repo_root()
    if precommit_path is None:
        precommit_path = root / ".pre-commit-config.yaml"
    if pixi_path is None:
        pixi_path = root / "pixi.toml"

    repos = load_precommit_config(precommit_path)
    external_hooks = extract_external_hooks(repos)
    pixi_versions = load_pixi_versions(pixi_path)
    return check_version_drift(external_hooks, pixi_versions)


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code: 0 for success, 1 for drift detected or error
    """
    parser = argparse.ArgumentParser(description="Check .pre-commit-config.yaml revs match pixi.toml versions")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to .pre-commit-config.yaml (default: repo root)",
    )
    parser.add_argument(
        "--pixi",
        type=Path,
        default=None,
        help="Path to pixi.toml (default: repo root)",
    )
    args = parser.parse_args(argv)

    try:
        issues = check_version_consistency(
            precommit_path=args.config,
            pixi_path=args.pixi,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if issues:
        print("Pre-commit version drift detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix: update the rev in .pre-commit-config.yaml or the version constraint in pixi.toml so they match.")
        return 1

    print("OK: all pre-commit hook versions are consistent with pixi.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
