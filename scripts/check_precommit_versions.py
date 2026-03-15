#!/usr/bin/env python3
"""Check for version drift between .pre-commit-config.yaml rev: values and pixi.lock.

Parses rev: tags for known external repos in .pre-commit-config.yaml and
compares them against the authoritative versions resolved in pixi.lock.
Fails with exit code 1 if any mismatch is detected.

Exit codes:
  0 - No version drift detected (or tool not tracked in pixi.lock)
  1 - Version drift found or required files missing

Usage:
    python scripts/check_precommit_versions.py [--repo-root PATH]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Maps pre-commit repo URL substring → pixi.lock conda package name
REPO_TO_PACKAGE: Dict[str, str] = {
    "mirrors-mypy": "mypy",
    "kynan/nbstripout": "nbstripout",
}


def get_repo_root() -> Path:
    """Get repository root by searching upward for a .git directory.

    Returns:
        Path to repository root.

    Raises:
        RuntimeError: If no .git directory found.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / ".git").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError("Could not find repository root (.git directory not found)")


def parse_precommit_revs(config_path: Path) -> Dict[str, str]:
    """Return {repo_url: rev} for all external repos with a rev: tag.

    Uses regex to avoid a yaml dependency.

    Args:
        config_path: Path to .pre-commit-config.yaml.

    Returns:
        Mapping of repo URL to rev string.
    """
    text = config_path.read_text()

    # Match blocks like:
    #   - repo: https://github.com/foo/bar
    #     rev: v1.2.3
    repo_rev_pattern = re.compile(
        r"-\s+repo:\s+(\S+)\s+rev:\s+(\S+)",
        re.MULTILINE,
    )

    result: Dict[str, str] = {}
    for match in repo_rev_pattern.finditer(text):
        repo_url = match.group(1)
        rev = match.group(2)
        # Skip local repos
        if repo_url != "local":
            result[repo_url] = rev

    return result


def parse_lock_versions(lock_path: Path) -> Dict[str, str]:
    """Return {package_name: version} from pixi.lock conda entries.

    Extracts versions from conda URL lines like:
      - conda: https://.../mypy-1.19.1-py314h5bd0f2a_0.conda

    Args:
        lock_path: Path to pixi.lock.

    Returns:
        Mapping of package name to version string.
    """
    text = lock_path.read_text()

    # Match conda URL lines, extracting package name and version
    # Pattern: /packagename-VERSION-buildstring.conda (or .tar.bz2)
    conda_pattern = re.compile(
        r"-\s+conda:\s+https?://\S+/([a-zA-Z0-9_\-]+)-(\d+\.\d+[\.\d]*)-[^/\s]+\.(?:conda|tar\.bz2)"
    )

    result: Dict[str, str] = {}
    for match in conda_pattern.finditer(text):
        pkg_name = match.group(1)
        version = match.group(2)
        # Keep first occurrence (avoid duplicates from multiple platforms/builds)
        if pkg_name not in result:
            result[pkg_name] = version

    return result


def normalize_rev(rev: str) -> str:
    """Strip leading 'v' from rev tags (v1.19.1 → 1.19.1).

    Args:
        rev: Version string, possibly with a leading 'v'.

    Returns:
        Version string without leading 'v'.
    """
    return rev.lstrip("v")


def check_drift(
    revs: Dict[str, str],
    lock_versions: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Return list of (repo_url, rev_version, lock_version) mismatches.

    Only repos with entries in REPO_TO_PACKAGE are checked.
    Repos not tracked in pixi.lock are silently skipped.

    Args:
        revs: Mapping of repo URL to rev string from .pre-commit-config.yaml.
        lock_versions: Mapping of package name to version from pixi.lock.

    Returns:
        List of (repo_url, precommit_version, lock_version) tuples for mismatches.
    """
    mismatches: List[Tuple[str, str, str]] = []

    for repo_url, rev in revs.items():
        # Find the package name for this repo URL
        pkg_name = None
        for url_substring, package in REPO_TO_PACKAGE.items():
            if url_substring in repo_url:
                pkg_name = package
                break

        if pkg_name is None:
            # Not tracked in pixi.lock — skip
            continue

        lock_version = lock_versions.get(pkg_name)
        if lock_version is None:
            # Package not in pixi.lock — skip
            continue

        normalized_rev = normalize_rev(rev)
        if normalized_rev != lock_version:
            mismatches.append((repo_url, normalized_rev, lock_version))

    return mismatches


def main(argv: Optional[List[str]] = None) -> int:
    """Run version drift check.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        0 if no drift detected, 1 if drift found or files missing.
    """
    parser = argparse.ArgumentParser(
        description="Check for version drift between .pre-commit-config.yaml and pixi.lock"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (default: auto-detected from script location)",
    )
    args = parser.parse_args(argv)

    root = args.repo_root if args.repo_root is not None else get_repo_root()

    config_path = root / ".pre-commit-config.yaml"
    lock_path = root / "pixi.lock"

    if not config_path.exists():
        print(f"ERROR: .pre-commit-config.yaml not found at {config_path}", file=sys.stderr)
        return 1

    if not lock_path.exists():
        print(f"ERROR: pixi.lock not found at {lock_path}", file=sys.stderr)
        return 1

    revs = parse_precommit_revs(config_path)
    lock_versions = parse_lock_versions(lock_path)

    mismatches = check_drift(revs, lock_versions)

    if mismatches:
        print("ERROR: Version drift detected between .pre-commit-config.yaml and pixi.lock:")
        for repo_url, precommit_ver, lock_ver in mismatches:
            print(f"  {repo_url}")
            print(f"    pre-commit rev: {precommit_ver}")
            print(f"    pixi.lock:      {lock_ver}")
        print()
        print("Fix: Update the rev: tag in .pre-commit-config.yaml to match pixi.lock")
        return 1

    # Report which tools were checked
    checked: List[str] = []
    for repo_url, _rev in revs.items():
        for url_substring in REPO_TO_PACKAGE:
            if url_substring in repo_url:
                checked.append(url_substring)
                break

    if checked:
        print(f"OK: No version drift detected (checked: {', '.join(checked)})")
    else:
        print("OK: No tracked repos found in .pre-commit-config.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
