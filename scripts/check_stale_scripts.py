#!/usr/bin/env python3
"""Detect scripts/*.py files with no references in .github/, justfile, or other scripts/.

A script is considered "possibly stale" if its basename does not appear in any of:
  - .github/**/*.yml
  - justfile
  - .pre-commit-config.yaml
  - other scripts/*.py files (excluding the script itself)

This check always exits 0 (warning only, not a hard failure) to surface candidates
without blocking commits. It follows up on manual audit rounds in #3148 and #3337.

Usage:
    python scripts/check_stale_scripts.py [--repo-root PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set

# Scripts that are imported by other scripts (not invoked directly).
# These will never be flagged as stale even if not mentioned in CI/justfile.
ALWAYS_ACTIVE: Set[str] = {"common.py", "check_stale_scripts.py"}


def get_all_scripts(scripts_dir: Path) -> List[str]:
    """Return basenames of all *.py files in scripts_dir.

    Args:
        scripts_dir: Path to the scripts/ directory.

    Returns:
        Sorted list of basenames (e.g. ["audit_shared_links.py", ...]).
    """
    return sorted(p.name for p in scripts_dir.glob("*.py") if p.is_file())


def get_reference_targets(repo_root: Path) -> List[Path]:
    """Collect files that may reference script names.

    Includes:
      - .github/**/*.yml
      - justfile
      - .pre-commit-config.yaml
      - scripts/*.py (all scripts, so cross-references are detected)

    Args:
        repo_root: Root of the repository.

    Returns:
        List of Path objects for files to search.
    """
    targets: List[Path] = []

    # GitHub Actions workflows
    github_dir = repo_root / ".github"
    if github_dir.is_dir():
        targets.extend(github_dir.rglob("*.yml"))

    # justfile
    justfile = repo_root / "justfile"
    if justfile.is_file():
        targets.append(justfile)

    # pre-commit config
    precommit = repo_root / ".pre-commit-config.yaml"
    if precommit.is_file():
        targets.append(precommit)

    # All scripts (for cross-references between scripts)
    scripts_dir = repo_root / "scripts"
    if scripts_dir.is_dir():
        targets.extend(scripts_dir.glob("*.py"))

    return targets


def find_references(script_name: str, targets: List[Path], scripts_dir: Path) -> bool:
    """Return True if script_name appears in any target file outside its own source.

    A self-reference (script_name appearing inside its own file) does not count.

    Args:
        script_name: Basename of the script to search for (e.g. "audit_shared_links.py").
        targets: List of files to search through.
        scripts_dir: Path to the scripts/ directory, used to identify the script's own file.

    Returns:
        True if at least one external reference is found.
    """
    own_path = scripts_dir / script_name
    for target in targets:
        if target.resolve() == own_path.resolve():
            continue
        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if script_name in content:
            return True
    return False


def find_stale_candidates(repo_root: Path) -> List[str]:
    """Return basenames of scripts with no external references.

    Scripts in ALWAYS_ACTIVE are excluded from consideration.

    Args:
        repo_root: Root of the repository.

    Returns:
        Sorted list of possibly-stale script basenames.
    """
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return []

    all_scripts = get_all_scripts(scripts_dir)
    targets = get_reference_targets(repo_root)

    stale: List[str] = []
    for script_name in all_scripts:
        if script_name in ALWAYS_ACTIVE:
            continue
        if not find_references(script_name, targets, scripts_dir):
            stale.append(script_name)

    return stale


def main(argv: List[str] | None = None) -> int:
    """Run stale-script detection and print warnings.

    Always returns 0 (warning only, never a hard failure).

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detected from this script's location)",
    )
    args = parser.parse_args(argv)

    if args.repo_root is not None:
        repo_root = args.repo_root
    else:
        # Auto-detect: scripts/ is one level below repo root
        repo_root = Path(__file__).resolve().parent.parent

    candidates = find_stale_candidates(repo_root)
    for candidate in candidates:
        print(f"WARNING: possibly stale: scripts/{candidate}")

    if candidates:
        print(f"\n{len(candidates)} possibly stale script(s) found (warnings only, not a failure).")
    else:
        print("No stale script candidates found.")

    return 0  # always exit 0 — warning, not hard failure


if __name__ == "__main__":
    sys.exit(main())
