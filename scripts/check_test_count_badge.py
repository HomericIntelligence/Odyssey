#!/usr/bin/env python3
"""
Check Test Count Badge — Validate README.md test count badge against actual test files.

This script counts test_*.mojo files in the repository and compares the count
against the shields.io badge URL in README.md. It fails if the badge is more than
10% out of date, and can auto-update the badge with --fix.

Exit codes:
  0 - Badge is current (within tolerance)
  1 - Badge is stale or validation error

Usage:
    python scripts/check_test_count_badge.py [--fix] [--tolerance FLOAT]

Arguments:
    --fix           Auto-update the badge in README.md to the actual count
    --tolerance     Allowed fractional drift before failing (default: 0.10)
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from common import get_repo_root

# Directories to exclude from test file discovery (matches validate_test_coverage.py)
_EXCLUDE_DIRS = [".pixi/", "build/", "dist/", ".git/", "worktrees/"]

# Regex to extract the numeric count from a shields.io tests badge URL.
# Matches patterns like: tests-247%2B, tests-122%2B, tests-300
_BADGE_COUNT_RE = re.compile(r"tests-(\d[\d,]*?)(?:%2B|-brightgreen|\+|\.svg|-[a-z])")

# Pattern that identifies the tests badge line in README.md
_BADGE_LINE_RE = re.compile(r"(https://img\.shields\.io/badge/tests-)(\d[\d,]*)(%2B-brightgreen\.svg)")


def count_test_files(repo_root: Path) -> int:
    """Count test_*.mojo files, excluding build artifacts and known non-test directories.

    Uses subprocess find to mirror the approach in validate_test_coverage.py and
    handle large directory trees safely.  Exclusions are checked against the path
    *relative to repo_root* so that worktree paths (which contain 'worktrees/' in
    their absolute form) are not incorrectly filtered.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        Number of test_*.mojo files found.
    """
    cmd = ["find", str(repo_root), "-name", "test_*.mojo", "-type", "f"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"❌ find command failed: {result.stderr}", file=sys.stderr)
        return 0

    repo_prefix = str(repo_root) + "/"
    count = 0
    for path in result.stdout.splitlines():
        # Build the relative path for exclusion checks so that worktree repo
        # roots (e.g. .worktrees/issue-3307/) are not falsely excluded.
        rel = path[len(repo_prefix) :] if path.startswith(repo_prefix) else path
        if any(excl in rel for excl in _EXCLUDE_DIRS):
            continue
        count += 1

    return count


def parse_badge_count(readme_path: Path) -> Optional[int]:
    """Extract the test count from the shields.io badge in README.md.

    Args:
        readme_path: Path to README.md.

    Returns:
        The integer count embedded in the badge URL, or None if not found.
    """
    if not readme_path.exists():
        print(f"❌ README not found: {readme_path}", file=sys.stderr)
        return None

    content = readme_path.read_text(encoding="utf-8")
    match = _BADGE_COUNT_RE.search(content)
    if not match:
        print("❌ Could not find tests badge in README.md", file=sys.stderr)
        return None

    raw = match.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        print(f"❌ Could not parse badge count from '{raw}'", file=sys.stderr)
        return None


def check_badge_drift(actual: int, badge: int, tolerance: float = 0.10) -> bool:
    """Return True if the badge count is within tolerance of the actual count.

    Args:
        actual: Real number of test files found.
        badge: Count recorded in the README badge.
        tolerance: Maximum allowed fractional deviation (default 10%).

    Returns:
        True if within tolerance, False if the badge is too stale.
    """
    if actual == 0:
        return badge == 0
    drift = abs(actual - badge) / actual
    return drift <= tolerance


def update_badge(readme_path: Path, new_count: int) -> None:
    """Rewrite the tests badge URL in README.md with new_count.

    Args:
        readme_path: Path to README.md.
        new_count: The count to embed in the badge URL.
    """
    content = readme_path.read_text(encoding="utf-8")
    updated = _BADGE_LINE_RE.sub(
        rf"\g<1>{new_count}\g<3>",
        content,
    )
    readme_path.write_text(updated, encoding="utf-8")


def main() -> int:
    """Orchestrate badge count validation.

    Returns:
        0 if badge is current, 1 if stale or error.
    """
    args = sys.argv[1:]
    fix_mode = "--fix" in args

    tolerance = 0.10
    if "--tolerance" in args:
        idx = args.index("--tolerance")
        try:
            tolerance = float(args[idx + 1])
        except (IndexError, ValueError):
            print("❌ --tolerance requires a float argument", file=sys.stderr)
            return 1

    repo_root = get_repo_root()
    readme_path = repo_root / "README.md"

    actual = count_test_files(repo_root)
    badge = parse_badge_count(readme_path)

    if badge is None:
        return 1

    if check_badge_drift(actual, badge, tolerance):
        print(f"✅ Test count badge is current: {badge}+ (actual: {actual})")
        return 0

    print(
        f"❌ Test count badge is stale: badge={badge}+, actual={actual} "
        f"(drift={abs(actual - badge) / actual:.1%}, tolerance={tolerance:.0%})"
    )

    if fix_mode:
        update_badge(readme_path, actual)
        print(f"✅ Updated badge to {actual}+ in README.md")
        return 0

    print("   Run with --fix to auto-update the badge.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
