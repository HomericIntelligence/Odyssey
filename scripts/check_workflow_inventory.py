#!/usr/bin/env python3
"""
Check workflow inventory drift between .github/workflows/*.yml files and README.md table.

Compares the set of .yml files on disk against filenames documented in the
workflow table in .github/workflows/README.md.  Fails if any file exists on
disk but is not documented, or if a documented filename has no corresponding
file on disk.

Exit codes:
  0 - No drift detected
  1 - Drift found (undocumented files or documented-but-missing files)

Usage:
    python3 scripts/check_workflow_inventory.py [--repo-root PATH]
"""

import argparse
import re
import sys
from pathlib import Path

# Add scripts directory to path for common.py import
sys.path.insert(0, str(Path(__file__).parent))
from common import get_repo_root

# Matches a .yml filename (with or without a markdown hyperlink) inside a
# pipe-delimited table cell.  Examples matched:
#   | validate-workflows.yml |
#   | [comprehensive-tests.yml](#comprehensive-tests) |
_TABLE_FILENAME_RE = re.compile(r"\|\s*\[?([a-zA-Z0-9_.-]+\.yml)\]?[^|]*\|")


def collect_yml_files(repo_root: Path) -> set[str]:
    """Return basenames of *.yml files in .github/workflows/, excluding worktrees.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        Set of .yml basenames (e.g. {"ci.yml", "release.yml"}).
    """
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return set()

    result: set[str] = set()
    for path in workflows_dir.glob("*.yml"):
        # Exclude files that live inside a worktree subdirectory.
        # Relative paths are used so that the worktrees/ prefix is detected
        # regardless of where the repo is checked out.
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        if any(part == "worktrees" for part in rel.parts):
            continue
        result.add(path.name)
    return result


def parse_readme_table(readme_path: Path) -> set[str]:
    """Parse .github/workflows/README.md and return documented .yml filenames.

    Only lines that contain a pipe-delimited table cell with a .yml filename
    are considered.  Both plain (``validate-workflows.yml``) and hyperlinked
    (``[comprehensive-tests.yml](#anchor)``) forms are matched.

    Args:
        readme_path: Path to the README.md file to parse.

    Returns:
        Set of documented .yml basenames.
    """
    if not readme_path.is_file():
        return set()

    content = readme_path.read_text(encoding="utf-8")
    found: set[str] = set()
    for line in content.splitlines():
        for match in _TABLE_FILENAME_RE.finditer(line):
            found.add(match.group(1))
    return found


def check_inventory(repo_root: Path) -> tuple[list[str], list[str]]:
    """Compare on-disk .yml files against the README table.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        A tuple (undocumented, missing_files) where:
          - undocumented: filenames present on disk but absent from README table
          - missing_files: filenames in README table but absent from disk
    """
    readme_path = repo_root / ".github" / "workflows" / "README.md"
    on_disk = collect_yml_files(repo_root)
    in_readme = parse_readme_table(readme_path)

    undocumented = sorted(on_disk - in_readme)
    missing_files = sorted(in_readme - on_disk)
    return undocumented, missing_files


def main() -> int:
    """Entry point.  Returns 0 on success, 1 on drift detected."""
    parser = argparse.ArgumentParser(
        description="Detect drift between .github/workflows/*.yml files and README.md table."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to repository root (default: auto-detected from script location)",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root if args.repo_root is not None else get_repo_root()

    undocumented, missing_files = check_inventory(repo_root)

    if not undocumented and not missing_files:
        print("✅ Workflow inventory is in sync.")
        return 0

    print("❌ Workflow inventory drift detected!\n")

    if undocumented:
        print("Files on disk but NOT documented in .github/workflows/README.md:")
        for name in undocumented:
            print(f"  + {name}")
        print()

    if missing_files:
        print("Files documented in README.md table but NOT present on disk:")
        for name in missing_files:
            print(f"  - {name}")
        print()

    print(
        "Fix: update the Workflow Summary table in .github/workflows/README.md "
        "so it exactly matches the *.yml files on disk."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
