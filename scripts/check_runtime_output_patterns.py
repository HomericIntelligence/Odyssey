#!/usr/bin/env python3
"""Check for misleading runtime output patterns in Mojo source files.

Scans .mojo and .🔥 files for print() calls containing patterns that mislead
users: WARNING:, HACK:, XXX:, or 'Not implemented' placeholder messages.
These patterns emerged from the #3084/#3194 audit series and this script
enforces the cleanup for the broader pattern set (issue #3704).

Usage:
    python scripts/check_runtime_output_patterns.py [directory]
    python scripts/check_runtime_output_patterns.py examples/
    python scripts/check_runtime_output_patterns.py  # defaults to repo root, checks source dirs
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Directories to exclude from scanning
EXCLUDED_DIRS = {".worktrees", ".pixi", "build", ".git", "__pycache__", ".mypy_cache"}

# Source directories to check when scanning from repo root
SOURCE_DIRS = ["benchmarks", "examples", "papers", "scripts", "shared", "tests"]

# Patterns that match misleading runtime output in print() calls.
# Matches lines that call print( and contain one of the banned prefixes.
# Only matches actual print calls, not comment lines (those start with #).
BANNED_PATTERNS = [
    re.compile(r'print\([^)]*WARNING\s*:', re.IGNORECASE),
    re.compile(r'print\([^)]*HACK\s*:', re.IGNORECASE),
    re.compile(r'print\([^)]*XXX\s*:', re.IGNORECASE),
    re.compile(r'print\([^)]*Not\s+implemented', re.IGNORECASE),
]


def is_excluded(path: Path) -> bool:
    """Check if a path is inside an excluded directory.

    Args:
        path: Path to check.

    Returns:
        True if the path is inside an excluded directory.
    """
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    return False


def is_comment_line(line: str) -> bool:
    """Check if a line is a comment (starts with # after optional whitespace).

    Args:
        line: Line of source code to check.

    Returns:
        True if the line is a comment line.
    """
    return line.lstrip().startswith("#")


def find_violations(search_path: Path) -> List[Tuple[Path, int, str]]:
    """Find all misleading runtime output violations in .mojo and .🔥 files.

    Args:
        search_path: Directory to scan recursively.

    Returns:
        List of (file_path, line_number, line_content) tuples for each violation.
    """
    violations: List[Tuple[Path, int, str]] = []

    for pattern in ["*.mojo", "*.🔥"]:
        for mojo_file in sorted(search_path.rglob(pattern)):
            if is_excluded(mojo_file):
                continue
            try:
                lines = mojo_file.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue

            for line_num, line in enumerate(lines, start=1):
                if is_comment_line(line):
                    continue
                for banned in BANNED_PATTERNS:
                    if banned.search(line):
                        violations.append((mojo_file, line_num, line.rstrip()))
                        break  # Only report each line once

    return violations


def scan_source_dirs(repo_root: Path) -> List[Tuple[Path, int, str]]:
    """Scan only the standard source directories from repo root.

    Args:
        repo_root: Repository root directory.

    Returns:
        List of (file_path, line_number, line_content) tuples for each violation.
    """
    violations: List[Tuple[Path, int, str]] = []
    for source_dir in SOURCE_DIRS:
        dir_path = repo_root / source_dir
        if dir_path.exists():
            violations.extend(find_violations(dir_path))
    return violations


def get_repo_root() -> Path:
    """Get the repository root by searching upward for .git directory.

    Returns:
        Path to repository root.

    Raises:
        RuntimeError: If repository root cannot be found.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory found)")


def main() -> int:
    """Main entry point. Returns exit code (0 = clean, 1 = violations found).

    Returns:
        0 if no violations found, 1 if violations exist.
    """
    parser = argparse.ArgumentParser(
        description="Check for misleading runtime output patterns in Mojo files",
        epilog="Banned patterns: WARNING:, HACK:, XXX:, Not implemented in print() calls",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Directory to scan (default: source dirs in repo root)",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.directory is not None:
        search_path = Path(args.directory).resolve()
        if not search_path.exists():
            print(f"Error: directory not found: {search_path}", file=sys.stderr)
            return 1
        violations = find_violations(search_path)
    else:
        violations = scan_source_dirs(repo_root)

    if not violations:
        return 0

    for file_path, line_num, line_content in violations:
        # Show relative path from repo root
        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
            rel_path = file_path
        print(f"{rel_path}:{line_num}: {line_content}")

    print(
        f"\n{len(violations)} violation(s) found. Remove WARNING:, HACK:, XXX:, or "
        f"'Not implemented' prefixes from print() calls.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
