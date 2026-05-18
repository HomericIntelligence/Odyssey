#!/usr/bin/env python3
"""Check NOTE format compliance in Mojo source files.

Scans .mojo and .🔥 files for # NOTE patterns that do not follow the required
# NOTE (Mojo vX.Y.Z): format, as enforced by issue #3285.

The compliant format requires a parenthesized version annotation:
    # NOTE (Mojo v0.26.1): explanation
Non-compliant patterns like # NOTE: or # NOTE some text will be flagged.

Usage:
    python scripts/check_note_format.py [directory]
    python scripts/check_note_format.py src/projectodyssey/
    python scripts/check_note_format.py  # defaults to repo root, checks source dirs
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Directories to exclude from scanning
EXCLUDED_DIRS = {".worktrees", ".pixi", "build", ".git", "__pycache__", ".mypy_cache"}

# Source directories to check when scanning from repo root
SOURCE_DIRS = ["benchmarks", "examples", "papers", "scripts", "src/projectodyssey", "tests"]

# Pattern that matches non-compliant NOTE comments.
# Compliant format requires # NOTE followed by optional space then '('.
# Uses negative lookahead to avoid flagging '# NOTE (' or '# NOTE(' patterns.
NOTE_VIOLATION_PATTERN = re.compile(r"# NOTE(?!\s*\()")


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


def find_violations(search_path: Path) -> List[Tuple[Path, int, str]]:
    """Find all NOTE format violations in .mojo and .🔥 files under search_path.

    Args:
        search_path: Directory to scan recursively.

    Returns:
        List of (file_path, line_number, line_content) tuples for each violation.
    """
    violations: List[Tuple[Path, int, str]] = []

    # Scan for both .mojo and .🔥 files
    for mojo_file in sorted(search_path.rglob("*.mojo")):
        if is_excluded(mojo_file):
            continue
        try:
            lines = mojo_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(lines, start=1):
            if NOTE_VIOLATION_PATTERN.search(line):
                violations.append((mojo_file, line_num, line.rstrip()))

    for mojo_file in sorted(search_path.rglob("*.🔥")):
        if is_excluded(mojo_file):
            continue
        try:
            lines = mojo_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(lines, start=1):
            if NOTE_VIOLATION_PATTERN.search(line):
                violations.append((mojo_file, line_num, line.rstrip()))

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
        description="Check NOTE format compliance in Mojo files",
        epilog="Compliant format: # NOTE (Mojo vX.Y.Z): explanation",
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
        f"\n{len(violations)} violation(s) found. Use '# NOTE (Mojo vX.Y.Z): ...' format.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
