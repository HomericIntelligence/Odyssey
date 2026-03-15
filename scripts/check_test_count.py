#!/usr/bin/env python3
"""
Check Mojo Test Count — Enforce ≤10 tests per file per ADR-009.

This script counts `fn test_` occurrences per staged Mojo test file and
fails (exit 1) if any file exceeds the ADR-009 heap-corruption threshold of
10 tests. Intended for use as a pre-commit hook with pass_filenames: true.

Exit codes:
  0 - All files within the 10-test limit
  1 - One or more files exceed the limit

Usage:
    python scripts/check_test_count.py [file1.mojo file2.mojo ...]

Reference:
    docs/adr/ADR-009-heap-corruption-workaround.md § Phase 2
"""

import re
import sys
from pathlib import Path
from typing import List

# Crash threshold is ~15 tests; ADR-009 sets limit to 10 for safety margin
LIMIT = 10

# Anchored to start of line to avoid false positives in strings/docstrings
_TEST_FN_RE = re.compile(r"^\s*fn test_", re.MULTILINE)


def is_mojo_test_file(path: Path) -> bool:
    """Return True if path is a Mojo test file under the tests/ directory.

    Args:
        path: File path to check.

    Returns:
        True if the file ends with .mojo and lives under a tests/ directory.
    """
    if path.suffix != ".mojo":
        return False
    # Accept both absolute paths containing /tests/ and relative paths
    return "tests" in path.parts or "/tests/" in str(path)


def count_tests_in_file(file_path: Path) -> int:
    """Count `fn test_` definitions in a Mojo source file.

    Args:
        file_path: Path to the Mojo file.

    Returns:
        Number of test functions found; 0 if the file is unreadable.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"⚠️  Warning: cannot read {file_path}: {exc}", file=sys.stderr)
        return 0
    return len(_TEST_FN_RE.findall(content))


def check_files(file_paths: List[str]) -> int:
    """Check each Mojo test file against the ADR-009 test-count limit.

    Args:
        file_paths: List of file path strings (typically from sys.argv[1:]).

    Returns:
        0 if all files are within the limit, 1 if any violation is found.
    """
    violations: List[str] = []
    checked = 0

    for raw in file_paths:
        path = Path(raw)
        if not is_mojo_test_file(path):
            continue
        checked += 1
        count = count_tests_in_file(path)
        if count > LIMIT:
            violations.append(
                f"❌  {path}: {count} tests found (limit: {LIMIT}) — split per ADR-009"
            )

    if violations:
        for msg in violations:
            print(msg)
        print(
            f"\nSee docs/adr/ADR-009-heap-corruption-workaround.md for guidance on splitting test files."
        )
        return 1

    print(f"✅  All {checked} test file(s) within the {LIMIT}-test limit.")
    return 0


def main() -> int:
    """Entry point: read file paths from argv and run the check.

    Returns:
        Exit code (0 = pass, 1 = violation).
    """
    return check_files(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
