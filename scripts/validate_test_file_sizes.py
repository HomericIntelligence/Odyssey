#!/usr/bin/env python3
"""Validate test file sizes to avoid heap corruption bug (#2942).

The Mojo 0.26.1 runtime has a bug causing heap corruption after running
~15 cumulative tests in a single file. This script ensures no test file
exceeds the safe threshold of 10 tests per file.

Usage:
    python scripts/validate_test_file_sizes.py [--threshold N] [--verbose]

See Also:
    - Issue #2942: Heap corruption bug report
    - ADR-009: Heap corruption workaround documentation
"""

import argparse
import re
import sys
from pathlib import Path

# Safe threshold below crash point (~15 tests)
DEFAULT_MAX_TESTS_PER_FILE = 10

# Pattern to match test function definitions
TEST_FUNCTION_PATTERN = re.compile(r"^fn\s+test_\w+\s*\(", re.MULTILINE)


def count_tests_in_file(filepath: Path) -> int:
    """Count the number of test functions in a Mojo file.

    Args:
        filepath: Path to the Mojo test file.

    Returns:
        Number of test functions found (functions starting with 'test_').
    """
    try:
        content = filepath.read_text(encoding="utf-8")
        matches = TEST_FUNCTION_PATTERN.findall(content)
        return len(matches)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return 0


def validate_test_files(
    test_dir: Path, threshold: int = DEFAULT_MAX_TESTS_PER_FILE, verbose: bool = False
) -> tuple[bool, list[tuple[Path, int]]]:
    """Validate that all test files have fewer tests than the threshold.

    Args:
        test_dir: Directory containing test files.
        threshold: Maximum allowed tests per file.
        verbose: If True, print all files checked.

    Returns:
        Tuple of (all_passed, list of (file, count) for violations).
    """
    violations = []
    all_files = []

    # Find all Mojo test files
    for test_file in test_dir.rglob("test_*.mojo"):
        # Skip deprecated files
        if ".DEPRECATED" in test_file.name:
            continue

        test_count = count_tests_in_file(test_file)
        all_files.append((test_file, test_count))

        if test_count > threshold:
            violations.append((test_file, test_count))

    if verbose:
        print(f"Checked {len(all_files)} test files (threshold: {threshold} tests)")
        for filepath, count in sorted(all_files, key=lambda x: -x[1]):
            status = "FAIL" if count > threshold else "OK"
            print(f"  [{status}] {filepath.relative_to(test_dir.parent)}: {count} tests")

    return len(violations) == 0, violations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate test file sizes to avoid heap corruption bug (#2942)",
        epilog="See Issue #2942 and ADR-009 for details on the Mojo runtime bug.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_MAX_TESTS_PER_FILE,
        help=f"Maximum tests per file (default: {DEFAULT_MAX_TESTS_PER_FILE})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all files checked")
    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        help="Test files to check (if not provided, scans tests/ directory)",
    )
    args = parser.parse_args()

    # Determine which files to check
    if args.files:
        # Files passed as arguments (from pre-commit hook)
        test_files = [f for f in args.files if f.exists() and f.suffix == ".mojo"]
    else:
        # Scan tests directory
        test_dir = Path("tests")
        if not test_dir.exists():
            print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
            sys.exit(1)
        test_files = list(test_dir.rglob("test_*.mojo"))

    if not test_files:
        print("No test files to validate", file=sys.stderr)
        sys.exit(0)

    # Check each file
    violations = []
    for filepath in sorted(test_files):
        # Skip deprecated files
        if ".DEPRECATED" in filepath.name:
            continue

        count = count_tests_in_file(filepath)
        if args.verbose:
            status = "FAIL" if count > args.threshold else "OK"
            print(f"  [{status}] {filepath}: {count} tests")

        if count > args.threshold:
            violations.append((filepath, count))

    if not violations:
        if args.verbose:
            print(f"\n✅ All {len(test_files)} test file(s) pass (under {args.threshold} tests each)")
        sys.exit(0)

    # Report violations
    print(
        f"Error: {len(violations)} file(s) exceed the {args.threshold}-test limit",
        file=sys.stderr,
    )
    print("\nViolations:", file=sys.stderr)
    for filepath, count in violations:
        print(f"  {filepath}: {count} tests (max: {args.threshold})", file=sys.stderr)
    print(
        "\nNote: The Mojo 0.26.1 runtime has a heap corruption bug that crashes",
        file=sys.stderr,
    )
    print(
        "after running ~15 cumulative tests. Keep files under 10 tests each.",
        file=sys.stderr,
    )
    print("\nSee: Issue #2942, ADR-009", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
