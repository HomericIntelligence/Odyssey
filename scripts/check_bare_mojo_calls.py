#!/usr/bin/env python3
"""Check for bare 'pixi run mojo test|run' calls in GitHub Actions workflow files.

Bare calls lack retry wrappers or just test-group routing and can cause
flaky CI failures (ADR-009 JIT crash issue, ref #3329). Suppress per-line with:
  # no-bare-mojo-lint: <reason>

Exit codes: 0 = clean, 1 = violations found

Usage:
    python3 scripts/check_bare_mojo_calls.py [files...]

Examples:
    python3 scripts/check_bare_mojo_calls.py .github/workflows/*.yml
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Matches: pixi run mojo test ... OR pixi run mojo run ...
# Does NOT match: --version, build, package, format, mojo-format (hyphenated)
BARE_PATTERN = re.compile(r"pixi run mojo\s+(test|run)\b")
SUPPRESSION = "# no-bare-mojo-lint"


def check_file(path: Path) -> List[Tuple[int, str]]:
    """Check a single workflow file for bare pixi run mojo calls.

    Args:
        path: Path to the workflow YAML file to check.

    Returns:
        List of (line_number, line_content) tuples for each violation found.
    """
    violations: List[Tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if SUPPRESSION in line:
            continue
        if BARE_PATTERN.search(line):
            violations.append((lineno, line.rstrip()))
    return violations


def main(argv: List[str]) -> int:
    """Run the bare mojo call checker on the given files.

    Args:
        argv: List of file paths to check (as strings).

    Returns:
        Exit code: 0 if no violations found, 1 if violations found.
    """
    files = [Path(f) for f in argv] if argv else []
    all_violations: List[Tuple[Path, int, str]] = []
    for f in files:
        if not f.exists():
            continue
        for lineno, line in check_file(f):
            all_violations.append((f, lineno, line))
    if all_violations:
        for path, lineno, line in all_violations:
            print(
                f"{path}:{lineno}: bare pixi run mojo call — wrap with just test-group or add retry, "
                f"or suppress with '# no-bare-mojo-lint: <reason>'"
            )
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
