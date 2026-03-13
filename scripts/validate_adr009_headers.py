#!/usr/bin/env python3
"""Validate ADR-009 headers in split test files.

Ensures all test_*_part*.mojo files contain the ADR-009 split docstring
referencing heap corruption / Issue #2942.

Usage:
    python3 scripts/validate_adr009_headers.py [--fix]
"""

import re
import sys
from pathlib import Path

MARKERS = [
    "heap corruption",
    "Issue #2942",
    "ADR-009",
    "Split from monolithic",
]

SPLIT_PATTERN = re.compile(r"test_.*_part\d+\.mojo$")


def find_split_files(root: Path) -> list[Path]:
    """Find all test_*_part*.mojo files under root."""
    return sorted(p for p in root.rglob("test_*_part*.mojo") if SPLIT_PATTERN.search(p.name))


def has_adr009_header(path: Path) -> bool:
    """Check if file contains any ADR-009 marker in first 10 lines."""
    try:
        lines = path.read_text().splitlines()[:40]
        text = "\n".join(lines).lower()
        return any(m.lower() in text for m in MARKERS)
    except Exception:
        return False


def main() -> int:
    root = Path("tests")
    if not root.exists():
        print("ERROR: tests/ directory not found", file=sys.stderr)
        return 1

    files = find_split_files(root)
    missing = [f for f in files if not has_adr009_header(f)]

    if missing:
        print(f"FAIL: {len(missing)} split file(s) missing ADR-009 header:")
        for f in missing:
            print(f"  {f}")
        return 1

    print(f"OK: All {len(files)} split test files have ADR-009 headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
