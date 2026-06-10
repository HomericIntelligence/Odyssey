#!/usr/bin/env python3
"""Check that every src/projectodyssey/**/*.mojo has a corresponding test_*.mojo.

This is a FILE-LEVEL workaround for the absence of Mojo coverage tooling
(ADR-008). It flags source files with no discoverable test file. It does NOT
measure line or branch coverage.

Exit codes:
  0 - Always (warn-only). Hard-gating is a follow-up after baseline is clean.

Usage:
    python scripts/check_source_coverage.py
"""

import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent))
from validate_test_coverage import find_test_files  # reuses existing exclusion list
from hephaestus.utils import get_repo_root

# Source files known to have no test by design (e.g. pure re-export shims)
SOURCE_EXCLUSIONS: Set[str] = set()


def find_source_files(repo_root: Path) -> List[Path]:
    """All non-__init__ .mojo files under src/projectodyssey/."""
    src = repo_root / "src" / "projectodyssey"
    return sorted(
        p.relative_to(repo_root)
        for p in src.rglob("*.mojo")
        if p.name != "__init__.mojo" and str(p.relative_to(repo_root)) not in SOURCE_EXCLUSIONS
    )


def expected_test_paths(source: Path) -> List[Path]:
    """Candidate test file locations for a given src path.

    For src/projectodyssey/core/foo.mojo, candidates are:
      tests/projectodyssey/core/test_foo.mojo            (mirror layout, primary)
      tests/projectodyssey/core/test_foo_part1.mojo      (split-file pattern)
      tests/core/test_foo.mojo                            (legacy flat layout)
      tests/models/test_foo.mojo                          (model-specific layout)
      tests/shared/<subpath>/test_foo.mojo                (shared-library layout)
    """
    # source like src/projectodyssey/core/foo.mojo -> subpath = core/foo.mojo
    parts = source.parts
    assert parts[:2] == ("src", "projectodyssey"), source
    subpath = Path(*parts[2:])
    stem = subpath.stem  # "foo"
    parent = subpath.parent  # "core"
    return [
        Path("tests/projectodyssey") / parent / f"test_{stem}.mojo",
        Path("tests/projectodyssey") / parent / f"test_{stem}_part1.mojo",
        Path("tests") / parent / f"test_{stem}.mojo",
        Path("tests/models") / f"test_{stem}.mojo",
        Path("tests/shared") / parent / f"test_{stem}.mojo",
    ]


def find_uncovered_sources(sources: List[Path], all_tests: Set[Path]) -> List[Path]:
    """Return source files whose every candidate test path is missing."""
    uncovered = []
    for src in sources:
        if not any(cand in all_tests for cand in expected_test_paths(src)):
            uncovered.append(src)
    return uncovered


def generate_report(uncovered: List[Path], total_sources: int) -> str:
    """Generate a human-readable report of source-to-test coverage."""
    if not uncovered:
        return f"✅ All {total_sources} source files have a matching test file."
    lines = [
        f"⚠️  {len(uncovered)} of {total_sources} source files have no matching test:",
        "",
    ]
    lines.extend(f"   • {p}" for p in uncovered)
    lines.append("")
    lines.append("Expected test paths for src/projectodyssey/X/Y.mojo (any one):")
    lines.append("   • tests/projectodyssey/X/test_Y.mojo")
    lines.append("   • tests/projectodyssey/X/test_Y_part1.mojo  (split-file)")
    lines.append("   • tests/X/test_Y.mojo  (legacy)")
    lines.append("   • tests/models/test_Y.mojo  (model-specific)")
    lines.append("   • tests/shared/X/test_Y.mojo  (shared-library)")
    return "\n".join(lines)


def main() -> int:
    """Run source-to-test mapping check."""
    repo_root = get_repo_root()
    sources = find_source_files(repo_root)
    tests = set(find_test_files(repo_root))
    # Sanity guard: this repo always has hundreds of .mojo test files. An empty
    # result means test discovery is misconfigured — e.g. running from inside a
    # `worktrees/` checkout, which find_test_files() deliberately excludes. In
    # that case every source would be falsely reported as uncovered, so fail
    # loudly instead of emitting a misleading 100%-uncovered report.
    if not tests:
        print(
            "❌ No test files discovered under tests/ — source-to-test mapping "
            "cannot run. This usually means the script is being invoked from a "
            "worktree checkout (excluded by find_test_files). Run from a primary "
            "repository checkout.",
            file=sys.stderr,
        )
        return 1
    uncovered = find_uncovered_sources(sources, tests)
    print(generate_report(uncovered, len(sources)))
    return 0  # warn-only on first land (ADR-008 + KISS rollout)


if __name__ == "__main__":
    sys.exit(main())
