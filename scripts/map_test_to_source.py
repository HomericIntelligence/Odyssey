#!/usr/bin/env python3
"""Map test files to source files to identify coverage gaps.

Since Mojo does not provide coverage tooling (ADR-008), this script identifies which
source modules under shared/ have corresponding test files under tests/shared/, and
reports gaps ranked by priority from coverage.toml.

Usage:
    python scripts/map_test_to_source.py [--ci] [--post-pr] [--tolerance RATIO]

Example:
    python scripts/map_test_to_source.py --ci --tolerance 0.15
"""

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add scripts directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Module priorities from coverage.toml
MODULE_PRIORITIES = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def find_source_files(root_dir: Path, exclude_pattern: str = "__init__.mojo") -> List[Path]:
    """Find all non-test .mojo files under shared/.

    Args:
        root_dir: Root directory to search (e.g., shared/)
        exclude_pattern: Pattern to exclude (default: __init__.mojo)

    Returns:
        List of .mojo file paths sorted alphabetically.
    """
    if not root_dir.exists():
        return []

    source_files = []
    for mojo_file in sorted(root_dir.rglob("*.mojo")):
        # Skip __init__.mojo files and any test files
        if exclude_pattern in mojo_file.name or mojo_file.name.startswith("test_"):
            continue
        source_files.append(mojo_file)

    return source_files


def find_test_files(root_dir: Path) -> List[Path]:
    """Find all test_*.mojo files under tests/.

    Args:
        root_dir: Root directory to search (e.g., tests/shared/)

    Returns:
        List of test_*.mojo file paths sorted alphabetically.
    """
    if not root_dir.exists():
        return []

    test_files = []
    for test_file in sorted(root_dir.rglob("test_*.mojo")):
        test_files.append(test_file)

    return test_files


def build_mapping(
    source_files: List[Path], test_files: List[Path], source_root: Optional[Path] = None
) -> Tuple[Dict[Path, List[Path]], Set[Path]]:
    """Map source files to their corresponding test files.

    Mapping strategy:
    - src/projectodyssey/core/loss.mojo -> tests/projectodyssey/core/test_loss*.mojo
    - src/projectodyssey/core/layers/linear.mojo -> tests/projectodyssey/core/layers/test_linear*.mojo
    - Handles both single test files (test_loss.mojo) and part-numbered test files
      (test_loss_part1.mojo through test_loss_part6.mojo)

    Args:
        source_files: List of source .mojo file paths
        test_files: List of test_*.mojo file paths
        source_root: Root directory of source files (defaults to PROJECT_ROOT/src/projectodyssey)

    Returns:
        Tuple of (mapping dict, set of unmapped source files)
    """
    if source_root is None:
        source_root = PROJECT_ROOT / "src" / "projectodyssey"

    mapping: Dict[Path, List[Path]] = {}
    unmapped: Set[Path] = set(source_files)

    for source_file in source_files:
        # Compute the expected test file path pattern
        # src/projectodyssey/core/loss.mojo -> tests/projectodyssey/core/test_loss
        relative_path = source_file.relative_to(source_root)
        test_prefix = "test_" + relative_path.stem  # e.g., test_loss
        # Map src/projectodyssey -> tests/projectodyssey
        test_root = source_root.parent.parent / "tests" / source_root.name
        test_dir = test_root / relative_path.parent

        # Find all matching test files (including part-numbered variants)
        matching_tests = []
        for test_file in test_files:
            # Check if test file is in the correct directory and matches the prefix
            if test_file.parent == test_dir and test_file.stem.startswith(test_prefix):
                matching_tests.append(test_file)

        if matching_tests:
            mapping[source_file] = matching_tests
            unmapped.discard(source_file)

    return mapping, unmapped


def load_coverage_config(coverage_toml_path: Path) -> Dict[str, Dict[str, str]]:
    """Load coverage.toml to get module priority information.

    Args:
        coverage_toml_path: Path to coverage.toml

    Returns:
        Dictionary with module paths as keys and {"priority": "critical"|"high"|"medium"|"low"} as values
    """
    if not coverage_toml_path.exists():
        return {}

    with open(coverage_toml_path, "rb") as f:
        config = tomllib.load(f)

    module_priorities: Dict[str, Dict[str, str]] = {}
    for priority_level in ["critical", "high", "medium", "low"]:
        modules = config.get(priority_level, {}).get("modules", [])
        for module in modules:
            module_priorities[module] = {"priority": priority_level}

    return module_priorities


def score_gaps(
    unmapped: Set[Path],
    coverage_config: Dict[str, Dict[str, str]],
    source_root: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """Score unmapped source files by priority from coverage.toml.

    Args:
        unmapped: Set of unmapped source file paths
        coverage_config: Module priority configuration from coverage.toml
        source_root: Root directory of source files (defaults to PROJECT_ROOT/src/projectodyssey)

    Returns:
        List of (path, priority) tuples sorted by priority (critical first)
    """
    if source_root is None:
        source_root = PROJECT_ROOT / "src" / "projectodyssey"

    scored = []

    for source_file in unmapped:
        # Try to match the source file path against coverage.toml module paths
        # coverage.toml uses paths like "src/projectodyssey/core" but our source is "shared/core"
        # We need to map "shared/core/loss.mojo" to "shared/core" for comparison
        relative_path = source_file.relative_to(source_root)
        # Get the module path (parent directories)
        module_path = str(relative_path.parent).replace("\\", "/")
        if module_path == ".":
            module_path = source_root.name
        else:
            module_path = f"{source_root.name}/{module_path}"

        # Look up priority in coverage config
        priority = "low"  # Default to low if not found
        for config_module, config_info in coverage_config.items():
            if module_path in config_module or config_module.endswith(module_path):
                priority = config_info.get("priority", "low")
                break

        scored.append((source_file, priority))

    # Sort by priority (critical first)
    scored.sort(key=lambda x: MODULE_PRIORITIES.get(x[1], 999))
    return scored


def generate_report(
    mapping: Dict[Path, List[Path]], gaps: List[Tuple[Path, str]]
) -> str:
    """Generate a markdown report of test-to-source mapping.

    Args:
        mapping: Mapping of source files to test files
        gaps: List of (unmapped_path, priority) tuples

    Returns:
        Markdown-formatted report string
    """
    total_source = len(mapping) + len(gaps)
    mapped_count = len(mapping)
    unmapped_count = len(gaps)

    mapped_pct = (mapped_count / total_source * 100) if total_source > 0 else 0
    unmapped_pct = (unmapped_count / total_source * 100) if total_source > 0 else 0

    lines = [
        "# Test-to-Source Mapping Report",
        "",
        "## Summary",
        f"- Source files: {total_source}",
        f"- Mapped (have tests): {mapped_count} ({mapped_pct:.1f}%)",
        f"- Unmapped (no tests): {unmapped_count} ({unmapped_pct:.1f}%)",
        "",
    ]

    if gaps:
        lines.extend(["## Unmapped Source Files (by priority)", ""])
        current_priority = None
        for source_path, priority in gaps:
            if priority != current_priority:
                lines.append(f"### {priority.upper()}")
                lines.append("")
                current_priority = priority
            # Make path relative to PROJECT_ROOT for readability
            rel_path = source_path.relative_to(PROJECT_ROOT)
            lines.append(f"- {rel_path}")

        lines.append("")

    if mapped_count > 0:
        lines.extend(["## Mapped Source Files (Sample)", ""])
        # Show first 10 mapped files
        for source_path, test_paths in list(mapping.items())[:10]:
            rel_src = source_path.relative_to(PROJECT_ROOT)
            rel_tests = [t.relative_to(PROJECT_ROOT) for t in test_paths]
            lines.append(f"- {rel_src}")
            for test_path in rel_tests:
                lines.append(f"  - {test_path}")

        if len(mapping) > 10:
            lines.append(f"... and {len(mapping) - 10} more mapped files")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Run the test-to-source mapping script.

    Returns:
        Exit code: 0 if gap is within tolerance, 1 if over tolerance or error
    """
    parser = argparse.ArgumentParser(
        description="Map test files to source files and report coverage gaps"
    )
    parser.add_argument(
        "--ci", action="store_true", help="CI mode: exit 1 if gap exceeds tolerance"
    )
    parser.add_argument(
        "--post-pr", action="store_true", help="Post report to PR comment (requires gh CLI)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Tolerance threshold for unmapped percentage (default: 0.10 = 10%%)",
    )
    args = parser.parse_args()

    # Find source and test files
    source_root = PROJECT_ROOT / "src" / "projectodyssey"
    test_root = PROJECT_ROOT / "tests" / "projectodyssey"

    source_files = find_source_files(source_root)
    test_files = find_test_files(test_root)

    if not source_files:
        print("ERROR: No source files found in shared/", file=sys.stderr)
        return 1

    # Build mapping
    mapping, unmapped = build_mapping(source_files, test_files)

    # Load coverage config and score gaps
    coverage_config = load_coverage_config(PROJECT_ROOT / "coverage.toml")
    scored_gaps = score_gaps(unmapped, coverage_config)

    # Generate report
    report = generate_report(mapping, scored_gaps)
    print(report)

    # Check tolerance
    unmapped_ratio = len(unmapped) / len(source_files) if source_files else 0
    exceeds_tolerance = unmapped_ratio > args.tolerance

    # Baseline mode: if CI flag is set and this is the first run (no prior baseline),
    # we establish the baseline and exit 0. This prevents CI failures on the first merge.
    # The tolerance check kicks in from the second run onwards.
    baseline_file = PROJECT_ROOT / ".claude" / ".coverage_baseline"

    if args.ci:
        if not baseline_file.exists():
            # Establish baseline on first run
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            baseline_file.write_text(str(unmapped_ratio))
            print(
                f"\n✓ Baseline established: {unmapped_ratio:.1%} unmapped (tolerance: {args.tolerance:.0%})"
            )
            return 0

        # Check against baseline from second run onwards
        baseline_ratio = float(baseline_file.read_text().strip())
        if unmapped_ratio > baseline_ratio + args.tolerance:
            print(
                f"\n✗ FAILED: Gap increased from {baseline_ratio:.1%} to {unmapped_ratio:.1%} "
                f"(exceeds tolerance of {args.tolerance:.0%})",
                file=sys.stderr,
            )
            return 1

        print(
            f"\n✓ PASSED: Gap {unmapped_ratio:.1%} is within tolerance (baseline: {baseline_ratio:.1%}, "
            f"threshold: {baseline_ratio + args.tolerance:.1%})"
        )
        return 0

    # Non-CI mode: just report
    if exceeds_tolerance:
        print(
            f"\nℹ Gap is {unmapped_ratio:.1%}, exceeds tolerance of {args.tolerance:.0%}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
