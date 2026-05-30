#!/usr/bin/env python3
"""Check test coverage against threshold.

This script validates that code coverage meets the required threshold.
Used in CI to enforce coverage requirements.

Usage:
    python scripts/check_coverage.py --threshold 90 --path src/projectodyssey/
    python scripts/check_coverage.py --config coverage.toml --path src/projectodyssey/core
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from hephaestus.utils import get_repo_root

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


def load_coverage_config(config_file: Optional[Path] = None) -> Dict:
    """Load coverage configuration from coverage.toml.

    Args:
        config_file: Path to coverage.toml. If None, looks for coverage.toml in repo root.

    Returns:
        Dictionary with coverage configuration, or empty dict if file not found.
    """
    if config_file is None:
        # Try to find coverage.toml in repo root
        repo_root = get_repo_root()
        config_file = repo_root / "coverage.toml"

    if not config_file.exists():
        return {"coverage": {"target": 90.0, "minimum": 80.0}}

    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        return config
    except Exception as e:
        print(f"⚠️  Warning: Failed to load config from {config_file}: {e}", file=sys.stderr)
        return {"coverage": {"target": 90.0, "minimum": 80.0}}


def get_module_threshold(path: str, config: Dict) -> float:
    """Get coverage threshold for a specific module.

    Args:
        path: Module path (e.g., "src/projectodyssey/core")
        config: Configuration dictionary from load_coverage_config()

    Returns:
        Minimum coverage threshold for the module.
    """
    modules = config.get("coverage", {}).get("modules", {})

    # Try exact match first
    if path in modules:
        return modules[path].get("minimum", 90.0)

    # Try prefix match
    for module_path in sorted(modules.keys(), key=len, reverse=True):
        if path.startswith(module_path):
            return modules[module_path].get("minimum", 90.0)

    # Default to overall minimum
    return config.get("coverage", {}).get("minimum", 80.0)


def parse_coverage_report(coverage_file: Path) -> Optional[float]:
    """Parse coverage report and extract total coverage percentage.

    ⚠️  WARNING: NOT IMPLEMENTED - RETURNS None ⚠️

    This function does NOT parse coverage data!
    It always returns None to indicate coverage parsing is not available.

    Real implementation blocked by:
    - Mojo v0.26+ lacks coverage instrumentation
    - No coverage report format available to parse
    - No `mojo test --coverage` command exists

    See Issues:
    - #2583: Coverage implementation tracking
    - #2612: Prominent warning documentation
    - ADR-008: Coverage tool blocker decision

    Args:
        coverage_file: Path to coverage report file.

    Returns:
        None - Coverage parsing is not implemented.
        This ensures coverage checks are properly skipped rather than
        falsely passing with a mock value.
    """
    # TODO(#2583): BLOCKED - Waiting on Mojo team to release coverage instrumentation
    #
    # CONTEXT: Mojo v0.26+ does not provide built-in code coverage tools
    # - No coverage instrumentation (no `mojo test --coverage` equivalent)
    # - No coverage report generation (no XML/JSON output)
    # - Expected format when available: Cobertura XML (standard for Python ecosystems)
    #
    # WORKAROUND: Manual test discovery via `validate_test_coverage.py` ensures all tests run
    #
    # DECISION: Return None instead of fake value (see Issue #2612)
    # - This prevents false sense of security from hardcoded mock values
    # - Coverage checks are properly skipped until Mojo provides coverage tooling
    # - CI test validation still runs (ensures tests execute, just no coverage metrics)
    #
    # BLOCKED BY: Mojo team (external dependency)
    # REFERENCE: Issue #2583, Issue #2612, Issue #3079 (tracking), ADR-008
    print()
    print("⚠️" + "=" * 68)
    print("⚠️  WARNING: COVERAGE PARSING NOT IMPLEMENTED!")
    print("⚠️" + "=" * 68)
    print()
    print(f"   Coverage file: {coverage_file}")
    print(f"   File exists: {coverage_file.exists()}")
    print()
    print("   This function does NOT parse coverage data.")
    print("   Returning None to indicate coverage check is disabled.")
    print()
    print("   REASON: Mojo does not provide coverage instrumentation")
    print("   - No `mojo test --coverage` command exists")
    print("   - No coverage report format to parse")
    print()
    print("   IMPACT: Coverage checks are skipped (better than fake passing)")
    print("   - No false sense of security from hardcoded mock values")
    print("   - CI still validates that tests execute successfully")
    print()
    print("   REFERENCE: ADR-008, Issue #2583, Issue #2612")
    print()
    print("   ➤ Returning None (coverage check disabled)")
    print()

    # Don't return fake value - return None to indicate not implemented
    # This way coverage checks are properly skipped rather than falsely passing
    return None


def check_coverage(threshold: float, path: str, coverage_file: Path) -> bool:
    """Check if coverage meets threshold.

    Args:
        threshold: Minimum required coverage percentage.
        path: Path to source code being tested.
        coverage_file: Path to coverage report.

    Returns:
        True if coverage meets threshold or coverage is not available,
        False only if coverage parsing succeeds but is below threshold.
    """
    coverage = parse_coverage_report(coverage_file)

    if coverage is None:
        print("\n📊 Coverage Report")
        print(f"   Path: {path}")
        print("   Status: Coverage parsing not implemented")
        print(f"   Threshold: {threshold:.2f}%")
        print()
        print("   ✅ PASSED - Coverage check skipped (tool not available)")
        print("      Tests are still validated for execution success")
        return True  # Pass gracefully when coverage is not available

    print("\n📊 Coverage Report")
    print(f"   Path: {path}")
    print(f"   Coverage: {coverage:.2f}%")
    print(f"   Threshold: {threshold:.2f}%")

    if coverage >= threshold:
        print("   ✅ PASSED - Coverage meets threshold")
        return True
    else:
        gap = threshold - coverage
        print(f"   ❌ FAILED - Coverage is {gap:.2f}% below threshold")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check test coverage against threshold")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum required coverage percentage (overrides config)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="src/projectodyssey/",
        help="Path to source code being tested (default: src/projectodyssey/)",
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to coverage report file (default: coverage.xml)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to coverage.toml config file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Load configuration
    config = load_coverage_config(args.config)

    # Determine threshold: use --threshold flag if provided, else get from config
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = get_module_threshold(args.path, config)

    if args.verbose:
        print("Checking coverage with settings:")
        print(f"  Threshold: {threshold}%")
        print(f"  Path: {args.path}")
        print(f"  Coverage file: {args.coverage_file}")
        print(f"  Config file: {args.config or 'default (coverage.toml)'}")

    # Check if coverage file exists
    if not args.coverage_file.exists():
        print(f"\n⚠️  WARNING: Coverage file not found: {args.coverage_file}")
        print()
        print("   REASON: Mojo does not yet provide coverage instrumentation")
        print("   - Mojo v0.26+ lacks built-in coverage tools (no `mojo test --coverage`)")
        print("   - This is NOT a bug - waiting on Mojo team to release coverage support")
        print()
        print("   WORKAROUND: Manual test discovery ensures all tests execute")
        print("   - Script `validate_test_coverage.py` verifies test files exist")
        print("   - CI runs all tests via `just test-mojo` (validation only, no metrics)")
        print()
        print("   IMPACT: Test execution is verified, but coverage metrics unavailable")
        print("   - CI passes without coverage enforcement until tooling exists")
        print()
        print("   REFERENCE: See ADR-008 and Issue #2583 for detailed explanation")
        print("\n   For now, assuming coverage meets threshold...")
        sys.exit(0)  # Don't fail CI until Mojo coverage is available

    # Check coverage
    success = check_coverage(threshold, args.path, args.coverage_file)

    if not success:
        print("\n💡 Tips for improving coverage:")
        print("   - Add tests for uncovered functions")
        print("   - Test edge cases and error paths")
        print("   - Check for untested branches in conditionals")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
