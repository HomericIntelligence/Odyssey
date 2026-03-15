#!/usr/bin/env python3
"""
Validate that GitHub Actions workflows match documentation.

This script detects drift between:
1. Workflows documented in .github/workflows/README.md
2. Actual workflow files in .github/workflows/

Prevents documentation from falling out of sync.

Usage:
    python3 scripts/validate_workflow_inventory.py [--verbose]

Exit codes:
    0 - Inventory is in sync
    1 - Drift detected (missing or extra workflows)
"""

import re
import sys
from pathlib import Path
from typing import Set, Tuple


def extract_documented_workflows(readme_path: Path) -> Set[str]:
    """Extract workflow filenames from README.md table."""
    documented = set()

    try:
        content = readme_path.read_text()
    except Exception as e:
        print(f"Error reading README: {e}", file=sys.stderr)
        return documented

    # Look for workflow filenames in the table
    # Patterns: [filename.yml], `filename.yml`, or just filename.yml in table rows
    patterns = [
        r'\[([a-z0-9_-]+\.yml)\]',  # [filename.yml] markdown links
        r'`([a-z0-9_-]+\.yml)`',    # `filename.yml` code blocks
        r'\| [a-z0-9_-]*([a-z0-9_-]+\.yml)',  # | filename.yml in table
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            filename = match.group(1)
            documented.add(filename)

    return documented


def find_actual_workflows(workflows_dir: Path) -> Set[str]:
    """Find all .yml workflow files in the directory."""
    workflows = set()

    if not workflows_dir.exists():
        return workflows

    for workflow_file in workflows_dir.glob("*.yml"):
        workflows.add(workflow_file.name)

    return workflows


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate workflow documentation matches actual files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/validate_workflow_inventory.py
    python3 scripts/validate_workflow_inventory.py --verbose

This detects when:
- A workflow file exists but is not documented in README.md
- A workflow is documented but no file exists on disk
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    workflows_dir = repo_root / ".github" / "workflows"
    readme_path = workflows_dir / "README.md"

    documented = extract_documented_workflows(readme_path)
    actual = find_actual_workflows(workflows_dir)

    missing_docs = actual - documented
    orphaned_docs = documented - actual

    if args.verbose:
        print(f"Documented workflows: {len(documented)}")
        print(f"Actual workflows: {len(actual)}\n")

    errors = False

    if missing_docs:
        errors = True
        print("✗ Workflows NOT documented in README.md:")
        for workflow in sorted(missing_docs):
            print(f"  - {workflow}")
        print()

    if orphaned_docs:
        errors = True
        print("✗ Documented workflows that don't exist on disk:")
        for workflow in sorted(orphaned_docs):
            print(f"  - {workflow}")
        print()

    if not errors:
        print("✓ Workflow inventory is in sync")
        return 0

    print("Please update .github/workflows/README.md to match actual files.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
