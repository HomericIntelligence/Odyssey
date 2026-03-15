#!/usr/bin/env python3
"""
Detect potentially stale scripts in the scripts/ directory.

A script is considered potentially stale if:
- It exists in scripts/ directory
- It is not referenced in justfile, .github/workflows, or other scripts
- It is not a known utility/library file (common.py, conftest.py, etc.)

This lightweight check helps surface unused scripts that could be deleted,
preventing code bloat and maintenance burden.

Usage:
    python3 scripts/detect_stale_scripts.py [--verbose] [--exclude PATTERN]

Exit codes:
    0 - No stale scripts detected
    1 - Stale scripts detected (warnings printed)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set


# Known utility/library scripts that don't need to be referenced
KNOWN_UTILITIES = {
    "common.py",
    "conftest.py",
    "__init__.py",
    "setup.py",
    "__pycache__",
}

# Known patterns for scripts that are referenced in special ways
SPECIAL_PATTERNS = {
    "test_",  # Test files (referenced by pytest)
    "conftest",  # Pytest config
}


def is_known_utility(script_name: str) -> bool:
    """Check if script is a known utility that doesn't need explicit references."""
    if script_name in KNOWN_UTILITIES:
        return True
    for pattern in SPECIAL_PATTERNS:
        if pattern in script_name:
            return True
    return False


def find_all_scripts(scripts_dir: Path) -> Set[str]:
    """Find all Python and shell scripts in scripts/ directory."""
    scripts = set()
    for script in scripts_dir.rglob("*"):
        if script.is_file():
            # Include .py and .sh files
            if script.suffix in {".py", ".sh", ".mojo"}:
                rel_name = script.relative_to(scripts_dir).name
                if rel_name and not rel_name.startswith("."):
                    scripts.add(rel_name)
    return scripts


def find_script_references(repo_root: Path) -> Set[str]:
    """Find all scripts that are referenced in the codebase."""
    referenced = set()

    # Search in justfile
    justfile = repo_root / "justfile"
    if justfile.exists():
        content = justfile.read_text()
        # Find script references (justfile calls like 'python3 scripts/name.py')
        for match in re.finditer(r"scripts/([a-zA-Z0-9_-]+\.(?:py|sh|mojo))", content):
            referenced.add(match.group(1))

    # Search in .github/workflows
    workflows_dir = repo_root / ".github" / "workflows"
    if workflows_dir.exists():
        for workflow_file in workflows_dir.glob("*.yml"):
            content = workflow_file.read_text()
            for match in re.finditer(r"scripts/([a-zA-Z0-9_-]+\.(?:py|sh|mojo))", content):
                referenced.add(match.group(1))

    # Search in other scripts (recursive imports/calls)
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        for script in scripts_dir.rglob("*.py"):
            content = script.read_text()
            for match in re.finditer(r"(?:from|import|run)\s+(?:\w+/)*([a-zA-Z0-9_]+)", content):
                script_name = match.group(1)
                if script_name and script_name != "__main__":
                    referenced.add(f"{script_name}.py")

    # Search in documentation
    docs_dir = repo_root / "docs"
    if docs_dir.exists():
        for doc_file in docs_dir.rglob("*.md"):
            content = doc_file.read_text()
            for match in re.finditer(r"scripts/([a-zA-Z0-9_-]+\.(?:py|sh|mojo))", content):
                referenced.add(match.group(1))

    return referenced


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect potentially stale scripts in scripts/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
A script is considered stale if it's not referenced in:
- justfile
- .github/workflows/*.yml
- Other scripts/ files
- docs/ documentation

This helps identify scripts that could be deleted during cleanup.

Examples:
    python3 scripts/detect_stale_scripts.py
    python3 scripts/detect_stale_scripts.py --verbose
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including all referenced scripts",
    )
    parser.add_argument(
        "--exclude",
        metavar="PATTERN",
        default=None,
        help="Exclude scripts matching this pattern (e.g., 'test_')",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    scripts_dir = repo_root / "scripts"

    if not scripts_dir.exists():
        print("Scripts directory not found")
        return 1

    all_scripts = find_all_scripts(scripts_dir)
    referenced = find_script_references(repo_root)

    # Filter out known utilities
    all_scripts = {s for s in all_scripts if not is_known_utility(s)}

    # Apply exclude pattern if provided
    if args.exclude:
        all_scripts = {s for s in all_scripts if args.exclude not in s}

    stale = all_scripts - referenced

    if args.verbose:
        print(f"Total scripts: {len(all_scripts)}")
        print(f"Referenced: {len(referenced)}")
        print(f"Stale: {len(stale)}\n")

    if stale:
        print(f"⚠ Found {len(stale)} potentially stale script(s):\n")
        for script in sorted(stale):
            print(f"  - {script}")
        print("\nConsider removing these scripts if they are no longer needed.")
        return 1

    print("✓ No stale scripts detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
