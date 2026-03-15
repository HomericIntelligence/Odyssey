#!/usr/bin/env python3
"""
Validate that deleted agents are not referenced in documentation.

This script helps with agent consolidation PRs by detecting stale references
to agents that have been deleted or merged. Run this when consolidating agents
to ensure no orphaned references remain in the codebase.

Usage:
    python3 scripts/validate_agent_consolidation.py <agent-name> [agent-name2...]
    python3 scripts/validate_agent_consolidation.py --help

Examples:
    # Check for stale references to a single deleted agent
    python3 scripts/validate_agent_consolidation.py old-agent-name

    # Check for multiple deleted agents
    python3 scripts/validate_agent_consolidation.py agent1 agent2 agent3

Exit codes:
    0 - No stale references found
    1 - Stale references detected in documentation
"""

import argparse
import sys
from pathlib import Path
from typing import List


def find_stale_references(agent_names: List[str], repo_root: Path = None) -> dict:
    """
    Find stale references to deleted agents in markdown files.

    Args:
        agent_names: List of deleted agent names to search for
        repo_root: Repository root path (defaults to CWD)

    Returns:
        Dictionary mapping file paths to lists of matching lines
    """
    if repo_root is None:
        repo_root = Path.cwd()

    stale_refs = {}

    # Search in markdown files in agents/ and docs/
    search_dirs = [
        repo_root / "agents",
        repo_root / "docs",
        repo_root / ".claude",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for md_file in search_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                matching_lines = []
                for line_num, line in enumerate(lines, 1):
                    for agent_name in agent_names:
                        # Search for agent name in links, references, and text
                        # Avoid false positives by checking context
                        if agent_name.lower() in line.lower():
                            # Skip if line is in a list of all agents or similar
                            if "agent-" not in line.lower() and "agent hierarchy" not in line.lower():
                                matching_lines.append((line_num, line.strip()))

                if matching_lines:
                    rel_path = md_file.relative_to(repo_root)
                    stale_refs[str(rel_path)] = matching_lines

            except Exception as e:
                print(f"Warning: Could not read {md_file}: {e}", file=sys.stderr)

    return stale_refs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate that deleted agents are not referenced in documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check for a single agent
    python3 scripts/validate_agent_consolidation.py old-agent-name

    # Check multiple agents at once
    python3 scripts/validate_agent_consolidation.py agent1 agent2

During agent consolidation:
    1. List the agents being deleted/merged
    2. Run this script with their names
    3. Review any stale references found
    4. Update documentation to remove old references
        """,
    )

    parser.add_argument(
        "agent_names",
        nargs="+",
        help="Name(s) of agents being deleted/consolidated",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: current directory)",
    )

    args = parser.parse_args()

    print(f"Checking for stale references to: {', '.join(args.agent_names)}\n")

    stale_refs = find_stale_references(args.agent_names, args.repo_root)

    if not stale_refs:
        print("✓ No stale references found. Agent consolidation is clean.")
        return 0

    print(f"✗ Found stale references in {len(stale_refs)} file(s):\n")

    for file_path in sorted(stale_refs.keys()):
        print(f"  {file_path}:")
        for line_num, line in stale_refs[file_path]:
            print(f"    Line {line_num}: {line[:80]}")
        print()

    print("Please review and update these references before merging the consolidation PR.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
