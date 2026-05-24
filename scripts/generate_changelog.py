#!/usr/bin/env python3
"""Generate changelog from git commit history.

ADR-001 Justification: Python required for:
- Subprocess execution to run git commands
- Text parsing and categorization of commits
- Formatted output generation

Usage:
    python scripts/generate_changelog.py                    # Since last tag
    python scripts/generate_changelog.py v0.2.0             # For specific version
    python scripts/generate_changelog.py v0.2.0 v0.1.0      # Between versions
    python scripts/generate_changelog.py --output CHANGELOG.md
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add scripts directory to path for common utilities
sys.path.insert(0, str(Path(__file__).parent))
from hephaestus.utils import get_repo_root


def run_git_command(args: List[str]) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=get_repo_root(),
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_latest_tag() -> Optional[str]:
    """Get the most recent release tag."""
    output = run_git_command(["describe", "--tags", "--abbrev=0"])
    return output if output else None


def get_previous_tag(current_tag: str) -> Optional[str]:
    """Get the tag before the current one."""
    output = run_git_command(["describe", "--tags", "--abbrev=0", f"{current_tag}^"])
    return output if output else None


def get_commits_between(from_ref: Optional[str], to_ref: str = "HEAD") -> List[str]:
    """Get commit messages between two refs."""
    if from_ref:
        range_spec = f"{from_ref}..{to_ref}"
    else:
        range_spec = to_ref

    output = run_git_command(
        [
            "log",
            range_spec,
            "--pretty=format:%h|%s|%an",
            "--no-merges",
        ]
    )

    if not output:
        return []

    return output.split("\n")


def parse_commit(commit_line: str) -> Tuple[str, str, str, str]:
    """Parse a commit line into (hash, type, scope, message).

    Handles conventional commits format: type(scope): message
    """
    parts = commit_line.split("|", 2)
    if len(parts) != 3:
        return ("", "other", "", commit_line)

    commit_hash, subject, author = parts

    # Parse conventional commit format
    commit_type = "other"
    scope = ""
    message = subject

    if ":" in subject:
        prefix, rest = subject.split(":", 1)
        message = rest.strip()

        # Extract type and optional scope
        if "(" in prefix and ")" in prefix:
            commit_type = prefix.split("(")[0].strip().lower()
            scope = prefix.split("(")[1].split(")")[0].strip()
        else:
            commit_type = prefix.strip().lower()

    return (commit_hash, commit_type, scope, message)


def categorize_commits(commits: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Categorize commits by type.

    Returns dict mapping category name to list of (hash, scope, message).
    """
    categories = defaultdict(list)

    type_to_category = {
        "feat": "Features",
        "fix": "Bug Fixes",
        "perf": "Performance",
        "docs": "Documentation",
        "refactor": "Refactoring",
        "test": "Testing",
        "ci": "CI/CD",
        "chore": "Maintenance",
        "build": "Build",
        "style": "Style",
    }

    for commit_line in commits:
        if not commit_line.strip():
            continue

        commit_hash, commit_type, scope, message = parse_commit(commit_line)

        category = type_to_category.get(commit_type, "Other")
        categories[category].append((commit_hash, scope, message))

    return dict(categories)


def generate_changelog(
    version: str,
    from_ref: Optional[str] = None,
    to_ref: str = "HEAD",
) -> str:
    """Generate changelog content.

    Args:
        version: Version string for the release (e.g., "v0.2.0")
        from_ref: Starting ref (tag/commit), defaults to previous tag
        to_ref: Ending ref, defaults to HEAD

    Returns:
        Formatted changelog as markdown string.
    """
    lines = []

    # Header
    lines.append(f"# Changelog for {version}")
    lines.append("")
    lines.append(f"**Release Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # Determine range
    if from_ref is None:
        from_ref = get_previous_tag(version) if version != "HEAD" else get_latest_tag()

    # Get commits
    commits = get_commits_between(from_ref, to_ref)

    if not commits:
        lines.append("No changes recorded.")
        return "\n".join(lines)

    # Categorize
    categories = categorize_commits(commits)

    # Priority order for categories
    category_order = [
        "Features",
        "Bug Fixes",
        "Performance",
        "Documentation",
        "Refactoring",
        "Testing",
        "CI/CD",
        "Build",
        "Maintenance",
        "Style",
        "Other",
    ]

    # Output categories
    for category in category_order:
        if category not in categories:
            continue

        commits_in_category = categories[category]
        if not commits_in_category:
            continue

        lines.append(f"## {category}")
        lines.append("")

        for commit_hash, scope, message in commits_in_category:
            if scope:
                lines.append(f"- **{scope}**: {message} ({commit_hash})")
            else:
                lines.append(f"- {message} ({commit_hash})")

        lines.append("")

    # Statistics
    total_commits = sum(len(c) for c in categories.values())
    lines.append("---")
    lines.append("")
    lines.append(f"**Total commits**: {total_commits}")
    if from_ref:
        lines.append(f"**Compare**: {from_ref}...{to_ref}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate changelog from git commit history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate changelog since last tag
    python scripts/generate_changelog.py

    # Generate changelog for specific version
    python scripts/generate_changelog.py v0.2.0

    # Generate changelog between two tags
    python scripts/generate_changelog.py v0.2.0 v0.1.0

    # Output to file
    python scripts/generate_changelog.py --output CHANGELOG.md
        """,
    )
    parser.add_argument(
        "version",
        nargs="?",
        default="HEAD",
        help="Version to generate changelog for (default: HEAD)",
    )
    parser.add_argument(
        "from_ref",
        nargs="?",
        default=None,
        help="Starting reference (default: previous tag)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    # Generate changelog
    changelog = generate_changelog(
        version=args.version,
        from_ref=args.from_ref,
        to_ref=args.version if args.version != "HEAD" else "HEAD",
    )

    # Output
    if args.output:
        args.output.write_text(changelog)
        print(f"Changelog written to {args.output}", file=sys.stderr)
    else:
        print(changelog)

    return 0


if __name__ == "__main__":
    sys.exit(main())
