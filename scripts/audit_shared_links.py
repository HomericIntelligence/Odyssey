#!/usr/bin/env python3
"""
Audit .claude/shared/ files for missing link-backs in CLAUDE.md

Verifies that every file in .claude/shared/ is listed in the Quick Links
section of CLAUDE.md. Exits non-zero if any shared file is missing.

Usage:
    python scripts/audit_shared_links.py [--claude-md PATH] [--shared-dir PATH]

Exit codes:
    0: All shared files are linked in CLAUDE.md Quick Links
    1: One or more shared files are missing from Quick Links
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

from hephaestus.utils import get_repo_root

# File suffixes to include in audit (extensible if new file types are added)
ALLOWED_SUFFIXES = {".md"}


def list_shared_files(shared_dir: Path, suffixes: Set[str] = None) -> List[str]:
    """
    List all shared documentation files in the shared directory.

    By default, only .md files are included. Use suffixes parameter to extend
    to other file types (e.g., {".md", ".yaml", ".sh"}).

    Args:
        shared_dir: Path to .claude/shared/ directory
        suffixes: Set of allowed file suffixes (defaults to {".md"})

    Returns:
        List of relative paths like '.claude/shared/foo.md'
    """
    if suffixes is None:
        suffixes = ALLOWED_SUFFIXES
    return sorted(f".claude/shared/{p.name}" for p in shared_dir.iterdir() if p.is_file() and p.suffix in suffixes)


def extract_quick_links_section(claude_md_content: str) -> str:
    """
    Extract the Quick Links section from CLAUDE.md content.

    Captures text from '## Quick Links' up to (but not including) the next
    level-2 heading.

    Args:
        claude_md_content: Full content of CLAUDE.md

    Returns:
        The Quick Links section as a string, or empty string if not found
    """
    match = re.search(
        r"^## Quick Links\b(.*?)(?=^## |\Z)",
        claude_md_content,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


def extract_linked_shared_paths(section: str) -> Set[str]:
    """
    Extract .claude/shared/... paths referenced as markdown links in a section.

    Handles both absolute links (/.claude/shared/foo.md) and relative links
    (.claude/shared/foo.md).

    Args:
        section: Markdown text to search

    Returns:
        Set of normalised paths like '.claude/shared/foo.md'
    """
    # Match markdown links: [text](url) or bare references in table cells.
    # Capture the filename, ignoring any trailing #anchor fragment.
    link_pattern = re.compile(r"\(/?\.claude/shared/([^)#\s]+)(?:#[^)]*)?\)")
    return {f".claude/shared/{m.group(1)}" for m in link_pattern.finditer(section)}


def audit(
    claude_md_content: str,
    shared_dir: Path,
) -> Tuple[List[str], List[str]]:
    """
    Audit shared files against CLAUDE.md Quick Links.

    Args:
        claude_md_content: Full text of CLAUDE.md
        shared_dir: Path to .claude/shared/ directory

    Returns:
        Tuple of (missing_files, present_files) where each element is a list
        of '.claude/shared/...' path strings
    """
    shared_files = list_shared_files(shared_dir)
    quick_links_section = extract_quick_links_section(claude_md_content)
    linked_paths = extract_linked_shared_paths(quick_links_section)

    missing = [f for f in shared_files if f not in linked_paths]
    present = [f for f in shared_files if f in linked_paths]
    return missing, present


def main(argv: List[str] | None = None) -> int:
    """
    Run the audit and print results.

    Returns:
        0 if all shared files are linked, 1 otherwise
    """
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = get_repo_root()
    parser.add_argument(
        "--claude-md",
        type=Path,
        default=repo_root / "CLAUDE.md",
        help="Path to CLAUDE.md (default: repo root)",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=repo_root / ".claude" / "shared",
        help="Path to .claude/shared/ directory (default: repo root)",
    )
    args = parser.parse_args(argv)

    claude_md_path: Path = args.claude_md
    shared_dir: Path = args.shared_dir

    if not claude_md_path.exists():
        print(f"ERROR: CLAUDE.md not found: {claude_md_path}", file=sys.stderr)
        return 1

    if not shared_dir.exists():
        print(f"ERROR: shared dir not found: {shared_dir}", file=sys.stderr)
        return 1

    content = claude_md_path.read_text(encoding="utf-8")
    missing, present = audit(content, shared_dir)

    if missing:
        print("AUDIT FAILED: The following .claude/shared/ files are not linked")
        print("in the Quick Links section of CLAUDE.md:\n")
        for path in missing:
            print(f"  - {path}")
        print(f"\n{len(missing)} file(s) missing, {len(present)} file(s) present.")
        print("\nAdd the missing files to the '### Core Guidelines' list in CLAUDE.md.")
        return 1

    print(f"AUDIT PASSED: All {len(present)} .claude/shared/ file(s) are linked in CLAUDE.md Quick Links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
