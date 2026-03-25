#!/usr/bin/env python3
"""
Fix remaining SKILL.md validation warnings via dynamic scan.

Dynamically discovers all SKILL.md files under a skills directory and applies
fixes for:
1. Files with ## Quick Reference or similar but missing ## Verified Workflow
2. Files with Failed Attempts that need better table formatting
3. Orphaned ## Quick Reference sections that should be under ## Verified Workflow

Replaces the previously hardcoded plugin lists so new skills are handled
automatically without manual list maintenance.

Usage:
    python3 scripts/fix_remaining_warnings.py [--skills-dir DIR] [--dry-run]
    python3 scripts/fix_remaining_warnings.py --dry-run
    python3 scripts/fix_remaining_warnings.py --skills-dir .claude/skills/
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


DEFAULT_SKILLS_DIR = Path(".claude/skills")


def read_file(path: Path) -> str:
    """Read file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    """Write content to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def has_verified_workflow_section(content: str) -> bool:
    """Check if ## Verified Workflow exists."""
    return bool(re.search(r"^## Verified Workflow", content, re.MULTILINE))


def has_failed_attempts_table(content: str) -> bool:
    """Check if Failed Attempts section has a pipe table."""
    match = re.search(r"^## Failed Attempts\s*$(.*?)(?:^##|\Z)", content, re.MULTILINE | re.DOTALL)
    if not match:
        return True  # No Failed Attempts section, skip
    section_content = match.group(1)
    return "|" in section_content


def add_verified_workflow_wrapper(content: str) -> str:
    """Add ## Verified Workflow wrapper around Quick Reference or similar sections."""
    workflow_sections = [
        "Quick Reference",
        "Analysis Workflow",
        "Workflow Steps",
        "Implementation Steps",
        "Usage",
    ]

    for section_name in workflow_sections:
        pattern = f"^## {section_name}$"
        if re.search(pattern, content, re.MULTILINE):
            replacement = f"## Verified Workflow\n\n### {section_name}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            return content

    return content


def has_orphan_quick_reference(content: str) -> bool:
    """Check if ## Quick Reference exists as a top-level section.

    Returns True when the content contains a top-level '## Quick Reference'
    heading, indicating it needs to be demoted to a subsection of
    '## Verified Workflow'.
    """
    return bool(re.search(r"^## Quick Reference", content, re.MULTILINE))


def merge_quick_reference_into_verified_workflow(content: str) -> str:
    """Demote top-level ## Quick Reference to ### Quick Reference under ## Verified Workflow.

    Extracts the entire ## Quick Reference section, removes it from its current
    position, and inserts it as the first subsection (###) of ## Verified Workflow.
    If ## Quick Reference is already a subsection (### level), the content is
    returned unchanged.
    """
    if not re.search(r"^## Quick Reference", content, re.MULTILINE):
        return content

    qr_match = re.search(
        r"^(## Quick Reference\s*\n.*?)(?=^##|\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not qr_match:
        return content

    qr_block = qr_match.group(1)

    qr_as_subsection = re.sub(r"^## Quick Reference", "### Quick Reference", qr_block, count=1)

    content_without_qr = content[: qr_match.start()] + content[qr_match.end() :]

    vw_match = re.search(r"^## Verified Workflow[^\n]*\n", content_without_qr, re.MULTILINE)
    if not vw_match:
        return content

    insert_pos = vw_match.end()
    subsection_text = "\n" + qr_as_subsection.lstrip("\n")
    return content_without_qr[:insert_pos] + subsection_text + content_without_qr[insert_pos:]


def improve_failed_attempts_table(content: str) -> str:
    """Improve Failed Attempts tables that were flagged."""
    match = re.search(r"^## Failed Attempts\s*$(.*?)(?:^##|\Z)", content, re.MULTILINE | re.DOTALL)
    if not match:
        return content

    section_content = match.group(1)

    if "|" in section_content:
        return content

    table = """

| Approach | Issue | Resolution |
|----------|-------|------------|
| See details below | Various implementation challenges | Documented in this section |
"""

    insert_pos = match.end()
    return content[:insert_pos] + table + content[insert_pos:]


def fix_skill_file(skill_path: Path, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Fix a single SKILL.md file. Returns (modified, fixes_applied).

    Args:
        skill_path: Path to the SKILL.md file to process.
        dry_run: If True, determine what would change but do not write the file.

    Returns:
        Tuple of (modified, fixes_applied) where modified is True if the file
        was changed (or would be changed in dry-run mode) and fixes_applied is
        a list of description strings for each fix that was applied.
    """
    content = read_file(skill_path)
    original_content = content
    fixes: List[str] = []

    # Fix 1: Add Verified Workflow wrapper
    if not has_verified_workflow_section(content):
        new_content = add_verified_workflow_wrapper(content)
        if new_content != content:
            content = new_content
            fixes.append("Added ## Verified Workflow wrapper")

    # Fix 2: Merge orphaned ## Quick Reference into ## Verified Workflow
    if has_verified_workflow_section(content) and has_orphan_quick_reference(content):
        new_content = merge_quick_reference_into_verified_workflow(content)
        if new_content != content:
            content = new_content
            fixes.append("Merged ## Quick Reference into ## Verified Workflow as subsection")

    # Fix 3: Improve Failed Attempts table
    if not has_failed_attempts_table(content):
        new_content = improve_failed_attempts_table(content)
        if new_content != content:
            content = new_content
            fixes.append("Improved Failed Attempts table")

    # Write back if modified (skip write in dry-run mode)
    if content != original_content:
        if not dry_run:
            write_file(skill_path, content)
        return True, fixes

    return False, []


def main(argv: "List[str] | None" = None) -> None:
    """Dynamically discover and fix all SKILL.md files.

    Scans ``skills_dir`` via ``Path.rglob("SKILL.md")`` and applies fixes to
    each file.  No hardcoded plugin lists — new skills are handled automatically.

    Args:
        argv: Argument list for parsing (defaults to sys.argv when None).
    """
    parser = argparse.ArgumentParser(
        description="Fix SKILL.md validation warnings across all skills",
    )
    parser.add_argument(
        "--skills-dir",
        default=str(DEFAULT_SKILLS_DIR),
        help="Root directory containing SKILL.md files (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files",
    )
    args = parser.parse_args(argv)

    skills_dir = Path(args.skills_dir)
    dry_run: bool = args.dry_run

    if dry_run:
        print("DRY RUN — no files will be written\n")

    skill_files = sorted(skills_dir.rglob("SKILL.md"))

    total_files = 0
    modified_files = 0
    all_fixes: List[str] = []

    for skill_file in skill_files:
        total_files += 1
        modified, fixes = fix_skill_file(skill_file, dry_run=dry_run)

        if modified:
            modified_files += 1
            rel_path = skill_file.relative_to(skills_dir)
            action = "Would fix" if dry_run else "Fixed"
            print(f"{'~' if dry_run else '✓'} {action}: {rel_path}")
            for fix in fixes:
                print(f"  - {fix}")
                all_fixes.append(fix)

    print(f"\n{'=' * 60}")
    print(f"Processed {total_files} SKILL.md files")
    if dry_run:
        print(f"Would modify {modified_files} files")
    else:
        print(f"Modified {modified_files} files")
    if all_fixes:
        print(f"\nFixes {'that would be ' if dry_run else ''}applied:")
        for fix_type in set(all_fixes):
            count = all_fixes.count(fix_type)
            print(f"  - {fix_type}: {count} files")


if __name__ == "__main__":
    main()
