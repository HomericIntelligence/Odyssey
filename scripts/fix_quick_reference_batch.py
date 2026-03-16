#!/usr/bin/env python3
"""
Bulk-fix SKILL.md files with orphaned top-level ## Quick Reference sections.

A "Quick Reference should be under Verified Workflow" warning is emitted when a
SKILL.md has **both** a top-level ``## Quick Reference`` heading *and* a
``## Verified Workflow`` heading.  The correct structure is to demote
``## Quick Reference`` to ``### Quick Reference`` as the first subsection inside
``## Verified Workflow``.

Usage:
    python3 scripts/fix_quick_reference_batch.py [skills_dir] [--dry-run]
    python3 scripts/fix_quick_reference_batch.py .claude/skills/
    python3 scripts/fix_quick_reference_batch.py skills/ --dry-run
    python3 scripts/fix_quick_reference_batch.py  # defaults to .claude/skills/
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


DEFAULT_SKILLS_DIR = Path(".claude/skills")


def has_orphan_quick_reference(content: str) -> bool:
    """Return True if content has a top-level ``## Quick Reference`` heading.

    Args:
        content: Full text of the SKILL.md file.

    Returns:
        True when a top-level h2 Quick Reference section exists.
    """
    return bool(re.search(r"^## Quick Reference", content, re.MULTILINE))


def has_verified_workflow(content: str) -> bool:
    """Return True if content has a ``## Verified Workflow`` section.

    Args:
        content: Full text of the SKILL.md file.

    Returns:
        True when a top-level h2 Verified Workflow section exists.
    """
    return bool(re.search(r"^## Verified Workflow", content, re.MULTILINE))


def collect_affected_files(skills_dir: Path) -> List[Path]:
    """Find all SKILL.md files that have both Quick Reference and Verified Workflow.

    Scans *skills_dir* recursively for SKILL.md files where both
    ``## Quick Reference`` and ``## Verified Workflow`` appear as top-level
    headings.

    Args:
        skills_dir: Root directory to scan for SKILL.md files.

    Returns:
        Sorted list of Path objects for affected SKILL.md files.
    """
    affected: List[Path] = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        content = skill_md.read_text(encoding="utf-8")
        if has_orphan_quick_reference(content) and has_verified_workflow(content):
            affected.append(skill_md)
    return affected


def merge_quick_reference_into_verified_workflow(content: str) -> str:
    """Demote top-level ``## Quick Reference`` to ``### Quick Reference`` under ``## Verified Workflow``.

    Extracts the entire ``## Quick Reference`` section (from its heading up to
    the next ``##``-level heading or EOF), demotes the heading from ``##`` to
    ``###``, removes it from its original position, and inserts it immediately
    after the ``## Verified Workflow`` heading line.

    If the content does not contain a top-level ``## Quick Reference`` or
    ``## Verified Workflow``, the content is returned unchanged.

    Args:
        content: Full text of the SKILL.md file to transform.

    Returns:
        Transformed content with Quick Reference as a subsection, or the
        original content if the preconditions are not met.
    """
    if not re.search(r"^## Quick Reference", content, re.MULTILINE):
        return content

    # Extract the full ## Quick Reference block (up to next ## or end of file)
    qr_match = re.search(
        r"^(## Quick Reference\s*\n.*?)(?=^##|\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not qr_match:
        return content

    qr_block = qr_match.group(1)

    # Demote heading from ## to ###
    qr_as_subsection = re.sub(r"^## Quick Reference", "### Quick Reference", qr_block, count=1)

    # Remove the original top-level block
    content_without_qr = content[: qr_match.start()] + content[qr_match.end() :]

    # Insert the demoted block immediately after the ## Verified Workflow heading line
    vw_match = re.search(r"^## Verified Workflow[^\n]*\n", content_without_qr, re.MULTILINE)
    if not vw_match:
        # No Verified Workflow section — return unchanged
        return content

    insert_pos = vw_match.end()
    subsection_text = "\n" + qr_as_subsection.lstrip("\n")
    return content_without_qr[:insert_pos] + subsection_text + content_without_qr[insert_pos:]


def fix_skill_file(skill_path: Path) -> Tuple[bool, str]:
    """Apply Quick Reference merge fix to a single SKILL.md file.

    Reads the file, applies ``merge_quick_reference_into_verified_workflow``,
    and writes the result back if the content changed.

    Args:
        skill_path: Path to the SKILL.md file to fix.

    Returns:
        Tuple of (was_modified, description) where *was_modified* is True when
        the file was changed and *description* briefly describes what happened.
    """
    content = skill_path.read_text(encoding="utf-8")
    new_content = merge_quick_reference_into_verified_workflow(content)
    if new_content == content:
        return False, "no change needed"
    skill_path.write_text(new_content, encoding="utf-8")
    return True, "merged ## Quick Reference into ## Verified Workflow"


def run_batch_fix(
    skills_dir: Path,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Fix all affected SKILL.md files under *skills_dir*.

    Discovers files with orphaned ``## Quick Reference`` sections, applies the
    merge transformation, and reports results.

    Args:
        skills_dir: Root directory to scan.
        dry_run: When True, report affected files without modifying them.

    Returns:
        Tuple of (total_scanned, fixed, skipped) counts.
    """
    if not skills_dir.exists():
        print(f"ERROR: skills directory not found: {skills_dir}", file=sys.stderr)
        return 0, 0, 0

    affected = collect_affected_files(skills_dir)
    total_skill_mds = sum(1 for _ in skills_dir.rglob("SKILL.md"))

    print(f"Scanned {total_skill_mds} SKILL.md files under {skills_dir}")
    print(f"Found {len(affected)} file(s) with orphaned ## Quick Reference")

    if not affected:
        print("Nothing to fix.")
        return total_skill_mds, 0, 0

    fixed = 0
    skipped = 0

    for path in affected:
        rel = path.relative_to(skills_dir.parent) if skills_dir.parent else path
        if dry_run:
            print(f"  [DRY RUN] Would fix: {rel}")
            skipped += 1
        else:
            modified, desc = fix_skill_file(path)
            if modified:
                print(f"  FIXED: {rel} — {desc}")
                fixed += 1
            else:
                print(f"  SKIP:  {rel} — {desc}")
                skipped += 1

    return total_skill_mds, fixed, skipped


def verify_no_warnings(skills_dir: Path) -> int:
    """Return the count of SKILL.md files that still have the orphaned pattern.

    Performs a second-pass check after fixes have been applied to confirm that
    no files still emit the Quick Reference warning.

    Args:
        skills_dir: Root directory to scan.

    Returns:
        Number of files that still have the problem (0 means all clear).
    """
    remaining = collect_affected_files(skills_dir)
    return len(remaining)


def main() -> int:
    """Entry point for the batch fix script.

    Returns:
        Exit code: 0 on success, 1 if any files remain unfixed after the run.
    """
    parser = argparse.ArgumentParser(description="Bulk-fix orphaned ## Quick Reference sections in SKILL.md files")
    parser.add_argument(
        "skills_dir",
        nargs="?",
        default=str(DEFAULT_SKILLS_DIR),
        help=f"Root skills directory to scan (default: {DEFAULT_SKILLS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report affected files without modifying them",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    total, fixed, skipped = run_batch_fix(skills_dir, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  SKILL.md files scanned: {total}")
    if args.dry_run:
        print(f"  Would fix:              {skipped}")
        print("[DRY RUN] No files were modified.")
    else:
        print(f"  Fixed:                  {fixed}")
        print(f"  Already clean:          {skipped}")

        # Second-pass validation
        remaining = verify_no_warnings(skills_dir)
        if remaining == 0:
            print("  Validation:             PASS — no warnings remain")
        else:
            print(f"  Validation:             FAIL — {remaining} file(s) still have the issue")
            return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
