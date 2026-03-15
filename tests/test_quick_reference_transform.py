#!/usr/bin/env python3
"""
Integration tests for Quick Reference section handling using real SKILL.md files.

Guards against regressions on real-world file shapes by testing the full
before→fix→after cycle with the actual worktree-create SKILL.md.

Follow-up from #3230.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _find_scripts_dir() -> Path:
    """Locate build/ProjectMnemosyne/scripts/ relative to the git repository root.

    In a worktree, Path(__file__).parent.parent is the worktree root, not the
    main repo root. We resolve the main repo root via `git rev-parse --git-common-dir`
    so that the build/ directory (which lives only in the main repo) is found
    regardless of which worktree this test runs from.
    """
    worktree_root = Path(__file__).parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=worktree_root,
            capture_output=True,
            text=True,
            check=True,
        )
        git_common_dir = Path(result.stdout.strip())
        # git-common-dir is <main-repo>/.git; main repo root is its parent
        main_repo_root = git_common_dir.parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: assume no worktree nesting (standard checkout)
        main_repo_root = worktree_root

    candidate = main_repo_root / "build" / "ProjectMnemosyne" / "scripts"
    if candidate.is_dir():
        return candidate

    # Last resort: use local path (handles non-worktree checkouts)
    return worktree_root / "build" / "ProjectMnemosyne" / "scripts"


sys.path.insert(0, str(_find_scripts_dir()))

from fix_remaining_warnings import (
    fix_skill_file,
    has_orphan_quick_reference,
)

# Path to the real worktree-create SKILL.md used as the integration fixture
WORKTREE_CREATE_SKILL = (
    Path(__file__).parent.parent
    / ".claude"
    / "skills"
    / "worktree-create"
    / "SKILL.md"
)


class TestRealWorktreeCreateIntegration:
    """Integration tests using the real worktree-create SKILL.md as a fixture.

    These tests verify that the fix functions handle real-world file shapes
    correctly, complementing the synthetic-fixture unit tests.
    """

    def test_fixture_file_exists(self) -> None:
        """The real SKILL.md fixture file must exist for all other tests to be valid."""
        assert WORKTREE_CREATE_SKILL.exists(), (
            f"Fixture file not found: {WORKTREE_CREATE_SKILL}\n"
            "This file is required for integration tests."
        )

    def test_real_file_has_orphan_quick_reference(self) -> None:
        """has_orphan_quick_reference() returns True on the unmodified real file.

        The worktree-create SKILL.md has '## Quick Reference' as a top-level
        section (not nested under '## Verified Workflow'), so it is an orphan
        that needs to be wrapped or merged.
        """
        content = WORKTREE_CREATE_SKILL.read_text(encoding="utf-8")
        assert has_orphan_quick_reference(content) is True, (
            "Expected the real worktree-create SKILL.md to have a top-level "
            "'## Quick Reference' section, but has_orphan_quick_reference() returned False."
        )

    def test_fix_skill_file_returns_modified_true(self, tmp_path: Path) -> None:
        """fix_skill_file() returns modified=True when applied to the real file."""
        target = tmp_path / "SKILL.md"
        shutil.copy2(WORKTREE_CREATE_SKILL, target)

        modified, fixes = fix_skill_file(target)

        assert modified is True, (
            f"fix_skill_file() returned modified=False; fixes applied: {fixes}"
        )
        assert len(fixes) > 0, "Expected at least one fix to be reported."

    def test_fix_skill_file_removes_orphan_quick_reference(self, tmp_path: Path) -> None:
        """has_orphan_quick_reference() returns False after fix_skill_file() runs."""
        target = tmp_path / "SKILL.md"
        shutil.copy2(WORKTREE_CREATE_SKILL, target)

        fix_skill_file(target)

        result_content = target.read_text(encoding="utf-8")
        assert has_orphan_quick_reference(result_content) is False, (
            "After fix_skill_file(), '## Quick Reference' should no longer appear "
            "as a top-level heading, but has_orphan_quick_reference() still returns True."
        )

    def test_round_trip_no_data_loss(self, tmp_path: Path) -> None:
        """All significant content from the original file survives the fix.

        Verifies that fix_skill_file() only restructures headings without
        dropping any meaningful text content.
        """
        original_content = WORKTREE_CREATE_SKILL.read_text(encoding="utf-8")

        target = tmp_path / "SKILL.md"
        shutil.copy2(WORKTREE_CREATE_SKILL, target)
        fix_skill_file(target)
        result_content = target.read_text(encoding="utf-8")

        # The file must actually have been changed (not a no-op)
        assert result_content != original_content, (
            "fix_skill_file() produced no change; the test fixture may already be fixed."
        )

        # Key content from the bash block in ## Quick Reference must survive
        bash_snippets = [
            "create_worktree.sh",
            "git worktree list",
            "git worktree add",
        ]
        for snippet in bash_snippets:
            assert snippet in result_content, (
                f"Content lost after fix: '{snippet}' not found in fixed file."
            )

        # Section names that should still be present
        section_names = [
            "When to Use",
            "Workflow",
            "Error Handling",
        ]
        for section in section_names:
            assert section in result_content, (
                f"Section lost after fix: '{section}' not found in fixed file."
            )

    def test_fix_skill_file_is_idempotent(self, tmp_path: Path) -> None:
        """Running fix_skill_file() twice produces no second modification.

        After the first fix the file should be in a valid state, so a second
        pass must return modified=False and leave the content unchanged.
        """
        target = tmp_path / "SKILL.md"
        shutil.copy2(WORKTREE_CREATE_SKILL, target)

        # First pass — should modify
        fix_skill_file(target)
        content_after_first = target.read_text(encoding="utf-8")

        # Second pass — must be a no-op
        modified, fixes = fix_skill_file(target)

        assert modified is False, (
            f"fix_skill_file() should be idempotent but returned modified=True "
            f"on the second pass; fixes: {fixes}"
        )
        assert target.read_text(encoding="utf-8") == content_after_first, (
            "File content changed on second fix_skill_file() call (not idempotent)."
        )
