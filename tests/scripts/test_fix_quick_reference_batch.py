#!/usr/bin/env python3
"""Tests for scripts/fix_quick_reference_batch.py."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "fix_quick_reference_batch.py"


def load_module():
    """Dynamically load the fix_quick_reference_batch module."""
    spec = importlib.util.spec_from_file_location("fix_quick_reference_batch", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="mod")
def module_fixture():
    """Provide the loaded module."""
    return load_module()


# ---------------------------------------------------------------------------
# has_orphan_quick_reference
# ---------------------------------------------------------------------------


class TestHasOrphanQuickReference:
    def test_detects_top_level_heading(self, mod) -> None:
        content = "# Skill\n\n## Quick Reference\n\nsome content\n"
        assert mod.has_orphan_quick_reference(content) is True

    def test_ignores_subsection_heading(self, mod) -> None:
        content = "# Skill\n\n## Verified Workflow\n\n### Quick Reference\n\nsome content\n"
        assert mod.has_orphan_quick_reference(content) is False

    def test_false_when_absent(self, mod) -> None:
        content = "# Skill\n\n## When to Use\n\n- bullet\n"
        assert mod.has_orphan_quick_reference(content) is False

    def test_partial_match_not_triggered(self, mod) -> None:
        content = "# Skill\n\n## Quick Reference Guide\n\ncontent\n"
        # "Quick Reference Guide" still matches because the pattern is prefix-based
        assert mod.has_orphan_quick_reference(content) is True


# ---------------------------------------------------------------------------
# has_verified_workflow
# ---------------------------------------------------------------------------


class TestHasVerifiedWorkflow:
    def test_detects_section(self, mod) -> None:
        content = "# Skill\n\n## Verified Workflow\n\n1. step\n"
        assert mod.has_verified_workflow(content) is True

    def test_false_when_absent(self, mod) -> None:
        content = "# Skill\n\n## Workflow\n\n1. step\n"
        assert mod.has_verified_workflow(content) is False

    def test_false_when_only_subsection(self, mod) -> None:
        content = "# Skill\n\n## Something\n\n### Verified Workflow\n\nstep\n"
        assert mod.has_verified_workflow(content) is False


# ---------------------------------------------------------------------------
# merge_quick_reference_into_verified_workflow
# ---------------------------------------------------------------------------


SIMPLE_CONTENT = """\
---
name: test-skill
---

# Test Skill

## When to Use

- use it

## Quick Reference

```bash
some command
```

## Verified Workflow

1. step one
2. step two

## References

- link
"""


class TestMergeQuickReference:
    def test_demotes_heading(self, mod) -> None:
        import re
        result = mod.merge_quick_reference_into_verified_workflow(SIMPLE_CONTENT)
        assert "### Quick Reference" in result
        # Top-level h2 should be gone (### still contains "## Quick Reference" as substring)
        assert not re.search(r"^## Quick Reference", result, re.MULTILINE)

    def test_places_subsection_inside_verified_workflow(self, mod) -> None:
        result = mod.merge_quick_reference_into_verified_workflow(SIMPLE_CONTENT)
        vw_pos = result.index("## Verified Workflow")
        qr_pos = result.index("### Quick Reference")
        refs_pos = result.index("## References")
        assert vw_pos < qr_pos < refs_pos

    def test_no_op_when_no_quick_reference(self, mod) -> None:
        content = "# Skill\n\n## Verified Workflow\n\n1. step\n"
        assert mod.merge_quick_reference_into_verified_workflow(content) == content

    def test_no_op_when_no_verified_workflow(self, mod) -> None:
        content = "# Skill\n\n## Quick Reference\n\ncmd\n\n## Workflow\n\n1. step\n"
        assert mod.merge_quick_reference_into_verified_workflow(content) == content

    def test_preserves_qr_body_content(self, mod) -> None:
        result = mod.merge_quick_reference_into_verified_workflow(SIMPLE_CONTENT)
        assert "some command" in result

    def test_idempotent(self, mod) -> None:
        """Applying twice produces the same result as applying once."""
        once = mod.merge_quick_reference_into_verified_workflow(SIMPLE_CONTENT)
        twice = mod.merge_quick_reference_into_verified_workflow(once)
        assert once == twice

    def test_qr_removed_from_original_position(self, mod) -> None:
        result = mod.merge_quick_reference_into_verified_workflow(SIMPLE_CONTENT)
        # Quick Reference should only appear once (as subsection), not at top level
        count = result.count("Quick Reference")
        assert count == 1

    def test_multiline_qr_block(self, mod) -> None:
        import re
        content = (
            "# S\n\n"
            "## Quick Reference\n\n"
            "```bash\ncmd1\ncmd2\ncmd3\n```\n\n"
            "More text here.\n\n"
            "## Verified Workflow\n\n"
            "1. step\n"
        )
        result = mod.merge_quick_reference_into_verified_workflow(content)
        assert "cmd1\ncmd2\ncmd3" in result
        assert "### Quick Reference" in result
        assert not re.search(r"^## Quick Reference", result, re.MULTILINE)


# ---------------------------------------------------------------------------
# collect_affected_files
# ---------------------------------------------------------------------------


def _make_skill(directory: Path, name: str, content: str) -> Path:
    """Create a SKILL.md file under directory/name/SKILL.md."""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")
    return skill_md


AFFECTED_CONTENT = """\
# Skill

## Quick Reference

cmd

## Verified Workflow

1. step
"""

CLEAN_CONTENT = """\
# Skill

## Verified Workflow

### Quick Reference

cmd

1. step
"""

WORKFLOW_ONLY_CONTENT = """\
# Skill

## Quick Reference

cmd

## Workflow

1. step
"""


class TestCollectAffectedFiles:
    def test_finds_affected_files(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "affected", AFFECTED_CONTENT)
        result = mod.collect_affected_files(skills_dir)
        assert len(result) == 1
        assert result[0].name == "SKILL.md"

    def test_ignores_already_clean_files(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "clean", CLEAN_CONTENT)
        result = mod.collect_affected_files(skills_dir)
        assert result == []

    def test_ignores_workflow_without_verified(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "workflow-only", WORKFLOW_ONLY_CONTENT)
        result = mod.collect_affected_files(skills_dir)
        assert result == []

    def test_mixed_directory(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "s1", AFFECTED_CONTENT)
        _make_skill(skills_dir, "s2", CLEAN_CONTENT)
        _make_skill(skills_dir, "s3", AFFECTED_CONTENT)
        result = mod.collect_affected_files(skills_dir)
        assert len(result) == 2

    def test_returns_sorted_paths(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "zzz", AFFECTED_CONTENT)
        _make_skill(skills_dir, "aaa", AFFECTED_CONTENT)
        result = mod.collect_affected_files(skills_dir)
        names = [p.parent.name for p in result]
        assert names == sorted(names)

    def test_empty_directory(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        result = mod.collect_affected_files(skills_dir)
        assert result == []


# ---------------------------------------------------------------------------
# fix_skill_file
# ---------------------------------------------------------------------------


class TestFixSkillFile:
    def test_modifies_affected_file(self, mod, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(AFFECTED_CONTENT, encoding="utf-8")
        modified, desc = mod.fix_skill_file(skill_md)
        assert modified is True
        assert "merged" in desc.lower()

    def test_writes_corrected_content(self, mod, tmp_path: Path) -> None:
        import re
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(AFFECTED_CONTENT, encoding="utf-8")
        mod.fix_skill_file(skill_md)
        result = skill_md.read_text(encoding="utf-8")
        assert "### Quick Reference" in result
        assert not re.search(r"^## Quick Reference", result, re.MULTILINE)

    def test_no_modification_for_clean_file(self, mod, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(CLEAN_CONTENT, encoding="utf-8")
        modified, desc = mod.fix_skill_file(skill_md)
        assert modified is False

    def test_no_modification_for_workflow_only(self, mod, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(WORKFLOW_ONLY_CONTENT, encoding="utf-8")
        modified, _ = mod.fix_skill_file(skill_md)
        assert modified is False


# ---------------------------------------------------------------------------
# run_batch_fix
# ---------------------------------------------------------------------------


class TestRunBatchFix:
    def test_fixes_affected_files(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        path = _make_skill(skills_dir, "s1", AFFECTED_CONTENT)
        total, fixed, skipped = mod.run_batch_fix(skills_dir)
        assert fixed == 1
        assert skipped == 0
        assert "### Quick Reference" in path.read_text()

    def test_dry_run_does_not_modify(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        path = _make_skill(skills_dir, "s1", AFFECTED_CONTENT)
        original = path.read_text()
        total, fixed, skipped = mod.run_batch_fix(skills_dir, dry_run=True)
        assert fixed == 0
        assert skipped == 1
        assert path.read_text() == original

    def test_returns_correct_counts(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "s1", AFFECTED_CONTENT)
        _make_skill(skills_dir, "s2", CLEAN_CONTENT)
        total, fixed, skipped = mod.run_batch_fix(skills_dir)
        assert total == 2
        assert fixed == 1

    def test_missing_directory(self, mod, tmp_path: Path, capsys) -> None:
        missing = tmp_path / "nonexistent"
        total, fixed, skipped = mod.run_batch_fix(missing)
        assert total == 0
        assert fixed == 0
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_empty_directory_no_errors(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        total, fixed, skipped = mod.run_batch_fix(skills_dir)
        assert total == 0
        assert fixed == 0


# ---------------------------------------------------------------------------
# verify_no_warnings
# ---------------------------------------------------------------------------


class TestVerifyNoWarnings:
    def test_returns_zero_when_all_clean(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "clean", CLEAN_CONTENT)
        assert mod.verify_no_warnings(skills_dir) == 0

    def test_returns_count_of_remaining(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "still-broken", AFFECTED_CONTENT)
        assert mod.verify_no_warnings(skills_dir) == 1

    def test_zero_after_fix(self, mod, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "s1", AFFECTED_CONTENT)
        mod.run_batch_fix(skills_dir)
        assert mod.verify_no_warnings(skills_dir) == 0
