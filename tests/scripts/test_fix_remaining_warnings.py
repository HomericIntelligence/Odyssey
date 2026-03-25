#!/usr/bin/env python3
"""Tests for scripts/fix_remaining_warnings.py.

Verifies that:
- main() discovers all SKILL.md files dynamically via rglob (no hardcoded lists)
- --dry-run reports what would change without writing any files
- Empty skills directory is handled gracefully
- Files without warnings are left untouched
- fix_skill_file() respects the dry_run parameter
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "fix_remaining_warnings.py"


def load_module() -> Any:
    """Dynamically load the fix_remaining_warnings module."""
    spec = importlib.util.spec_from_file_location("fix_remaining_warnings", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="mod")
def module_fixture() -> Any:
    """Provide the loaded module."""
    return load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLEAN_SKILL_MD = """\
---
name: test-skill
description: "A test skill."
category: tooling
date: 2026-01-01
user-invocable: false
---

# Test Skill

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-01-01 |
| Objective | Test |
| Outcome | Pass |

## When to Use

- When testing

## Verified Workflow

### Step 1

Do the thing.

## Failed Attempts

| Attempt | Why Failed | Lesson |
|---------|------------|--------|
| N/A | No failures | Document as they occur |

## Results & Parameters

N/A
"""

NEEDS_WORKFLOW_FIX = """\
---
name: needs-fix-skill
description: "A skill missing ## Verified Workflow."
category: tooling
date: 2026-01-01
user-invocable: false
---

# Needs Fix Skill

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-01-01 |
| Objective | Test |
| Outcome | Fix |

## When to Use

- When testing dynamic scan

## Quick Reference

```bash
git status
```

## Failed Attempts

| Attempt | Why Failed | Lesson |
|---------|------------|--------|
| N/A | No failures | N/A |

## Results & Parameters

N/A
"""

NEEDS_TABLE_FIX = """\
---
name: needs-table-skill
description: "A skill with Failed Attempts missing a pipe table."
category: tooling
date: 2026-01-01
user-invocable: false
---

# Needs Table Skill

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-01-01 |
| Objective | Test |
| Outcome | Fix |

## When to Use

- When testing

## Verified Workflow

### Step 1

Do the thing.

## Failed Attempts

No failures recorded.

## Results & Parameters

N/A
"""


def make_skill_file(directory: Path, content: str) -> Path:
    """Write a SKILL.md into *directory* and return its path."""
    skill_file = directory / "SKILL.md"
    skill_file.write_text(content)
    return skill_file


# ---------------------------------------------------------------------------
# Tests: fix_skill_file() dry_run parameter
# ---------------------------------------------------------------------------


class TestFixSkillFileDryRun:
    def test_dry_run_does_not_write_file(self, tmp_path: Path, mod: Any) -> None:
        """File must remain unchanged when dry_run=True even if fixes apply."""
        skill_file = make_skill_file(tmp_path, NEEDS_WORKFLOW_FIX)
        original_content = skill_file.read_text()

        modified, fixes = mod.fix_skill_file(skill_file, dry_run=True)

        assert modified is True, "Expected fix to be detected"
        assert fixes, "Expected at least one fix description"
        assert skill_file.read_text() == original_content, "File must not be written in dry-run mode"

    def test_dry_run_returns_same_fixes_as_live_run(self, tmp_path: Path, mod: Any) -> None:
        """dry_run mode should report the same fixes as a live run."""
        content = NEEDS_WORKFLOW_FIX

        dry_file = tmp_path / "dry" / "SKILL.md"
        dry_file.parent.mkdir()
        dry_file.write_text(content)

        live_file = tmp_path / "live" / "SKILL.md"
        live_file.parent.mkdir()
        live_file.write_text(content)

        _, dry_fixes = mod.fix_skill_file(dry_file, dry_run=True)
        _, live_fixes = mod.fix_skill_file(live_file, dry_run=False)

        assert dry_fixes == live_fixes

    def test_live_run_writes_file(self, tmp_path: Path, mod: Any) -> None:
        """File must be written when dry_run=False (default)."""
        skill_file = make_skill_file(tmp_path, NEEDS_WORKFLOW_FIX)
        original_content = skill_file.read_text()

        modified, fixes = mod.fix_skill_file(skill_file, dry_run=False)

        assert modified is True
        assert skill_file.read_text() != original_content

    def test_dry_run_no_change_when_file_is_clean(self, tmp_path: Path, mod: Any) -> None:
        """A clean file reports no modifications in dry-run mode."""
        skill_file = make_skill_file(tmp_path, CLEAN_SKILL_MD)

        modified, fixes = mod.fix_skill_file(skill_file, dry_run=True)

        assert modified is False
        assert fixes == []
        assert skill_file.read_text() == CLEAN_SKILL_MD

    def test_dry_run_table_fix_detected_but_not_written(self, tmp_path: Path, mod: Any) -> None:
        """Table fix is reported but the file is not modified in dry-run mode."""
        skill_file = make_skill_file(tmp_path, NEEDS_TABLE_FIX)
        original_content = skill_file.read_text()

        modified, fixes = mod.fix_skill_file(skill_file, dry_run=True)

        assert modified is True
        assert any("Failed Attempts" in fix or "table" in fix.lower() for fix in fixes)
        assert skill_file.read_text() == original_content


# ---------------------------------------------------------------------------
# Tests: main() dynamic discovery
# ---------------------------------------------------------------------------


class TestMainDynamicDiscovery:
    @staticmethod
    def _build_skills_tree(root: Path) -> list:
        """Create a nested skills directory tree with several SKILL.md files."""
        files = []
        for category, skill_name, content in [
            ("tooling", "skill-clean", CLEAN_SKILL_MD),
            ("tooling", "skill-needs-fix", NEEDS_WORKFLOW_FIX),
            ("testing", "skill-needs-table", NEEDS_TABLE_FIX),
            ("documentation", "skill-also-clean", CLEAN_SKILL_MD),
        ]:
            skill_dir = root / category / skill_name
            skill_dir.mkdir(parents=True)
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(content)
            files.append(skill_file)
        return files

    def test_discovers_all_skill_files_dynamically(self, tmp_path: Path, mod: Any) -> None:
        """main() must process every SKILL.md in the tree without a hardcoded list."""
        self._build_skills_tree(tmp_path)

        mod.main(["--skills-dir", str(tmp_path)])

        needs_fix_path = tmp_path / "tooling" / "skill-needs-fix" / "SKILL.md"
        assert "## Verified Workflow" in needs_fix_path.read_text()

    def test_dry_run_does_not_modify_any_files(self, tmp_path: Path, mod: Any) -> None:
        """--dry-run must leave every discovered file untouched."""
        self._build_skills_tree(tmp_path)

        before = {p: p.read_text() for p in tmp_path.rglob("SKILL.md")}

        mod.main(["--skills-dir", str(tmp_path), "--dry-run"])

        after = {p: p.read_text() for p in tmp_path.rglob("SKILL.md")}

        assert before == after, "No files should be modified during dry-run"

    def test_empty_skills_dir_handled_gracefully(self, tmp_path: Path, mod: Any) -> None:
        """main() must not raise when there are no SKILL.md files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        mod.main(["--skills-dir", str(empty_dir)])

    def test_processes_files_in_subdirectories(self, tmp_path: Path, mod: Any) -> None:
        """main() discovers SKILL.md files nested arbitrarily deep."""
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        skill_file = deep_dir / "SKILL.md"
        skill_file.write_text(NEEDS_WORKFLOW_FIX)

        mod.main(["--skills-dir", str(tmp_path)])

        assert "## Verified Workflow" in skill_file.read_text()

    def test_clean_files_are_not_modified(self, tmp_path: Path, mod: Any) -> None:
        """main() must not touch files that already pass all checks."""
        clean_dir = tmp_path / "category" / "clean-skill"
        clean_dir.mkdir(parents=True)
        skill_file = clean_dir / "SKILL.md"
        skill_file.write_text(CLEAN_SKILL_MD)

        mod.main(["--skills-dir", str(tmp_path)])

        assert skill_file.read_text() == CLEAN_SKILL_MD

    def test_multiple_categories_all_processed(self, tmp_path: Path, mod: Any) -> None:
        """Files in distinct category subdirectories are all discovered."""
        self._build_skills_tree(tmp_path)

        discovered = sorted(tmp_path.rglob("SKILL.md"))
        assert len(discovered) == 4, f"Expected 4 SKILL.md files, found {len(discovered)}"

        mod.main(["--skills-dir", str(tmp_path), "--dry-run"])

    def test_dry_run_flag_output_contains_would_fix(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        mod: Any,
    ) -> None:
        """--dry-run output should indicate what would be fixed."""
        fix_dir = tmp_path / "cat" / "skill-a"
        fix_dir.mkdir(parents=True)
        (fix_dir / "SKILL.md").write_text(NEEDS_WORKFLOW_FIX)

        mod.main(["--skills-dir", str(tmp_path), "--dry-run"])

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Would fix" in captured.out or "would" in captured.out.lower()

    def test_no_dry_run_output_shows_fixed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        mod: Any,
    ) -> None:
        """Normal run output should report files that were fixed."""
        fix_dir = tmp_path / "cat" / "skill-b"
        fix_dir.mkdir(parents=True)
        (fix_dir / "SKILL.md").write_text(NEEDS_WORKFLOW_FIX)

        mod.main(["--skills-dir", str(tmp_path)])

        captured = capsys.readouterr()
        assert "Fixed" in captured.out or "✓" in captured.out
