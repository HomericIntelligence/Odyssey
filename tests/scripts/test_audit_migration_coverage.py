#!/usr/bin/env python3
"""Tests for the --audit flag in migrate_odyssey_skills.py.

Covers:
- find_skill_in_mnemosyne()
- load_skip_list()
- run_audit()
- AuditResult properties
- print_audit_table() / print_audit_summary() output
- main() exit codes for --audit mode
"""

import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from migrate_odyssey_skills import (
    AuditResult,
    SkillAuditEntry,
    find_skill_in_mnemosyne,
    load_skip_list,
    print_audit_summary,
    print_audit_table,
    run_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mnemosyne_dir(tmp_path: Path) -> Path:
    """Create a minimal fake Mnemosyne skills/ directory."""
    skills_dir = tmp_path / "skills"
    # category: tooling — contains skill-a
    (skills_dir / "tooling" / "skill-a").mkdir(parents=True)
    # category: ci-cd — contains skill-b
    (skills_dir / "ci-cd" / "skill-b").mkdir(parents=True)
    return skills_dir


@pytest.fixture()
def source_skill_md(tmp_path: Path) -> Path:
    """Create a minimal SKILL.md file."""
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: skill-a\ndescription: Test skill\n---\n# skill-a\n")
    return p


@pytest.fixture()
def source_skills(source_skill_md: Path) -> list[tuple[str, Path, None]]:
    """Three source skills: skill-a, skill-b, skill-c."""
    parent = source_skill_md.parent
    md_b = parent / "skill-b.md"
    md_b.write_text("---\nname: skill-b\n---\n")
    md_c = parent / "skill-c.md"
    md_c.write_text("---\nname: skill-c\n---\n")
    return [
        ("skill-a", source_skill_md, None),
        ("skill-b", md_b, None),
        ("skill-c", md_c, None),
    ]


# ---------------------------------------------------------------------------
# find_skill_in_mnemosyne
# ---------------------------------------------------------------------------


class TestFindSkillInMnemosyne:
    def test_returns_category_when_skill_exists(self, mnemosyne_dir: Path) -> None:
        result = find_skill_in_mnemosyne("skill-a", mnemosyne_dir)
        assert result == "tooling"

    def test_returns_category_for_different_category(self, mnemosyne_dir: Path) -> None:
        result = find_skill_in_mnemosyne("skill-b", mnemosyne_dir)
        assert result == "ci-cd"

    def test_returns_none_when_skill_missing(self, mnemosyne_dir: Path) -> None:
        result = find_skill_in_mnemosyne("nonexistent-skill", mnemosyne_dir)
        assert result is None

    def test_returns_none_when_mnemosyne_dir_missing(self, tmp_path: Path) -> None:
        result = find_skill_in_mnemosyne("skill-a", tmp_path / "does-not-exist")
        assert result is None

    def test_ignores_files_in_category_dir(self, mnemosyne_dir: Path) -> None:
        # A file named like a skill should not match (only dirs count)
        (mnemosyne_dir / "tooling" / "skill-file").write_text("not a dir")
        result = find_skill_in_mnemosyne("skill-file", mnemosyne_dir)
        assert result is None


# ---------------------------------------------------------------------------
# load_skip_list
# ---------------------------------------------------------------------------


class TestLoadSkipList:
    def test_returns_empty_set_when_file_missing(self, tmp_path: Path) -> None:
        result = load_skip_list(tmp_path / "no-such-file")
        assert result == set()

    def test_reads_skill_names(self, tmp_path: Path) -> None:
        skip_file = tmp_path / ".audit-skip"
        skip_file.write_text("skill-x\nskill-y\n")
        result = load_skip_list(skip_file)
        assert result == {"skill-x", "skill-y"}

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        skip_file = tmp_path / ".audit-skip"
        skip_file.write_text("\nskill-x\n\nskill-y\n\n")
        result = load_skip_list(skip_file)
        assert result == {"skill-x", "skill-y"}

    def test_ignores_comment_lines(self, tmp_path: Path) -> None:
        skip_file = tmp_path / ".audit-skip"
        skip_file.write_text("# This is a comment\nskill-x\n# another comment\n")
        result = load_skip_list(skip_file)
        assert result == {"skill-x"}

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        skip_file = tmp_path / ".audit-skip"
        skip_file.write_text("  skill-x  \n  skill-y\n")
        result = load_skip_list(skip_file)
        assert result == {"skill-x", "skill-y"}


# ---------------------------------------------------------------------------
# run_audit
# ---------------------------------------------------------------------------


class TestRunAudit:
    def test_present_skill_marked_present(
        self,
        source_skills: list[tuple[str, Path, None]],
        mnemosyne_dir: Path,
    ) -> None:
        result = run_audit(source_skills, mnemosyne_dir, skip_list=set())
        entry = next(e for e in result.skills if e.name == "skill-a")
        assert entry.status == "present"
        assert entry.mnemosyne_category == "tooling"

    def test_missing_skill_marked_missing(
        self,
        source_skills: list[tuple[str, Path, None]],
        mnemosyne_dir: Path,
    ) -> None:
        result = run_audit(source_skills, mnemosyne_dir, skip_list=set())
        entry = next(e for e in result.skills if e.name == "skill-c")
        assert entry.status == "missing"
        assert entry.mnemosyne_category is None

    def test_skipped_skill_marked_skipped(
        self,
        source_skills: list[tuple[str, Path, None]],
        mnemosyne_dir: Path,
    ) -> None:
        result = run_audit(source_skills, mnemosyne_dir, skip_list={"skill-c"})
        entry = next(e for e in result.skills if e.name == "skill-c")
        assert entry.status == "skipped"

    def test_counts_are_correct(
        self,
        source_skills: list[tuple[str, Path, None]],
        mnemosyne_dir: Path,
    ) -> None:
        # skill-a: present, skill-b: present, skill-c: missing
        result = run_audit(source_skills, mnemosyne_dir, skip_list=set())
        assert result.total == 3
        assert result.present == 2
        assert result.missing == 1
        assert result.skipped == 0

    def test_skipped_excluded_from_coverage_denominator(
        self,
        source_skills: list[tuple[str, Path, None]],
        mnemosyne_dir: Path,
    ) -> None:
        # skill-c missing but skipped => coverage = 2/2 = 100%
        result = run_audit(source_skills, mnemosyne_dir, skip_list={"skill-c"})
        assert result.coverage_pct == 100.0

    def test_deduplicates_skills_by_name(
        self,
        source_skill_md: Path,
        mnemosyne_dir: Path,
    ) -> None:
        duplicate_skills = [
            ("skill-a", source_skill_md, None),
            ("skill-a", source_skill_md, "1"),  # duplicate
        ]
        result = run_audit(duplicate_skills, mnemosyne_dir, skip_list=set())
        assert result.total == 1

    def test_empty_source_skills(self, mnemosyne_dir: Path) -> None:
        result = run_audit([], mnemosyne_dir, skip_list=set())
        assert result.total == 0
        assert result.coverage_pct == 100.0

    def test_empty_skip_list_marks_all_unmatched_as_missing(
        self,
        mnemosyne_dir: Path,
        tmp_path: Path,
    ) -> None:
        md = tmp_path / "x.md"
        md.write_text("")
        skills = [("no-such-skill", md, None)]
        result = run_audit(skills, mnemosyne_dir, skip_list=set())
        assert result.missing == 1
        assert result.present == 0


# ---------------------------------------------------------------------------
# AuditResult properties
# ---------------------------------------------------------------------------


class TestAuditResult:
    def _make_result(self, statuses: list[str]) -> AuditResult:
        result = AuditResult()
        for i, status in enumerate(statuses):
            result.skills.append(
                SkillAuditEntry(
                    name=f"skill-{i}",
                    source_path=Path(f"/fake/skill-{i}/SKILL.md"),
                    tier=None,
                    mnemosyne_category="tooling" if status == "present" else None,
                    status=status,
                )
            )
        return result

    def test_total(self) -> None:
        r = self._make_result(["present", "missing", "skipped"])
        assert r.total == 3

    def test_present_count(self) -> None:
        r = self._make_result(["present", "present", "missing"])
        assert r.present == 2

    def test_missing_count(self) -> None:
        r = self._make_result(["present", "missing", "missing"])
        assert r.missing == 2

    def test_skipped_count(self) -> None:
        r = self._make_result(["skipped", "skipped", "present"])
        assert r.skipped == 2

    def test_coverage_pct_all_present(self) -> None:
        r = self._make_result(["present", "present"])
        assert r.coverage_pct == 100.0

    def test_coverage_pct_half_present(self) -> None:
        r = self._make_result(["present", "missing"])
        assert r.coverage_pct == pytest.approx(50.0)

    def test_coverage_pct_excludes_skipped(self) -> None:
        r = self._make_result(["present", "skipped"])
        # 1 present / (2 total - 1 skipped) = 100%
        assert r.coverage_pct == 100.0

    def test_coverage_pct_all_skipped(self) -> None:
        r = self._make_result(["skipped", "skipped"])
        assert r.coverage_pct == 100.0


# ---------------------------------------------------------------------------
# print_audit_table
# ---------------------------------------------------------------------------


class TestPrintAuditTable:
    def _capture(self, result: AuditResult) -> str:
        buf = StringIO()
        sys.stdout = buf
        try:
            print_audit_table(result, no_color=True)
        finally:
            sys.stdout = sys.__stdout__
        return buf.getvalue()

    def _make_result_with_entry(self, name: str, status: str, category: str | None) -> AuditResult:
        result = AuditResult()
        result.skills.append(
            SkillAuditEntry(
                name=name,
                source_path=Path(f"/fake/{name}/SKILL.md"),
                tier=None,
                mnemosyne_category=category,
                status=status,
            )
        )
        return result

    def test_header_present(self) -> None:
        output = self._capture(AuditResult())
        assert "Skill" in output
        assert "Category" in output
        assert "Status" in output

    def test_present_skill_shows_present(self) -> None:
        result = self._make_result_with_entry("my-skill", "present", "tooling")
        output = self._capture(result)
        assert "my-skill" in output
        assert "tooling" in output
        assert "PRESENT" in output

    def test_missing_skill_shows_missing(self) -> None:
        result = self._make_result_with_entry("my-skill", "missing", None)
        output = self._capture(result)
        assert "MISSING" in output
        assert "-" in output  # category placeholder

    def test_skipped_skill_shows_skipped(self) -> None:
        result = self._make_result_with_entry("my-skill", "skipped", None)
        output = self._capture(result)
        assert "SKIPPED" in output

    def test_skills_sorted_alphabetically(self) -> None:
        result = AuditResult()
        for name in ["zzz-skill", "aaa-skill", "mmm-skill"]:
            result.skills.append(
                SkillAuditEntry(
                    name=name,
                    source_path=Path(f"/fake/{name}/SKILL.md"),
                    tier=None,
                    mnemosyne_category="tooling",
                    status="present",
                )
            )
        output = self._capture(result)
        pos_aaa = output.index("aaa-skill")
        pos_mmm = output.index("mmm-skill")
        pos_zzz = output.index("zzz-skill")
        assert pos_aaa < pos_mmm < pos_zzz


# ---------------------------------------------------------------------------
# print_audit_summary
# ---------------------------------------------------------------------------


class TestPrintAuditSummary:
    def _capture(self, result: AuditResult) -> str:
        buf = StringIO()
        sys.stdout = buf
        try:
            print_audit_summary(result, no_color=True)
        finally:
            sys.stdout = sys.__stdout__
        return buf.getvalue()

    def _make_result(self, present: int, missing: int, skipped: int) -> AuditResult:
        result = AuditResult()
        for i in range(present):
            result.skills.append(SkillAuditEntry(f"present-{i}", Path("/f"), None, "tooling", "present"))
        for i in range(missing):
            result.skills.append(SkillAuditEntry(f"missing-{i}", Path("/f"), None, None, "missing"))
        for i in range(skipped):
            result.skills.append(SkillAuditEntry(f"skipped-{i}", Path("/f"), None, None, "skipped"))
        return result

    def test_shows_counts(self) -> None:
        result = self._make_result(present=3, missing=1, skipped=1)
        output = self._capture(result)
        assert "3" in output  # present
        assert "1" in output  # missing / skipped

    def test_pass_message_when_no_missing(self) -> None:
        result = self._make_result(present=2, missing=0, skipped=0)
        output = self._capture(result)
        assert "PASS" in output

    def test_fail_message_when_missing(self) -> None:
        result = self._make_result(present=1, missing=1, skipped=0)
        output = self._capture(result)
        assert "FAIL" in output

    def test_coverage_percentage_in_output(self) -> None:
        result = self._make_result(present=1, missing=1, skipped=0)
        output = self._capture(result)
        assert "50.0%" in output


# ---------------------------------------------------------------------------
# Integration: main() exit codes via subprocess
# ---------------------------------------------------------------------------


SCRIPT = str(Path(__file__).parent.parent.parent / "scripts" / "migrate_odyssey_skills.py")


class TestMainAuditExitCodes:
    def _run_audit(
        self,
        source_dir: str,
        target_dir: str,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            SCRIPT,
            "--audit",
            "--no-color",
            "--source-dir",
            source_dir,
            "--target-dir",
            target_dir,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_exit_zero_when_all_skills_present(self, tmp_path: Path) -> None:
        # Create source: one skill
        source = tmp_path / "source"
        skill_dir = source / "skill-x"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-x\n---\n")

        # Create target: skill-x exists in tooling
        target = tmp_path / "target"
        (target / "skills" / "tooling" / "skill-x").mkdir(parents=True)

        proc = self._run_audit(str(source), str(target))
        assert proc.returncode == 0

    def test_exit_one_when_skills_missing(self, tmp_path: Path) -> None:
        # Create source: one skill
        source = tmp_path / "source"
        skill_dir = source / "skill-y"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-y\n---\n")

        # Create empty target: skill-y NOT present
        target = tmp_path / "target"
        (target / "skills").mkdir(parents=True)

        proc = self._run_audit(str(source), str(target))
        assert proc.returncode == 1

    def test_exit_zero_when_missing_skill_is_skipped(self, tmp_path: Path) -> None:
        # Create source: one skill
        source = tmp_path / "source"
        skill_dir = source / "skill-z"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-z\n---\n")

        # Empty target
        target = tmp_path / "target"
        (target / "skills").mkdir(parents=True)

        # Skip list
        skip_file = tmp_path / ".audit-skip"
        skip_file.write_text("skill-z\n")

        proc = self._run_audit(str(source), str(target), extra_args=["--audit-skip", str(skip_file)])
        assert proc.returncode == 0

    def test_output_contains_skill_name_and_status(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        skill_dir = source / "skill-q"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-q\n---\n")

        target = tmp_path / "target"
        (target / "skills" / "tooling" / "skill-q").mkdir(parents=True)

        proc = self._run_audit(str(source), str(target))
        assert "skill-q" in proc.stdout
        assert "PRESENT" in proc.stdout

    def test_output_contains_summary_section(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        skill_dir = source / "skill-r"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-r\n---\n")

        target = tmp_path / "target"
        (target / "skills" / "tooling" / "skill-r").mkdir(parents=True)

        proc = self._run_audit(str(source), str(target))
        assert "Audit Summary" in proc.stdout
        assert "Coverage" in proc.stdout

    def test_exit_one_when_source_dir_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        proc = self._run_audit(str(tmp_path / "no-such-source"), str(target))
        assert proc.returncode == 1

    def test_exit_one_when_target_dir_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        proc = self._run_audit(str(source), str(tmp_path / "no-such-target"))
        assert proc.returncode == 1
