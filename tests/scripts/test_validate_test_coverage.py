#!/usr/bin/env python3
"""Tests for validate_test_coverage.py — stale pattern detection (issue #3358).

Verifies that check_stale_patterns() correctly identifies CI matrix entries
that match zero existing test files.
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from validate_test_coverage import (
    check_stale_patterns,
    expand_pattern,
    generate_report,
    group_split_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo-like directory tree with a few test files."""
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_foo.mojo").touch()
    (tmp_path / "tests" / "unit" / "test_bar.mojo").touch()
    (tmp_path / "tests" / "integration").mkdir(parents=True)
    (tmp_path / "tests" / "integration" / "test_baz.mojo").touch()
    return tmp_path


# ---------------------------------------------------------------------------
# group_split_files
# ---------------------------------------------------------------------------


class TestGroupSplitFiles:
    """Unit tests for group_split_files()."""

    def test_empty_input_returns_empty_dict(self) -> None:
        """No files → empty mapping."""
        assert group_split_files([]) == {}

    def test_no_part_files_returns_empty_dict(self) -> None:
        """Files without _partN suffix are not grouped."""
        files = [
            Path("tests/core/test_foo.mojo"),
            Path("tests/core/test_bar.mojo"),
        ]
        assert group_split_files(files) == {}

    def test_single_part_file_forms_group(self) -> None:
        """A single _part1 file creates a group with one member."""
        files = [Path("tests/core/test_foo_part1.mojo")]
        groups = group_split_files(files)
        assert "tests/core/test_foo" in groups
        assert groups["tests/core/test_foo"] == [Path("tests/core/test_foo_part1.mojo")]

    def test_multiple_parts_grouped_together(self) -> None:
        """Six part files are grouped under a single logical key."""
        files = [
            Path(f"tests/shared/core/test_elementwise_dispatch_part{i}.mojo")
            for i in range(1, 7)
        ]
        groups = group_split_files(files)
        key = "tests/shared/core/test_elementwise_dispatch"
        assert key in groups
        assert len(groups[key]) == 6

    def test_parts_are_sorted(self) -> None:
        """Part files within a group are returned in sorted order."""
        files = [
            Path("tests/core/test_foo_part3.mojo"),
            Path("tests/core/test_foo_part1.mojo"),
            Path("tests/core/test_foo_part2.mojo"),
        ]
        groups = group_split_files(files)
        result = groups["tests/core/test_foo"]
        assert result == sorted(result)

    def test_non_part_files_excluded_from_groups(self) -> None:
        """Regular test files mixed with part files are not included in groups."""
        files = [
            Path("tests/core/test_bar.mojo"),
            Path("tests/core/test_foo_part1.mojo"),
            Path("tests/core/test_foo_part2.mojo"),
        ]
        groups = group_split_files(files)
        assert "tests/core/test_foo" in groups
        assert len(groups) == 1  # test_bar.mojo not grouped

    def test_multiple_distinct_groups(self) -> None:
        """Two different base names produce two separate groups."""
        files = [
            Path("tests/core/test_alpha_part1.mojo"),
            Path("tests/core/test_alpha_part2.mojo"),
            Path("tests/core/test_beta_part1.mojo"),
            Path("tests/core/test_beta_part2.mojo"),
        ]
        groups = group_split_files(files)
        assert set(groups.keys()) == {
            "tests/core/test_alpha",
            "tests/core/test_beta",
        }

    def test_different_directories_create_separate_groups(self) -> None:
        """Same base name in different directories yields distinct keys."""
        files = [
            Path("tests/unit/test_foo_part1.mojo"),
            Path("tests/integration/test_foo_part1.mojo"),
        ]
        groups = group_split_files(files)
        assert "tests/unit/test_foo" in groups
        assert "tests/integration/test_foo" in groups
        assert len(groups) == 2

    def test_group_key_preserves_directory(self) -> None:
        """The group key includes the parent directory, not just the base name."""
        files = [Path("tests/shared/core/test_ops_part1.mojo")]
        groups = group_split_files(files)
        assert "tests/shared/core/test_ops" in groups

    def test_multidigit_part_numbers_matched(self) -> None:
        """Part numbers with more than one digit (e.g., part10) are matched."""
        files = [
            Path("tests/core/test_foo_part10.mojo"),
            Path("tests/core/test_foo_part11.mojo"),
        ]
        groups = group_split_files(files)
        assert "tests/core/test_foo" in groups
        assert len(groups["tests/core/test_foo"]) == 2


# ---------------------------------------------------------------------------
# check_stale_patterns
# ---------------------------------------------------------------------------


class TestCheckStalePatterns:
    """Unit tests for check_stale_patterns()."""

    def test_no_stale_when_all_patterns_match(self, tmp_repo: Path) -> None:
        """Groups that match existing files are not reported as stale."""
        ci_groups = {
            "Unit Tests": {"path": "tests/unit", "pattern": "test_*.mojo"},
            "Integration Tests": {
                "path": "tests/integration",
                "pattern": "test_*.mojo",
            },
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == []

    def test_stale_when_path_does_not_exist(self, tmp_repo: Path) -> None:
        """A pattern whose base path does not exist is reported as stale."""
        ci_groups = {
            "Ghost Tests": {"path": "tests/nonexistent", "pattern": "test_*.mojo"},
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == ["Ghost Tests"]

    def test_stale_when_pattern_matches_no_files(self, tmp_repo: Path) -> None:
        """A pattern that globs to zero files in an existing path is stale."""
        ci_groups = {
            "Unit Tests": {"path": "tests/unit", "pattern": "test_missing_*.mojo"},
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == ["Unit Tests"]

    def test_partial_stale(self, tmp_repo: Path) -> None:
        """Only the groups with zero matches are returned."""
        ci_groups = {
            "Good Group": {"path": "tests/unit", "pattern": "test_*.mojo"},
            "Stale Group": {"path": "tests/deleted", "pattern": "test_*.mojo"},
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == ["Stale Group"]

    def test_multiple_stale_groups_sorted(self, tmp_repo: Path) -> None:
        """Multiple stale groups are returned in sorted order."""
        ci_groups = {
            "Zebra Tests": {"path": "tests/z_gone", "pattern": "test_*.mojo"},
            "Alpha Tests": {"path": "tests/a_gone", "pattern": "test_*.mojo"},
            "Good Tests": {"path": "tests/unit", "pattern": "test_*.mojo"},
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == ["Alpha Tests", "Zebra Tests"]

    def test_empty_ci_groups_returns_empty(self, tmp_repo: Path) -> None:
        """An empty CI groups mapping yields no stale patterns."""
        stale = check_stale_patterns({}, tmp_repo)
        assert stale == []

    def test_returns_list_of_strings(self, tmp_repo: Path) -> None:
        """Return type is always a list of strings."""
        ci_groups = {
            "Unit Tests": {"path": "tests/unit", "pattern": "test_*.mojo"},
        }
        result = check_stale_patterns(ci_groups, tmp_repo)
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_specific_file_pattern_stale_after_rename(self, tmp_repo: Path) -> None:
        """A group referencing a specific file that was renamed is stale."""
        # tests/unit/test_renamed.mojo no longer exists
        ci_groups = {
            "Renamed Test": {
                "path": "tests/unit",
                "pattern": "test_renamed.mojo",
            },
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == ["Renamed Test"]

    def test_specific_file_pattern_not_stale_when_file_exists(self, tmp_repo: Path) -> None:
        """A group referencing a specific existing file is not stale."""
        ci_groups = {
            "Foo Test": {"path": "tests/unit", "pattern": "test_foo.mojo"},
        }
        stale = check_stale_patterns(ci_groups, tmp_repo)
        assert stale == []


# ---------------------------------------------------------------------------
# generate_report — stale_patterns argument
# ---------------------------------------------------------------------------


class TestGenerateReportStalePatterns:
    """Tests for the stale_patterns argument of generate_report()."""

    def _make_coverage(self) -> dict:
        return {"Unit Tests": {Path("tests/unit/test_foo.mojo")}}

    def test_no_stale_section_when_none_passed(self) -> None:
        """stale_patterns=None produces no '### Stale CI Patterns' heading."""
        report = generate_report(set(), [], self._make_coverage(), stale_patterns=None)
        assert "### Stale CI Patterns" not in report

    def test_no_stale_section_when_empty_list(self) -> None:
        """stale_patterns=[] produces no '### Stale CI Patterns' heading."""
        report = generate_report(set(), [], self._make_coverage(), stale_patterns=[])
        assert "### Stale CI Patterns" not in report

    def test_stale_section_appended_when_patterns_exist(self) -> None:
        """Non-empty stale_patterns appends a stale section with all names."""
        report = generate_report(set(), [], self._make_coverage(), stale_patterns=["Alpha Tests", "Zebra Tests"])
        assert "### Stale CI Patterns" in report
        assert "- Alpha Tests" in report
        assert "- Zebra Tests" in report

    def test_stale_section_with_uncovered_files(self, tmp_path: Path) -> None:
        """Both uncovered files and stale patterns appear in the report."""
        uncovered = {Path("tests/unit/test_missing.mojo")}
        report = generate_report(
            uncovered,
            [Path("tests/unit/test_missing.mojo")],
            {},
            stale_patterns=["Ghost Group"],
        )
        assert "### Uncovered Tests" in report
        assert "### Stale CI Patterns" in report
        assert "- Ghost Group" in report

    def test_stale_only_no_uncovered(self) -> None:
        """Only stale patterns (no uncovered files) still produces the stale section."""
        report = generate_report(set(), [], self._make_coverage(), stale_patterns=["Deleted Group"])
        assert "### Stale CI Patterns" in report
        assert "- Deleted Group" in report


# ---------------------------------------------------------------------------
# expand_pattern (regression guard — unchanged behaviour)
# ---------------------------------------------------------------------------


class TestExpandPattern:
    """Regression tests ensuring expand_pattern still works correctly."""

    def test_wildcard_finds_files(self, tmp_repo: Path) -> None:
        matched = expand_pattern("tests/unit", "test_*.mojo", tmp_repo)
        assert len(matched) == 2

    def test_wildcard_finds_no_files_in_missing_dir(self, tmp_repo: Path) -> None:
        matched = expand_pattern("tests/missing", "test_*.mojo", tmp_repo)
        assert matched == set()

    def test_direct_file_found(self, tmp_repo: Path) -> None:
        matched = expand_pattern("tests/unit", "test_foo.mojo", tmp_repo)
        assert len(matched) == 1

    def test_direct_file_missing(self, tmp_repo: Path) -> None:
        matched = expand_pattern("tests/unit", "test_gone.mojo", tmp_repo)
        assert matched == set()
