#!/usr/bin/env python3
"""Tests for validate_test_coverage.py — stale pattern detection (issue #3358)
and group overlap detection (issue #4459).

Verifies that:
- check_stale_patterns() correctly identifies CI matrix entries that match
  zero existing test files.
- check_group_overlaps() correctly identifies CI matrix groups whose resolved
  file sets overlap.
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from validate_test_coverage import (
    _paths_overlap,
    check_group_overlaps,
    check_stale_patterns,
    expand_pattern,
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


# ---------------------------------------------------------------------------
# _paths_overlap
# ---------------------------------------------------------------------------


class TestPathsOverlap:
    """Unit tests for _paths_overlap()."""

    def test_identical_paths_overlap(self) -> None:
        assert _paths_overlap("tests/shared/data", "tests/shared/data") is True

    def test_parent_child_overlap(self) -> None:
        assert _paths_overlap("tests/shared", "tests/shared/data") is True

    def test_child_parent_overlap_reversed(self) -> None:
        assert _paths_overlap("tests/shared/data", "tests/shared") is True

    def test_sibling_paths_do_not_overlap(self) -> None:
        assert _paths_overlap("tests/shared/data", "tests/shared/core") is False

    def test_unrelated_paths_do_not_overlap(self) -> None:
        assert _paths_overlap("benchmarks/ops", "tests/shared") is False

    def test_partial_name_match_does_not_count(self) -> None:
        # "tests/shared" should NOT overlap with "tests/shared2"
        assert _paths_overlap("tests/shared", "tests/shared2") is False


# ---------------------------------------------------------------------------
# check_group_overlaps
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo_with_subdirs(tmp_path: Path) -> Path:
    """Create a repo tree with nested test directories."""
    (tmp_path / "tests" / "shared" / "data" / "datasets").mkdir(parents=True)
    (tmp_path / "tests" / "shared" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "shared" / "core").mkdir(parents=True)
    (tmp_path / "tests" / "shared" / "data" / "test_loader.mojo").touch()
    (tmp_path / "tests" / "shared" / "data" / "datasets" / "test_cifar.mojo").touch()
    (tmp_path / "tests" / "shared" / "core" / "test_tensor.mojo").touch()
    return tmp_path


class TestCheckGroupOverlaps:
    """Unit tests for check_group_overlaps()."""

    def test_no_overlap_when_groups_are_disjoint(self, tmp_repo_with_subdirs: Path) -> None:
        root = tmp_repo_with_subdirs
        ci_groups = {
            "Data": {"path": "tests/shared/data", "pattern": "test_*.mojo"},
            "Core": {"path": "tests/shared/core", "pattern": "test_*.mojo"},
        }
        coverage = {
            "Data": expand_pattern("tests/shared/data", "test_*.mojo", root),
            "Core": expand_pattern("tests/shared/core", "test_*.mojo", root),
        }
        result = check_group_overlaps(ci_groups, coverage)
        assert result == []

    def test_detects_overlap_between_parent_and_child_groups(self, tmp_repo_with_subdirs: Path) -> None:
        root = tmp_repo_with_subdirs
        # "Data" uses a subdirectory wildcard that picks up datasets/test_cifar.mojo
        # "Datasets" explicitly targets that same file
        ci_groups = {
            "Data": {"path": "tests/shared/data", "pattern": "datasets/test_*.mojo"},
            "Datasets": {
                "path": "tests/shared/data/datasets",
                "pattern": "test_*.mojo",
            },
        }
        coverage = {
            "Data": expand_pattern("tests/shared/data", "datasets/test_*.mojo", root),
            "Datasets": expand_pattern("tests/shared/data/datasets", "test_*.mojo", root),
        }
        result = check_group_overlaps(ci_groups, coverage)
        assert len(result) == 1
        group_a, group_b, f = result[0]
        assert set([group_a, group_b]) == {"Data", "Datasets"}
        assert f == Path("tests/shared/data/datasets/test_cifar.mojo")

    def test_no_false_positive_across_unrelated_dirs(self, tmp_repo_with_subdirs: Path) -> None:
        root = tmp_repo_with_subdirs
        ci_groups = {
            "Benchmarks": {"path": "benchmarks/ops", "pattern": "test_*.mojo"},
            "Core": {"path": "tests/shared/core", "pattern": "test_*.mojo"},
        }
        coverage = {
            "Benchmarks": expand_pattern("benchmarks/ops", "test_*.mojo", root),
            "Core": expand_pattern("tests/shared/core", "test_*.mojo", root),
        }
        result = check_group_overlaps(ci_groups, coverage)
        assert result == []

    def test_empty_groups_returns_empty(self) -> None:
        result = check_group_overlaps({}, {})
        assert result == []

    def test_overlap_result_is_sorted(self, tmp_repo_with_subdirs: Path) -> None:
        """Overlapping files are returned in deterministic sorted order."""
        root = tmp_repo_with_subdirs
        # Add a second overlapping file
        (root / "tests" / "shared" / "data" / "datasets" / "test_mnist.mojo").touch()
        ci_groups = {
            "Data": {"path": "tests/shared/data", "pattern": "datasets/test_*.mojo"},
            "Datasets": {
                "path": "tests/shared/data/datasets",
                "pattern": "test_*.mojo",
            },
        }
        coverage = {
            "Data": expand_pattern("tests/shared/data", "datasets/test_*.mojo", root),
            "Datasets": expand_pattern("tests/shared/data/datasets", "test_*.mojo", root),
        }
        result = check_group_overlaps(ci_groups, coverage)
        files = [f for _, _, f in result]
        assert files == sorted(files)

    def test_overlap_triples_contain_group_names_and_path(self, tmp_repo_with_subdirs: Path) -> None:
        root = tmp_repo_with_subdirs
        ci_groups = {
            "Data": {"path": "tests/shared/data", "pattern": "datasets/test_*.mojo"},
            "Datasets": {
                "path": "tests/shared/data/datasets",
                "pattern": "test_*.mojo",
            },
        }
        coverage = {
            "Data": expand_pattern("tests/shared/data", "datasets/test_*.mojo", root),
            "Datasets": expand_pattern("tests/shared/data/datasets", "test_*.mojo", root),
        }
        result = check_group_overlaps(ci_groups, coverage)
        assert len(result) == 1
        group_a, group_b, f = result[0]
        assert isinstance(group_a, str)
        assert isinstance(group_b, str)
        assert isinstance(f, Path)
