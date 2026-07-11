"""Tests for scripts/check_source_coverage.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import check_source_coverage
from check_source_coverage import (
    expected_test_paths,
    find_uncovered_sources,
    generate_report,
    main,
)


def test_expected_test_paths_returns_all_five_layouts() -> None:
    """Test that expected_test_paths covers all five layout candidates."""
    paths = expected_test_paths(Path("src/odyssey/core/foo.mojo"))
    names = {str(p) for p in paths}
    assert "tests/odyssey/core/test_foo.mojo" in names
    assert "tests/odyssey/core/test_foo_part1.mojo" in names
    assert "tests/core/test_foo.mojo" in names
    assert "tests/models/test_foo.mojo" in names
    assert "tests/shared/core/test_foo.mojo" in names


def test_find_uncovered_sources_flags_source_with_no_test() -> None:
    """Test that sources with no matching test are flagged."""
    sources = [Path("src/odyssey/core/orphan.mojo")]
    tests = {Path("tests/odyssey/core/test_unrelated.mojo")}
    assert find_uncovered_sources(sources, tests) == [Path("src/odyssey/core/orphan.mojo")]


def test_find_uncovered_sources_accepts_split_part1_pattern() -> None:
    """Test that split-file pattern test_foo_part1.mojo is accepted."""
    sources = [Path("src/odyssey/core/foo.mojo")]
    tests = {Path("tests/odyssey/core/test_foo_part1.mojo")}
    assert find_uncovered_sources(sources, tests) == []


def test_find_uncovered_sources_accepts_legacy_layout() -> None:
    """Test that legacy flat layout tests/X/test_Y.mojo is accepted."""
    sources = [Path("src/odyssey/core/foo.mojo")]
    tests = {Path("tests/core/test_foo.mojo")}
    assert find_uncovered_sources(sources, tests) == []


def test_generate_report_clean_baseline() -> None:
    """Test report generation when all sources are covered."""
    out = generate_report([], total_sources=42)
    assert "✅" in out and "42" in out


def test_generate_report_lists_each_uncovered_file() -> None:
    """Test that report lists each uncovered file and shows the count."""
    out = generate_report([Path("src/odyssey/x/y.mojo")], total_sources=10)
    assert "src/odyssey/x/y.mojo" in out
    assert "1 of 10" in out


def test_main_returns_zero_even_when_uncovered_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() returns 0 (warn-only contract) when test discovery succeeds.

    Stub find_test_files so the contract is verified deterministically,
    independent of the current checkout (a worktree checkout would otherwise
    yield zero discovered tests — see the sanity-guard test below).
    """
    monkeypatch.setattr(
        check_source_coverage,
        "find_test_files",
        lambda _root: [Path("tests/odyssey/core/test_something.mojo")],
    )
    assert main() == 0


def test_main_fails_when_no_tests_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() returns non-zero when test discovery yields nothing.

    An empty test set is never legitimate in this repo; it signals a
    misconfigured invocation (e.g. from a worktree checkout, which
    find_test_files excludes). The script must fail loudly rather than emit a
    misleading 100%-uncovered report.
    """
    monkeypatch.setattr(check_source_coverage, "find_test_files", lambda _root: [])
    assert main() == 1
