#!/usr/bin/env python3
"""Unit tests for scripts/check_test_count_badge.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from check_test_count_badge import (
    check_badge_drift,
    count_test_files,
    main,
    parse_badge_count,
    update_badge,
)


# ---------------------------------------------------------------------------
# count_test_files
# ---------------------------------------------------------------------------


def test_count_test_files_finds_mojo_files(tmp_path: Path) -> None:
    """Should count test_*.mojo files in a directory tree."""
    (tmp_path / "test_alpha.mojo").write_text("")
    (tmp_path / "test_beta.mojo").write_text("")
    (tmp_path / "not_a_test.mojo").write_text("")  # must be excluded by name
    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "test_gamma.mojo").write_text("")

    count = count_test_files(tmp_path)
    assert count == 3


def test_count_test_files_excludes_pixi(tmp_path: Path) -> None:
    """Files under .pixi/ should be excluded."""
    pixi = tmp_path / ".pixi" / "lib"
    pixi.mkdir(parents=True)
    (pixi / "test_hidden.mojo").write_text("")
    (tmp_path / "test_real.mojo").write_text("")

    count = count_test_files(tmp_path)
    assert count == 1


def test_count_test_files_excludes_worktrees(tmp_path: Path) -> None:
    """Files under worktrees/ should be excluded."""
    wt = tmp_path / "worktrees" / "branch"
    wt.mkdir(parents=True)
    (wt / "test_branch.mojo").write_text("")
    (tmp_path / "test_main.mojo").write_text("")

    count = count_test_files(tmp_path)
    assert count == 1


def test_count_test_files_empty_dir(tmp_path: Path) -> None:
    """Should return 0 for directories with no matching files."""
    assert count_test_files(tmp_path) == 0


# ---------------------------------------------------------------------------
# parse_badge_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "badge_url, expected",
    [
        ("[![Tests](https://img.shields.io/badge/tests-247%2B-brightgreen.svg)](tests/)", 247),
        ("[![Tests](https://img.shields.io/badge/tests-122%2B-brightgreen.svg)](tests/)", 122),
        ("[![Tests](https://img.shields.io/badge/tests-300%2B-brightgreen.svg)](tests/)", 300),
        ("[![Tests](https://img.shields.io/badge/tests-1000%2B-brightgreen.svg)](tests/)", 1000),
    ],
)
def test_parse_badge_count_various_values(tmp_path: Path, badge_url: str, expected: int) -> None:
    """Should extract numeric count from various badge URL formats."""
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{badge_url}\n")
    assert parse_badge_count(readme) == expected


def test_parse_badge_count_missing_badge(tmp_path: Path) -> None:
    """Should return None when no tests badge is present."""
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nNo badge here.\n")
    assert parse_badge_count(readme) is None


def test_parse_badge_count_nonexistent_file(tmp_path: Path) -> None:
    """Should return None for a non-existent README."""
    assert parse_badge_count(tmp_path / "MISSING.md") is None


# ---------------------------------------------------------------------------
# check_badge_drift
# ---------------------------------------------------------------------------


def test_check_badge_drift_within_tolerance() -> None:
    """Should return True when drift is within 10%."""
    # 5% drift
    assert check_badge_drift(actual=200, badge=190) is True


def test_check_badge_drift_exactly_at_tolerance() -> None:
    """Should return True when drift equals the tolerance boundary exactly."""
    # 10% below
    assert check_badge_drift(actual=200, badge=180, tolerance=0.10) is True


def test_check_badge_drift_just_over_tolerance() -> None:
    """Should return False when drift exceeds tolerance."""
    # ~10.5% drift
    assert check_badge_drift(actual=200, badge=179, tolerance=0.10) is False


def test_check_badge_drift_zero_actual() -> None:
    """Should handle zero actual count: matches only when badge is also zero."""
    assert check_badge_drift(actual=0, badge=0) is True
    assert check_badge_drift(actual=0, badge=1) is False


def test_check_badge_drift_custom_tolerance() -> None:
    """Should respect custom tolerance values."""
    assert check_badge_drift(actual=100, badge=95, tolerance=0.05) is True
    assert check_badge_drift(actual=100, badge=94, tolerance=0.05) is False


# ---------------------------------------------------------------------------
# update_badge
# ---------------------------------------------------------------------------


def test_update_badge_rewrites_count(tmp_path: Path) -> None:
    """Should update the badge count in README.md."""
    readme = tmp_path / "README.md"
    readme.write_text("[![Tests](https://img.shields.io/badge/tests-122%2B-brightgreen.svg)](tests/)\n")
    update_badge(readme, 300)
    content = readme.read_text()
    assert "tests-300%2B-brightgreen.svg" in content
    assert "tests-122" not in content


def test_update_badge_preserves_rest_of_file(tmp_path: Path) -> None:
    """Should not modify any content outside the badge URL."""
    readme = tmp_path / "README.md"
    original = (
        "# Title\n\n"
        "[![Tests](https://img.shields.io/badge/tests-247%2B-brightgreen.svg)](tests/)\n\n"
        "Some other content.\n"
    )
    readme.write_text(original)
    update_badge(readme, 250)
    content = readme.read_text()
    assert "# Title" in content
    assert "Some other content." in content
    assert "tests-250%2B-brightgreen.svg" in content


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


def _write_readme(path: Path, count: int) -> None:
    path.write_text(f"[![Tests](https://img.shields.io/badge/tests-{count}%2B-brightgreen.svg)](tests/)\n")


def test_main_badge_current_exits_zero(tmp_path: Path) -> None:
    """Should exit 0 when badge matches actual count within tolerance."""
    readme = tmp_path / "README.md"
    _write_readme(readme, 200)

    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=200),
        patch("sys.argv", ["check_test_count_badge.py"]),
    ):
        assert main() == 0


def test_main_badge_stale_exits_one(tmp_path: Path) -> None:
    """Should exit 1 when badge is stale beyond tolerance."""
    readme = tmp_path / "README.md"
    _write_readme(readme, 100)

    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=200),
        patch("sys.argv", ["check_test_count_badge.py"]),
    ):
        assert main() == 1


def test_main_fix_mode_updates_badge(tmp_path: Path) -> None:
    """Should update badge and exit 0 when --fix is passed."""
    readme = tmp_path / "README.md"
    _write_readme(readme, 100)

    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=200),
        patch("sys.argv", ["check_test_count_badge.py", "--fix"]),
    ):
        result = main()

    assert result == 0
    content = readme.read_text()
    assert "tests-200%2B-brightgreen.svg" in content


def test_main_no_badge_in_readme_exits_one(tmp_path: Path) -> None:
    """Should exit 1 when README has no tests badge."""
    readme = tmp_path / "README.md"
    readme.write_text("# No badge here\n")

    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=100),
        patch("sys.argv", ["check_test_count_badge.py"]),
    ):
        assert main() == 1


def test_main_custom_tolerance(tmp_path: Path) -> None:
    """Should respect --tolerance argument."""
    readme = tmp_path / "README.md"
    _write_readme(readme, 190)  # 5% drift from 200

    # With 3% tolerance, 5% drift should fail
    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=200),
        patch("sys.argv", ["check_test_count_badge.py", "--tolerance", "0.03"]),
    ):
        assert main() == 1

    # With 10% tolerance, 5% drift should pass
    with (
        patch("check_test_count_badge.get_repo_root", return_value=tmp_path),
        patch("check_test_count_badge.count_test_files", return_value=200),
        patch("sys.argv", ["check_test_count_badge.py", "--tolerance", "0.10"]),
    ):
        assert main() == 0
