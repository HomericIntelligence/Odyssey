#!/usr/bin/env python3
"""Tests for scripts/check_bare_mojo_calls.py."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_bare_mojo_calls import BARE_PATTERN, SUPPRESSION, check_file, main


@pytest.fixture
def tmp_workflow(tmp_path: Path):
    """Return a factory for creating temporary workflow YAML files."""

    def _make(content: str) -> Path:
        f = tmp_path / "test.yml"
        f.write_text(content)
        return f

    return _make


class TestBarePattern:
    def test_pattern_compiled(self) -> None:
        """BARE_PATTERN compiles without error."""
        assert BARE_PATTERN is not None

    def test_suppression_constant(self) -> None:
        """SUPPRESSION constant has expected value."""
        assert SUPPRESSION == "# no-bare-mojo-lint"


class TestCheckFile:
    def test_bare_mojo_test_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          if pixi run mojo test -I . tests/\n")
        violations = check_file(f)
        assert len(violations) == 1
        assert violations[0][0] == 1

    def test_bare_mojo_run_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          if pixi run mojo run -I . bench.mojo\n")
        violations = check_file(f)
        assert len(violations) == 1
        assert violations[0][0] == 1

    def test_version_not_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          pixi run mojo --version\n")
        violations = check_file(f)
        assert violations == []

    def test_build_not_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          pixi run mojo build -I . src/main.mojo\n")
        violations = check_file(f)
        assert violations == []

    def test_package_not_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          pixi run mojo package src/shared -o shared.mojopkg\n")
        violations = check_file(f)
        assert violations == []

    def test_format_not_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow("          pixi run mojo format src/\n")
        violations = check_file(f)
        assert violations == []

    def test_hyphenated_not_flagged(self, tmp_workflow) -> None:
        """pixi run mojo-format should not be flagged (near-miss case)."""
        f = tmp_workflow("          pixi run mojo-format src/\n")
        violations = check_file(f)
        assert violations == []

    def test_suppression_skips_line(self, tmp_workflow) -> None:
        f = tmp_workflow(
            "          if pixi run mojo test -I . tests/  # no-bare-mojo-lint: inside retry loop\n"
        )
        violations = check_file(f)
        assert violations == []

    def test_just_test_group_not_flagged(self, tmp_workflow) -> None:
        f = tmp_workflow('          just test-group "tests/shared/core" "test_*.mojo"\n')
        violations = check_file(f)
        assert violations == []

    def test_empty_file(self, tmp_workflow) -> None:
        f = tmp_workflow("")
        violations = check_file(f)
        assert violations == []

    def test_multiple_violations(self, tmp_workflow) -> None:
        content = "\n".join(
            [
                "          if pixi run mojo test -I . tests/unit/",
                "          if pixi run mojo test -I . tests/integration/",
                "          if pixi run mojo run -I . bench.mojo",
            ]
        )
        f = tmp_workflow(content)
        violations = check_file(f)
        assert len(violations) == 3

    def test_violation_line_numbers(self, tmp_workflow) -> None:
        """Line numbers in violations are 1-indexed and correct."""
        content = "line one\n          if pixi run mojo test tests/\nline three\n"
        f = tmp_workflow(content)
        violations = check_file(f)
        assert len(violations) == 1
        assert violations[0][0] == 2

    def test_violation_contains_line_content(self, tmp_workflow) -> None:
        line = "          if pixi run mojo test -I . tests/unit/ --verbose"
        f = tmp_workflow(line + "\n")
        violations = check_file(f)
        assert len(violations) == 1
        assert violations[0][1] == line

    def test_mixed_violations_and_clean(self, tmp_workflow) -> None:
        content = "\n".join(
            [
                "          pixi run mojo --version",
                "          if pixi run mojo test -I . tests/",
                "          just test-group tests/ test_*.mojo",
                "          if pixi run mojo run bench.mojo  # no-bare-mojo-lint: benchmark",
            ]
        )
        f = tmp_workflow(content)
        violations = check_file(f)
        assert len(violations) == 1
        assert "mojo test" in violations[0][1]

    def test_nonexistent_file_skipped_in_main(self, tmp_path: Path) -> None:
        """main() skips files that do not exist."""
        result = main([str(tmp_path / "nonexistent.yml")])
        assert result == 0


class TestMain:
    def test_main_no_files_returns_zero(self) -> None:
        result = main([])
        assert result == 0

    def test_main_clean_file_returns_zero(self, tmp_workflow) -> None:
        f = tmp_workflow("          pixi run mojo --version\n")
        result = main([str(f)])
        assert result == 0

    def test_main_violation_returns_one(self, tmp_workflow, capsys) -> None:
        f = tmp_workflow("          if pixi run mojo test -I . tests/\n")
        result = main([str(f)])
        assert result == 1

    def test_main_prints_violation_path(self, tmp_workflow, capsys) -> None:
        f = tmp_workflow("          if pixi run mojo test -I . tests/\n")
        main([str(f)])
        captured = capsys.readouterr()
        assert str(f) in captured.out
        assert "bare pixi run mojo call" in captured.out

    def test_main_multiple_files(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.yml"
        clean.write_text("          pixi run mojo --version\n")
        dirty = tmp_path / "dirty.yml"
        dirty.write_text("          if pixi run mojo test tests/\n")
        result = main([str(clean), str(dirty)])
        assert result == 1

    def test_main_all_suppressed_returns_zero(self, tmp_workflow) -> None:
        f = tmp_workflow(
            "          if pixi run mojo test tests/  # no-bare-mojo-lint: retry handled above\n"
        )
        result = main([str(f)])
        assert result == 0
