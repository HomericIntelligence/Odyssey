#!/usr/bin/env python3
"""Tests for scripts/bench_precommit.py."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from bench_precommit import (
    check_threshold,
    emit_warning,
    format_summary_table,
    main,
    write_step_summary,
)


class TestFormatSummaryTable:
    """Tests for format_summary_table()."""

    def test_contains_elapsed(self) -> None:
        result = format_summary_table(45, 100, "passed")
        assert "45s" in result

    def test_contains_file_count(self) -> None:
        result = format_summary_table(45, 300, "passed")
        assert "300" in result

    def test_contains_passed_status(self) -> None:
        result = format_summary_table(45, 100, "passed")
        assert "passed" in result

    def test_contains_failed_status(self) -> None:
        result = format_summary_table(45, 100, "failed")
        assert "failed" in result

    def test_passed_has_checkmark(self) -> None:
        result = format_summary_table(45, 100, "passed")
        assert "✅" in result

    def test_failed_has_cross(self) -> None:
        result = format_summary_table(45, 100, "failed")
        assert "❌" in result

    def test_is_markdown_table(self) -> None:
        result = format_summary_table(45, 100, "passed")
        assert "|" in result
        assert "---" in result

    def test_ends_with_newline(self) -> None:
        result = format_summary_table(45, 100, "passed")
        assert result.endswith("\n")


class TestCheckThreshold:
    """Tests for check_threshold()."""

    def test_fast_returns_false(self) -> None:
        assert check_threshold(60, threshold_s=120) is False

    def test_slow_returns_true(self) -> None:
        assert check_threshold(150, threshold_s=120) is True

    def test_exactly_at_threshold_returns_false(self) -> None:
        assert check_threshold(120, threshold_s=120) is False

    def test_default_threshold_is_120(self) -> None:
        assert check_threshold(119) is False
        assert check_threshold(121) is True


class TestEmitWarning:
    """Tests for emit_warning()."""

    def test_warning_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_warning("Hook is slow")
        captured = capsys.readouterr()
        assert captured.out.startswith("::warning::")

    def test_message_included(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_warning("Hook is slow")
        captured = capsys.readouterr()
        assert "Hook is slow" in captured.out


class TestWriteStepSummary:
    """Tests for write_step_summary()."""

    def test_writes_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as f:
            path = f.name
        write_step_summary("## Summary\n", summary_path=path)
        content = Path(path).read_text()
        assert "## Summary" in content

    def test_no_file_when_path_empty(self) -> None:
        # Should not raise even when no path is configured
        write_step_summary("content", summary_path=None)

    def test_appends_to_existing_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("existing\n")
            path = f.name
        write_step_summary("appended\n", summary_path=path)
        content = Path(path).read_text()
        assert "existing" in content
        assert "appended" in content


class TestMain:
    """Tests for the main() CLI entry-point."""

    def test_exits_zero_when_fast(self) -> None:
        rc = main(["--elapsed", "30", "--files", "100", "--status", "passed"])
        assert rc == 0

    def test_exits_zero_even_when_slow(self) -> None:
        rc = main(["--elapsed", "200", "--files", "100", "--status", "passed"])
        assert rc == 0

    def test_exits_zero_when_failed_status(self) -> None:
        rc = main(["--elapsed", "30", "--files", "100", "--status", "failed"])
        assert rc == 0

    def test_warning_emitted_when_slow(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--elapsed", "200", "--threshold", "120", "--status", "passed"])
        captured = capsys.readouterr()
        assert "::warning::" in captured.out

    def test_no_warning_when_fast(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--elapsed", "50", "--threshold", "120", "--status", "passed"])
        captured = capsys.readouterr()
        assert "::warning::" not in captured.out

    def test_custom_threshold_respected(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--elapsed", "60", "--threshold", "30", "--status", "passed"])
        captured = capsys.readouterr()
        assert "::warning::" in captured.out

    def test_subprocess_exits_zero(self) -> None:
        """Verify script exits 0 even when slow, via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "scripts" / "bench_precommit.py"),
                "--elapsed",
                "200",
                "--files",
                "100",
                "--status",
                "passed",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
