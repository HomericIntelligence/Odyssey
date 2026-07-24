#!/usr/bin/env python3
"""Tests for scripts/check_note_format.py NOTE format compliance checker."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_note_format import NOTE_VIOLATION_PATTERN, find_violations, is_excluded, scan_source_dirs


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test .mojo files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def make_mojo_file(directory: Path, name: str, content: str) -> Path:
    """Helper to create a .mojo file with given content."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


class TestNoteViolationPattern:
    """Test the regex pattern used to detect violations."""

    def test_detects_note_colon(self):
        """Plain # NOTE: without parentheses is a violation."""
        assert NOTE_VIOLATION_PATTERN.search("    # NOTE: some text")

    def test_detects_note_space_text(self):
        """# NOTE followed by space and text (no parens) is a violation."""
        assert NOTE_VIOLATION_PATTERN.search("    # NOTE some text without version")

    def test_does_not_flag_compliant_format(self):
        """# NOTE (Mojo vX.Y.Z): is compliant and must not be flagged."""
        assert not NOTE_VIOLATION_PATTERN.search("    # NOTE (Mojo v0.26.1): explanation")

    def test_does_not_flag_compliant_with_issue(self):
        """# NOTE (Mojo vX.Y.Z, #1234): is compliant."""
        assert not NOTE_VIOLATION_PATTERN.search("    # NOTE (Mojo v0.26.1, #3092): reason")

    def test_does_not_flag_note_with_open_paren(self):
        """Any # NOTE( pattern is compliant (has opening paren)."""
        assert not NOTE_VIOLATION_PATTERN.search("    # NOTE(#3092): issue ref")

    def test_detects_note_without_space(self):
        """# NOTE followed by any non-paren char is a violation."""
        assert NOTE_VIOLATION_PATTERN.search("    # NOTEsomething")


class TestIsExcluded:
    """Test directory exclusion logic."""

    def test_excludes_worktrees(self):
        """Paths inside .worktrees should be excluded."""
        path = Path("/repo/.worktrees/issue-123/shared/foo.mojo")
        assert is_excluded(path)

    def test_excludes_pixi(self):
        """Paths inside .pixi should be excluded."""
        path = Path("/repo/.pixi/env/lib/foo.mojo")
        assert is_excluded(path)

    def test_excludes_git(self):
        """Paths inside .git should be excluded."""
        path = Path("/repo/.git/hooks/foo.mojo")
        assert is_excluded(path)

    def test_excludes_build(self):
        """Paths inside build/ should be excluded."""
        path = Path("/repo/build/output/foo.mojo")
        assert is_excluded(path)

    def test_does_not_exclude_shared(self):
        """Paths inside src/odyssey/ should not be excluded."""
        path = Path("/repo/shared/core/foo.mojo")
        assert not is_excluded(path)

    def test_does_not_exclude_tests(self):
        """Paths inside tests/ should not be excluded."""
        path = Path("/repo/tests/models/foo.mojo")
        assert not is_excluded(path)


class TestFindViolations:
    """Test find_violations() scanning logic."""

    def test_no_violations_in_clean_directory(self, temp_dir):
        """Returns empty list when all NOTE comments are compliant."""
        make_mojo_file(
            temp_dir,
            "clean.mojo",
            "    # NOTE (Mojo v0.26.1): This is compliant\n    var x = 1\n",
        )
        assert find_violations(temp_dir) == []

    def test_detects_single_violation(self, temp_dir):
        """Detects a single non-compliant NOTE comment."""
        make_mojo_file(
            temp_dir,
            "bad.mojo",
            "    var x = 1\n    # NOTE: missing version\n    var y = 2\n",
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        file_path, line_num, line_content = violations[0]
        assert file_path.name == "bad.mojo"
        assert line_num == 2
        assert "# NOTE: missing version" in line_content

    def test_detects_multiple_violations_in_one_file(self, temp_dir):
        """Detects multiple violations within a single file."""
        make_mojo_file(
            temp_dir,
            "multi.mojo",
            "# NOTE: first bad\n# NOTE (Mojo v0.26.1): ok\n# NOTE: second bad\n",
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 2
        assert violations[0][1] == 1
        assert violations[1][1] == 3

    def test_detects_violations_in_subdirectory(self, temp_dir):
        """Recursively finds violations in subdirectories."""
        subdir = temp_dir / "sub"
        subdir.mkdir()
        make_mojo_file(subdir, "nested.mojo", "# NOTE: nested violation\n")
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert violations[0][0].name == "nested.mojo"

    def test_ignores_excluded_directories(self, temp_dir):
        """Skips .worktrees, .pixi, build, .git subdirectories."""
        for excluded in [".worktrees", ".pixi", "build", ".git"]:
            excl_dir = temp_dir / excluded
            excl_dir.mkdir()
            make_mojo_file(excl_dir, "should_skip.mojo", "# NOTE: violation in excluded\n")
        violations = find_violations(temp_dir)
        assert violations == []

    def test_ignores_non_mojo_files(self, temp_dir):
        """Only scans .mojo files; ignores .py, .txt, etc."""
        (temp_dir / "not_mojo.py").write_text("# NOTE: this is python\n")
        (temp_dir / "also_not.txt").write_text("# NOTE: plain text\n")
        violations = find_violations(temp_dir)
        assert violations == []

    def test_mixed_compliant_and_violations(self, temp_dir):
        """Only returns violations, not compliant lines."""
        make_mojo_file(
            temp_dir,
            "mixed.mojo",
            "# NOTE (Mojo v0.26.1): compliant\n# NOTE: violation\n# NOTE (Mojo v1.0.0): also fine\n",
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert "# NOTE: violation" in violations[0][2]

    def test_returns_correct_line_numbers(self, temp_dir):
        """Line numbers in violations are 1-based."""
        make_mojo_file(
            temp_dir,
            "lines.mojo",
            "var a = 1\nvar b = 2\n# NOTE: bad line 3\nvar c = 3\n",
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert violations[0][1] == 3

    def test_empty_directory(self, temp_dir):
        """Returns empty list for a directory with no .mojo files."""
        assert find_violations(temp_dir) == []


class TestScanSourceDirs:
    """Test scan_source_dirs() with a mock repo layout."""

    def test_scans_existing_source_dirs(self, temp_dir):
        """Scans all present source dirs and aggregates violations."""
        shared_dir = temp_dir / "src" / "odyssey"
        shared_dir.mkdir(parents=True)
        make_mojo_file(shared_dir, "foo.mojo", "# NOTE: violation in shared\n")

        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        make_mojo_file(tests_dir, "test_foo.mojo", "# NOTE (Mojo v0.26.1): ok\n")

        violations = scan_source_dirs(temp_dir)
        assert len(violations) == 1
        assert violations[0][0].name == "foo.mojo"

    def test_skips_missing_source_dirs(self, temp_dir):
        """Does not fail when source dirs are absent."""
        # temp_dir has no source dirs at all
        violations = scan_source_dirs(temp_dir)
        assert violations == []


class TestMainExitCodes:
    """Test main() exit code behavior via subprocess."""

    def test_exit_zero_when_clean(self, temp_dir):
        """main() returns 0 when no violations found."""
        make_mojo_file(temp_dir, "clean.mojo", "# NOTE (Mojo v0.26.1): fine\n")
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/check_note_format.py", str(temp_dir)],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0

    def test_exit_one_when_violations(self, temp_dir):
        """main() returns 1 when violations are found."""
        make_mojo_file(temp_dir, "bad.mojo", "# NOTE: no version\n")
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/check_note_format.py", str(temp_dir)],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 1

    def test_violation_printed_to_stdout(self, temp_dir):
        """Violations are printed as file:line: content to stdout."""
        make_mojo_file(temp_dir, "bad.mojo", "# NOTE: no version\n")
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/check_note_format.py", str(temp_dir)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert "bad.mojo" in result.stdout
        assert ":1:" in result.stdout

    def test_summary_printed_to_stderr(self, temp_dir):
        """Summary count is printed to stderr on failure."""
        make_mojo_file(temp_dir, "bad.mojo", "# NOTE: no version\n")
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/check_note_format.py", str(temp_dir)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert "violation" in result.stderr

    def test_exit_one_for_nonexistent_directory(self):
        """main() returns 1 for a nonexistent directory argument."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/check_note_format.py", "/nonexistent/path/xyz"],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
