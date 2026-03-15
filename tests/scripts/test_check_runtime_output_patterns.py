#!/usr/bin/env python3
"""Tests for scripts/check_runtime_output_patterns.py misleading output pattern checker."""

import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_runtime_output_patterns import (
    BANNED_PATTERNS,
    find_violations,
    is_comment_line,
    is_excluded,
    scan_source_dirs,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test .mojo files."""
    return tmp_path


def make_mojo_file(directory: Path, name: str, content: str) -> Path:
    """Create a .mojo file with the given content."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


class TestBannedPatterns:
    """Test that the banned regex patterns match the right strings."""

    def test_detects_warning_colon(self) -> None:
        """print() with WARNING: is a violation."""
        line = '    print("WARNING: something bad happened")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_warning_case_insensitive(self) -> None:
        """WARNING: is matched case-insensitively."""
        line = '    print("warning: lowercase")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_hack_colon(self) -> None:
        """print() with HACK: is a violation."""
        line = '    print("HACK: this is a workaround")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_xxx_colon(self) -> None:
        """print() with XXX: is a violation."""
        line = '    print("XXX: fix this")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_not_implemented(self) -> None:
        """print() with 'Not implemented' is a violation."""
        line = '    print("Not implemented")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_not_implemented_case_insensitive(self) -> None:
        """'not implemented' is matched case-insensitively."""
        line = '    print("not implemented yet")'
        assert any(p.search(line) for p in BANNED_PATTERNS)

    def test_does_not_flag_clean_print(self) -> None:
        """A print() without any banned pattern is not flagged."""
        line = '    print("Gradient overflow detected, skipping parameter update")'
        assert not any(p.search(line) for p in BANNED_PATTERNS)

    def test_does_not_flag_warning_without_print(self) -> None:
        """WARNING: outside of a print() call is not flagged."""
        line = "    # WARNING: this is a comment"
        assert not any(p.search(line) for p in BANNED_PATTERNS)

    def test_detects_not_implemented_with_extra_words(self) -> None:
        """'Not implemented' anywhere in a print() is caught."""
        line = '    print("Function foo is not implemented yet")'
        assert any(p.search(line) for p in BANNED_PATTERNS)


class TestIsCommentLine:
    """Test comment-line detection."""

    def test_detects_hash_comment(self) -> None:
        """Lines starting with # (after whitespace) are comments."""
        assert is_comment_line("    # WARNING: this is a comment")

    def test_detects_comment_no_whitespace(self) -> None:
        """Lines starting with # at column 0 are comments."""
        assert is_comment_line("# WARNING: comment")

    def test_does_not_flag_code_line(self) -> None:
        """Actual print() calls are not comments."""
        assert not is_comment_line('    print("WARNING: real violation")')

    def test_does_not_flag_empty_line(self) -> None:
        """Empty lines are not comments."""
        assert not is_comment_line("")

    def test_does_not_flag_blank_line(self) -> None:
        """Lines with only whitespace are not comments."""
        assert not is_comment_line("    ")


class TestIsExcluded:
    """Test directory exclusion logic."""

    def test_excludes_worktrees(self) -> None:
        """Paths inside .worktrees should be excluded."""
        assert is_excluded(Path("/repo/.worktrees/issue-123/examples/foo.mojo"))

    def test_excludes_pixi(self) -> None:
        """Paths inside .pixi should be excluded."""
        assert is_excluded(Path("/repo/.pixi/env/lib/foo.mojo"))

    def test_excludes_git(self) -> None:
        """Paths inside .git should be excluded."""
        assert is_excluded(Path("/repo/.git/hooks/foo.mojo"))

    def test_excludes_build(self) -> None:
        """Paths inside build/ should be excluded."""
        assert is_excluded(Path("/repo/build/output/foo.mojo"))

    def test_does_not_exclude_examples(self) -> None:
        """Paths inside examples/ are not excluded."""
        assert not is_excluded(Path("/repo/examples/lenet/run_train.mojo"))

    def test_does_not_exclude_tests(self) -> None:
        """Paths inside tests/ are not excluded."""
        assert not is_excluded(Path("/repo/tests/models/foo.mojo"))


class TestFindViolations:
    """Test find_violations() scanning logic."""

    def test_no_violations_in_clean_directory(self, temp_dir: Path) -> None:
        """Returns empty list for a clean directory."""
        make_mojo_file(
            temp_dir,
            "clean.mojo",
            'print("Gradient overflow detected, skipping parameter update")\n',
        )
        assert find_violations(temp_dir) == []

    def test_detects_warning_violation(self, temp_dir: Path) -> None:
        """Detects WARNING: in a print() call."""
        make_mojo_file(
            temp_dir,
            "warn.mojo",
            'print("WARNING: something went wrong")\n',
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert violations[0][0].name == "warn.mojo"
        assert violations[0][1] == 1

    def test_detects_hack_violation(self, temp_dir: Path) -> None:
        """Detects HACK: in a print() call."""
        make_mojo_file(temp_dir, "hack.mojo", 'print("HACK: dirty workaround")\n')
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert "HACK" in violations[0][2]

    def test_detects_xxx_violation(self, temp_dir: Path) -> None:
        """Detects XXX: in a print() call."""
        make_mojo_file(temp_dir, "xxx.mojo", 'print("XXX: fix me")\n')
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert "XXX" in violations[0][2]

    def test_detects_not_implemented_violation(self, temp_dir: Path) -> None:
        """Detects 'Not implemented' in a print() call."""
        make_mojo_file(temp_dir, "notimpl.mojo", 'print("Not implemented")\n')
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert "Not implemented" in violations[0][2]

    def test_ignores_comment_lines(self, temp_dir: Path) -> None:
        """Comment lines with banned patterns are not flagged."""
        make_mojo_file(
            temp_dir,
            "commented.mojo",
            '# print("WARNING: this is commented out")\n',
        )
        assert find_violations(temp_dir) == []

    def test_detects_multiple_violations_in_one_file(self, temp_dir: Path) -> None:
        """Detects multiple violations in a single file."""
        make_mojo_file(
            temp_dir,
            "multi.mojo",
            'print("WARNING: first")\nprint("clean")\nprint("HACK: second")\n',
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 2
        assert violations[0][1] == 1
        assert violations[1][1] == 3

    def test_detects_violations_in_subdirectory(self, temp_dir: Path) -> None:
        """Recursively finds violations in subdirectories."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        make_mojo_file(subdir, "nested.mojo", 'print("XXX: nested violation")\n')
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert violations[0][0].name == "nested.mojo"

    def test_ignores_excluded_directories(self, temp_dir: Path) -> None:
        """Skips .worktrees, .pixi, build, .git subdirectories."""
        for excluded in [".worktrees", ".pixi", "build", ".git"]:
            excl_dir = temp_dir / excluded
            excl_dir.mkdir()
            make_mojo_file(excl_dir, "skip.mojo", 'print("WARNING: in excluded")\n')
        assert find_violations(temp_dir) == []

    def test_ignores_non_mojo_files(self, temp_dir: Path) -> None:
        """Only scans .mojo files; ignores .py, .txt, etc."""
        (temp_dir / "not_mojo.py").write_text('print("WARNING: python file")\n')
        (temp_dir / "also_not.txt").write_text('print("HACK: text file")\n')
        assert find_violations(temp_dir) == []

    def test_empty_directory(self, temp_dir: Path) -> None:
        """Returns empty list for a directory with no .mojo files."""
        assert find_violations(temp_dir) == []

    def test_each_line_reported_once(self, temp_dir: Path) -> None:
        """A line matching multiple patterns is only reported once."""
        # Contrived: WARNING: and HACK: in same print (unusual but possible)
        make_mojo_file(
            temp_dir,
            "multi_match.mojo",
            'print("WARNING: HACK: double")\n',
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1

    def test_returns_correct_line_numbers(self, temp_dir: Path) -> None:
        """Line numbers in violations are 1-based."""
        make_mojo_file(
            temp_dir,
            "lines.mojo",
            "var a = 1\nvar b = 2\n" + 'print("WARNING: on line 3")\n',
        )
        violations = find_violations(temp_dir)
        assert len(violations) == 1
        assert violations[0][1] == 3


class TestScanSourceDirs:
    """Test scan_source_dirs() with a mock repo layout."""

    def test_scans_existing_source_dirs(self, temp_dir: Path) -> None:
        """Scans all present source dirs and aggregates violations."""
        examples_dir = temp_dir / "examples"
        examples_dir.mkdir()
        make_mojo_file(examples_dir, "bad.mojo", 'print("WARNING: example violation")\n')

        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        make_mojo_file(tests_dir, "ok.mojo", 'print("clean output")\n')

        violations = scan_source_dirs(temp_dir)
        assert len(violations) == 1
        assert violations[0][0].name == "bad.mojo"

    def test_skips_missing_source_dirs(self, temp_dir: Path) -> None:
        """Does not fail when source dirs are absent."""
        violations = scan_source_dirs(temp_dir)
        assert violations == []


class TestMainExitCodes:
    """Test main() exit code behavior via subprocess."""

    def _run_script(self, *args: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        """Run the check_runtime_output_patterns.py script with given args."""
        return subprocess.run(
            [sys.executable, "scripts/check_runtime_output_patterns.py", *args],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

    def test_exit_zero_when_clean(self, temp_dir: Path) -> None:
        """main() returns 0 when no violations found."""
        make_mojo_file(temp_dir, "clean.mojo", 'print("clean output")\n')
        result = self._run_script(str(temp_dir))
        assert result.returncode == 0

    def test_exit_one_on_warning_violation(self, temp_dir: Path) -> None:
        """main() returns 1 when WARNING: violation found."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("WARNING: bad")\n')
        result = self._run_script(str(temp_dir))
        assert result.returncode == 1

    def test_exit_one_on_hack_violation(self, temp_dir: Path) -> None:
        """main() returns 1 when HACK: violation found."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("HACK: workaround")\n')
        result = self._run_script(str(temp_dir))
        assert result.returncode == 1

    def test_exit_one_on_xxx_violation(self, temp_dir: Path) -> None:
        """main() returns 1 when XXX: violation found."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("XXX: todo")\n')
        result = self._run_script(str(temp_dir))
        assert result.returncode == 1

    def test_exit_one_on_not_implemented_violation(self, temp_dir: Path) -> None:
        """main() returns 1 when 'Not implemented' violation found."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("Not implemented")\n')
        result = self._run_script(str(temp_dir))
        assert result.returncode == 1

    def test_comment_line_not_flagged(self, temp_dir: Path) -> None:
        """main() returns 0 when banned pattern only appears in a comment."""
        make_mojo_file(
            temp_dir,
            "commented.mojo",
            '# print("WARNING: commented out")\n',
        )
        result = self._run_script(str(temp_dir))
        assert result.returncode == 0

    def test_violation_printed_to_stdout(self, temp_dir: Path) -> None:
        """Violations are printed as file:line: content to stdout."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("WARNING: output")\n')
        result = self._run_script(str(temp_dir))
        assert "bad.mojo" in result.stdout
        assert ":1:" in result.stdout

    def test_summary_printed_to_stderr(self, temp_dir: Path) -> None:
        """Summary count is printed to stderr on failure."""
        make_mojo_file(temp_dir, "bad.mojo", 'print("HACK: workaround")\n')
        result = self._run_script(str(temp_dir))
        assert "violation" in result.stderr

    def test_exit_one_for_nonexistent_directory(self) -> None:
        """main() returns 1 for a nonexistent directory argument."""
        result = self._run_script("/nonexistent/path/xyz")
        assert result.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
