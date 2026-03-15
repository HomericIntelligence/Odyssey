#!/usr/bin/env python3
"""
Unit tests for scripts/check_test_count.py

Verifies ADR-009 Phase 2 pre-commit hook behaviour: count `fn test_` per
Mojo test file and fail if any file exceeds the 10-test limit.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_test_count import LIMIT, check_files, count_tests_in_file, is_mojo_test_file


def _write(directory: str, filename: str, content: str) -> Path:
    """Write *content* to *directory/filename* and return the Path."""
    path = Path(directory) / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestIsMojoTestFile(unittest.TestCase):
    """Tests for is_mojo_test_file()."""

    def test_mojo_under_tests_dir(self) -> None:
        """Accepts a .mojo file whose path contains tests/."""
        self.assertTrue(is_mojo_test_file(Path("tests/models/test_conv.mojo")))

    def test_absolute_path_under_tests(self) -> None:
        """Accepts an absolute path that contains /tests/."""
        self.assertTrue(is_mojo_test_file(Path("/repo/tests/test_foo.mojo")))

    def test_not_mojo_extension(self) -> None:
        """Rejects non-Mojo files even when under tests/."""
        self.assertFalse(is_mojo_test_file(Path("tests/test_foo.py")))

    def test_mojo_outside_tests_dir(self) -> None:
        """Rejects .mojo files not under a tests/ directory."""
        self.assertFalse(is_mojo_test_file(Path("shared/core/tensor.mojo")))

    def test_empty_path(self) -> None:
        """Rejects an empty path."""
        self.assertFalse(is_mojo_test_file(Path("")))

    def test_just_filename_no_tests_parent(self) -> None:
        """Rejects a bare filename with no directory component."""
        self.assertFalse(is_mojo_test_file(Path("test_foo.mojo")))


class TestCountTestsInFile(unittest.TestCase):
    """Tests for count_tests_in_file()."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def test_counts_fn_test_functions(self) -> None:
        """Counts standard `fn test_` definitions."""
        path = _write(
            self._tmpdir,
            "test_foo.mojo",
            "fn test_a():\n    pass\nfn test_b():\n    pass\n",
        )
        self.assertEqual(count_tests_in_file(path), 2)

    def test_counts_indented_fn_test(self) -> None:
        """Counts indented `fn test_` (e.g. inside a struct)."""
        path = _write(
            self._tmpdir,
            "test_indent.mojo",
            "struct S:\n    fn test_inner():\n        pass\n",
        )
        self.assertEqual(count_tests_in_file(path), 1)

    def test_ignores_non_test_fns(self) -> None:
        """Does not count non-test functions."""
        path = _write(
            self._tmpdir,
            "test_nonfn.mojo",
            "fn helper():\n    pass\nfn run():\n    pass\n",
        )
        self.assertEqual(count_tests_in_file(path), 0)

    def test_does_not_count_string_mention(self) -> None:
        """Ignores `fn test_` in string literals / comments."""
        path = _write(
            self._tmpdir,
            "test_str.mojo",
            '# fn test_fake is not a real test\nlet s = "fn test_x"\n',
        )
        self.assertEqual(count_tests_in_file(path), 0)

    def test_returns_zero_for_missing_file(self) -> None:
        """Returns 0 (with a warning) for a non-existent file."""
        missing = Path(self._tmpdir) / "nonexistent.mojo"
        self.assertEqual(count_tests_in_file(missing), 0)

    def test_empty_file(self) -> None:
        """Returns 0 for an empty file."""
        path = _write(self._tmpdir, "test_empty.mojo", "")
        self.assertEqual(count_tests_in_file(path), 0)


class TestCheckFiles(unittest.TestCase):
    """Tests for check_files() exit-code logic."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        # Create a tests/ subdirectory so is_mojo_test_file passes
        self._tests_dir = Path(self._tmpdir) / "tests"
        self._tests_dir.mkdir()

    def _mojo(self, filename: str, n_tests: int) -> str:
        """Write a .mojo file with n_tests `fn test_` functions; return its path."""
        fns = "".join(f"fn test_{i}():\n    pass\n" for i in range(n_tests))
        path = self._tests_dir / filename
        path.write_text(fns, encoding="utf-8")
        return str(path)

    def test_empty_argv_exits_zero(self) -> None:
        """No files → exit 0."""
        self.assertEqual(check_files([]), 0)

    def test_file_at_limit_passes(self) -> None:
        """File with exactly LIMIT tests → exit 0."""
        p = self._mojo("test_at_limit.mojo", LIMIT)
        self.assertEqual(check_files([p]), 0)

    def test_file_below_limit_passes(self) -> None:
        """File with fewer than LIMIT tests → exit 0."""
        p = self._mojo("test_below.mojo", LIMIT - 1)
        self.assertEqual(check_files([p]), 0)

    def test_file_above_limit_fails(self) -> None:
        """File with more than LIMIT tests → exit 1."""
        p = self._mojo("test_over.mojo", LIMIT + 1)
        self.assertEqual(check_files([p]), 1)

    def test_non_test_mojo_is_skipped(self) -> None:
        """Non-test .mojo file (not under tests/) is silently skipped."""
        # Write file directly in tmpdir, not under tests/
        other = Path(self._tmpdir) / "shared" / "core.mojo"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("fn test_a():\n    pass\n" * (LIMIT + 5), encoding="utf-8")
        self.assertEqual(check_files([str(other)]), 0)

    def test_non_mojo_file_is_skipped(self) -> None:
        """Python files are skipped even if passed explicitly."""
        py_file = self._tests_dir / "test_foo.py"
        py_file.write_text("def test_a(): pass\n" * (LIMIT + 5), encoding="utf-8")
        self.assertEqual(check_files([str(py_file)]), 0)

    def test_mixed_files_one_violation(self) -> None:
        """One violating file among several valid files → exit 1."""
        ok = self._mojo("test_ok.mojo", LIMIT)
        bad = self._mojo("test_bad.mojo", LIMIT + 2)
        self.assertEqual(check_files([ok, bad]), 1)

    def test_all_files_pass(self) -> None:
        """Multiple files all within limit → exit 0."""
        files = [self._mojo(f"test_part{i}.mojo", 5) for i in range(3)]
        self.assertEqual(check_files(files), 0)

    def test_limit_constant_is_ten(self) -> None:
        """LIMIT must equal 10 per ADR-009."""
        self.assertEqual(LIMIT, 10)


if __name__ == "__main__":
    unittest.main()
