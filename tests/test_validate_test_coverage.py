#!/usr/bin/env python3
"""
Unit tests for scripts/validate_test_coverage.py

Tests the coverage tracking logic, with special focus on split file naming
conventions (part1/part2 and suffix variants like _cmd_run, _parser).
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_test_coverage import find_test_files, expand_pattern, check_coverage


MINIMAL_CI_GROUPS = {
    "Utils": {"path": "tests/shared/utils", "pattern": "test_*.mojo"},
}


class TestFindTestFiles(unittest.TestCase):
    """Test suite for find_test_files function."""

    def setUp(self) -> None:
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_file(self, rel_path: str) -> Path:
        """Create a stub test file at the given relative path."""
        full_path = self.test_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("# stub\n")
        return full_path

    def test_find_empty_directory(self) -> None:
        """No test files in an empty directory."""
        files = find_test_files(self.test_dir)
        self.assertEqual(files, [])

    def test_find_canonical_test_file(self) -> None:
        """A canonical test file is discovered."""
        self._create_file("tests/shared/utils/test_arg_parser.mojo")
        files = find_test_files(self.test_dir)
        self.assertEqual(len(files), 1)
        self.assertIn(Path("tests/shared/utils/test_arg_parser.mojo"), files)

    def test_find_part1_part2_split_files(self) -> None:
        """Part1/part2 split test files are both discovered."""
        self._create_file("tests/shared/utils/test_arg_parser_part1.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_part2.mojo")
        files = find_test_files(self.test_dir)
        self.assertEqual(len(files), 2)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_part1.mojo"), files)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_part2.mojo"), files)

    def test_find_suffix_variant_naming(self) -> None:
        """Suffix variant test files (e.g. _cmd_run, _parser) are discovered."""
        self._create_file("tests/shared/utils/test_arg_parser_cmd_run.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_parser.mojo")
        files = find_test_files(self.test_dir)
        self.assertEqual(len(files), 2)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_cmd_run.mojo"), files)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_parser.mojo"), files)


class TestExpandPattern(unittest.TestCase):
    """Test suite for expand_pattern function."""

    def setUp(self) -> None:
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_file(self, rel_path: str) -> Path:
        """Create a stub test file at the given relative path."""
        full_path = self.test_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("# stub\n")
        return full_path

    def test_wildcard_matches_canonical_file(self) -> None:
        """Wildcard pattern test_*.mojo matches a canonical filename."""
        self._create_file("tests/shared/utils/test_arg_parser.mojo")
        matched = expand_pattern("tests/shared/utils", "test_*.mojo", self.test_dir)
        self.assertIn(Path("tests/shared/utils/test_arg_parser.mojo"), matched)

    def test_wildcard_matches_part1_part2(self) -> None:
        """Wildcard pattern test_*.mojo matches part1 and part2 split files."""
        self._create_file("tests/shared/utils/test_arg_parser_part1.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_part2.mojo")
        matched = expand_pattern("tests/shared/utils", "test_*.mojo", self.test_dir)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_part1.mojo"), matched)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_part2.mojo"), matched)

    def test_wildcard_matches_suffix_variants(self) -> None:
        """Wildcard pattern test_*.mojo matches suffix variant filenames."""
        self._create_file("tests/shared/utils/test_arg_parser_cmd_run.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_parser.mojo")
        matched = expand_pattern("tests/shared/utils", "test_*.mojo", self.test_dir)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_cmd_run.mojo"), matched)
        self.assertIn(Path("tests/shared/utils/test_arg_parser_parser.mojo"), matched)

    def test_no_match_wrong_directory(self) -> None:
        """Files in a different directory are not matched."""
        self._create_file("tests/shared/other/test_arg_parser_part1.mojo")
        matched = expand_pattern("tests/shared/utils", "test_*.mojo", self.test_dir)
        self.assertEqual(len(matched), 0)


class TestCheckCoverage(unittest.TestCase):
    """Test suite for check_coverage function."""

    def setUp(self) -> None:
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_file(self, rel_path: str) -> Path:
        """Create a stub test file at the given relative path."""
        full_path = self.test_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("# stub\n")
        return full_path

    def test_split_files_in_covered_directory_are_covered(self) -> None:
        """Part1/part2 split files in a covered directory are not reported as uncovered."""
        self._create_file("tests/shared/utils/test_arg_parser_part1.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_part2.mojo")

        test_files = [
            Path("tests/shared/utils/test_arg_parser_part1.mojo"),
            Path("tests/shared/utils/test_arg_parser_part2.mojo"),
        ]
        uncovered, _ = check_coverage(test_files, MINIMAL_CI_GROUPS, self.test_dir)
        self.assertEqual(len(uncovered), 0)

    def test_split_files_in_uncovered_directory_are_flagged(self) -> None:
        """Part1/part2 split files in a directory with no CI group are flagged as uncovered."""
        self._create_file("tests/shared/other/test_arg_parser_part1.mojo")
        self._create_file("tests/shared/other/test_arg_parser_part2.mojo")

        test_files = [
            Path("tests/shared/other/test_arg_parser_part1.mojo"),
            Path("tests/shared/other/test_arg_parser_part2.mojo"),
        ]
        uncovered, _ = check_coverage(test_files, MINIMAL_CI_GROUPS, self.test_dir)
        self.assertEqual(len(uncovered), 2)
        self.assertIn(Path("tests/shared/other/test_arg_parser_part1.mojo"), uncovered)
        self.assertIn(Path("tests/shared/other/test_arg_parser_part2.mojo"), uncovered)

    def test_canonical_file_in_covered_directory_is_covered(self) -> None:
        """A canonical (non-split) test file in a covered directory is not flagged."""
        self._create_file("tests/shared/utils/test_arg_parser.mojo")

        test_files = [Path("tests/shared/utils/test_arg_parser.mojo")]
        uncovered, _ = check_coverage(test_files, MINIMAL_CI_GROUPS, self.test_dir)
        self.assertEqual(len(uncovered), 0)

    def test_suffix_variant_files_in_covered_directory_are_covered(self) -> None:
        """Suffix variant files (_cmd_run, _parser) in a covered directory are not flagged."""
        self._create_file("tests/shared/utils/test_arg_parser_cmd_run.mojo")
        self._create_file("tests/shared/utils/test_arg_parser_parser.mojo")

        test_files = [
            Path("tests/shared/utils/test_arg_parser_cmd_run.mojo"),
            Path("tests/shared/utils/test_arg_parser_parser.mojo"),
        ]
        uncovered, _ = check_coverage(test_files, MINIMAL_CI_GROUPS, self.test_dir)
        self.assertEqual(len(uncovered), 0)

    def test_mixed_covered_and_uncovered(self) -> None:
        """Only files outside covered directories are reported as uncovered."""
        self._create_file("tests/shared/utils/test_arg_parser_part1.mojo")
        self._create_file("tests/shared/other/test_something_part1.mojo")

        test_files = [
            Path("tests/shared/utils/test_arg_parser_part1.mojo"),
            Path("tests/shared/other/test_something_part1.mojo"),
        ]
        uncovered, _ = check_coverage(test_files, MINIMAL_CI_GROUPS, self.test_dir)
        self.assertEqual(len(uncovered), 1)
        self.assertIn(Path("tests/shared/other/test_something_part1.mojo"), uncovered)


if __name__ == "__main__":
    unittest.main()
