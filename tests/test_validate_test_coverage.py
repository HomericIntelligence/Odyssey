#!/usr/bin/env python3
"""
Unit tests for scripts/validate_test_coverage.py

Tests the split file naming recognition and coverage validation logic.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import yaml

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_test_coverage import (
    find_test_files,
    parse_ci_matrix,
    expand_pattern,
    check_coverage,
)


class TestSplitFileNaming(unittest.TestCase):
    """Test suite for split file naming recognition in test coverage validation."""

    def setUp(self):
        """Create temporary directory structure for testing."""
        self.test_root = Path(tempfile.mkdtemp())
        self.tests_dir = self.test_root / "tests" / "shared" / "core"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir = self.test_root / ".github" / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_root)

    def _create_workflow_file(self, test_group_path: str, pattern: str):
        """Create a minimal comprehensive-tests.yml workflow file."""
        workflow_content = {
            "name": "comprehensive-tests",
            "on": ["push", "pull_request"],
            "jobs": {
                "test-mojo-comprehensive": {
                    "strategy": {
                        "matrix": {
                            "test-group": [
                                {
                                    "name": "Core Tensors",
                                    "path": test_group_path,
                                    "pattern": pattern,
                                }
                            ]
                        }
                    }
                }
            },
        }

        workflow_file = self.workflows_dir / "comprehensive-tests.yml"
        with open(workflow_file, "w") as f:
            yaml.dump(workflow_content, f)
        return workflow_file

    def test_split_files_in_covered_directory(self):
        """Test that split files in a covered directory are recognized as covered."""
        # Create split test files in the tests directory
        (self.tests_dir / "test_arg_parser_part1.mojo").write_text(
            "fn test_arg_parser_basic() raises:\n    pass\n"
        )
        (self.tests_dir / "test_arg_parser_part2.mojo").write_text(
            "fn test_arg_parser_edge_cases() raises:\n    pass\n"
        )

        # Create workflow that covers this directory with wildcard pattern
        workflow_file = self._create_workflow_file(
            "tests/shared/core", "test_*.mojo"
        )

        # Check coverage
        groups = parse_ci_matrix(workflow_file)
        uncovered = check_coverage(self.test_root, groups)

        # Both split files should be covered by the wildcard pattern
        uncovered_names = {f.name for f in uncovered}
        self.assertNotIn(
            "test_arg_parser_part1.mojo",
            uncovered_names,
            "Split file part1 should be covered by wildcard pattern",
        )
        self.assertNotIn(
            "test_arg_parser_part2.mojo",
            uncovered_names,
            "Split file part2 should be covered by wildcard pattern",
        )

    def test_split_files_in_uncovered_directory(self):
        """Test that split files in an uncovered directory are flagged as uncovered."""
        # Create split files in tests/shared/core
        (self.tests_dir / "test_arg_parser_part1.mojo").write_text(
            "fn test_arg_parser_basic() raises:\n    pass\n"
        )
        (self.tests_dir / "test_arg_parser_part2.mojo").write_text(
            "fn test_arg_parser_edge_cases() raises:\n    pass\n"
        )

        # Create workflow that covers a DIFFERENT directory
        workflow_file = self._create_workflow_file(
            "tests/shared/utils", "test_*.mojo"
        )

        # Check coverage
        groups = parse_ci_matrix(workflow_file)
        uncovered = check_coverage(self.test_root, groups)

        # Both split files should be uncovered (different directory)
        uncovered_names = {f.name for f in uncovered}
        self.assertIn(
            "test_arg_parser_part1.mojo",
            uncovered_names,
            "Split file part1 in uncovered directory should be flagged",
        )
        self.assertIn(
            "test_arg_parser_part2.mojo",
            uncovered_names,
            "Split file part2 in uncovered directory should be flagged",
        )

    def test_single_canonical_file_covered(self):
        """Test baseline: single canonical test file is recognized as covered."""
        # Create a single canonical test file
        (self.tests_dir / "test_arg_parser.mojo").write_text(
            "fn test_arg_parser_basic() raises:\n    pass\n"
        )

        # Create workflow covering this directory
        workflow_file = self._create_workflow_file(
            "tests/shared/core", "test_*.mojo"
        )

        # Check coverage
        groups = parse_ci_matrix(workflow_file)
        uncovered = check_coverage(self.test_root, groups)

        # Single file should be covered
        uncovered_names = {f.name for f in uncovered}
        self.assertNotIn(
            "test_arg_parser.mojo",
            uncovered_names,
            "Canonical test file should be covered by wildcard pattern",
        )

    def test_suffix_variant_naming(self):
        """Test that suffix naming variants (e.g., _cmd_run, _parser) are covered."""
        # Create test files with suffix variants
        (self.tests_dir / "test_arg_parser_cmd_run.mojo").write_text(
            "fn test_arg_parser_cmd_run() raises:\n    pass\n"
        )
        (self.tests_dir / "test_arg_parser_parser.mojo").write_text(
            "fn test_arg_parser_parser() raises:\n    pass\n"
        )

        # Create workflow covering this directory
        workflow_file = self._create_workflow_file(
            "tests/shared/core", "test_*.mojo"
        )

        # Check coverage
        groups = parse_ci_matrix(workflow_file)
        uncovered = check_coverage(self.test_root, groups)

        # Both suffix variants should be covered
        uncovered_names = {f.name for f in uncovered}
        self.assertNotIn(
            "test_arg_parser_cmd_run.mojo",
            uncovered_names,
            "Suffix variant _cmd_run should be covered by wildcard pattern",
        )
        self.assertNotIn(
            "test_arg_parser_parser.mojo",
            uncovered_names,
            "Suffix variant _parser should be covered by wildcard pattern",
        )

    def test_mixed_split_and_canonical_files(self):
        """Test that mixed split and canonical files are all recognized correctly."""
        # Create a mix of canonical, split, and suffix variant files
        (self.tests_dir / "test_utils.mojo").write_text(
            "fn test_utils_basic() raises:\n    pass\n"
        )
        (self.tests_dir / "test_parser_part1.mojo").write_text(
            "fn test_parser_basic() raises:\n    pass\n"
        )
        (self.tests_dir / "test_parser_part2.mojo").write_text(
            "fn test_parser_edge_cases() raises:\n    pass\n"
        )
        (self.tests_dir / "test_formatter_cmd_run.mojo").write_text(
            "fn test_formatter_cmd_run() raises:\n    pass\n"
        )

        # Create workflow covering this directory
        workflow_file = self._create_workflow_file(
            "tests/shared/core", "test_*.mojo"
        )

        # Check coverage
        groups = parse_ci_matrix(workflow_file)
        uncovered = check_coverage(self.test_root, groups)

        # All files should be covered
        uncovered_names = {f.name for f in uncovered}
        self.assertEqual(
            len(uncovered_names),
            0,
            f"All files should be covered, but found uncovered: {uncovered_names}",
        )


class TestFindTestFiles(unittest.TestCase):
    """Test suite for find_test_files function."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.test_root = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_root)

    def test_find_mojo_files_in_directory(self):
        """Test finding .mojo test files in a directory."""
        tests_dir = self.test_root / "tests"
        tests_dir.mkdir()

        # Create test files
        (tests_dir / "test_foo.mojo").write_text("")
        (tests_dir / "test_bar.mojo").write_text("")
        (tests_dir / "not_a_test.mojo").write_text("")

        files = find_test_files(tests_dir)
        names = {f.name for f in files}

        self.assertIn("test_foo.mojo", names)
        self.assertIn("test_bar.mojo", names)
        self.assertNotIn("not_a_test.mojo", names)

    def test_find_files_in_nested_directories(self):
        """Test finding test files in nested directory structures."""
        tests_dir = self.test_root / "tests"
        core_dir = tests_dir / "shared" / "core"
        core_dir.mkdir(parents=True)

        (core_dir / "test_tensor.mojo").write_text("")
        (core_dir / "test_matrix.mojo").write_text("")

        files = find_test_files(tests_dir)
        names = {f.name for f in files}

        self.assertIn("test_tensor.mojo", names)
        self.assertIn("test_matrix.mojo", names)


if __name__ == "__main__":
    unittest.main()
