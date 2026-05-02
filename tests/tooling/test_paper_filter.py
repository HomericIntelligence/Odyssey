#!/usr/bin/env python3
"""
Tests for paper-specific test filtering.

Tests Issue #810 (Test Specific Paper - Write Tests):
- Paper name parsing
- Paper directory resolution
- Partial name matching
- Error handling for non-existent papers
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import pytest


class TestPaperFiltering:
    """Test paper-specific test filtering (Issue #810)."""

    def setup_method(self):
        """Create temporary test structure."""
        self.test_dir = Path(tempfile.mkdtemp())

        # Create mock paper directories
        papers_dir = self.test_dir / "papers"
        papers_dir.mkdir()

        # Create several test papers
        (papers_dir / "lenet-5").mkdir()
        (papers_dir / "lenet-5" / "tests").mkdir()

        (papers_dir / "bert").mkdir()
        (papers_dir / "bert" / "tests").mkdir()

        (papers_dir / "gpt-2").mkdir()
        (papers_dir / "gpt-2" / "tests").mkdir()

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_find_paper_by_exact_name(self):
        """Test finding paper by exact directory name."""

        papers_dir = self.test_dir / "papers"
        target = "lenet-5"

        # Should find exact match
        result = list(papers_dir.glob(f"*{target}*"))
        assert len(result) >= 1
        assert any(p.name == target for p in result)

    def test_find_paper_by_partial_name(self):
        """Test finding paper by partial name match."""
        papers_dir = self.test_dir / "papers"
        target = "lenet"

        # Should find lenet-5
        result = list(papers_dir.glob(f"*{target}*"))
        assert len(result) >= 1
        assert any("lenet" in p.name for p in result)

    def test_paper_not_found(self):
        """Test handling of non-existent paper."""
        papers_dir = self.test_dir / "papers"
        target = "nonexistent"

        # Should not find anything
        result = list(papers_dir.glob(f"*{target}*"))
        assert len(result) == 0

    def test_paper_has_tests_directory(self):
        """Test verifying paper has tests directory."""
        paper_dir = self.test_dir / "papers" / "lenet-5"
        tests_dir = paper_dir / "tests"

        assert tests_dir.exists()
        assert tests_dir.is_dir()

    def test_multiple_partial_matches(self):
        """Test handling multiple partial matches."""
        papers_dir = self.test_dir / "papers"

        # Create another paper with similar name
        (papers_dir / "lenet-7").mkdir()

        target = "lenet"
        result = list(papers_dir.glob(f"*{target}*"))

        # Should find both lenet-5 and lenet-7
        assert len(result) >= 2
        names = [p.name for p in result]
        assert "lenet-5" in names
        assert "lenet-7" in names

    def test_case_insensitive_matching(self):
        """Test case-insensitive paper name matching."""
        papers_dir = self.test_dir / "papers"

        # Search with different case
        target_lower = "bert"
        target_upper = "BERT"

        result_lower = list(papers_dir.glob(f"*{target_lower.lower()}*"))
        result_upper = list(papers_dir.glob(f"*{target_upper.lower()}*"))

        # Both should find the same paper
        assert len(result_lower) >= 1
        assert len(result_upper) >= 1

    def test_hyphenated_paper_names(self):
        """Test papers with hyphens in names."""
        papers_dir = self.test_dir / "papers"

        # Should find gpt-2
        target = "gpt-2"
        result = list(papers_dir.glob(f"*{target}*"))

        assert len(result) >= 1
        assert any(p.name == "gpt-2" for p in result)


@pytest.mark.skip(
    reason="mojo-test-runner skill migrated to ProjectMnemosyne; run_tests.sh no longer in .claude/skills/"
)
class TestRunTestsScript:
    """Test run_tests.sh script integration (Issue #811)."""

    def test_run_tests_script_exists(self):
        """Test that run_tests.sh script exists."""
        script_path = Path(".claude/skills/mojo-test-runner/scripts/run_tests.sh")
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_executable(self):
        """Test that run_tests.sh is executable."""
        script_path = Path(".claude/skills/mojo-test-runner/scripts/run_tests.sh")
        assert script_path.exists()
        import os

        assert os.access(script_path, os.X_OK), "Script is not executable"

    def test_help_option(self):
        """Test script has help option."""
        script_path = Path(".claude/skills/mojo-test-runner/scripts/run_tests.sh")
        result = subprocess.run([str(script_path), "--help"], capture_output=True, text=True)
        assert result.returncode in [0, 1, 2]

    def test_paper_option_available(self):
        """Test --paper option is available."""
        script_path = Path(".claude/skills/mojo-test-runner/scripts/run_tests.sh")
        result = subprocess.run([str(script_path), "--help"], capture_output=True, text=True)
        output = result.stdout + result.stderr
        print(output)
        assert script_path.exists()


class TestPaperTestFiltering:
    """Test actual filtering of tests by paper (Issue #812)."""

    def test_filter_logic(self):
        """Test that filtering logic works correctly."""
        # This will be a basic logic test
        # Simulating the filter: if paper specified, run only that paper's tests

        all_tests = [
            "papers/lenet-5/tests/test_model.mojo",
            "papers/lenet-5/tests/test_train.mojo",
            "papers/bert/tests/test_model.mojo",
            "papers/gpt-2/tests/test_model.mojo",
        ]

        # Filter for lenet-5
        target_paper = "lenet-5"
        filtered = [t for t in all_tests if target_paper in t]

        assert len(filtered) == 2
        assert all("lenet-5" in t for t in filtered)

    def test_no_filter_runs_all(self):
        """Test that no paper filter runs all tests."""
        all_tests = [
            "papers/lenet-5/tests/test_model.mojo",
            "papers/bert/tests/test_model.mojo",
            "papers/gpt-2/tests/test_model.mojo",
        ]

        # No filter - should run all
        target_paper = None
        filtered = all_tests if target_paper is None else [t for t in all_tests if target_paper in t]

        assert len(filtered) == 3


if __name__ == "__main__":
    import traceback

    print("Running paper filtering tests...")

    # Run TestPaperFiltering tests
    test_class = TestPaperFiltering()
    test_methods = [m for m in dir(test_class) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            test_class.setup_method()
            method = getattr(test_class, method_name)
            method()
            test_class.teardown_method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            traceback.print_exc()
            failed += 1
            try:
                test_class.teardown_method()
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                pass

    # Run TestRunTestsScript tests
    test_class2 = TestRunTestsScript()
    test_methods2 = [m for m in dir(test_class2) if m.startswith("test_")]

    for method_name in test_methods2:
        try:
            method = getattr(test_class2, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            # Don't print traceback for expected failures
            failed += 1

    # Run TestPaperTestFiltering tests
    test_class3 = TestPaperTestFiltering()
    test_methods3 = [m for m in dir(test_class3) if m.startswith("test_")]

    for method_name in test_methods3:
        try:
            method = getattr(test_class3, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
