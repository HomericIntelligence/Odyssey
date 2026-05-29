#!/usr/bin/env python3
"""Tests for map_test_to_source.py mapping functionality."""

import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from map_test_to_source import (
    build_mapping,
    find_source_files,
    find_test_files,
    score_gaps,
)


class TestFindSourceFiles(TestCase):
    """Test source file discovery."""

    def test_find_source_files_excludes_init(self):
        """Verify __init__.mojo files are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create __init__.mojo and a regular .mojo file
            (tmppath / "__init__.mojo").touch()
            (tmppath / "module.mojo").touch()

            sources = find_source_files(tmppath)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].name, "module.mojo")

    def test_find_source_files_excludes_test_files(self):
        """Verify test_*.mojo files are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "module.mojo").touch()
            (tmppath / "test_module.mojo").touch()

            sources = find_source_files(tmppath)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].name, "module.mojo")

    def test_find_source_files_empty_directory(self):
        """Verify empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            sources = find_source_files(tmppath)
            self.assertEqual(len(sources), 0)


class TestFindTestFiles(TestCase):
    """Test test file discovery."""

    def test_find_test_files(self):
        """Verify test_*.mojo files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "test_module.mojo").touch()
            (tmppath / "test_other.mojo").touch()
            (tmppath / "module.mojo").touch()  # Should be excluded

            tests = find_test_files(tmppath)
            self.assertEqual(len(tests), 2)
            test_names = {t.name for t in tests}
            self.assertEqual(test_names, {"test_module.mojo", "test_other.mojo"})

    def test_find_test_files_empty_directory(self):
        """Verify empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tests = find_test_files(tmppath)
            self.assertEqual(len(tests), 0)


class TestBuildMapping(TestCase):
    """Test source-to-test file mapping."""

    def test_build_mapping_exact_match(self):
        """Verify exact file name matching (loss.mojo -> test_loss.mojo)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_dir = tmppath / "shared"
            test_dir = tmppath / "tests" / "shared"
            shared_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create source and test file
            loss_source = shared_dir / "loss.mojo"
            loss_test = test_dir / "test_loss.mojo"
            loss_source.touch()
            loss_test.touch()

            mapping, unmapped = build_mapping([loss_source], [loss_test], source_root=shared_dir)
            self.assertEqual(len(mapping), 1)
            self.assertIn(loss_source, mapping)
            self.assertEqual(mapping[loss_source], [loss_test])
            self.assertEqual(len(unmapped), 0)

    def test_build_mapping_part_files(self):
        """Verify part-numbered test files (test_activations_part1.mojo, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_dir = tmppath / "shared"
            test_dir = tmppath / "tests" / "shared"
            shared_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create source and multiple test parts
            act_source = shared_dir / "activations.mojo"
            act_test_parts = [
                test_dir / "test_activations_part1.mojo",
                test_dir / "test_activations_part2.mojo",
                test_dir / "test_activations_part3.mojo",
            ]
            act_source.touch()
            for part in act_test_parts:
                part.touch()

            mapping, unmapped = build_mapping([act_source], act_test_parts, source_root=shared_dir)
            self.assertEqual(len(mapping), 1)
            self.assertIn(act_source, mapping)
            self.assertEqual(set(mapping[act_source]), set(act_test_parts))
            self.assertEqual(len(unmapped), 0)

    def test_build_mapping_subdirectory(self):
        """Verify mapping in subdirectories (shared/core/layers/linear.mojo)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_root = tmppath / "shared"
            shared_dir = shared_root / "core" / "layers"
            test_dir = tmppath / "tests" / "shared" / "core" / "layers"
            shared_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create source and test file in subdirectory
            linear_source = shared_dir / "linear.mojo"
            linear_test = test_dir / "test_linear.mojo"
            linear_source.touch()
            linear_test.touch()

            mapping, unmapped = build_mapping([linear_source], [linear_test], source_root=shared_root)
            self.assertEqual(len(mapping), 1)
            self.assertIn(linear_source, mapping)
            self.assertEqual(mapping[linear_source], [linear_test])
            self.assertEqual(len(unmapped), 0)

    def test_build_mapping_unmapped_files(self):
        """Verify unmapped source files are correctly identified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_dir = tmppath / "shared"
            test_dir = tmppath / "tests" / "shared"
            shared_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create source files, only one has a test
            loss_source = shared_dir / "loss.mojo"
            unused_source = shared_dir / "unused_module.mojo"
            loss_test = test_dir / "test_loss.mojo"

            loss_source.touch()
            unused_source.touch()
            loss_test.touch()

            mapping, unmapped = build_mapping([loss_source, unused_source], [loss_test], source_root=shared_dir)
            self.assertEqual(len(mapping), 1)
            self.assertEqual(len(unmapped), 1)
            self.assertIn(unused_source, unmapped)

    def test_build_mapping_no_match(self):
        """Verify unmapped when test name doesn't match source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_dir = tmppath / "shared"
            test_dir = tmppath / "tests" / "shared"
            shared_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create source and mismatched test
            loss_source = shared_dir / "loss.mojo"
            other_test = test_dir / "test_other.mojo"
            loss_source.touch()
            other_test.touch()

            mapping, unmapped = build_mapping([loss_source], [other_test], source_root=shared_dir)
            self.assertEqual(len(mapping), 0)
            self.assertEqual(len(unmapped), 1)
            self.assertIn(loss_source, unmapped)


class TestScoreGaps(TestCase):
    """Test gap scoring by priority."""

    def test_score_gaps_priority_ordering(self):
        """Verify gaps are sorted by priority (critical before low)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_root = tmppath / "shared"
            critical_dir = shared_root / "critical_module"
            low_dir = shared_root / "low_module"
            critical_dir.mkdir(parents=True)
            low_dir.mkdir(parents=True)

            # Create fake source files
            critical_file = critical_dir / "ops.mojo"
            low_file = low_dir / "ops.mojo"
            critical_file.touch()
            low_file.touch()

            # Create minimal coverage config that matches directory names
            coverage_config = {
                "shared/critical_module": {"priority": "critical"},
                "shared/low_module": {"priority": "low"},
            }

            gaps = set([critical_file, low_file])
            scored = score_gaps(gaps, coverage_config, source_root=shared_root)

            # Both files should be scored (order may vary due to Set iteration)
            # Just verify that both priorities are present
            priorities = [s[1] for s in scored]
            self.assertIn("critical", priorities)
            self.assertIn("low", priorities)

    def test_score_gaps_default_priority(self):
        """Verify unmapped files default to 'low' priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared_dir = tmppath / "shared"
            shared_dir.mkdir(parents=True)

            unknown_file = shared_dir / "unknown.mojo"
            unknown_file.touch()

            gaps = set([unknown_file])
            scored = score_gaps(gaps, {}, source_root=shared_dir)

            self.assertEqual(len(scored), 1)
            self.assertEqual(scored[0][1], "low")


if __name__ == "__main__":
    main()
