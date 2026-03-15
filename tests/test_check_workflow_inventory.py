#!/usr/bin/env python3
"""
Unit tests for scripts/check_workflow_inventory.py

Tests cover:
- collect_yml_files: basic discovery, worktree exclusion
- parse_readme_table: plain and hyperlinked filenames
- check_inventory: in-sync, undocumented files, missing files, both at once
- main: exit codes
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_workflow_inventory import (
    check_inventory,
    collect_yml_files,
    main,
    parse_readme_table,
)


def _make_workflows_dir(base: Path) -> Path:
    """Create .github/workflows/ under base and return its path."""
    workflows = base / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    return workflows


def _write_readme(workflows: Path, table_rows: list[str]) -> Path:
    """Write a minimal README.md with the given table rows."""
    header = (
        "# Workflows\n\n"
        "| Workflow | Trigger |\n"
        "|----------|---------|\n"
    )
    readme = workflows / "README.md"
    readme.write_text(header + "\n".join(table_rows) + "\n")
    return readme


class TestCollectYmlFiles(unittest.TestCase):
    """Tests for collect_yml_files()."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_empty_workflows_dir(self) -> None:
        _make_workflows_dir(self.tmp)
        self.assertEqual(collect_yml_files(self.tmp), set())

    def test_single_yml_returned(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        self.assertEqual(collect_yml_files(self.tmp), {"ci.yml"})

    def test_multiple_yml_files(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        for name in ("ci.yml", "release.yml", "security.yml"):
            (workflows / name).write_text("name: x")
        self.assertEqual(collect_yml_files(self.tmp), {"ci.yml", "release.yml", "security.yml"})

    def test_non_yml_files_excluded(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        (workflows / "README.md").write_text("# docs")
        (workflows / "schema.json").write_text("{}")
        self.assertEqual(collect_yml_files(self.tmp), {"ci.yml"})

    def test_worktrees_subdir_excluded(self) -> None:
        """Files under a worktrees/ directory must not appear in results."""
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        # Simulate a worktree path (worktrees/ as an ancestor of workflows/)
        worktree_workflows = self.tmp / "worktrees" / "issue-1" / ".github" / "workflows"
        worktree_workflows.mkdir(parents=True)
        (worktree_workflows / "ci.yml").write_text("name: CI")
        # Only the root-level ci.yml should be found
        self.assertEqual(collect_yml_files(self.tmp), {"ci.yml"})

    def test_missing_workflows_dir_returns_empty(self) -> None:
        self.assertEqual(collect_yml_files(self.tmp), set())


class TestParseReadmeTable(unittest.TestCase):
    """Tests for parse_readme_table()."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_plain_filename(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        _write_readme(workflows, ["| ci.yml | push |"])
        self.assertEqual(parse_readme_table(workflows / "README.md"), {"ci.yml"})

    def test_hyperlinked_filename(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        _write_readme(workflows, ["| [ci.yml](#ci) | push |"])
        self.assertEqual(parse_readme_table(workflows / "README.md"), {"ci.yml"})

    def test_multiple_filenames(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        rows = [
            "| [ci.yml](#ci) | push |",
            "| release.yml | tag |",
            "| [security.yml](#security) | weekly |",
        ]
        _write_readme(workflows, rows)
        self.assertEqual(
            parse_readme_table(workflows / "README.md"),
            {"ci.yml", "release.yml", "security.yml"},
        )

    def test_section_header_rows_ignored(self) -> None:
        """Category header rows like '| **Test Workflows** | | |' must not match."""
        workflows = _make_workflows_dir(self.tmp)
        _write_readme(
            workflows,
            [
                "| **Test Workflows** | | |",
                "| ci.yml | push |",
            ],
        )
        self.assertEqual(parse_readme_table(workflows / "README.md"), {"ci.yml"})

    def test_missing_readme_returns_empty(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        self.assertEqual(parse_readme_table(workflows / "README.md"), set())

    def test_readme_without_table_returns_empty(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "README.md").write_text("# Just a heading\n\nNo table here.\n")
        self.assertEqual(parse_readme_table(workflows / "README.md"), set())

    def test_bold_category_row_not_matched(self) -> None:
        """Rows like '| **Build & Release** | | | |' should not be extracted."""
        workflows = _make_workflows_dir(self.tmp)
        readme = workflows / "README.md"
        readme.write_text(
            "| **Build & Release Workflows** | | | |\n"
            "| release.yml | tag | Build | < 5 min |\n"
        )
        self.assertEqual(parse_readme_table(readme), {"release.yml"})


class TestCheckInventory(unittest.TestCase):
    """Tests for check_inventory()."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _setup(self, disk_files: list[str], readme_rows: list[str]) -> None:
        workflows = _make_workflows_dir(self.tmp)
        for name in disk_files:
            (workflows / name).write_text("name: x")
        _write_readme(workflows, readme_rows)

    def test_in_sync_returns_empty_lists(self) -> None:
        self._setup(["ci.yml"], ["| ci.yml | push |"])
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, [])
        self.assertEqual(missing, [])

    def test_undocumented_file_detected(self) -> None:
        """File on disk but not in README should appear in undocumented."""
        self._setup(["ci.yml", "extra.yml"], ["| ci.yml | push |"])
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, ["extra.yml"])
        self.assertEqual(missing, [])

    def test_missing_file_detected(self) -> None:
        """README entry with no corresponding file should appear in missing_files."""
        self._setup(["ci.yml"], ["| ci.yml | push |", "| ghost.yml | manual |"])
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, [])
        self.assertEqual(missing, ["ghost.yml"])

    def test_both_mismatches_at_once(self) -> None:
        self._setup(
            ["ci.yml", "new.yml"],
            ["| ci.yml | push |", "| ghost.yml | manual |"],
        )
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, ["new.yml"])
        self.assertEqual(missing, ["ghost.yml"])

    def test_multiple_files_all_documented(self) -> None:
        names = ["ci.yml", "release.yml", "security.yml"]
        rows = [f"| {n} | push |" for n in names]
        self._setup(names, rows)
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, [])
        self.assertEqual(missing, [])

    def test_hyperlinked_entries_recognised(self) -> None:
        self._setup(["ci.yml"], ["| [ci.yml](#ci) | push |"])
        undocumented, missing = check_inventory(self.tmp)
        self.assertEqual(undocumented, [])
        self.assertEqual(missing, [])

    def test_undocumented_list_is_sorted(self) -> None:
        self._setup(["z.yml", "a.yml", "m.yml"], [])
        undocumented, _ = check_inventory(self.tmp)
        self.assertEqual(undocumented, ["a.yml", "m.yml", "z.yml"])

    def test_missing_list_is_sorted(self) -> None:
        self._setup([], ["| z.yml | push |", "| a.yml | push |"])
        _, missing = check_inventory(self.tmp)
        self.assertEqual(missing, ["a.yml", "z.yml"])

    def test_readme_not_counted_as_yml(self) -> None:
        """README.md is not a .yml file and must never appear in undocumented."""
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        _write_readme(workflows, ["| ci.yml | push |"])
        undocumented, _ = check_inventory(self.tmp)
        self.assertNotIn("README.md", undocumented)


class TestMain(unittest.TestCase):
    """Integration tests for main() exit codes."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_main_returns_0_when_in_sync(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        _write_readme(workflows, ["| ci.yml | push |"])
        # Patch sys.argv so argparse doesn't pick up pytest arguments
        old_argv = sys.argv
        sys.argv = ["check_workflow_inventory.py", "--repo-root", str(self.tmp)]
        try:
            self.assertEqual(main(), 0)
        finally:
            sys.argv = old_argv

    def test_main_returns_1_on_drift(self) -> None:
        workflows = _make_workflows_dir(self.tmp)
        (workflows / "ci.yml").write_text("name: CI")
        (workflows / "undoc.yml").write_text("name: x")
        _write_readme(workflows, ["| ci.yml | push |"])
        old_argv = sys.argv
        sys.argv = ["check_workflow_inventory.py", "--repo-root", str(self.tmp)]
        try:
            self.assertEqual(main(), 1)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
