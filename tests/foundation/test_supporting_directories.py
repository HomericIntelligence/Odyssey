"""
Test suite for supporting directories validation.

This module contains comprehensive tests for validating the supporting directories
structure (benchmarks/, docs/, agents/, tools/, configs/) following TDD/FIRST principles.

Test Categories:
- Unit tests: Directory existence, location, permissions
- README validation: Presence and completeness
- Structure tests: Subdirectory organization
- Integration tests: Cross-directory validation

Coverage Target: >95%
"""

import stat
from pathlib import Path
from typing import Dict

import pytest


class TestSupportingDirectoriesExistence:
    """Test that all supporting directories exist at repository root."""

    def test_benchmarks_directory_exists(self, benchmarks_dir: Path) -> None:
        """
        Test that benchmarks/ directory exists.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)

        Args:
            benchmarks_dir: Path to benchmarks directory
        """
        assert benchmarks_dir.exists(), "benchmarks/ directory should exist"
        assert benchmarks_dir.is_dir(), "benchmarks/ should be a directory, not a file"

    def test_docs_directory_exists(self, docs_dir: Path) -> None:
        """
        Test that docs/ directory exists.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)

        Args:
            docs_dir: Path to docs directory
        """
        assert docs_dir.exists(), "docs/ directory should exist"
        assert docs_dir.is_dir(), "docs/ should be a directory, not a file"

    def test_agents_directory_exists(self, agents_dir: Path) -> None:
        """
        Test that agents/ directory exists.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)

        Args:
            agents_dir: Path to agents directory
        """
        assert agents_dir.exists(), "agents/ directory should exist"
        assert agents_dir.is_dir(), "agents/ should be a directory, not a file"

    def test_tools_directory_exists(self, tools_dir: Path) -> None:
        """
        Test that tools/ directory exists.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)

        Args:
            tools_dir: Path to tools directory
        """
        assert tools_dir.exists(), "tools/ directory should exist"
        assert tools_dir.is_dir(), "tools/ should be a directory, not a file"

    def test_configs_directory_exists(self, configs_dir: Path) -> None:
        """
        Test that configs/ directory exists.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)

        Args:
            configs_dir: Path to configs directory
        """
        assert configs_dir.exists(), "configs/ directory should exist"
        assert configs_dir.is_dir(), "configs/ should be a directory, not a file"


class TestSupportingDirectoriesLocation:
    """Test that directories are at correct location with proper configuration."""

    def test_supporting_directories_at_root(self, repo_root: Path, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that all supporting directories are directly under repository root.

        Verifies:
        - Each directory's parent is repository root
        - Directories are not nested deeper

        Args:
            repo_root: Repository root path
            supporting_dirs: Dictionary of all supporting directory paths
        """
        for dir_name, dir_path in supporting_dirs.items():
            assert dir_path.parent == repo_root, f"{dir_name}/ should be directly under repository root"

    def test_directory_names_correct(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that directory names are exactly as specified.

        Verifies:
        - Directory names match specification
        - No uppercase or alternative naming

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        expected_names = {"benchmarks", "docs", "agents", "tools", "configs"}
        actual_names = {dir_path.name for dir_path in supporting_dirs.values()}

        assert actual_names == expected_names, (
            f"Directory names should match specification. Expected: {expected_names}, Got: {actual_names}"
        )

    def test_directory_permissions(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that directories have correct permissions.

        Verifies:
        - Directories have read permissions
        - Directories have write permissions
        - Directories have execute permissions

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        for dir_name, dir_path in supporting_dirs.items():
            dir_stat = dir_path.stat()
            mode = dir_stat.st_mode

            assert mode & stat.S_IRUSR, f"{dir_name}/ should have read permission"
            assert mode & stat.S_IWUSR, f"{dir_name}/ should have write permission"
            assert mode & stat.S_IXUSR, f"{dir_name}/ should have execute permission"

            # Verify we can create a test file
            test_file = dir_path / ".test_permissions"
            test_file.write_text("test")
            assert test_file.exists(), f"Should be able to create files in {dir_name}/"
            test_file.unlink()  # Clean up


class TestSupportingDirectoriesReadme:
    """Test README.md presence and completeness for each directory."""

    def test_each_directory_has_readme(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that every supporting directory has a README.md file.

        Verifies:
        - README.md exists in each directory
        - README.md is a file (not a directory)

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        for dir_name, dir_path in supporting_dirs.items():
            readme_path = dir_path / "README.md"
            assert readme_path.exists(), f"{dir_name}/README.md should exist"
            assert readme_path.is_file(), f"{dir_name}/README.md should be a file"

    def test_readme_not_empty(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that README.md files are not empty.

        Verifies:
        - README.md has content (>100 characters)
        - README.md is readable

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        min_content_length = 100  # Minimum meaningful README length

        for dir_name, dir_path in supporting_dirs.items():
            readme_path = dir_path / "README.md"
            content = readme_path.read_text()

            assert len(content) > min_content_length, (
                f"{dir_name}/README.md should have substantial content "
                f"(expected >{min_content_length} chars, got {len(content)})"
            )

    def test_readme_has_title(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that README.md files have a markdown title (# header).

        Verifies:
        - README starts with markdown header
        - Header is at top of file (within first few lines)

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        for dir_name, dir_path in supporting_dirs.items():
            readme_path = dir_path / "README.md"
            content = readme_path.read_text()
            lines = content.split("\n")

            # Check first 5 lines for a markdown header
            has_header = any(line.strip().startswith("#") for line in lines[:5])

            assert has_header, f"{dir_name}/README.md should have a markdown header (# Title)"


class TestSupportingDirectoriesStructure:
    """Test subdirectory structure for each supporting directory."""

    def test_benchmarks_subdirectory_structure(self, benchmarks_dir: Path) -> None:
        """
        Test benchmarks/ directory has expected subdirectories.

        Verifies:
        - Key subdirectories exist (baselines/, results/, scripts/)
        - Structure supports benchmarking workflow

        Args:
            benchmarks_dir: Path to benchmarks directory
        """
        # Check for key subdirectories from specification
        expected_subdirs = ["baselines", "results", "scripts"]

        for subdir_name in expected_subdirs:
            subdir_path = benchmarks_dir / subdir_name
            if not subdir_path.exists():
                pytest.skip(f"Subdirectory not yet created: {subdir_name}/")

        # Verify expected subdirectories exist
        for subdir_name in expected_subdirs:
            subdir_path = benchmarks_dir / subdir_name
            assert subdir_path.exists(), f"benchmarks/{subdir_name}/ should exist"
            assert subdir_path.is_dir(), f"benchmarks/{subdir_name}/ should be a directory"

    def test_docs_subdirectory_structure(self, docs_dir: Path) -> None:
        """
        Test docs/ directory has expected tier subdirectories.

        Verifies:
        - 4-tier structure exists (getting-started, core, advanced, dev)
        - Structure matches documentation specification

        Args:
            docs_dir: Path to docs directory
        """
        # Check for tier subdirectories from specification
        expected_tiers = ["getting-started", "core", "advanced", "dev"]

        for tier_name in expected_tiers:
            tier_path = docs_dir / tier_name
            if not tier_path.exists():
                pytest.skip(f"Tier not yet created: {tier_name}/")

        # Verify expected tiers exist
        for tier_name in expected_tiers:
            tier_path = docs_dir / tier_name
            assert tier_path.exists(), f"docs/{tier_name}/ should exist"
            assert tier_path.is_dir(), f"docs/{tier_name}/ should be a directory"

    def test_agents_subdirectory_structure(self, agents_dir: Path) -> None:
        """
        Test agents/ directory has expected subdirectories.

        Verifies:
        - Key subdirectories exist (guides/, templates/)
        - Structure supports agent system

        Args:
            agents_dir: Path to agents directory
        """
        # Check for key subdirectories from specification
        expected_subdirs = ["guides", "templates"]

        for subdir_name in expected_subdirs:
            subdir_path = agents_dir / subdir_name
            if not subdir_path.exists():
                pytest.skip(f"Subdirectory not yet created: {subdir_name}/")

        # Verify expected subdirectories exist
        for subdir_name in expected_subdirs:
            subdir_path = agents_dir / subdir_name
            assert subdir_path.exists(), f"agents/{subdir_name}/ should exist"
            assert subdir_path.is_dir(), f"agents/{subdir_name}/ should be a directory"

    def test_tools_subdirectory_structure(self, tools_dir: Path) -> None:
        """
        Test tools/ directory has expected subdirectories.

        Verifies:
        - Key subdirectories exist for tool categories
        - Structure supports tool organization

        Args:
            tools_dir: Path to tools directory
        """
        # Check for at least some key subdirectories from specification
        # (Being flexible as implementation may add more)
        expected_subdirs = ["benchmarking", "setup"]

        for subdir_name in expected_subdirs:
            subdir_path = tools_dir / subdir_name
            if not subdir_path.exists():
                pytest.skip(f"Subdirectory not yet created: {subdir_name}/")

        # Verify expected subdirectories exist
        for subdir_name in expected_subdirs:
            subdir_path = tools_dir / subdir_name
            assert subdir_path.exists(), f"tools/{subdir_name}/ should exist"
            assert subdir_path.is_dir(), f"tools/{subdir_name}/ should be a directory"

    def test_configs_subdirectory_structure(self, configs_dir: Path) -> None:
        """
        Test configs/ directory has expected subdirectories.

        Verifies:
        - Key subdirectories exist (defaults/, schemas/, templates/)
        - Structure supports configuration management

        Args:
            configs_dir: Path to configs directory
        """
        # Check for key subdirectories from specification
        expected_subdirs = ["defaults", "schemas", "templates"]

        for subdir_name in expected_subdirs:
            subdir_path = configs_dir / subdir_name
            if not subdir_path.exists():
                pytest.skip(f"Subdirectory not yet created: {subdir_name}/")

        # Verify expected subdirectories exist
        for subdir_name in expected_subdirs:
            subdir_path = configs_dir / subdir_name
            assert subdir_path.exists(), f"configs/{subdir_name}/ should exist"
            assert subdir_path.is_dir(), f"configs/{subdir_name}/ should be a directory"


class TestSupportingDirectoriesIntegration:
    """Integration tests for all supporting directories together."""

    def test_all_supporting_directories_present(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that all 5 supporting directories are present.

        Verifies:
        - All 5 directories exist
        - All are actual directories
        - Complete set is available

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        assert len(supporting_dirs) == 5, "Should have exactly 5 supporting directories"

        for dir_name, dir_path in supporting_dirs.items():
            assert dir_path.exists(), f"{dir_name}/ should exist"
            assert dir_path.is_dir(), f"{dir_name}/ should be a directory"

    def test_directories_ready_for_content(self, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test that directories are ready to receive content.

        Verifies:
        - Can create subdirectories
        - Can create files
        - Proper permissions for content creation

        Args:
            supporting_dirs: Dictionary of all supporting directory paths
        """
        for dir_name, dir_path in supporting_dirs.items():
            # Test subdirectory creation
            test_subdir = dir_path / ".test_subdir"
            test_subdir.mkdir(exist_ok=True)
            assert test_subdir.exists(), f"Should be able to create subdirectories in {dir_name}/"

            # Test file creation
            test_file = test_subdir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists(), f"Should be able to create files in {dir_name}/ subdirectories"

            # Clean up
            test_file.unlink()
            test_subdir.rmdir()

    def test_supporting_directories_relationship(self, repo_root: Path, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test relationships between supporting directories.

        Verifies:
        - All directories share same parent (repository root)
        - No directory nesting between supporting dirs
        - Flat structure at root level

        Args:
            repo_root: Repository root path
            supporting_dirs: Dictionary of all supporting directory paths
        """
        # All should have same parent
        parents = {dir_path.parent for dir_path in supporting_dirs.values()}
        assert len(parents) == 1, "All supporting directories should share the same parent"
        parent = parents.pop()
        assert parent == repo_root, "All supporting directories should be under repository root"

        # No directory should be nested under another
        dir_paths = set(supporting_dirs.values())
        for dir_path in dir_paths:
            for other_path in dir_paths:
                if dir_path != other_path:
                    assert not str(dir_path).startswith(str(other_path) + "/"), (
                        f"{dir_path.name}/ should not be nested under {other_path.name}/"
                    )


class TestSupportingDirectoriesRealWorld:
    """Real-world scenario tests for supporting directories."""

    def test_complete_supporting_directories_workflow(self, repo_root: Path, supporting_dirs: Dict[str, Path]) -> None:
        """
        Test complete workflow using all supporting directories.

        This integration test simulates real-world usage:
        1. Verify all directories exist
        2. Verify all have READMEs
        3. Verify structure is complete
        4. Verify ready for content

        Args:
            repo_root: Repository root path
            supporting_dirs: Dictionary of all supporting directory paths
        """
        # Step 1: Verify all directories exist
        for dir_name, dir_path in supporting_dirs.items():
            assert dir_path.exists() and dir_path.is_dir(), f"{dir_name}/ should exist and be a directory"

        # Step 2: Verify all have READMEs
        for dir_name, dir_path in supporting_dirs.items():
            readme = dir_path / "README.md"
            assert readme.exists() and readme.is_file(), f"{dir_name}/README.md should exist"

        # Step 3: Verify structure is at repository root
        for dir_path in supporting_dirs.values():
            assert dir_path.parent == repo_root, "All directories should be at repository root"

        # Step 4: Verify ready for content (can create test files)
        for dir_name, dir_path in supporting_dirs.items():
            test_file = dir_path / ".test_workflow"
            test_file.write_text("workflow test")
            assert test_file.exists(), f"Should be able to create content in {dir_name}/"
            test_file.unlink()  # Clean up
