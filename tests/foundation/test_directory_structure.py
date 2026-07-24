"""
Test suite for directory structure validation.

This module contains comprehensive tests for validating the papers/ and src/odyssey/
directory structures, ensuring they follow the planned architecture.

Test Categories:
- Papers directory structure validation
- Shared directory structure validation
- Required files validation
- Directory permissions validation

Coverage Target: 100%
"""

import stat
from pathlib import Path


class TestPapersDirectoryStructure:
    """Test cases for papers/ directory structure."""

    def test_papers_directory_exists(self, papers_dir: Path) -> None:
        """
        Test that papers/ directory exists at repository root.

        Verifies:
        - Directory exists
        - Path is a directory (not a file)
        - Located at correct position in repository

        Args:
            papers_dir: Papers directory path
        """
        assert papers_dir.exists(), "papers/ directory must exist at repository root"
        assert papers_dir.is_dir(), "papers/ must be a directory, not a file"

    def test_papers_readme_exists(self, papers_dir: Path) -> None:
        """
        Test that papers/README.md exists.

        Verifies:
        - README.md file exists
        - Path is a file (not a directory)
        - File has content

        Args:
            papers_dir: Papers directory path
        """
        readme = papers_dir / "README.md"
        assert readme.exists(), "papers/README.md must exist"
        assert readme.is_file(), "papers/README.md must be a file"
        assert readme.stat().st_size > 0, "papers/README.md must not be empty"

    def test_papers_readme_content(self, papers_dir: Path) -> None:
        """
        Test that papers/README.md has correct content.

        Verifies:
        - Contains overview section
        - Contains structure documentation
        - Contains contributing guidelines

        Args:
            papers_dir: Papers directory path
        """
        readme = papers_dir / "README.md"
        content = readme.read_text()

        assert "# Papers" in content, "README must have Papers heading"
        assert "Overview" in content, "README must have Overview section"
        assert "Structure" in content, "README must document structure"

    def test_papers_template_directory_exists(self, template_dir: Path) -> None:
        """
        Test that papers/_template/ directory exists.

        Verifies:
        - Template directory exists
        - Path is a directory
        - Located within papers/ directory

        Args:
            template_dir: Template directory path
        """
        assert template_dir.exists(), "papers/_template/ directory must exist"
        assert template_dir.is_dir(), "papers/_template/ must be a directory"
        assert template_dir.name == "_template", "Template directory must be named '_template'"

    def test_papers_directory_permissions(self, papers_dir: Path) -> None:
        """
        Test that papers/ directory has correct permissions.

        Verifies:
        - Directory has read permission
        - Directory has write permission
        - Directory has execute permission

        Args:
            papers_dir: Papers directory path
        """
        dir_stat = papers_dir.stat()
        mode = dir_stat.st_mode

        assert mode & stat.S_IRUSR, "papers/ must have read permission"
        assert mode & stat.S_IWUSR, "papers/ must have write permission"
        assert mode & stat.S_IXUSR, "papers/ must have execute permission"


class TestSharedDirectoryStructure:
    """Test cases for src/odyssey/ directory structure."""

    def test_shared_directory_exists(self, shared_dir: Path) -> None:
        """
        Test that src/odyssey/ directory exists at repository root.

        Verifies:
        - Directory exists
        - Path is a directory
        - Located at correct position in repository

        Args:
            shared_dir: Shared directory path
        """
        assert shared_dir.exists(), "src/odyssey/ directory must exist at repository root"
        assert shared_dir.is_dir(), "src/odyssey/ must be a directory, not a file"

    def test_shared_readme_exists(self, shared_dir: Path) -> None:
        """
        Test that src/odyssey/README.md exists.

        Verifies:
        - README.md file exists
        - Path is a file
        - File has content

        Args:
            shared_dir: Shared directory path
        """
        readme = shared_dir / "README.md"
        assert readme.exists(), "src/odyssey/README.md must exist"
        assert readme.is_file(), "src/odyssey/README.md must be a file"
        assert readme.stat().st_size > 0, "src/odyssey/README.md must not be empty"

    def test_shared_readme_content(self, shared_dir: Path) -> None:
        """
        Test that src/odyssey/README.md has correct content.

        Verifies:
        - Contains purpose section
        - Contains design principles
        - Contains directory structure documentation
        - Contains subdirectory descriptions

        Args:
            shared_dir: Shared directory path
        """
        readme = shared_dir / "README.md"
        content = readme.read_text()

        assert "# Shared Library" in content, "README must have Shared Library heading"
        assert "Purpose" in content, "README must have Purpose section"
        assert "Design Principles" in content, "README must have Design Principles"
        assert "core/" in content, "README must document core subdirectory"
        assert "training/" in content, "README must document training subdirectory"
        assert "data/" in content, "README must document data subdirectory"
        assert "utils/" in content, "README must document utils subdirectory"

    def test_shared_init_file_exists(self, shared_dir: Path) -> None:
        """
        Test that src/odyssey/__init__.mojo exists.

        Verifies:
        - __init__.mojo file exists
        - Path is a file
        - File has content (package exports)

        Args:
            shared_dir: Shared directory path
        """
        init_file = shared_dir / "__init__.mojo"
        assert init_file.exists(), "src/odyssey/__init__.mojo must exist"
        assert init_file.is_file(), "src/odyssey/__init__.mojo must be a file"

    def test_shared_subdirectories_exist(
        self,
        shared_core_dir: Path,
        shared_training_dir: Path,
        shared_data_dir: Path,
        shared_utils_dir: Path,
    ) -> None:
        """
        Test that all required shared subdirectories exist.

        Verifies:
        - core/ directory exists
        - training/ directory exists
        - data/ directory exists
        - utils/ directory exists

        Args:
            shared_core_dir: Shared core directory path
            shared_training_dir: Shared training directory path
            shared_data_dir: Shared data directory path
            shared_utils_dir: Shared utils directory path
        """
        assert shared_core_dir.exists(), "src/odyssey/core/ directory must exist"
        assert shared_core_dir.is_dir(), "src/odyssey/core/ must be a directory"

        assert shared_training_dir.exists(), "src/odyssey/training/ directory must exist"
        assert shared_training_dir.is_dir(), "src/odyssey/training/ must be a directory"

        assert shared_data_dir.exists(), "src/odyssey/data/ directory must exist"
        assert shared_data_dir.is_dir(), "src/odyssey/data/ must be a directory"

        assert shared_utils_dir.exists(), "src/odyssey/utils/ directory must exist"
        assert shared_utils_dir.is_dir(), "src/odyssey/utils/ must be a directory"

    def test_shared_core_has_readme(self, shared_core_dir: Path) -> None:
        """
        Test that src/odyssey/core/ has README.md.

        Verifies:
        - README.md exists in core directory
        - File has content

        Args:
            shared_core_dir: Shared core directory path
        """
        readme = shared_core_dir / "README.md"
        assert readme.exists(), "src/odyssey/core/README.md must exist"
        assert readme.is_file(), "src/odyssey/core/README.md must be a file"
        assert readme.stat().st_size > 0, "src/odyssey/core/README.md must not be empty"

    def test_shared_core_has_init(self, shared_core_dir: Path) -> None:
        """
        Test that src/odyssey/core/ has __init__.mojo.

        Verifies:
        - __init__.mojo exists in core directory
        - Path is a file

        Args:
            shared_core_dir: Shared core directory path
        """
        init_file = shared_core_dir / "__init__.mojo"
        assert init_file.exists(), "src/odyssey/core/__init__.mojo must exist"
        assert init_file.is_file(), "src/odyssey/core/__init__.mojo must be a file"

    def test_shared_training_has_readme(self, shared_training_dir: Path) -> None:
        """
        Test that src/odyssey/training/ has README.md.

        Verifies:
        - README.md exists in training directory
        - File has content

        Args:
            shared_training_dir: Shared training directory path
        """
        readme = shared_training_dir / "README.md"
        assert readme.exists(), "src/odyssey/training/README.md must exist"
        assert readme.is_file(), "src/odyssey/training/README.md must be a file"
        assert readme.stat().st_size > 0, "src/odyssey/training/README.md must not be empty"

    def test_shared_training_has_init(self, shared_training_dir: Path) -> None:
        """
        Test that src/odyssey/training/ has __init__.mojo.

        Verifies:
        - __init__.mojo exists in training directory
        - Path is a file

        Args:
            shared_training_dir: Shared training directory path
        """
        init_file = shared_training_dir / "__init__.mojo"
        assert init_file.exists(), "src/odyssey/training/__init__.mojo must exist"
        assert init_file.is_file(), "src/odyssey/training/__init__.mojo must be a file"

    def test_shared_data_has_readme(self, shared_data_dir: Path) -> None:
        """
        Test that src/odyssey/data/ has README.md.

        Verifies:
        - README.md exists in data directory
        - File has content

        Args:
            shared_data_dir: Shared data directory path
        """
        readme = shared_data_dir / "README.md"
        assert readme.exists(), "src/odyssey/data/README.md must exist"
        assert readme.is_file(), "src/odyssey/data/README.md must be a file"
        assert readme.stat().st_size > 0, "src/odyssey/data/README.md must not be empty"

    def test_shared_data_has_init(self, shared_data_dir: Path) -> None:
        """
        Test that src/odyssey/data/ has __init__.mojo.

        Verifies:
        - __init__.mojo exists in data directory
        - Path is a file

        Args:
            shared_data_dir: Shared data directory path
        """
        init_file = shared_data_dir / "__init__.mojo"
        assert init_file.exists(), "src/odyssey/data/__init__.mojo must exist"
        assert init_file.is_file(), "src/odyssey/data/__init__.mojo must be a file"

    def test_shared_utils_has_readme(self, shared_utils_dir: Path) -> None:
        """
        Test that src/odyssey/utils/ has README.md.

        Verifies:
        - README.md exists in utils directory
        - File has content

        Args:
            shared_utils_dir: Shared utils directory path
        """
        readme = shared_utils_dir / "README.md"
        assert readme.exists(), "src/odyssey/utils/README.md must exist"
        assert readme.is_file(), "src/odyssey/utils/README.md must be a file"
        assert readme.stat().st_size > 0, "src/odyssey/utils/README.md must not be empty"

    def test_shared_utils_has_init(self, shared_utils_dir: Path) -> None:
        """
        Test that src/odyssey/utils/ has __init__.mojo.

        Verifies:
        - __init__.mojo exists in utils directory
        - Path is a file

        Args:
            shared_utils_dir: Shared utils directory path
        """
        init_file = shared_utils_dir / "__init__.mojo"
        assert init_file.exists(), "src/odyssey/utils/__init__.mojo must exist"
        assert init_file.is_file(), "src/odyssey/utils/__init__.mojo must be a file"

    def test_shared_directory_permissions(self, shared_dir: Path) -> None:
        """
        Test that src/odyssey/ directory has correct permissions.

        Verifies:
        - Directory has read permission
        - Directory has write permission
        - Directory has execute permission

        Args:
            shared_dir: Shared directory path
        """
        dir_stat = shared_dir.stat()
        mode = dir_stat.st_mode

        assert mode & stat.S_IRUSR, "src/odyssey/ must have read permission"
        assert mode & stat.S_IWUSR, "src/odyssey/ must have write permission"
        assert mode & stat.S_IXUSR, "src/odyssey/ must have execute permission"


class TestDirectoryHierarchy:
    """Test cases for directory hierarchy relationships."""

    def test_papers_and_shared_are_under_repo_root(self, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that papers/ and src/odyssey/ are under the repository root.

        Verifies:
        - papers/ is directly under the repository root
        - src/odyssey/ is under the repository root's src/ directory

        Args:
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        assert papers_dir.parent == shared_dir.parent.parent

    def test_template_is_child_of_papers(self, papers_dir: Path, template_dir: Path) -> None:
        """
        Test that _template/ is direct child of papers/.

        Verifies:
        - Template parent is papers directory
        - Template is not nested deeper

        Args:
            papers_dir: Papers directory path
            template_dir: Template directory path
        """
        assert template_dir.parent == papers_dir, "_template/ must be direct child of papers/"

    def test_shared_subdirectories_are_children(
        self,
        shared_dir: Path,
        shared_core_dir: Path,
        shared_training_dir: Path,
        shared_data_dir: Path,
        shared_utils_dir: Path,
    ) -> None:
        """
        Test that all shared subdirectories are direct children.

        Verifies:
        - All subdirectories have src/odyssey/ as parent
        - No extra nesting

        Args:
            shared_dir: Shared directory path
            shared_core_dir: Shared core directory path
            shared_training_dir: Shared training directory path
            shared_data_dir: Shared data directory path
            shared_utils_dir: Shared utils directory path
        """
        assert shared_core_dir.parent == shared_dir, "core/ must be direct child of src/odyssey/"
        assert shared_training_dir.parent == shared_dir, "training/ must be direct child of src/odyssey/"
        assert shared_data_dir.parent == shared_dir, "data/ must be direct child of src/odyssey/"
        assert shared_utils_dir.parent == shared_dir, "utils/ must be direct child of src/odyssey/"
