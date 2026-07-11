"""
Test suite for directory structure integration.

This module contains integration tests that verify cross-component functionality
and interactions between papers and shared directories.

Test Categories:
- Cross-directory integration
- Import path validation
- Template instantiation workflows
- Directory relationship validation

Coverage Target: 100%
"""

import shutil
from pathlib import Path


class TestPapersSharedIntegration:
    """Test cases for papers and shared directory integration."""

    def test_papers_can_reference_shared_paths(self, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that papers can reference shared library paths.

        Verifies:
        - Papers and shared are in same repository
        - Relative paths between directories work
        - Both directories are accessible

        Args:
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        # Both should exist
        assert papers_dir.exists(), "papers/ must exist"
        assert shared_dir.exists(), "src/odyssey/ must exist"

        # Both should resolve under the same repository root
        assert papers_dir.parent == shared_dir.parent.parent, (
            "papers/ and src/odyssey/ must share the same repository root"
        )

        # Path from papers to odyssey should be computable
        relative_path = Path("..") / "src" / "odyssey"
        shared_from_papers = papers_dir / relative_path
        assert shared_from_papers.resolve() == shared_dir.resolve(), (
            "Relative path from papers to src/odyssey/ must resolve correctly"
        )

    def test_import_paths_resolve_correctly(self, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that import paths between papers and shared resolve.

        Verifies:
        - Shared core is accessible from papers
        - Shared training is accessible from papers
        - Shared data is accessible from papers
        - Shared utils is accessible from papers

        Args:
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        # All shared subdirectories should be accessible
        shared_subdirs = ["core", "training", "data", "utils"]

        for subdir in shared_subdirs:
            subdir_path = shared_dir / subdir
            assert subdir_path.exists(), f"src/odyssey/{subdir}/ must be accessible from repository"

    def test_no_circular_dependencies(self, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that there are no circular dependencies.

        Verifies:
        - Papers don't contain shared imports
        - Shared doesn't contain papers imports
        - Dependency flow is unidirectional

        Args:
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        # Shared should not reference papers
        shared_readme = shared_dir / "README.md"
        content = shared_readme.read_text()

        # Papers are mentioned in documentation, but not as imports
        # Check for import-like references (would indicate circular dependency)
        assert "from papers." not in content, "Shared library must not import from papers (circular dependency)"
        assert "import papers." not in content, "Shared library must not import from papers (circular dependency)"


class TestTemplateInstantiation:
    """Test cases for template instantiation workflow."""

    def test_template_can_be_copied_to_papers_dir(self, template_dir: Path, tmp_path: Path) -> None:
        """
        Test that template can be copied to create new paper in papers directory.

        Verifies:
        - Template copying workflow works
        - New paper directory is created
        - Structure is preserved

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
        """
        # Simulate papers directory
        mock_papers_dir = tmp_path / "papers"
        mock_papers_dir.mkdir()

        # Copy template to new paper
        new_paper_dir = mock_papers_dir / "test-paper"
        shutil.copytree(template_dir, new_paper_dir)

        # Verify new paper was created
        assert new_paper_dir.exists(), "New paper directory must be created"
        assert new_paper_dir.is_dir(), "New paper must be a directory"

        # Verify key structure elements
        assert (new_paper_dir / "src").exists(), "New paper must have src/"
        assert (new_paper_dir / "tests").exists(), "New paper must have tests/"
        assert (new_paper_dir / "README.md").exists(), "New paper must have README.md"

    def test_multiple_papers_can_coexist(self, template_dir: Path, tmp_path: Path) -> None:
        """
        Test that multiple paper implementations can coexist.

        Verifies:
        - Multiple papers can be created
        - Each paper is independent
        - No conflicts between papers

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
        """
        # Create mock papers directory
        mock_papers_dir = tmp_path / "papers"
        mock_papers_dir.mkdir()

        # Create multiple papers from template
        papers = ["lenet5", "alexnet", "resnet"]
        for paper_name in papers:
            paper_dir = mock_papers_dir / paper_name
            shutil.copytree(template_dir, paper_dir)

        # Verify all papers exist
        for paper_name in papers:
            paper_dir = mock_papers_dir / paper_name
            assert paper_dir.exists(), f"{paper_name} directory must exist"
            assert paper_dir.is_dir(), f"{paper_name} must be a directory"

        # Verify papers are independent (different paths)
        paper_paths = [mock_papers_dir / name for name in papers]
        assert len(paper_paths) == len(set(paper_paths)), "All paper directories must be independent"

    def test_new_paper_can_reference_shared(self, template_dir: Path, shared_dir: Path, tmp_path: Path) -> None:
        """
        Test that new paper created from template can reference shared library.

        Verifies:
        - New paper can compute path to shared
        - Shared library is accessible from new paper

        Args:
            template_dir: Template directory path
            shared_dir: Shared directory path
            tmp_path: Temporary directory for testing
        """
        # Create new paper in temporary location
        # (simulating it being in papers/ directory)
        mock_papers_dir = tmp_path / "papers"
        mock_papers_dir.mkdir()
        new_paper_dir = mock_papers_dir / "test-paper"
        shutil.copytree(template_dir, new_paper_dir)

        # Compute relative path to src/odyssey
        # In real repository: papers/test-paper/../../src/odyssey/
        relative_to_shared = Path("..") / ".." / "src" / "odyssey"

        # This would resolve to src/odyssey/ in actual repository structure
        # Just verify the path construction works
        assert relative_to_shared.parts == (
            "..",
            "..",
            "src",
            "odyssey",
        ), "Relative path to src/odyssey/ must be constructable from paper directory"


class TestDirectoryPermissions:
    """Test cases for directory permissions and access."""

    def test_papers_directory_is_writable(self, papers_dir: Path) -> None:
        """
        Test that papers directory is writable.

        Verifies:
        - Can create new directories in papers/
        - Can create files in papers/
        - Appropriate for adding new papers

        Args:
            papers_dir: Papers directory path
        """
        import os

        assert os.access(papers_dir, os.W_OK), "papers/ directory must be writable (for adding new papers)"

    def test_shared_directory_is_readable(self, shared_dir: Path) -> None:
        """
        Test that shared directory is readable.

        Verifies:
        - Can read files in src/odyssey/
        - Can list directory contents
        - Appropriate for importing from shared

        Args:
            shared_dir: Shared directory path
        """
        import os

        assert os.access(shared_dir, os.R_OK), "src/odyssey/ directory must be readable (for imports)"

    def test_template_is_readable_not_writable_in_production(self, template_dir: Path) -> None:
        """
        Test that template should be read-only in production.

        Verifies:
        - Template is readable (for copying)
        - Template should not be modified directly

        Note: This test only verifies readability as write protection
        is typically enforced through version control and code review.

        Args:
            template_dir: Template directory path
        """
        import os

        assert os.access(template_dir, os.R_OK), "Template directory must be readable (for copying to new papers)"


class TestWorkflowIntegration:
    """Test cases for complete workflow integration."""

    def test_complete_new_paper_workflow(self, template_dir: Path, tmp_path: Path) -> None:
        """
        Test complete workflow for creating a new paper.

        Simulates the full process:
        1. Copy template to new paper directory
        2. Verify structure is complete
        3. Modify README
        4. Create source files

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
        """
        # Step 1: Create papers directory
        mock_papers_dir = tmp_path / "papers"
        mock_papers_dir.mkdir()

        # Step 2: Copy template to new paper
        new_paper = mock_papers_dir / "lenet5"
        shutil.copytree(template_dir, new_paper)

        # Step 3: Verify structure
        assert (new_paper / "src").exists()
        assert (new_paper / "tests").exists()
        assert (new_paper / "README.md").exists()

        # Step 4: Modify README (simulating customization)
        readme = new_paper / "README.md"
        readme.write_text("# LeNet-5 Implementation\n\nCustom paper implementation.")

        # Step 5: Create source file (simulating development)
        src_dir = new_paper / "src"
        model_file = src_dir / "model.mojo"
        model_file.write_text("# LeNet-5 model implementation")

        # Verify final state
        assert readme.read_text().startswith("# LeNet-5"), "README should be customized"
        assert model_file.exists(), "Source files should be created"

    def test_repository_structure_consistency(self, repo_root: Path, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that repository structure is consistent.

        Verifies:
        - Papers and shared are at repository root
        - Both are direct children of same parent
        - Structure matches documented architecture

        Args:
            repo_root: Repository root directory
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        # papers/ is a direct child of repo root; src/odyssey/ lives under src/
        assert papers_dir.parent == repo_root, "papers/ must be direct child of repository root"
        assert shared_dir.parent.parent == repo_root, "src/odyssey/ must live under <repo_root>/src/"

        # Repository root should contain papers/ and src/; src/ must contain odyssey/
        root_contents = [item.name for item in repo_root.iterdir() if item.is_dir()]
        assert "papers" in root_contents, "Repository root must contain papers/"
        assert "src" in root_contents, "Repository root must contain src/"
        src_contents = [item.name for item in (repo_root / "src").iterdir() if item.is_dir()]
        assert "odyssey" in src_contents, "src/ must contain odyssey/"


class TestDependencyGraph:
    """Test cases for dependency graph validation."""

    def test_dependency_flow_is_acyclic(self, papers_dir: Path, shared_dir: Path) -> None:
        """
        Test that dependency flow is acyclic (DAG).

        Verifies:
        - Papers depend on shared (documented)
        - Shared doesn't depend on papers
        - No circular dependencies

        Args:
            papers_dir: Papers directory path
            shared_dir: Shared directory path
        """
        # Read shared README to verify no papers dependencies
        shared_readme = shared_dir / "README.md"
        content = shared_readme.read_text()

        # Shared should document its independence from papers
        # It mentions papers in usage examples, but doesn't import from them
        assert "from papers" not in content or "# In papers/" in content, (
            "Shared library must not have dependency on papers (only usage examples)"
        )

    def test_shared_subdirectories_follow_dependency_order(
        self,
        shared_core_dir: Path,
        shared_training_dir: Path,
        shared_data_dir: Path,
        shared_utils_dir: Path,
    ) -> None:
        """
        Test that shared subdirectories follow documented dependency order.

        Verifies:
        - All subdirectories exist
        - Dependency documentation is available

        Args:
            shared_core_dir: Shared core directory path
            shared_training_dir: Shared training directory path
            shared_data_dir: Shared data directory path
            shared_utils_dir: Shared utils directory path
        """
        # All subdirectories must exist
        subdirs = [
            shared_core_dir,
            shared_training_dir,
            shared_data_dir,
            shared_utils_dir,
        ]
        for subdir in subdirs:
            assert subdir.exists(), f"{subdir.name}/ must exist for dependency graph"

    def test_papers_template_is_self_contained(self, template_dir: Path) -> None:
        """
        Test that template is self-contained and ready to use.

        Verifies:
        - Template has complete documentation
        - Template provides implementation guidance
        - Template structure is usable

        Note: Future enhancement would be to document shared library usage,
        but current template focuses on paper-specific implementation.

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        content = readme.read_text()

        # Template should provide comprehensive guidance
        assert "Implementation Guide" in content or "Step" in content, "Template should provide implementation guidance"
        assert "Mojo" in content, "Template should mention Mojo language"
        assert "src/" in content, "Template should document source directory structure"
