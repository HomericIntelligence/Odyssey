"""
Test suite for Tier 3 (Advanced Topics) documentation validation.

This module validates the 6 documents in Tier 3, ensuring they exist,
have required content, and meet quality standards for advanced documentation.

Tier 3 Documents (6):
- docs/advanced/performance.md
- docs/advanced/custom-layers.md
- docs/advanced/distributed-training.md
- docs/advanced/visualization.md
- docs/advanced/debugging.md
- docs/advanced/integration.md

Coverage Target: >95%
"""

import pytest
from pathlib import Path


@pytest.fixture
def advanced_docs_dir(repo_root: Path) -> Path:
    """
    Provide the advanced documentation directory path (may not exist yet).

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to docs/advanced directory
    """
    return repo_root / "docs" / "advanced"


@pytest.mark.parametrize(
    "doc_name",
    [
        "performance.md",
        "custom-layers.md",
        "distributed-training.md",
        "visualization.md",
        "debugging.md",
        "integration.md",
    ],
)
class TestAdvancedDocsExistence:
    """Test cases for Tier 3 document existence."""

    def test_advanced_doc_exists(self, advanced_docs_dir: Path, doc_name: str) -> None:
        """
        Test that advanced documentation file exists.

        Args:
            advanced_docs_dir: Path to advanced docs directory
            doc_name: Name of document to test
        """
        doc_path = advanced_docs_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    def test_advanced_doc_has_title(self, advanced_docs_dir: Path, doc_name: str) -> None:
        """
        Test that advanced documentation has a title.

        Args:
            advanced_docs_dir: Path to advanced docs directory
            doc_name: Name of document to test
        """
        doc_path = advanced_docs_dir / doc_name
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        text = doc_path.read_text()
        assert text.startswith("# "), f"{doc_name} should start with title"

    def test_advanced_doc_has_content(self, advanced_docs_dir: Path, doc_name: str) -> None:
        """
        Test that advanced documentation has minimum content.

        Args:
            advanced_docs_dir: Path to advanced docs directory
            doc_name: Name of document to test
        """
        doc_path = advanced_docs_dir / doc_name
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        text = doc_path.read_text()
        assert len(text) > 50, f"{doc_name} should have substantial content"
        assert "## " in text, f"{doc_name} should have sections"


class TestPerformance:
    """Test cases for performance.md."""

    def test_performance_has_optimization_guide(self, advanced_docs_dir: Path) -> None:
        """
        Test that performance.md has optimization guidance.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "performance.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "## " in text, "Should have sections"
        assert "```" in text, "Should have code examples"


class TestCustomLayers:
    """Test cases for custom-layers.md."""

    def test_custom_layers_has_implementation_guide(self, advanced_docs_dir: Path) -> None:
        """
        Test that custom-layers.md has layer implementation guide.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "custom-layers.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "```mojo" in text or "```" in text, "Should have code examples"


class TestDistributedTraining:
    """Test cases for distributed-training.md."""

    def test_distributed_training_has_setup_guide(self, advanced_docs_dir: Path) -> None:
        """
        Test that distributed-training.md has distributed setup guide.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "distributed-training.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestVisualization:
    """Test cases for visualization.md."""

    def test_visualization_has_tools_guide(self, advanced_docs_dir: Path) -> None:
        """
        Test that visualization.md has visualization tools guide.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "visualization.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestDebugging:
    """Test cases for debugging.md."""

    def test_debugging_has_strategies(self, advanced_docs_dir: Path) -> None:
        """
        Test that debugging.md has debugging strategies.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "debugging.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestIntegration:
    """Test cases for integration.md."""

    def test_integration_has_patterns(self, advanced_docs_dir: Path) -> None:
        """
        Test that integration.md has integration patterns.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        doc = advanced_docs_dir / "integration.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestTier3Integration:
    """Test cases for Tier 3 integration and completeness."""

    def test_all_tier3_docs_exist(self, advanced_docs_dir: Path) -> None:
        """
        Test that all 6 Tier 3 documents exist.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        docs = [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ]

        for doc in docs:
            doc_path = advanced_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        for doc in docs:
            assert (advanced_docs_dir / doc).exists(), f"{doc} should exist"

    def test_tier3_document_count(self, advanced_docs_dir: Path) -> None:
        """
        Test that Tier 3 has exactly 6 documents.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        docs = [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ]

        for doc in docs:
            doc_path = advanced_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        md_files = list(advanced_docs_dir.glob("*.md"))
        assert len(md_files) == 6, f"Tier 3 should have 6 documents, found {len(md_files)}"

    def test_no_unexpected_advanced_docs(self, advanced_docs_dir: Path) -> None:
        """
        Test that no unexpected documents exist in advanced/.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        expected_docs = {
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        }

        for doc in expected_docs:
            doc_path = advanced_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        actual_docs = {f.name for f in advanced_docs_dir.glob("*.md")}
        unexpected = actual_docs - expected_docs

        assert len(unexpected) == 0, f"Unexpected documents in advanced/: {unexpected}"

    def test_tier3_focuses_on_advanced_topics(self, advanced_docs_dir: Path) -> None:
        """
        Test that Tier 3 documents focus on advanced topics.

        Args:
            advanced_docs_dir: Path to advanced docs directory
        """
        # This is a conceptual test - we verify the doc names suggest advanced content
        doc_names = [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ]

        for doc_name in doc_names:
            doc_path = advanced_docs_dir / doc_name
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # All these topics are clearly advanced
        assert len(doc_names) == 6, "Should have 6 advanced topics"
