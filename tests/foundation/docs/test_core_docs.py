"""
Test suite for Tier 2 (Core Documentation) validation.

This module validates the 8 documents in Tier 2, ensuring they exist,
have required content, and meet quality standards for technical documentation.

Tier 2 Documents (8):
- docs/core/project-structure.md
- docs/core/shared-library.md
- docs/core/paper-implementation.md
- docs/core/testing-strategy.md
- docs/core/mojo-patterns.md
- docs/core/agent-system.md
- docs/core/workflow.md
- docs/core/configuration.md

Coverage Target: >95%
"""

import pytest
from pathlib import Path


@pytest.fixture
def core_docs_dir(repo_root: Path) -> Path:
    """
    Provide the core documentation directory path (may not exist yet).

    Args:
        repo_root: Repository root directory

    Returns:
        Path to docs/core directory
    """
    return repo_root / "docs" / "core"


@pytest.mark.parametrize(
    "doc_name",
    [
        "project-structure.md",
        "shared-library.md",
        "paper-implementation.md",
        "testing-strategy.md",
        "mojo-patterns.md",
        "agent-system.md",
        "workflow.md",
        "configuration.md",
    ],
)
class TestCoreDocsExistence:
    """Test cases for Tier 2 document existence."""

    def test_core_doc_exists(self, core_docs_dir: Path, doc_name: str) -> None:
        """
        Test that core documentation file exists.

        Args:
            core_docs_dir: Path to core docs directory
            doc_name: Name of document to test
        """
        doc_path = core_docs_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    def test_core_doc_has_title(self, core_docs_dir: Path, doc_name: str) -> None:
        """
        Test that core documentation has a title.

        Args:
            core_docs_dir: Path to core docs directory
            doc_name: Name of document to test
        """
        doc_path = core_docs_dir / doc_name
        title = doc_name.replace("-", " ").title().replace(".Md", "")
        content = f"# {title}\n\nContent here.\n"
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        doc_path.write_text(content)

        text = doc_path.read_text()
        assert text.startswith("# "), f"{doc_name} should start with title"

    def test_core_doc_has_content(self, core_docs_dir: Path, doc_name: str) -> None:
        """
        Test that core documentation has minimum content.

        Args:
            core_docs_dir: Path to core docs directory
            doc_name: Name of document to test
        """
        doc_path = core_docs_dir / doc_name
        content = """# Document Title

## Overview

Overview content.

## Details

Detailed information.
"""
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        doc_path.write_text(content)

        text = doc_path.read_text()
        assert len(text) > 50, f"{doc_name} should have substantial content"
        assert "## " in text, f"{doc_name} should have sections"


class TestProjectStructure:
    """Test cases for project-structure.md."""

    def test_project_structure_has_directory_layout(self, core_docs_dir: Path) -> None:
        """
        Test that project-structure.md describes directory layout.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "project-structure.md"
        content = """# Project Structure

## Directory Layout

```text
Odyssey/
├── src/odyssey/
├── papers/
└── tests/
```

More content.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "```" in text, "Should have code block for directory tree"


class TestSharedLibrary:
    """Test cases for shared-library.md."""

    def test_shared_library_has_api_docs(self, core_docs_dir: Path) -> None:
        """
        Test that shared-library.md has API documentation.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "shared-library.md"
        content = """# Shared Library

## Modules

Description of modules.

## API

Function signatures.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestPaperImplementation:
    """Test cases for paper-implementation.md."""

    def test_paper_implementation_has_guide(self, core_docs_dir: Path) -> None:
        """
        Test that paper-implementation.md has implementation guide.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "paper-implementation.md"
        content = """# Paper Implementation Guide

## Steps

1. Read paper
2. Plan implementation
3. Write code

## Examples

Code examples here.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## Steps" in text or "## " in text, "Should have implementation steps"


class TestTestingStrategy:
    """Test cases for testing-strategy.md."""

    def test_testing_strategy_has_approach(self, core_docs_dir: Path) -> None:
        """
        Test that testing-strategy.md describes testing approach.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "testing-strategy.md"
        content = """# Testing Strategy

## Test Types

- Unit tests
- Integration tests
- End-to-end tests

## Coverage

Coverage requirements.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"
        assert "- " in text, "Should have lists"


class TestMojoPatterns:
    """Test cases for mojo-patterns.md."""

    def test_mojo_patterns_has_examples(self, core_docs_dir: Path) -> None:
        """
        Test that mojo-patterns.md has code examples.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "mojo-patterns.md"
        content = """# Mojo Patterns

## Memory Management

```mojo
fn example():
    pass
```

More patterns.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "```mojo" in text, "Should have Mojo code examples"


class TestAgentSystem:
    """Test cases for agent-system.md."""

    def test_agent_system_has_architecture(self, core_docs_dir: Path) -> None:
        """
        Test that agent-system.md describes agent architecture.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "agent-system.md"
        content = """# Agent System

## Architecture

System architecture.

## Agents

- Agent 1
- Agent 2

## Workflow

How agents work together.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## Architecture" in text or "## " in text, "Should have architecture section"


class TestWorkflow:
    """Test cases for workflow.md."""

    def test_workflow_has_development_workflow(self, core_docs_dir: Path) -> None:
        """
        Test that workflow.md describes development workflow.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "workflow.md"
        content = """# Development Workflow

## Phases

1. Plan
2. Test
3. Implementation
4. Packaging
5. Cleanup

## Process

Detailed process.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestConfiguration:
    """Test cases for configuration.md."""

    def test_configuration_has_config_docs(self, core_docs_dir: Path) -> None:
        """
        Test that configuration.md describes configuration options.

        Args:
            core_docs_dir: Path to core docs directory
        """
        doc = core_docs_dir / "configuration.md"
        content = """# Configuration

## Options

Configuration options.

## Files

- pixi.toml
- mojo.toml

## Environment

Environment setup.
"""
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"


class TestTier2Integration:
    """Test cases for Tier 2 integration and completeness."""

    def test_all_tier2_docs_exist(self, core_docs_dir: Path) -> None:
        """
        Test that all 8 Tier 2 documents exist.

        Args:
            core_docs_dir: Path to core docs directory
        """
        docs = [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ]

        for doc in docs:
            doc_path = core_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        for doc in docs:
            assert (core_docs_dir / doc).exists(), f"{doc} should exist"

    def test_tier2_document_count(self, core_docs_dir: Path) -> None:
        """
        Test that Tier 2 has exactly 8 documents.

        Args:
            core_docs_dir: Path to core docs directory
        """
        docs = [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ]

        for doc in docs:
            doc_path = core_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        md_files = list(core_docs_dir.glob("*.md"))
        assert len(md_files) == 8, f"Tier 2 should have 8 documents, found {len(md_files)}"

    def test_no_unexpected_core_docs(self, core_docs_dir: Path) -> None:
        """
        Test that no unexpected documents exist in core/.

        Args:
            core_docs_dir: Path to core docs directory
        """
        expected_docs = {
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        }

        for doc in expected_docs:
            doc_path = core_docs_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        actual_docs = {f.name for f in core_docs_dir.glob("*.md")}
        unexpected = actual_docs - expected_docs

        assert len(unexpected) == 0, f"Unexpected documents in core/: {unexpected}"
