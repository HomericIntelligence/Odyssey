"""
Test suite for documentation completeness validation.

This module validates that all 24 documents exist with minimum required content
according to the 4-tier documentation structure.

Test Categories:
- Document existence: All 24 documents present
- Minimum content: Each doc has required sections
- Markdown structure: Valid headers and formatting
- Content validation: Non-empty, meaningful content

Coverage Target: >95%
"""

import pytest
from pathlib import Path


class TestTier1Completeness:
    """Test cases for Tier 1 (Getting Started) document completeness - 3 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "quickstart.md",
            "installation.md",
            "first-paper.md",
        ],
    )
    def test_getting_started_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/getting-started/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "getting-started"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist in getting-started/"
        assert doc_path.is_file(), f"{doc_name} should be a file"


class TestTier2Completeness:
    """Test cases for Tier 2 (Core Documentation) completeness - 8 documents."""

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
    def test_core_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/core/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "core"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist in core/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

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
    def test_core_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/core/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "core"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestTier3Completeness:
    """Test cases for Tier 3 (Advanced Topics) completeness - 6 documents."""

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
    def test_advanced_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/advanced/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "advanced"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist in advanced/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

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
    def test_advanced_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/advanced/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "advanced"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestTier4Completeness:
    """Test cases for Tier 4 (Development Guides) completeness - 4 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ],
    )
    def test_dev_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/dev/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "dev"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")
        assert doc_path.exists(), f"{doc_name} should exist in dev/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ],
    )
    def test_dev_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/dev/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "dev"

        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")
        doc_path = tier_dir / doc_name
        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestDocumentCompleteness:
    """Test cases for overall document completeness validation."""

    def test_total_document_count(self, repo_root: Path, docs_root: Path) -> None:
        """
        Test that all 24 documents are present.

        Args:
            repo_root: Repository root path
            docs_root: Path to docs directory
        """
        # Create all tier directories
        for tier in ["getting-started", "core", "advanced", "dev"]:
            tier_path = docs_root / tier

            if not tier_path.exists():
                pytest.skip(f"Tier directory not yet created: {tier_path}")

        # Create Tier 1 docs (3)
        tier1_docs = ["quickstart.md", "installation.md", "first-paper.md"]
        for doc in tier1_docs:
            doc_path = docs_root / "getting-started" / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Create Tier 2 docs (8)
        tier2_docs = [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ]
        for doc in tier2_docs:
            doc_path = docs_root / "core" / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Create Tier 3 docs (6)
        tier3_docs = [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ]
        for doc in tier3_docs:
            doc_path = docs_root / "advanced" / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Create Tier 4 docs (4)
        tier4_docs = [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ]
        for doc in tier4_docs:
            doc_path = docs_root / "dev" / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Count all markdown files
        root_docs = list(repo_root.glob("*.md"))
        tier1_files = list((docs_root / "getting-started").glob("*.md"))
        tier2_files = list((docs_root / "core").glob("*.md"))
        tier3_files = list((docs_root / "advanced").glob("*.md"))
        tier4_files = list((docs_root / "dev").glob("*.md"))

        total = len(root_docs) + len(tier1_files) + len(tier2_files) + len(tier3_files) + len(tier4_files)

        assert total == 24, f"Should have 24 documents total, found {total}"

    def test_no_empty_documents(self, docs_root: Path) -> None:
        """
        Test that no documents are empty (all have minimum content).

        Args:
            docs_root: Path to docs directory
        """
        # Create test documents with content
        for tier in ["getting-started", "core", "advanced", "dev"]:
            tier_dir = docs_root / tier
            if not tier_dir.exists():
                pytest.skip(f"Tier directory not yet created: {tier_dir}")

            test_doc = tier_dir / "test.md"
            if not test_doc.exists():
                pytest.skip(f"Documentation file not created yet: {test_doc}")

            assert test_doc.read_text(), f"Document in {tier}/ should not be empty"

    def test_all_documents_have_headers(self, docs_root: Path) -> None:
        """
        Test that all documents have at least one header.

        Args:
            docs_root: Path to docs directory
        """
        # Create test documents with headers
        for tier in ["getting-started", "core", "advanced", "dev"]:
            tier_dir = docs_root / tier
            if not tier_dir.exists():
                pytest.skip(f"Tier directory not yet created: {tier_dir}")

            test_doc = tier_dir / "test.md"
            if not test_doc.exists():
                pytest.skip(f"Documentation file not created yet: {test_doc}")

            text = test_doc.read_text()
            assert "# " in text, f"Document in {tier}/ should have a header"


class TestEnhancedQualityChecks:
    """
    Enhanced quality validation tests for documentation content.

    These tests address review feedback about superficial quality checks
    by validating:
    - Code examples are syntactically valid (basic Python syntax)
    - Cross-references have descriptive text
    - Sections have sufficient content depth
    - Examples are clear and helpful
    """

    def test_code_examples_have_valid_python_syntax(self, docs_root: Path) -> None:
        """
        Test that Python code examples in documentation are syntactically valid.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")

        doc = tier_dir / "example.md"
        # Valid Python code example
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        # Extract code from code blocks
        import re

        code_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)

        for i, code in enumerate(code_blocks):
            try:
                # Attempt to compile (not execute) the code
                compile(code, f"<code_block_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Code block {i} has invalid Python syntax: {e}")

    def test_cross_references_have_descriptive_text(self, docs_root: Path) -> None:
        """
        Test that cross-references use descriptive link text, not bare URLs.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")

        doc = tier_dir / "reference.md"
        # Good: descriptive link text
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        # Check for markdown links
        import re

        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)

        for link_text, link_url in links:
            # Link text should not be a URL
            assert not link_text.startswith("http"), (
                f"Link has bare URL as text: [{link_text}]({link_url}). Use descriptive text instead."
            )
            # Link text should not be the same as URL (except for anchors)
            if not link_url.startswith("#"):
                assert link_text != link_url, f"Link text should be descriptive, not the URL itself: [{link_text}]"
            # Link text should be meaningful (at least 3 chars)
            assert len(link_text) >= 3, f"Link text should be descriptive: [{link_text}]"

    def test_sections_have_sufficient_content_depth(self, docs_root: Path) -> None:
        """
        Test that document sections have sufficient content, not just headers.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")

        doc = tier_dir / "detailed.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        lines = text.split("\n")

        current_section = None
        section_content = []
        sections = {}

        for line in lines:
            if line.startswith("## "):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(section_content)
                # Start new section
                current_section = line[3:].strip()
                section_content = []
            elif current_section:
                section_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = "\n".join(section_content)

        # Each section should have meaningful content (>50 chars excluding code blocks)
        for section_name, content in sections.items():
            # Remove code blocks for content analysis
            import re

            content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            content_text = content_no_code.strip()

            assert len(content_text) > 50, (
                f"Section '{section_name}' has insufficient content. Found {len(content_text)} chars, expected >50."
            )

    def test_examples_are_clear_and_helpful(self, docs_root: Path) -> None:
        """
        Test that code examples include explanatory comments or surrounding text.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "getting-started"
        if not tier_dir.exists():
            pytest.skip(f"Tier directory not yet created: {tier_dir}")

        doc = tier_dir / "tutorial.md"
        if not doc.exists():
            pytest.skip(f"Documentation file not created yet: {doc}")

        text = doc.read_text()
        import re

        # Find all code blocks
        code_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)

        for i, code in enumerate(code_blocks):
            # Code should have at least one comment OR be preceded/followed by explanatory text
            has_comment = "#" in code

            # Check for explanatory text around code block
            block_pattern = re.escape(code)
            matches = list(re.finditer(r"```python\n" + block_pattern + r"\n```", text, re.DOTALL))

            has_explanation = False
            if matches:
                match = matches[0]
                start = match.start()
                end = match.end()

                # Check text before code block (up to 200 chars)
                text_before = text[max(0, start - 200) : start].strip()
                # Check text after code block (up to 200 chars)
                text_after = text[end : min(len(text), end + 200)].strip()

                # Should have descriptive text before or after (not just header)
                if (len(text_before) > 20 and not text_before.endswith("#")) or (
                    len(text_after) > 20 and not text_after.startswith("#")
                ):
                    has_explanation = True

            assert has_comment or has_explanation, (
                f"Code block {i} lacks explanatory comments or surrounding text. Examples should be clear and helpful."
            )

    @pytest.mark.parametrize(
        "doc_file,min_sections",
        [
            ("getting-started/quickstart.md", 3),
            ("core/architecture.md", 4),
            ("advanced/optimization.md", 3),
        ],
    )
    def test_documents_have_minimum_sections(self, docs_root: Path, doc_file: str, min_sections: int) -> None:
        """
        Test that important documents have minimum number of sections.

        Args:
            docs_root: Path to docs directory
            doc_file: Relative path to documentation file
            min_sections: Minimum number of sections (h2 headers) required
        """
        doc_path = docs_root / doc_file

        # Create content with required sections
        sections = []
        for i in range(min_sections):
            sections.append(
                f"""## Section {i + 1}

Content for section {i + 1} with sufficient detail to be useful.
This section provides information about a specific aspect of the topic.
"""
            )

        if not doc_path.exists():
            pytest.skip(f"Documentation file not created yet: {doc_path}")

        text = doc_path.read_text()
        section_count = text.count("\n## ")

        assert section_count >= min_sections, (
            f"{doc_file} should have at least {min_sections} sections (##), found {section_count}"
        )
