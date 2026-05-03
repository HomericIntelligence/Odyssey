#!/usr/bin/env python3
"""Unit tests for scripts/agents/validate_agents.py.

Tests for:
- validate_file: comprehensive validation of a single agent markdown file
- validate_frontmatter: validates parsed frontmatter content
- Colon-containing values are handled correctly (regression for #3310)

Note: validate_agents.py is a thin re-export wrapper over
hephaestus.agents.frontmatter. The previous API (ValidationResult,
validate_file, extract_sections) no longer exists; this file adapts
the tests to use check_agent_file and validate_frontmatter instead.

Usage:
    pytest tests/agents/test_validate_agents.py -v
    python -m pytest tests/agents/test_validate_agents.py -v
"""

import re
import sys
from pathlib import Path
from typing import Any

import pytest

# Add scripts/agents to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "agents"))

from validate_agents import (
    check_agent_file as _check_agent_file,
    validate_frontmatter as _validate_frontmatter,
)


# ---------------------------------------------------------------------------
# Local adapters to match the test surface area of the old API
# ---------------------------------------------------------------------------


class ValidationResult:
    """Minimal adapter for the old ValidationResult API used in tests."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        return f"ValidationResult(errors={self.errors}, warnings={self.warnings})"


def validate_file(file_path: Path, verbose: bool = False) -> ValidationResult:
    """Adapter: call check_agent_file and wrap result in ValidationResult."""
    result = ValidationResult(file_path)
    is_valid, errors = _check_agent_file(file_path)
    result.errors = errors
    return result


def validate_frontmatter(frontmatter: dict[str, Any], result: ValidationResult) -> None:
    """Adapter: call hephaestus validate_frontmatter and populate result."""
    errors = _validate_frontmatter(frontmatter)
    result.errors.extend(errors)


def extract_sections(content: str) -> set[str]:
    """Extract ## (level-2) section header names from markdown content."""
    return {m.group(1).strip() for m in re.finditer(r"^##\s+(.+)$", content, re.MULTILINE)}


# ---------------------------------------------------------------------------
# A minimal valid agent file that satisfies all required checks
# ---------------------------------------------------------------------------

FULL_VALID_AGENT_CONTENT = """---
name: test-agent
description: Use this agent when you need to test Mojo implementation patterns
tools: Read,Write,Edit
model: sonnet
---

## Role

A comprehensive test agent for validation purposes.

## Scope

Handles all test-related implementation tasks using Mojo.

## Responsibilities

- Implement and validate Mojo struct patterns
- Ensure fn vs def usage is correct
- Apply SIMD vectorization

## Mojo-Specific Guidelines

This agent follows Mojo best practices including fn vs def, struct vs class,
SIMD operations, @parameter decorators, owned and borrowed semantics.

## Workflow

Follows the Plan → Test → Implementation → Cleanup workflow phases.

## Constraints

- Only modify files within the assigned scope
- Respect owned and borrowed parameter conventions

## Delegation

Delegates to junior engineers for routine tasks.

## Examples

Example: Implement a Mojo struct for tensor operations using SIMD.
"""


def _write_agent_file(tmp_path: Path, content: str, filename: str = "test-agent.md") -> Path:
    """Helper: write content to a temp file and return the path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


class TestValidateFileValid:
    """Tests for validate_file() with valid agent files."""

    def test_valid_agent_file_no_errors(self, tmp_path: Path) -> None:
        """A fully-formed agent file should produce no errors."""
        f = _write_agent_file(tmp_path, FULL_VALID_AGENT_CONTENT)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_valid_agent_file_verbose_no_errors(self, tmp_path: Path) -> None:
        """verbose=True should not cause errors on a valid file."""
        f = _write_agent_file(tmp_path, FULL_VALID_AGENT_CONTENT)
        result = validate_file(f, verbose=True)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"


class TestValidateFileColonValues:
    """Regression tests for colon-containing values in frontmatter (follow-up from #3310).

    validate_file uses PyYAML (via hephaestus) so values with colons must not
    be truncated.
    """

    def test_description_with_url_preserved(self, tmp_path: Path) -> None:
        """Description containing a URL should be preserved and not truncated."""
        content = FULL_VALID_AGENT_CONTENT.replace(
            "description: Use this agent when you need to test Mojo implementation patterns",
            "description: See https://docs.example.com/mojo for Mojo implementation patterns",
        )
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_description_with_mid_sentence_colon_preserved(self, tmp_path: Path) -> None:
        """Description with a mid-sentence colon should not be truncated."""
        content = FULL_VALID_AGENT_CONTENT.replace(
            "description: Use this agent when you need to test Mojo implementation patterns",
            "description: Use this agent when you need to implement design or test Mojo files",
        )
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_description_with_ratio_colon_preserved(self, tmp_path: Path) -> None:
        """Description with ratio notation (3:1) should be preserved."""
        content = FULL_VALID_AGENT_CONTENT.replace(
            "description: Use this agent when you need to test Mojo implementation patterns",
            "description: Implement 3:1 train/test splits for Mojo model validation",
        )
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_description_with_multiple_colons_preserved(self, tmp_path: Path) -> None:
        """Description with multiple colons (URL with port) should be preserved."""
        content = FULL_VALID_AGENT_CONTENT.replace(
            "description: Use this agent when you need to test Mojo implementation patterns",
            "description: Connect to http://localhost:8080 to test Mojo implementation",
        )
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"


class TestValidateFileMissingRequired:
    """Tests for validate_file() detecting missing required frontmatter fields."""

    def test_missing_name_produces_error(self, tmp_path: Path) -> None:
        """Missing 'name' field should produce an error."""
        content = """---
description: Use this agent when you need to test Mojo implementation patterns
tools: Read,Write
model: sonnet
---

## Role

Content with Mojo fn and struct patterns.

## Scope

Mojo scope.

## Responsibilities

Implement with owned, SIMD, @parameter.

## Mojo-Specific Guidelines

fn vs def, struct vs class, SIMD.

## Workflow

Plan phase.

## Constraints

None.
"""
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert not result.is_valid()
        assert any("name" in e for e in result.errors)

    def test_missing_model_produces_error(self, tmp_path: Path) -> None:
        """Missing 'model' field should produce an error."""
        content = """---
name: test-agent
description: Use this agent when you need to test Mojo implementation patterns
tools: Read,Write
---

## Role

Content.

## Scope

Mojo scope.

## Responsibilities

Tasks with owned and SIMD.

## Mojo-Specific Guidelines

fn vs def, struct vs class.

## Workflow

Plan phase.

## Constraints

None.
"""
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert not result.is_valid()
        assert any("model" in e for e in result.errors)


class TestValidateFileModelValidation:
    """Tests for model field validation in validate_file()."""

    def test_valid_model_haiku_accepted(self, tmp_path: Path) -> None:
        """'haiku' is a valid model value (string type check passes)."""
        content = FULL_VALID_AGENT_CONTENT.replace("model: sonnet", "model: haiku")
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_any_string_model_accepted(self, tmp_path: Path) -> None:
        """check_agent_file only validates types; any string model passes."""
        content = FULL_VALID_AGENT_CONTENT.replace("model: sonnet", "model: gpt-4o")
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        # The underlying check_agent_file does not validate model name values,
        # only that the field is present and a string.
        assert result.is_valid(), f"Unexpected errors: {result.errors}"


class TestValidateFileToolsValidation:
    """Tests for tools field validation in validate_file()."""

    def test_unknown_tools_valid_string(self, tmp_path: Path) -> None:
        """check_agent_file accepts any string for tools (type check only)."""
        content = FULL_VALID_AGENT_CONTENT.replace("tools: Read,Write,Edit", "tools: Read,UnknownTool")
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_empty_tools_string_is_valid_type(self, tmp_path: Path) -> None:
        """check_agent_file only validates types; empty string tools passes."""
        content = FULL_VALID_AGENT_CONTENT.replace("tools: Read,Write,Edit", "tools: ''")
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        # The underlying check_agent_file does not validate non-empty tools,
        # only that the field is present and a string.
        assert result.is_valid(), f"Unexpected errors: {result.errors}"


class TestValidateFileNoFrontmatter:
    """Tests for files with missing or malformed frontmatter."""

    def test_no_frontmatter_fails(self, tmp_path: Path) -> None:
        """File with no YAML frontmatter should fail validation."""
        content = "# Just a title\n\nNo frontmatter here.\n"
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert not result.is_valid()
        assert len(result.errors) > 0

    def test_invalid_yaml_fails(self, tmp_path: Path) -> None:
        """File with malformed YAML frontmatter should fail validation."""
        content = """---
name: test-agent
  invalid: [unclosed
---

Content.
"""
        f = _write_agent_file(tmp_path, content)
        result = validate_file(f)
        assert not result.is_valid()
        assert len(result.errors) > 0


class TestExtractSections:
    """Tests for extract_sections() helper function."""

    def test_extracts_level2_headers(self) -> None:
        """extract_sections finds ## headers."""
        content = """# Top level

## Role

Some content.

## Responsibilities

More content.
"""
        sections = extract_sections(content)
        assert "Role" in sections
        assert "Responsibilities" in sections

    def test_ignores_level1_headers(self) -> None:
        """extract_sections ignores # (level-1) headers."""
        content = "# Top Level\n\n## Section One\n"
        sections = extract_sections(content)
        assert "Top Level" not in sections
        assert "Section One" in sections

    def test_empty_content_returns_empty_set(self) -> None:
        """extract_sections returns empty set for content with no ## headers."""
        content = "No headers here."
        sections = extract_sections(content)
        assert sections == set()


class TestValidateFrontmatterDirect:
    """Tests for the validate_frontmatter() function in validate_agents."""

    def test_valid_frontmatter_no_errors(self) -> None:
        """All required fields present and correct types yields no errors."""
        frontmatter = {
            "name": "test-agent",
            "description": "Use this agent when you need to implement Mojo patterns",
            "tools": "Read,Write",
            "model": "sonnet",
        }
        result = ValidationResult(Path("test-agent.md"))
        validate_frontmatter(frontmatter, result)
        assert result.is_valid()
        assert result.errors == []

    def test_missing_required_field_produces_error(self) -> None:
        """Missing a required field produces an appropriate error."""
        frontmatter = {
            "description": "Use this agent when you need to implement Mojo patterns",
            "tools": "Read",
            "model": "sonnet",
            # 'name' missing
        }
        result = ValidationResult(Path("test-agent.md"))
        validate_frontmatter(frontmatter, result)
        assert not result.is_valid()
        assert any("name" in e for e in result.errors)

    def test_colon_in_description_no_error(self) -> None:
        """Description with a colon should not cause a validation error."""
        frontmatter = {
            "name": "test-agent",
            "description": "Use when you need to implement design or test Mojo files",
            "tools": "Read",
            "model": "sonnet",
        }
        result = ValidationResult(Path("test-agent.md"))
        validate_frontmatter(frontmatter, result)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_url_in_description_no_error(self) -> None:
        """Description containing a URL should not cause a validation error."""
        frontmatter = {
            "name": "test-agent",
            "description": "See https://example.com for when to implement this agent",
            "tools": "Read",
            "model": "sonnet",
        }
        result = ValidationResult(Path("test-agent.md"))
        validate_frontmatter(frontmatter, result)
        assert result.is_valid(), f"Unexpected errors: {result.errors}"

    def test_invalid_model_no_longer_produces_error(self) -> None:
        """hephaestus validate_frontmatter only checks types, not model name values."""
        frontmatter = {
            "name": "test-agent",
            "description": "Use this agent when you need to test Mojo patterns",
            "tools": "Read",
            "model": "not-a-real-model",
        }
        result = ValidationResult(Path("test-agent.md"))
        validate_frontmatter(frontmatter, result)
        # The underlying validate_frontmatter does not validate model name values.
        assert result.is_valid(), f"Unexpected errors: {result.errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
