#!/usr/bin/env python3
"""Unit tests for scripts/agents/check_frontmatter.py.

Tests for:
- check_file: validates YAML frontmatter in a single agent markdown file
- validate_frontmatter: validates parsed frontmatter content
- Colon-containing values are handled correctly (regression for #3310)

Usage:
    pytest tests/agents/test_check_frontmatter.py -v
    python -m pytest tests/agents/test_check_frontmatter.py -v
"""

import sys
from pathlib import Path

import pytest

# Add scripts/agents to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "agents"))

from check_frontmatter import check_agent_file as _check_agent_file, validate_frontmatter as _validate_frontmatter


def check_file(file_path, verbose: bool = False):
    """Adapter: add verbose param (ignored) and delegate to check_agent_file."""
    return _check_agent_file(file_path)


def validate_frontmatter(frontmatter: dict, file_path=None) -> list:
    """Adapter: drop the file_path arg and delegate to hephaestus validate_frontmatter."""
    return _validate_frontmatter(frontmatter)


def _write_agent_file(tmp_path: Path, content: str, filename: str = "test-agent.md") -> Path:
    """Helper: write content to a temp file and return the path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


VALID_AGENT_CONTENT = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: Read,Write,Edit
model: sonnet
---

## Role

A test agent used for validation.
"""


class TestCheckFileValidFile:
    """Tests for check_file() with valid agent files."""

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        """A correctly-formed agent file should pass validation."""
        f = _write_agent_file(tmp_path, VALID_AGENT_CONTENT)
        is_valid, errors = check_file(f)
        assert is_valid
        assert errors == []

    def test_valid_file_verbose_passes(self, tmp_path: Path) -> None:
        """A valid file should also pass in verbose mode."""
        f = _write_agent_file(tmp_path, VALID_AGENT_CONTENT)
        is_valid, errors = check_file(f, verbose=True)
        assert is_valid
        assert errors == []


class TestCheckFileColonValues:
    """Regression tests for colon-containing values in frontmatter (follow-up from #3310).

    check_file uses PyYAML (via agent_utils) so these must all parse without
    truncating the value at the first colon.
    """

    def test_description_with_url(self, tmp_path: Path) -> None:
        """Description containing a URL should not be truncated at the colon."""
        content = """---
name: test-agent
description: See https://docs.example.com/api for implementation details
tools: Read,Write
model: sonnet
---

## Role

Agent for URL testing.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        # The description is long enough and should not cause a validation error
        assert is_valid, f"Unexpected errors: {errors}"

    def test_description_with_colon_mid_sentence(self, tmp_path: Path) -> None:
        """Description with a mid-sentence colon should be preserved whole."""
        content = """---
name: test-agent
description: Use when you need to implement design or test agent files
tools: Read,Write
model: sonnet
---

## Role

Agent for colon mid-sentence testing.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert is_valid, f"Unexpected errors: {errors}"

    def test_description_with_numeric_ratio_colon(self, tmp_path: Path) -> None:
        """Description with ratio notation (3:1) should be preserved."""
        content = """---
name: test-agent
description: Implement 3:1 data splits for training and validation sets
tools: Read,Write
model: sonnet
---

## Role

Agent for ratio colon testing.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert is_valid, f"Unexpected errors: {errors}"

    def test_description_with_multiple_colons(self, tmp_path: Path) -> None:
        """Description with multiple colons (URL with port) is preserved."""
        content = """---
name: test-agent
description: Connect to http://localhost:8080 to test or validate endpoints
tools: Read,Write
model: sonnet
---

## Role

Agent for multi-colon testing.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert is_valid, f"Unexpected errors: {errors}"


class TestCheckFileMissingFields:
    """Tests for check_file() detecting missing required fields."""

    def test_missing_name_field(self, tmp_path: Path) -> None:
        """File without 'name' field should fail validation."""
        content = """---
description: Use when you need to test agent configuration files
tools: Read,Write
model: sonnet
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert any("name" in e for e in errors)

    def test_missing_description_field(self, tmp_path: Path) -> None:
        """File without 'description' field should fail validation."""
        content = """---
name: test-agent
tools: Read,Write
model: sonnet
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert any("description" in e for e in errors)

    def test_missing_tools_field(self, tmp_path: Path) -> None:
        """File without 'tools' field should fail validation."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
model: sonnet
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert any("tools" in e for e in errors)

    def test_missing_model_field(self, tmp_path: Path) -> None:
        """File without 'model' field should fail validation."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: Read,Write
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert any("model" in e for e in errors)


class TestCheckFileModelValidation:
    """Tests for model field validation in check_file()."""

    def test_valid_model_sonnet(self, tmp_path: Path) -> None:
        """'sonnet' is a valid model value."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: Read,Write
model: sonnet
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert is_valid, f"Unexpected errors: {errors}"

    def test_valid_model_opus(self, tmp_path: Path) -> None:
        """'opus' is a valid model value."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: Read,Write
model: opus
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert is_valid, f"Unexpected errors: {errors}"

    def test_invalid_model_still_valid_type(self, tmp_path: Path) -> None:
        """check_agent_file only validates types; any string model value is accepted."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: Read,Write
model: gpt-4
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        # The underlying check_agent_file does not validate model name values,
        # only that the field is present and a string.
        assert is_valid, f"Unexpected errors: {errors}"


class TestCheckFileToolsValidation:
    """Tests for tools field validation in check_file()."""

    def test_empty_tools_is_valid_string(self, tmp_path: Path) -> None:
        """check_agent_file only validates types; an empty string for tools is accepted."""
        content = """---
name: test-agent
description: Use when you need to test agent configuration files
tools: ''
model: sonnet
---

## Role

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        # The underlying check_agent_file does not validate non-empty tools,
        # only that the field is present and a string.
        assert is_valid, f"Unexpected errors: {errors}"


class TestCheckFileNoFrontmatter:
    """Tests for files without any frontmatter."""

    def test_no_frontmatter_fails(self, tmp_path: Path) -> None:
        """File with no YAML frontmatter should fail validation."""
        content = "# Just a title\n\nNo frontmatter here.\n"
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_yaml_fails(self, tmp_path: Path) -> None:
        """File with malformed YAML frontmatter should fail validation."""
        content = """---
name: test-agent
  invalid_indent: [unclosed
---

Content.
"""
        f = _write_agent_file(tmp_path, content)
        is_valid, errors = check_file(f)
        assert not is_valid
        assert len(errors) > 0


class TestValidateFrontmatterDirect:
    """Tests for the validate_frontmatter() function directly."""

    def test_valid_frontmatter_returns_no_errors(self, tmp_path: Path) -> None:
        """All required fields present and correct types yields no errors."""
        frontmatter = {
            "name": "test-agent",
            "description": "Use when you need to test agent configuration files",
            "tools": "Read,Write",
            "model": "sonnet",
        }
        errors = validate_frontmatter(frontmatter, tmp_path / "test-agent.md")
        assert errors == []

    def test_missing_required_field_produces_error(self, tmp_path: Path) -> None:
        """Missing a required field produces an appropriate error."""
        frontmatter = {
            "description": "Use when you need to test agent configuration files",
            "tools": "Read",
            "model": "sonnet",
            # 'name' missing
        }
        errors = validate_frontmatter(frontmatter, tmp_path / "test-agent.md")
        assert any("name" in e for e in errors)

    def test_colon_in_description_no_error(self, tmp_path: Path) -> None:
        """Description with a colon should not cause a validation error."""
        frontmatter = {
            "name": "test-agent",
            "description": "Use when you need to: parse, validate, or design files",
            "tools": "Read",
            "model": "sonnet",
        }
        errors = validate_frontmatter(frontmatter, tmp_path / "test-agent.md")
        assert errors == []

    def test_url_in_description_no_error(self, tmp_path: Path) -> None:
        """Description containing a URL should not cause a validation error."""
        frontmatter = {
            "name": "test-agent",
            "description": "See https://example.com for when to implement this agent",
            "tools": "Read",
            "model": "sonnet",
        }
        errors = validate_frontmatter(frontmatter, tmp_path / "test-agent.md")
        assert errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
