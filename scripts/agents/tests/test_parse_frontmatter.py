#!/usr/bin/env python3
"""
Tests for frontmatter parsing in agent utilities.

Validates that YAML frontmatter parsing correctly handles edge cases,
including colon-containing values and special characters, following
the pattern established in #3310.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import agent_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_utils import (
    extract_frontmatter_raw,
    extract_frontmatter_with_lines,
    extract_frontmatter_parsed,
    extract_frontmatter_full,
    AgentInfo,
)


class TestExtractFrontmatterRaw:
    """Test raw frontmatter extraction."""

    def test_extract_valid_frontmatter(self):
        """Test extracting valid frontmatter."""
        content = """---
name: test-agent
description: A test agent
---
# Content"""
        result = extract_frontmatter_raw(content)
        assert result is not None
        assert "name: test-agent" in result
        assert "description: A test agent" in result

    def test_extract_frontmatter_with_colons_in_value(self):
        """Test extraction handles colons in field values correctly (#3310)."""
        content = """---
name: test-agent
description: "Create PR linked to issue: #123"
tools: "gh-create-pr-linked, gh-check-ci-status"
---
# Content"""
        result = extract_frontmatter_raw(content)
        assert result is not None
        assert 'description: "Create PR linked to issue: #123"' in result

    def test_extract_no_frontmatter(self):
        """Test extraction returns None when no frontmatter present."""
        content = "# Content without frontmatter"
        result = extract_frontmatter_raw(content)
        assert result is None

    def test_extract_incomplete_frontmatter(self):
        """Test extraction returns None for incomplete frontmatter."""
        content = """---
name: test-agent
# Missing closing ---"""
        result = extract_frontmatter_raw(content)
        assert result is None


class TestExtractFrontmatterWithLines:
    """Test frontmatter extraction with line tracking."""

    def test_extract_with_line_numbers(self):
        """Test line number tracking."""
        content = """---
name: test-agent
description: Test
---
# Content"""
        result = extract_frontmatter_with_lines(content)
        assert result is not None
        frontmatter, start_line, end_line = result
        assert start_line == 1
        assert end_line == 4
        assert "name: test-agent" in frontmatter

    def test_extract_multiline_values(self):
        """Test line tracking with multiline YAML values."""
        content = """---
name: test-agent
description: |
  Multi-line
  description
---
# Content"""
        result = extract_frontmatter_with_lines(content)
        assert result is not None
        frontmatter, start_line, end_line = result
        assert "Multi-line" in frontmatter


class TestExtractFrontmatterParsed:
    """Test frontmatter extraction with YAML parsing."""

    def test_parse_valid_yaml(self):
        """Test parsing valid YAML frontmatter."""
        content = """---
name: test-agent
description: Test description
model: sonnet
tools: "tool1,tool2"
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert parsed["name"] == "test-agent"
        assert parsed["description"] == "Test description"

    def test_parse_with_colon_values(self):
        """Test parsing YAML with colon-containing values (#3310)."""
        content = """---
name: test-agent
description: "Create PR linked to issue: #123"
tools: "gh-create-pr-linked, gh-check-ci-status"
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        # Verify the colon is preserved in the parsed value
        assert parsed["description"] == "Create PR linked to issue: #123"
        assert ":" in parsed["description"]

    def test_parse_with_quoted_strings(self):
        """Test parsing YAML with quoted strings containing special characters."""
        content = '''---
name: test-agent
description: "A 'quoted' string with: colons and special chars!"
tools: "tool-1,tool-2,tool-3"
---
# Content'''
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert parsed["description"] == "A 'quoted' string with: colons and special chars!"

    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML returns None."""
        content = """---
name: test-agent
description: [invalid yaml here {{{
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is None

    def test_parse_no_frontmatter(self):
        """Test parsing returns None when no frontmatter present."""
        content = "# No frontmatter here"
        result = extract_frontmatter_parsed(content)
        assert result is None


class TestExtractFrontmatterFull:
    """Test full frontmatter extraction with parsing and line tracking."""

    def test_full_extraction(self):
        """Test full extraction with all metadata."""
        content = """---
name: test-agent
description: "Test: comprehensive frontmatter"
model: opus
---
# Content"""
        result = extract_frontmatter_full(content)
        assert result is not None
        frontmatter_text, parsed, start_line, end_line = result
        assert parsed["name"] == "test-agent"
        assert parsed["description"] == "Test: comprehensive frontmatter"
        assert start_line == 1
        assert end_line == 4

    def test_full_extraction_complex_values(self):
        """Test full extraction handles complex YAML values."""
        content = """---
name: pr-review-specialist
description: "Review PRs and provide feedback: issues, patterns, suggestions"
model: sonnet
tools: "gh-get-review-comments, gh-reply-review-comment, review-pr-changes"
level: 3
---
# Content"""
        result = extract_frontmatter_full(content)
        assert result is not None
        frontmatter_text, parsed, start_line, end_line = result
        assert parsed["description"] == "Review PRs and provide feedback: issues, patterns, suggestions"
        assert ":" in parsed["description"]
        assert parsed["level"] == 3


class TestAgentInfo:
    """Test AgentInfo class initialization."""

    def test_agent_info_from_frontmatter(self):
        """Test AgentInfo initialization."""
        frontmatter = {
            "name": "test-agent",
            "description": "A test agent",
            "tools": "tool1,tool2",
            "model": "sonnet",
            "level": 3,
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.name == "test-agent"
        assert agent.description == "A test agent"
        assert agent.tools == "tool1,tool2"
        assert agent.model == "sonnet"
        assert agent.level == 3

    def test_agent_info_tools_list(self):
        """Test AgentInfo.get_tools_list() parsing."""
        frontmatter = {
            "name": "test-agent",
            "description": "Test",
            "tools": "tool1, tool2, tool3",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        tools = agent.get_tools_list()
        assert tools == ["tool1", "tool2", "tool3"]

    def test_agent_info_with_colon_description(self):
        """Test AgentInfo handles colon-containing descriptions (#3310)."""
        frontmatter = {
            "name": "pr-reviewer",
            "description": "Review PRs and flag issues: security, quality, scope",
            "tools": "gh-review-pr,gh-reply-review-comment",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.description == "Review PRs and flag issues: security, quality, scope"
        assert ":" in agent.description

    def test_agent_info_level_inference(self):
        """Test AgentInfo level inference from name."""
        test_cases = [
            ("chief-architect", 0),
            ("section-orchestrator", 1),
            ("design-agent", 2),
            ("component-specialist", 3),
            ("senior-engineer", 4),
            ("junior-engineer", 5),
        ]
        for name, expected_level in test_cases:
            frontmatter = {
                "name": name,
                "description": "Test",
                "tools": "tool1",
                "model": "sonnet",
            }
            agent = AgentInfo(Path("test.md"), frontmatter)
            assert agent.level == expected_level


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_frontmatter(self):
        """Test handling of empty frontmatter."""
        content = """---

---
# Content"""
        result = extract_frontmatter_parsed(content)
        # Empty YAML parses to None, which fails the dict check
        assert result is None

    def test_frontmatter_with_special_characters(self):
        """Test frontmatter with special YAML characters."""
        content = """---
name: test-agent
description: "Special chars: !@#$%^&*() and <quotes>'single' and \"double\""
tools: "tool-1,tool-2"
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert "!@#$%^&*()" in parsed["description"]

    def test_unicode_in_frontmatter(self):
        """Test frontmatter with Unicode characters."""
        content = """---
name: test-agent
description: "Test with émojis 🚀 and spëcial çharacters"
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert "🚀" in parsed["description"]

    def test_yaml_boolean_values(self):
        """Test YAML boolean and numeric values."""
        content = """---
name: test-agent
description: "Test"
enabled: true
level: 3
threshold: 0.95
---
# Content"""
        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert parsed["enabled"] is True
        assert parsed["level"] == 3
        assert parsed["threshold"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
