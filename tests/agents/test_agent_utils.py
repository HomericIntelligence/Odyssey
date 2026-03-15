#!/usr/bin/env python3
"""Unit tests for agent_utils module.

Tests for:
- AgentInfo class initialization and methods
- Level inference from agent names
- Tools list parsing
- Frontmatter extraction and parsing
- Agent file discovery

Usage:
    pytest tests/agents/test_agent_utils.py -v
    python -m pytest tests/agents/test_agent_utils.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/agents to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "agents"))

from agent_utils import (
    AgentInfo,
    extract_frontmatter_raw,
    extract_frontmatter_parsed,
    extract_frontmatter_full,
    find_agent_files,
    load_agent,
    load_all_agents,
    validate_frontmatter_structure,
)


class TestAgentInfoInitialization:
    """Tests for AgentInfo class initialization."""

    def test_init_basic_frontmatter(self):
        """Test basic AgentInfo initialization from frontmatter dict."""
        frontmatter = {
            "name": "test-agent",
            "description": "A test agent",
            "tools": "Read,Write,Edit",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        assert agent.name == "test-agent"
        assert agent.description == "A test agent"
        assert agent.tools == "Read,Write,Edit"
        assert agent.model == "sonnet"
        assert agent.file_path == Path("test.md")

    def test_init_with_missing_fields(self):
        """Test AgentInfo with missing optional fields."""
        frontmatter = {
            "name": "minimal-agent",
            "model": "haiku",
        }
        agent = AgentInfo(Path("minimal.md"), frontmatter)

        assert agent.name == "minimal-agent"
        assert agent.description == "No description"
        assert agent.tools == ""
        assert agent.model == "haiku"

    def test_init_with_all_defaults(self):
        """Test AgentInfo with empty frontmatter."""
        frontmatter = {}
        agent = AgentInfo(Path("empty.md"), frontmatter)

        assert agent.name == "unknown"
        assert agent.description == "No description"
        assert agent.tools == ""
        assert agent.model == "unknown"

    def test_init_with_explicit_level(self):
        """Test AgentInfo with explicit level in frontmatter."""
        frontmatter = {
            "name": "test-agent",
            "description": "Test",
            "tools": "",
            "model": "sonnet",
            "level": 2,
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 2


class TestLevelInference:
    """Tests for agent level inference."""

    def test_level_inference_chief_architect(self):
        """Test level 0 inference for chief-architect."""
        frontmatter = {
            "name": "chief-architect",
            "description": "",
            "tools": "",
            "model": "opus",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 0

    def test_level_inference_orchestrator(self):
        """Test level 1 inference for orchestrators."""
        frontmatter = {
            "name": "foundation-orchestrator",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 1

    def test_level_inference_design_agent(self):
        """Test level 2 inference for design agents."""
        frontmatter = {
            "name": "architecture-design",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 2

    def test_level_inference_specialist(self):
        """Test level 3 inference for specialists."""
        frontmatter = {
            "name": "test-specialist",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 3

    def test_level_inference_senior_engineer(self):
        """Test level 4 inference for senior engineers."""
        frontmatter = {
            "name": "senior-implementation-engineer",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 4

    def test_level_inference_junior_engineer(self):
        """Test level 5 inference for junior engineers."""
        frontmatter = {
            "name": "junior-implementation-engineer",
            "description": "",
            "tools": "",
            "model": "haiku",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 5

    def test_level_inference_generic_engineer(self):
        """Test level inference for generic 'engineer' agents defaults to 4."""
        frontmatter = {
            "name": "implementation-engineer",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 4

    def test_level_inference_unknown_defaults_to_3(self):
        """Test that unknown agent types default to level 3."""
        frontmatter = {
            "name": "mysterious-agent",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 3

    def test_level_explicit_overrides_inference(self):
        """Test that explicit level in frontmatter overrides inference."""
        frontmatter = {
            "name": "chief-architect",
            "description": "",
            "tools": "",
            "model": "opus",
            "level": 5,
        }
        agent = AgentInfo(Path("test.md"), frontmatter)
        assert agent.level == 5  # Explicit level overrides inference


class TestToolsListParsing:
    """Tests for tools list parsing."""

    def test_get_tools_list_with_spaces(self):
        """Test get_tools_list() with spaces around commas."""
        frontmatter = {
            "name": "test",
            "description": "",
            "tools": "Read, Write, Edit, Bash",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        tools = agent.get_tools_list()
        assert tools == ["Read", "Write", "Edit", "Bash"]

    def test_get_tools_list_no_spaces(self):
        """Test get_tools_list() without spaces."""
        frontmatter = {
            "name": "test",
            "description": "",
            "tools": "Read,Write,Edit",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        tools = agent.get_tools_list()
        assert tools == ["Read", "Write", "Edit"]

    def test_get_tools_list_empty(self):
        """Test get_tools_list() with empty tools."""
        frontmatter = {
            "name": "test",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        tools = agent.get_tools_list()
        assert tools == []

    def test_get_tools_list_single_tool(self):
        """Test get_tools_list() with single tool."""
        frontmatter = {
            "name": "test",
            "description": "",
            "tools": "Read",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        tools = agent.get_tools_list()
        assert tools == ["Read"]


class TestAgentInfoRepr:
    """Tests for AgentInfo string representation."""

    def test_repr_format(self):
        """Test __repr__ includes level and name."""
        frontmatter = {
            "name": "test-agent",
            "description": "",
            "tools": "",
            "model": "sonnet",
        }
        agent = AgentInfo(Path("test.md"), frontmatter)

        repr_str = repr(agent)
        assert "AgentInfo" in repr_str
        assert "test-agent" in repr_str
        assert "level=" in repr_str


class TestFrontmatterExtraction:
    """Tests for frontmatter extraction functions."""

    def test_extract_frontmatter_raw_valid(self):
        """Test extract_frontmatter_raw with valid frontmatter."""
        content = """---
name: test-agent
description: A test agent
tools: Read,Write
model: sonnet
---
# Content here"""

        result = extract_frontmatter_raw(content)
        assert result is not None
        assert "name: test-agent" in result
        assert "model: sonnet" in result

    def test_extract_frontmatter_raw_no_frontmatter(self):
        """Test extract_frontmatter_raw with no frontmatter."""
        content = "# No frontmatter\nJust content"

        result = extract_frontmatter_raw(content)
        assert result is None

    def test_extract_frontmatter_parsed_valid(self):
        """Test extract_frontmatter_parsed returns dict and text."""
        content = """---
name: test-agent
description: A test agent
tools: Read,Write
model: sonnet
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is not None
        frontmatter_text, parsed = result
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test-agent"
        assert parsed["model"] == "sonnet"

    def test_extract_frontmatter_parsed_invalid_yaml(self):
        """Test extract_frontmatter_parsed with invalid YAML."""
        content = """---
name: test-agent
  invalid yaml: [
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is None

    def test_extract_frontmatter_full(self):
        """Test extract_frontmatter_full returns dict, text, and line numbers."""
        content = """---
name: test
---
Content"""

        result = extract_frontmatter_full(content)
        assert result is not None
        frontmatter_text, parsed, start_line, end_line = result
        assert isinstance(parsed, dict)
        assert start_line == 1
        assert end_line > start_line


class TestAgentFileDiscovery:
    """Tests for agent file discovery."""

    def test_find_agent_files_returns_markdown_files(self):
        """Test find_agent_files finds markdown files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "agent1.md").write_text("# Agent 1")
            (tmppath / "agent2.md").write_text("# Agent 2")
            (tmppath / "readme.txt").write_text("Not a markdown")

            files = find_agent_files(tmppath)
            assert len(files) == 2
            assert all(f.suffix == ".md" for f in files)

    def test_find_agent_files_sorted_order(self):
        """Test find_agent_files returns sorted results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files in non-sorted order
            (tmppath / "z-agent.md").write_text("# Z")
            (tmppath / "a-agent.md").write_text("# A")
            (tmppath / "m-agent.md").write_text("# M")

            files = find_agent_files(tmppath)
            names = [f.name for f in files]
            assert names == ["a-agent.md", "m-agent.md", "z-agent.md"]

    def test_find_agent_files_empty_directory(self):
        """Test find_agent_files with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            files = find_agent_files(tmppath)
            assert files == []


class TestLoadAgent:
    """Tests for agent loading."""

    def test_load_agent_valid_file(self):
        """Test load_agent with valid agent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            agent_file = tmppath / "test-agent.md"
            agent_file.write_text(
                """---
name: test-agent
description: A test agent
tools: Read,Write
model: sonnet
---
# Agent Content"""
            )

            agent = load_agent(agent_file)
            assert agent is not None
            assert agent.name == "test-agent"
            assert agent.description == "A test agent"

    def test_load_agent_invalid_yaml(self):
        """Test load_agent with invalid YAML frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            agent_file = tmppath / "bad-agent.md"
            agent_file.write_text(
                """---
name: test-agent
  invalid: [
---
# Content"""
            )

            agent = load_agent(agent_file)
            assert agent is None

    def test_load_agent_no_frontmatter(self):
        """Test load_agent with no frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            agent_file = tmppath / "no-fm.md"
            agent_file.write_text("# Just content\nNo frontmatter")

            agent = load_agent(agent_file)
            assert agent is None

    def test_load_agent_missing_file(self):
        """Test load_agent with non-existent file."""
        agent = load_agent(Path("/nonexistent/agent.md"))
        assert agent is None


class TestLoadAllAgents:
    """Tests for loading all agents from a directory."""

    def test_load_all_agents_multiple_files(self):
        """Test load_all_agents with multiple valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create multiple agent files
            for i in range(3):
                agent_file = tmppath / f"agent{i}.md"
                agent_file.write_text(
                    f"""---
name: agent-{i}
description: Test agent {i}
tools: Read,Write
model: sonnet
---
# Content"""
                )

            agents = load_all_agents(tmppath)
            assert len(agents) == 3
            names = sorted([a.name for a in agents])
            assert names == ["agent-0", "agent-1", "agent-2"]

    def test_load_all_agents_skips_invalid(self):
        """Test load_all_agents skips invalid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Valid file
            valid_file = tmppath / "valid.md"
            valid_file.write_text(
                """---
name: valid-agent
description: Valid
tools: Read
model: sonnet
---
# Content"""
            )

            # Invalid file
            invalid_file = tmppath / "invalid.md"
            invalid_file.write_text("# No frontmatter")

            agents = load_all_agents(tmppath)
            assert len(agents) == 1
            assert agents[0].name == "valid-agent"


class TestValidateFrontmatterStructure:
    """Tests for frontmatter validation."""

    def test_validate_required_fields_present(self):
        """Test validation with all required fields present."""
        frontmatter = {
            "name": "test",
            "description": "Test agent",
            "tools": "Read,Write",
            "model": "sonnet",
        }
        errors = validate_frontmatter_structure(frontmatter)
        assert len(errors) == 0

    def test_validate_missing_required_field(self):
        """Test validation detects missing required field."""
        frontmatter = {
            "name": "test",
            "description": "Test agent",
            "model": "sonnet",
            # Missing 'tools'
        }
        errors = validate_frontmatter_structure(frontmatter)
        assert len(errors) > 0
        assert any("tools" in error for error in errors)

    def test_validate_wrong_field_type(self):
        """Test validation detects wrong field type."""
        frontmatter = {
            "name": "test",
            "description": "Test agent",
            "tools": "Read,Write",
            "model": 123,  # Should be string
        }
        errors = validate_frontmatter_structure(frontmatter)
        assert len(errors) > 0
        assert any("model" in error for error in errors)

    def test_validate_optional_field_correct_type(self):
        """Test validation accepts optional field with correct type."""
        frontmatter = {
            "name": "test",
            "description": "Test agent",
            "tools": "Read,Write",
            "model": "sonnet",
            "level": 3,
        }
        errors = validate_frontmatter_structure(frontmatter)
        assert len(errors) == 0

    def test_validate_optional_field_wrong_type(self):
        """Test validation detects optional field with wrong type."""
        frontmatter = {
            "name": "test",
            "description": "Test agent",
            "tools": "Read,Write",
            "model": "sonnet",
            "level": "three",  # Should be int
        }
        errors = validate_frontmatter_structure(frontmatter)
        assert len(errors) > 0
        assert any("level" in error for error in errors)

    def test_validate_custom_required_fields(self):
        """Test validation with custom required fields."""
        frontmatter = {
            "name": "test",
            "custom_field": "value",
        }
        custom_required = {"custom_field": str}
        errors = validate_frontmatter_structure(frontmatter, required_fields=custom_required, optional_fields={})
        assert len(errors) == 0


class TestFrontmatterColonValues:
    """Regression tests for colon-containing values in frontmatter (follow-up from #3310).

    These tests verify that agent_utils correctly handles YAML values that
    contain colons (URLs, ratio notation, mid-sentence colons) without truncation.
    """

    def test_extract_frontmatter_parsed_url_in_description(self):
        """Description with a URL should not be truncated at the colon."""
        content = """---
name: test-agent
description: See https://docs.example.com/api for details
tools: Read,Write
model: sonnet
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is not None
        _, parsed = result
        assert parsed["description"] == "See https://docs.example.com/api for details"

    def test_extract_frontmatter_parsed_colon_in_quoted_value(self):
        """Description with a colon inside a quoted YAML string should be preserved whole."""
        content = """---
name: test-agent
description: "Use when you need to: parse, validate, or check files"
tools: Read,Write
model: sonnet
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is not None
        _, parsed = result
        assert parsed["description"] == "Use when you need to: parse, validate, or check files"

    def test_extract_frontmatter_parsed_numeric_ratio_colon(self):
        """Description with numeric ratio notation (3:1) should be preserved."""
        content = """---
name: test-agent
description: Handles ratio 3:1 splits for training and validation
tools: Read,Write
model: sonnet
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is not None
        _, parsed = result
        assert parsed["description"] == "Handles ratio 3:1 splits for training and validation"

    def test_extract_frontmatter_parsed_multiple_colons(self):
        """Value with multiple colons (e.g. URL with port) should be preserved."""
        content = """---
name: test-agent
description: Connect to http://localhost:8080/api for testing
tools: Read,Write
model: sonnet
---
# Content"""

        result = extract_frontmatter_parsed(content)
        assert result is not None
        _, parsed = result
        assert parsed["description"] == "Connect to http://localhost:8080/api for testing"

    def test_extract_frontmatter_full_url_in_description(self):
        """extract_frontmatter_full also preserves colon-containing descriptions."""
        content = """---
name: test-agent
description: See https://example.com for more info
tools: Read
model: haiku
---
# Content"""

        result = extract_frontmatter_full(content)
        assert result is not None
        _, parsed, _, _ = result
        assert parsed["description"] == "See https://example.com for more info"

    def test_extract_frontmatter_raw_colon_value_unaffected(self):
        """extract_frontmatter_raw returns raw text regardless of colon content."""
        content = """---
name: test-agent
description: See https://example.com for details
tools: Read
model: sonnet
---
# Content"""

        raw = extract_frontmatter_raw(content)
        assert raw is not None
        assert "https://example.com" in raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
