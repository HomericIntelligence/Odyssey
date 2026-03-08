#!/usr/bin/env python3
"""
Integration tests for agent configuration files.

Tests that all agent files exist, are readable, have valid YAML frontmatter,
and that all references (skills, agents) resolve correctly.
"""

from pathlib import Path
from typing import List

import pytest

# Import helper functions from conftest (not fixtures, which are auto-discovered)
from conftest import (
    extract_agent_references,
    extract_skill_references,
    parse_frontmatter,
    resolve_relative_path,
    validate_frontmatter_keys,
)


# ============================================================================
# Agent File Existence Tests
# ============================================================================


@pytest.mark.integration
class TestAgentFilesExist:
    """Test that all agent files exist and are readable."""

    def test_agents_directory_exists(self, agents_dir: Path):
        """Test that .claude/agents directory exists."""
        assert agents_dir.exists(), f"Agents directory not found: {agents_dir}"
        assert agents_dir.is_dir(), f"Agents path is not a directory: {agents_dir}"

    def test_agent_files_discovered(self, all_agent_files: List[Path]):
        """Test that at least one agent file is discovered."""
        assert len(all_agent_files) > 0, "No agent files found in .claude/agents/"

    def test_agent_file_readable(self, all_agent_files: List[Path]):
        """Test that each agent file is readable."""
        for agent_file in all_agent_files:
            assert agent_file.exists(), f"Agent file not found: {agent_file}"
            assert agent_file.is_file(), f"Agent path is not a file: {agent_file}"

            # Test readability
            try:
                content = agent_file.read_text()
                assert len(content) > 0, f"Agent file is empty: {agent_file}"
            except Exception as e:
                pytest.fail(f"Failed to read agent file {agent_file}: {e}")


# ============================================================================
# YAML Frontmatter Tests
# ============================================================================


@pytest.mark.integration
class TestAgentFrontmatter:
    """Test YAML frontmatter parsing and validation."""

    def test_parse_frontmatter_helper(self, sample_valid_agent_content: str):
        """Test the frontmatter parsing helper function."""
        frontmatter, body = parse_frontmatter(sample_valid_agent_content)

        assert frontmatter is not None, "Failed to parse valid frontmatter"
        assert "name" in frontmatter, "Frontmatter missing 'name' key"
        assert len(body) > 0, "Body content is empty"

    def test_parse_frontmatter_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "# Just a heading\n\nSome content"
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter is None, "Should return None for missing frontmatter"
        assert body == content, "Body should be original content"

    def test_agent_has_frontmatter(self, all_agent_files: List[Path]):
        """Test that each agent file has valid YAML frontmatter."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, body = parse_frontmatter(content)

            assert frontmatter is not None, f"Agent file missing frontmatter: {agent_file.name}"
            assert isinstance(frontmatter, dict), f"Frontmatter is not a dict: {agent_file.name}"
            assert len(body) > 0, f"Agent file has no body content: {agent_file.name}"

    def test_agent_required_frontmatter_keys(self, all_agent_files: List[Path]):
        """Test that agent frontmatter has all required keys."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, _ = parse_frontmatter(content)

            required_keys = ["name", "description", "tools", "model"]
            optional_keys: List[str] = []  # Add optional keys if needed

            errors = validate_frontmatter_keys(frontmatter, required_keys, optional_keys)

            assert not errors, f"Frontmatter validation failed for {agent_file.name}: {'; '.join(errors)}"

    def test_agent_name_matches_filename(self, all_agent_files: List[Path]):
        """Test that agent name in frontmatter matches filename."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, _ = parse_frontmatter(content)

            expected_name = agent_file.stem
            actual_name = frontmatter.get("name")

            assert actual_name == expected_name, f"Agent name '{actual_name}' doesn't match filename '{expected_name}'"

    def test_agent_description_not_empty(self, all_agent_files: List[Path]):
        """Test that agent description is not empty."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, _ = parse_frontmatter(content)

            description = frontmatter.get("description", "")
            assert len(description) > 0, f"Agent {agent_file.name} has empty description"
            assert len(description) > 10, f"Agent {agent_file.name} description too short: '{description}'"

    def test_agent_tools_valid(self, all_agent_files: List[Path]):
        """Test that agent tools list is valid."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, _ = parse_frontmatter(content)

            tools = frontmatter.get("tools", "")
            assert isinstance(tools, str), f"Tools must be a string in {agent_file.name}"

            # Valid tools from Claude Code
            valid_tools = {
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Grep",
                "Glob",
                "NotebookEdit",
                "WebFetch",
                "WebSearch",
                "AskUserQuestion",
            }

            if tools:  # Empty string is valid (no tools)
                tool_list = [t.strip() for t in tools.split(",")]
                invalid_tools = set(tool_list) - valid_tools

                assert not invalid_tools, f"Invalid tools in {agent_file.name}: {invalid_tools}"

    def test_agent_model_valid(self, all_agent_files: List[Path]):
        """Test that agent model is valid."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            frontmatter, _ = parse_frontmatter(content)

            model = frontmatter.get("model", "")
            valid_models = ["sonnet", "opus", "haiku"]  # Claude model types

            assert model in valid_models, f"Invalid model '{model}' in {agent_file.name}. Valid: {valid_models}"


# ============================================================================
# Skill Reference Tests
# ============================================================================


@pytest.mark.integration
class TestSkillReferences:
    """Test that all skill references resolve correctly."""

    def test_skills_directory_exists(self, skills_dir: Path):
        """Test that .claude/skills directory exists."""
        assert skills_dir.exists(), f"Skills directory not found: {skills_dir}"

    def test_skill_references_resolve(self, all_agent_files: List[Path], repo_root: Path):
        """Test that all skill references in agent files resolve to existing files."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            skill_refs = extract_skill_references(body)

            for skill_ref in skill_refs:
                # Resolve relative path from agent file
                skill_path = resolve_relative_path(agent_file, skill_ref)

                assert skill_path.exists(), f"Skill reference broken in {agent_file.name}: {skill_ref} -> {skill_path}"

    def test_skill_references_valid_format(self, all_agent_files: List[Path]):
        """Test that skill references follow the correct format."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            skill_refs = extract_skill_references(body)

            for skill_ref in skill_refs:
                # Should point to ../skills/.../SKILL.md
                assert skill_ref.startswith("../skills/"), (
                    f"Skill reference in {agent_file.name} should start with '../skills/': {skill_ref}"
                )
                assert skill_ref.endswith("/SKILL.md"), (
                    f"Skill reference in {agent_file.name} should end with '/SKILL.md': {skill_ref}"
                )


# ============================================================================
# Agent Reference Tests
# ============================================================================


@pytest.mark.integration
class TestAgentReferences:
    """Test that all internal agent references are valid."""

    def test_agent_references_resolve(self, all_agent_files: List[Path]):
        """Test that all agent references resolve to existing files."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            agent_refs = extract_agent_references(body)

            for agent_ref in agent_refs:
                # Skip external references
                if agent_ref.startswith("http"):
                    continue

                # Resolve relative path from agent file
                ref_path = resolve_relative_path(agent_file, agent_ref)

                assert ref_path.exists(), f"Agent reference broken in {agent_file.name}: {agent_ref} -> {ref_path}"

    def test_agent_references_valid_format(self, all_agent_files: List[Path]):
        """Test that agent references use correct path format."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            agent_refs = extract_agent_references(body)

            for agent_ref in agent_refs:
                # Should be either ./file.md or ../agents/file.md
                assert agent_ref.startswith(("./", "../agents/")), (
                    f"Agent reference in {agent_file.name} has invalid format: {agent_ref}"
                )
                assert agent_ref.endswith(".md"), (
                    f"Agent reference in {agent_file.name} should end with '.md': {agent_ref}"
                )


# ============================================================================
# Agent Structure Tests
# ============================================================================


@pytest.mark.integration
class TestAgentStructure:
    """Test that agent files follow the expected structure."""

    def test_agent_has_role_section(self, all_agent_files: List[Path]):
        """Test that agent file has a Role section."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            assert "## Role" in body, f"Agent {agent_file.name} missing '## Role' section"

    def test_agent_has_scope_section(self, all_agent_files: List[Path]):
        """Test that agent file has a Scope section."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            assert "## Scope" in body, f"Agent {agent_file.name} missing '## Scope' section"

    def test_agent_has_responsibilities_section(self, all_agent_files: List[Path]):
        """Test that agent file has a Responsibilities section."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            assert "## Responsibilities" in body, f"Agent {agent_file.name} missing '## Responsibilities' section"

    def test_agent_has_workflow_phase(self, all_agent_files: List[Path]):
        """Test that agent file specifies workflow phase."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            assert "## Workflow Phase" in body, f"Agent {agent_file.name} missing '## Workflow Phase' section"

    def test_agent_has_success_criteria(self, all_agent_files: List[Path]):
        """Test that agent file has success criteria."""
        for agent_file in all_agent_files:
            content = agent_file.read_text()
            _, body = parse_frontmatter(content)

            assert "## Success Criteria" in body, f"Agent {agent_file.name} missing '## Success Criteria' section"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_yaml_frontmatter(self):
        """Test parsing invalid YAML frontmatter."""
        content = """---
name: test
invalid: yaml: structure: here
---

# Body
"""
        with pytest.raises(ValueError, match="Invalid YAML"):
            parse_frontmatter(content)

    def test_malformed_frontmatter_missing_end(self):
        """Test parsing frontmatter with missing end marker."""
        content = """---
name: test
description: test

# Body without closing ---
"""
        frontmatter, _ = parse_frontmatter(content)
        # Should return None since frontmatter is malformed
        assert frontmatter is None

    def test_empty_file(self, tmp_path: Path):
        """Test handling empty agent file."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")

        content = empty_file.read_text()
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter is None
        assert body == ""

    def test_validate_frontmatter_keys_with_missing(self):
        """Test frontmatter validation with missing required keys."""
        frontmatter = {"name": "test"}
        required = ["name", "description", "tools"]

        errors = validate_frontmatter_keys(frontmatter, required)

        assert len(errors) == 2
        assert any("description" in err for err in errors)
        assert any("tools" in err for err in errors)

    def test_validate_frontmatter_keys_with_unknown(self):
        """Test frontmatter validation with unknown keys."""
        frontmatter = {"name": "test", "description": "test", "unknown_key": "value"}
        required = ["name", "description"]
        optional: list[str] = []

        errors = validate_frontmatter_keys(frontmatter, required, optional)

        assert len(errors) == 1
        assert "unknown_key" in errors[0]
