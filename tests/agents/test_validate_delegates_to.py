#!/usr/bin/env python3
"""Tests for delegates_to file-existence validation in validate_configs.py.

Verifies that AgentConfigValidator catches stale delegates_to references
(agent names that have no corresponding .md file in .claude/agents/).

Usage:
    pytest tests/agents/test_validate_delegates_to.py -v
    python -m pytest tests/agents/test_validate_delegates_to.py -v
"""

import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Add tests/agents to path so we can import validate_configs directly
sys.path.insert(0, str(Path(__file__).parent))

from validate_configs import AgentConfigValidator, ValidationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_FRONTMATTER_TEMPLATE = """\
---
name: {name}
description: A test agent that does something useful when invoked
tools: Read,Write
model: sonnet
delegates_to: {delegates_to}
---
# Role
Test role.

# Responsibilities
Test responsibilities.

# Scope
Test scope.

# Delegation
Test delegation.

# Workflow
Test workflow.

# Examples
Test examples.
"""


def _make_agent_file(directory: Path, name: str, delegates_to: str = "[]") -> Path:
    """Write a minimal valid agent .md file with the given delegates_to value."""
    content = _VALID_FRONTMATTER_TEMPLATE.format(name=name, delegates_to=delegates_to)
    path = directory / f"{name}.md"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Tests: existing_agents set initialisation
# ---------------------------------------------------------------------------


class TestExistingAgentsSetInit:
    """AgentConfigValidator should build existing_agents from the directory."""

    def test_existing_agents_populated_from_directory(self) -> None:
        """existing_agents contains stems of all .md files in agents_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "alpha-engineer")
            _make_agent_file(agents_dir, "beta-specialist")

            validator = AgentConfigValidator(agents_dir)

            assert "alpha-engineer" in validator.existing_agents
            assert "beta-specialist" in validator.existing_agents

    def test_existing_agents_empty_for_nonexistent_dir(self) -> None:
        """existing_agents is empty when directory does not exist."""
        validator = AgentConfigValidator(Path("/nonexistent/agents"))
        assert validator.existing_agents == set()

    def test_existing_agents_empty_for_empty_dir(self) -> None:
        """existing_agents is empty when directory has no .md files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = AgentConfigValidator(Path(tmpdir))
            assert validator.existing_agents == set()

    def test_existing_agents_ignores_non_md_files(self) -> None:
        """existing_agents only counts .md files, not .txt or other files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            (agents_dir / "some-agent.txt").write_text("not an agent")
            (agents_dir / "readme.rst").write_text("also not an agent")
            _make_agent_file(agents_dir, "real-engineer")

            validator = AgentConfigValidator(agents_dir)

            assert validator.existing_agents == {"real-engineer"}


# ---------------------------------------------------------------------------
# Tests: delegates_to validation — valid cases
# ---------------------------------------------------------------------------


class TestDelegatesToValidationValid:
    """delegates_to pointing at real agents should not produce errors."""

    def test_empty_delegates_to_is_valid(self) -> None:
        """delegates_to: [] produces no errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "solo-engineer", delegates_to="[]")

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            errors = [e for r in results for e in r.errors if "delegates_to" in e]
            assert errors == []

    def test_single_valid_reference(self) -> None:
        """delegates_to: [existing-agent] produces no errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "target-engineer")
            _make_agent_file(agents_dir, "source-specialist", delegates_to="[target-engineer]")

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            errors = [e for r in results for e in r.errors if "delegates_to" in e]
            assert errors == []

    def test_multiple_valid_references(self) -> None:
        """delegates_to with several real agent names produces no errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "alpha-engineer")
            _make_agent_file(agents_dir, "beta-engineer")
            _make_agent_file(agents_dir, "gamma-engineer")
            _make_agent_file(
                agents_dir,
                "delegating-specialist",
                delegates_to="[alpha-engineer, beta-engineer, gamma-engineer]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            errors = [e for r in results for e in r.errors if "delegates_to" in e]
            assert errors == []

    def test_delegates_to_with_spaces_around_commas(self) -> None:
        """Spaces around commas in YAML inline list are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "worker-engineer")
            _make_agent_file(
                agents_dir,
                "manager-specialist",
                delegates_to="[ worker-engineer ]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            errors = [e for r in results for e in r.errors if "delegates_to" in e]
            assert errors == []


# ---------------------------------------------------------------------------
# Tests: delegates_to validation — invalid cases (stale references)
# ---------------------------------------------------------------------------


class TestDelegatesToValidationInvalid:
    """Stale delegates_to references should produce hard errors."""

    def test_single_missing_reference_produces_error(self) -> None:
        """delegates_to with one non-existent agent name is an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "real-engineer")
            _make_agent_file(
                agents_dir,
                "source-specialist",
                delegates_to="[ghost-engineer]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            source_result = next(r for r in results if r.file_path.stem == "source-specialist")
            assert not source_result.is_valid
            assert any("ghost-engineer" in e for e in source_result.errors)

    def test_error_message_contains_agent_name(self) -> None:
        """Error message includes the missing agent name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(
                agents_dir,
                "broken-specialist",
                delegates_to="[junior-documentation-engineer]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            broken_result = next(r for r in results if r.file_path.stem == "broken-specialist")
            assert any("junior-documentation-engineer" in e for e in broken_result.errors)

    def test_error_message_contains_expected_filename(self) -> None:
        """Error message mentions the expected .md file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(
                agents_dir,
                "broken-specialist",
                delegates_to="[missing-agent]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            broken_result = next(r for r in results if r.file_path.stem == "broken-specialist")
            assert any("missing-agent.md" in e for e in broken_result.errors)

    def test_mixed_valid_and_invalid_references(self) -> None:
        """Only missing references produce errors; valid ones do not."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(agents_dir, "existing-engineer")
            _make_agent_file(
                agents_dir,
                "mixed-specialist",
                delegates_to="[existing-engineer, ghost-engineer]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            mixed_result = next(r for r in results if r.file_path.stem == "mixed-specialist")
            assert not mixed_result.is_valid
            delegates_errors = [e for e in mixed_result.errors if "delegates_to" in e]
            assert len(delegates_errors) == 1
            assert "ghost-engineer" in delegates_errors[0]
            assert "existing-engineer" not in delegates_errors[0]

    def test_multiple_missing_references_each_produce_error(self) -> None:
        """Each missing agent in delegates_to produces a separate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(
                agents_dir,
                "broken-specialist",
                delegates_to="[ghost-a, ghost-b, ghost-c]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            broken_result = next(r for r in results if r.file_path.stem == "broken-specialist")
            delegates_errors = [e for e in broken_result.errors if "delegates_to" in e]
            assert len(delegates_errors) == 3
            assert any("ghost-a" in e for e in delegates_errors)
            assert any("ghost-b" in e for e in delegates_errors)
            assert any("ghost-c" in e for e in delegates_errors)

    def test_stale_reference_marks_file_invalid(self) -> None:
        """A file with a stale delegates_to reference has is_valid == False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            _make_agent_file(
                agents_dir,
                "broken-specialist",
                delegates_to="[senior-implementation-engineer]",
            )

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            broken_result = next(r for r in results if r.file_path.stem == "broken-specialist")
            assert not broken_result.is_valid


# ---------------------------------------------------------------------------
# Tests: delegates_to format validation
# ---------------------------------------------------------------------------


class TestDelegatesToFormatValidation:
    """Malformed delegates_to values should produce errors."""

    def test_non_list_value_produces_error(self) -> None:
        """delegates_to value not wrapped in [] produces an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            content = """\
---
name: bad-format-specialist
description: A test agent that does something useful when invoked
tools: Read,Write
model: sonnet
delegates_to: some-engineer
---
# Role
Test.
"""
            (agents_dir / "bad-format-specialist.md").write_text(content)

            validator = AgentConfigValidator(agents_dir)
            results = validator.validate_all()

            bad_result = next(r for r in results if r.file_path.stem == "bad-format-specialist")
            assert not bad_result.is_valid
            assert any("delegates_to" in e for e in bad_result.errors)


# ---------------------------------------------------------------------------
# Tests: integration with real agent directory
# ---------------------------------------------------------------------------


class TestDelegatesToRealAgents:
    """Integration test: all agents in .claude/agents/ have valid delegates_to."""

    def _find_agents_dir(self) -> Path:
        """Locate the .claude/agents/ directory relative to this file."""
        base = Path(__file__).parent
        for candidate in [base, *base.parents]:
            agents_dir = candidate / ".claude" / "agents"
            if agents_dir.is_dir():
                return agents_dir
        pytest.skip(".claude/agents/ not found — skipping integration test")

    def test_all_agent_delegates_to_references_exist(self) -> None:
        """Every delegates_to name in .claude/agents/ maps to a real .md file."""
        agents_dir = self._find_agents_dir()
        validator = AgentConfigValidator(agents_dir)
        results = validator.validate_all()

        stale_errors: List[str] = []
        for result in results:
            for error in result.errors:
                if "delegates_to references non-existent agent" in error:
                    stale_errors.append(f"{result.file_path.name}: {error}")

        assert stale_errors == [], (
            "Stale delegates_to references found:\n" + "\n".join(f"  {e}" for e in stale_errors)
        )
