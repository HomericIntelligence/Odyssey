#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for agent testing.

This module provides common fixtures, helpers, and configuration for all agent tests.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import yaml

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hephaestus.utils import get_repo_root


# ============================================================================
# Path Configuration
# ============================================================================


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Get repository root directory."""
    return get_repo_root()


@pytest.fixture(scope="session")
def agents_dir(repo_root: Path) -> Path:
    """Get .claude/agents directory."""
    return repo_root / ".claude" / "agents"


@pytest.fixture(scope="session")
def skills_dir(repo_root: Path) -> Path:
    """Get .claude/skills directory."""
    return repo_root / ".claude" / "skills"


@pytest.fixture(scope="session")
def docs_agents_dir(repo_root: Path) -> Path:
    """Get agents/ documentation directory."""
    return repo_root / "agents"


# ============================================================================
# Agent File Discovery
# ============================================================================


@pytest.fixture(scope="session")
def all_agent_files(agents_dir: Path) -> List[Path]:
    """Discover all agent configuration files."""
    if not agents_dir.exists():
        return []
    return sorted(agents_dir.glob("*.md"))


@pytest.fixture(scope="session")
def all_skill_files(skills_dir: Path) -> List[Path]:
    """Discover all skill configuration files."""
    if not skills_dir.exists():
        return []
    return sorted(skills_dir.glob("**/SKILL.md"))


@pytest.fixture(scope="session")
def all_doc_files(docs_agents_dir: Path) -> List[Path]:
    """Discover all documentation markdown files."""
    if not docs_agents_dir.exists():
        return []
    return sorted(docs_agents_dir.glob("**/*.md"))


# ============================================================================
# YAML Frontmatter Parsing
# ============================================================================


def parse_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown file content

    Returns:
        Tuple of (frontmatter_dict, body_content)
        Returns (None, content) if no frontmatter found
    """
    # Match YAML frontmatter pattern: ---\n...\n---
    pattern = r"^---\s*\n(.*?\n)---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None, content

    try:
        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2)
        return frontmatter, body
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}")


@pytest.fixture
def parse_agent_file(agents_dir: Path):
    """
    Factory fixture to parse agent configuration files.

    Returns:
        Function that takes a filename and returns (frontmatter, body).
    """

    def _parse(filename: str) -> Tuple[Optional[Dict], str]:
        filepath = agents_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Agent file not found: {filepath}")

        content = filepath.read_text()
        return parse_frontmatter(content)

    return _parse


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_frontmatter_keys(
    frontmatter: Dict,
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None,
) -> List[str]:
    """
    Validate that frontmatter contains required keys.

    Args:
        frontmatter: Parsed YAML frontmatter
        required_keys: List of required keys
        optional_keys: List of optional keys (for checking unknowns)

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check required keys
    for key in required_keys:
        if key not in frontmatter:
            errors.append(f"Missing required key: {key}")

    # Check for unknown keys
    if optional_keys is not None:
        allowed_keys = set(required_keys) | set(optional_keys)
        unknown_keys = set(frontmatter.keys()) - allowed_keys
        if unknown_keys:
            errors.append(f"Unknown keys: {', '.join(sorted(unknown_keys))}")

    return errors


def extract_links(content: str) -> List[str]:
    """
    Extract all markdown links from content.

    Args:
        content: Markdown content

    Returns:
        List of link targets (URLs or paths).
    """
    # Match [text](link) pattern
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(pattern, content)
    return [link for _, link in matches]


def extract_skill_references(content: str) -> List[str]:
    """
    Extract skill references from content.

    Pattern: [`skill_name`](../skills/.../SKILL.md)

    Args:
        content: Markdown content

    Returns:
        List of skill reference paths
    """
    # Match skill reference pattern
    pattern = r"\[`([^`]+)`\]\(([^)]+/SKILL\.md)\)"
    matches = re.findall(pattern, content)
    return [link for _, link in matches]


def extract_agent_references(content: str) -> List[str]:
    """
    Extract agent references from content.

    Pattern: [Agent Name](./agent-file.md) or [Agent Name](../agents/agent-file.md)

    Args:
        content: Markdown content

    Returns:
        List of agent reference paths
    """
    # Match agent reference pattern
    pattern = r"\[([^\]]+)\]\((\./[^)]+\.md|\.\.\/agents\/[^)]+\.md)\)"
    matches = re.findall(pattern, content)
    return [link for _, link in matches]


def resolve_relative_path(base_path: Path, relative_path: str) -> Path:
    """
    Resolve a relative path from a base path.

    Args:
        base_path: Base file path
        relative_path: Relative path string

    Returns:
        Resolved absolute path
    """
    # Handle relative paths
    if relative_path.startswith("./"):
        return (base_path.parent / relative_path[2:]).resolve()
    elif relative_path.startswith("../"):
        return (base_path.parent / relative_path).resolve()
    else:
        return Path(relative_path).resolve()


@pytest.fixture
def validate_link_exists(repo_root: Path):
    """
    Factory fixture to validate that a link target exists.

    Returns:
        Function that takes (source_file, link) and returns bool
    """

    def _validate(source_file: Path, link: str) -> bool:
        # Skip external URLs
        if link.startswith(("http://", "https://", "mailto:")):
            return True

        # Skip anchors
        if link.startswith("#"):
            return True

        # Resolve relative path
        target = resolve_relative_path(source_file, link)
        return target.exists()

    return _validate


# ============================================================================
# Test Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "documentation: mark test as documentation test")
    config.addinivalue_line("markers", "scripts: mark test as script validation test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ============================================================================
# Test Data Helpers
# ============================================================================


@pytest.fixture
def sample_valid_frontmatter() -> Dict:
    """Provide sample valid agent frontmatter."""
    return {
        "name": "test-agent",
        "description": "Test agent for validation",
        "tools": "Read,Write,Edit,Bash,Grep,Glob",
        "model": "sonnet",
    }


@pytest.fixture
def sample_valid_agent_content(sample_valid_frontmatter: Dict) -> str:
    """Provide sample valid agent file content."""
    frontmatter_yaml = yaml.dump(sample_valid_frontmatter)
    return f"""---
{frontmatter_yaml}---

# Test Agent

## Role
Test agent for validation.

## Scope
- Test scope

## Responsibilities
- Test responsibility

## Workflow Phase
**Test**

## Success Criteria
- Test passes
"""


# ============================================================================
# Parametrization Helpers
# ============================================================================


def generate_agent_test_ids(agent_files: List[Path]) -> List[str]:
    """Generate test IDs from agent file paths."""
    return [f.stem for f in agent_files]


def generate_doc_test_ids(doc_files: List[Path]) -> List[str]:
    """Generate test IDs from documentation file paths."""
    return [str(f.relative_to(f.parents[2])) for f in doc_files]
