#!/usr/bin/env python3
"""Regression tests for extract_frontmatter() in ProjectMnemosyne migrate_to_skills.py.

Guards against the partition(':') bug that silently truncated YAML values
containing colons (e.g. description: "Create PR linked to issue: #123").
"""

import importlib.util
import os
from pathlib import Path

import pytest

# Locate the script relative to common build locations and the local checkout.
_CANDIDATES = [
    # Local ProjectMnemosyne checkout
    Path.home() / "ProjectMnemosyne" / "scripts" / "migrate_to_skills.py",
    # PID-scoped build directory (used by /advise and /retrospective skills)
    Path(__file__).parent.parent.parent / "build" / str(os.getpid()) / "ProjectMnemosyne" / "scripts" / "migrate_to_skills.py",
]

_SCRIPT_PATH: Path | None = next((p for p in _CANDIDATES if p.exists()), None)

pytestmark = pytest.mark.skipif(
    _SCRIPT_PATH is None,
    reason="ProjectMnemosyne/scripts/migrate_to_skills.py not found; skipping",
)


def _load_module():
    """Dynamically load migrate_to_skills without adding it to sys.modules."""
    assert _SCRIPT_PATH is not None
    spec = importlib.util.spec_from_file_location("migrate_to_skills", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def migrate_module():
    """Provide the loaded migrate_to_skills module."""
    return _load_module()


class TestExtractFrontmatter:
    """Regression tests for extract_frontmatter() colon handling."""

    def test_plain_value(self, migrate_module) -> None:
        """Basic key: value with no colon in the value."""
        content = "---\nname: my-skill\ncategory: tooling\n---\n# Body"
        fm = migrate_module.extract_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["category"] == "tooling"

    def test_colon_in_quoted_value(self, migrate_module) -> None:
        """Regression: description with a colon inside a quoted value must not be truncated."""
        content = '---\nname: gh-create-pr-linked\ndescription: "Create PR linked to issue: #123"\n---\n# Body'
        fm = migrate_module.extract_frontmatter(content)
        assert fm["description"] == "Create PR linked to issue: #123"

    def test_colon_in_unquoted_value(self, migrate_module) -> None:
        """YAML bare string with embedded colon is returned in full."""
        content = "---\nname: my-skill\ndescription: Use when condition A is true\n---\n"
        fm = migrate_module.extract_frontmatter(content)
        assert fm["description"] == "Use when condition A is true"

    def test_no_frontmatter(self, migrate_module) -> None:
        """Content without --- delimiter returns empty dict."""
        content = "# Just a heading\nSome text"
        fm = migrate_module.extract_frontmatter(content)
        assert fm == {}

    def test_unclosed_frontmatter(self, migrate_module) -> None:
        """Single --- with no closing delimiter returns empty dict."""
        content = "---\nname: my-skill\n"
        fm = migrate_module.extract_frontmatter(content)
        assert fm == {}

    def test_invalid_yaml_returns_empty_dict(self, migrate_module) -> None:
        """Malformed YAML in frontmatter returns {} without raising."""
        content = "---\n: invalid: yaml: [\n---\n# Body"
        fm = migrate_module.extract_frontmatter(content)
        assert fm == {}

    def test_multiple_colons_in_value(self, migrate_module) -> None:
        """Values with multiple colons are returned fully intact."""
        content = '---\ndescription: "Step 1: do this; Step 2: do that"\n---\n'
        fm = migrate_module.extract_frontmatter(content)
        assert fm["description"] == "Step 1: do this; Step 2: do that"
