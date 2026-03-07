#!/usr/bin/env python3
"""Tests for migrate_odyssey_skills.py - auxiliary subdirectory copying."""

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

# Import the module under test
import importlib.util

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "migrate_odyssey_skills.py"


def load_migrate_module():
    """Load the migrate_odyssey_skills module dynamically."""
    spec = importlib.util.spec_from_file_location("migrate_odyssey_skills", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migrate_module():
    """Provide the loaded migration module."""
    return load_migrate_module()


@pytest.fixture
def fake_skill_tree(tmp_path: Path):
    """Create a fake Odyssey2 skill directory structure."""
    odyssey_skills = tmp_path / "odyssey_skills"
    mnemosyne = tmp_path / "mnemosyne"
    mnemosyne_skills = mnemosyne / "skills"
    mnemosyne_skills.mkdir(parents=True)

    return odyssey_skills, mnemosyne, mnemosyne_skills


def make_skill_dir(skills_root: Path, skill_name: str, extra_subdirs: Optional[dict] = None) -> Path:
    """Create a fake skill directory with SKILL.md and optional subdirectories.

    Args:
        skills_root: Root .claude/skills directory.
        skill_name: Name of the skill.
        extra_subdirs: Dict of {subdir_name: {filename: content}} for extra subdirs.

    Returns:
        Path to the skill directory.
    """
    skill_dir = skills_root / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal SKILL.md with frontmatter
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"""---
name: {skill_name}
description: Test skill for {skill_name}
category: tooling
user-invocable: false
---

# {skill_name}

## When to Use

- Use when testing.

## Verified Workflow

1. Run the skill.

## Failed Attempts

| Attempt | Why Failed | Lesson Learned |
|---------|-----------|----------------|
| N/A | No recorded failures | N/A |

## Results & Parameters

See workflow above.
"""
    )

    if extra_subdirs:
        for subdir_name, files in extra_subdirs.items():
            subdir = skill_dir / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)
            for filename, content in files.items():
                (subdir / filename).write_text(content)

    return skill_dir


class TestParseFrontmatter:
    """Tests for parse_frontmatter() colon handling."""

    def test_plain_value(self, migrate_module) -> None:
        """Basic key: value with no colon in the value."""
        content = "---\nname: my-skill\ncategory: tooling\n---\n# Body"
        fm, remaining = migrate_module.parse_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["category"] == "tooling"
        assert "Body" in remaining

    def test_colon_in_quoted_value(self, migrate_module) -> None:
        """Regression: description with a colon inside quoted value must not be truncated."""
        content = '---\nname: gh-create-pr-linked\ndescription: "Create PR linked to issue: #123"\n---\n# Body'
        fm, _ = migrate_module.parse_frontmatter(content)
        assert fm["description"] == "Create PR linked to issue: #123"

    def test_colon_in_unquoted_value(self, migrate_module) -> None:
        """YAML bare string with colon (parsed as string by yaml.safe_load)."""
        content = "---\nname: my-skill\ndescription: Use when condition A is true\n---\n"
        fm, _ = migrate_module.parse_frontmatter(content)
        assert fm["description"] == "Use when condition A is true"

    def test_no_frontmatter(self, migrate_module) -> None:
        """Content without --- delimiter returns empty dict and full content."""
        content = "# Just a heading\nSome text"
        fm, remaining = migrate_module.parse_frontmatter(content)
        assert fm == {}
        assert remaining == content

    def test_unclosed_frontmatter(self, migrate_module) -> None:
        """Single --- with no closing delimiter returns empty dict."""
        content = "---\nname: my-skill\n"
        fm, remaining = migrate_module.parse_frontmatter(content)
        assert fm == {}

    def test_invalid_yaml_returns_empty_dict(self, migrate_module) -> None:
        """Malformed YAML in frontmatter returns {} without raising."""
        content = "---\n: invalid: yaml: [\n---\n# Body"
        fm, _ = migrate_module.parse_frontmatter(content)
        assert fm == {}

    def test_remaining_content_preserved(self, migrate_module) -> None:
        """Text after closing --- is returned as remaining content."""
        content = "---\nname: skill\n---\n\n# Title\n\nSome body text."
        _, remaining = migrate_module.parse_frontmatter(content)
        assert "Title" in remaining
        assert "Some body text." in remaining


class TestMigrateSkillAuxiliaryDirs:
    """Tests that auxiliary subdirectories are copied during migration."""

    def test_skill_with_only_skill_md(self, tmp_path: Path, migrate_module) -> None:
        """A skill with only SKILL.md migrates correctly (baseline behavior)."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(odyssey_skills, "simple-skill")

        skill_md = odyssey_skills / "simple-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="simple-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        plugin_dir = mnemosyne_skills / "tooling" / "simple-skill"
        assert (plugin_dir / ".claude-plugin" / "plugin.json").exists()
        assert (plugin_dir / "skills" / "simple-skill" / "SKILL.md").exists()

    def test_skill_with_scripts_subdir_copies_scripts(self, tmp_path: Path, migrate_module) -> None:
        """A skill with scripts/ subdir must have scripts/ copied into skills/<name>/scripts/."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "gh-create-pr-linked",
            extra_subdirs={"scripts": {"create_pr.sh": "#!/bin/bash\necho 'create pr'"}},
        )

        skill_md = odyssey_skills / "gh-create-pr-linked" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="gh-create-pr-linked",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        expected_scripts = (
            mnemosyne_skills / "tooling" / "gh-create-pr-linked" / "skills" / "gh-create-pr-linked" / "scripts"
        )
        assert expected_scripts.exists(), f"Expected scripts/ at {expected_scripts}"
        assert (expected_scripts / "create_pr.sh").exists()

    def test_skill_with_templates_subdir_copies_templates(self, tmp_path: Path, migrate_module) -> None:
        """A skill with templates/ subdir must have templates/ copied into skills/<name>/templates/."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "gh-create-pr-linked",
            extra_subdirs={"templates": {"pr_body.md": "## PR Template\n\nCloses #123"}},
        )

        skill_md = odyssey_skills / "gh-create-pr-linked" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="gh-create-pr-linked",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        expected_templates = (
            mnemosyne_skills / "tooling" / "gh-create-pr-linked" / "skills" / "gh-create-pr-linked" / "templates"
        )
        assert expected_templates.exists(), f"Expected templates/ at {expected_templates}"
        assert (expected_templates / "pr_body.md").exists()

    def test_skill_with_references_subdir_copies_to_plugin_root(self, tmp_path: Path, migrate_module) -> None:
        """A skill with references/ subdir must have it copied to <plugin>/references/."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "some-skill",
            extra_subdirs={"references": {"notes.md": "# Reference notes"}},
        )

        skill_md = odyssey_skills / "some-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="some-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        # references/ goes to plugin root, not inside skills/<name>/
        expected_refs = mnemosyne_skills / "tooling" / "some-skill" / "references"
        assert expected_refs.exists(), f"Expected references/ at {expected_refs}"
        assert (expected_refs / "notes.md").exists()
        # And NOT inside skills/<name>/references/
        wrong_location = mnemosyne_skills / "tooling" / "some-skill" / "skills" / "some-skill" / "references"
        assert not wrong_location.exists(), "references/ should not be inside skills/<name>/"

    def test_skill_with_multiple_subdirs_copies_all(self, tmp_path: Path, migrate_module) -> None:
        """A skill with scripts/, templates/, and references/ gets all copied correctly."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "gh-create-pr-linked",
            extra_subdirs={
                "scripts": {"create_pr.sh": "#!/bin/bash\necho hi"},
                "templates": {"pr_body.md": "# PR"},
                "references": {"notes.md": "# Notes"},
            },
        )

        skill_md = odyssey_skills / "gh-create-pr-linked" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="gh-create-pr-linked",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        plugin_dir = mnemosyne_skills / "tooling" / "gh-create-pr-linked"
        skill_inner_dir = plugin_dir / "skills" / "gh-create-pr-linked"

        # scripts/ inside skill inner dir
        assert (skill_inner_dir / "scripts" / "create_pr.sh").exists()
        # templates/ inside skill inner dir
        assert (skill_inner_dir / "templates" / "pr_body.md").exists()
        # references/ at plugin root
        assert (plugin_dir / "references" / "notes.md").exists()

    def test_skill_with_nested_scripts_content(self, tmp_path: Path, migrate_module) -> None:
        """Nested files within scripts/ are all copied."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "agent-run-orchestrator",
            extra_subdirs={
                "scripts": {
                    "run_orchestrator.sh": "#!/bin/bash\necho run",
                    "helper.sh": "#!/bin/bash\necho help",
                }
            },
        )

        skill_md = odyssey_skills / "agent-run-orchestrator" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="agent-run-orchestrator",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        scripts_dir = (
            mnemosyne_skills / "tooling" / "agent-run-orchestrator" / "skills" / "agent-run-orchestrator" / "scripts"
        )
        assert (scripts_dir / "run_orchestrator.sh").exists()
        assert (scripts_dir / "helper.sh").exists()

    def test_dry_run_does_not_copy_files(self, tmp_path: Path, migrate_module) -> None:
        """dry_run=True does not write any files."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "gh-create-pr-linked",
            extra_subdirs={"scripts": {"create_pr.sh": "#!/bin/bash\necho hi"}},
        )

        skill_md = odyssey_skills / "gh-create-pr-linked" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="gh-create-pr-linked",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=True,
            )

        assert result is True
        # Nothing should be created
        plugin_dir = mnemosyne_skills / "tooling" / "gh-create-pr-linked"
        assert not plugin_dir.exists(), "dry_run should not create any directories"

    def test_missing_skill_md_returns_false(self, tmp_path: Path, migrate_module) -> None:
        """migrate_skill() returns False when SKILL.md does not exist."""
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        nonexistent_skill_md = tmp_path / "nonexistent" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="nonexistent",
                source_skill_md=nonexistent_skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is False

    def test_hidden_directories_not_copied(self, tmp_path: Path, migrate_module) -> None:
        """Hidden subdirectories (starting with .) are not copied."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "some-skill",
            extra_subdirs={
                "scripts": {"run.sh": "#!/bin/bash"},
                ".hidden": {"secret.txt": "hidden content"},
            },
        )

        skill_md = odyssey_skills / "some-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="some-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        skill_inner_dir = mnemosyne_skills / "tooling" / "some-skill" / "skills" / "some-skill"
        # scripts/ should be copied
        assert (skill_inner_dir / "scripts" / "run.sh").exists()
        # .hidden/ should NOT be copied
        assert not (skill_inner_dir / ".hidden").exists()

    def test_custom_subdir_hooks_copied(self, tmp_path: Path, migrate_module) -> None:
        """A hooks/ subdir is treated like scripts/ and copied into skills/<name>/hooks/."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "phase-test-tdd",
            extra_subdirs={"hooks": {"pre_commit.sh": "#!/bin/bash\necho hook"}},
        )

        skill_md = odyssey_skills / "phase-test-tdd" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="phase-test-tdd",
                source_skill_md=skill_md,
                category="ci-cd",
                dry_run=False,
            )

        assert result is True
        expected_hooks = mnemosyne_skills / "ci-cd" / "phase-test-tdd" / "skills" / "phase-test-tdd" / "hooks"
        assert expected_hooks.exists()
        assert (expected_hooks / "pre_commit.sh").exists()

    def test_existing_destination_does_not_raise(self, tmp_path: Path, migrate_module) -> None:
        """If target subdir already exists, copytree with dirs_exist_ok=True does not error."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "some-skill",
            extra_subdirs={"scripts": {"run.sh": "#!/bin/bash\necho hi"}},
        )

        # Pre-create the destination scripts/ directory with a different file
        pre_existing = mnemosyne_skills / "tooling" / "some-skill" / "skills" / "some-skill" / "scripts"
        pre_existing.mkdir(parents=True)
        (pre_existing / "existing_file.sh").write_text("#!/bin/bash\necho existing")

        skill_md = odyssey_skills / "some-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="some-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        # Both files should exist (merge, not overwrite)
        assert (pre_existing / "run.sh").exists()
        assert (pre_existing / "existing_file.sh").exists()
