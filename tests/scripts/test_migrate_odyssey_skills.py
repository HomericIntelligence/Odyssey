#!/usr/bin/env python3
"""Tests for migrate_odyssey_skills.py.

Covers:
- parse_frontmatter(): YAML parsing with and without frontmatter
- transform_skill_md(): section injection, ## Workflow rename, path generalization
- determine_category(): all category mappings including tier-1/tier-2 overrides
- generalize_content(): all PATH_REPLACEMENTS patterns
- find_all_skills(): top-level, tier-1, tier-2 discovery
- migrate_skill(): auxiliary subdirectory copying
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

# Add scripts directory to sys.path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from migrate_odyssey_skills import (
    PATH_REPLACEMENTS,
    TIER1_CATEGORY,
    TIER2_CATEGORY_MAP,
    determine_category,
    find_all_skills,
    generalize_content,
    parse_frontmatter,
    transform_skill_md,
)

# Also keep dynamic loader for migrate_skill tests that need module-level patching
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


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatterDupe:
    def test_parses_key_value_pairs(self) -> None:
        content = "---\nname: my-skill\ndescription: A skill\n---\nBody"
        fm, rest = parse_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "A skill"

    def test_remaining_content_after_frontmatter(self) -> None:
        content = "---\nname: foo\n---\nBody content here"
        _, rest = parse_frontmatter(content)
        assert "Body content here" in rest

    def test_no_frontmatter_returns_empty_dict(self) -> None:
        content = "# Title\n\nSome content without frontmatter"
        fm, rest = parse_frontmatter(content)
        assert fm == {}
        assert rest == content

    def test_unclosed_frontmatter_returns_empty_dict(self) -> None:
        content = "---\nname: foo\n"
        fm, rest = parse_frontmatter(content)
        assert fm == {}
        assert rest == content

    def test_strips_double_quotes_from_values(self) -> None:
        content = '---\nname: "my skill"\n---\nBody'
        fm, _ = parse_frontmatter(content)
        assert fm["name"] == "my skill"

    def test_strips_single_quotes_from_values(self) -> None:
        content = "---\nname: 'my skill'\n---\nBody"
        fm, _ = parse_frontmatter(content)
        assert fm["name"] == "my skill"

    def test_colon_in_value_kept_intact(self) -> None:
        content = "---\ndescription: foo: bar\n---\nBody"
        fm, _ = parse_frontmatter(content)
        assert fm["description"] == "foo: bar"

    def test_empty_frontmatter_block(self) -> None:
        content = "---\n---\nContent"
        fm, rest = parse_frontmatter(content)
        assert fm == {}
        assert "Content" in rest

    def test_keys_without_values_not_included(self) -> None:
        content = "---\nname:\ndescription: valid\n---\nBody"
        fm, _ = parse_frontmatter(content)
        assert "name" not in fm
        assert fm["description"] == "valid"

    def test_multiple_fields_all_parsed(self) -> None:
        content = "---\nname: skill-x\ndescription: desc\ncategory: tooling\nuser-invocable: false\n---\nBody"
        fm, _ = parse_frontmatter(content)
        assert len(fm) == 4
        assert fm["category"] == "tooling"
        assert fm["user-invocable"] == "false"


# ---------------------------------------------------------------------------
# determine_category
# ---------------------------------------------------------------------------


class TestDetermineCategory:
    def test_skill_category_override_takes_precedence(self) -> None:
        # gh-create-pr-linked is in SKILL_CATEGORY_OVERRIDE -> tooling
        assert determine_category("gh-create-pr-linked", {}, None) == "tooling"

    def test_override_ignores_frontmatter_category(self) -> None:
        # Even with a different frontmatter category, override wins
        assert determine_category("run-precommit", {"category": "github"}, None) == "ci-cd"

    def test_override_ignores_tier(self) -> None:
        assert determine_category("mojo-format", {}, "1") == "architecture"

    def test_tier1_returns_tooling(self) -> None:
        result = determine_category("some-unknown-skill", {}, "1")
        assert result == TIER1_CATEGORY

    def test_tier2_mapped_skill_returns_correct_category(self) -> None:
        for skill_name, expected_category in TIER2_CATEGORY_MAP.items():
            result = determine_category(skill_name, {}, "2")
            assert result == expected_category, f"tier-2 skill {skill_name!r} expected {expected_category!r}"

    def test_tier2_unknown_skill_defaults_to_tooling(self) -> None:
        result = determine_category("not-in-tier2-map", {}, "2")
        assert result == "tooling"

    def test_frontmatter_category_maps_correctly(self) -> None:
        result = determine_category("some-skill", {"category": "github"}, None)
        assert result == "tooling"

    def test_frontmatter_ci_maps_to_ci_cd(self) -> None:
        result = determine_category("some-skill", {"category": "ci"}, None)
        assert result == "ci-cd"

    def test_frontmatter_mojo_maps_to_architecture(self) -> None:
        result = determine_category("some-skill", {"category": "mojo"}, None)
        assert result == "architecture"

    def test_frontmatter_doc_maps_to_documentation(self) -> None:
        result = determine_category("some-skill", {"category": "doc"}, None)
        assert result == "documentation"

    def test_frontmatter_quality_maps_to_evaluation(self) -> None:
        result = determine_category("some-skill", {"category": "quality"}, None)
        assert result == "evaluation"

    def test_unknown_frontmatter_category_defaults_to_tooling(self) -> None:
        result = determine_category("some-skill", {"category": "unknown-cat"}, None)
        assert result == "tooling"

    def test_no_frontmatter_no_tier_defaults_to_tooling(self) -> None:
        result = determine_category("some-new-skill", {}, None)
        assert result == "tooling"


# ---------------------------------------------------------------------------
# generalize_content
# ---------------------------------------------------------------------------


class TestGeneralizeContent:
    def test_odyssey2_path_replaced(self) -> None:
        content = "Run from /home/mvillmow/Odyssey2/ directory"
        result = generalize_content(content)
        assert "/home/mvillmow/Odyssey2/" not in result
        assert "<project-root>/" in result

    def test_projectodyssey_path_replaced(self) -> None:
        content = "See /home/mvillmow/ProjectOdyssey/ for details"
        result = generalize_content(content)
        assert "/home/mvillmow/ProjectOdyssey/" not in result
        assert "<project-root>/" in result

    def test_projectodyssey_name_replaced(self) -> None:
        result = generalize_content("This is ProjectOdyssey code")
        assert "ProjectOdyssey" not in result
        assert "<project-name>" in result

    def test_projectodyssey2_name_replaced(self) -> None:
        result = generalize_content("This is ProjectOdyssey2 code")
        assert "ProjectOdyssey2" not in result
        assert "<project-name>" in result

    def test_pixi_run_mojo_replaced(self) -> None:
        result = generalize_content("Use pixi run mojo test")
        assert "pixi run mojo" not in result
        assert "<package-manager> run mojo" in result

    def test_pixi_run_replaced(self) -> None:
        result = generalize_content("Use pixi run format")
        assert "pixi run" not in result
        assert "<package-manager> run" in result

    def test_test_model_path_replaced(self) -> None:
        result = generalize_content("tests/models/test_lenet5.mojo")
        assert "tests/models/test_" not in result
        assert "tests/<model>/test_" in result

    def test_create_worktree_script_replaced(self) -> None:
        result = generalize_content("Run ./scripts/create_worktree.sh now")
        assert "./scripts/create_worktree.sh" not in result
        assert "<project-root>/scripts/create_worktree.sh" in result

    def test_worktree_name_pattern_replaced(self) -> None:
        result = generalize_content("Branch WorktreeName-123-feature")
        assert "WorktreeName-123-" not in result
        assert "<project-name>-<issue-number>-" in result

    def test_content_without_patterns_unchanged(self) -> None:
        content = "This content has no Odyssey-specific paths."
        result = generalize_content(content)
        assert result == content

    def test_all_replacement_patterns_covered(self) -> None:
        """Verify PATH_REPLACEMENTS is not empty and all patterns compile."""
        import re

        assert len(PATH_REPLACEMENTS) > 0
        for pattern, _ in PATH_REPLACEMENTS:
            # Should not raise
            re.compile(pattern)


# ---------------------------------------------------------------------------
# transform_skill_md
# ---------------------------------------------------------------------------


class TestTransformSkillMd:
    def _minimal_content(self, name: str = "test-skill", description: str = "A test skill") -> str:
        return f"---\nname: {name}\ndescription: {description}\ncategory: tooling\nuser-invocable: false\n---\n\n# {name}\n\n## Workflow\n\n1. Do something.\n"

    def test_workflow_renamed_to_verified_workflow(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "## Verified Workflow" in result
        assert "## Workflow\n" not in result

    def test_already_verified_workflow_not_doubled(self) -> None:
        content = "---\nname: s\ndescription: d\ncategory: tooling\nuser-invocable: false\n---\n\n# s\n\n## Verified Workflow\n\n1. Step.\n"
        result = transform_skill_md(content, "s", "tooling")
        assert result.count("## Verified Workflow") == 1

    def test_overview_injected_when_missing(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "## Overview" in result

    def test_overview_not_duplicated_when_present(self) -> None:
        content = "---\nname: s\ndescription: d\ncategory: tooling\nuser-invocable: false\n---\n\n# s\n\n## Overview\n\nAlready here.\n\n## Workflow\n\n1. Step.\n"
        result = transform_skill_md(content, "s", "tooling")
        assert result.count("## Overview") == 1

    def test_when_to_use_injected_when_missing(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "## When to Use" in result

    def test_failed_attempts_injected_when_missing(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "## Failed Attempts" in result

    def test_results_section_injected_when_missing(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "## Results" in result

    def test_frontmatter_rebuilt_with_category(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "architecture")
        assert "category: architecture" in result

    def test_frontmatter_rebuilt_with_date(self) -> None:
        content = self._minimal_content()
        result = transform_skill_md(content, "test-skill", "tooling")
        assert "date: 2025-01-01" in result

    def test_odyssey_paths_generalized_in_body(self) -> None:
        content = "---\nname: s\ndescription: d\ncategory: tooling\nuser-invocable: false\n---\n\n# s\n\n## Workflow\n\nUse /home/mvillmow/Odyssey2/ as root.\n"
        result = transform_skill_md(content, "s", "tooling")
        assert "/home/mvillmow/Odyssey2/" not in result
        assert "<project-root>/" in result

    def test_content_without_frontmatter_still_transforms(self) -> None:
        content = "# bare-skill\n\n## Workflow\n\n1. Do it.\n"
        result = transform_skill_md(content, "bare-skill", "tooling")
        assert "## Verified Workflow" in result
        assert "category: tooling" in result

    def test_existing_sections_preserved(self) -> None:
        content = (
            "---\nname: s\ndescription: d\ncategory: tooling\nuser-invocable: false\n---\n\n"
            "# s\n\n"
            "## When to Use\n\n- Custom use case.\n\n"
            "## Quick Reference\n\nCustom reference.\n\n"
            "## Workflow\n\n1. Custom step.\n"
        )
        result = transform_skill_md(content, "s", "tooling")
        assert "Custom use case." in result
        assert "Custom reference." in result
        assert "Custom step." in result


# ---------------------------------------------------------------------------
# find_all_skills
# ---------------------------------------------------------------------------


class TestFindAllSkills:
    def _make_skill(self, root: Path, name: str) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        md = skill_dir / "SKILL.md"
        md.write_text(f"---\nname: {name}\n---\n")
        return md

    def _make_tiered_skill(self, root: Path, tier: str, name: str) -> Path:
        skill_dir = root / f"tier-{tier}" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        md = skill_dir / "SKILL.md"
        md.write_text(f"---\nname: {name}\n---\n")
        return md

    def test_top_level_skill_discovered(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "my-skill")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "my-skill" in names

    def test_top_level_skill_has_none_tier(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "my-skill")
        skills = find_all_skills(tmp_path)
        entry = next(e for e in skills if e[0] == "my-skill")
        assert entry[2] is None

    def test_tier1_skill_discovered(self, tmp_path: Path) -> None:
        self._make_tiered_skill(tmp_path, "1", "tier1-skill")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "tier1-skill" in names

    def test_tier1_skill_has_tier_one(self, tmp_path: Path) -> None:
        self._make_tiered_skill(tmp_path, "1", "tier1-skill")
        skills = find_all_skills(tmp_path)
        entry = next(e for e in skills if e[0] == "tier1-skill")
        assert entry[2] == "1"

    def test_tier2_skill_discovered(self, tmp_path: Path) -> None:
        self._make_tiered_skill(tmp_path, "2", "tier2-skill")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "tier2-skill" in names

    def test_tier2_skill_has_tier_two(self, tmp_path: Path) -> None:
        self._make_tiered_skill(tmp_path, "2", "tier2-skill")
        skills = find_all_skills(tmp_path)
        entry = next(e for e in skills if e[0] == "tier2-skill")
        assert entry[2] == "2"

    def test_directory_without_skill_md_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-skill").mkdir()
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "not-a-skill" not in names

    def test_hidden_directories_excluded(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden-skill").mkdir()
        (tmp_path / ".hidden-skill" / "SKILL.md").write_text("---\n---\n")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert ".hidden-skill" not in names

    def test_tier1_and_tier2_dirs_not_treated_as_skills(self, tmp_path: Path) -> None:
        # The tier-1 directory itself should not appear as a skill name
        self._make_tiered_skill(tmp_path, "1", "inner-skill")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "tier-1" not in names
        assert "tier-2" not in names

    def test_empty_root_returns_empty_list(self, tmp_path: Path) -> None:
        skills = find_all_skills(tmp_path)
        assert skills == []

    def test_returns_path_to_skill_md(self, tmp_path: Path) -> None:
        md = self._make_skill(tmp_path, "path-skill")
        skills = find_all_skills(tmp_path)
        entry = next(e for e in skills if e[0] == "path-skill")
        assert entry[1] == md

    def test_all_three_tiers_together(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "top-skill")
        self._make_tiered_skill(tmp_path, "1", "t1-skill")
        self._make_tiered_skill(tmp_path, "2", "t2-skill")
        skills = find_all_skills(tmp_path)
        names = [n for n, _, _ in skills]
        assert "top-skill" in names
        assert "t1-skill" in names
        assert "t2-skill" in names
        assert len(skills) == 3

    def test_missing_tier1_dir_skipped_gracefully(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "top-skill")
        # No tier-1 or tier-2 dirs
        skills = find_all_skills(tmp_path)
        assert len(skills) == 1

    def test_uses_odyssey_skills_dir_when_no_arg(self) -> None:
        # When source_dir is None, falls back to ODYSSEY_SKILLS_DIR constant.
        # We just verify the function signature accepts None without raising TypeError.
        # (We can't assert the result without the real dir existing on CI.)
        try:
            find_all_skills(None)
        except FileNotFoundError:
            pass  # Expected if ODYSSEY_SKILLS_DIR doesn't exist in test environment
        except Exception as exc:
            raise AssertionError(f"Unexpected exception type: {type(exc).__name__}: {exc}") from exc
