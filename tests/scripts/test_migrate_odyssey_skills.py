#!/usr/bin/env python3
"""Tests for migrate_odyssey_skills.py - auxiliary subdirectory copying and path resolution."""

import importlib.util
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

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


def make_tiered_skill_dir(
    skills_root: Path,
    tier: str,
    skill_name: str,
    extra_subdirs: Optional[dict] = None,
) -> Path:
    """Create a fake tiered skill directory with SKILL.md and optional subdirectories.

    Args:
        skills_root: Root .claude/skills directory.
        tier: Tier string ("1" or "2").
        skill_name: Name of the skill.
        extra_subdirs: Dict of {subdir_name: {filename: content}} for extra subdirs.

    Returns:
        Path to the skill directory (skills_root/tier-{tier}/{skill_name}).
    """
    skill_dir = skills_root / f"tier-{tier}" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"""---
name: {skill_name}
description: Test tier-{tier} skill for {skill_name}
tier: "{tier}"
user-invocable: false
---

# {skill_name}

## When to Use

- Use when testing tier-{tier} skills.

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

    def test_skill_with_deeply_nested_scripts_subdir(self, tmp_path: Path, migrate_module) -> None:
        """scripts/utils/helper.sh (subdir inside scripts/) is preserved after migration."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        skill_dir = make_skill_dir(odyssey_skills, "agent-run-orchestrator")
        # Create a subdirectory *within* scripts/ to test multi-level nesting
        nested_dir = skill_dir / "scripts" / "utils"
        nested_dir.mkdir(parents=True)
        (nested_dir / "helper.sh").write_text("#!/bin/bash\necho helper")

        skill_md = odyssey_skills / "agent-run-orchestrator" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="agent-run-orchestrator",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        nested_helper = (
            mnemosyne_skills
            / "tooling"
            / "agent-run-orchestrator"
            / "skills"
            / "agent-run-orchestrator"
            / "scripts"
            / "utils"
            / "helper.sh"
        )
        assert nested_helper.exists(), f"Expected deeply nested helper at {nested_helper}"

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

    def test_warns_when_dest_subdir_already_exists(
        self, tmp_path: Path, migrate_module, capsys: pytest.CaptureFixture
    ) -> None:
        """Warning is printed to stderr when destination subdir pre-exists before copytree."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "some-skill",
            extra_subdirs={"scripts": {"run.sh": "#!/bin/bash\necho hi"}},
        )

        # Pre-create the destination scripts/ directory
        pre_existing = mnemosyne_skills / "tooling" / "some-skill" / "skills" / "some-skill" / "scripts"
        pre_existing.mkdir(parents=True)

        skill_md = odyssey_skills / "some-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="some-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=False,
            )

        assert result is True
        captured = capsys.readouterr()
        assert "WARNING: destination subdir already exists and will be merged:" in captured.err
        assert str(pre_existing) in captured.err

    def test_warns_when_dest_subdir_already_exists_dry_run(
        self, tmp_path: Path, migrate_module, capsys: pytest.CaptureFixture
    ) -> None:
        """Warning fires in dry-run mode too when destination subdir pre-exists."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_skill_dir(
            odyssey_skills,
            "some-skill",
            extra_subdirs={"scripts": {"run.sh": "#!/bin/bash\necho hi"}},
        )

        # Pre-create the destination scripts/ directory
        pre_existing = mnemosyne_skills / "tooling" / "some-skill" / "skills" / "some-skill" / "scripts"
        pre_existing.mkdir(parents=True)

        skill_md = odyssey_skills / "some-skill" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="some-skill",
                source_skill_md=skill_md,
                category="tooling",
                dry_run=True,
            )

        assert result is True
        captured = capsys.readouterr()
        assert "WARNING: destination subdir already exists and will be merged:" in captured.err
        assert str(pre_existing) in captured.err


class TestResolveMnemosyneDir:
    """Tests for resolve_mnemosyne_dir() path resolution priority."""

    def test_explicit_target_takes_priority(self, migrate_module, tmp_path: Path) -> None:
        """--target-dir value is used even when MNEMOSYNE_DIR env var is set."""
        env_path = str(tmp_path / "env_path")
        explicit_path = str(tmp_path / "explicit_path")
        with patch.dict("os.environ", {"MNEMOSYNE_DIR": env_path}):
            result = migrate_module.resolve_mnemosyne_dir(explicit_path)
        assert result == Path(explicit_path)

    def test_env_var_used_when_no_target(self, migrate_module, tmp_path: Path) -> None:
        """MNEMOSYNE_DIR env var is used when --target-dir is not provided."""
        env_path = str(tmp_path / "from_env")
        with patch.dict("os.environ", {"MNEMOSYNE_DIR": env_path}):
            result = migrate_module.resolve_mnemosyne_dir(None)
        assert result == Path(env_path)

    def test_default_used_when_neither_set(self, migrate_module) -> None:
        """Default /tmp/ProjectMnemosyne is used when neither arg nor env var is set."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "MNEMOSYNE_DIR"}
        with patch.dict("os.environ", env, clear=True):
            result = migrate_module.resolve_mnemosyne_dir(None)
        assert result == Path("/tmp/ProjectMnemosyne")  # nosec B108

    def test_empty_env_var_falls_back_to_default(self, migrate_module) -> None:
        """An unset MNEMOSYNE_DIR env var falls back to the default."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "MNEMOSYNE_DIR"}
        with patch.dict("os.environ", env, clear=True):
            result = migrate_module.resolve_mnemosyne_dir(None)
        assert result == migrate_module.DEFAULT_MNEMOSYNE_DIR


class TestSkillAlreadyExistsWithPath:
    """Tests for skill_already_exists() with explicit mnemosyne_skills_dir."""

    def test_returns_false_when_skills_dir_missing(self, migrate_module, tmp_path: Path) -> None:
        """Returns False (not an error) when the skills directory does not exist."""
        nonexistent = tmp_path / "no_such_dir" / "skills"
        assert migrate_module.skill_already_exists("my-skill", nonexistent) is False

    def test_returns_true_when_skill_present(self, migrate_module, tmp_path: Path) -> None:
        """Returns True when the skill exists under any category."""
        skills_dir = tmp_path / "skills"
        (skills_dir / "tooling" / "my-skill").mkdir(parents=True)
        assert migrate_module.skill_already_exists("my-skill", skills_dir) is True

    def test_returns_false_when_skill_absent(self, migrate_module, tmp_path: Path) -> None:
        """Returns False when the skill does not exist in any category."""
        skills_dir = tmp_path / "skills"
        (skills_dir / "tooling").mkdir(parents=True)
        assert migrate_module.skill_already_exists("absent-skill", skills_dir) is False


class TestMainErrorMessage:
    """Tests for improved error messages in main() when target dir is missing."""

    def test_missing_target_dir_prints_hint(
        self, migrate_module, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """main() prints --target-dir hint when target directory does not exist."""
        source_dir = tmp_path / "source_skills"
        source_dir.mkdir()
        missing_target = str(tmp_path / "nonexistent_mnemosyne")

        # Call main via sys.argv patch
        import sys

        orig_argv = sys.argv
        try:
            sys.argv = [
                "migrate_odyssey_skills.py",
                "--source-dir",
                str(source_dir),
                "--target-dir",
                missing_target,
            ]
            result = migrate_module.main()
        finally:
            sys.argv = orig_argv

        assert result == 1
        captured = capsys.readouterr()
        assert "--target-dir" in captured.err or "MNEMOSYNE_DIR" in captured.err


class TestMigrateSkillDescriptionWithColon:
    """End-to-end tests for migrate_skill() with descriptions containing colons.

    Regression tests for issue #3310 / #3931: Verify that descriptions with colons
    are preserved through the full migration pipeline and appear untruncated in both
    plugin.json and SKILL.md.
    """

    def test_migrate_skill_preserves_colon_in_description_in_plugin_json(self, tmp_path: Path, migrate_module) -> None:
        """End-to-end: description with colon appears fully in plugin.json."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        # Create skill with description containing a colon
        skill_dir = odyssey_skills / "gh-create-pr-linked"
        skill_dir.mkdir(parents=True)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: gh-create-pr-linked
description: "Create GitHub PR linked to issue: matches issue number from branch"
category: github
user-invocable: true
---

# Create GitHub PR

When to use this skill.
"""
        )

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="gh-create-pr-linked",
                source_skill_md=skill_md,
                category="github",
                dry_run=False,
            )

        assert result is True

        # Verify plugin.json contains the full description with colon
        plugin_json_path = mnemosyne_skills / "github" / "gh-create-pr-linked" / ".claude-plugin" / "plugin.json"
        assert plugin_json_path.exists()

        import json

        plugin_data = json.loads(plugin_json_path.read_text())
        assert plugin_data["description"] == ("Create GitHub PR linked to issue: matches issue number from branch")

    def test_migrate_skill_preserves_colon_in_description_in_skill_md(self, tmp_path: Path, migrate_module) -> None:
        """End-to-end: description with colon appears fully in migrated SKILL.md."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        # Create skill with description containing a colon
        skill_dir = odyssey_skills / "mojo-format"
        skill_dir.mkdir(parents=True)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: mojo-format
description: "Format Mojo code: uses pixi run mojo format"
category: mojo
user-invocable: false
---

# Format Mojo Code

Detailed information about the skill.
"""
        )

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="mojo-format",
                source_skill_md=skill_md,
                category="mojo",
                dry_run=False,
            )

        assert result is True

        # Verify migrated SKILL.md frontmatter contains the full description
        migrated_skill_md = mnemosyne_skills / "mojo" / "mojo-format" / "skills" / "mojo-format" / "SKILL.md"
        assert migrated_skill_md.exists()

        migrated_content = migrated_skill_md.read_text()
        # Verify description appears untruncated in the migrated file
        # (Note: The migrated SKILL.md should quote the description for valid YAML,
        # but the key point is the description is not truncated)
        assert "Format Mojo code: uses pixi run mojo format" in migrated_content
        assert migrated_content.count("Format Mojo code: uses pixi run mojo format") >= 2  # In frontmatter and Overview


# ---------------------------------------------------------------------------
# Parametrize tuples: (skill_name, tier, extra_subdirs_spec, expected_category)
# ---------------------------------------------------------------------------
_TIERED_AUXILIARY_CASES = [
    pytest.param(
        "run-tests",
        "1",
        {"scripts": {"run_tests.sh": "#!/bin/bash\necho 'running tests'"}},
        "tooling",
        id="tier1-run-tests-scripts",
    ),
    pytest.param(
        "lint-code",
        "1",
        {"templates": {"lint_config.yaml": "rules:\n  - no-unused-vars"}},
        "tooling",
        id="tier1-lint-code-templates",
    ),
    pytest.param(
        "generate-tests",
        "2",
        {"scripts": {"gen_tests.py": "#!/usr/bin/env python3\n# generate tests"}},
        "testing",
        id="tier2-generate-tests-scripts",
    ),
    pytest.param(
        "generate-docstrings",
        "2",
        {"templates": {"docstring.md": "# Docstring Template\n\nArgs:\n"}},
        "documentation",
        id="tier2-generate-docstrings-templates",
    ),
    pytest.param(
        "profile-code",
        "2",
        {"references": {"profiling_notes.md": "# Profiling Reference\n\nUse cProfile."}},
        "optimization",
        id="tier2-profile-code-references",
    ),
    pytest.param(
        "generate-tests",
        "2",
        {
            "scripts": {"gen_tests.py": "#!/usr/bin/env python3\n# generate tests"},
            "templates": {"test_template.py": "def test_example():\n    pass"},
        },
        "testing",
        id="tier2-generate-tests-scripts-and-templates",
    ),
]


class TestMigrateSkillTieredAuxiliaryDirs:
    """Tests that auxiliary subdirectories are copied for tier-1 and tier-2 skills."""

    @pytest.mark.parametrize(
        "skill_name,tier,extra_subdirs_spec,expected_category",
        _TIERED_AUXILIARY_CASES,
    )
    def test_tier_skill_auxiliary_subdirs_copied(
        self,
        tmp_path: Path,
        migrate_module,
        skill_name: str,
        tier: str,
        extra_subdirs_spec: dict,
        expected_category: str,
    ) -> None:
        """Auxiliary subdirs in tier-1/tier-2 skills are copied to the correct destinations."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_tiered_skill_dir(odyssey_skills, tier, skill_name, extra_subdirs=extra_subdirs_spec)
        source_skill_md = odyssey_skills / f"tier-{tier}" / skill_name / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name=skill_name,
                source_skill_md=source_skill_md,
                category=expected_category,
                dry_run=False,
                tier=tier,
            )

        assert result is True

        plugin_dir = mnemosyne_skills / expected_category / skill_name
        skill_inner_dir = plugin_dir / "skills" / skill_name

        for subdir_name, files in extra_subdirs_spec.items():
            if subdir_name == "references":
                dest_dir = plugin_dir / "references"
            else:
                dest_dir = skill_inner_dir / subdir_name

            assert dest_dir.exists(), f"Expected {subdir_name}/ at {dest_dir}"
            for filename, content in files.items():
                dest_file = dest_dir / filename
                assert dest_file.exists(), f"Expected file {filename} at {dest_file}"
                assert dest_file.read_text() == content

    def test_tier1_skill_dry_run_no_files_created(self, tmp_path: Path, migrate_module) -> None:
        """dry_run=True for a tier-1 skill with scripts/ creates no files."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_tiered_skill_dir(
            odyssey_skills,
            "1",
            "run-tests",
            extra_subdirs={"scripts": {"run_tests.sh": "#!/bin/bash\necho run"}},
        )
        source_skill_md = odyssey_skills / "tier-1" / "run-tests" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="run-tests",
                source_skill_md=source_skill_md,
                category="tooling",
                dry_run=True,
                tier="1",
            )

        assert result is True
        plugin_dir = mnemosyne_skills / "tooling" / "run-tests"
        assert not plugin_dir.exists(), "dry_run should not create any directories for tier-1 skill"

    def test_tier2_references_not_inside_skill_inner_dir(self, tmp_path: Path, migrate_module) -> None:
        """For a tier-2 skill, references/ lands at plugin root, NOT inside skills/<name>/."""
        odyssey_skills = tmp_path / "odyssey_skills"
        mnemosyne_skills = tmp_path / "mnemosyne" / "skills"
        mnemosyne_skills.mkdir(parents=True)

        make_tiered_skill_dir(
            odyssey_skills,
            "2",
            "profile-code",
            extra_subdirs={"references": {"notes.md": "# Profiling Reference"}},
        )
        source_skill_md = odyssey_skills / "tier-2" / "profile-code" / "SKILL.md"

        with patch.object(migrate_module, "MNEMOSYNE_SKILLS_DIR", mnemosyne_skills):
            result = migrate_module.migrate_skill(
                skill_name="profile-code",
                source_skill_md=source_skill_md,
                category="optimization",
                dry_run=False,
                tier="2",
            )

        assert result is True

        plugin_dir = mnemosyne_skills / "optimization" / "profile-code"
        skill_inner_dir = plugin_dir / "skills" / "profile-code"

        # references/ should be at plugin root
        assert (plugin_dir / "references" / "notes.md").exists()
        # references/ should NOT be inside skills/<name>/references/
        wrong_location = skill_inner_dir / "references"
        assert not wrong_location.exists(), "references/ must not be nested inside skills/<name>/ for tier-2 skills"
