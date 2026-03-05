#!/usr/bin/env python3
"""
Migrate Odyssey2 skills to ProjectMnemosyne plugin format.

Reads source SKILL.md files from Odyssey2 .claude/skills/ and creates
the proper plugin structure in ProjectMnemosyne skills/ directory.

Usage:
    python3 scripts/migrate_odyssey_skills.py [--dry-run] [--skill SKILL_NAME]
    python3 scripts/migrate_odyssey_skills.py --dry-run
    python3 scripts/migrate_odyssey_skills.py --skill gh-create-pr-linked

Source structure (Odyssey2):
    .claude/skills/<skill-name>/SKILL.md

Target structure (ProjectMnemosyne):
    skills/<category>/<skill-name>/.claude-plugin/plugin.json
    skills/<category>/<skill-name>/skills/<skill-name>/SKILL.md
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# Source Odyssey2 skills directory
ODYSSEY_SKILLS_DIR = Path("/home/mvillmow/Odyssey2/.claude/skills")

# Target ProjectMnemosyne directory
MNEMOSYNE_DIR = Path("/home/mvillmow/Odyssey2/build/ProjectMnemosyne")
MNEMOSYNE_SKILLS_DIR = MNEMOSYNE_DIR / "skills"

# Category mapping from Odyssey categories to Mnemosyne valid categories
CATEGORY_MAP = {
    "github": "tooling",
    "worktree": "tooling",
    "ci": "ci-cd",
    "phase": "ci-cd",
    "mojo": "architecture",
    "doc": "documentation",
    "quality": "evaluation",
    "review": "evaluation",
    "testing": "testing",
    "analysis": "optimization",
    "ml": "optimization",
    "agent": "tooling",
    "plan": "tooling",
    "generation": "tooling",
    "tooling": "tooling",
    "training": "training",
    "evaluation": "evaluation",
    "optimization": "optimization",
    "debugging": "debugging",
    "architecture": "architecture",
    "ci-cd": "ci-cd",
    "documentation": "documentation",
}

# Category for tier-1 and tier-2 skills based on their purpose
TIER1_CATEGORY = "tooling"
TIER2_CATEGORY_MAP = {
    "analyze-code-structure": "evaluation",
    "analyze-equations": "optimization",
    "benchmark-functions": "optimization",
    "calculate-coverage": "evaluation",
    "check-dependencies": "tooling",
    "detect-code-smells": "evaluation",
    "evaluate-model": "evaluation",
    "extract-algorithm": "optimization",
    "extract-dependencies": "tooling",
    "extract-hyperparameters": "optimization",
    "generate-api-docs": "documentation",
    "generate-changelog": "documentation",
    "generate-docstrings": "documentation",
    "generate-tests": "testing",
    "identify-architecture": "architecture",
    "prepare-dataset": "training",
    "profile-code": "optimization",
    "refactor-code": "tooling",
    "scan-vulnerabilities": "debugging",
    "suggest-optimizations": "optimization",
    "train-model": "training",
    "validate-inputs": "testing",
}

# Skill-to-category overrides for top-level skills
SKILL_CATEGORY_OVERRIDE = {
    "analyze-ci-failure-logs": "ci-cd",
    "build-run-local": "ci-cd",
    "fix-ci-failures": "ci-cd",
    "install-workflow": "ci-cd",
    "run-precommit": "ci-cd",
    "validate-workflow": "ci-cd",
    "agent-coverage-check": "tooling",
    "agent-hierarchy-diagram": "tooling",
    "agent-run-orchestrator": "tooling",
    "agent-test-delegation": "tooling",
    "agent-validate-config": "tooling",
    "analyze-simd-usage": "architecture",
    "check-memory-safety": "architecture",
    "mojo-build-package": "architecture",
    "mojo-format": "architecture",
    "mojo-lint-syntax": "architecture",
    "mojo-memory-check": "architecture",
    "mojo-simd-optimize": "architecture",
    "mojo-test-runner": "architecture",
    "mojo-type-safety": "architecture",
    "validate-mojo-patterns": "architecture",
    "doc-generate-adr": "documentation",
    "doc-issue-readme": "documentation",
    "doc-update-blog": "documentation",
    "doc-validate-markdown": "documentation",
    "create-review-checklist": "evaluation",
    "quality-complexity-check": "evaluation",
    "quality-coverage-report": "evaluation",
    "quality-fix-formatting": "evaluation",
    "quality-run-linters": "evaluation",
    "quality-security-scan": "evaluation",
    "review-pr-changes": "evaluation",
    "track-implementation-progress": "evaluation",
    "extract-test-failures": "testing",
    "generate-fix-suggestions": "testing",
    "test-diff-analyzer": "testing",
    "phase-cleanup": "ci-cd",
    "phase-implement": "ci-cd",
    "phase-package": "ci-cd",
    "phase-plan-generate": "ci-cd",
    "phase-test-tdd": "ci-cd",
    "plan-create-component": "tooling",
    "plan-regenerate-issues": "tooling",
    "plan-validate-structure": "tooling",
    "gh-batch-merge-by-labels": "tooling",
    "gh-check-ci-status": "tooling",
    "gh-create-pr-linked": "tooling",
    "gh-fix-pr-feedback": "tooling",
    "gh-get-review-comments": "tooling",
    "gh-implement-issue": "tooling",
    "gh-post-issue-update": "tooling",
    "gh-read-issue-context": "tooling",
    "gh-reply-review-comment": "tooling",
    "gh-review-pr": "tooling",
    "verify-pr-ready": "tooling",
    "worktree-cleanup": "tooling",
    "worktree-create": "tooling",
    "worktree-switch": "tooling",
    "worktree-sync": "tooling",
}

# Replacements for hardcoded Odyssey paths
PATH_REPLACEMENTS = [
    (r"/home/mvillmow/Odyssey2/", "<project-root>/"),
    (r"/home/mvillmow/ProjectOdyssey/", "<project-root>/"),
    (r"\bProjectOdyssey2\b", "<project-name>"),
    (r"\bProjectOdyssey\b", "<project-name>"),
    (r"\bpixi run mojo\b", "<package-manager> run mojo"),
    (r"\bpixi run\b", "<package-manager> run"),
    (r"tests/models/test_", "tests/<model>/test_"),
    (r"\./scripts/create_worktree\.sh", "<project-root>/scripts/create_worktree.sh"),
    (r"WorktreeName-\d+-", "<project-name>-<issue-number>-"),
]


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter_dict, remaining_content)
    """
    if not content.startswith("---"):
        return {}, content

    lines = content.split("\n")
    end_idx = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, content

    frontmatter_lines = lines[1:end_idx]
    remaining = "\n".join(lines[end_idx + 1:])

    frontmatter: dict = {}
    for line in frontmatter_lines:
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                frontmatter[key] = value

    return frontmatter, remaining


def determine_category(skill_name: str, frontmatter: dict, tier: Optional[str] = None) -> str:
    """Determine the Mnemosyne category for a skill."""
    # Check explicit overrides first
    if skill_name in SKILL_CATEGORY_OVERRIDE:
        return SKILL_CATEGORY_OVERRIDE[skill_name]

    # Check tier-2 mapping
    if tier == "2" and skill_name in TIER2_CATEGORY_MAP:
        return TIER2_CATEGORY_MAP[skill_name]

    if tier == "1":
        return TIER1_CATEGORY

    # Map from frontmatter category
    fm_category = frontmatter.get("category", "").lower()
    if fm_category in CATEGORY_MAP:
        return CATEGORY_MAP[fm_category]

    # Default to tooling
    return "tooling"


def generalize_content(content: str) -> str:
    """Replace hardcoded Odyssey-specific paths and references."""
    for pattern, replacement in PATH_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    return content


def transform_skill_md(source_content: str, skill_name: str, category: str, tier: Optional[str] = None) -> str:
    """Transform Odyssey SKILL.md into Mnemosyne format.

    Key transformations:
    - Remove Odyssey-specific frontmatter fields
    - Rename ## Workflow -> ## Verified Workflow
    - Add ## Overview table if missing
    - Add ## Failed Attempts stub if missing
    - Add ## Results & Parameters stub if missing
    - Generalize hardcoded paths
    """
    frontmatter, body = parse_frontmatter(source_content)

    # Build clean frontmatter with only Mnemosyne fields
    name = frontmatter.get("name", skill_name)
    description = frontmatter.get("description", f"{skill_name} skill")
    user_invocable = frontmatter.get("user-invocable", "false")

    new_frontmatter = f"""---
name: {name}
description: {description}
category: {category}
date: 2025-01-01
user-invocable: {user_invocable}
---"""

    # Generalize the body content
    body = generalize_content(body)

    # Rename ## Workflow -> ## Verified Workflow (only if not already Verified Workflow)
    body = re.sub(r"^## Workflow\b", "## Verified Workflow", body, flags=re.MULTILINE)

    # Check for required sections
    has_overview = "## Overview" in body
    has_when_to_use = "## When to Use" in body
    has_verified_workflow = "## Verified Workflow" in body
    has_failed_attempts = "## Failed Attempts" in body
    has_results = "## Results" in body

    # Build new body with required sections injected
    sections = []

    # Split body into title + rest
    body_lines = body.strip().split("\n")
    title_line = ""
    rest_lines = []

    for i, line in enumerate(body_lines):
        if line.startswith("# "):
            title_line = line
            rest_lines = body_lines[i + 1:]
            break
    else:
        rest_lines = body_lines

    if title_line:
        sections.append(title_line)
        sections.append("")

    # Add Overview if missing
    if not has_overview:
        overview = f"""## Overview

| Item | Details |
|------|---------|
| Name | {name} |
| Category | {category} |
| Description | {description.strip('"')} |
"""
        sections.append(overview)

    # Reconstruct remaining body
    remaining_body = "\n".join(rest_lines).strip()

    # Split remaining body into section chunks
    # We'll parse sections and insert missing ones at right places
    section_pattern = re.compile(r"^## ", re.MULTILINE)
    section_positions = [m.start() for m in section_pattern.finditer(remaining_body)]

    if not section_positions:
        # No sections found, just append
        if remaining_body:
            sections.append(remaining_body)
        if not has_when_to_use:
            sections.append("\n## When to Use\n\n- Use when this skill's functionality is needed.\n")
        if not has_verified_workflow:
            sections.append("\n## Verified Workflow\n\n1. Identify the task requiring this skill.\n2. Follow the Quick Reference commands above.\n3. Verify results.\n")
        if not has_failed_attempts:
            sections.append("\n## Failed Attempts\n\n| Attempt | Why Failed | Lesson Learned |\n|---------|-----------|----------------|\n| N/A | No recorded failures | N/A |\n")
        if not has_results:
            sections.append("\n## Results & Parameters\n\nSee Quick Reference section above for commands and parameters.\n")
    else:
        # Parse out existing sections and re-order/add missing
        existing_sections: list[tuple[str, str]] = []
        for i, pos in enumerate(section_positions):
            end_pos = section_positions[i + 1] if i + 1 < len(section_positions) else len(remaining_body)
            section_text = remaining_body[pos:end_pos].strip()
            # Get section header
            first_newline = section_text.find("\n")
            if first_newline != -1:
                header = section_text[:first_newline].strip()
            else:
                header = section_text.strip()
            existing_sections.append((header, section_text))

        # Build ordered output
        # Order: When to Use, (other sections), Verified Workflow, Failed Attempts, Results
        ordered_headers = [
            "## When to Use",
            "## Verified Workflow",
            "## Failed Attempts",
            "## Results",
        ]

        # Sections to place before Verified Workflow
        pre_workflow = []
        post_workflow_order = []
        workflow_section = None
        failed_section = None
        results_section = None
        when_to_use_section = None

        other_sections = []
        for header, text in existing_sections:
            if header == "## When to Use":
                when_to_use_section = text
            elif header in ("## Verified Workflow", "## Workflow"):
                text = text.replace("## Workflow\n", "## Verified Workflow\n", 1)
                workflow_section = text
            elif header == "## Failed Attempts":
                failed_section = text
            elif header.startswith("## Results"):
                results_section = text
            else:
                other_sections.append(text)

        # Build final section order
        if when_to_use_section:
            sections.append(when_to_use_section)
            sections.append("")
        elif not has_when_to_use:
            sections.append("## When to Use\n\n- Use when this skill's functionality is needed.\n")

        for s in other_sections:
            sections.append(s)
            sections.append("")

        if workflow_section:
            sections.append(workflow_section)
            sections.append("")
        elif not has_verified_workflow:
            sections.append("## Verified Workflow\n\n1. Identify the task.\n2. Execute the relevant commands.\n3. Verify results.\n")

        if failed_section:
            # Ensure it has a pipe table
            if "|" not in failed_section:
                failed_section += "\n\n| Attempt | Why Failed | Lesson Learned |\n|---------|-----------|----------------|\n| N/A | No recorded failures | N/A |\n"
            sections.append(failed_section)
            sections.append("")
        else:
            sections.append("## Failed Attempts\n\n| Attempt | Why Failed | Lesson Learned |\n|---------|-----------|----------------|\n| N/A | No recorded failures | N/A |\n")

        if results_section:
            sections.append(results_section)
        else:
            sections.append("## Results & Parameters\n\nSee the workflow sections above for commands and expected outputs.\n")

    final_body = "\n".join(sections)
    return f"{new_frontmatter}\n\n{final_body}\n"


def create_plugin_json(skill_name: str, description: str, category: str, tags: list[str]) -> dict:
    """Create plugin.json content."""
    # Ensure description is long enough
    if len(description) < 20:
        description = f"{skill_name}: {description}"

    return {
        "name": skill_name,
        "version": "1.0.0",
        "description": description,
        "category": category,
        "date": "2025-01-01",
        "tags": tags,
    }


def get_tags_for_skill(skill_name: str, category: str, frontmatter: dict) -> list[str]:
    """Generate relevant tags for a skill."""
    tags = []

    # Add category as tag
    tags.append(category)

    # Extract tags from skill name parts
    parts = skill_name.split("-")
    if len(parts) >= 2:
        prefix = parts[0]
        if prefix in ("gh", "mojo", "doc", "quality", "agent", "phase", "plan", "worktree"):
            tags.append(prefix)

    # Add specific tags based on skill name
    tag_map = {
        "gh-": ["github", "gh-cli"],
        "mojo-": ["mojo", "language"],
        "doc-": ["documentation"],
        "quality-": ["code-quality", "linting"],
        "agent-": ["agent-system"],
        "phase-": ["workflow", "development-phases"],
        "plan-": ["planning"],
        "worktree-": ["git-worktree", "parallel-development"],
        "tier-": ["automation"],
    }

    for prefix, prefix_tags in tag_map.items():
        if skill_name.startswith(prefix):
            tags.extend(prefix_tags)
            break

    # Deduplicate while preserving order
    seen: set = set()
    result = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)

    return result[:6]  # Cap at 6 tags


def migrate_skill(
    skill_name: str,
    source_skill_md: Path,
    category: str,
    dry_run: bool = False,
    tier: Optional[str] = None,
) -> bool:
    """Migrate a single skill to ProjectMnemosyne format.

    Returns:
        True if migration succeeded, False otherwise.
    """
    if not source_skill_md.exists():
        print(f"  ERROR: Source SKILL.md not found: {source_skill_md}")
        return False

    source_content = source_skill_md.read_text()
    frontmatter, _ = parse_frontmatter(source_content)

    description = frontmatter.get("description", f"Skill: {skill_name}").strip('"').strip("'")
    if len(description) < 20:
        description = f"{skill_name}: automated skill for {category} workflows"

    # Determine output paths
    plugin_dir = MNEMOSYNE_SKILLS_DIR / category / skill_name
    plugin_json_path = plugin_dir / ".claude-plugin" / "plugin.json"
    skill_md_dir = plugin_dir / "skills" / skill_name
    skill_md_path = skill_md_dir / "SKILL.md"

    print(f"  Migrating {skill_name} -> skills/{category}/{skill_name}/")

    if not dry_run:
        # Create directories
        plugin_json_path.parent.mkdir(parents=True, exist_ok=True)
        skill_md_dir.mkdir(parents=True, exist_ok=True)

        # Write plugin.json
        tags = get_tags_for_skill(skill_name, category, frontmatter)
        plugin_data = create_plugin_json(skill_name, description, category, tags)
        with open(plugin_json_path, "w") as f:
            json.dump(plugin_data, f, indent=2)
            f.write("\n")

        # Transform and write SKILL.md
        transformed = transform_skill_md(source_content, skill_name, category, tier=tier)
        skill_md_path.write_text(transformed)

        print(f"  OK: {plugin_json_path}")
        print(f"  OK: {skill_md_path}")
    else:
        print(f"  [DRY RUN] Would create: {plugin_dir}")

    return True


def find_all_skills() -> list[tuple[str, Path, Optional[str]]]:
    """Find all skills in Odyssey2, returning (name, skill_md_path, tier) tuples."""
    skills = []

    # Top-level skills
    for skill_dir in sorted(ODYSSEY_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        if skill_dir.name.startswith("."):
            continue
        if skill_dir.name in ("tier-1", "tier-2", "SKILL_FORMAT_TEMPLATE.md"):
            continue

        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            skills.append((skill_dir.name, skill_md, None))

    # Tier-1 skills
    tier1_dir = ODYSSEY_SKILLS_DIR / "tier-1"
    if tier1_dir.exists():
        for skill_dir in sorted(tier1_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skills.append((skill_dir.name, skill_md, "1"))

    # Tier-2 skills
    tier2_dir = ODYSSEY_SKILLS_DIR / "tier-2"
    if tier2_dir.exists():
        for skill_dir in sorted(tier2_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skills.append((skill_dir.name, skill_md, "2"))

    return skills


def skill_already_exists(skill_name: str) -> bool:
    """Check if a skill already exists in ProjectMnemosyne."""
    for category_dir in MNEMOSYNE_SKILLS_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        skill_path = category_dir / skill_name
        if skill_path.exists():
            return True
    return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Odyssey2 skills to ProjectMnemosyne format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skill",
        metavar="SKILL_NAME",
        help="Migrate only a specific skill",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip skills that already exist in Mnemosyne (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skills",
    )
    args = parser.parse_args()

    if not ODYSSEY_SKILLS_DIR.exists():
        print(f"ERROR: Odyssey skills directory not found: {ODYSSEY_SKILLS_DIR}")
        return 1

    if not MNEMOSYNE_DIR.exists():
        print(f"ERROR: ProjectMnemosyne directory not found: {MNEMOSYNE_DIR}")
        return 1

    all_skills = find_all_skills()
    print(f"Found {len(all_skills)} skills in Odyssey2")

    # Filter to specific skill if requested
    if args.skill:
        all_skills = [(n, p, t) for n, p, t in all_skills if n == args.skill]
        if not all_skills:
            print(f"ERROR: Skill '{args.skill}' not found")
            return 1

    succeeded = 0
    skipped = 0
    failed = 0

    for skill_name, skill_md_path, tier in all_skills:
        # Check if already exists
        if not args.force and skill_already_exists(skill_name):
            print(f"SKIP: {skill_name} (already exists in Mnemosyne)")
            skipped += 1
            continue

        # Determine category
        source_content = skill_md_path.read_text()
        frontmatter, _ = parse_frontmatter(source_content)
        category = determine_category(skill_name, frontmatter, tier=tier)

        print(f"Processing: {skill_name} (category={category}, tier={tier})")

        ok = migrate_skill(
            skill_name=skill_name,
            source_skill_md=skill_md_path,
            category=category,
            dry_run=args.dry_run,
            tier=tier,
        )

        if ok:
            succeeded += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Migration Summary:")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
