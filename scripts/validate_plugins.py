#!/usr/bin/env python3
"""
Validate plugin structure and content in the ProjectMnemosyne marketplace.

This script validates:
- plugin.json exists and has required fields (name, version, description)
- SKILL.md exists and has required sections
- Failed Attempts section is present (required)
- Description is specific (20+ chars)
- Category is valid if present (one of 8 approved)

Note: category and date are optional - category is derived from directory structure.

Usage:
    python3 scripts/validate_plugins.py [plugins_dir]

    If plugins_dir not provided, checks plugins/
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


# Validation thresholds
MIN_DESCRIPTION_LENGTH = 20

# Valid categories
VALID_CATEGORIES = {
    "training",
    "evaluation",
    "optimization",
    "debugging",
    "architecture",
    "tooling",
    "ci-cd",
    "testing",
    "documentation",
}

# Required plugin.json fields (per official Claude Code plugin docs)
# Note: category is derived from directory structure, date is optional metadata
REQUIRED_PLUGIN_FIELDS = {"name", "version", "description"}


@dataclass
class ValidationResult:
    """Result of validating a plugin."""

    plugin_path: Path
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __str__(self) -> str:
        """Format validation result as string."""
        status = "PASS" if self.is_valid else "FAIL"
        output = [f"\n{status}: {self.plugin_path.name}"]

        if self.errors:
            output.append("  Errors:")
            for error in self.errors:
                output.append(f"    - {error}")

        if self.warnings:
            output.append("  Warnings:")
            for warning in self.warnings:
                output.append(f"    - {warning}")

        return "\n".join(output)


def validate_plugin_json(plugin_dir: Path) -> Tuple[List[str], List[str], dict]:
    """Validate plugin.json file."""
    errors: List[str] = []
    warnings: List[str] = []
    data: dict = {}

    plugin_json_path = plugin_dir / ".claude-plugin" / "plugin.json"

    if not plugin_json_path.exists():
        errors.append("Missing .claude-plugin/plugin.json")
        return errors, warnings, data

    try:
        with open(plugin_json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in plugin.json: {e}")
        return errors, warnings, data

    # Check required fields
    missing_fields = REQUIRED_PLUGIN_FIELDS - set(data.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(sorted(missing_fields))}")

    # Validate name format
    if "name" in data:
        name = data["name"]
        if not re.match(r"^[a-z0-9-]+$", name):
            errors.append(f"Invalid name format '{name}' (use lowercase, numbers, hyphens)")

    # Validate description length
    if "description" in data:
        desc = data["description"]
        if len(desc) < MIN_DESCRIPTION_LENGTH:
            errors.append(f"Description too short ({len(desc)} chars, min {MIN_DESCRIPTION_LENGTH})")

    # Validate category
    if "category" in data:
        category = data["category"]
        if category not in VALID_CATEGORIES:
            errors.append(f"Invalid category '{category}'. Valid: {', '.join(sorted(VALID_CATEGORIES))}")

    # Validate date format
    if "date" in data:
        date = data["date"]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            errors.append(f"Invalid date format '{date}' (use YYYY-MM-DD)")

    # Validate tags is a list
    if "tags" in data:
        if not isinstance(data["tags"], list):
            errors.append("tags must be a list")
        elif len(data["tags"]) == 0:
            warnings.append("No tags provided - reduces searchability")

    return errors, warnings, data


def validate_skill_md(plugin_dir: Path, plugin_data: dict) -> Tuple[List[str], List[str]]:
    """Validate SKILL.md file."""
    errors: List[str] = []
    warnings: List[str] = []

    # Find SKILL.md in skills directory
    skills_dir = plugin_dir / "skills"
    if not skills_dir.exists():
        errors.append("Missing skills/ directory")
        return errors, warnings

    skill_files = list(skills_dir.glob("*/SKILL.md"))
    if not skill_files:
        errors.append("No SKILL.md found in skills/ subdirectories")
        return errors, warnings

    skill_md_path = skill_files[0]

    try:
        content = skill_md_path.read_text()
    except Exception as e:
        errors.append(f"Failed to read SKILL.md: {e}")
        return errors, warnings

    # Check for YAML frontmatter
    if not content.startswith("---"):
        errors.append("SKILL.md missing YAML frontmatter (must start with ---)")

    # Check for required sections
    required_sections = [
        ("## Overview", "Overview table"),
        ("## When to Use", "When to Use section"),
        ("## Verified Workflow", "Verified Workflow section"),
        ("## Failed Attempts", "Failed Attempts section (REQUIRED)"),
        ("## Results", "Results & Parameters section"),
    ]

    for section_marker, section_name in required_sections:
        if section_marker not in content:
            if "Failed Attempts" in section_name:
                errors.append(f"Missing {section_name}")
            else:
                warnings.append(f"Missing {section_name}")

    # Check Failed Attempts has actual content (table)
    if "## Failed Attempts" in content:
        failed_section = content.split("## Failed Attempts")[1].split("##")[0]
        if "|" not in failed_section:
            warnings.append("Failed Attempts section should contain a table")

    # Error if ## Quick Reference appears as a top-level section alongside ## Verified Workflow.
    # This was previously a warning but is now an error so CI rejects new submissions
    # with orphaned Quick Reference sections (see #3778, follow-up from #3230).
    if "## Verified Workflow" in content and re.search(r"^## Quick Reference", content, re.MULTILINE):
        errors.append(
            "## Quick Reference should be a subsection (### Quick Reference) of "
            "## Verified Workflow, not a top-level section"
        )

    return errors, warnings


def validate_plugin(plugin_dir: Path) -> ValidationResult:
    """Validate a single plugin."""
    errors = []
    warnings = []

    # Validate plugin.json
    json_errors, json_warnings, plugin_data = validate_plugin_json(plugin_dir)
    errors.extend(json_errors)
    warnings.extend(json_warnings)

    # Validate SKILL.md
    skill_errors, skill_warnings = validate_skill_md(plugin_dir, plugin_data)
    errors.extend(skill_errors)
    warnings.extend(skill_warnings)

    return ValidationResult(
        plugin_path=plugin_dir,
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def find_plugins(plugins_dir: Path) -> List[Path]:
    """Find all plugin directories."""
    plugins = []

    for category_dir in plugins_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if category_dir.name.startswith("."):
            continue

        for plugin_dir in category_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
            if plugin_dir.name.startswith("."):
                continue

            # Check if it has plugin structure
            if (plugin_dir / ".claude-plugin").exists() or (plugin_dir / "skills").exists():
                plugins.append(plugin_dir)

    return plugins


def main() -> int:
    """Main entry point.

    Usage: validate_plugins.py [scan_dir ...]
    Defaults: skills/ plugins/
    """
    scan_dir_args = sys.argv[1:] if len(sys.argv) > 1 else ["skills", "plugins"]

    all_plugins: List[Path] = []
    for dir_arg in scan_dir_args:
        d = Path(dir_arg)
        if not d.exists():
            continue
        all_plugins.extend(find_plugins(d))

    if not all_plugins:
        print(f"No plugins found in: {scan_dir_args}")
        print("This is OK if the marketplace is empty.")
        return 0

    # Deduplicate by name (keep first occurrence)
    seen: set = set()
    plugins: List[Path] = []
    for p in all_plugins:
        if p.name not in seen:
            seen.add(p.name)
            plugins.append(p)

    results = [validate_plugin(p) for p in plugins]

    # Print results
    total = len(results)
    passed = sum(1 for r in results if r.is_valid)
    failed = total - passed

    print("=" * 60)
    print("PLUGIN VALIDATION")
    print("=" * 60)
    print(f"Total plugins: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    for result in results:
        print(result)

    print("\n" + "=" * 60)
    if failed == 0:
        print("ALL VALIDATIONS PASSED")
    else:
        print(f"VALIDATION FAILED: {failed} plugin(s) with errors")
    print("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
