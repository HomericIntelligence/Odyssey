#!/usr/bin/env python3
"""
Validate agent configuration files in .claude/agents/.

This script validates:
- YAML frontmatter syntax
- Required fields (name, description, tools, model)
- Tool specifications
- Description quality (clear enough for auto-invocation)
- File naming conventions
- Mojo-specific guidelines presence

Usage:
    python3 tests/agents/validate_configs.py [agent_dir]

    If agent_dir not provided, checks .claude/agents/
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import yaml


# Security limits
MAX_FILE_SIZE = 102400  # 100KB max file size to prevent DoS

# Validation thresholds
MIN_DESCRIPTION_LENGTH = 20  # Minimum description characters
MIN_TOOL_COUNT = 1  # Minimum number of tools
MAX_TOOL_COUNT = 10  # Maximum number of tools (sanity check)


@dataclass
class ValidationResult:
    """Result of validating an agent configuration."""

    file_path: Path
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __str__(self) -> str:
        """Format validation result as string."""
        status = "PASS" if self.is_valid else "FAIL"
        output = [f"\n{status}: {self.file_path.name}"]

        if self.errors:
            output.append("  Errors:")
            for error in self.errors:
                output.append(f"    - {error}")

        if self.warnings:
            output.append("  Warnings:")
            for warning in self.warnings:
                output.append(f"    - {warning}")

        return "\n".join(output)


class AgentConfigValidator:
    """Validator for agent configuration files."""

    # Required frontmatter fields
    REQUIRED_FIELDS = {"name", "description", "tools", "model"}

    # Valid Claude Code tools
    # This list must be kept in sync with available tools in Claude Code.
    # Reference: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
    #
    # To update when new tools are added:
    # 1. Check Claude Code documentation for new tools
    # 2. Add tool name to this list (comma-separated)
    # 3. Update tests if tool validation logic changes
    #
    # Common tools: Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch, etc.
    VALID_TOOLS = {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
        "WebFetch",
        "WebSearch",
        "NotebookEdit",
        "AskUserQuestion",
        "TodoWrite",
        "Task",
        "Skill",
        "SlashCommand",
    }

    # Valid model values
    VALID_MODELS = {"sonnet", "opus", "haiku"}

    # Agent levels and their expected name patterns
    LEVEL_PATTERNS = {
        0: r"chief-architect",
        1: r".*-orchestrator",
        2: r".*-design",
        3: r".*-specialist",
        4: r".*-engineer",
        5: r"junior-.*-engineer",
    }

    # Mojo-specific keywords that should appear in relevant agents
    MOJO_KEYWORDS = {
        "fn vs def",
        "struct vs class",
        "SIMD",
        "@parameter",
        "owned",
        "borrowed",
        "performance",
        "optimization",
        "vectorization",
    }

    def __init__(self, agents_dir: Path):
        """Initialize validator.

        Args:
            agents_dir: Directory containing agent .md files
        """
        self.agents_dir = agents_dir
        self.results: List[ValidationResult] = []
        self.existing_agents: set = {f.stem for f in agents_dir.glob("*.md")} if agents_dir.exists() else set()

    def validate_all(self) -> List[ValidationResult]:
        """Validate all agent configuration files.

        Returns:
            List of validation results
        """
        if not self.agents_dir.exists():
            print(f"Error: Directory {self.agents_dir} does not exist")
            return []

        agent_files = sorted(self.agents_dir.glob("*.md"))

        if not agent_files:
            print(f"Warning: No .md files found in {self.agents_dir}")
            return []

        for agent_file in agent_files:
            # Check file size before reading
            file_size = agent_file.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logging.warning(f"Skipping {agent_file.name} (file too large: {file_size} bytes)")
                self.results.append(
                    ValidationResult(
                        agent_file,
                        False,
                        [f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})"],
                        [],
                    )
                )
                continue

            result = self.validate_file(agent_file)
            self.results.append(result)

        return self.results

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a single agent configuration file.

        Args:
            file_path: Path to agent .md file

        Returns:
            ValidationResult for the file
        """
        errors = []
        warnings = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            logging.error(f"Failed to read {file_path.name}: {e}")
            return ValidationResult(file_path, False, errors, warnings)

        # Validate YAML frontmatter
        frontmatter_errors, frontmatter_warnings, frontmatter = self._validate_frontmatter(content)
        errors.extend(frontmatter_errors)
        warnings.extend(frontmatter_warnings)

        # Validate file naming
        naming_errors, naming_warnings = self._validate_naming(file_path, frontmatter)
        errors.extend(naming_errors)
        warnings.extend(naming_warnings)

        # Validate Mojo patterns
        mojo_warnings = self._validate_mojo_patterns(content, frontmatter)
        warnings.extend(mojo_warnings)

        # Validate content structure
        content_warnings = self._validate_content_structure(content)
        warnings.extend(content_warnings)

        is_valid = len(errors) == 0
        return ValidationResult(file_path, is_valid, errors, warnings)

    def _validate_frontmatter(self, content: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate YAML frontmatter.

        Args:
            content: File content

        Returns:
            Tuple of (errors, warnings, frontmatter_dict).
        """
        errors = []
        warnings = []
        frontmatter: Dict[str, Any] = {}

        # Extract frontmatter
        frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            errors.append("No YAML frontmatter found (must start with ---)")
            return errors, warnings, frontmatter

        frontmatter_text = frontmatter_match.group(1)

        # Parse frontmatter using PyYAML to correctly handle colon-containing values
        try:
            parsed = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML frontmatter: {e}")
            return errors, warnings, frontmatter

        if not isinstance(parsed, dict):
            errors.append("Frontmatter must be a YAML mapping")
            return errors, warnings, frontmatter

        frontmatter = parsed

        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(frontmatter.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(sorted(missing_fields))}")

        # Validate specific fields
        if "name" in frontmatter:
            name = frontmatter["name"]
            if not re.match(r"^[a-z0-9-]+$", name):
                errors.append(f"Invalid name format '{name}' (use lowercase, numbers, hyphens only)")

        if "description" in frontmatter:
            desc = frontmatter["description"]
            if len(desc) < MIN_DESCRIPTION_LENGTH:
                warnings.append(f"Description too short ({len(desc)} chars) - may not trigger auto-invocation")
            if not any(word in desc.lower() for word in ["when", "for", "to", "implement", "design", "test"]):
                warnings.append("Description should clearly state when to use this agent")

        if "tools" in frontmatter:
            tools = [t.strip() for t in frontmatter["tools"].split(",")]
            invalid_tools = [t for t in tools if t not in self.VALID_TOOLS]
            if invalid_tools:
                errors.append(f"Invalid tools: {', '.join(invalid_tools)}")
            if not tools or len(tools) < MIN_TOOL_COUNT:
                errors.append("No tools specified")
            if len(tools) > MAX_TOOL_COUNT:
                warnings.append(f"Large number of tools specified ({len(tools)}), verify this is intentional")

        if "model" in frontmatter:
            model = frontmatter["model"].lower()
            if model not in self.VALID_MODELS:
                errors.append(f"Invalid model '{model}' (use: {', '.join(self.VALID_MODELS)})")

        if "delegates_to" in frontmatter:
            raw = frontmatter["delegates_to"].strip()
            # Parse YAML inline list: [name-a, name-b] or empty []
            if raw.startswith("[") and raw.endswith("]"):
                inner = raw[1:-1].strip()
                if inner:
                    names = [n.strip() for n in inner.split(",") if n.strip()]
                    for name in names:
                        if name not in self.existing_agents:
                            errors.append(
                                f"delegates_to references non-existent agent: '{name}'"
                                f" (no .claude/agents/{name}.md)"
                            )
            else:
                errors.append(f"delegates_to must be a YAML inline list, got: '{raw}'")

        return errors, warnings, frontmatter

    def _validate_naming(self, file_path: Path, frontmatter: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate file naming conventions.

        Args:
            file_path: Path to agent file
            frontmatter: Parsed frontmatter

        Returns:
            Tuple of (errors, warnings).
        """
        errors = []
        warnings = []

        filename = file_path.stem  # Without .md extension

        # Check that filename matches frontmatter name
        if "name" in frontmatter:
            if filename != frontmatter["name"]:
                errors.append(f"Filename '{filename}' doesn't match name '{frontmatter['name']}'")

        # Check for level patterns
        level_matched = False
        for level, pattern in self.LEVEL_PATTERNS.items():
            if re.match(pattern, filename):
                level_matched = True
                break

        if not level_matched:
            warnings.append(f"Filename '{filename}' doesn't match standard level patterns")

        return errors, warnings

    def _validate_mojo_patterns(self, content: str, frontmatter: Dict[str, Any]) -> List[str]:
        """Validate Mojo-specific guidelines presence.

        Args:
            content: File content
            frontmatter: Parsed frontmatter

        Returns:
            List of warnings
        """
        warnings = []

        # Check if this is an implementation-level agent
        name = frontmatter.get("name", "")
        is_implementation = any(word in name for word in ["engineer", "specialist", "implementation"])

        if not is_implementation:
            return warnings  # Only check implementation agents

        # Check for Mojo keywords
        content_lower = content.lower()
        found_keywords = [kw for kw in self.MOJO_KEYWORDS if kw.lower() in content_lower]

        # Commented out: Mojo guidance check is too strict for current agent design
        if not found_keywords:
            warnings.append(
                "Implementation agent should include Mojo-specific guidance (fn vs def, struct vs class, SIMD, etc.)"
            )
        elif len(found_keywords) < 3:
            warnings.append(f"Limited Mojo guidance found (only {len(found_keywords)} keywords)")

        # Check for specific sections
        if "mojo" not in content_lower:
            warnings.append("No explicit 'Mojo' section found - consider adding Mojo-specific guidelines")

        return warnings

    def _validate_content_structure(self, content: str) -> List[str]:
        """Validate content structure and completeness.

        Args:
            content: File content

        Returns:
            List of warnings
        """
        warnings = []

        # Expected sections
        expected_sections = [
            "Role",
            "Responsibilities",
            "Scope",
            "Delegation",
            "Workflow",
            "Examples",
        ]

        content_lower = content.lower()
        missing_sections = []

        for section in expected_sections:
            # Check for section headers (## Section or # Section)
            if not re.search(rf"^#+ {section}", content, re.MULTILINE | re.IGNORECASE):
                missing_sections.append(section)

        if missing_sections:
            warnings.append(f"Missing recommended sections: {', '.join(missing_sections)}")

        # Check for delegation information
        if "delegate" not in content_lower and "escalate" not in content_lower:
            warnings.append("No delegation/escalation guidance found")

        # Check for examples
        if "example" not in content_lower:
            warnings.append("No examples provided")

        return warnings

    def print_summary(self) -> None:
        """Print validation summary."""
        if not self.results:
            print("No results to summarize")
            return

        total = len(self.results)
        passed = sum(1 for r in self.results if r.is_valid)
        failed = total - passed

        total_errors = sum(len(r.errors) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total files: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total errors: {total_errors}")
        print(f"Total warnings: {total_warnings}")
        print("=" * 80)

        # Print individual results
        for result in self.results:
            print(result)

        # Print final status
        print("\n" + "=" * 80)
        if failed == 0:
            print("ALL VALIDATIONS PASSED")
        else:
            print(f"VALIDATION FAILED: {failed} file(s) with errors")
        print("=" * 80)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Determine agents directory
    agents_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Validate input
    if agents_dir_arg:
        if not agents_dir_arg:
            logger.error("agents_dir path is required")
            return 1

        agents_dir = Path(agents_dir_arg)
        if not agents_dir.exists():
            logger.error(f"Directory does not exist: {agents_dir_arg}")
            return 1

        if not agents_dir.is_dir():
            logger.error(f"Path is not a directory: {agents_dir_arg}")
            return 1
    else:
        # Default to .claude/agents/ in current or parent directories
        current = Path.cwd()
        agents_dir = current / ".claude" / "agents"

        if not agents_dir.exists():
            # Try parent directories
            for parent in current.parents:
                test_dir = parent / ".claude" / "agents"
                if test_dir.exists():
                    agents_dir = test_dir
                    break

    print(f"Validating agent configurations in: {agents_dir}")

    validator = AgentConfigValidator(agents_dir)
    results = validator.validate_all()

    if not results:
        print("No agent files found to validate")
        return 1

    validator.print_summary()

    # Return non-zero if any validations failed
    failed = sum(1 for r in results if not r.is_valid)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
