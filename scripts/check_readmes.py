#!/usr/bin/env python3
"""
README completeness validation script for ML Odyssey

Validates that all READMEs have required sections and follow markdown standards.

Usage:
    python scripts/check_readmes.py [--directory DIR] [--verbose]

Exit codes:
    0: All validations passed
    1: One or more validation failures
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hephaestus.utils import get_repo_root

# Required sections for different README types
REQUIRED_SECTIONS = {
    "default": [
        "Overview",
        "Quick Start",
        "Usage",
    ],
    "directory": [
        "Overview",
        "Structure",
        "Usage",
    ],
    "tool": [
        "Overview",
        "Installation",
        "Usage",
        "Examples",
    ],
}

# Markdown linting checks
MARKDOWN_CHECKS = [
    ("code_blocks_language", r"```\n", "Code blocks must specify language"),
    ("blank_lines_lists", r"[^\n]\n[-*]", "Lists must have blank line before"),
    ("blank_lines_headings", r"[^\n]\n#{1,6} ", "Headings must have blank line before"),
]


def find_readmes(directory: Path) -> List[Path]:
    """Find all README.md files in directory tree"""
    return list(directory.rglob("README.md"))


def extract_sections(content: str) -> List[str]:
    """Extract section headings from markdown content"""
    # Match markdown headings (## Section Name)
    heading_pattern = r"^#{1,6}\s+(.+)$"
    sections = []

    for line in content.split("\n"):
        match = re.match(heading_pattern, line)
        if match:
            sections.append(match.group(1).strip())

    return sections


def check_required_sections(readme_path: Path, sections: List[str]) -> Tuple[bool, List[str]]:
    """Check if README has required sections"""
    # Determine README type based on location
    readme_type = "default"
    parent_dir = readme_path.parent.name

    if parent_dir in [
        "benchmarks",
        "docs",
        "agents",
        "tools",
        "configs",
        "shared",
        "tests",
    ]:
        readme_type = "directory"
    elif readme_path.parent.parent.name == "tools":
        readme_type = "tool"

    required = REQUIRED_SECTIONS.get(readme_type, REQUIRED_SECTIONS["default"])

    missing = []
    for section in required:
        # Check for exact match or case-insensitive partial match
        found = any(section.lower() in existing.lower() for existing in sections)
        if not found:
            missing.append(section)

    return len(missing) == 0, missing


def check_markdown_formatting(content: str) -> List[str]:
    """Check markdown formatting issues"""
    issues = []

    # Check for code blocks without language
    if re.search(r"```\s*\n", content):
        issues.append("Code blocks missing language specification")

    # Check for lists without blank lines (simplified check)
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and line.strip().startswith(("-", "*", "1.")):
            prev_line = lines[i - 1].strip()
            if prev_line and not prev_line.startswith(("#", "-", "*", "1.")):
                issues.append(f"Line {i + 1}: List without blank line before")
                break  # Only report first occurrence

    # Check for headings without blank lines (simplified check)
    for i, line in enumerate(lines):
        if i > 0 and line.strip().startswith("#"):
            prev_line = lines[i - 1].strip()
            if prev_line and not prev_line.startswith("#"):
                issues.append(f"Line {i + 1}: Heading without blank line before")
                break  # Only report first occurrence

    return issues


def validate_readme(readme_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate a single README file

    Returns:
        Dictionary with validation results
    """
    result: Dict[str, Any] = {
        "path": str(readme_path),
        "passed": True,
        "issues": [],
    }

    try:
        content = readme_path.read_text()

        # Extract sections
        sections = extract_sections(content)

        # Check required sections
        sections_ok, missing = check_required_sections(readme_path, sections)
        if not sections_ok:
            result["passed"] = False
            result["issues"].append(f"Missing sections: {', '.join(missing)}")

        # Check markdown formatting
        formatting_issues = check_markdown_formatting(content)
        if formatting_issues:
            result["passed"] = False
            result["issues"].extend(formatting_issues)

        # Check file ends with newline
        if not content.endswith("\n"):
            result["passed"] = False
            result["issues"].append("File must end with newline")

    except Exception as e:
        result["passed"] = False
        result["issues"].append(f"Error reading file: {str(e)}")

    return result


def validate_all_readmes(directory: Path, verbose: bool = False) -> Dict[str, Any]:
    """Validate all READMEs in directory tree"""
    results: Dict[str, Any] = {"passed": [], "failed": []}

    readmes = find_readmes(directory)

    if not readmes:
        print(f"No README.md files found in {directory}")
        return results

    print(f"Found {len(readmes)} README files\n")

    for readme_path in readmes:
        relative_path = readme_path.relative_to(directory)
        result = validate_readme(readme_path, verbose)

        if result["passed"]:
            results["passed"].append(str(relative_path))
            if verbose:
                print(f"✓ {relative_path}")
        else:
            results["failed"].append(result)
            print(f"✗ {relative_path}")
            for issue in result["issues"]:
                print(f"    - {issue}")

    return results


def print_summary(results: Dict[str, List]) -> None:
    """Print validation summary"""
    total = len(results["passed"]) + len(results["failed"])
    passed = len(results["passed"])
    failed = len(results["failed"])

    print("\n" + "=" * 70)
    print("README VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total READMEs: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\nFailed READMEs ({failed}):")
        for result in results["failed"]:
            print(f"  {result['path']}")
            for issue in result["issues"]:
                print(f"    - {issue}")

    print("=" * 70)


def main() -> int:
    """Main validation function"""
    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Get directory to check
    directory = None
    for i, arg in enumerate(sys.argv):
        if arg in ["--directory", "-d"] and i + 1 < len(sys.argv):
            directory = Path(sys.argv[i + 1])
            break

    if directory is None:
        directory = get_repo_root()

    print(f"Validating READMEs in: {directory}\n")

    results = validate_all_readmes(directory, verbose)
    print_summary(results)

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
