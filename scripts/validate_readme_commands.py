#!/usr/bin/env python3
"""
README Command Validation Script

Extracts and validates commands from README.md code blocks to ensure
documented commands actually work.

Usage:
    python scripts/validate_readme_commands.py [--level quick|comprehensive] README.md

Validation Levels:
    quick:         Syntax check and binary availability (nightly)
    comprehensive: Full command execution with timeout (weekly)

Exit codes:
    0: All validations passed
    1: One or more validation failures
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from hephaestus.utils import get_repo_root


# Command classification by language tag
EXECUTE_LANGUAGES = {"bash", "shell", "sh"}
SKIP_LANGUAGES = {"text", "plaintext", "output", "console", "markdown", ""}
SYNTAX_CHECK_LANGUAGES = {"mojo", "python"}

# Skip markers - commands with these comments are not executed
SKIP_MARKERS = ["# SKIP-VALIDATION", "# OPTIONAL", "# EXAMPLE"]

# Safety: blocked patterns (never execute)
BLOCKED_PATTERNS = [
    r"\brm\s",
    r"\bmv\s",
    r"\bcp\s",
    r">",
    r">>",
    r"\bgit\s+(commit|push|checkout|reset)",
    r"\bsudo\b",
    r"\bpip\s+install",
    r"\bnpm\s+install",
    r"\bcurl\s+.*\|\s*(bash|sh)",  # Pipe to shell
]

# Safety: allowed command prefixes (execute only these)
ALLOWED_PREFIXES = [
    "uv run",
    "uv sync",
    "uv lock",
    "uv run mojo build",
    "uv run mojo test",
    "uv run mojo format",
    "uv run mojo --version",
    "just precommit",
    "uv run pre-commit install",
    "python3 -m py_compile",
    "python3 --version",
    "gh auth status",
    "gh issue list",
    "gh issue view",
    "gh pr list",
    "gh pr view",
    "echo",
    "cat",
    "ls",
    "pwd",
    "which",
]


@dataclass
class CodeBlock:
    """Represents a fenced code block from markdown."""

    language: str
    content: str
    line_number: int

    def commands(self) -> List[str]:
        """Extract individual commands from the code block."""
        lines = []
        for line in self.content.strip().split("\n"):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Skip continuation lines (handled with previous)
            if line.startswith("\\"):
                continue
            lines.append(line)
        return lines

    def has_skip_marker(self) -> bool:
        """Check if block contains a skip marker."""
        return any(marker in self.content for marker in SKIP_MARKERS)


@dataclass
class ValidationResult:
    """Result of validating a command."""

    command: str
    passed: bool
    check_type: str  # "syntax", "availability", "execution"
    error_message: Optional[str] = None
    line_number: int = 0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class ValidationReport:
    """Full validation report."""

    level: str
    timestamp: str
    total_blocks: int = 0
    total_commands: int = 0
    skipped_commands: int = 0
    passed: int = 0
    failed: int = 0
    results: List[ValidationResult] = field(default_factory=list)


def extract_code_blocks(markdown_path: Path) -> List[CodeBlock]:
    """Extract fenced code blocks from markdown file.

    Args:
        markdown_path: Path to markdown file

    Returns:
        List of CodeBlock objects
    """
    content = markdown_path.read_text()
    blocks = []

    # Match fenced code blocks: ```language\n...\n```
    pattern = r"^```(\w*)\n(.*?)^```"
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

    for match in matches:
        language = match.group(1).lower()
        block_content = match.group(2)

        # Calculate line number
        line_number = content[: match.start()].count("\n") + 1

        blocks.append(CodeBlock(language=language, content=block_content, line_number=line_number))

    return blocks


def is_blocked_command(command: str) -> bool:
    """Check if command matches blocked patterns.

    Args:
        command: Command string to check

    Returns:
        True if command is blocked
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return True
    return False


def is_allowed_command(command: str) -> bool:
    """Check if command starts with an allowed prefix.

    Args:
        command: Command string to check

    Returns:
        True if command is allowed
    """
    for prefix in ALLOWED_PREFIXES:
        if command.startswith(prefix):
            return True
    return False


def is_safe_command(command: str) -> tuple[bool, str]:
    """Check if command is safe to execute.

    Args:
        command: Command string to check

    Returns:
        Tuple of (is_safe, reason).
    """
    if is_blocked_command(command):
        return False, "matches blocked pattern"

    if not is_allowed_command(command):
        return False, "not in allowed prefixes"

    return True, "allowed"


def get_binary_from_command(command: str) -> str:
    """Extract the binary/executable from a command.

    Args:
        command: Command string

    Returns:
        Binary name
    """
    parts = command.split()
    if not parts:
        return ""
    return parts[0]


def validate_syntax(command: str) -> ValidationResult:
    """Validate bash syntax of a command.

    Args:
        command: Command string to validate

    Returns:
        ValidationResult
    """
    try:
        result = subprocess.run(
            ["bash", "-n", "-c", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return ValidationResult(
            command=command,
            passed=result.returncode == 0,
            check_type="syntax",
            error_message=result.stderr if result.returncode != 0 else None,
            exit_code=result.returncode,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            command=command,
            passed=False,
            check_type="syntax",
            error_message="Syntax check timed out",
        )
    except Exception as e:
        return ValidationResult(
            command=command,
            passed=False,
            check_type="syntax",
            error_message=str(e),
        )


def validate_availability(command: str) -> ValidationResult:
    """Check if command binary is available.

    Args:
        command: Command string to check

    Returns:
        ValidationResult
    """
    binary = get_binary_from_command(command)
    if not binary:
        return ValidationResult(
            command=command,
            passed=False,
            check_type="availability",
            error_message="Could not extract binary from command",
        )

    found = shutil.which(binary) is not None
    return ValidationResult(
        command=command,
        passed=found,
        check_type="availability",
        error_message=f"Binary not found: {binary}" if not found else None,
    )


def validate_execution(command: str, timeout: int = 60) -> ValidationResult:
    """Execute command and validate it succeeds.

    Args:
        command: Command string to execute
        timeout: Timeout in seconds

    Returns:
        ValidationResult
    """
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=get_repo_root(),  # Run from repo root
        )
        return ValidationResult(
            command=command,
            passed=result.returncode == 0,
            check_type="execution",
            error_message=result.stderr if result.returncode != 0 else None,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            command=command,
            passed=False,
            check_type="execution",
            error_message=f"Command timed out after {timeout}s",
        )
    except Exception as e:
        return ValidationResult(
            command=command,
            passed=False,
            check_type="execution",
            error_message=str(e),
        )


def validate_quick(blocks: List[CodeBlock]) -> ValidationReport:
    """Quick validation: syntax and availability checks.

    Args:
        blocks: List of code blocks to validate

    Returns:
        ValidationReport
    """
    report = ValidationReport(
        level="quick",
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_blocks=len(blocks),
    )

    for block in blocks:
        # Skip non-executable blocks
        if block.language not in EXECUTE_LANGUAGES:
            continue

        # Skip blocks with skip markers
        if block.has_skip_marker():
            report.skipped_commands += len(block.commands())
            continue

        for command in block.commands():
            report.total_commands += 1

            # Check safety
            is_safe, reason = is_safe_command(command)
            if not is_safe:
                report.skipped_commands += 1
                continue

            # Syntax check
            syntax_result = validate_syntax(command)
            syntax_result.line_number = block.line_number
            report.results.append(syntax_result)

            if syntax_result.passed:
                # Availability check
                avail_result = validate_availability(command)
                avail_result.line_number = block.line_number
                report.results.append(avail_result)

                if avail_result.passed:
                    report.passed += 1
                else:
                    report.failed += 1
            else:
                report.failed += 1

    return report


def validate_comprehensive(blocks: List[CodeBlock]) -> ValidationReport:
    """Comprehensive validation: full command execution.

    Args:
        blocks: List of code blocks to validate

    Returns:
        ValidationReport
    """
    report = ValidationReport(
        level="comprehensive",
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_blocks=len(blocks),
    )

    for block in blocks:
        # Skip non-executable blocks
        if block.language not in EXECUTE_LANGUAGES:
            continue

        # Skip blocks with skip markers
        if block.has_skip_marker():
            report.skipped_commands += len(block.commands())
            continue

        for command in block.commands():
            report.total_commands += 1

            # Check safety
            is_safe, reason = is_safe_command(command)
            if not is_safe:
                report.skipped_commands += 1
                continue

            # Full execution
            exec_result = validate_execution(command)
            exec_result.line_number = block.line_number
            report.results.append(exec_result)

            if exec_result.passed:
                report.passed += 1
            else:
                report.failed += 1

    return report


def generate_report(report: ValidationReport, output_path: Path) -> None:
    """Generate markdown validation report.

    Args:
        report: ValidationReport to format
        output_path: Path to write report
    """
    lines = [
        "# README.md Command Validation Results",
        "",
        f"**Validation Level**: {report.level.title()}",
        f"**Timestamp**: {report.timestamp} UTC",
        "",
        "## Summary",
        "",
        f"- Total code blocks: {report.total_blocks}",
        f"- Total commands found: {report.total_commands}",
        f"- Commands validated: {report.passed + report.failed}",
        f"- Commands skipped: {report.skipped_commands}",
        f"- **Passed**: {report.passed}",
        f"- **Failed**: {report.failed}",
        "",
    ]

    # Failed commands section
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.extend(["## Failed Commands", ""])
        for i, result in enumerate(failed, 1):
            lines.extend(
                [
                    f"### {i}. {result.check_type.title()} Failure (line {result.line_number})",
                    "",
                    "```bash",
                    result.command,
                    "```",
                    "",
                    f"**Error**: {result.error_message}",
                    "",
                ]
            )
            if result.stderr:
                lines.extend(
                    [
                        "**Stderr**:",
                        "```",
                        result.stderr[:500],  # Truncate long output
                        "```",
                        "",
                    ]
                )
    else:
        lines.extend(["## All Commands Passed!", ""])

    # Write report
    output_path.write_text("\n".join(lines))


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="Validate README.md commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "readme",
        type=Path,
        help="Path to README.md file",
    )
    parser.add_argument(
        "--level",
        choices=["quick", "comprehensive"],
        default="quick",
        help="Validation level (default: quick)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation-report.md"),
        help="Output report path (default: validation-report.md)",
    )

    args = parser.parse_args()

    if not args.readme.exists():
        print(f"Error: README file not found: {args.readme}", file=sys.stderr)
        return 1

    # Extract code blocks
    print(f"Extracting code blocks from {args.readme}...")
    blocks = extract_code_blocks(args.readme)
    print(f"Found {len(blocks)} code blocks")

    # Filter to executable blocks
    executable_blocks = [b for b in blocks if b.language in EXECUTE_LANGUAGES]
    print(f"Found {len(executable_blocks)} executable blocks (bash/shell/sh)")

    # Run validation
    print(f"Running {args.level} validation...")
    if args.level == "quick":
        report = validate_quick(blocks)
    else:
        report = validate_comprehensive(blocks)

    # Generate report
    generate_report(report, args.output)
    print(f"Report written to {args.output}")

    # Summary
    print()
    print("=" * 50)
    print(f"Validation Level: {report.level.title()}")
    print(f"Commands found: {report.total_commands}")
    print(f"Commands validated: {report.passed + report.failed}")
    print(f"Commands skipped: {report.skipped_commands}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print("=" * 50)

    if report.failed > 0:
        print()
        print("VALIDATION FAILED - see report for details")
        return 1

    print()
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
