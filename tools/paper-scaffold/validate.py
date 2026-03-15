#!/usr/bin/env python3
"""
Module: paper-scaffold/validate.py
Purpose: Validation logic for generated paper structures

Validates that generated paper directories are complete, correctly formatted,
and follow repository conventions per Issue #754.

Language: Python
Justification: File I/O, regex validation, no performance requirements
Reference: ADR-001
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ValidationStatus(Enum):
    """Status of validation result."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class FileValidationError:
    """Error found in a specific file."""

    file_path: Path
    error_message: str
    line_number: Optional[int] = None
    suggestion: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    status: ValidationStatus
    missing_directories: List[Path] = field(default_factory=list)
    missing_files: List[Path] = field(default_factory=list)
    invalid_files: List[FileValidationError] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format validation report for display."""
        lines = []

        if self.status == ValidationStatus.PASS:
            lines.append("✓ VALIDATION PASSED")
            if self.warnings:
                lines.append("")
                lines.append("Warnings:")
                for warning in self.warnings:
                    lines.append(f"  ⚠ {warning}")
        else:
            lines.append("✗ VALIDATION FAILED")

        if self.missing_directories:
            lines.append("")
            lines.append(f"Missing Directories ({len(self.missing_directories)}):")
            for dir_path in self.missing_directories:
                lines.append(f"  - {dir_path}")

        if self.missing_files:
            lines.append("")
            lines.append(f"Missing Files ({len(self.missing_files)}):")
            for file_path in self.missing_files:
                lines.append(f"  - {file_path}")

        if self.invalid_files:
            lines.append("")
            lines.append(f"Invalid Files ({len(self.invalid_files)}):")
            for error in self.invalid_files:
                lines.append(f"  - {error.file_path}")
                if error.line_number:
                    lines.append(f"    Error at line {error.line_number}: {error.error_message}")
                else:
                    lines.append(f"    Error: {error.error_message}")
                if error.suggestion:
                    lines.append(f"    Suggestion: {error.suggestion}")

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)


class PaperStructureValidator:
    """Validates generated paper structures."""

    # Required directories relative to paper root
    REQUIRED_DIRS = [
        "src",
        "tests",
    ]

    # Required files relative to paper root
    REQUIRED_FILES = [
        "README.md",
    ]

    # Optional but recommended directories
    RECOMMENDED_DIRS = [
        "docs",
        "data",
        "configs",
        "notebooks",
    ]

    def __init__(self, paper_path: Path):
        """
        Initialize validator.

        Args:
            paper_path: Path to the paper directory to validate
        """
        self.paper_path = paper_path

    def validate(self) -> ValidationReport:
        """
        Validate the paper structure.

        Returns:
            ValidationReport with all validation results
        """
        report = ValidationReport(status=ValidationStatus.PASS)

        # Check if paper directory exists
        if not self.paper_path.exists():
            report.status = ValidationStatus.FAIL
            report.suggestions.append(f"Paper directory does not exist: {self.paper_path}")
            return report

        # Validate directory structure
        self._validate_directories(report)

        # Validate required files
        self._validate_files(report)

        # Validate file content
        self._validate_content(report)

        # Set final status
        if report.missing_directories or report.missing_files or report.invalid_files:
            report.status = ValidationStatus.FAIL

        # Add overall suggestions
        if report.status == ValidationStatus.FAIL:
            self._add_fix_suggestions(report)

        return report

    def _validate_directories(self, report: ValidationReport) -> None:
        """Validate directory structure."""
        # Check required directories
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.paper_path / dir_name
            if not dir_path.exists():
                report.missing_directories.append(dir_path)

        # Check recommended directories (warnings only)
        for dir_name in self.RECOMMENDED_DIRS:
            dir_path = self.paper_path / dir_name
            if not dir_path.exists():
                report.warnings.append(f"Recommended directory missing: {dir_name}/")

    def _validate_files(self, report: ValidationReport) -> None:
        """Validate required files exist."""
        for file_name in self.REQUIRED_FILES:
            file_path = self.paper_path / file_name
            if not file_path.exists():
                report.missing_files.append(file_path)

    def _validate_content(self, report: ValidationReport) -> None:
        """Validate file content and format."""
        # Validate README.md structure
        readme_path = self.paper_path / "README.md"
        if readme_path.exists():
            self._validate_readme(readme_path, report)

        # Validate Mojo files syntax (basic check)
        for mojo_file in self.paper_path.rglob("*.mojo"):
            self._validate_mojo_file(mojo_file, report)

    def _validate_readme(self, readme_path: Path, report: ValidationReport) -> None:
        """Validate README.md structure."""
        try:
            content = readme_path.read_text()

            # Check for required sections
            required_sections = ["#", "Overview", "Implementation"]
            for section in required_sections:
                if section.lower() not in content.lower():
                    report.warnings.append(f"README.md missing recommended section: {section}")

            # Check it's not empty
            if len(content.strip()) < 50:
                report.invalid_files.append(
                    FileValidationError(
                        file_path=readme_path,
                        error_message="README.md appears to be empty or too short",
                        suggestion="Add paper overview and implementation details",
                    )
                )

        except Exception as e:
            report.invalid_files.append(
                FileValidationError(
                    file_path=readme_path,
                    error_message=f"Could not read file: {e}",
                    suggestion="Ensure file is valid UTF-8 text",
                )
            )

    def _validate_mojo_file(self, mojo_path: Path, report: ValidationReport) -> None:
        """Basic validation of Mojo files."""
        try:
            content = mojo_path.read_text()

            # Check for common syntax errors
            if content.strip() and not content.strip().startswith("#"):
                # Basic check: Mojo files should have some structure
                if "fn " not in content and "def " not in content and "struct " not in content:
                    report.warnings.append(f"{mojo_path.name}: No function or struct definitions found")

        except Exception as e:
            report.invalid_files.append(
                FileValidationError(
                    file_path=mojo_path,
                    error_message=f"Could not read file: {e}",
                    suggestion="Ensure file is valid UTF-8 text",
                )
            )

    def _add_fix_suggestions(self, report: ValidationReport) -> None:
        """Add actionable fix suggestions to report."""
        if report.missing_directories:
            dirs_to_create = " ".join(str(d) for d in report.missing_directories)
            report.suggestions.append(f"Create missing directories: mkdir -p {dirs_to_create}")

        if report.missing_files:
            if self.paper_path / "README.md" in report.missing_files:
                report.suggestions.append("Create README.md from template: cp papers/_template/README.md .")


def validate_paper_structure(paper_path: Path) -> ValidationReport:
    """
    Validate a generated paper structure.

    Args:
        paper_path: Path to the paper directory

    Returns:
        ValidationReport with validation results
    """
    validator = PaperStructureValidator(paper_path)
    return validator.validate()
