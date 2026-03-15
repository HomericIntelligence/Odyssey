#!/usr/bin/env python3
"""
Tool: paper-scaffold/scaffold_enhanced.py
Purpose: Enhanced directory generator with validation and comprehensive reporting

Implements Issues #744-763 (Directory Generator) with:
- Idempotent directory creation (#744)
- Template-based file generation (#749)
- Comprehensive validation (#754)

Language: Python
Justification: File I/O, template processing, subprocess limitations in Mojo
Reference: ADR-001
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from validate import validate_paper_structure, ValidationReport, ValidationStatus


@dataclass
class CreationResult:
    """Result of directory creation operation."""

    success: bool
    created_dirs: List[Path] = field(default_factory=list)
    created_files: List[Path] = field(default_factory=list)
    skipped_dirs: List[Path] = field(default_factory=list)
    skipped_files: List[Path] = field(default_factory=list)
    errors: List[Tuple[Path, str]] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("GENERATION SUMMARY")
        lines.append("=" * 60)

        if self.created_dirs:
            lines.append(f"\n✓ Created {len(self.created_dirs)} directories:")
            for dir_path in self.created_dirs:
                lines.append(f"  - {dir_path}")

        if self.skipped_dirs:
            lines.append(f"\n⚠ Skipped {len(self.skipped_dirs)} existing directories:")
            for dir_path in self.skipped_dirs:
                lines.append(f"  - {dir_path}")

        if self.created_files:
            lines.append(f"\n✓ Created {len(self.created_files)} files:")
            for file_path in self.created_files:
                lines.append(f"  - {file_path}")

        if self.skipped_files:
            lines.append(f"\n⚠ Skipped {len(self.skipped_files)} existing files:")
            for file_path in self.skipped_files:
                lines.append(f"  - {file_path}")

        if self.errors:
            lines.append(f"\n✗ Encountered {len(self.errors)} errors:")
            for path, error in self.errors:
                lines.append(f"  - {path}: {error}")

        if self.validation_report:
            lines.append("\n" + "=" * 60)
            lines.append("VALIDATION RESULTS")
            lines.append("=" * 60)
            lines.append(str(self.validation_report))

        lines.append("\n" + "=" * 60)
        if self.success and (not self.validation_report or self.validation_report.status == ValidationStatus.PASS):
            lines.append("✓ SUCCESS: Paper structure generated and validated")
        elif self.success:
            lines.append("⚠ PARTIAL SUCCESS: Structure generated but validation found issues")
        else:
            lines.append("✗ FAILED: Could not complete generation")
        lines.append("=" * 60)

        return "\n".join(lines)


def normalize_paper_name(name: str) -> str:
    """
    Normalize paper name to valid directory name.

    Follows specification from Issue #744:
    - Convert to lowercase
    - Replace spaces/special chars with hyphens
    - Remove consecutive hyphens
    - Trim leading/trailing hyphens

    Args:
        name: Original paper name

    Returns:
        Normalized directory name

    Examples:
        >>> normalize_paper_name("LeNet-5")
        'lenet-5'
        >>> normalize_paper_name("BERT: Pre-training")
        'bert-pre-training'
    """
    import re

    # Convert to lowercase
    name = name.lower()

    # Replace special characters and spaces with hyphens
    name = re.sub(r"[^\w\s-]", "-", name)
    name = re.sub(r"[\s_]+", "-", name)

    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)

    # Trim leading/trailing hyphens
    name = name.strip("-")

    return name


def render_template(template_content: str, variables: Dict[str, str]) -> str:
    """
    Render template by replacing {{VARIABLE}} placeholders.

    Args:
        template_content: Template string with {{VAR}} placeholders
        variables: Dictionary of variable replacements

    Returns:
        Rendered template string
    """
    result = template_content
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, value)
    return result


def load_template(template_path: Path) -> str:
    """
    Load template from file.

    Args:
        template_path: Path to template file

    Returns:
        Template content

    Raises:
        FileNotFoundError: If template doesn't exist
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


class DirectoryGenerator:
    """
    Enhanced directory generator with validation.

    Implements the three-stage pipeline from Issue #759:
    1. Create Structure
    2. Generate Files
    3. Validate Output
    """

    def __init__(self, base_path: Path, verbose: bool = True):
        """
        Initialize directory generator.

        Args:
            base_path: Base directory for paper (e.g., papers/)
            verbose: Enable verbose output
        """
        self.base_path = base_path
        self.verbose = verbose

    def generate(
        self,
        paper_name: str,
        paper_metadata: Dict[str, str],
        templates_dir: Optional[Path] = None,
        validate: bool = True,
        dry_run: bool = False,
    ) -> CreationResult:
        """
        Generate complete paper structure with validation.

        Args:
            paper_name: Original paper name (will be normalized)
            paper_metadata: Metadata for template rendering
            templates_dir: Directory containing templates (auto-detect if None)
            validate: Run validation after generation
            dry_run: Report what would be done without creating anything

        Returns:
            CreationResult with status and details
        """
        result = CreationResult(success=True)

        # Normalize paper name
        normalized_name = normalize_paper_name(paper_name)
        paper_path = self.base_path / normalized_name

        if self.verbose:
            print(f"Generating paper structure: {paper_name} → {normalized_name}")
            if dry_run:
                print("DRY RUN MODE - No files will be created")

        # Stage 1: Create directory structure
        try:
            self._create_structure(paper_path, result, dry_run)
        except Exception as e:
            result.success = False
            result.errors.append((paper_path, f"Structure creation failed: {e}"))
            return result

        # Stage 2: Generate files from templates
        try:
            self._generate_files(
                paper_path,
                normalized_name,
                paper_metadata,
                templates_dir,
                result,
                dry_run,
            )
        except Exception as e:
            result.success = False
            result.errors.append((paper_path, f"File generation failed: {e}"))
            return result

        # Stage 3: Validate output
        if validate and not dry_run:
            if self.verbose:
                print("\nValidating generated structure...")
            result.validation_report = validate_paper_structure(paper_path)

        return result

    def _create_structure(self, paper_path: Path, result: CreationResult, dry_run: bool) -> None:
        """
        Create directory structure (Stage 1).

        Uses os.makedirs(exist_ok=True) for idempotent operations per Issue #744.
        """
        if self.verbose:
            print("\nCreating directory structure...")

        # Directory structure based on papers/_template/
        directories = [
            paper_path,
            paper_path / "src",
            paper_path / "tests",
            paper_path / "scripts",
            paper_path / "configs",
            paper_path / "data",
            paper_path / "data" / "raw",
            paper_path / "data" / "processed",
            paper_path / "data" / "cache",
            paper_path / "notebooks",
            paper_path / "examples",
        ]

        for dir_path in directories:
            if dir_path.exists():
                result.skipped_dirs.append(dir_path)
                if self.verbose:
                    print(f"  ⚠ {dir_path} (already exists)")
            else:
                if not dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                result.created_dirs.append(dir_path)
                if self.verbose:
                    print(f"  ✓ {dir_path}")

    def _generate_files(
        self,
        paper_path: Path,
        paper_name: str,
        metadata: Dict[str, str],
        templates_dir: Optional[Path],
        result: CreationResult,
        dry_run: bool,
    ) -> None:
        """
        Generate files from templates (Stage 2).

        Implements Issue #749 specifications.
        """
        if self.verbose:
            print("\nGenerating files from templates...")

        # Auto-detect templates directory if not specified
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        # Prepare template variables
        model_name = "".join(word.capitalize() for word in paper_name.split("-"))
        variables = {
            "PAPER_NAME": paper_name,
            "MODEL_NAME": model_name,
            **metadata,  # Include all user-provided metadata
        }

        # Files to generate (template -> output path)
        template_files = {
            "README.md.tmpl": paper_path / "README.md",
            "model.mojo.tmpl": paper_path / "src" / "model.mojo",
            "train.mojo.tmpl": paper_path / "src" / "train.mojo",
            "test_model.mojo.tmpl": paper_path / "tests" / "test_model.mojo",
        }

        for template_name, output_path in template_files.items():
            try:
                template_path = templates_dir / template_name

                if not template_path.exists():
                    if self.verbose:
                        print(f"  ⚠ Template not found: {template_name} (skipping)")
                    continue

                # Check if file already exists
                if output_path.exists():
                    result.skipped_files.append(output_path)
                    if self.verbose:
                        print(f"  ⚠ {output_path.relative_to(paper_path)} (already exists)")
                    continue

                # Load and render template
                template_content = load_template(template_path)
                rendered_content = render_template(template_content, variables)

                # Write output file
                if not dry_run:
                    output_path.write_text(rendered_content, encoding="utf-8")

                result.created_files.append(output_path)
                if self.verbose:
                    print(f"  ✓ {output_path.relative_to(paper_path)}")

            except Exception as e:
                result.errors.append((output_path, str(e)))
                if self.verbose:
                    print(f"  ✗ {output_path.relative_to(paper_path)}: {e}")


def main() -> int:
    """Main CLI interface with enhanced features."""
    parser = argparse.ArgumentParser(
        description="Generate paper implementation directory structure with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for all metadata)
  python scaffold_enhanced.py

  # Interactive mode with some pre-filled values
  python scaffold_enhanced.py --paper "LeNet-5"

  # Non-interactive mode (all arguments provided)
  python scaffold_enhanced.py \\
    --paper "LeNet-5" \\
    --title "LeNet-5: Gradient-Based Learning" \\
    --authors "LeCun et al." \\
    --year 1998

  # Force interactive mode even with arguments
  python scaffold_enhanced.py --interactive --paper "BERT"

  # Dry run to see what would be created
  python scaffold_enhanced.py --paper "BERT" --dry-run

  # Skip validation
  python scaffold_enhanced.py --paper "GPT-2" --no-validate
        """,
    )

    parser.add_argument(
        "--paper",
        required=False,
        default=None,
        help="Paper name (will be normalized to lowercase-with-hyphens)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode (prompt for all metadata)",
    )
    parser.add_argument("--title", default="TODO: Add paper title", help="Full paper title")
    parser.add_argument("--authors", default="TODO: Add authors", help="Paper authors")
    parser.add_argument("--year", default="TODO", help="Publication year")
    parser.add_argument("--url", default="TODO: Add paper URL", help="URL to original paper")
    parser.add_argument("--description", default="TODO: Add description", help="Brief paper description")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("papers"),
        help="Output directory (default: papers/)",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Templates directory (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating it",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip validation after generation")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    try:
        # Check if interactive mode is needed
        needs_interactive = (
            args.interactive or not args.paper or args.title.startswith("TODO") or args.authors.startswith("TODO")
        )

        # Use interactive prompts if needed
        if needs_interactive:
            from prompts import InteractivePrompter

            prompter = InteractivePrompter(quiet=args.quiet)

            # Prepare existing args for prompter
            existing_args = {
                "paper": args.paper,
                "title": args.title,
                "authors": args.authors,
                "year": args.year,
                "url": args.url,
                "description": args.description,
            }

            # Collect metadata interactively
            metadata = prompter.collect_metadata(existing=existing_args)

            # Extract paper name from metadata
            paper_name = metadata["PAPER_NAME"]
        else:
            # Use CLI arguments directly
            paper_name = args.paper
            metadata = {
                "PAPER_TITLE": args.title,
                "AUTHORS": args.authors,
                "YEAR": args.year,
                "PAPER_URL": args.url,
                "DESCRIPTION": args.description,
            }

        # Create generator
        generator = DirectoryGenerator(base_path=args.output, verbose=not args.quiet)

        # Generate structure
        result = generator.generate(
            paper_name=paper_name,
            paper_metadata=metadata,
            templates_dir=args.templates,
            validate=not args.no_validate,
            dry_run=args.dry_run,
        )

        # Print summary
        print("\n" + result.summary())

        # Return exit code
        if not result.success:
            return 1
        if result.validation_report and result.validation_report.status != ValidationStatus.PASS:
            return 2  # Validation failures
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
