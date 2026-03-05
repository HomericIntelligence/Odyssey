#!/usr/bin/env python3
"""
DEPRECATED: This script is no longer used.

The notes/plan/ directory has been removed. Planning is now done directly
through GitHub issues. See .claude/shared/github-issue-workflow.md for the
new workflow.

To update issues, use the GitHub CLI directly:
    gh issue edit <number> --title "..." --body "..."

---

Regenerate all github_issue.md files from their corresponding plan.md files.
(DEPRECATED)

This script consolidates the functionality of all legacy issue update scripts
and can regenerate all github_issue.md files dynamically from plan.md sources.
"""

import sys
from pathlib import Path
import re
from datetime import datetime
import json
import argparse


def read_plan_file(plan_path):
    """Read and parse a plan.md file to extract all sections."""
    with open(plan_path, "r") as f:
        content = f.read()

    sections = {}

    # Extract title (first h1)
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    sections["title"] = title_match.group(1) if title_match else "Unknown"

    # Extract overview
    overview_match = re.search(r"## Overview\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["overview"] = overview_match.group(1).strip() if overview_match else ""

    # Extract inputs
    inputs_match = re.search(r"## Inputs\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["inputs"] = inputs_match.group(1).strip() if inputs_match else ""

    # Extract outputs
    outputs_match = re.search(r"## Outputs\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["outputs"] = outputs_match.group(1).strip() if outputs_match else ""

    # Extract steps
    steps_match = re.search(r"## Steps\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["steps"] = steps_match.group(1).strip() if steps_match else ""

    # Extract success criteria
    criteria_match = re.search(r"## Success Criteria\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["success_criteria"] = criteria_match.group(1).strip() if criteria_match else ""

    # Extract notes
    notes_match = re.search(r"## Notes\n(.+?)(?=\n## |$)", content, re.DOTALL)
    sections["notes"] = notes_match.group(1).strip() if notes_match else ""

    return sections


def generate_plan_body(sections):
    """Generate the Plan issue body."""
    body = f"""## Overview
{sections["overview"]}

## Objectives
This planning phase will:
- Define detailed specifications and requirements
- Design the architecture and approach
- Document API contracts and interfaces
- Create comprehensive design documentation

## Inputs
{sections["inputs"]}

## Expected Outputs
{sections["outputs"]}

## Success Criteria
{sections["success_criteria"]}

## Additional Notes
{sections["notes"]}"""
    return body


def generate_test_body(sections):
    """Generate the Test issue body."""
    body = f"""## Overview
{sections["overview"]}

## Testing Objectives
This phase focuses on:
- Writing comprehensive test cases following TDD principles
- Creating test fixtures and mock data
- Defining test scenarios for edge cases
- Setting up test infrastructure

## What to Test
Based on the expected outputs:
{sections["outputs"]}

## Test Success Criteria
{sections["success_criteria"]}

## Implementation Steps
{sections["steps"]}

## Notes
{sections["notes"]}"""
    return body


def generate_implementation_body(sections):
    """Generate the Implementation issue body."""
    body = f"""## Overview
{sections["overview"]}

## Implementation Goals
- Implement the functionality to pass all tests
- Follow Mojo best practices and coding standards
- Ensure code is clean, documented, and maintainable
- Meet all requirements specified in the plan

## Required Inputs
{sections["inputs"]}

## Expected Outputs
{sections["outputs"]}

## Implementation Steps
{sections["steps"]}

## Success Criteria
{sections["success_criteria"]}

## Notes
{sections["notes"]}"""
    return body


def generate_packaging_body(sections):
    """Generate the Packaging issue body."""
    body = f"""## Overview
{sections["overview"]}

## Packaging Objectives
- Integrate the implementation with existing codebase
- Ensure all dependencies are properly configured
- Verify compatibility with other components
- Package for deployment/distribution

## Integration Requirements
Based on outputs:
{sections["outputs"]}

## Integration Steps
{sections["steps"]}

## Success Criteria
{sections["success_criteria"]}

## Notes
{sections["notes"]}"""
    return body


def generate_cleanup_body(sections):
    """Generate the Cleanup issue body."""
    body = f"""## Overview
{sections["overview"]}

## Cleanup Objectives
- Refactor code for optimal quality and maintainability
- Remove technical debt and temporary workarounds
- Ensure comprehensive documentation
- Perform final validation and optimization

## Cleanup Tasks
- Code review and refactoring
- Documentation finalization
- Performance optimization
- Final testing and validation

## Success Criteria
{sections["success_criteria"]}

## Notes
{sections["notes"]}"""
    return body


def generate_github_issue_content(plan_sections):
    """Generate the complete github_issue.md file content."""
    title = plan_sections["title"]

    # Generate all issue bodies
    plan_body = generate_plan_body(plan_sections)
    test_body = generate_test_body(plan_sections)
    impl_body = generate_implementation_body(plan_sections)
    package_body = generate_packaging_body(plan_sections)
    cleanup_body = generate_cleanup_body(plan_sections)

    # Create the complete github_issue.md content
    content = f"""# GitHub Issues

**Plan Issue**:
- Title: [Plan] {title} - Design and Documentation
- Body:
```
{plan_body}
```
- Labels: planning, documentation
- URL: [to be filled]

**Test Issue**:
- Title: [Test] {title} - Write Tests
- Body:
```
{test_body}
```
- Labels: testing, tdd
- URL: [to be filled]

**Implementation Issue**:
- Title: [Impl] {title} - Implementation
- Body:
```
{impl_body}
```
- Labels: implementation
- URL: [to be filled]

**Packaging Issue**:
- Title: [Package] {title} - Integration and Packaging
- Body:
```
{package_body}
```
- Labels: packaging, integration
- URL: [to be filled]

**Cleanup Issue**:
- Title: [Cleanup] {title} - Refactor and Finalize
- Body:
```
{cleanup_body}
```
- Labels: cleanup, documentation
- URL: [to be filled]
"""
    return content


def save_state(state_data, logs_dir):
    """Save processing state to logs directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = logs_dir / f".issue_creation_state_{timestamp}.json"

    with open(state_file, "w") as f:
        json.dump(state_data, f, indent=2)

    return state_file


def load_latest_state(logs_dir):
    """Load the most recent state file from logs directory."""
    state_files = sorted(logs_dir.glob(".issue_creation_state_*.json"))

    if not state_files:
        return None

    latest_state_file = state_files[-1]

    try:
        with open(latest_state_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(
            f"Warning: Could not load state file {latest_state_file}: {e}",
            file=sys.stderr,
        )
        return None


def process_plan_directory(plan_dir, section=None, dry_run=False, resume=False):
    """
    Process all plan.md files and generate corresponding github_issue.md files.

    Args:
        plan_dir: Path to notes/plan directory
        section: Optional specific section to process (e.g., '01-foundation')
        dry_run: If True, show what would be done without making changes
        resume: If True, attempt to resume from last saved state

    Returns:
        Tuple of (success_count, error_count, errors_list).
    """
    plan_path = Path(plan_dir)
    logs_dir = plan_path.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Load resume state if requested
    processed_files = set()
    if resume:
        state = load_latest_state(logs_dir)
        if state:
            processed_files = set(state.get("processed", []))
            print(
                f"Resuming from previous state: {len(processed_files)} files already processed",
                file=sys.stderr,
            )

    # Find all plan.md files
    if section:
        pattern = f"{section}/**/plan.md"
        plan_files = sorted(plan_path.glob(pattern))
    else:
        plan_files = sorted(plan_path.rglob("plan.md"))

    total = len(plan_files)
    success_count = 0
    error_count = 0
    errors = []

    print(f"Found {total} plan.md files to process", file=sys.stderr)

    if dry_run:
        print("DRY RUN MODE - No files will be modified", file=sys.stderr)

    print("=" * 80, file=sys.stderr)

    for i, plan_file in enumerate(plan_files, 1):
        plan_file_str = str(plan_file)

        # Skip if already processed (resume mode)
        if plan_file_str in processed_files:
            continue

        issue_file = plan_file.parent / "github_issue.md"

        try:
            # Read and parse plan
            plan_sections = read_plan_file(plan_file)

            # Generate github_issue.md content
            issue_content = generate_github_issue_content(plan_sections)

            if dry_run:
                print(f"[{i}/{total}] Would update: {issue_file}", file=sys.stderr)
            else:
                # Write the file
                with open(issue_file, "w") as f:
                    f.write(issue_content)

                success_count += 1
                processed_files.add(plan_file_str)

                # Show progress
                if success_count <= 5 or i % 50 == 0 or i == total:
                    print(f"[{i}/{total}] Updated: {issue_file}", file=sys.stderr)

                # Save state periodically (every 50 files)
                if success_count % 50 == 0:
                    state_file = save_state({"processed": list(processed_files)}, logs_dir)
                    print(f"State saved to: {state_file}", file=sys.stderr)

        except Exception as e:
            error_count += 1
            error_msg = f"{plan_file}: {str(e)}"
            errors.append(error_msg)
            print(f"[{i}/{total}] ERROR: {error_msg}", file=sys.stderr)

    # Save final state
    if not dry_run and success_count > 0:
        state_file = save_state({"processed": list(processed_files), "completed": True}, logs_dir)
        print(f"\nFinal state saved to: {state_file}", file=sys.stderr)

    return success_count, error_count, errors


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Regenerate all github_issue.md files from plan.md files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run to see what would be changed
  python regenerate_github_issues.py --dry-run

  # Process only one section
  python regenerate_github_issues.py --section 01-foundation

  # Resume from previous run
  python regenerate_github_issues.py --resume

  # Process all files
  python regenerate_github_issues.py
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument("--section", type=str, help="Process only one section (e.g., 01-foundation)")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved state")
    parser.add_argument(
        "--plan-dir",
        type=str,
        default=None,
        help="Path to plan directory (default: auto-detected from repository root)",
    )

    args = parser.parse_args()

    # Get plan directory (use provided path or auto-detect)
    if args.plan_dir:
        plan_dir = Path(args.plan_dir)
    else:
        print("ERROR: This script is DEPRECATED.", file=sys.stderr)
        print("The notes/plan/ directory has been removed.", file=sys.stderr)
        print("Planning is now done directly through GitHub issues.", file=sys.stderr)
        print(
            "See .claude/shared/github-issue-workflow.md for the new workflow.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("To update issues, use the GitHub CLI directly:", file=sys.stderr)
        print("    gh issue edit <number> --title '...' --body '...'", file=sys.stderr)
        return 1

    # Validate plan directory exists
    if not plan_dir.exists():
        print(f"ERROR: Plan directory not found: {plan_dir}", file=sys.stderr)
        return 1

    print(f"Regenerating github_issue.md files from: {plan_dir}", file=sys.stderr)
    if args.section:
        print(f"Section filter: {args.section}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Process files
    success_count, error_count, errors = process_plan_directory(
        args.plan_dir, section=args.section, dry_run=args.dry_run, resume=args.resume
    )

    # Print summary
    print("\n" + "=" * 80, file=sys.stderr)
    print("Summary:", file=sys.stderr)
    print(f"  Successfully processed: {success_count}", file=sys.stderr)
    print(f"  Errors: {error_count}", file=sys.stderr)

    if errors:
        print("\nErrors encountered:", file=sys.stderr)
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)

    if args.dry_run:
        print("\nDRY RUN COMPLETE - No files were modified", file=sys.stderr)
    else:
        print("\nRegeneration complete!", file=sys.stderr)

    # Exit with error code if there were errors
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
