#!/usr/bin/env python3
"""
Validate Test Coverage - Ensure all test_*.mojo files are covered by CI

This script finds all test_*.mojo files in the repository and verifies they are
included in the CI test matrix in .github/workflows/comprehensive-tests.yml.

IMPORTANT: Always grep live test files rather than relying on cached metadata.
Issue #3477 mistakenly stated test_conv.mojo had 15 test functions when the actual
count was 20. This script verifies live file counts, not stale issue metadata.
See Issue #4325 for details on this discrepancy.

Exit codes:
  0 - All tests covered (or only warnings with --warn-stale)
  1 - Uncovered tests found, validation errors, or stale patterns with --error-stale

Usage:
    python scripts/validate_test_coverage.py [options]

Options:
    --post-pr       Post validation report to GitHub PR if tests are missing
    --warn-stale    Treat stale patterns as warnings (default, exit 0)
    --error-stale   Treat stale patterns as errors (exit 1 if found)
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from common import get_repo_root, EXCLUDE_DIRS


def find_test_files(root_dir: Path) -> List[Path]:
    """Find all test_*.mojo files, excluding build artifacts and examples.

    MAINTENANCE NOTE: This function uses hardcoded lists of excluded test files
    (E2E tests, training tests, dataset-dependent tests, etc.). These lists must
    be manually updated whenever test files are:
    - Renamed (update in exclude_files list)
    - Split per ADR-009 (add new part files to list)
    - Added/removed (add/remove from appropriate list)

    This is a known maintenance burden. Alternatives to consider:
    - Switch to glob-based discovery with .gitignore-style pattern matching
    - Use a separate config file (JSON/YAML) for excludes instead of hardcoded list
    - Add a test naming convention to auto-identify excluded tests

    See Issue #4369 for discussion and ADR-009 for test splitting guidelines.
    """
    test_files = []

    # Exclude patterns for directories we don't want to scan
    exclude_patterns = list(EXCLUDE_DIRS) + ["__pycache__/"]

    # Exclude specific test files that require external datasets
    # These tests need datasets/ directory which must be downloaded separately
    # They run in the weekly E2E workflow, not per-PR
    exclude_files = [
        # EMNIST example tests (require dataset download)
        "examples/lenet-emnist/test_gradients.mojo",
        "examples/lenet-emnist/test_loss_decrease.mojo",
        "examples/lenet-emnist/test_predictions.mojo",
        "examples/lenet-emnist/test_training_metrics.mojo",
        "examples/lenet-emnist/test_weight_updates.mojo",
        "examples/googlenet-cifar10/test_model.mojo",
        "examples/mobilenetv1-cifar10/test_model.mojo",
        "examples/resnet18-cifar10/test_model.mojo",
    ]

    # Exclude E2E model tests (run weekly, not per-PR)
    # These test full training loops and require significant runtime
    exclude_e2e_patterns = [
        "tests/models/test_alexnet_e2e.mojo",
        "tests/models/test_googlenet_e2e.mojo",
        "tests/models/test_lenet5_e2e_part1.mojo",
        "tests/models/test_lenet5_e2e_part2.mojo",
        "tests/models/test_googlenet_e2e_part1.mojo",
        "tests/models/test_googlenet_e2e_part2.mojo",
        "tests/models/test_lenet5_e2e.mojo",
        "tests/models/test_mobilenetv1_e2e.mojo",
        "tests/models/test_resnet18_e2e.mojo",
        "tests/models/test_vgg16_e2e.mojo",
        # Heap corruption debugging test (used with git bisect, not regular CI)
        "tests/models/test_heap_corruption_combined.mojo",
    ]
    exclude_files.extend(exclude_e2e_patterns)

    # Exclude training tests (run weekly, require dataset downloads)
    exclude_training_patterns = [
        "tests/shared/training/test_accuracy_bugs.mojo",
        "tests/shared/training/test_base.mojo",
        "tests/shared/training/test_callbacks_part1.mojo",
        "tests/shared/training/test_callbacks_part2.mojo",
        "tests/shared/training/test_callbacks_part3.mojo",
        "tests/shared/training/test_callbacks.mojo",
        "tests/shared/training/test_checkpoint.mojo",
        "tests/shared/training/test_checkpointing.mojo",
        "tests/shared/training/test_config.mojo",
        "tests/shared/training/test_confusion_matrix_bugs.mojo",
        "tests/shared/training/test_csv_metrics_logger.mojo",
        "tests/shared/training/test_dtype_utils.mojo",
        "tests/shared/training/test_early_stopping_part1.mojo",
        "tests/shared/training/test_early_stopping_part2.mojo",
        "tests/shared/training/test_evaluate.mojo",
        "tests/shared/training/test_evaluation_part1.mojo",
        "tests/shared/training/test_evaluation_part2.mojo",
        "tests/shared/training/test_evaluation.mojo",
        "tests/shared/training/test_exponential_scheduler.mojo",
        "tests/shared/training/test_gradient_clipping.mojo",
        "tests/shared/training/test_gradient_ops.mojo",
        "tests/shared/training/test_lars.mojo",
        "tests/shared/training/test_logging_callback_part1.mojo",
        "tests/shared/training/test_logging_callback_part2.mojo",
        "tests/shared/training/test_loops.mojo",
        "tests/shared/training/test_metrics_part1.mojo",
        "tests/shared/training/test_metrics_part2.mojo",
        "tests/shared/training/test_mixed_precision.mojo",
        # Removed: test_mixed_precision_part1.mojo and test_mixed_precision_part2.mojo
        # These files now comply with ADR-009 (≤8 tests each) and run in per-PR CI
        # See issue #4409
        "tests/shared/training/test_optimizer_utils_part1.mojo",
        "tests/shared/training/test_optimizer_utils_part2.mojo",
        "tests/shared/training/test_mixed_precision_simd.mojo",
        "tests/shared/training/test_multistep_scheduler.mojo",
        "tests/shared/training/test_optimizer_utils.mojo",
        "tests/shared/training/test_optimizers.mojo",
        "tests/shared/training/test_precision_checkpoint.mojo",
        "tests/shared/training/test_precision_config.mojo",
        "tests/shared/training/test_results_printer_part1.mojo",
        "tests/shared/training/test_results_printer_part2.mojo",
        "tests/shared/training/test_results_printer_part3.mojo",
        "tests/shared/training/test_rmsprop.mojo",
        "tests/shared/training/test_schedulers_part1.mojo",
        "tests/shared/training/test_schedulers_part2.mojo",
        "tests/shared/training/test_schedulers_part3.mojo",
        "tests/shared/training/test_step_scheduler.mojo",
        "tests/shared/training/test_trainer_interface_bugs.mojo",
        "tests/shared/training/test_training_loop_part1.mojo",
        "tests/shared/training/test_training_loop_part2.mojo",
        "tests/shared/training/test_training_loop_part3.mojo",
        "tests/shared/training/test_validation_loop.mojo",
        "tests/shared/training/test_warmup_composite_scheduler.mojo",
        "tests/shared/training/test_warmup_scheduler.mojo",
    ]
    exclude_files.extend(exclude_training_patterns)

    for test_file in root_dir.rglob("test_*.mojo"):
        # Check if file is in an excluded directory
        if any(exclude in str(test_file) for exclude in exclude_patterns):
            continue

        # Check if file is explicitly excluded (dataset-dependent tests)
        rel_path = test_file.relative_to(root_dir)
        if str(rel_path) in exclude_files:
            continue

        test_files.append(rel_path)

    return sorted(test_files)


def parse_ci_matrix(workflow_file: Path) -> Dict[str, Dict[str, str]]:
    """Parse the CI workflow YAML to extract test groups and their patterns."""

    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)

    # Navigate to the test matrix
    try:
        jobs = workflow["jobs"]
        test_job = jobs.get("test-mojo-comprehensive", {})
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        test_groups = matrix.get("test-group", [])
    except (KeyError, TypeError) as e:
        print(f"❌ Error parsing workflow file: {e}", file=sys.stderr)
        sys.exit(1)

    # Build a mapping of group name -> (path, pattern)
    groups = {}
    for group in test_groups:
        name = group.get("name")
        path = group.get("path")
        pattern = group.get("pattern")

        if name and path and pattern:
            groups[name] = {"path": path, "pattern": pattern}

    # Also parse separate test jobs (test-configs, test-training, test-benchmarks, test-core-layers, test-example-arithmetic)
    # These are standalone jobs outside the matrix
    for job_name in [
        "test-configs",
        "test-training",
        "test-benchmarks",
        "test-core-layers",
        "test-example-arithmetic",
    ]:
        job = jobs.get(job_name, {})
        if job:
            # Extract the test command from the "Run X tests" step
            steps = job.get("steps", [])
            for step in steps:
                run_cmd = step.get("run", "")
                # Parse command like: just test-group tests/configs "test_*.mojo"
                if "test-group" in run_cmd:
                    parts = run_cmd.split()
                    if len(parts) >= 3:
                        path = parts[2]  # tests/configs or tests/shared/training
                        pattern = parts[3].strip('"')  # "test_*.mojo"
                        name = job.get("name", job_name)
                        groups[name] = {"path": path, "pattern": pattern}

    return groups


def expand_pattern(base_path: str, pattern: str, root_dir: Path) -> Set[Path]:
    """Expand a test pattern to actual file paths."""
    matched_files = set()

    # Split pattern by spaces (multiple patterns)
    patterns = pattern.split()

    for pat in patterns:
        # Handle wildcard patterns
        if "*" in pat:
            # Construct the full glob pattern
            full_pattern = f"{base_path}/{pat}"
            for match in sorted(root_dir.glob(full_pattern)):
                if match.is_file():
                    matched_files.add(match.relative_to(root_dir))
        else:
            # Direct file reference or subdirectory pattern
            if "/" in pat:
                # Subdirectory pattern like "datasets/test_*.mojo"
                full_pattern = f"{base_path}/{pat}"
                for match in sorted(root_dir.glob(full_pattern)):
                    if match.is_file():
                        matched_files.add(match.relative_to(root_dir))
            else:
                # Direct file
                full_path = root_dir / base_path / pat
                if full_path.is_file():
                    matched_files.add(full_path.relative_to(root_dir))

    return matched_files


def _paths_overlap(path_a: str, path_b: str) -> bool:
    """Return True if one path is a prefix of the other (or they are equal)."""
    a, b = Path(path_a), Path(path_b)
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return True
    except ValueError:
        pass
    return False


def check_group_overlaps(
    ci_groups: Dict[str, Dict[str, str]],
    coverage_by_group: Dict[str, Set[Path]],
) -> List[Tuple[str, str, Path]]:
    """Detect CI matrix groups whose resolved file sets overlap.

    Only compares groups with overlapping ``path:`` prefixes to avoid
    false positives between unrelated directories (e.g., ``benchmarks/``
    vs ``tests/``).

    Args:
        ci_groups: Mapping of group name to its ``path`` and ``pattern`` config.
        coverage_by_group: Mapping of group name to the set of files it covers.

    Returns:
        Sorted list of ``(group_a_name, group_b_name, file)`` triples for every
        file matched by more than one group.
    """
    overlaps: List[Tuple[str, str, Path]] = []
    group_names = sorted(coverage_by_group.keys())

    for i, name_a in enumerate(group_names):
        path_a = ci_groups[name_a]["path"]
        files_a = coverage_by_group[name_a]

        for name_b in group_names[i + 1 :]:
            path_b = ci_groups[name_b]["path"]
            files_b = coverage_by_group[name_b]

            # Only compare groups with overlapping path prefixes
            if not _paths_overlap(path_a, path_b):
                continue

            common = files_a & files_b
            for f in sorted(common):
                overlaps.append((name_a, name_b, f))

    return overlaps


def check_stale_patterns(
    ci_groups: Dict[str, Dict[str, str]],
    root_dir: Path,
) -> List[str]:
    """Detect CI matrix groups whose patterns match zero existing test files.

    Args:
        ci_groups: Mapping of group name to its ``path`` and ``pattern`` config.
        root_dir: Repository root directory used for glob expansion.

    Returns:
        Sorted list of group names whose patterns match no files.
    """
    stale: List[str] = []
    for group_name, group_info in ci_groups.items():
        matched = expand_pattern(group_info["path"], group_info["pattern"], root_dir)
        if not matched:
            stale.append(group_name)
    return sorted(stale)


def check_coverage(
    test_files: List[Path], ci_groups: Dict[str, Dict[str, str]], root_dir: Path
) -> Tuple[Set[Path], Dict[str, Set[Path]], List[Tuple[str, str, str]]]:
    """
    Check which test files are covered by CI matrix and detect stale patterns.

    Returns:
        (uncovered_files, group_coverage_map, stale_patterns)
        where stale_patterns is a list of (group_name, path, pattern) tuples
        that don't match any actual files
    """
    all_covered = set()
    coverage_by_group = {}
    stale_patterns = []

    for group_name, group_info in ci_groups.items():
        covered = expand_pattern(group_info["path"], group_info["pattern"], root_dir)
        coverage_by_group[group_name] = covered
        all_covered.update(covered)

        # Check if the pattern matched any files
        if not covered:
            stale_patterns.append((group_name, group_info["path"], group_info["pattern"]))

    uncovered = set(test_files) - all_covered

    return uncovered, coverage_by_group, stale_patterns


def generate_report(
    uncovered: Set[Path],
    test_files: List[Path],
    coverage_by_group: Dict[str, Set[Path]],
    stale_patterns: Optional[List[Tuple[str, str, str]]] = None,
    overlaps: Optional[List[Tuple[str, str, Path]]] = None,
) -> str:
    """Generate a detailed validation report."""
    report_lines = []
    report_lines.append("## Test Coverage Validation Report")
    report_lines.append("")

    effective_overlaps = overlaps or []

    if not uncovered and not stale_patterns and not effective_overlaps:
        report_lines.append("✅ All test files are covered by CI!")
        report_lines.append("")
        report_lines.append(f"- Total test files: {len(test_files)}")
        report_lines.append(f"- Covered by {len(coverage_by_group)} test groups")
        report_lines.append("")
        report_lines.append("### Coverage by Test Group")
        report_lines.append("")
        for group_name in sorted(coverage_by_group.keys()):
            count = len(coverage_by_group[group_name])
            report_lines.append(f"- {group_name}: {count} test(s)")
    else:
        if effective_overlaps:
            report_lines.append(f"❌ Found {len(effective_overlaps)} overlapping test file(s) across CI matrix groups")
            report_lines.append("")
            report_lines.append("### Overlapping Test Groups")
            report_lines.append("")
            report_lines.append("The following files are matched by more than one CI matrix group.")
            report_lines.append("Each group must use explicit, non-overlapping patterns.")
            report_lines.append("")
            for group_a, group_b, f in effective_overlaps:
                report_lines.append(f"- `{f}` matched by both **{group_a}** and **{group_b}**")
            report_lines.append("")
            report_lines.append("### How to Fix Overlaps")
            report_lines.append("")
            report_lines.append("Remove subdirectory wildcards from parent groups so each test file")
            report_lines.append("is owned by exactly one CI matrix group.")
            report_lines.append("")

        if uncovered:
            report_lines.append(f"❌ Found {len(uncovered)} uncovered test file(s)")
            report_lines.append("")
            report_lines.append("### Uncovered Tests")
            report_lines.append("")
            for test_file in sorted(uncovered):
                report_lines.append(f"- {test_file}")
            report_lines.append("")
            report_lines.append("### Recommendations")
            report_lines.append("")
            report_lines.append("Add missing test files to `.github/workflows/comprehensive-tests.yml`")
            report_lines.append("by updating the appropriate test group or creating a new one.")
            report_lines.append("")
            report_lines.append("#### Example Test Groups to Consider")
            report_lines.append("")

            # Suggest groups based on uncovered paths
            suggestions: Dict[str, Any] = {}
            for test_file in sorted(uncovered):
                parts = test_file.parts
                if len(parts) >= 2:
                    suggested_group = parts[1]
                    if suggested_group not in suggestions:
                        suggestions[suggested_group] = []
                    suggestions[suggested_group].append(test_file)

            report_lines.append("```yaml")
            for group, files in sorted(suggestions.items()):
                report_lines.append(f'- name: "{group.title()}"')
                report_lines.append(f'  path: "{files[0].parent}"')
                report_lines.append('  pattern: "test_*.mojo"')
            report_lines.append("```")

        if stale_patterns:
            report_lines.append("")
            report_lines.append("### Stale CI Patterns")
            report_lines.append("")
            report_lines.append("The following test groups have patterns that don't match any test files:")
            report_lines.append("")
            for group_name, path, pattern in sorted(stale_patterns):
                report_lines.append(f"- **{group_name}**: `{path}` with pattern `{pattern}`")
            report_lines.append("")
            report_lines.append(
                "These patterns may be outdated or incorrect. "
                "Consider updating or removing them from the CI configuration."
            )

    return "\n".join(report_lines)


def post_to_pr(report: str) -> bool:
    """Post validation report to GitHub PR if running in CI."""
    try:
        # Check if we're in GitHub Actions
        github_ref = os.environ.get("GITHUB_REF", "")
        pr_number = None

        # Extract PR number from GitHub Actions context
        # In PR events, GITHUB_REF is refs/pull/{pr_number}/merge
        if "/pull/" in github_ref:
            match = re.search(r"refs/pull/(\d+)/", github_ref)
            if match:
                pr_number = match.group(1)

        if not pr_number:
            print("ℹ️  Not a PR context. Skipping PR comment.", file=sys.stderr)
            return False

        # Use gh CLI to post comment
        comment_body = f"{report}\n\n---\n*This check runs automatically on pull requests.*"

        result = subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                pr_number,
                "--body",
                comment_body,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Posted validation report to PR", file=sys.stderr)
            return True
        else:
            print(f"⚠️  Failed to post comment to PR: {result.stderr}", file=sys.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Timeout posting comment to PR", file=sys.stderr)
        return False
    except Exception as e:
        print(f"⚠️  Error posting comment to PR: {e}", file=sys.stderr)
        return False


def main():
    """Main validation logic."""
    # Parse arguments
    post_pr = "--post-pr" in sys.argv

    # Determine repository root
    repo_root = get_repo_root()

    # Find all test files (quietly)
    test_files = find_test_files(repo_root)

    # Parse CI workflow
    workflow_file = repo_root / ".github" / "workflows" / "comprehensive-tests.yml"
    if not workflow_file.exists():
        print(f"❌ Workflow file not found: {workflow_file}", file=sys.stderr)
        sys.exit(1)

    ci_groups = parse_ci_matrix(workflow_file)

    # Check coverage
    uncovered, coverage_by_group, stale_patterns = check_coverage(test_files, ci_groups, repo_root)

    # Check for overlapping patterns across CI matrix groups
    overlaps = check_group_overlaps(ci_groups, coverage_by_group)

    # Parse command line flags
    post_pr = "--post-pr" in sys.argv
    warn_stale = "--warn-stale" in sys.argv or "--error-stale" not in sys.argv
    error_stale = "--error-stale" in sys.argv

    has_errors = bool(uncovered or overlaps)

    if overlaps:
        print("=" * 70)
        print("CI Matrix Group Overlap Detection")
        print("=" * 70)
        print()
        print(f"❌ Found {len(overlaps)} file(s) matched by multiple CI matrix groups:")
        print()
        for group_a, group_b, f in overlaps:
            print(f"   • {f}")
            print(f"     → matched by '{group_a}' AND '{group_b}'")
        print()
        print("Each test file must be owned by exactly one CI matrix group.")
        print("Remove subdirectory wildcards from parent groups so patterns are explicit.")
        print()

    # Only print detailed report if tests are missing or stale patterns exist
    if uncovered or (stale_patterns and warn_stale):
        print("=" * 70)
        print("Test Coverage Validation")
        print("=" * 70)
        print()
        print(f"❌ Found {len(uncovered)} uncovered test file(s):")
        print()

        for test_file in sorted(uncovered):
            print(f"   • {test_file}")

        print()
        print("=" * 70)
        print("Recommendations")
        print("=" * 70)
        print()
        print("Add missing test files to .github/workflows/comprehensive-tests.yml")
        print("by updating the appropriate test group or creating a new one.")
        print()
        print("Example test groups to consider:")
        print()

        # Suggest groups based on uncovered paths
        suggestions: Dict[str, Any] = {}
        for test_file in sorted(uncovered):
            parts = test_file.parts
            if len(parts) >= 2:
                suggested_group = parts[1]  # e.g., "shared", "configs", etc.
                if suggested_group not in suggestions:
                    suggestions[suggested_group] = []
                suggestions[suggested_group].append(test_file)

        for group, files in sorted(suggestions.items()):
            print(f'  - name: "{group.title()}"')
            print(f'    path: "{files[0].parent}"')
            print('    pattern: "test_*.mojo"')
            print()

        print()

    if has_errors:
        # Generate report for PR
        report = generate_report(uncovered, test_files, coverage_by_group, stale_patterns, overlaps)

        # Post to PR if requested (post if uncovered files or stale patterns exist)
        if post_pr and (uncovered or stale_patterns):
            post_to_pr(report)

        # Exit with error code
        # - Always exit 1 if uncovered files exist
        # - Exit 1 if stale patterns exist AND --error-stale flag is set
        if uncovered:
            return 1
        if error_stale and stale_patterns:
            return 1

    # Handle case where only stale patterns exist without uncovered files
    if stale_patterns:
        if error_stale:
            return 1
        # For --warn-stale (default), exit with success

    # All tests are covered with no overlaps - exit quietly with success
    return 0


if __name__ == "__main__":
    sys.exit(main())
