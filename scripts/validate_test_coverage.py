#!/usr/bin/env python3
"""
Validate Test Coverage - Ensure all test_*.mojo files are covered by CI

This script finds all test_*.mojo files in the repository and verifies they are
included in the CI test matrix in .github/workflows/comprehensive-tests.yml.

Exit codes:
  0 - All tests covered
  1 - Uncovered tests found or validation errors

Usage:
    python scripts/validate_test_coverage.py [--post-pr]

Arguments:
    --post-pr   Post validation report to GitHub PR if tests are missing
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
from hephaestus.utils import get_repo_root


def find_test_files(root_dir: Path) -> List[Path]:
    """Find all test_*.mojo files, excluding build artifacts and examples."""
    test_files = []

    # Exclude patterns for directories we don't want to scan
    exclude_patterns = [
        ".pixi/",
        "build/",
        "dist/",
        "__pycache__/",
        ".git/",
        "worktrees/",
    ]

    # Exclude specific test files that require external datasets
    # These tests need datasets/ directory which must be downloaded separately
    # They run in the weekly E2E workflow, not per-PR
    exclude_files = [
        # EMNIST example tests (require dataset download)
        "examples/lenet_emnist/test_gradients.mojo",
        "examples/lenet_emnist/test_loss_decrease.mojo",
        "examples/lenet_emnist/test_predictions.mojo",
        "examples/lenet_emnist/test_training_metrics.mojo",
        "examples/lenet_emnist/test_weight_updates.mojo",
        "examples/googlenet_cifar10/test_model.mojo",
        "examples/mobilenetv1_cifar10/test_model.mojo",
        "examples/resnet18_cifar10/test_model.mojo",
        "examples/resnet18_cifar10/test_forward_cache_velocities.mojo",
        "examples/resnet18_cifar10/test_backward.mojo",
        # GoogLeNet backward tests (#3184): concatenate_depthwise_backward
        # SPLIT_EXACT/round-trip unit tests + a full 9-module convergence run.
        # Excluded from the SRC coverage validator (an example-dir test, not a
        # src/ unit test), but EXECUTED per-PR by the "example-backward-tests"
        # job in .github/workflows/comprehensive-tests.yml via
        # `just test-example-backward` — it uses synthetic interleaved-class
        # data with no dataset download.
        "examples/googlenet_cifar10/test_backward.mojo",
        # MobileNetV1 BatchNorm running-stat persistence regression (#5537).
        # Example-dir test (imports `from model`), EXECUTED per-PR by the
        # "example-backward-tests" job via `just test-example-backward`
        # (EXAMPLE_TESTS list) with synthetic data — not a src/ unit test.
        "examples/mobilenetv1_cifar10/test_bn_persistence.mojo",
        # Conda recipe smoke test — executed by rattler-build's `tests:`
        # block when the package is built (see conda.recipe/recipe.yaml),
        # not by the per-PR CI test matrix.
        "conda.recipe/test_import.mojo",
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
        "tests/projectodyssey/training/test_accuracy_bugs.mojo",
        "tests/projectodyssey/training/test_base.mojo",
        "tests/projectodyssey/training/test_callbacks_part1.mojo",
        "tests/projectodyssey/training/test_callbacks_part2.mojo",
        "tests/projectodyssey/training/test_callbacks_part3.mojo",
        "tests/projectodyssey/training/test_callbacks.mojo",
        "tests/projectodyssey/training/test_checkpoint.mojo",
        "tests/projectodyssey/training/test_checkpointing.mojo",
        "tests/projectodyssey/training/test_config.mojo",
        "tests/projectodyssey/training/test_confusion_matrix_bugs.mojo",
        "tests/projectodyssey/training/test_csv_metrics_logger.mojo",
        "tests/projectodyssey/training/test_dtype_utils.mojo",
        "tests/projectodyssey/training/test_early_stopping_part1.mojo",
        "tests/projectodyssey/training/test_early_stopping_part2.mojo",
        "tests/projectodyssey/training/test_evaluate.mojo",
        "tests/projectodyssey/training/test_evaluation_part1.mojo",
        "tests/projectodyssey/training/test_evaluation_part2.mojo",
        "tests/projectodyssey/training/test_evaluation.mojo",
        "tests/projectodyssey/training/test_exponential_scheduler.mojo",
        "tests/projectodyssey/training/test_gradient_clipping.mojo",
        "tests/projectodyssey/training/test_gradient_ops.mojo",
        "tests/projectodyssey/training/test_lars.mojo",
        "tests/projectodyssey/training/test_logging_callback_part1.mojo",
        "tests/projectodyssey/training/test_logging_callback_part2.mojo",
        "tests/projectodyssey/training/test_loops.mojo",
        "tests/projectodyssey/training/test_metrics_part1.mojo",
        "tests/projectodyssey/training/test_metrics_part2.mojo",
        "tests/projectodyssey/training/test_mixed_precision.mojo",
        "tests/projectodyssey/training/test_optimizer_utils_part1.mojo",
        "tests/projectodyssey/training/test_optimizer_utils_part2.mojo",
        "tests/projectodyssey/training/test_mixed_precision_simd.mojo",
        "tests/projectodyssey/training/test_multistep_scheduler.mojo",
        "tests/projectodyssey/training/test_optimizer_utils.mojo",
        "tests/projectodyssey/training/test_optimizers.mojo",
        "tests/projectodyssey/training/test_precision_checkpoint.mojo",
        "tests/projectodyssey/training/test_precision_config.mojo",
        "tests/projectodyssey/training/test_results_printer_part1.mojo",
        "tests/projectodyssey/training/test_results_printer_part2.mojo",
        "tests/projectodyssey/training/test_results_printer_part3.mojo",
        "tests/projectodyssey/training/test_rmsprop.mojo",
        "tests/projectodyssey/training/test_schedulers_part1.mojo",
        "tests/projectodyssey/training/test_schedulers_part2.mojo",
        "tests/projectodyssey/training/test_schedulers_part3.mojo",
        "tests/projectodyssey/training/test_step_scheduler.mojo",
        "tests/projectodyssey/training/test_trainer_interface_bugs.mojo",
        "tests/projectodyssey/training/test_training_loop_part1.mojo",
        "tests/projectodyssey/training/test_training_loop_part2.mojo",
        "tests/projectodyssey/training/test_training_loop_part3.mojo",
        "tests/projectodyssey/training/test_validation_loop.mojo",
        "tests/projectodyssey/training/test_warmup_composite_scheduler.mojo",
        "tests/projectodyssey/training/test_warmup_scheduler.mojo",
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

    # When no matrix groups exist, parse sequential steps of test-mojo-comprehensive directly.
    # Each step runs `just test-group <path> "<pattern>"` (possibly multi-line with \ continuation).
    # Extract path+pattern from the run command to determine which test files are covered.
    if not groups:
        steps = test_job.get("steps", [])
        for step in steps:
            run_cmd = step.get("run", "")
            if "test-group" not in run_cmd:
                continue
            step_name = step.get("name", "unknown")
            # Collapse backslash-newline continuations and strip leading whitespace per line
            collapsed = run_cmd.replace("\\\n", " ")
            collapsed = " ".join(line.strip() for line in collapsed.splitlines())
            # Find `just test-group "<path>" "<patterns...>"` — quotes may or may not be present
            import re as _re

            # Match: just test-group <path> <rest-of-patterns>
            m = _re.search(r'just\s+test-group\s+"?([^\s"]+)"?\s+"?(.+?)(?:"?\s*$|$)', collapsed)
            if m:
                path = m.group(1)
                pattern_raw = m.group(2)
            else:
                # Fallback: tokenize and find test-group token
                parts = collapsed.split()
                try:
                    tg_idx = parts.index("test-group")
                    path = parts[tg_idx + 1].strip('"')
                    pattern_raw = " ".join(parts[tg_idx + 2 :])
                except (ValueError, IndexError):
                    continue
            # Strip quotes, backslash artifacts from YAML multi-line folding
            pattern_raw = pattern_raw.replace("\\ ", " ").replace("\\", " ")
            pattern_raw = pattern_raw.strip('"').strip("'").strip()
            key = f"{step_name}::{path}"
            groups[key] = {"path": path, "pattern": pattern_raw}

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
                # or: NATIVE=1 just test-group tests/configs "test_*.mojo"
                if "test-group" in run_cmd:
                    parts = run_cmd.split()
                    try:
                        tg_idx = parts.index("test-group")
                        path = parts[tg_idx + 1]
                        pattern = parts[tg_idx + 2].strip('"')
                        name = job.get("name", job_name)
                        groups[name] = {"path": path, "pattern": pattern}
                    except (ValueError, IndexError):
                        pass

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
            for match in root_dir.glob(full_pattern):
                if match.is_file():
                    matched_files.add(match.relative_to(root_dir))
        else:
            # Direct file reference or subdirectory pattern
            if "/" in pat:
                # Subdirectory pattern like "datasets/test_*.mojo"
                full_pattern = f"{base_path}/{pat}"
                for match in root_dir.glob(full_pattern):
                    if match.is_file():
                        matched_files.add(match.relative_to(root_dir))
            else:
                # Direct file
                full_path = root_dir / base_path / pat
                if full_path.is_file():
                    matched_files.add(full_path.relative_to(root_dir))

    return matched_files


def group_split_files(test_files: List[Path]) -> Dict[str, List[Path]]:
    """Group test_*_partN.mojo files by their logical base name.

    Files matching the pattern ``test_<base>_part<N>.mojo`` are grouped under
    the key ``<parent_dir>/<base>``.  Files that do not match the pattern are
    ignored (they remain as individual entries in the main coverage logic).

    Args:
        test_files: All test file paths (relative to repo root).

    Returns:
        Mapping of logical group key → sorted list of constituent part files.
        Returns an empty dict when no split-file groups are found.

    Example:
        >>> files = [Path("tests/core/test_foo_part1.mojo"),
        ...          Path("tests/core/test_foo_part2.mojo"),
        ...          Path("tests/core/test_bar.mojo")]
        >>> group_split_files(files)
        {'tests/core/test_foo': [Path('tests/core/test_foo_part1.mojo'),
                                  Path('tests/core/test_foo_part2.mojo')]}
    """
    part_pattern = re.compile(r"^(.+)_part(\d+)\.mojo$")
    groups: Dict[str, List[Path]] = {}
    for f in test_files:
        m = part_pattern.match(f.name)
        if m:
            base_key = str(f.parent / m.group(1))
            groups.setdefault(base_key, []).append(f)
    for key in groups:
        groups[key].sort()
    return groups


def check_stale_patterns(
    ci_groups: Dict[str, Dict[str, str]],
    root_dir: Path,
) -> List[str]:
    """Return names of CI groups (and sub-patterns) that match zero existing files.

    For single-pattern groups or groups where ALL sub-patterns are stale, returns
    the group name.  For multi-pattern groups with only some stale sub-patterns,
    returns ``"GroupName (sub-pattern: <pat>)"`` for each dead sub-pattern so
    callers can identify exactly which sub-pattern is stale without the group
    itself being silently hidden behind a surviving sibling pattern.

    Args:
        ci_groups: Mapping of group name to a dict with ``path`` and ``pattern``
            keys, as returned by :func:`parse_ci_matrix`.
        root_dir: Repository root used to resolve file paths.

    Returns:
        Sorted list of stale group names and/or sub-pattern identifiers.
    """
    stale: List[str] = []

    for group_name, group_info in ci_groups.items():
        sub_patterns = group_info["pattern"].split()
        stale_subs: List[str] = []
        live_subs: List[str] = []

        for sub_pat in sub_patterns:
            matched = expand_pattern(group_info["path"], sub_pat, root_dir)
            if not matched:
                stale_subs.append(sub_pat)
            else:
                live_subs.append(sub_pat)

        if len(stale_subs) == len(sub_patterns):
            # Entire group matches nothing — original group-level staleness
            stale.append(group_name)
        else:
            # Partial staleness: report each dead sub-pattern individually
            for sub_pat in stale_subs:
                stale.append(f"{group_name} (sub-pattern: {sub_pat})")

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
    split_groups: Optional[Dict[str, List[Path]]] = None,
) -> str:
    """Generate a detailed validation report.

    Args:
        uncovered: Test files not matched by any CI group.
        test_files: All test files found in the repository.
        coverage_by_group: Mapping of CI group name → set of covered files.
        split_groups: Optional mapping from :func:`group_split_files` that
            describes logical file groups composed of ``_partN.mojo`` parts.
            When provided, a "Split File Groups" section is appended to the
            report so reviewers can see which groups contain part files.
    """
    if split_groups is None:
        split_groups = {}

    report_lines = []
    report_lines.append("## Test Coverage Validation Report")
    report_lines.append("")

    if not uncovered and not stale_patterns:
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
                path_parts: Tuple[str, ...] = test_file.parts
                if len(path_parts) >= 2:
                    suggested_group: str = path_parts[1]
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
            if uncovered:
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

    # Append split file groups section when part files are present
    if split_groups:
        report_lines.append("")
        report_lines.append("### Split File Groups")
        report_lines.append("")
        report_lines.append("The following logical test groups are split across multiple `_partN.mojo` files:")
        report_lines.append("")
        for group_key in sorted(split_groups.keys()):
            group_parts: List[Path] = split_groups[group_key]
            part_names = ", ".join(p.name for p in group_parts)
            report_lines.append(f"- `{group_key}` ({len(group_parts)} parts: {part_names})")

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

    # Group split (part) files for coverage reporting
    split_groups = group_split_files(test_files)

    # Parse CI workflow
    workflow_file = repo_root / ".github" / "workflows" / "comprehensive-tests.yml"
    if not workflow_file.exists():
        print(f"❌ Workflow file not found: {workflow_file}", file=sys.stderr)
        sys.exit(1)

    ci_groups = parse_ci_matrix(workflow_file)

    # Check coverage
    uncovered, coverage_by_group, stale_patterns = check_coverage(test_files, ci_groups, repo_root)

    # Only print detailed report if tests are missing or stale patterns exist
    if uncovered or stale_patterns:
        print("=" * 70)
        print("Test Coverage Validation")
        print("=" * 70)
        print()

        if uncovered:
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

        if stale_patterns:
            print("⚠️  Found stale CI patterns (patterns that don't match any test files):")
            print()
            for group_name, path, pattern in sorted(stale_patterns):
                print(f"   • {group_name}: {path} / {pattern}")
            print()
            print("These patterns may be outdated or incorrect.")
            print()
            print()

        # Generate report for PR
        report = generate_report(uncovered, test_files, coverage_by_group, stale_patterns, split_groups)

        # Post to PR if requested (post if uncovered files or stale patterns exist)
        if post_pr and (uncovered or stale_patterns):
            post_to_pr(report)

        # Exit with error code
        return 1

    # All tests are covered - exit quietly with success
    return 0


if __name__ == "__main__":
    sys.exit(main())
