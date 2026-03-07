#!/usr/bin/env python3
"""
Validate that composite action references follow checkout-first ordering.

Local composite actions (uses: ./.github/actions/X) require the repository to
be checked out before they can be referenced. This script enforces that
actions/checkout always appears as a step before any ./.github/actions/ reference
within every job of every scanned workflow file.

Usage:
    python scripts/validate_workflow_checkout_order.py [path ...]

    path: One or more workflow YAML files or directories containing them.
          Defaults to .github/workflows/ relative to the repo root.

Exit codes:
    0: All workflows pass the checkout-first invariant
    1: One or more violations found
"""

import sys
from pathlib import Path
from typing import List, NamedTuple

import yaml


# Security limit: skip files larger than 1 MB
MAX_FILE_SIZE = 1_048_576


class Violation(NamedTuple):
    """A single checkout-order violation."""

    workflow_file: Path
    job_name: str
    step_index: int
    step_name: str
    composite_action: str


def _is_checkout_step(step: object) -> bool:
    """Return True if the step uses actions/checkout (any version or hash)."""
    if not isinstance(step, dict):
        return False
    uses = step.get("uses", "")
    return isinstance(uses, str) and uses.startswith("actions/checkout")


def _is_composite_action_step(step: object) -> bool:
    """Return True if the step references a local composite action."""
    if not isinstance(step, dict):
        return False
    uses = step.get("uses", "")
    return isinstance(uses, str) and uses.startswith("./.github/actions/")


def validate_workflow(workflow_file: Path) -> List[Violation]:
    """
    Validate checkout-first ordering for all jobs in a workflow file.

    Args:
        workflow_file: Path to the workflow YAML file.

    Returns:
        List of Violation objects; empty list means the file passes.
    """
    if workflow_file.stat().st_size > MAX_FILE_SIZE:
        print(f"WARNING: Skipping {workflow_file} (exceeds {MAX_FILE_SIZE} byte limit)")
        return []

    with open(workflow_file, encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            print(f"WARNING: Skipping {workflow_file} (YAML parse error: {exc})")
            return []

    if not isinstance(data, dict):
        return []

    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        return []

    violations: List[Violation] = []

    for job_name, job_data in jobs.items():
        if not isinstance(job_data, dict):
            continue

        steps = job_data.get("steps")
        if not isinstance(steps, list):
            continue

        checked_out = False
        for idx, step in enumerate(steps):
            if _is_checkout_step(step):
                checked_out = True
                continue

            if _is_composite_action_step(step):
                if not checked_out:
                    step_name = (
                        step.get("name", f"(unnamed step {idx + 1})") if isinstance(step, dict) else f"(step {idx + 1})"
                    )
                    composite_action = step.get("uses", "") if isinstance(step, dict) else ""
                    violations.append(
                        Violation(
                            workflow_file=workflow_file,
                            job_name=str(job_name),
                            step_index=idx + 1,
                            step_name=str(step_name),
                            composite_action=str(composite_action),
                        )
                    )

    return violations


def collect_workflow_files(paths: List[str]) -> List[Path]:
    """
    Expand the given paths into a list of workflow YAML files.

    Args:
        paths: File paths or directory paths. Directories are searched for
               *.yml and *.yaml files non-recursively.

    Returns:
        Deduplicated list of Path objects for each candidate file.
    """
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.yml")))
            files.extend(sorted(p.glob("*.yaml")))
        else:
            print(f"WARNING: Path not found: {p}", file=sys.stderr)

    # Deduplicate while preserving order
    seen: set = set()
    result: List[Path] = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


def _default_workflow_dir() -> str:
    """Return the default workflow directory relative to the repo root."""
    # Walk up from this script's location to find the repo root
    here = Path(__file__).resolve().parent
    candidate = here.parent / ".github" / "workflows"
    if candidate.is_dir():
        return str(candidate)
    return ".github/workflows"


def main(argv: List[str] | None = None) -> int:
    """
    Entry point for the checkout-order validation script.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for violations found.
    """
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        args = [_default_workflow_dir()]

    workflow_files = collect_workflow_files(args)

    if not workflow_files:
        print("No workflow files found to validate.")
        return 0

    all_violations: List[Violation] = []
    for wf_file in workflow_files:
        violations = validate_workflow(wf_file)
        all_violations.extend(violations)

    if all_violations:
        for v in all_violations:
            print(
                f"\nERROR: {v.workflow_file} :: job '{v.job_name}' :: step {v.step_index} "
                f"uses '{v.composite_action}'\n"
                f"       but actions/checkout is not a preceding step.\n"
                f"       Composite actions require the repository to be checked out first."
            )
        print(f"\nFound {len(all_violations)} violation(s) in {len(workflow_files)} file(s).")
        return 1

    print(f"OK: {len(workflow_files)} workflow file(s) checked. All pass checkout-first invariant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
