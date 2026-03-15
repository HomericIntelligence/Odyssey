#!/usr/bin/env python3
"""
Validate that all 'pixi run mojo' test/run/build/package calls in workflow files
are protected by a retry loop (issue #3329, ADR-009).

The Mojo v0.26.1 JIT compiler occasionally crashes with a libKGENCompilerRTShared.so
error. Every invocation that runs Mojo code (test, run, build, package) must be
wrapped in a 3-attempt exponential-backoff retry loop. Version-check calls
('pixi run mojo --version') do not compile Mojo code and are exempt.

A call is considered protected when a 'while [' or 'attempt=' line appears
within the same run: block as the 'pixi run mojo' line.

Usage:
    python scripts/validate_mojo_retry_pattern.py [path ...]

    path: One or more workflow YAML files or directories containing them.
          Defaults to .github/workflows/ relative to the repo root.

Exit codes:
    0: All workflows pass -- every non-exempt pixi run mojo call has retry protection
    1: One or more violations found
"""

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import yaml


# Security limit: skip files larger than 1 MB
MAX_FILE_SIZE = 1_048_576

# Mojo subcommands that compile/execute code -- these MUST be retry-wrapped
COMPILING_SUBCOMMANDS = {"test", "run", "build", "package"}

# Marker strings that indicate a retry loop is present in the block
RETRY_MARKERS = ("while [", "attempt=")


class Violation(NamedTuple):
    """A single missing-retry violation."""

    workflow_file: Path
    job_name: str
    step_name: str
    line_number: int
    mojo_call: str


def _extract_run_blocks(step: object) -> List[str]:
    """Return the list of run: script strings from a step dict."""
    if not isinstance(step, dict):
        return []
    run = step.get("run")
    if isinstance(run, str):
        return [run]
    return []


def _is_version_check(line: str) -> bool:
    """Return True if the line is a version-check call (exempt from retry requirement)."""
    stripped = line.strip()
    # Matches: pixi run mojo --version  (with optional leading $(...) or |& tee etc.)
    return "pixi run mojo --version" in stripped or "pixi run mojo --version" in stripped


def _subcommand(line: str) -> Optional[str]:
    """
    Extract the Mojo subcommand from a line containing 'pixi run mojo'.

    Returns the subcommand string (e.g. 'test', 'run') or None if not parseable.
    """
    parts = line.strip().split()
    try:
        idx = next(i for i, p in enumerate(parts) if p == "mojo")
        if idx + 1 < len(parts):
            cmd = parts[idx + 1].lstrip("-")  # skip flag-like tokens
            # If the token after 'mojo' starts with '-', it's a flag not a subcommand
            if parts[idx + 1].startswith("-"):
                return None
            return cmd
    except StopIteration:
        pass
    return None


def _is_docker_run_call(line: str) -> bool:
    """Return True if pixi run mojo appears inside a docker run command.

    When mojo runs inside docker, the retry wrapper must wrap the docker run
    command itself -- not the inner pixi call.  These calls are exempt from
    the bare-call check.

    Handles both single-line forms and line-continuation forms where
    'pixi run mojo' appears as the last argument on a continuation line
    (preceded only by whitespace/backslash from the previous line).
    """
    stripped = line.strip()
    if stripped.startswith("docker run") and "pixi run mojo" in stripped:
        return True
    # Continuation line: indented and not starting with a shell keyword,
    # i.e. pixi run mojo is an argument to a prior docker run command.
    # We detect this by checking the line has no leading shell command token.
    if stripped.startswith("pixi run mojo"):
        # Not a top-level command; it's a continuation argument
        # (leading whitespace in the original multi-line string)
        return line != line.lstrip()
    return False


def _is_echo_or_comment(line: str) -> bool:
    """Return True if pixi run mojo appears only inside a string/comment, not as a command."""
    stripped = line.strip()
    # echo/printf with quoted content containing the pattern
    if stripped.startswith("echo ") or stripped.startswith("printf "):
        return True
    # Shell comment
    if stripped.startswith("#"):
        return True
    return False


def _is_compiling_call(line: str) -> bool:
    """Return True if the line calls pixi run mojo with a compiling subcommand."""
    if "pixi run mojo" not in line:
        return False
    if _is_version_check(line):
        return False
    if _is_echo_or_comment(line):
        return False
    if _is_docker_run_call(line):
        return False
    cmd = _subcommand(line)
    if cmd is None:
        # Cannot parse -- conservatively flag it
        return True
    return cmd in COMPILING_SUBCOMMANDS


def _has_retry_protection(block: str) -> bool:
    """Return True if the run: block contains retry-loop markers."""
    return any(marker in block for marker in RETRY_MARKERS)


def _find_violations_in_step(
    workflow_file: Path,
    job_name: str,
    step: object,
) -> List[Violation]:
    """Find missing-retry violations in a single workflow step."""
    if not isinstance(step, dict):
        return []

    step_name = str(step.get("name", "(unnamed step)"))
    violations: List[Violation] = []

    for run_block in _extract_run_blocks(step):
        if not _has_retry_protection(run_block):
            for line_num, line in enumerate(run_block.splitlines(), start=1):
                if _is_compiling_call(line):
                    violations.append(
                        Violation(
                            workflow_file=workflow_file,
                            job_name=job_name,
                            step_name=step_name,
                            line_number=line_num,
                            mojo_call=line.strip(),
                        )
                    )

    return violations


def validate_workflow(workflow_file: Path) -> List[Violation]:
    """
    Validate retry protection for all pixi run mojo calls in a workflow file.

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

        for step in steps:
            violations.extend(_find_violations_in_step(workflow_file, str(job_name), step))

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
    here = Path(__file__).resolve().parent
    candidate = here.parent / ".github" / "workflows"
    if candidate.is_dir():
        return str(candidate)
    return ".github/workflows"


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the mojo retry pattern validation script.

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
                f"\nERROR: {v.workflow_file} :: job '{v.job_name}' :: step '{v.step_name}'"
                f" :: line {v.line_number}\n"
                f"  Bare pixi run mojo call without retry loop: {v.mojo_call}\n"
                f"  Wrap with 3-attempt exponential-backoff retry (issue #3329, ADR-009)."
            )
        print(f"\nFound {len(all_violations)} violation(s) across {len(workflow_files)} file(s).")
        return 1

    print(f"OK: {len(workflow_files)} workflow file(s) checked. All pixi run mojo calls are retry-protected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
