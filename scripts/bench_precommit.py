#!/usr/bin/env python3
"""
Pre-commit hook performance benchmark helper.

Accepts elapsed time, file count, and hook status; emits a Markdown summary
table and a GitHub Actions warning annotation if runtime exceeds the threshold.

Usage:
    python scripts/bench_precommit.py --elapsed 45 --files 300 --status passed
    python scripts/bench_precommit.py --elapsed 150 --files 300 --status passed --threshold 120
"""

import argparse
import os
import sys
from typing import Optional


def format_summary_table(elapsed_s: int, file_count: int, hook_status: str) -> str:
    """
    Format a Markdown table summarising the pre-commit benchmark run.

    Args:
        elapsed_s: Wall-clock seconds the hooks took to complete.
        file_count: Number of files processed.
        hook_status: Result string, e.g. ``"passed"`` or ``"failed"``.

    Returns:
        Markdown-formatted table string including a trailing newline.
    """
    status_icon = "✅" if hook_status == "passed" else "❌"
    return (
        "## Pre-commit Hook Benchmark\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Hook status | {status_icon} {hook_status} |\n"
        f"| Elapsed time | {elapsed_s}s |\n"
        f"| Files processed | {file_count} |\n"
    )


def check_threshold(elapsed_s: int, threshold_s: int = 120) -> bool:
    """
    Return ``True`` if the elapsed time exceeds the threshold.

    Args:
        elapsed_s: Measured runtime in seconds.
        threshold_s: Maximum acceptable runtime in seconds (default 120).

    Returns:
        ``True`` when slow (elapsed_s > threshold_s), ``False`` otherwise.
    """
    return elapsed_s > threshold_s


def emit_warning(message: str) -> None:
    """
    Emit a GitHub Actions warning annotation to stdout.

    Args:
        message: Warning text to emit.
    """
    print(f"::warning::{message}")


def write_step_summary(content: str, summary_path: Optional[str] = None) -> None:
    """
    Append content to the GitHub Actions step summary file if the path is set.

    Args:
        content: Markdown content to write.
        summary_path: Path to the summary file; defaults to ``$GITHUB_STEP_SUMMARY``.
    """
    path = summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(content)


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry-point for the pre-commit benchmark helper.

    Accepts --elapsed, --files, --status, and --threshold arguments, writes
    a Markdown table to $GITHUB_STEP_SUMMARY (when set), emits a warning if
    the run was slow, and always exits 0.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Always 0 — timing regressions are non-blocking.
    """
    parser = argparse.ArgumentParser(description="Report pre-commit hook benchmark results.")
    parser.add_argument(
        "--elapsed",
        type=int,
        required=True,
        help="Elapsed time in seconds.",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=0,
        help="Number of files processed.",
    )
    parser.add_argument(
        "--status",
        default="passed",
        help='Hook exit status string, e.g. "passed" or "failed".',
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=120,
        help="Warning threshold in seconds (default: 120).",
    )

    args = parser.parse_args(argv)

    table = format_summary_table(args.elapsed, args.files, args.status)
    print(table)
    write_step_summary(table)

    if check_threshold(args.elapsed, args.threshold):
        emit_warning(
            f"Pre-commit hooks took {args.elapsed}s, "
            f"which exceeds the {args.threshold}s threshold. "
            "Consider reviewing hook configuration for performance regressions."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
