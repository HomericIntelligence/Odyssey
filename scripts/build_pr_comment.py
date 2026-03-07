#!/usr/bin/env python3
"""Build PR comment markdown file from metrics content.

Combines a plain-text header with metrics.md content and a footer,
writing the result to an output file. This replaces the fragile
printf emoji byte-escape approach in coverage.yml.

Usage:
    python scripts/build_pr_comment.py --metrics-file metrics.md --output-file pr_comment.md
"""

import argparse
import sys
from pathlib import Path

HEADER = "## Test Metrics Report\n\n"
FOOTER = "\n\n---\n*Note: Full code coverage requires Mojo coverage tooling (blocked - see ADR-008)*\n"


def build_comment(metrics_file: Path, output_file: Path) -> int:
    """Build the PR comment file from metrics content.

    Args:
        metrics_file: Path to the input metrics markdown file.
        output_file: Path where the PR comment markdown will be written.

    Returns:
        0 on success, 1 if metrics_file does not exist.
    """
    if not metrics_file.exists():
        print(f"Error: metrics file not found: {metrics_file}", file=sys.stderr)
        return 1

    metrics_content = metrics_file.read_text(encoding="utf-8")
    comment = HEADER + metrics_content + FOOTER
    output_file.write_text(comment, encoding="utf-8")
    return 0


def main() -> int:
    """Parse arguments and build the PR comment file."""
    parser = argparse.ArgumentParser(description="Build PR comment markdown from metrics file.")
    parser.add_argument(
        "--metrics-file",
        required=True,
        type=Path,
        help="Path to the input metrics markdown file",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Path to write the PR comment markdown file",
    )
    args = parser.parse_args()
    return build_comment(args.metrics_file, args.output_file)


if __name__ == "__main__":
    sys.exit(main())
