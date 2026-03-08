#!/usr/bin/env python3
"""Analyze remaining docstring warnings and categorize them."""

import subprocess
import re
from collections import defaultdict
from pathlib import Path


def get_build_warnings():
    """Run build and capture warnings."""
    import os

    env = os.environ.copy()
    env["NATIVE"] = "1"

    result = subprocess.run(["just", "build", "debug"], capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    warning_lines = [line for line in output.split("\n") if "warning:" in line]
    return warning_lines


def categorize_warnings(warnings):
    """Categorize warnings by type."""
    categories = defaultdict(list)

    for warning in warnings:
        if "section body should end" in warning:
            categories["section_body_ending"].append(warning)
        elif "unknown argument" in warning:
            categories["unknown_argument"].append(warning)
        elif "description should end with a period" in warning:
            categories["missing_period"].append(warning)
        elif "is defined at index" in warning:
            categories["parameter_order"].append(warning)
        elif "doc string summary should end" in warning:
            categories["summary_period"].append(warning)
        elif "has been deprecated" in warning:
            categories["deprecated_syntax"].append(warning)
        elif "is overindented" in warning:
            categories["indentation"].append(warning)
        else:
            categories["other"].append(warning)

    return categories


def print_summary(categories):
    """Print summary of warning categories."""
    total = sum(len(warnings) for warnings in categories.values())

    print("=" * 80)
    print(f"DOCSTRING WARNING ANALYSIS - Total: {total} warnings")
    print("=" * 80)
    print()

    # Sort by count descending
    sorted_cats = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)

    for category, warnings in sorted_cats:
        count = len(warnings)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category:25s} {count:4d} warnings ({percentage:5.1f}%)")

    print()
    print("=" * 80)
    print("TOP 10 FILES BY WARNING COUNT")
    print("=" * 80)
    print()

    # Count warnings per file
    file_counts: defaultdict[str, int] = defaultdict(int)
    for warnings in categories.values():
        for warning in warnings:
            # Extract file path
            match = re.match(r"([^:]+):\d+:\d+:", warning)
            if match:
                file_path = match.group(1)
                file_counts[file_path] += 1

    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    repo_root = Path(__file__).resolve().parent.parent
    for file_path, count in sorted_files[:10]:
        # Show just the filename relative to repo root
        short_path = file_path.replace(str(repo_root) + "/", "")
        print(f"{count:4d} warnings - {short_path}")

    print()
    print("=" * 80)
    print("EXAMPLES BY CATEGORY")
    print("=" * 80)

    for category, warnings in sorted_cats:
        if warnings:
            print(f"\n{category} ({len(warnings)} total):")
            print("-" * 80)
            for warning in warnings[:3]:  # Show first 3 examples
                print(f"  {warning}")


def main():
    print("Fetching build warnings...")
    warnings = get_build_warnings()

    print(f"Found {len(warnings)} total warnings\n")

    categories = categorize_warnings(warnings)
    print_summary(categories)


if __name__ == "__main__":
    main()
