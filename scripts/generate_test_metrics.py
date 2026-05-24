#!/usr/bin/env python3
"""Generate test metrics report as coverage proxy.

Tracks test file counts, test function counts, and test LOC
as proxy metrics until Mojo provides coverage instrumentation.

Usage:
    python scripts/generate_test_metrics.py [--output metrics.json] [--format json|markdown]
"""

import json
import re
from pathlib import Path
from typing import Any, Dict
import sys

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from hephaestus.utils import get_repo_root


def count_test_functions(file_path: Path) -> int:
    """Count fn test_* functions in Mojo test file."""
    try:
        content = file_path.read_text()
        # Match 'fn test_' pattern for Mojo test functions
        return len(re.findall(r"^\s*fn\s+test_", content, re.MULTILINE))
    except Exception:
        return 0


def count_lines(file_path: Path) -> int:
    """Count total lines in file."""
    try:
        return len(file_path.read_text().splitlines())
    except Exception:
        return 0


def generate_metrics(repo_root: Path) -> Dict[str, Any]:
    """Generate comprehensive test metrics."""
    metrics: Dict[str, Any] = {"total_test_files": 0, "total_test_functions": 0, "total_test_lines": 0, "by_module": {}}

    # Scan test directories
    test_dirs = ["tests/shared", "tests/models", "tests/configs", "examples"]

    for test_dir in test_dirs:
        dir_path = repo_root / test_dir
        if not dir_path.exists():
            continue

        for test_file in sorted(dir_path.rglob("test_*.mojo")):
            # Skip E2E and dataset-dependent tests
            if "_e2e.mojo" in test_file.name:
                continue
            if any(skip in str(test_file) for skip in ["emnist", "cifar", "examples/", ".pixi", "build"]):
                continue

            metrics["total_test_files"] += 1
            test_count = count_test_functions(test_file)
            metrics["total_test_functions"] += test_count
            lines_count = count_lines(test_file)
            metrics["total_test_lines"] += lines_count

            # Track by module
            try:
                parts = test_file.parent.relative_to(repo_root).parts
                if len(parts) > 1:
                    module = parts[1]
                else:
                    module = "root"
            except ValueError:
                module = "root"

            if module not in metrics["by_module"]:
                metrics["by_module"][module] = {"files": 0, "functions": 0, "lines": 0}
            metrics["by_module"][module]["files"] += 1
            metrics["by_module"][module]["functions"] += test_count
            metrics["by_module"][module]["lines"] += lines_count

    return metrics


def format_markdown(metrics: Dict[str, Any]) -> str:
    """Format metrics as markdown."""
    lines = []

    lines.append("## 📊 Test Metrics Report")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Total Test Files**: {metrics['total_test_files']}")
    lines.append(f"- **Total Test Functions**: {metrics['total_test_functions']}")
    lines.append(f"- **Total Test Lines**: {metrics['total_test_lines']}")
    lines.append("")

    if metrics["total_test_files"] > 0:
        avg_tests_per_file = metrics["total_test_functions"] / metrics["total_test_files"]
        avg_lines_per_file = metrics["total_test_lines"] / metrics["total_test_files"]
        lines.append(f"- **Average Tests per File**: {avg_tests_per_file:.1f}")
        lines.append(f"- **Average Lines per File**: {avg_lines_per_file:.1f}")
    lines.append("")

    lines.append("### By Module")
    lines.append("")
    lines.append("| Module | Files | Functions | Lines |")
    lines.append("|--------|-------|-----------|-------|")

    for module in sorted(metrics["by_module"].keys()):
        mod_data = metrics["by_module"][module]
        lines.append(f"| {module} | {mod_data['files']} | {mod_data['functions']} | {mod_data['lines']} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "**Note**: These are test metrics (count-based). "
        "Full code coverage requires Mojo coverage tooling (blocked - see ADR-008)"
    )

    return "\n".join(lines)


def format_json(metrics: Dict[str, Any]) -> str:
    """Format metrics as JSON."""
    return json.dumps(metrics, indent=2)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate test metrics report as coverage proxy")
    parser.add_argument("--output", type=Path, default=None, help="Output file path (default: print to stdout)")
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="markdown", help="Output format (default: markdown)"
    )

    args = parser.parse_args()

    # Determine repo root
    repo_root = get_repo_root()

    # Generate metrics
    metrics = generate_metrics(repo_root)

    # Format output
    if args.format == "json":
        output = format_json(metrics)
    else:
        output = format_markdown(metrics)

    # Write or print
    if args.output:
        args.output.write_text(output)
        print(f"✅ Metrics written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
