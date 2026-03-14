#!/usr/bin/env python3
"""
Lint rule to prevent bare pixi run mojo calls in CI workflows.

This script checks GitHub Actions workflow files for bare `pixi run mojo` calls
that are not wrapped in a retry loop or routed through `just test-group`.

Usage:
    python scripts/check_bare_pixi_mojo.py [--fix]

Exit Codes:
    0 - No bare pixi run mojo calls found
    1 - One or more bare calls found
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

def check_workflows() -> Tuple[bool, List[str]]:
    """
    Check GitHub Actions workflows for bare pixi run mojo calls.

    Returns:
        (has_issues, error_messages)
    """
    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        return False, []

    has_issues = False
    errors = []

    # Pattern to match bare pixi run mojo calls (not in retry loop, not through just)
    # This matches: pixi run mojo [args] that are not:
    #   - Inside a retry loop (for statement, while loop, etc.)
    #   - Inside a just test-group call
    #   - Inside a pixi run pre-commit command
    bare_mojo_pattern = r'pixi\s+run\s+mojo\s+[^-]'
    safe_patterns = [
        r'for\s+',  # for loop (retry)
        r'while\s+',  # while loop (retry)
        r'attempt=',  # attempt counter
        r'just\s+test-group',  # routed through just
        r'pixi\s+run\s+pre-commit',  # pre-commit hook
    ]

    for workflow_file in workflows_dir.glob("*.yml"):
        with open(workflow_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check if line contains bare pixi run mojo
            if re.search(bare_mojo_pattern, line):
                # Check if it's in a safe context
                is_safe = False

                # Look at surrounding context (previous 5 lines for retry loops)
                start_line = max(0, i - 6)
                context = '\n'.join(lines[start_line:i])

                # Check for retry loop patterns in context
                if re.search(r'for\s+\w+\s+in\s+', context) or \
                   re.search(r'while\s+\[', context) or \
                   re.search(r'attempt=', context) or \
                   re.search(r'if\s+pixi\s+run\s+mojo.*then', context):
                    is_safe = True

                # Check if it's routed through just or pre-commit
                if 'just test-group' in line or 'pixi run pre-commit' in line:
                    is_safe = True

                # Check if it has error handling (|| or &&)
                if ' || ' in line or ' && ' in line or '|| echo' in line:
                    is_safe = True

                if not is_safe:
                    error_msg = (
                        f"{workflow_file.name}:{i}: bare `pixi run mojo` call "
                        f"without retry loop or just test-group routing\n"
                        f"  {line.strip()}\n"
                        f"  Wrap in retry loop or use `just test-group` instead. "
                        f"See Issue #3956."
                    )
                    errors.append(error_msg)
                    has_issues = True

    return has_issues, errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for bare pixi run mojo calls in CI workflows"
    )
    parser.add_argument("--fix", action="store_true", help="Fix (not implemented)")
    args = parser.parse_args()

    has_issues, errors = check_workflows()

    if has_issues:
        print("❌ Found bare pixi run mojo calls in workflows:")
        print()
        for error in errors:
            print(error)
            print()
        return 1
    else:
        print("✅ No bare pixi run mojo calls found in workflows")
        return 0


if __name__ == "__main__":
    sys.exit(main())
