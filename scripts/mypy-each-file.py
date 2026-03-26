#!/usr/bin/env python3
"""Run mypy on each file individually to avoid duplicate module name errors.

This wrapper is needed for examples/ subdirectories (e.g. examples/alexnet_cifar10/)
where multiple files share the same basename (download_cifar10.py). Passing them all to a
single mypy invocation causes a "Duplicate module named" blocker error because mypy cannot
resolve the directory as a Python package component.

Usage:
    python scripts/mypy-each-file.py [mypy-args...] file1.py file2.py ...

The script separates mypy flags (starting with '-') from file paths, then runs mypy once
per file, aggregating exit codes.
"""

import sys
import subprocess


def main() -> int:
    args = sys.argv[1:]

    # Separate mypy flags from file paths (flags start with '-')
    flags: list[str] = []
    files: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            flags.append(arg)
            # Some flags take a value argument (e.g. --python-version 3.10)
            if arg in ("--python-version", "--config-file", "--shadow-file", "--exclude"):
                i += 1
                if i < len(args):
                    flags.append(args[i])
        else:
            files.append(arg)
        i += 1

    if not files:
        print("mypy-each-file: no files to check", file=sys.stderr)
        return 0

    overall_rc = 0
    for filepath in files:
        cmd = [sys.executable, "-m", "mypy"] + flags + [filepath]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            overall_rc = result.returncode

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
