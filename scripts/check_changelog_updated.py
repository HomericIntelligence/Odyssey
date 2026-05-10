#!/usr/bin/env python3
"""Check that CHANGELOG.md is updated when user-facing files change.

Exits 0 always — this is a reminder, not a hard gate.
In CI (GITHUB_BASE_REF set) it prints a warning; otherwise it is silent.
"""

import os
import subprocess
import sys


def changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


WATCHED_PREFIXES = ("shared/", "models/", "tools/")


def main() -> int:
    base_ref = os.environ.get("GITHUB_BASE_REF", "")
    if not base_ref:
        # Not in CI — silent pass
        return 0

    files = changed_files(base_ref)
    has_user_facing = any(f.startswith(WATCHED_PREFIXES) for f in files)
    has_changelog = "CHANGELOG.md" in files

    if has_user_facing and not has_changelog:
        print(
            "::warning::CHANGELOG.md was not updated. "
            "If this PR includes user-facing changes (feat/fix), "
            "please add a curated entry to CHANGELOG.md.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
