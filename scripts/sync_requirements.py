#!/usr/bin/env python3
"""Synchronize pip-only requirements exports from ``uv.lock``.

Python is used for this automation because Mojo cannot capture subprocess
stdout or perform the line filtering needed here (ADR-001). The exporter uses
only the Python standard library and treats ``uv.lock`` as authoritative.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

RUNTIME_HEADER = """# AUTO-GENERATED from uv.lock (ADR-018) — do not edit manually.
# Regenerate with:
#   uv export --frozen --no-hashes --no-emit-project --no-dev \\
#     | grep -viE '^(mojo|mojo-compiler|mojo-compiler-mojo-libs|mojo-lldb-libs|mblack)==' > requirements.txt
# These files exist for pip-only contexts (Docker fallback, CI). The Mojo
# compiler chain is intentionally excluded (it installs only from the
# Modular PyPI index via uv, not plain pip). uv.lock is the source of truth.
#
"""

DEV_HEADER = """# AUTO-GENERATED from uv.lock (ADR-018) — do not edit manually.
# Regenerate with:
#   uv export --frozen --no-hashes --no-emit-project --all-groups \\
#     | grep -viE '^(mojo|mojo-compiler|mojo-compiler-mojo-libs|mojo-lldb-libs|mblack)==' > requirements-dev.txt
# Includes the [dependency-groups] dev + notebook extras. Mojo chain excluded
# (Modular-index only). uv.lock is the source of truth.
#
"""

GENERATED_FILES = ("requirements.txt", "requirements-dev.txt")
EXCLUDED_PACKAGE_LINE = re.compile(
    r"^(?:mojo|mojo-compiler|mojo-compiler-mojo-libs|mojo-lldb-libs|mblack)==",
    re.IGNORECASE,
)
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _export_command(*, include_all_groups: bool) -> list[str]:
    """Build one frozen uv export command without changing ``uv.lock``."""
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-hashes",
        "--no-header",
        "--no-emit-project",
    ]
    command.append("--all-groups" if include_all_groups else "--no-dev")
    return command


def _filter_export(output: str) -> str:
    """Remove Modular-only package rows while preserving uv comment layout."""
    filtered = "".join(line for line in output.splitlines(keepends=True) if not EXCLUDED_PACKAGE_LINE.match(line))
    if filtered and not filtered.endswith("\n"):
        filtered += "\n"
    return filtered


def generate_requirements(
    repo_root: Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    """Render both pip-only requirements files from the frozen uv lockfile."""
    exports: dict[str, str] = {}
    specifications = (
        ("requirements.txt", RUNTIME_HEADER, False),
        ("requirements-dev.txt", DEV_HEADER, True),
    )
    for filename, header, include_all_groups in specifications:
        completed = runner(
            _export_command(include_all_groups=include_all_groups),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        exports[filename] = header + _filter_export(completed.stdout)
    return exports


def sync_requirements(
    repo_root: Path,
    *,
    runner: Runner = subprocess.run,
) -> list[Path]:
    """Write changed requirements exports and leave matching files untouched."""
    generated = generate_requirements(repo_root, runner=runner)
    paths: list[Path] = []
    for filename in GENERATED_FILES:
        path = repo_root / filename
        content = generated[filename]
        if path.exists() and path.read_text(encoding="utf-8") == content:
            print(f"Unchanged {path}")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"Wrote {path}")
        paths.append(path)
    return paths


def check_requirements_up_to_date(
    repo_root: Path,
    *,
    runner: Runner = subprocess.run,
) -> bool:
    """Return whether both checked-in exports exactly match ``uv.lock``."""
    generated = generate_requirements(repo_root, runner=runner)
    current = True
    for filename in GENERATED_FILES:
        path = repo_root / filename
        if not path.is_file():
            print(f"ERROR: missing generated file: {path}", file=sys.stderr)
            current = False
            continue
        if path.read_text(encoding="utf-8") != generated[filename]:
            print(
                f"ERROR: {path} is stale; run 'python scripts/sync_requirements.py --repo-root {repo_root}'",
                file=sys.stderr,
            )
            current = False
    return current


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronize exports, or verify them without writing via ``--check``."""
    parser = argparse.ArgumentParser(
        description="Synchronize requirements*.txt from uv.lock",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify both exports are current without writing",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root containing pyproject.toml and uv.lock",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    try:
        if args.check:
            if check_requirements_up_to_date(repo_root):
                print("OK: requirements exports are up-to-date")
                return 0
            return 1

        sync_requirements(repo_root)
        if not check_requirements_up_to_date(repo_root):
            print("ERROR: requirements export verification failed", file=sys.stderr)
            return 1
        print("OK: requirements exports synchronized and verified")
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: unable to export requirements: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
