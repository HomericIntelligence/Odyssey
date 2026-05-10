#!/usr/bin/env python3
"""
Check that version numbers are in sync across all version files.

Source of truth: pyproject.toml [project] version field.
Verified against: pixi.toml, mojo.toml, VERSION

Usage:
    python scripts/check_version_sync.py

Exit codes:
    0 — all files in sync
    1 — version drift detected
"""

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent

VERSION_FILE = REPO_ROOT / "VERSION"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
PIXI_TOML = REPO_ROOT / "pixi.toml"
MOJO_TOML = REPO_ROOT / "mojo.toml"


def read_pyproject_version() -> str:
    """Read version from pyproject.toml [project] section (authoritative source)."""
    text = PYPROJECT_TOML.read_text()
    # Match version under [project] section only
    match = re.search(
        r"^\[project\].*?^version\s*=\s*[\"']([^\"']+)[\"']",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Could not find [project] version in {PYPROJECT_TOML}")
    return match.group(1)


def read_pixi_version() -> str:
    """Read version from pixi.toml [package] section."""
    text = PIXI_TOML.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {PIXI_TOML}")
    return match.group(1)


def read_mojo_version() -> str:
    """Read version from mojo.toml."""
    text = MOJO_TOML.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {MOJO_TOML}")
    return match.group(1)


def read_version_file() -> str:
    """Read version from VERSION plain-text file."""
    return VERSION_FILE.read_text().strip()


def check_sync() -> int:
    """Check all version files. Returns 0 if in sync, 1 if drift detected."""
    canonical = read_pyproject_version()

    checks = [
        ("pixi.toml", read_pixi_version),
        ("mojo.toml", read_mojo_version),
        ("VERSION", read_version_file),
    ]

    drift = False
    print(f"Canonical version (pyproject.toml): {canonical}")
    for name, reader in checks:
        try:
            found = reader()
            status = "OK" if found == canonical else "DRIFT"
            print(f"  {name}: {found} [{status}]")
            if found != canonical:
                drift = True
        except ValueError as e:
            print(f"  ERROR reading {name}: {e}")
            drift = True

    if drift:
        print("\nVersion drift detected! Run:\n  just bump-version <new_version>\nto update all files atomically.")
        return 1

    print("\nAll version files are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(check_sync())
