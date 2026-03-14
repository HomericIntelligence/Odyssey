"""Utility helpers for foundation tests."""

import re
from pathlib import Path


def assert_pkg_absent(dockerfile: Path, pkg: str) -> None:
    """Assert a package is absent from a Dockerfile.

    This helper checks that a package is not listed in apt-get install
    commands or installed via direct install commands (e.g., 'pkg install').
    Ignores occurrences in comments (lines starting with #).

    Args:
        dockerfile: Path to the Dockerfile to check.
        pkg: Package name to verify is absent (word-bounded).

    Raises:
        AssertionError: If the package is found in apt-get install or
                       as 'pkg install' command (excluding comments).
    """
    content = dockerfile.read_text()
    lines = content.splitlines()

    # Check apt-get install
    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            continue
        apt_match = re.search(rf"apt-get install[^\n]*\b{pkg}\b", line)
        if apt_match is not None:
            raise AssertionError(
                f"Found '{pkg}' in apt-get install in {dockerfile.name}:{line_num}: {apt_match.group()!r}"
            )

    # Check direct pkg install (e.g., 'cargo install', 'rustc install')
    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            continue
        install_match = re.search(rf"\b{pkg}\s+install\b", line)
        if install_match is not None:
            raise AssertionError(f"Found '{pkg} install' in {dockerfile.name}:{line_num}: {install_match.group()!r}")
