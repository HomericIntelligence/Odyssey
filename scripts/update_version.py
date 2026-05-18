#!/usr/bin/env python3
"""
Update version across all ML Odyssey version files.

This script updates version numbers in:
- VERSION (root file)
- src/projectodyssey/version.mojo (Mojo version module)

Usage:
    python3 scripts/update_version.py <new_version>

Example:
    ```python
    python3 scripts/update_version.py 0.2.0
    ```
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

from common import get_repo_root


def parse_version(version: str) -> Tuple[int, int, int]:
    """
    Parse version string into components.

    Args:
        version: Version string in format "MAJOR.MINOR.PATCH"

    Returns:
        Tuple of (major, minor, patch)

    Raises:
        ValueError: If version format is invalid
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}. Expected format: MAJOR.MINOR.PATCH (e.g., 0.1.0)")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))

    return (major, minor, patch)


def update_version_file(repo_root: Path, version: str) -> None:
    """
    Update VERSION file in repository root.

    Args:
        repo_root: Path to repository root
        version: New version string
    """
    version_file = repo_root / "VERSION"

    print(f"Updating {version_file}...")
    version_file.write_text(f"{version}\n")
    print(f"  ✓ Updated to {version}")


def update_version_mojo(repo_root: Path, version: str, major: int, minor: int, patch: int) -> None:
    """
    Update src/projectodyssey/version.mojo with new version.

    Args:
        repo_root: Path to repository root
        version: Version string
        major: Major version number
        minor: Minor version number
        patch: Patch version number
    """
    version_mojo = repo_root / "shared" / "version.mojo"

    if not version_mojo.exists():
        print(f"  ⚠️  Warning: {version_mojo} not found, skipping")
        return

    print(f"Updating {version_mojo}...")

    content = version_mojo.read_text()

    # Update version string
    content = re.sub(r'alias VERSION = "[^"]+"', f'alias VERSION = "{version}"', content)

    # Update version components
    content = re.sub(r"alias VERSION_MAJOR = \d+", f"alias VERSION_MAJOR = {major}", content)
    content = re.sub(r"alias VERSION_MINOR = \d+", f"alias VERSION_MINOR = {minor}", content)
    content = re.sub(r"alias VERSION_PATCH = \d+", f"alias VERSION_PATCH = {patch}", content)

    version_mojo.write_text(content)
    print(f"  ✓ Updated VERSION = {version}")
    print(f"  ✓ Updated VERSION_MAJOR = {major}")
    print(f"  ✓ Updated VERSION_MINOR = {minor}")
    print(f"  ✓ Updated VERSION_PATCH = {patch}")


def verify_version_files(repo_root: Path, version: str) -> bool:
    """
    Verify that all version files are consistent.

    Args:
        repo_root: Path to repository root
        version: Expected version string

    Returns:
        True if all files consistent, False otherwise
    """
    print("\nVerifying version files...")

    version_file = repo_root / "VERSION"
    version_mojo = repo_root / "shared" / "version.mojo"

    success = True

    # Check VERSION file
    if version_file.exists():
        content = version_file.read_text().strip()
        if content == version:
            print(f"  ✓ VERSION: {content}")
        else:
            print(f"  ✗ VERSION: {content} (expected {version})")
            success = False
    else:
        print("  ✗ VERSION file not found")
        success = False

    # Check version.mojo
    if version_mojo.exists():
        content = version_mojo.read_text()
        if f'alias VERSION = "{version}"' in content:
            print(f"  ✓ src/projectodyssey/version.mojo: {version}")
        else:
            print("  ✗ src/projectodyssey/version.mojo: version mismatch")
            success = False
    else:
        print("  ⚠️  src/projectodyssey/version.mojo not found (optional)")

    return success


def main() -> int:
    """
    Main entry point for version update script.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(description="Update ML Odyssey version across all version files")
    parser.add_argument("version", help="New version string (format: MAJOR.MINOR.PATCH, e.g., 0.1.0)")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify version consistency, don't update",
    )

    args = parser.parse_args()

    try:
        # Parse and validate version
        major, minor, patch = parse_version(args.version)
        print(f"Parsed version: {args.version} (major={major}, minor={minor}, patch={patch})")

        # Get repository root
        repo_root = get_repo_root()
        print(f"Repository root: {repo_root}\n")

        if args.verify_only:
            # Verify only
            if verify_version_files(repo_root, args.version):
                print("\n✅ All version files are consistent")
                return 0
            else:
                print("\n❌ Version files are inconsistent")
                return 1
        else:
            # Update all version files
            update_version_file(repo_root, args.version)
            update_version_mojo(repo_root, args.version, major, minor, patch)

            # Verify updates
            if verify_version_files(repo_root, args.version):
                print("\n✅ All version files updated successfully")
                return 0
            else:
                print("\n❌ Version update incomplete")
                return 1

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
