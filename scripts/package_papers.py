#!/usr/bin/env python3
"""
Package papers/ directory into a distributable source tarball.

This script creates a compressed tarball of the papers/ directory structure,
making it easy to distribute and install the papers collection.

Usage:
    python scripts/package_papers.py [--output OUTPUT_DIR]
"""

import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from hephaestus.utils import get_repo_root


def create_papers_tarball(repo_root: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Create a compressed tarball of the papers/ directory.

    Args:
        repo_root: Path to the repository root
        output_dir: Optional output directory for tarball (defaults to repo_root/dist)

    Returns:
        Path to the created tarball

    Raises:
        FileNotFoundError: If papers/ directory doesn't exist
        PermissionError: If lacking permissions to create tarball
    """
    papers_dir = repo_root / "papers"

    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    if not papers_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {papers_dir}")

    # Default output to dist/ directory
    if output_dir is None:
        output_dir = repo_root / "dist"

    # Create dist directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tarball name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    tarball_name = f"papers-{timestamp}.tar.gz"
    tarball_path = output_dir / tarball_name

    # Create the tarball
    print(f"Creating tarball: {tarball_path}")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(papers_dir, arcname="papers")

    print(f"Successfully created tarball: {tarball_path}")
    print(f"Size: {tarball_path.stat().st_size:,} bytes")

    return tarball_path


def main() -> int:
    """
    Main entry point for the packaging script.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(description="Package papers/ directory into a distributable tarball")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for tarball (default: <repo>/dist)",
    )

    args = parser.parse_args()

    try:
        # Determine repository root
        repo_root = get_repo_root()
        print(f"Repository root: {repo_root}")

        # Create tarball
        tarball_path = create_papers_tarball(repo_root, args.output)

        print("\nPackaging complete!")
        print(f"Tarball: {tarball_path}")
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Please check your permissions and try again.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
