#!/usr/bin/env python3
"""
Directory structure validation script for ML Odyssey

Validates that all required supporting directories exist with proper structure
and required files.

Usage:
    python scripts/validate_structure.py [--verbose]

Exit codes:
    0: All validations passed
    1: One or more validation failures
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hephaestus.utils import get_repo_root

# Required top-level directories
REQUIRED_DIRECTORIES = [
    "benchmarks",
    "docs",
    "agents",
    "tools",
    "configs",
    "papers",
    "shared",
    "tests",
    "scripts",
    "examples",
    "notes",
    ".claude",
    ".github",
]

# Required files in each directory
REQUIRED_FILES = {
    "benchmarks": ["README.md"],
    "docs": ["README.md", "index.md"],
    "agents": ["README.md", "hierarchy.md", "delegation-rules.md"],
    "tools": ["README.md"],
    "configs": ["README.md"],
    "papers": ["README.md", "_template/README.md"],
    "shared": ["README.md"],
    "tests": ["README.md"],
    "scripts": ["README.md"],
}

# Required subdirectories
REQUIRED_SUBDIRS = {
    "benchmarks": ["scripts", "baselines", "results"],
    "docs": ["getting-started", "core", "advanced", "dev"],
    "agents": ["templates", "guides", "docs"],
    "tools": ["paper-scaffold", "test-utils", "benchmarking", "codegen"],
    "configs": ["defaults", "papers", "experiments", "templates"],
    "papers": ["_template"],
    "shared": ["core", "training", "data", "utils"],
    "tests": ["foundation", "shared", "agents", "tools"],
    ".claude": ["agents"],
}


def check_directory_exists(base_path: Path, dir_name: str) -> Tuple[bool, str]:
    """Check if a directory exists"""
    dir_path = base_path / dir_name
    if not dir_path.exists():
        return False, f"Missing directory: {dir_name}/"
    if not dir_path.is_dir():
        return False, f"Not a directory: {dir_name}"
    return True, f"✓ {dir_name}/"


def check_file_exists(base_path: Path, dir_name: str, file_name: str) -> Tuple[bool, str]:
    """Check if a required file exists"""
    file_path = base_path / dir_name / file_name
    if not file_path.exists():
        return False, f"Missing file: {dir_name}/{file_name}"
    if not file_path.is_file():
        return False, f"Not a file: {dir_name}/{file_name}"
    return True, f"✓ {dir_name}/{file_name}"


def check_subdirectory_exists(base_path: Path, parent_dir: str, subdir: str) -> Tuple[bool, str]:
    """Check if a required subdirectory exists"""
    subdir_path = base_path / parent_dir / subdir
    if not subdir_path.exists():
        return False, f"Missing subdirectory: {parent_dir}/{subdir}/"
    if not subdir_path.is_dir():
        return False, f"Not a directory: {parent_dir}/{subdir}"
    return True, f"✓ {parent_dir}/{subdir}/"


def validate_structure(repo_root: Path, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Validate repository directory structure

    Returns:
        Dictionary with 'passed' and 'failed' lists of validation messages
    """
    results: Dict[str, Any] = {"passed": [], "failed": []}

    print("Validating ML Odyssey directory structure...\n")

    # Check required top-level directories
    print("Checking top-level directories...")
    for dir_name in REQUIRED_DIRECTORIES:
        passed, message = check_directory_exists(repo_root, dir_name)
        if passed:
            results["passed"].append(message)
            if verbose:
                print(f"  {message}")
        else:
            results["failed"].append(message)
            print(f"  ✗ {message}")

    # Check required files
    print("\nChecking required files...")
    for dir_name, files in REQUIRED_FILES.items():
        for file_name in files:
            passed, message = check_file_exists(repo_root, dir_name, file_name)
            if passed:
                results["passed"].append(message)
                if verbose:
                    print(f"  {message}")
            else:
                results["failed"].append(message)
                print(f"  ✗ {message}")

    # Check required subdirectories
    print("\nChecking required subdirectories...")
    for parent_dir, subdirs in REQUIRED_SUBDIRS.items():
        for subdir in subdirs:
            passed, message = check_subdirectory_exists(repo_root, parent_dir, subdir)
            if passed:
                results["passed"].append(message)
                if verbose:
                    print(f"  {message}")
            else:
                results["failed"].append(message)
                print(f"  ✗ {message}")

    return results


def print_summary(results: Dict[str, List[str]]) -> None:
    """Print validation summary"""
    total_checks = len(results["passed"]) + len(results["failed"])
    passed = len(results["passed"])
    failed = len(results["failed"])

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed checks:")
        for message in results["failed"]:
            print(f"  - {message}")

    print("=" * 70)


def main() -> int:
    """Main validation function"""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    repo_root = get_repo_root()
    print(f"Repository root: {repo_root}\n")

    results = validate_structure(repo_root, verbose)
    print_summary(results)

    # Return exit code based on results
    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
