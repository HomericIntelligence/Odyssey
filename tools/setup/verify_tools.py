#!/usr/bin/env python3

"""
Tool: setup/verify_tools.py
Purpose: Verify ML Odyssey tools installation and configuration

Language: Python
Justification:
  - Subprocess execution for tool verification
  - Environment detection and validation
  - Cross-platform compatibility
  - No performance requirements

Reference: ADR-001
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Tuple, Optional


class Color:
    """ANSI color codes"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def run_command(cmd: list) -> Tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr

    Args:
        cmd: Command to run as list of arguments (secure from injection)

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"
    except Exception as e:
        return 1, "", str(e)


def check_item(name: str, passed: bool, details: str = "", verbose: bool = False):
    """Print check result"""
    status = f"{Color.GREEN}✓{Color.RESET}" if passed else f"{Color.RED}✗{Color.RESET}"
    print(f"{status} {name}", end="")
    if details and (verbose or not passed):
        print(f" - {details}")
    elif not verbose:
        print()
    else:
        print()


def get_repo_root() -> Optional[Path]:
    """Get git repository root"""
    code, stdout, _ = run_command(["git", "rev-parse", "--show-toplevel"])
    if code == 0:
        return Path(stdout)
    return None


def check_prerequisites(verbose: bool = False) -> int:
    """Check system prerequisites"""
    print(f"\n{Color.BOLD}Prerequisites{Color.RESET}")

    errors = 0

    # Python version
    version = sys.version_info
    python_ok = version.major == 3 and version.minor >= 8
    check_item(
        "Python",
        python_ok,
        f"{version.major}.{version.minor}.{version.micro} (requires 3.8+)",
        verbose,
    )
    if not python_ok:
        errors += 1

    # Mojo
    code, stdout, _ = run_command(["mojo", "--version"])
    mojo_ok = code == 0
    check_item("Mojo", mojo_ok, stdout if mojo_ok else "Not found", verbose)
    if not mojo_ok:
        errors += 1

    # Git
    code, stdout, _ = run_command(["git", "--version"])
    git_ok = code == 0
    check_item("Git", git_ok, stdout.split()[2] if git_ok else "Not found", verbose)
    if not git_ok:
        errors += 1

    # Repository root
    repo_root = get_repo_root()
    repo_ok = repo_root is not None
    check_item("Repository", repo_ok, str(repo_root) if repo_ok else "Not in git repo", verbose)
    if not repo_ok:
        errors += 1

    return errors


def check_python_dependencies(verbose: bool = False) -> int:
    """Check Python package dependencies"""
    print(f"\n{Color.BOLD}Python Dependencies{Color.RESET}")

    # Whitelist of allowed package names (prevents any injection attempts)
    ALLOWED_PACKAGES = {"jinja2", "yaml", "click"}

    packages = [
        ("jinja2", "Template engine"),
        ("yaml", "YAML parser"),
        ("click", "CLI framework (optional)"),
    ]

    errors = 0
    for package, description in packages:
        # Validate package name against whitelist
        if package not in ALLOWED_PACKAGES:
            check_item(package, False, f"Invalid package name - {description}", verbose)
            errors += 1
            continue

        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            check_item(package, True, f"{version} - {description}", verbose)
        except ImportError:
            check_item(package, False, f"Not installed - {description}", verbose)
            if package != "click":  # click is optional
                errors += 1

    return errors


def check_tool_structure(repo_root: Path, verbose: bool = False) -> int:
    """Check tools directory structure"""
    print(f"\n{Color.BOLD}Tool Structure{Color.RESET}")

    required_items = [
        ("tools/README.md", "Main documentation"),
        ("tools/INTEGRATION.md", "Integration guide"),
        ("tools/CATALOG.md", "Tool catalog"),
        ("tools/INSTALL.md", "Installation guide"),
        ("tools/paper-scaffold/", "Paper scaffolding"),
        ("tools/test-utils/", "Test utilities"),
        ("tools/benchmarking/", "Benchmarking tools"),
        ("tools/codegen/", "Code generation"),
        ("tools/setup/", "Setup scripts"),
    ]

    errors = 0
    for item, description in required_items:
        full_path = repo_root / item
        exists = full_path.exists()
        check_item(item, exists, description, verbose)
        if not exists:
            errors += 1

    return errors


def check_tool_readmes(repo_root: Path, verbose: bool = False) -> int:
    """Check that each tool category has a README"""
    print(f"\n{Color.BOLD}Tool Documentation{Color.RESET}")

    tool_dirs = [
        "paper-scaffold",
        "test-utils",
        "benchmarking",
        "codegen",
    ]

    errors = 0
    for tool_dir in tool_dirs:
        readme_path = repo_root / "tools" / tool_dir / "README.md"
        exists = readme_path.exists()
        check_item(f"{tool_dir}/README.md", exists, "", verbose)
        if not exists:
            errors += 1

    return errors


def check_directories(repo_root: Path, verbose: bool = False) -> int:
    """Check required directories exist"""
    print(f"\n{Color.BOLD}Output Directories{Color.RESET}")

    dirs = [
        ("benchmarks", "Benchmark output"),
        ("logs", "Log files"),
    ]

    errors = 0
    for dirname, description in dirs:
        dir_path = repo_root / dirname
        exists = dir_path.exists() and dir_path.is_dir()
        check_item(dirname, exists, description, verbose)
        if not exists:
            errors += 1

    return errors


def main():
    """Main verification function"""
    parser = argparse.ArgumentParser(description="Verify ML Odyssey tools installation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"{Color.BOLD}{Color.BLUE}ML Odyssey Tools Verification{Color.RESET}")

    # Get repository root
    repo_root = get_repo_root()
    if not repo_root:
        print(f"\n{Color.RED}Error: Not in a git repository{Color.RESET}")
        print("Please run from Odyssey directory")
        return 1

    # Run checks
    total_errors = 0
    total_errors += check_prerequisites(args.verbose)
    total_errors += check_python_dependencies(args.verbose)
    total_errors += check_tool_structure(repo_root, args.verbose)
    total_errors += check_tool_readmes(repo_root, args.verbose)
    total_errors += check_directories(repo_root, args.verbose)

    # Summary
    print(f"\n{Color.BOLD}Summary{Color.RESET}")
    if total_errors == 0:
        print(f"{Color.GREEN}All checks passed!{Color.RESET}")
        print("\nTools are ready to use.")
        print(f"  - Integration guide: {repo_root}/tools/INTEGRATION.md")
        print(f"  - Tool catalog: {repo_root}/tools/CATALOG.md")
        return 0
    else:
        print(f"{Color.RED}{total_errors} error(s) found{Color.RESET}")
        print("\nPlease fix the errors above.")
        print(f"  - Installation guide: {repo_root}/tools/INSTALL.md")
        print(f"  - Run installer: python3 {repo_root}/tools/setup/install_tools.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
