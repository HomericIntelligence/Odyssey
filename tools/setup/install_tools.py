#!/usr/bin/env python3

"""
Tool: setup/install_tools.py
Purpose: Automated installation and setup for ML Odyssey tools

Language: Python
Justification:
  - Subprocess management for environment detection
  - Cross-platform compatibility checks
  - Package installation automation
  - File system operations
  - No performance requirements

Reference: ADR-001
"""

import sys
import subprocess
import platform
from pathlib import Path
from typing import Tuple, Optional


class Color:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(msg: str):
    """Print section header"""
    print(f"\n{Color.BOLD}{Color.BLUE}{'=' * 60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{msg}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'=' * 60}{Color.RESET}\n")


def print_success(msg: str):
    """Print success message"""
    print(f"{Color.GREEN}✓{Color.RESET} {msg}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"{Color.YELLOW}⚠{Color.RESET} {msg}")


def print_error(msg: str):
    """Print error message"""
    print(f"{Color.RED}✗{Color.RESET} {msg}")


def run_command(cmd: list, check: bool = False) -> Tuple[int, str, str]:
    """
    Run shell command and return exit code, stdout, stderr

    Args:
        cmd: Command to run as list of arguments (secure from injection)
        check: Raise exception on non-zero exit

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def get_repo_root() -> Optional[Path]:
    """Get repository root directory"""
    code, stdout, _ = run_command(["git", "rev-parse", "--show-toplevel"])
    if code == 0:
        return Path(stdout)
    return None


def check_python_version() -> bool:
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_mojo() -> bool:
    """Check if Mojo is installed"""
    code, stdout, _ = run_command(["mojo", "--version"])
    if code == 0:
        print_success(f"Mojo {stdout}")
        return True
    else:
        print_warning("Mojo not found (required for Mojo tools)")
        print("  Install from: https://docs.modular.com/mojo/")
        return False


def check_git() -> bool:
    """Check if Git is installed"""
    code, stdout, _ = run_command(["git", "--version"])
    if code == 0:
        print_success(f"Git {stdout.split()[2]}")
        return True
    else:
        print_error("Git not found")
        return False


def detect_platform() -> str:
    """Detect operating system platform"""
    system = platform.system()
    if system == "Linux":
        return "Linux"
    elif system == "Darwin":
        return "macOS"
    elif system == "Windows":
        return "Windows (WSL)" if "microsoft" in platform.uname().release.lower() else "Windows"
    else:
        return system


def install_python_dependencies(repo_root: Path) -> bool:
    """Install Python dependencies from requirements.txt"""
    req_file = repo_root / "tools" / "requirements.txt"

    if not req_file.exists():
        print_warning(f"No requirements.txt found at {req_file}")
        print("  Creating minimal requirements.txt...")

        # Create basic requirements file
        with open(req_file, "w") as f:
            f.write("# Python dependencies for ML Odyssey tools\n\n")
            f.write("# Template engine (paper scaffolding, code generation)\n")
            f.write("jinja2>=3.0.0\n\n")
            f.write("# YAML parsing (configuration)\n")
            f.write("pyyaml>=6.0\n\n")
            f.write("# Optional: CLI framework\n")
            f.write("click>=8.0.0\n\n")
            f.write("# Optional: Visualization (benchmarking reports)\n")
            f.write("# matplotlib>=3.5.0\n")
            f.write("# pandas>=1.3.0\n")

        print_success(f"Created {req_file}")

    print(f"\nInstalling Python dependencies from {req_file}...")
    code, stdout, stderr = run_command(["pip", "install", "-r", str(req_file)])

    if code == 0:
        print_success("Python dependencies installed")
        return True
    else:
        print_error("Failed to install Python dependencies")
        print(f"  Error: {stderr}")
        return False


def create_directories(repo_root: Path) -> bool:
    """Create necessary directories for tools"""
    dirs = [
        repo_root / "benchmarks",
        repo_root / "logs",
    ]

    all_created = True
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Directory: {directory.relative_to(repo_root)}")
        except Exception as e:
            print_error(f"Failed to create {directory}: {e}")
            all_created = False

    return all_created


def verify_tool_structure(repo_root: Path) -> bool:
    """Verify tools directory structure exists"""
    required_dirs = [
        "tools",
        "tools/paper-scaffold",
        "tools/test-utils",
        "tools/benchmarking",
        "tools/codegen",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        if full_path.exists():
            print_success(f"Found: {dir_path}/")
        else:
            print_error(f"Missing: {dir_path}/")
            all_exist = False

    return all_exist


def main():
    """Main installation function"""
    print_header("ML Odyssey Tools Installation")

    # Detect platform
    platform_name = detect_platform()
    print(f"Platform: {platform_name}\n")

    # Check prerequisites
    print_header("Checking Prerequisites")

    python_ok = check_python_version()
    git_ok = check_git()
    mojo_ok = check_mojo()

    if not python_ok or not git_ok:
        print_error("\nCritical prerequisites missing. Please install and retry.")
        return 1

    # Get repository root
    repo_root = get_repo_root()
    if not repo_root:
        print_error("Not in a git repository. Please run from Odyssey directory.")
        return 1

    print_success(f"Repository root: {repo_root}")

    # Verify tool structure
    print_header("Verifying Tools Structure")
    if not verify_tool_structure(repo_root):
        print_error("\nTools directory structure incomplete.")
        print("  This is expected during development. Tools will be added incrementally.")

    # Create directories
    print_header("Creating Directories")
    if not create_directories(repo_root):
        print_warning("Some directories could not be created")

    # Install Python dependencies
    print_header("Installing Python Dependencies")
    deps_ok = install_python_dependencies(repo_root)

    # Final status
    print_header("Installation Summary")

    if python_ok and git_ok and deps_ok:
        print_success("Installation completed successfully!")

        if not mojo_ok:
            print_warning("Mojo not found - some tools will not work")
            print("  Install Mojo: https://docs.modular.com/mojo/")

        print("\nNext steps:")
        print(f"  1. Run verification: python3 {repo_root}/tools/setup/verify_tools.py")
        print(f"  2. Read integration guide: {repo_root}/tools/INTEGRATION.md")
        print(f"  3. Browse tool catalog: {repo_root}/tools/CATALOG.md")
        return 0
    else:
        print_error("Installation completed with errors")
        print("\nPlease fix the errors above and retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
