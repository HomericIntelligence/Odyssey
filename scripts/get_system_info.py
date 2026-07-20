#!/usr/bin/env python3

"""
System Information Collection Script

Collects comprehensive system information for bug reports and issue creation.
Gathers OS details, Python/Mojo/Pixi versions, Git information, and environment variables.

Usage:
    python3 scripts/get_system_info.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple


def run_command(cmd: list, capture_output: bool = True) -> Tuple[bool, str]:
    """
    Run a shell command and return success status and output.

    Args:
        cmd: Command as list of strings
        capture_output: Whether to capture stdout/stderr

    Returns:
        Tuple of (success: bool, output: str).
    """
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=5)
        return (result.returncode == 0, result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return (False, "")


def get_command_path(cmd: str) -> Optional[str]:
    """
    Get the full path of a command if it exists.

    Args:
        cmd: Command name to locate

    Returns:
        Full path to command or None if not found
    """
    success, output = run_command(["which", cmd])
    return output if success and output else None


def get_os_info() -> str:
    """Get operating system information."""
    system = platform.system()

    if system == "Linux":
        # Try to read /etc/os-release
        os_release_path = Path("/etc/os-release")
        if os_release_path.exists():
            try:
                with open(os_release_path) as f:
                    lines = f.readlines()
                    os_data = {}
                    for line in lines:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            os_data[key] = value.strip('"')

                    name = os_data.get("NAME", "Linux")
                    version = os_data.get("VERSION", "")
                    return f"{name} {version}".strip()
            except Exception:
                pass
        return "Linux (unknown distribution)"

    elif system == "Darwin":
        # macOS
        success, version = run_command(["sw_vers", "-productVersion"])
        if success:
            return f"macOS {version}"
        return "macOS (unknown version)"

    elif system == "Windows":
        return f"Windows {platform.release()}"

    else:
        return f"{system} (unknown)"


def get_tool_info(
    tool_name: str,
    version_flag: str = "--version",
    version_extract: Optional[Callable[[str], str]] = None,
) -> Tuple[str, str]:
    """
    Get version and path information for a tool.

    Args:
        tool_name: Name of the tool
        version_flag: Flag to get version (default: --version)
        version_extract: Optional function to extract version from output

    Returns:
        Tuple of (version: str, path: str).
    """
    path = get_command_path(tool_name)

    if not path:
        return ("Not found", "")

    success, output = run_command([tool_name, version_flag])

    if not success or not output:
        version = "Unable to determine"
    elif version_extract:
        version = version_extract(output)
    else:
        version = output

    return (version, path)


def extract_version_word(output: str, word_index: int = 1) -> str:
    """Extract version by splitting output and taking word at index."""
    parts = output.split()
    return parts[word_index] if len(parts) > word_index else output


def main():
    """Main function to collect and display system information."""
    print("=== System Information ===")
    print()

    # Operating System
    print("OS Information:")
    print(f"  OS: {get_os_info()}")
    print()

    # Kernel/Architecture
    print("Architecture:")
    print(f"  Kernel: {platform.system()}")
    print(f"  Machine: {platform.machine()}")
    print()

    # Python version
    print("Python:")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_path = sys.executable
    print(f"  Version: {python_version}")
    print(f"  Path: {python_path}")
    print()

    # Mojo version
    print("Mojo:")
    mojo_version, mojo_path = get_tool_info("mojo")
    if mojo_path:
        print(f"  Version: {mojo_version}")
        print(f"  Path: {mojo_path}")
    else:
        print("  Not found")
    print()

    # uv version
    print("uv:")
    uv_version, uv_path = get_tool_info("uv", version_extract=lambda x: extract_version_word(x, 1))
    if uv_path:
        print(f"  Version: {uv_version}")
        print(f"  Path: {uv_path}")
    else:
        print("  Not found")
    print()

    # Git version
    print("Git:")
    git_version, git_path = get_tool_info("git", version_extract=lambda x: extract_version_word(x, 2))
    if git_path:
        print(f"  Version: {git_version}")
        print(f"  Path: {git_path}")
    else:
        print("  Not found")
    print()

    # Current directory and git status
    print("Current Directory:")
    current_dir = os.getcwd()
    print(f"  Path: {current_dir}")

    # Check if in a git repository
    success, _ = run_command(["git", "rev-parse", "--git-dir"])
    if success:
        print("  Git Repository: Yes")

        # Get current branch
        branch_success, branch = run_command(["git", "branch", "--show-current"])
        if branch_success and branch:
            print(f"  Branch: {branch}")

        # Get current commit
        commit_success, commit = run_command(["git", "rev-parse", "--short", "HEAD"])
        if commit_success and commit:
            print(f"  Commit: {commit}")
    else:
        print("  Git Repository: No")
    print()

    # Environment variables (selected)
    print("Environment:")
    shell = os.environ.get("SHELL", "Not set")
    lang = os.environ.get("LANG", "Not set")
    print(f"  SHELL: {shell}")
    print(f"  LANG: {lang}")
    print()

    print("=== End System Information ===")


if __name__ == "__main__":
    main()
