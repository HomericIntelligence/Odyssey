#!/usr/bin/env python3
"""
Shared utilities and constants for ML Odyssey scripts

This module provides common functionality used across multiple scripts to avoid duplication.

Centralized Utilities:
- LABEL_COLORS: GitHub label colors for 5-phase workflow
- get_repo_root(): Repository root finder (used by 20+ scripts)
- get_agents_dir(): .claude/agents path helper
- Colors: ANSI terminal colors with disable() method

Related Issues:
- #2601: Colors class centralization
- #2603: get_repo_root() audit
- #2634: Script utilities consolidation
"""

from pathlib import Path


# Label colors for GitHub issues (5-phase development workflow)
# Used by: create_issues.py, create_single_component_issues.py
LABEL_COLORS = {
    "planning": "d4c5f9",  # Light purple
    "documentation": "0075ca",  # Blue
    "testing": "fbca04",  # Yellow
    "tdd": "fbca04",  # Yellow
    "implementation": "1d76db",  # Dark blue
    "packaging": "c2e0c6",  # Light green
    "integration": "c2e0c6",  # Light green
    "cleanup": "d93f0b",  # Red
}


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Searches upward from the current file location until finding a directory
    containing a .git folder.

    Returns:
        Path to repository root

    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = Path(__file__).resolve().parent

    # Search upward for .git directory
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    raise RuntimeError("Could not find repository root (no .git directory found)")


def get_agents_dir() -> Path:
    """
    Get the .claude/agents directory path.

    Returns:
        Path to .claude/agents directory

    Raises:
        RuntimeError: If agents directory doesn't exist
    """
    repo_root = get_repo_root()
    agents_dir = repo_root / ".claude" / "agents"

    if not agents_dir.exists():
        raise RuntimeError(f"Agents directory not found: {agents_dir}")

    return agents_dir


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def disable() -> None:
        """Disable colors for non-terminal output.

        Sets all color codes to empty strings, useful for piping
        output to files or non-TTY streams.
        """
        Colors.HEADER = ""
        Colors.OKBLUE = ""
        Colors.OKCYAN = ""
        Colors.OKGREEN = ""
        Colors.WARNING = ""
        Colors.FAIL = ""
        Colors.ENDC = ""
        Colors.BOLD = ""
        Colors.UNDERLINE = ""
