#!/usr/bin/env python3
"""Repository-specific constants and helpers for ML Odyssey scripts.

General-purpose utilities that used to live here — ``get_repo_root()`` and the
``Colors`` ANSI helper — have been migrated to the ``hephaestus`` library
(issue #5061):

- ``get_repo_root`` → ``from hephaestus.utils import get_repo_root``
- ``Colors``        → ``from hephaestus.cli.colors import Colors``

Only the symbols that are specific to this repository's layout and GitHub
workflow remain here, since ``hephaestus`` has no equivalent for them.

Related Issues:
- #5061: Replace duplicated validation scripts with hephaestus
"""

from pathlib import Path

from hephaestus.utils import get_repo_root

# Label colors for GitHub issues (5-phase development workflow).
# Repo-specific: tied to ML Odyssey's planning labels, not a hephaestus concern.
# Used by: create_issues.py
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

# Directories to exclude from test file discovery.
# Used by: check_test_count_badge.py
EXCLUDE_DIRS = [".pixi/", "build/", "dist/", ".git/", "worktrees/"]


def get_agents_dir() -> Path:
    """Get the ``.claude/agents`` directory path.

    Returns:
        Path to the ``.claude/agents`` directory.

    Raises:
        RuntimeError: If the agents directory does not exist.
    """
    agents_dir = get_repo_root() / ".claude" / "agents"

    if not agents_dir.exists():
        raise RuntimeError(f"Agents directory not found: {agents_dir}")

    return agents_dir
