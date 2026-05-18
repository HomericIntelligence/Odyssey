#!/usr/bin/env python3
"""Orchestrate parallel implementation of GitHub issues using worktrees.

This script parses an epic issue to extract dependencies, spawns parallel
workers in isolated git worktrees, and uses Claude haiku to implement each issue.

Design goals (following plan_issues.py and analyze_issues_claude.py patterns):
- Dependency-aware scheduling (topological order)
- Thread-safe worktree management
- Live status display with StatusTracker
- Resumable state persistence
- Pause on CI failure for manual intervention

Mojo Requirements:
- Requires Mojo v0.26.1 or later
- Language reference: https://docs.modular.com/mojo/manual/
- Uses modern Mojo syntax (list literals, @fieldwise_init, etc.)
"""

from __future__ import annotations

import argparse
import atexit
import collections
import curses
import dataclasses
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile

# Enable importing from scripts/common.py
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import get_repo_root
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import FrameType
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, TextIO

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[import,no-redef]

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_PARALLEL_DEFAULT = 4
ISSUE_TIMEOUT = 1800  # 30 minutes per issue
PR_POLL_INTERVAL = 30  # seconds between PR status checks
PR_MAX_WAIT = 3600  # 1 hour max wait for CI
PLAN_COMMENT_HEADER = "## Detailed Implementation Plan"
MAX_RETRIES = 3
STATUS_REFRESH_INTERVAL = 0.1
API_DELAY_DEFAULT = 1.0  # seconds between GitHub API calls to avoid rate limiting

# Rate limit detection
RATE_LIMIT_RE = re.compile(
    r"Limit reached.*resets\s+(?P<time>[0-9:apm]+)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)

ALLOWED_TIMEZONES = {
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Phoenix",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Asia/Tokyo",
}

# Time constants
NOON_HOUR = 12
MIDNIGHT_HOUR = 0


# ---------------------------------------------------------------------
# Secure file helpers
# ---------------------------------------------------------------------


def write_secure(path: pathlib.Path, content: str) -> None:
    """Write content to file with secure permissions (owner-only read/write).

    Uses os.open() with mode flags to avoid TOCTOU race condition.
    File is created with 0o600 permissions atomically.
    """
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open with secure permissions from the start (no TOCTOU window)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)


# ---------------------------------------------------------------------
# Dependency Validation
# ---------------------------------------------------------------------


def check_dependencies() -> None:
    """Verify required external dependencies are available.

    Raises:
        RuntimeError: If any required command is missing.
    """
    required = ["gh", "git", "claude", "python3"]
    missing = []

    for cmd in required:
        if not shutil.which(cmd):
            missing.append(cmd)

    if missing:
        raise RuntimeError(
            f"Required command(s) not found: {', '.join(missing)}\n"
            f"Please install missing dependencies before running this script."
        )

    # Verify gh CLI is authenticated
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("GitHub CLI (gh) is not authenticated.\nRun 'gh auth login' to authenticate.")


def health_check() -> int:
    """Check and display status of all required dependencies.

    Displays version information and availability status for:
    - gh (GitHub CLI)
    - git (Version control)
    - claude (Claude Code CLI)
    - python3 (Python interpreter)

    Exit codes:
        0: All dependencies available and working
        1: One or more dependencies missing or non-functional
    """
    required = {
        "gh": ["gh", "--version"],
        "git": ["git", "--version"],
        "claude": ["claude", "--version"],
        "python3": ["python3", "--version"],
    }

    log("INFO", "=" * 80)
    log("INFO", "DEPENDENCY HEALTH CHECK")
    log("INFO", "=" * 80)
    log("INFO", "")

    all_ok = True
    for cmd, version_cmd in required.items():
        # Check if command exists
        cmd_path = shutil.which(cmd)

        if not cmd_path:
            log("INFO", f"✗ {cmd:12} NOT FOUND")
            all_ok = False
            continue

        # Get version information
        try:
            result = subprocess.run(version_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract first line of version output
                version = result.stdout.strip().split("\n")[0]
                log("INFO", f"✓ {cmd:12} {version}")
            else:
                log("INFO", f"⚠ {cmd:12} FOUND but version check failed")
                all_ok = False
        except Exception as e:
            log("INFO", f"⚠ {cmd:12} FOUND but error getting version: {e}")
            all_ok = False

    log("INFO", "")
    log("INFO", "-" * 80)
    log("INFO", "GitHub CLI Authentication")
    log("INFO", "-" * 80)

    # Check gh auth status
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse auth status output
            auth_lines = result.stderr.strip().split("\n")  # gh outputs to stderr
            log("INFO", "✓ GitHub CLI authenticated")
            for line in auth_lines[:3]:  # Show first 3 lines
                log("INFO", f"  {line}")
        else:
            log("INFO", "✗ GitHub CLI NOT authenticated")
            log("INFO", "  Run 'gh auth login' to authenticate")
            all_ok = False
    except Exception as e:
        log("INFO", f"⚠ Error checking gh auth: {e}")
        all_ok = False

    log("INFO", "")
    log("INFO", "=" * 80)
    if all_ok:
        log("INFO", "✓ ALL DEPENDENCIES OK")
        log("INFO", "=" * 80)
        return 0
    else:
        log("INFO", "✗ SOME DEPENDENCIES MISSING OR NON-FUNCTIONAL")
        log("INFO", "=" * 80)
        return 1


def export_dependency_graph(issues: dict[int, IssueInfo], output_path: str) -> int:
    """Export dependency graph to Graphviz DOT format.

    Args:
        issues: Dictionary of issue number to IssueInfo
        output_path: Path to output DOT file

    Returns:
        0 on success, 1 on failure
    """
    try:
        with open(output_path, "w") as f:
            f.write("digraph Dependencies {\n")
            f.write("  rankdir=LR;\n")
            f.write("  node [shape=box];\n")
            f.write("\n")

            # Write nodes
            for issue_num, info in sorted(issues.items()):
                # Escape quotes in title
                title = info.title.replace('"', '\\"')
                # Truncate long titles
                if len(title) > 40:
                    title = title[:37] + "..."

                # Color by priority
                color = {
                    "P0": "red",
                    "P1": "orange",
                    "P2": "yellow",
                }.get(info.priority, "gray")

                # Style by status
                style = "solid"
                if info.status == "completed":
                    style = "filled"
                    color = "lightgreen"

                f.write(f'  {issue_num} [label="#{issue_num}: {title}\\n({info.priority})", ')
                f.write(f'color="{color}", style="{style}"];\n')

            f.write("\n")

            # Write edges
            for issue_num, info in sorted(issues.items()):
                for dep in sorted(info.depends_on):
                    # Only draw edge if dependency is in the issue set
                    if dep in issues:
                        f.write(f"  {dep} -> {issue_num};\n")
                    else:
                        # External dependency - show as dashed line
                        f.write(f'  ext{dep} [label="#{dep} (external)", shape=ellipse, color=blue];\n')
                        f.write(f"  ext{dep} -> {issue_num} [style=dashed, color=blue];\n")

            f.write("}\n")

        log("INFO", f"✓ Exported dependency graph to {output_path}")
        log("INFO", f"  Visualize with: dot -Tpng {output_path} -o graph.png")
        return 0

    except Exception as e:
        log("ERROR", f"Failed to export graph: {e}")
        return 1


def rollback_issue(issue_number: int, state_dir: pathlib.Path, repo_root: pathlib.Path) -> int:
    """Rollback implementation of a specific issue.

    Actions:
    - Remove worktree
    - Delete local and remote branch
    - Remove issue from state
    - Post rollback comment to issue

    Args:
        issue_number: Issue number to rollback
        state_dir: Directory containing state file
        repo_root: Repository root directory

    Returns:
        0 on success, 1 on failure
    """
    log("INFO", "=" * 80)
    log("INFO", f"ROLLBACK ISSUE #{issue_number}")
    log("INFO", "=" * 80)
    log("INFO", "")

    # Load state to check if issue was implemented
    from ._state import State

    state_file = state_dir / "implement_state.json"
    state = State.load(state_file) if state_file.exists() else State()

    # Check if issue is in state
    if (
        issue_number not in state.completed
        and issue_number not in state.failed
        and issue_number not in state.in_progress
    ):
        log("ERROR", f"Issue #{issue_number} not found in state (not yet processed or already rolled back)")
        return 1

    # Get issue status
    if issue_number in state.completed:
        status = "completed"
    elif issue_number in state.failed:
        status = "failed"
    else:
        status = "in_progress"

    log("INFO", f"Issue #{issue_number} status: {status}")
    log("INFO", "")

    # Confirm rollback
    log("INFO", "⚠️  WARNING: This will DELETE:")
    log("INFO", f"  - Local worktree (worktrees/{issue_number}-*)")
    log("INFO", f"  - Local branch ({issue_number}-*)")
    log("INFO", f"  - Remote branch (origin/{issue_number}-*)")
    log("INFO", "  - Issue from state file")
    log("INFO", "")

    response = input("Continue with rollback? (yes/no): ")
    if response.lower() != "yes":
        log("INFO", "Rollback cancelled")
        return 0

    log("INFO", "")
    log("INFO", "Starting rollback...")
    log("INFO", "")

    # 1. Remove worktree
    worktree_mgr = WorktreeManager(repo_root)
    if worktree_mgr.remove(issue_number):
        log("INFO", f"✓ Removed worktree for issue #{issue_number}")
    else:
        log("WARN", f"No worktree found for issue #{issue_number}")

    # 2. Delete remote branch
    # Find branch name matching issue
    branch_name = None
    result = subprocess.run(
        ["git", "branch", "-r"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if f"origin/{issue_number}-" in line:
                branch_name = line.strip().replace("origin/", "")
                break

    if branch_name:
        result = subprocess.run(
            ["git", "push", "origin", "--delete", branch_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log("INFO", f"✓ Deleted remote branch: origin/{branch_name}")
        else:
            log("WARN", f"Failed to delete remote branch: {result.stderr}")
    else:
        log("WARN", f"No remote branch found for issue #{issue_number}")

    # 3. Remove from state
    if issue_number in state.completed:
        del state.completed[issue_number]
    if issue_number in state.failed:
        del state.failed[issue_number]
    if issue_number in state.in_progress:
        del state.in_progress[issue_number]

    state.save(state_file)
    log("INFO", f"✓ Removed issue #{issue_number} from state")

    # 4. Post comment to issue
    comment = """## Implementation Rolled Back

This issue's implementation has been rolled back.

**Actions taken:**
- Deleted worktree
- Deleted branch
- Removed from state

The issue is now ready for re-implementation.
"""

    result = subprocess.run(
        ["gh", "issue", "comment", str(issue_number), "--body", comment],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log("INFO", f"✓ Posted rollback comment to issue #{issue_number}")
    else:
        log("WARN", f"Failed to post comment: {result.stderr}")

    log("INFO", "")
    log("INFO", "=" * 80)
    log("INFO", f"✓ ROLLBACK COMPLETE FOR ISSUE #{issue_number}")
    log("INFO", "=" * 80)

    return 0


# ---------------------------------------------------------------------
# Log Buffer for Curses UI
# ---------------------------------------------------------------------


class LogBuffer:
    """Thread-safe circular buffer for log messages with optional file output.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(self, max_lines: int = 1000, log_file: Optional[pathlib.Path] = None) -> None:
        self._lock = threading.Lock()
        self._messages: collections.deque[tuple[str, str, str]] = collections.deque(maxlen=max_lines)
        self._log_file = log_file
        self._file_handle: Optional[TextIO] = None
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(log_file, "a", encoding="utf-8")

    def __enter__(self) -> "LogBuffer":
        """Enter context manager - returns self."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager - ensures file handle is closed."""
        self.close()

    def append(self, level: str, msg: str) -> None:
        """Append a log message with timestamp."""
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._messages.append((ts, level, msg))
            # Also write to log file if configured
            if self._file_handle:
                self._file_handle.write(f"[{level:5}] {ts} {msg}\n")
                self._file_handle.flush()

    def get_messages(self, n: int) -> list[tuple[str, str, str]]:
        """Get the last n messages."""
        with self._lock:
            return list(self._messages)[-n:]

    def close(self) -> None:
        """Close the log file handle (safe to call multiple times)."""
        with self._lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except (OSError, IOError) as e:
                    # Log error but don't raise - cleanup should continue
                    print(f"Warning: Failed to close log file {self._log_file}: {e}", file=sys.stderr)
                finally:
                    self._file_handle = None


class ThreadLogManager:
    """Manages per-thread log buffers for parallel execution."""

    MAIN_THREAD_ID = -1  # Special ID for main/orchestrator thread

    def __init__(self, log_dir: pathlib.Path, log_prefix: str, max_workers: int) -> None:
        """Initialize thread log manager.

        Args:
            log_dir: Directory to store log files
            log_prefix: Prefix for log filenames (e.g., "implement-epic-2784-20250127-120000")
            max_workers: Number of worker threads (determines number of buffers to create)
        """
        self._lock = threading.Lock()
        self._buffers: dict[int, LogBuffer] = {}
        self._log_dir = log_dir
        self._log_prefix = log_prefix
        self._max_workers = max_workers

        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create buffer for main thread
        main_log_file = log_dir / f"{log_prefix}-main.log"
        self._buffers[self.MAIN_THREAD_ID] = LogBuffer(log_file=main_log_file)

        # Create buffers for worker threads
        for worker_id in range(max_workers):
            worker_log_file = log_dir / f"{log_prefix}-worker-{worker_id}.log"
            self._buffers[worker_id] = LogBuffer(log_file=worker_log_file)

    def get_buffer(self, worker_id: int) -> LogBuffer:
        """Get the log buffer for a specific worker ID.

        Args:
            worker_id: Worker ID (0 to max_workers-1) or MAIN_THREAD_ID for main thread

        Returns:
            LogBuffer for the specified worker
        """
        with self._lock:
            return self._buffers.get(worker_id, self._buffers[self.MAIN_THREAD_ID])

    def get_all_buffers(self) -> dict[int, LogBuffer]:
        """Get all log buffers indexed by worker ID."""
        with self._lock:
            return dict(self._buffers)

    def get_all_messages_merged(self, n: int) -> list[tuple[str, str, str]]:
        """Get the last n messages from all buffers, merged chronologically.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of (timestamp, level, message) tuples, sorted chronologically
        """
        with self._lock:
            all_messages = []
            for buffer in self._buffers.values():
                all_messages.extend(buffer.get_messages(10000))  # Get many messages

            # Sort by timestamp (first element of tuple)
            all_messages.sort(key=lambda x: x[0])

            # Return last n messages
            return all_messages[-n:] if len(all_messages) > n else all_messages

    def close_all(self) -> None:
        """Close all log file handles."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.close()

    def merge_logs_on_success(self, output_file: pathlib.Path) -> None:
        """Merge all per-thread logs into a single chronological log file.

        This should only be called on successful completion with no errors.
        After merging, individual thread log files are deleted.

        Args:
            output_file: Path to the merged output log file
        """
        with self._lock:
            # Collect all log lines with timestamps
            all_lines: list[tuple[str, str]] = []  # (timestamp_sortable, line)

            for worker_id, buffer in self._buffers.items():
                log_file = buffer._log_file
                if log_file and log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            # Extract timestamp from line format: [LEVEL] HH:MM:SS message
                            if line.startswith("[") and "]" in line:
                                # Extract HH:MM:SS timestamp
                                parts = line.split("]", 1)
                                if len(parts) == 2:
                                    timestamp_part = parts[1].strip().split(" ", 1)
                                    if timestamp_part:
                                        timestamp = timestamp_part[0]
                                        all_lines.append((timestamp, line))

            # Sort by timestamp
            all_lines.sort(key=lambda x: x[0])

            # Write merged log
            with open(output_file, "w", encoding="utf-8") as f:
                for _, line in all_lines:
                    f.write(line)

            # Delete individual log files
            for worker_id, buffer in self._buffers.items():
                log_file = buffer._log_file
                if log_file and log_file.exists():
                    log_file.unlink()


# Global log buffer for curses mode (deprecated - use thread log manager)
_log_buffer: Optional[LogBuffer] = None

# Global thread log manager for per-thread logging
_thread_log_manager: Optional[ThreadLogManager] = None

# Thread-local storage to track which worker slot the current thread is using
_thread_local = threading.local()


def set_log_buffer(buffer: LogBuffer | None) -> None:
    """Set the global log buffer for curses mode (deprecated)."""
    global _log_buffer
    _log_buffer = buffer


def set_thread_log_manager(manager: ThreadLogManager | None) -> None:
    """Set the global thread log manager."""
    global _thread_log_manager
    _thread_log_manager = manager


def set_worker_slot(slot: int) -> None:
    """Set the worker slot ID for the current thread.

    Args:
        slot: Worker slot ID (0 to max_workers-1) or ThreadLogManager.MAIN_THREAD_ID for main thread
    """
    _thread_local.worker_slot = slot


def get_worker_slot() -> int:
    """Get the worker slot ID for the current thread.

    Returns:
        Worker slot ID, or MAIN_THREAD_ID if not set
    """
    return getattr(_thread_local, "worker_slot", ThreadLogManager.MAIN_THREAD_ID)


# ---------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------


# Global verbose/debug flags (set from Options)
_verbose_mode = False
_debug_mode = False

# Global lock for stdout/stderr to prevent output races
_output_lock = threading.Lock()

# Global shutdown flag for graceful termination
_shutdown_requested = False
_shutdown_lock = threading.Lock()


def set_shutdown_requested() -> None:
    """Set the global shutdown flag to request graceful termination."""
    global _shutdown_requested
    with _shutdown_lock:
        _shutdown_requested = True


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    global _shutdown_requested
    with _shutdown_lock:
        return _shutdown_requested


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with more aggressive shutdown behavior.

    Uses wait=False and cancels futures to allow faster shutdown,
    even if worker threads are stuck in subprocess calls.
    """

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Shutdown with default cancel_futures=True for faster exit."""
        # Always use cancel_futures=True for aggressive shutdown
        super().shutdown(wait=False, cancel_futures=True)


def set_verbose(verbose: bool, debug: bool = False) -> None:
    """Set the global verbose and debug modes."""
    global _verbose_mode, _debug_mode
    _verbose_mode = verbose
    _debug_mode = debug


def log(level: str, msg: str) -> None:
    """Log a message with timestamp and level prefix.

    Routes messages to the appropriate per-thread log buffer based on the current
    thread's worker slot ID.
    """
    if level == "DEBUG" and not _debug_mode:
        return

    # Route to thread-specific buffer if thread log manager is available
    if _thread_log_manager is not None:
        worker_slot = get_worker_slot()
        buffer = _thread_log_manager.get_buffer(worker_slot)
        buffer.append(level, msg)
        return

    # Fallback: Route to single log buffer if in curses mode (deprecated path)
    if _log_buffer is not None:
        _log_buffer.append(level, msg)
        return

    # Fallback for non-curses mode: print to stdout/stderr
    ts = time.strftime("%H:%M:%S")
    out = sys.stderr if level in {"WARN", "ERROR"} else sys.stdout
    with _output_lock:
        # Clear current line and move to new line before printing
        # This ensures log messages don't overlap with status line
        if sys.stdout.isatty():
            sys.stdout.write("\r\033[K")  # Clear the status line
            sys.stdout.flush()
        print(f"[{level}] {ts} {msg}", file=out, flush=True)


# ---------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------


def run(
    cmd: list[str],
    *,
    timeout: int | None = None,
    cwd: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture output."""
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
    )


def clean_stale_git_locks(repo_root: pathlib.Path, ref_pattern: str = "*") -> int:
    """Clean up stale git lock files.

    Args:
        repo_root: Path to the git repository root
        ref_pattern: Pattern to match ref lock files (e.g., "fix-test-issues-*")

    Returns:
        Number of stale lock files removed
    """
    git_dir = repo_root / ".git"
    refs_dir = git_dir / "refs" / "remotes" / "origin"

    if not refs_dir.exists():
        return 0

    stale_threshold = 300  # 5 minutes in seconds
    current_time = time.time()
    removed_count = 0

    # Find all .lock files matching pattern
    for lock_file in refs_dir.glob(f"{ref_pattern}.lock"):
        try:
            # Check file age
            file_age = current_time - lock_file.stat().st_mtime
            if file_age > stale_threshold:
                log("DEBUG", f"  Removing stale lock file: {lock_file.name} (age: {file_age:.0f}s)")
                lock_file.unlink()
                removed_count += 1
        except (OSError, FileNotFoundError):
            # File may have been removed by another process
            pass

    if removed_count > 0:
        log("INFO", f"  Cleaned {removed_count} stale git lock file(s)")

    return removed_count


def safe_git_fetch(
    repo_root: pathlib.Path,
    branch: str,
    max_retries: int = 3,
) -> tuple[bool, str]:
    """Safely fetch a git branch with retry logic and lock cleanup.

    Implements exponential backoff: 1s, 2s, 4s delays between retries.
    Cleans stale lock files before each retry.

    Args:
        repo_root: Path to the git repository root
        branch: Branch name to fetch
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (success: bool, error_msg: str)
        - success: True if fetch succeeded, False otherwise
        - error_msg: Error message if fetch failed, empty string on success
    """
    delays = [1, 2, 4]  # Exponential backoff delays in seconds

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        # Clean stale locks before attempting fetch (except on first attempt)
        if attempt > 0:
            clean_stale_git_locks(repo_root, branch)
            delay = delays[attempt - 1] if attempt - 1 < len(delays) else delays[-1]
            log("INFO", f"  Retrying git fetch after {delay}s delay (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

        # Attempt git fetch
        try:
            cp = run(
                ["git", "fetch", "origin", f"{branch}:refs/remotes/origin/{branch}"],
                timeout=60,
                cwd=repo_root,
            )

            if cp.returncode == 0:
                if attempt > 0:
                    log("INFO", f"  Git fetch succeeded on attempt {attempt + 1}")
                return (True, "")

            # Fetch failed, check if it's a lock error
            error_output = cp.stderr + cp.stdout
            is_lock_error = "cannot lock ref" in error_output or ".lock" in error_output

            if is_lock_error:
                log("WARN", f"  Git fetch failed (lock error, attempt {attempt + 1}/{max_retries + 1})")
                if attempt >= max_retries:
                    # Final attempt failed
                    return (False, f"git fetch failed after {max_retries + 1} attempts: {error_output[:200]}")
                # Continue to next retry
            else:
                # Non-lock error, fail immediately
                return (False, f"git fetch failed: {error_output[:200]}")

        except subprocess.TimeoutExpired:
            return (False, "git fetch timed out after 60 seconds")
        except Exception as e:
            return (False, f"git fetch exception: {str(e)}")

    # Should not reach here, but just in case
    return (False, "git fetch failed: unknown error")


class CachedIssueState(NamedTuple):
    """Cached issue state with timestamp for TTL support."""

    is_closed: bool
    cached_at: float  # epoch seconds


# Cache for external issue states (avoid repeated API calls)
# Now includes timestamps for TTL support (5-minute expiration)
_external_issue_cache: dict[int, CachedIssueState] = {}
_external_issue_cache_lock = threading.Lock()  # Thread-safe access to cache
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Cache for repo owner/name
_repo_info_cache: tuple[str, str] | None = None


def get_repo_info() -> tuple[str, str]:
    """Get the repo owner and name from git remote.

    Returns (owner, name) tuple. Caches result for subsequent calls.
    """
    global _repo_info_cache
    if _repo_info_cache is not None:
        return _repo_info_cache

    # Try gh repo view first (most reliable)
    cp = run(["gh", "repo", "view", "--json", "owner,name"])
    if cp.returncode == 0:
        try:
            data = json.loads(cp.stdout)
            owner = data.get("owner", {}).get("login", "")
            name = data.get("name", "")
            if owner and name:
                _repo_info_cache = (owner, name)
                return _repo_info_cache
        except json.JSONDecodeError:
            pass

    # Fallback: parse git remote URL
    cp = run(["git", "remote", "get-url", "origin"])
    if cp.returncode == 0:
        url = cp.stdout.strip()
        # Handle both SSH and HTTPS URLs
        # SSH: git@github.com:owner/repo.git
        # HTTPS: https://github.com/owner/repo.git
        match = re.search(r"[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
        if match:
            _repo_info_cache = (match.group(1), match.group(2))
            return _repo_info_cache

    # Last resort fallback
    log("WARN", "Could not detect repo owner/name, using defaults")
    _repo_info_cache = ("mvillmow", "ProjectOdyssey")
    return _repo_info_cache


def prefetch_issue_states(issue_nums: list[int]) -> None:
    """Batch fetch issue states using GraphQL to populate cache.

    This reduces API calls when checking multiple external dependencies.
    Cache entries are timestamped for TTL support.
    """
    # Filter to issues not already cached or expired (with thread-safe access)
    now = time.time()
    to_fetch = []
    with _external_issue_cache_lock:
        for n in issue_nums:
            if n not in _external_issue_cache:
                to_fetch.append(n)
            else:
                cached = _external_issue_cache[n]
                if now - cached.cached_at >= _CACHE_TTL_SECONDS:
                    to_fetch.append(n)

    if not to_fetch:
        log("DEBUG", "All issues cached and fresh, skipping prefetch")
        return

    log("DEBUG", f"Prefetching states for {len(to_fetch)} external issues...")

    # Get repo info from git
    repo_owner, repo_name = get_repo_info()

    # Batch in groups of 50
    batch_size = 50
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]

        fragments = []
        for j, num in enumerate(batch):
            fragments.append(f"issue{j}: issue(number: {num}) {{ number state }}")

        query = f"""
        query {{
          repository(owner: "{repo_owner}", name: "{repo_name}") {{
            {chr(10).join(fragments)}
          }}
        }}
        """

        try:
            cp = run(["gh", "api", "graphql", "-f", f"query={query}"], timeout=30)
            if cp.returncode == 0:
                try:
                    data = json.loads(cp.stdout)
                    repo_data = data.get("data", {}).get("repository", {})
                    # Update cache with thread-safe access
                    with _external_issue_cache_lock:
                        for j, num in enumerate(batch):
                            issue_data = repo_data.get(f"issue{j}")
                            if issue_data:
                                is_closed = issue_data.get("state", "OPEN").upper() == "CLOSED"
                                _external_issue_cache[num] = CachedIssueState(is_closed=is_closed, cached_at=now)
                            else:
                                _external_issue_cache[num] = CachedIssueState(is_closed=False, cached_at=now)
                except (json.JSONDecodeError, KeyError):
                    # Mark all as not closed on parse error (with thread-safe access)
                    with _external_issue_cache_lock:
                        for num in batch:
                            if num not in _external_issue_cache:
                                _external_issue_cache[num] = CachedIssueState(is_closed=False, cached_at=now)
            else:
                # Mark all as not closed on API error (with thread-safe access)
                with _external_issue_cache_lock:
                    for num in batch:
                        if num not in _external_issue_cache:
                            _external_issue_cache[num] = CachedIssueState(is_closed=False, cached_at=now)
        except subprocess.TimeoutExpired:
            log("WARN", f"GitHub API timeout for batch {i // batch_size + 1}, marking as not closed")
            # Mark all as not closed on timeout (with thread-safe access)
            with _external_issue_cache_lock:
                for num in batch:
                    if num not in _external_issue_cache:
                        _external_issue_cache[num] = CachedIssueState(is_closed=False, cached_at=now)


def is_issue_closed(issue_num: int, force_refresh: bool = False) -> bool:
    """Check if an external issue is closed via GitHub API.

    Results are cached for 5 minutes to avoid repeated API calls.
    Use prefetch_issue_states() first for batch operations.

    Args:
        issue_num: Issue number to check
        force_refresh: Bypass cache and fetch fresh data

    Returns:
        True if issue is closed
    """
    now = time.time()

    # Check cache if not forcing refresh (with thread-safe access)
    if not force_refresh:
        with _external_issue_cache_lock:
            if issue_num in _external_issue_cache:
                cached = _external_issue_cache[issue_num]
                age = now - cached.cached_at

                if age < _CACHE_TTL_SECONDS:
                    log("DEBUG", f"Cache hit for issue #{issue_num} (age: {age:.0f}s)")
                    return cached.is_closed
                else:
                    log("DEBUG", f"Cache expired for issue #{issue_num} (age: {age:.0f}s)")

    # Fetch from API (outside lock - slow operation)
    log("DEBUG", f"Fetching state for issue #{issue_num}")
    cp = run(["gh", "issue", "view", str(issue_num), "--json", "state"])

    if cp.returncode == 0:
        try:
            data = json.loads(cp.stdout)
            is_closed = data.get("state", "OPEN").upper() == "CLOSED"

            # Update cache with timestamp (with thread-safe access)
            with _external_issue_cache_lock:
                _external_issue_cache[issue_num] = CachedIssueState(is_closed=is_closed, cached_at=now)
            return is_closed
        except json.JSONDecodeError as e:
            log("ERROR", f"Failed to parse issue #{issue_num} state: {e}")
            raise RuntimeError(f"Invalid JSON from GitHub API for issue #{issue_num}: {e}")

    # Fetch failed - distinguish error types
    stderr_lower = cp.stderr.lower()

    # Check if issue doesn't exist
    if "not found" in stderr_lower or "could not resolve" in stderr_lower:
        log("ERROR", f"Issue #{issue_num} does not exist")
        raise ValueError(f"External dependency issue #{issue_num} does not exist")

    # Check for network/temporary errors
    if any(msg in stderr_lower for msg in ["connection", "network", "timeout", "eof", "dns", "rate limit"]):
        log("ERROR", f"Network error fetching issue #{issue_num}: {cp.stderr[:200]}")
        raise RuntimeError(f"Network error checking external dependency #{issue_num}: {cp.stderr[:200]}")

    # Unknown error - raise
    log("ERROR", f"Unknown error fetching issue #{issue_num}: {cp.stderr[:200]}")
    raise RuntimeError(f"Failed to check external dependency #{issue_num}: {cp.stderr[:200]}")


# ---------------------------------------------------------------------
# Rate-limit handling (from plan_issues.py)
# ---------------------------------------------------------------------


def parse_reset_epoch(time_str: str, tz: str) -> int:
    """Parse a rate limit reset time string and return epoch seconds.

    Supports multiple formats:
    - ISO 8601 timestamps (GitHub API format)
    - Unix epoch seconds (integer)
    - Human-readable time (e.g., "3:45pm")

    Args:
        time_str: Time string to parse
        tz: Timezone identifier (e.g., "America/Los_Angeles")

    Returns:
        Epoch seconds for reset time
    """
    # Strategy 1: Try ISO 8601 timestamp (GitHub API format)
    try:
        parsed = dt.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(parsed.timestamp())
    except (ValueError, AttributeError):
        pass

    # Strategy 2: Try unix epoch (integer seconds)
    try:
        return int(time_str)
    except ValueError:
        pass

    # Strategy 3: Human-readable time (e.g., "3:45pm")
    # Validate timezone
    if tz not in ALLOWED_TIMEZONES:
        log("WARN", f"Unknown timezone '{tz}', falling back to America/Los_Angeles")
        tz = "America/Los_Angeles"

    now_utc = dt.datetime.now(dt.timezone.utc)
    today = now_utc.astimezone(ZoneInfo(tz)).date()

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", time_str, re.IGNORECASE)
    if not m:
        log("WARN", f"Could not parse rate limit reset time '{time_str}', using 12-hour fallback")
        return int(time.time()) + 43200  # 12 hours - conservative fallback

    hour, minute, ampm = m.groups()
    hour = int(hour)
    minute = int(minute or 0)

    # Validate ranges BEFORE conversion
    if hour < 1 or hour > 12:
        log("WARN", f"Invalid hour {hour} in time '{time_str}', using 12-hour fallback")
        return int(time.time()) + 43200
    if minute < 0 or minute > 59:
        log("WARN", f"Invalid minute {minute} in time '{time_str}', using 12-hour fallback")
        return int(time.time()) + 43200

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < NOON_HOUR:
            hour += NOON_HOUR
        if ampm == "am" and hour == NOON_HOUR:
            hour = MIDNIGHT_HOUR

    try:
        local = dt.datetime.combine(
            today,
            dt.time(hour, minute),
            tzinfo=ZoneInfo(tz),
        )
    except (ValueError, KeyError) as e:
        log("ERROR", f"Failed to construct datetime for '{time_str}' in {tz}: {e}")
        return int(time.time()) + 43200

    if local < now_utc.astimezone(ZoneInfo(tz)):
        local += dt.timedelta(days=1)

    return int(local.timestamp())


def detect_rate_limit(text: str) -> int | None:
    """Detect rate limit message in text and return reset epoch if found."""
    m = RATE_LIMIT_RE.search(text)
    if not m:
        return None
    return parse_reset_epoch(m.group("time"), m.group("tz"))


def detect_claude_usage_limit(text: str) -> bool:
    """Detect Claude API usage limit errors in output.

    Returns:
        True if usage limit detected, False otherwise
    """
    usage_limit_indicators = [
        "usage limit",
        "rate limit exceeded",
        "429",  # HTTP 429 Too Many Requests
        "quota exceeded",
        "too many requests",
        "overloaded_error",  # Claude-specific error type
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in usage_limit_indicators)


def wait_until(epoch: int) -> None:
    """Wait until the given epoch time, showing a countdown."""
    interrupted = False

    def handler(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    old_handler = signal.signal(signal.SIGINT, handler)
    try:
        while True:
            if interrupted:
                print("\n[INFO] Wait interrupted by user")
                raise KeyboardInterrupt
            remaining = epoch - int(time.time())
            if remaining <= 0:
                print()
                return
            h, r = divmod(remaining, 3600)
            m, s = divmod(r, 60)
            print(
                f"\r[INFO] Rate limit resets in {h:02d}:{m:02d}:{s:02d}",
                end="",
                flush=True,
            )
            time.sleep(1)
    finally:
        signal.signal(signal.SIGINT, old_handler)


def kill_process_safely(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Safely kill a subprocess, escalating to SIGKILL if needed.

    Args:
        proc: The process to kill
        timeout: How long to wait for graceful termination

    This prevents orphaned processes by ensuring the process is terminated.
    """
    try:
        proc.kill()  # Send SIGTERM (or SIGKILL on Windows)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Process didn't die - force kill with SIGKILL
        log("WARN", f"Process {proc.pid} didn't terminate, forcing SIGKILL")
        try:
            proc.send_signal(signal.SIGKILL)
            proc.wait()  # This should succeed now
        except (OSError, subprocess.SubprocessError) as e:
            log("ERROR", f"Failed to force-kill process {proc.pid}: {e}")


# ---------------------------------------------------------------------
# Status Tracker (from plan_issues.py)
# ---------------------------------------------------------------------


class StatusTracker:
    """Thread-safe status tracker for parallel processing.

    This class is a pure state container. Display is handled by CursesUI.
    """

    MAIN_THREAD = -1

    def __init__(self, max_workers: int) -> None:
        """Initialize the status tracker."""
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._slots: dict[int, tuple[int, str, str, float]] = {}
        self._slot_to_item: dict[int, int] = {}  # Track which item owns which slot
        self._slot_available = threading.Condition(self._lock)  # For waiting on slots
        self._update_event = threading.Event()
        self._completed_count = 0
        self._total_count = 0
        # Track cleanup phase (after slot release but before future completes)
        self._cleanup_items: dict[int, tuple[int, float]] = {}  # item_id -> (worker_id, start_time)

    def set_total(self, total: int) -> None:
        """Set the total number of items to process."""
        with self._lock:
            self._total_count = total

    def increment_completed(self) -> None:
        """Increment the completed count."""
        with self._lock:
            self._completed_count += 1
            self._update_event.set()

    def update_main(self, stage: str, info: str = "") -> None:
        """Update the main thread status."""
        with self._lock:
            self._slots[self.MAIN_THREAD] = (0, stage, info, time.time())
            self._update_event.set()

    def acquire_slot(self, item_id: int, timeout: float = 60.0) -> int:
        """Acquire a display slot for an item. Returns slot number.

        Args:
            item_id: ID of item needing a slot
            timeout: Max seconds to wait for slot (default 60s)

        Returns:
            Slot number (0 to max_workers-1)

        Raises:
            RuntimeError: If no slot becomes available within timeout
        """
        deadline = time.time() + timeout

        with self._slot_available:  # Uses condition variable
            while True:
                # Try to find an available slot
                for slot in range(self._max_workers):
                    if slot not in self._slot_to_item:
                        # Slot is available - claim it
                        self._slots[slot] = (item_id, "Starting", "", time.time())
                        self._slot_to_item[slot] = item_id
                        self._update_event.set()
                        wait_time = time.time() - (deadline - timeout)
                        log("DEBUG", f"Item #{item_id} acquired slot {slot} (waited {wait_time:.1f}s)")
                        return slot

                # No slots available - wait for one to be released
                remaining = deadline - time.time()
                if remaining <= 0:
                    # Build detailed error message showing occupied slots
                    occupied = []
                    for s, i in self._slot_to_item.items():
                        stage, info = "", ""
                        if s in self._slots:
                            _, stage, info, _ = self._slots[s]
                        occupied.append(f"Slot {s}: #{i} ({stage})")

                    raise RuntimeError(
                        f"Timeout waiting for slot (item #{item_id}). "
                        f"All {self._max_workers} slots occupied:\n  " + "\n  ".join(occupied) + "\n"
                        "Consider reducing --parallel count."
                    )

                occupied_items = list(self._slot_to_item.values())
                log(
                    "DEBUG",
                    f"Item #{item_id} waiting for slot (timeout in {remaining:.0f}s, occupied: {occupied_items})",
                )
                # Wait proportionally to remaining time (min 1s, max remaining/4)
                wait_time = min(remaining, max(1.0, remaining / 4))
                self._slot_available.wait(timeout=wait_time)

    def update(self, slot: int, item_id: int, stage: str, info: str = "") -> None:
        """Update the status for a slot.

        Validates that the slot is owned by the item before updating.
        """
        with self._lock:
            # Verify slot ownership
            if slot not in self._slot_to_item:
                log("WARN", f"Update to unoccupied slot {slot} by item #{item_id}")
                return
            if self._slot_to_item[slot] != item_id:
                log(
                    "ERROR",
                    f"Slot ownership mismatch! Slot {slot} owned by item #{self._slot_to_item[slot]}, not #{item_id}",
                )
                return

            self._slots[slot] = (item_id, stage, info, time.time())
            self._update_event.set()

    def set_cleanup(self, item_id: int, worker_id: int) -> None:
        """Mark an item as in cleanup phase.

        Call this after releasing the slot but before cleanup completes.
        Allows UI to show cleanup status instead of "Idle".

        Args:
            item_id: ID of item being cleaned up
            worker_id: Worker slot number (for UI display)
        """
        with self._lock:
            self._cleanup_items[item_id] = (worker_id, time.time())
            self._update_event.set()
            log("DEBUG", f"Item #{item_id} entered cleanup phase (worker {worker_id})")

    def clear_cleanup(self, item_id: int) -> None:
        """Clear cleanup status when item is fully complete.

        Call this in main loop after future.result() completes.

        Args:
            item_id: ID of item to clear
        """
        with self._lock:
            if item_id in self._cleanup_items:
                worker_id, _ = self._cleanup_items[item_id]
                del self._cleanup_items[item_id]
                self._update_event.set()
                log("DEBUG", f"Item #{item_id} cleanup complete (was worker {worker_id})")

    def release_slot(self, slot: int) -> None:
        """Release a slot when done."""
        with self._slot_available:  # Uses condition variable
            if slot in self._slots:
                item_id = self._slot_to_item.get(slot, -1)
                log("DEBUG", f"Item #{item_id} released slot {slot}")
                del self._slots[slot]
                if slot in self._slot_to_item:
                    del self._slot_to_item[slot]
                self._update_event.set()
                self._slot_available.notify_all()  # Wake ALL waiting threads
            else:
                log("WARN", f"Attempted to release unoccupied slot {slot}")

    def release_by_item(self, item_id: int) -> bool:
        """Release the slot owned by a specific item.

        Args:
            item_id: ID of the item to release

        Returns:
            True if slot was found and released, False otherwise
        """
        with self._slot_available:
            # Find slot for this item
            for slot, owner in self._slot_to_item.items():
                if owner == item_id:
                    # Found it - release
                    log("DEBUG", f"Item #{item_id} releasing slot {slot}")
                    if slot in self._slots:
                        del self._slots[slot]
                    del self._slot_to_item[slot]
                    self._update_event.set()
                    self._slot_available.notify_all()  # Wake ALL waiting threads
                    return True

            log("WARN", f"No slot found for item #{item_id}")
            return False

    def get_status_data(self) -> dict:
        """Return current state snapshot for external rendering."""
        with self._lock:
            # Detect slot leaks (exclude MAIN_THREAD which doesn't use slot_to_item tracking)
            leaked_slots = set(self._slots.keys()) - set(self._slot_to_item.keys()) - {self.MAIN_THREAD}
            if leaked_slots:
                leaked_details = {slot: self._slots[slot] for slot in leaked_slots}
                log("ERROR", f"Slot leak detected! Slots without owners: {leaked_slots}")
                log("ERROR", f"Leaked slot details: {leaked_details}")

            # Detect orphaned ownership (slot_to_item without corresponding slot)
            orphaned_items = set(self._slot_to_item.keys()) - set(self._slots.keys())
            if orphaned_items:
                orphaned_details = {slot: self._slot_to_item[slot] for slot in orphaned_items}
                log("ERROR", f"Orphaned ownership detected! Tracked items without slots: {orphaned_items}")
                log("ERROR", f"Orphaned details: {orphaned_details}")

            return {
                "completed": self._completed_count,
                "total": self._total_count,
                "slots": dict(self._slots),
                "cleanup_items": dict(self._cleanup_items),  # NEW: Track cleanup phase
                "max_workers": self._max_workers,
            }


# ---------------------------------------------------------------------
# Curses UI
# ---------------------------------------------------------------------


class CursesUI:
    """Curses-based UI with scrolling logs and fixed status panel."""

    # Color pairs
    COLOR_HEADER = 1
    COLOR_WORKER_BASE = 2  # Workers use 2-9
    COLOR_ERROR = 10
    COLOR_WARN = 11
    COLOR_INFO = 12

    # Worker colors cycle: green, yellow, blue, magenta, cyan, white, red
    WORKER_COLORS = [
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_BLUE,
        curses.COLOR_MAGENTA,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_RED,
        curses.COLOR_GREEN,
    ]

    def __init__(
        self,
        status_tracker: StatusTracker,
        log_manager: ThreadLogManager | LogBuffer,
        max_workers: int,
    ) -> None:
        """Initialize the curses UI.

        Args:
            status_tracker: Status tracker for worker progress
            log_manager: ThreadLogManager for per-thread logs or LogBuffer for single log (deprecated)
            max_workers: Number of worker threads
        """
        self._status_tracker = status_tracker
        self._max_workers = max_workers
        self._stop_event = threading.Event()
        self._screen: curses.window | None = None

        # Support both ThreadLogManager (new) and LogBuffer (deprecated)
        self._thread_log_manager: Optional[ThreadLogManager] = None
        if isinstance(log_manager, ThreadLogManager):
            self._thread_log_manager = log_manager
            self._log_buffer = log_manager.get_buffer(ThreadLogManager.MAIN_THREAD_ID)
        else:
            self._log_buffer = log_manager

        # View cycling state for Tab key
        self._current_view_index = 0
        self._build_view_list()

    def run(self, main_func: Callable[[], int]) -> int:
        """Run UI with curses.wrapper, executing main_func in background thread."""
        return curses.wrapper(self._curses_main, main_func)

    def _curses_main(self, stdscr: curses.window, main_func: Callable[[], int]) -> int:
        """Main curses loop."""
        self._screen = stdscr
        self._setup_colors()
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)

        # Run business logic in background thread
        result_holder = [0]

        def run_main() -> None:
            try:
                log("DEBUG", "Background thread: Starting main_func")
                result_holder[0] = main_func()
                log("DEBUG", "Background thread: main_func completed")
            except Exception as e:
                log("ERROR", f"Background thread: Exception in main_func: {e}")
                result_holder[0] = 1
            finally:
                log("DEBUG", "Background thread: Setting stop event")
                self._stop_event.set()
                set_shutdown_requested()  # Signal shutdown when background thread completes
                log("DEBUG", "Background thread: Exiting run_main")

        main_thread = threading.Thread(target=run_main, name="MainWorker")
        main_thread.start()

        # Render loop on main thread (curses requirement)
        log("DEBUG", "Curses UI: Starting render loop")
        self._render_loop()
        log("DEBUG", "Curses UI: Render loop exited")

        # Wait for background thread with timeout
        log("DEBUG", "Curses UI: Waiting for background thread to complete")
        main_thread.join(timeout=5.0)
        if main_thread.is_alive():
            log("WARN", "Curses UI: Background thread still alive after 5s timeout")
            # Give it a bit more time
            main_thread.join(timeout=5.0)
            if main_thread.is_alive():
                log("ERROR", "Curses UI: Background thread failed to exit after 10s, forcing exit")
        else:
            log("DEBUG", "Curses UI: Background thread completed successfully")

        return result_holder[0]

    def _setup_colors(self) -> None:
        """Initialize color pairs for the UI."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_HEADER, curses.COLOR_CYAN, -1)
        for i, color in enumerate(self.WORKER_COLORS):
            curses.init_pair(self.COLOR_WORKER_BASE + i, color, -1)
        curses.init_pair(self.COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_WARN, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_INFO, curses.COLOR_WHITE, -1)

    def _render_loop(self) -> None:
        """Main render loop running on the main thread."""
        while not self._stop_event.is_set() and not is_shutdown_requested():
            try:
                # Handle keyboard input (non-blocking)
                self._handle_keyboard_input()

                # Check for resize
                if self._screen is not None:
                    new_size = self._screen.getmaxyx()
                    if curses.is_term_resized(*new_size):
                        curses.resizeterm(*new_size)

                # Render UI
                self._render()
            except curses.error:
                pass
            self._status_tracker._update_event.wait(0.1)
            self._status_tracker._update_event.clear()

    def _render(self) -> None:
        """Render the full UI."""
        if self._screen is None:
            return

        height, width = self._screen.getmaxyx()
        status_height = self._max_workers + 3  # Header + Main + workers
        log_height = max(1, height - status_height - 1)  # -1 for separator

        self._screen.erase()

        # Render logs (top section) using current view
        messages = self._get_messages_for_current_view(log_height)
        for i, (ts, level, msg) in enumerate(messages):
            color = self._get_log_color(level)
            line = f"[{level:5}] {ts} {msg}"[: width - 1]
            try:
                self._screen.addstr(i, 0, line, color)
            except curses.error:
                pass

        # Separator line
        try:
            self._screen.hline(log_height, 0, curses.ACS_HLINE, width)
        except curses.error:
            pass

        # Status panel (bottom section)
        data = self._status_tracker.get_status_data()
        view_name = self._get_current_view_name()
        header = f" Progress: [{data['completed']}/{data['total']}] | View: {view_name} | Tab: cycle views"
        try:
            self._screen.addstr(
                log_height + 1,
                0,
                header[: width - 1],  # Truncate to screen width
                curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD,
            )
        except curses.error:
            pass

        # Main thread status
        if StatusTracker.MAIN_THREAD in data["slots"]:
            _, stage, info, start = data["slots"][StatusTracker.MAIN_THREAD]
            elapsed = self._format_elapsed(start)
            info_str = f" - {info}" if info else ""
            line = f"  Main:     [{stage:12}] ({elapsed}){info_str}"[: width - 1]
            try:
                self._screen.addstr(log_height + 2, 0, line, curses.color_pair(self.COLOR_HEADER))
            except curses.error:
                pass
        else:
            try:
                self._screen.addstr(
                    log_height + 2,
                    0,
                    "  Main:     [Idle        ]",
                    curses.color_pair(self.COLOR_HEADER),
                )
            except curses.error:
                pass

        # Worker status lines
        for slot in range(data["max_workers"]):
            row = log_height + 3 + slot
            color = curses.color_pair(self.COLOR_WORKER_BASE + (slot % 8))

            if slot in data["slots"]:
                # Worker has active slot
                item_id, stage, info, start = data["slots"][slot]
                elapsed = self._format_elapsed(start)
                info_str = f" - {info}" if info else ""
                line = f"  Worker {slot}: [{stage:12}] #{item_id} ({elapsed}){info_str}"[: width - 1]
            else:
                # Check if any item is in cleanup and was using this worker
                cleanup_line = None
                for item_id, (worker_id, start) in data.get("cleanup_items", {}).items():
                    if worker_id == slot:
                        elapsed = self._format_elapsed(start)
                        cleanup_line = f"  Worker {slot}: [Cleanup     ] #{item_id} ({elapsed})"[: width - 1]
                        break

                if cleanup_line:
                    line = cleanup_line
                else:
                    line = f"  Worker {slot}: [Idle        ]"

            try:
                self._screen.addstr(row, 0, line, color)
            except curses.error:
                pass

        self._screen.refresh()

    def _get_log_color(self, level: str) -> int:
        """Get the color attribute for a log level."""
        if level == "ERROR":
            return curses.color_pair(self.COLOR_ERROR) | curses.A_BOLD
        elif level == "WARN":
            return curses.color_pair(self.COLOR_WARN)
        return curses.color_pair(self.COLOR_INFO)

    def _format_elapsed(self, start_time: float) -> str:
        """Format elapsed time as MM:SS."""
        elapsed = max(0, int(time.time() - start_time))
        m, s = divmod(elapsed, 60)
        return f"{m:02d}:{s:02d}"

    def _build_view_list(self) -> None:
        """Build the list of available views based on configuration."""
        self._view_list = ["ALL", "MAIN"]

        # Add worker views if thread log manager is available
        if self._thread_log_manager:
            for worker_id in range(self._max_workers):
                self._view_list.append(f"WORKER-{worker_id}")

        # Add filter views
        self._view_list.extend(["ERRORS", "DEBUG"])

    def _get_current_view_name(self) -> str:
        """Get the name of the current view."""
        if self._current_view_index < len(self._view_list):
            return self._view_list[self._current_view_index]
        return "ALL"

    def _cycle_view(self) -> None:
        """Cycle to the next view."""
        self._current_view_index = (self._current_view_index + 1) % len(self._view_list)
        self._status_tracker._update_event.set()  # Trigger immediate re-render

    def _get_messages_for_current_view(self, n: int) -> list[tuple[str, str, str]]:
        """Get log messages for the currently selected view.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of (timestamp, level, message) tuples
        """
        view_name = self._get_current_view_name()

        if view_name == "ALL":
            # Merged view from all threads
            if self._thread_log_manager:
                return self._thread_log_manager.get_all_messages_merged(n)
            else:
                return self._log_buffer.get_messages(n)

        elif view_name == "MAIN":
            # Main thread only
            if self._thread_log_manager:
                buffer = self._thread_log_manager.get_buffer(ThreadLogManager.MAIN_THREAD_ID)
                return buffer.get_messages(n)
            else:
                return self._log_buffer.get_messages(n)

        elif view_name.startswith("WORKER-"):
            # Specific worker thread
            if self._thread_log_manager:
                worker_id = int(view_name.split("-")[1])
                buffer = self._thread_log_manager.get_buffer(worker_id)
                return buffer.get_messages(n)
            else:
                return self._log_buffer.get_messages(n)

        elif view_name == "ERRORS":
            # Filter to ERROR level only
            if self._thread_log_manager:
                all_messages = self._thread_log_manager.get_all_messages_merged(10000)
            else:
                all_messages = self._log_buffer.get_messages(10000)
            filtered = [m for m in all_messages if m[1] == "ERROR"]
            return filtered[-n:] if len(filtered) > n else filtered

        elif view_name == "DEBUG":
            # Filter to DEBUG level only
            if self._thread_log_manager:
                all_messages = self._thread_log_manager.get_all_messages_merged(10000)
            else:
                all_messages = self._log_buffer.get_messages(10000)
            filtered = [m for m in all_messages if m[1] == "DEBUG"]
            return filtered[-n:] if len(filtered) > n else filtered

        # Default fallback
        return self._log_buffer.get_messages(n)

    def _handle_keyboard_input(self) -> None:
        """Process keyboard input (non-blocking)."""
        if self._screen is None:
            return

        try:
            ch = self._screen.getch()
            if ch == -1:
                # No key pressed
                return

            # Tab key: cycle views
            if ch == ord("\t") or ch == 9:
                self._cycle_view()

            # 'q' key: quit (optional - could be disabled to prevent accidental quits)
            # elif ch == ord('q'):
            #     self._stop_event.set()

        except curses.error:
            pass

    def stop(self) -> None:
        """Signal the UI to stop."""
        self._stop_event.set()


# ---------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------


@dataclasses.dataclass
class IssueInfo:
    """Information about a single issue from the epic."""

    number: int
    title: str = ""
    depends_on: set[int] = dataclasses.field(default_factory=set)
    priority: str = "P2"  # P0, P1, P2
    status: str = "pending"  # pending, ready, in_progress, completed, paused, blocked_external

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "number": self.number,
            "title": self.title,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IssueInfo":
        """Create from dict."""
        return cls(
            number=data["number"],
            title=data.get("title", ""),
            depends_on=set(data.get("depends_on", [])),
            priority=data.get("priority", "P2"),
            status=data.get("status", "pending"),
        )


@dataclasses.dataclass
class PausedIssue:
    """Information about a paused issue."""

    worktree: str
    pr: int | None
    reason: str

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {"worktree": self.worktree, "pr": self.pr, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict) -> "PausedIssue":
        """Create from dict."""
        return cls(
            worktree=data.get("worktree", ""),
            pr=data.get("pr"),
            reason=data.get("reason", ""),
        )


@dataclasses.dataclass
class ImplementationState:
    """Persistent state for resumable execution."""

    epic_number: int
    issues: dict[int, IssueInfo] = dataclasses.field(default_factory=dict)
    completed_issues: set[int] = dataclasses.field(default_factory=set)
    paused_issues: dict[int, PausedIssue] = dataclasses.field(default_factory=dict)
    in_progress: dict[int, str] = dataclasses.field(default_factory=dict)  # issue -> worktree
    pr_numbers: dict[int, int] = dataclasses.field(default_factory=dict)  # issue -> PR number
    started_at: str = ""
    last_updated: str = ""
    _save_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, init=False, repr=False)

    def save(self, path: pathlib.Path) -> None:
        """Save state to JSON file with atomic write.

        Uses thread lock + file lock + atomic rename for safety.
        """
        with self._save_lock:  # Thread safety
            log("DEBUG", f"State.save(): Preparing data for {path}")
            data = {
                "epic_number": self.epic_number,
                "issues": {str(k): v.to_dict() for k, v in self.issues.items()},
                "completed_issues": list(self.completed_issues),
                "paused_issues": {str(k): v.to_dict() for k, v in self.paused_issues.items()},
                "in_progress": {str(k): v for k, v in self.in_progress.items()},
                "pr_numbers": {str(k): v for k, v in self.pr_numbers.items()},
                "started_at": self.started_at,
                "last_updated": dt.datetime.now().isoformat(),
            }

            json_content = json.dumps(data, indent=2)
            log("DEBUG", f"State.save(): JSON serialized, {len(json_content)} bytes")

            # Atomic write: write to temp file, then rename
            path.parent.mkdir(parents=True, exist_ok=True)
            log("DEBUG", f"State.save(): Creating temp file in {path.parent}")
            temp_fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            log("DEBUG", f"State.save(): Temp file created: {temp_path}")

            try:
                # Write to temp file with exclusive lock
                with os.fdopen(temp_fd, "w") as f:
                    # Acquire exclusive lock (blocks other processes)
                    import fcntl

                    log("DEBUG", "State.save(): Acquiring file lock...")
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    log("DEBUG", "State.save(): File lock acquired")
                    try:
                        f.write(json_content)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Set permissions before rename
                os.chmod(temp_path, 0o600)

                # Atomic rename
                os.replace(temp_path, str(path))
                log("DEBUG", f"Saved state to {path}")

            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise RuntimeError(f"Failed to save state: {e}")

    @classmethod
    def load(cls, path: pathlib.Path) -> "ImplementationState":
        """Load state from JSON file."""
        if not path.exists():
            return cls(epic_number=0)
        try:
            data = json.loads(path.read_text())
            return cls(
                epic_number=data.get("epic_number", 0),
                issues={int(k): IssueInfo.from_dict(v) for k, v in data.get("issues", {}).items()},
                completed_issues=set(data.get("completed_issues", [])),
                paused_issues={int(k): PausedIssue.from_dict(v) for k, v in data.get("paused_issues", {}).items()},
                in_progress={int(k): v for k, v in data.get("in_progress", {}).items()},
                pr_numbers={int(k): v for k, v in data.get("pr_numbers", {}).items()},
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            log("WARN", f"Failed to load state: {e}")
            return cls(epic_number=0)


@dataclasses.dataclass
class WorkerResult:
    """Result of processing a single issue."""

    issue: int
    status: str  # completed, paused, skipped, error
    pr_number: int | None = None
    error: str | None = None
    duration: float = 0.0


@dataclasses.dataclass(frozen=True)
class Options:
    """Configuration options for the implementer."""

    epic: int
    priority: str | None  # P0, P1, P2, or None for all
    issues: list[int] | None  # Specific issues to process
    parallel: int  # 1 = sequential, N = parallel workers
    dry_run: bool
    analyze: bool  # Just show dependency graph
    export_graph: str | None  # Export dependency graph to DOT file
    resume: bool
    timeout: int
    cleanup: bool
    state_dir: pathlib.Path | None
    api_delay: float  # seconds between GitHub API calls
    verbose: bool  # Enable verbose/debug output


# ---------------------------------------------------------------------
# Dependency Resolver
# ---------------------------------------------------------------------


class DependencyResolver:
    """Resolves issue dependencies and determines execution order."""

    def __init__(self, issues: dict[int, IssueInfo]) -> None:
        """Initialize the resolver."""
        self.issues = issues
        self._completed: set[int] = set()
        self._paused: set[int] = set()
        self._in_progress: set[int] = set()
        self._lock = threading.Lock()

    def initialize_from_state(self, state: ImplementationState) -> None:
        """Initialize resolver state from persistent state."""
        with self._lock:
            self._completed = state.completed_issues.copy()
            self._paused = set(state.paused_issues.keys())
            self._in_progress = set(state.in_progress.keys())

    def get_all_issue_numbers(self) -> set[int]:
        """Return all issue numbers in the graph."""
        return set(self.issues.keys())

    def get_ready_issues(self) -> list[int]:
        """Return issues whose dependencies are all completed.

        Algorithm:
        1. Filter issues not yet completed, paused, or in progress
        2. For each pending issue, check if all depends_on are in completed set
        3. Check external dependencies (not in epic) - if all are closed, treat as satisfied
        4. Return issues sorted by priority (P0 first) then by number
        """
        all_issues = self.get_all_issue_numbers()

        # Collect all external dependencies across ALL issues and prefetch their states
        all_external_deps = set()
        for info in self.issues.values():
            external_deps = info.depends_on - all_issues
            all_external_deps.update(external_deps)

        # Prefetch all external dependencies at once (batch API call)
        if all_external_deps:
            prefetch_issue_states(list(all_external_deps))

        priority_order = {"P0": 0, "P1": 1, "P2": 2}

        with self._lock:
            ready = []
            for num, info in self.issues.items():
                # Skip completed, paused, or in-progress issues
                if num in self._completed or num in self._paused or num in self._in_progress:
                    continue

                # Check for external dependencies (not in epic)
                external_deps = info.depends_on - all_issues
                if external_deps:
                    # Check if ALL external deps are closed (uses cached data from prefetch)
                    open_external = {d for d in external_deps if not is_issue_closed(d)}
                    if open_external:
                        info.status = "blocked_external"
                        continue
                    # All external deps are closed, treat as satisfied

                # All internal dependencies must be completed
                internal_deps = info.depends_on & all_issues
                if internal_deps.issubset(self._completed):
                    ready.append(num)

            return sorted(ready, key=lambda n: (priority_order.get(self.issues[n].priority, 2), n))

    def get_blocked_by_external(self) -> dict[int, set[int]]:
        """Return issues blocked by OPEN external dependencies.

        External dependencies that are closed are considered satisfied.
        """
        all_issues = self.get_all_issue_numbers()
        blocked = {}
        for num, info in self.issues.items():
            external_deps = info.depends_on - all_issues
            if external_deps:
                # Filter out closed external issues
                open_external = {d for d in external_deps if not is_issue_closed(d)}
                if open_external:
                    blocked[num] = open_external
        return blocked

    def get_all_pending_issues(self) -> list[int]:
        """Return all pending issues regardless of dependencies.

        Used for checking existing PRs that might need CI fixes.
        """
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        with self._lock:
            pending = []
            for num in self.issues:
                if num not in self._completed and num not in self._paused and num not in self._in_progress:
                    pending.append(num)
            return sorted(
                pending,
                key=lambda n: (priority_order.get(self.issues[n].priority, 2), n),
            )

    def mark_in_progress(self, issue: int) -> None:
        """Mark an issue as in progress."""
        with self._lock:
            self._in_progress.add(issue)
            self.issues[issue].status = "in_progress"

    def mark_completed(self, issue: int) -> None:
        """Mark an issue as completed, potentially unlocking dependents."""
        with self._lock:
            self._completed.add(issue)
            self._in_progress.discard(issue)
            self.issues[issue].status = "completed"

    def mark_paused(self, issue: int) -> None:
        """Mark an issue as paused (CI failed)."""
        with self._lock:
            self._paused.add(issue)
            self._in_progress.discard(issue)
            self.issues[issue].status = "paused"

    def has_cycle(self) -> bool:
        """Detect cycles using DFS with coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.issues}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for dep in self.issues[node].depends_on:
                if dep not in color:
                    continue  # External dependency
                if color[dep] == GRAY:
                    return True  # Back edge = cycle
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in self.issues)

    def get_topological_order(self) -> list[int]:
        """Return issues in dependency order (no issue before its dependencies).

        Raises:
            ValueError: If a dependency cycle is detected
        """
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        all_issues = self.get_all_issue_numbers()

        # Kahn's algorithm - only count internal dependencies
        in_degree = {n: len(info.depends_on & all_issues) for n, info in self.issues.items()}
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            # Pick by priority, then number
            queue.sort(key=lambda n: (priority_order.get(self.issues[n].priority, 2), n))
            node = queue.pop(0)
            result.append(node)

            for other, info in self.issues.items():
                if node in info.depends_on:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        # CRITICAL: Detect dependency cycles
        # If not all issues were processed, there's a cycle
        if len(result) != len(self.issues):
            unprocessed = set(self.issues.keys()) - set(result)
            # Build cycle description for error message
            cycle_details = []
            for issue_num in sorted(unprocessed):
                deps = self.issues[issue_num].depends_on & all_issues
                cycle_details.append(f"  #{issue_num} depends on: {sorted(deps)}")

            raise ValueError(
                f"Dependency cycle detected! {len(unprocessed)} issue(s) cannot be processed:\n"
                + "\n".join(cycle_details)
                + "\n\nPlease fix the circular dependencies in your epic issue."
            )

        return result


# ---------------------------------------------------------------------
# Worktree Manager
# ---------------------------------------------------------------------


class WorktreeManager:
    """Thread-safe worktree creation and cleanup."""

    def __init__(self, repo_root: pathlib.Path) -> None:
        """Initialize the manager."""
        self.repo_root = repo_root
        self.worktree_base = repo_root / "worktrees"
        self._lock = threading.Lock()

        # Ensure worktrees directory exists
        self.worktree_base.mkdir(exist_ok=True)

    def create(self, issue: int, description: str) -> pathlib.Path:
        """Create a worktree for an issue. Thread-safe."""
        with self._lock:
            # Sanitize description for branch name
            safe_desc = re.sub(r"[^a-zA-Z0-9-]", "-", description.lower())[:30]
            branch_name = f"{issue}-{safe_desc}"
            worktree_path = self.worktree_base / branch_name

            if worktree_path.exists():
                log("WARN", f"Worktree already exists: {worktree_path}")
                return worktree_path

            # Check if branch exists
            cp = run(
                ["git", "show-ref", "--verify", f"refs/heads/{branch_name}"],
                cwd=self.repo_root,
            )

            if cp.returncode == 0:
                # Branch exists, use it
                cp = run(
                    ["git", "worktree", "add", str(worktree_path), branch_name],
                    cwd=self.repo_root,
                )
            else:
                # Create new branch from main
                cp = run(
                    [
                        "git",
                        "worktree",
                        "add",
                        "-b",
                        branch_name,
                        str(worktree_path),
                        "origin/main",
                    ],
                    cwd=self.repo_root,
                )

            if cp.returncode != 0:
                raise RuntimeError(f"Failed to create worktree: {cp.stderr}")

            return worktree_path

    def remove(self, issue: int) -> bool:
        """Remove a worktree by issue number. Thread-safe."""
        with self._lock:
            # Find worktree matching issue
            for path in self.worktree_base.iterdir():
                if path.name.startswith(f"{issue}-"):
                    branch_name = path.name
                    cp = run(
                        ["git", "worktree", "remove", "--force", str(path)],
                        cwd=self.repo_root,
                    )
                    if cp.returncode != 0:
                        log("WARN", f"Failed to remove worktree: {cp.stderr}")
                        return False

                    # Delete local branch
                    run(["git", "branch", "-D", branch_name], cwd=self.repo_root)

                    return True
            return False

    def create_for_existing_branch(self, issue: int, remote_branch: str) -> pathlib.Path:
        """Create a worktree for an existing remote branch. Thread-safe."""
        with self._lock:
            worktree_path = self.worktree_base / f"{issue}-{remote_branch}"

            # Check if any existing worktree already has this branch checked out
            cp = run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root)
            for line in cp.stdout.split("\n"):
                if line.startswith("worktree "):
                    existing_path = pathlib.Path(line.split(" ", 1)[1])
                    # Check what branch this worktree is on
                    branch_cp = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=existing_path)
                    if branch_cp.returncode == 0 and branch_cp.stdout.strip() == remote_branch:
                        log(
                            "DEBUG",
                            f"Found existing worktree for branch {remote_branch}: {existing_path}",
                        )
                        # Update it to latest
                        success, error_msg = safe_git_fetch(self.repo_root, remote_branch)
                        if not success:
                            log("WARN", f"Failed to fetch branch {remote_branch}: {error_msg}")
                        else:
                            run(
                                ["git", "reset", "--hard", f"origin/{remote_branch}"],
                                cwd=existing_path,
                            )
                        return existing_path

            if worktree_path.exists():
                log("DEBUG", f"Worktree path exists: {worktree_path}")
                # Make sure it's on the right branch and up to date
                success, error_msg = safe_git_fetch(self.repo_root, remote_branch)
                if not success:
                    log("WARN", f"Failed to fetch branch {remote_branch}: {error_msg}")
                else:
                    run(
                        ["git", "reset", "--hard", f"origin/{remote_branch}"],
                        cwd=worktree_path,
                    )
                return worktree_path

            # Fetch the remote branch first
            success, error_msg = safe_git_fetch(self.repo_root, remote_branch)
            if not success:
                raise RuntimeError(f"Failed to fetch branch {remote_branch}: {error_msg}")

            # Create worktree tracking the remote branch
            cp = run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--track",
                    "-b",
                    remote_branch,
                    str(worktree_path),
                    f"origin/{remote_branch}",
                ],
                cwd=self.repo_root,
            )

            if cp.returncode != 0:
                # Branch might already exist locally, try without -b
                cp = run(
                    ["git", "worktree", "add", str(worktree_path), remote_branch],
                    cwd=self.repo_root,
                )
                if cp.returncode != 0:
                    raise RuntimeError(f"Failed to create worktree for {remote_branch}: {cp.stderr}")

            return worktree_path

    def list_active(self) -> dict[int, pathlib.Path]:
        """List active worktrees mapped to issue numbers."""
        result = {}
        cp = run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root)

        for line in cp.stdout.split("\n"):
            if line.startswith("worktree "):
                path = pathlib.Path(line.split(" ", 1)[1])
                if path.parent == self.worktree_base:
                    # Extract issue number from name
                    match = re.match(r"(\d+)-", path.name)
                    if match:
                        result[int(match.group(1))] = path

        return result

    def cleanup_stale(self) -> int:
        """Remove stale worktrees. Returns count of removed worktrees."""
        count = 0
        cp = run(["git", "worktree", "prune"], cwd=self.repo_root)
        if cp.returncode == 0:
            log("INFO", "Pruned stale worktrees")

        # Remove any leftover directories
        for path in self.worktree_base.iterdir():
            if not path.is_dir():
                continue
            # Check if this worktree is registered
            cp = run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root)
            if str(path) not in cp.stdout:
                shutil.rmtree(path, ignore_errors=True)
                count += 1
                log("INFO", f"Removed orphaned worktree: {path}")

        return count


# ---------------------------------------------------------------------
# Issue Implementer
# ---------------------------------------------------------------------


class IssueImplementer:
    """Implements individual issues using Claude sub-agents."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        tempdir: pathlib.Path,
        opts: Options,
        state: ImplementationState,
        resolver: Optional["DependencyResolver"],
        worktree_manager: "WorktreeManager",
        status_tracker: Optional["StatusTracker"] = None,
    ) -> None:
        """Initialize the implementer."""
        self.repo_root = repo_root
        self.tempdir = tempdir
        self.opts = opts
        self.state = state
        self.resolver = resolver
        self.worktree_manager = worktree_manager
        self.status_tracker = status_tracker
        self.state_file = (opts.state_dir or tempdir) / "implementation_state.json"

        self._api_lock = threading.Lock()
        self._last_api_call = 0.0

    def _gh_call(self, args: list[str], retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
        """Make a rate-limited GitHub CLI call with retries.

        CRITICAL: The API call must happen INSIDE the lock to prevent concurrent calls.
        """
        # Execute with retries
        for attempt in range(1, retries + 1):
            with self._api_lock:
                # Wait for rate limit delay
                elapsed = time.time() - self._last_api_call
                if elapsed < self.opts.api_delay:
                    time.sleep(self.opts.api_delay - elapsed)

                # Execute API call INSIDE lock to prevent concurrent calls
                cp = run(args)
                self._last_api_call = time.time()

            if cp.returncode == 0:
                return cp

            # Check for network/rate limit errors
            stderr_lower = cp.stderr.lower()
            is_network_error = any(
                msg in stderr_lower
                for msg in [
                    "connection",
                    "network",
                    "timeout",
                    "eof",
                    "dns",
                    "resolve",
                    "rate limit",
                ]
            )
            if is_network_error and attempt < retries:
                delay = 2**attempt
                log(
                    "WARN",
                    f"API call failed (attempt {attempt}/{retries}), retrying in {delay}s...",
                )
                time.sleep(delay)
                continue
            break
        return cp

    def _fetch_issues_batch(self, issue_nums: list[int]) -> dict[int, dict] | None:
        """Fetch multiple issues in one GraphQL query.

        Returns:
            dict mapping issue number to {title, state} if successful
            None if fetch failed (allows caller to distinguish error from empty result)
        """
        if not issue_nums:
            return {}

        # Build GraphQL query for batch fetch
        # GitHub GraphQL allows fetching multiple nodes by ID
        repo_owner, repo_name = get_repo_info()

        # Build query fragments for each issue
        fragments = []
        for i, num in enumerate(issue_nums):
            fragments.append(f"issue{i}: issue(number: {num}) {{ number title state }}")

        query = f"""
        query {{
          repository(owner: "{repo_owner}", name: "{repo_name}") {{
            {chr(10).join(fragments)}
          }}
        }}
        """

        cp = self._gh_call(["gh", "api", "graphql", "-f", f"query={query}"])
        if cp.returncode != 0:
            log("WARN", f"GraphQL batch fetch failed: {cp.stderr}")
            return None  # Explicit error signal

        try:
            data = json.loads(cp.stdout)
            repo_data = data.get("data", {}).get("repository", {})
            result = {}
            for i, num in enumerate(issue_nums):
                issue_data = repo_data.get(f"issue{i}")
                if issue_data:
                    result[num] = {
                        "title": issue_data.get("title", f"Issue #{num}"),
                        "state": issue_data.get("state", "OPEN"),
                    }
            return result
        except (json.JSONDecodeError, KeyError) as e:
            log("WARN", f"Failed to parse GraphQL response: {e}")
            return None  # Explicit error signal

    def _fetch_issue_title(self, issue_num: int) -> str:
        """Fetch title for a single issue (fallback, rate-limited)."""
        cp = self._gh_call(["gh", "issue", "view", str(issue_num), "--json", "title"])
        if cp.returncode == 0:
            try:
                data = json.loads(cp.stdout)
                # Validate data is a dict
                if not isinstance(data, dict):
                    log("WARN", f"Expected dict from gh issue view, got {type(data)}")
                    return f"Issue #{issue_num}"
                return data.get("title", f"Issue #{issue_num}")
            except json.JSONDecodeError as e:
                log("WARN", f"Failed to parse issue #{issue_num} title: {e}")
        return f"Issue #{issue_num}"

    def fetch_titles_for_issues(self, issue_nums: list[int]) -> None:
        """Fetch titles for a list of issues using batched GraphQL queries."""
        if not issue_nums:
            return

        # Filter to only issues that don't have titles yet
        to_fetch = [n for n in issue_nums if n in self.state.issues and not self.state.issues[n].title]
        if not to_fetch:
            return

        log("INFO", f"Fetching titles for {len(to_fetch)} issues...")

        # Batch fetch in groups of 50 (GraphQL limit is higher but be conservative)
        batch_size = 50
        fetched = 0
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i : i + batch_size]
            results = self._fetch_issues_batch(batch)

            for issue_num in batch:
                if results is not None and issue_num in results:
                    if issue_num in self.state.issues:
                        self.state.issues[issue_num].title = results[issue_num]["title"]
                else:
                    # Fallback to individual fetch if batch missed this one or batch failed
                    title = self._fetch_issue_title(issue_num)
                    if issue_num in self.state.issues:
                        self.state.issues[issue_num].title = title

            fetched += len(batch)
            print(f"\r  Fetched: {fetched}/{len(to_fetch)}", end="", flush=True)

        print()  # Newline after progress

    def parse_epic(self, epic_number: int) -> dict[int, IssueInfo]:
        """Parse an epic issue to extract issues with dependencies."""
        log("INFO", f"Parsing epic #{epic_number}...")

        # Retry with exponential backoff for network errors
        for attempt in range(1, MAX_RETRIES + 1):
            cp = run(["gh", "issue", "view", str(epic_number), "--json", "body,title"])
            if cp.returncode == 0:
                break

            # Check for network-related errors
            stderr_lower = cp.stderr.lower()
            is_network_error = any(
                msg in stderr_lower for msg in ["connection", "network", "timeout", "eof", "dns", "resolve"]
            )

            if is_network_error:
                if attempt < MAX_RETRIES:
                    delay = 2**attempt
                    log(
                        "WARN",
                        f"Network error fetching epic #{epic_number} (attempt {attempt}/{MAX_RETRIES})",
                    )
                    log("INFO", f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    log("ERROR", "Network connection failed. Please check:")
                    log("ERROR", "  1. Your internet connection")
                    log("ERROR", "  2. GitHub status: https://githubstatus.com")
                    log("ERROR", "  3. Your VPN/proxy settings")
                    sys.exit(1)
            else:
                # Non-network error, fail immediately
                log("ERROR", f"Failed to fetch epic #{epic_number}: {cp.stderr}")
                sys.exit(1)

        data = json.loads(cp.stdout)
        body = data.get("body", "")

        issues: dict[int, IssueInfo] = {}
        current_priority = "P2"

        # Match priority section headers
        priority_re = re.compile(r"###\s*(P[012]):\s*(\w+)")
        # Match issue lines: - [ ] #123 or - [x] #123 (depends on: #456, #789)
        issue_re = re.compile(r"-\s*\[([x ])\]\s*#(\d+)(?:\s*\(depends on:\s*([^)]+)\))?")

        for line in body.split("\n"):
            # Check for priority header
            priority_match = priority_re.search(line)
            if priority_match:
                current_priority = priority_match.group(1)
                continue

            # Check for issue line
            issue_match = issue_re.search(line)
            if issue_match:
                checked, issue_num_str, deps_str = issue_match.groups()
                issue_num = int(issue_num_str)

                # Parse dependencies
                depends_on: set[int] = set()
                if deps_str:
                    for dep_match in re.finditer(r"#(\d+)", deps_str):
                        depends_on.add(int(dep_match.group(1)))

                # Mark as completed if checked
                status = "completed" if checked == "x" else "pending"

                issues[issue_num] = IssueInfo(
                    number=issue_num,
                    depends_on=depends_on,
                    priority=current_priority,
                    status=status,
                )

        # Validate that at least one issue was found
        if not issues:
            log("ERROR", f"No issues found in epic #{epic_number}")
            log("ERROR", "Expected format in epic body:")
            log("ERROR", "  ### P0: Foundation")
            log("ERROR", "  - [ ] #123 (depends on: #456, #789)")
            log("ERROR", "  - [x] #124")
            log("ERROR", "")
            log("ERROR", "Epic body preview (first 500 chars):")
            log("ERROR", f"{body[:500]}")
            raise ValueError(
                f"No issues found in epic #{epic_number}. "
                "Check the epic issue format - expected checklist items like '- [ ] #123'"
            )

        # Titles are fetched lazily when needed (to avoid rate limiting)
        log("INFO", f"Parsed {len(issues)} issues from epic #{epic_number}")
        return issues

    def _update_status(self, slot: int, issue: int, stage: str, info: str = "") -> None:
        """Update status tracker if available."""
        if self.status_tracker and slot >= 0:
            self.status_tracker.update(slot, issue, stage, info)

    def _check_or_create_plan(self, issue: int, slot: int) -> str | None:
        """Check if issue has a plan, create one if missing. Returns plan content."""
        self._update_status(slot, issue, "Plan", "checking")

        # Fetch issue comments
        cp = run(["gh", "issue", "view", str(issue), "--json", "comments"])
        if cp.returncode != 0:
            return None

        data = json.loads(cp.stdout)
        comments = data.get("comments", [])

        # Look for plan comment
        for comment in comments:
            body = comment.get("body", "")
            if PLAN_COMMENT_HEADER in body:
                log("INFO", f"  Found existing plan for #{issue}")
                return body

        # No plan found, generate one
        log("INFO", f"  No plan found for #{issue}, generating...")
        self._update_status(slot, issue, "Plan", "generating")

        # Call plan_issues.py to generate plan
        scripts_dir = self.repo_root / "scripts"
        cp = run(
            [
                "python3",
                str(scripts_dir / "plan_issues.py"),
                "--issues",
                str(issue),
                "--auto",
            ],
            timeout=600,  # 10 minute timeout for plan generation
            cwd=self.repo_root,
        )

        if cp.returncode != 0:
            log("WARN", f"  Failed to generate plan for #{issue}: {cp.stderr}")
            return None

        # Fetch the newly created plan
        cp = run(["gh", "issue", "view", str(issue), "--json", "comments"])
        if cp.returncode != 0:
            return None

        data = json.loads(cp.stdout)
        for comment in data.get("comments", []):
            body = comment.get("body", "")
            if PLAN_COMMENT_HEADER in body:
                log("INFO", f"  Generated plan for #{issue}")
                return body

        return None

    def _spawn_claude_agent(
        self,
        worktree: pathlib.Path,
        prompt: str,
        slot: int,
        issue: int,
    ) -> tuple[bool, str]:
        """Spawn a Claude haiku agent to implement the issue."""
        log("DEBUG", f"  _spawn_claude_agent: Starting for issue #{issue}, shutdown_flag={is_shutdown_requested()}")
        self._update_status(slot, issue, "Claude", "starting")

        cmd = [
            "claude",
            "--model",
            "haiku",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Read,Write,Edit,Glob,Grep,Bash",
            "--add-dir",
            str(worktree),
            "-p",
            prompt,
        ]

        log_file = self.tempdir / f"issue-{issue}-claude.log"
        log("DEBUG", f"  Claude command: {' '.join(cmd[:6])}...")
        log("DEBUG", f"  Claude log file: {log_file}")

        # Output limits to prevent memory exhaustion
        MAX_OUTPUT_LINES = 10000
        MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=worktree,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line-buffered
            )

            deadline = time.time() + self.opts.timeout
            line_count = 0
            total_bytes = 0
            last_meaningful_line = ""
            last_update_time = time.time()
            update_interval = 0.5  # Update at most every 0.5 seconds

            with log_file.open("w") as log_handle:
                # Read output line-by-line, streaming to file (not memory)
                while True:
                    # Check shutdown flag first - allow graceful exit
                    if is_shutdown_requested():
                        log("DEBUG", "  Shutdown requested, killing Claude subprocess")
                        proc.kill()
                        proc.wait(timeout=5)
                        return False, "Shutdown requested"

                    # Check timeout
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        kill_process_safely(proc, timeout=5)
                        log("DEBUG", "  Claude timed out")
                        return False, "Timeout exceeded"

                    # Non-blocking read with timeout using select
                    import select

                    readable, _, _ = select.select([proc.stdout], [], [], 0.5)

                    if readable and proc.stdout is not None:
                        line = proc.stdout.readline()
                        if not line:  # EOF
                            break

                        # Write to file (not memory)
                        log_handle.write(line)
                        log_handle.flush()

                        # Track size limits
                        line_count += 1
                        total_bytes += len(line.encode("utf-8"))

                        if line_count > MAX_OUTPUT_LINES:
                            kill_process_safely(proc, timeout=5)
                            return False, f"Output exceeded {MAX_OUTPUT_LINES} lines"

                        if total_bytes > MAX_OUTPUT_SIZE:
                            kill_process_safely(proc, timeout=5)
                            return False, f"Output exceeded {MAX_OUTPUT_SIZE / 1024 / 1024:.0f}MB"

                        # Extract meaningful output for status
                        stripped = line.strip()
                        if stripped and not stripped.startswith("{") and len(stripped) < 200:
                            display_line = stripped[:60]
                            last_meaningful_line = display_line

                            if self.opts.verbose:
                                print(f"    [Claude] {stripped[:100]}", flush=True)

                        # Update status with throttling
                        now = time.time()
                        if last_meaningful_line and (now - last_update_time >= update_interval):
                            self._update_status(slot, issue, "Claude", last_meaningful_line)
                            last_update_time = now
                    else:
                        # Check if process exited
                        if proc.poll() is not None:
                            break

            # Process finished - get exit code
            returncode = proc.wait(timeout=5)
            log("DEBUG", f"  Claude finished with {line_count} lines, exit code {returncode}")

            # Read last 500 lines from log file for error checking
            with log_file.open("r") as f:
                lines = f.readlines()
                tail = "".join(lines[-500:])  # Last 500 lines only

            # Check for Claude API usage limit (FATAL - must stop processing)
            if detect_claude_usage_limit(tail):
                log("ERROR", "Claude API usage limit detected - stopping all processing")
                raise RuntimeError(
                    "Claude API usage limit reached. Please wait before running again.\n\n"
                    f"Error details:\n{tail[-500:]}"
                )

            # Check for GitHub rate limit
            reset = detect_rate_limit(tail)
            if reset:
                wait_until(reset)
                return False, "Rate limited - retry needed"

            if returncode == 0:
                return True, tail
            else:
                return False, f"Exit code {returncode}: {tail[-500:]}"

        except (KeyboardInterrupt, SystemExit):
            # Clean shutdown - kill process and re-raise
            kill_process_safely(proc, timeout=5)
            raise
        except subprocess.TimeoutExpired as e:
            log("ERROR", f"  Claude timed out: {e}")
            return False, "Timeout exceeded"
        except OSError as e:
            log("ERROR", f"  Claude process error: {e}")
            return False, f"Process error: {e}"
        except Exception as e:
            import traceback

            log("ERROR", f"  Claude execution failed: {e}\n{traceback.format_exc()}")
            # Ensure process is dead
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            return False, str(e)

    def _get_summary(self, worktree: pathlib.Path, slot: int, issue: int) -> str:
        """Get a concise summary of changes from Claude."""
        self._update_status(slot, issue, "Summary", "generating")

        prompt = """Concisely summarize the changes you just made in 2-3 sentences.
Focus on what functionality was added or fixed. Do not include implementation details."""

        cmd = [
            "claude",
            "--model",
            "haiku",
            "-p",
            prompt,
            "--add-dir",
            str(worktree),
        ]

        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=worktree,
            )
            if cp.returncode == 0:
                return cp.stdout.strip()
        except (KeyboardInterrupt, SystemExit):
            raise
        except subprocess.TimeoutExpired:
            log("WARN", "  Summary generation timed out")
        except (OSError, subprocess.SubprocessError) as e:
            log("WARN", f"  Failed to get summary: {e}")

        return "Implementation completed."

    # ============================================================================
    # DISABLED: PR creation functionality was removed (see line 2743, 2840)
    # These methods (_check_existing_pr, _analyze_and_fix_pr) are preserved
    # for potential future restoration but are NOT called anywhere.
    # If PR creation is re-enabled, uncomment the call sites and update logic.
    # ============================================================================

    def _check_existing_pr(self, issue: int) -> dict | None:
        """Check if a PR already exists for this issue.

        **DISABLED**: This method is not called - PR creation was removed.

        Returns PR info dict if found, None otherwise.
        Checks both by issue number in title/body AND by branch name pattern.
        """
        # First try searching by issue number in title/body
        cp = self._gh_call(
            [
                "gh",
                "pr",
                "list",
                "--search",
                f"#{issue} in:title,body",
                "--json",
                "number,state,headRefName,statusCheckRollup",
            ]
        )
        if cp.returncode == 0:
            try:
                data = json.loads(cp.stdout)
                # Validate data is a list
                if not isinstance(data, list):
                    log("WARN", f"Expected list from gh pr list, got {type(data)}")
                    prs = []
                else:
                    prs = data

                # Find open PR for this issue
                for pr in prs:
                    # Validate pr is a dict
                    if not isinstance(pr, dict):
                        log("WARN", f"Skipping non-dict PR entry: {type(pr)}")
                        continue
                    if pr.get("state") == "OPEN":
                        return pr

                # Check for merged PRs too
                for pr in prs:
                    if not isinstance(pr, dict):
                        continue
                    if pr.get("state") == "MERGED":
                        return pr
            except json.JSONDecodeError as e:
                log("WARN", f"Failed to parse PR list: {e}")
            except TypeError as e:
                log("ERROR", f"Type error iterating PRs: {e}")

        # Also check by branch name pattern (issue number prefix)
        cp = self._gh_call(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "all",
                "--json",
                "number,state,headRefName,statusCheckRollup",
            ]
        )
        if cp.returncode == 0:
            try:
                data = json.loads(cp.stdout)
                # Validate data is a list
                if not isinstance(data, list):
                    log("WARN", f"Expected list from gh pr list, got {type(data)}")
                    prs = []
                else:
                    prs = data

                for pr in prs:
                    # Validate pr is a dict
                    if not isinstance(pr, dict):
                        log("WARN", f"Skipping non-dict PR entry: {type(pr)}")
                        continue

                    branch = pr.get("headRefName", "")
                    # Check if branch starts with issue number
                    if branch.startswith(f"{issue}-") or branch.startswith(f"{issue}_"):
                        state = pr.get("state", "")
                        if state == "OPEN":
                            log(
                                "DEBUG",
                                f"  Found existing PR by branch: #{pr['number']} ({branch})",
                            )
                            return pr
                        elif state == "MERGED":
                            log(
                                "DEBUG",
                                f"  Found merged PR by branch: #{pr['number']} ({branch})",
                            )
                            return pr
            except json.JSONDecodeError as e:
                log("WARN", f"Failed to parse PR list: {e}")
            except TypeError as e:
                log("ERROR", f"Type error iterating PRs: {e}")

        return None

    def _analyze_and_fix_pr(self, pr_info: dict, issue: int, slot: int) -> WorkerResult:
        """Analyze CI failures on an existing PR and attempt to fix them.

        **DISABLED**: This method is not called - PR creation was removed.
        """
        start_time = time.time()
        pr_number = pr_info["number"]
        branch = pr_info.get("headRefName", "")

        self._update_status(slot, issue, "PR", f"analyzing #{pr_number}")
        log("INFO", f"  Found existing PR #{pr_number} for issue #{issue}")
        log("DEBUG", f"  PR branch: {branch}")

        # Check CI status
        checks = pr_info.get("statusCheckRollup", []) or []
        failing = [c for c in checks if c.get("conclusion") == "FAILURE"]
        log("DEBUG", f"  Total checks: {len(checks)}, failing: {len(failing)}")

        if not failing:
            # No failures - PR is in good state, move to next issue
            log("DEBUG", "  No CI failures detected, PR is in good state")
            log("INFO", f"  PR #{pr_number} has no CI failures - moving to next issue")
            duration = time.time() - start_time
            return WorkerResult(issue, "completed", pr_number, None, duration)

        # Get failure details
        self._update_status(slot, issue, "CI", "fetching logs")
        failure_info = []
        for check in failing:
            name = check.get("name", "unknown")
            failure_info.append(f"- {name}: FAILURE")
            log("DEBUG", f"  Failing check: {name}")

        log("INFO", f"  CI failures: {len(failing)} checks failed")

        # Get or create worktree for the existing PR branch
        log("DEBUG", f"  Setting up worktree for existing branch: {branch}")
        self._update_status(slot, issue, "Worktree", "setting up")
        worktree = self.worktree_manager.create_for_existing_branch(issue, branch)
        log("DEBUG", f"  Worktree ready: {worktree}")
        # Update state with synchronization, then save (save() handles its own locking)
        with self.state._save_lock:
            self.state.in_progress[issue] = str(worktree)
            self.state.pr_numbers[issue] = pr_number
        self.state.save(self.state_file)

        # Fetch CI logs
        log("DEBUG", "  Fetching CI run info")
        self._update_status(slot, issue, "CI", "analyzing failures")
        cp = run(
            [
                "gh",
                "run",
                "list",
                "--branch",
                branch,
                "--limit",
                "1",
                "--json",
                "databaseId",
            ]
        )
        run_id = None
        if cp.returncode == 0:
            try:
                runs = json.loads(cp.stdout)
                if runs:
                    run_id = runs[0].get("databaseId")
                    log("DEBUG", f"  Found CI run ID: {run_id}")
            except json.JSONDecodeError:
                log("DEBUG", "  Failed to parse CI run list")

        log_content = ""
        if run_id:
            log("DEBUG", f"  Fetching failed logs for run {run_id}")
            cp = run(["gh", "run", "view", str(run_id), "--log-failed"], timeout=60)
            if cp.returncode == 0:
                log_content = cp.stdout[-5000:]  # Last 5000 chars
                log("DEBUG", f"  Got {len(log_content)} chars of logs")
            else:
                log("DEBUG", f"  Failed to get logs: {cp.stderr}")
        else:
            log("DEBUG", "  No CI run ID found")

        # Build fix prompt
        log("DEBUG", "  Building fix prompt for Claude")
        fix_prompt = f"""You are a code fix agent. Your task is to fix CI failures for GitHub issue #{issue}.

## Working Directory
Your current working directory is: {worktree}
All file operations should be relative to this directory.

## Branch
You are on branch: {branch}

## PR Information
PR #{pr_number} has failing CI checks:
{chr(10).join(failure_info)}

## Failure Logs
```
{log_content}
```

## Instructions
1. Analyze the failure logs to understand what's broken
2. Use the Read tool to examine the failing code
3. Use the Edit tool to fix the issues in the code
4. Run tests with `pixi run mojo test` to verify fixes
5. COMMIT your changes before finishing using: git add -A && git commit -m "fix: Address CI failures for #{issue}"
6. DO NOT just output text - you MUST make actual file changes AND commit them

## Critical Rules
- You MUST edit files to fix the issues - do not just describe what to do
- You MUST commit your changes before finishing - branches with uncommitted changes are not acceptable
- Use absolute paths starting with {worktree}
- Follow Mojo v0.26.1+ syntax (out self for constructors, mut self for mutating methods)
- If logs are empty, run `pixi run mojo test` to see the actual errors
"""

        # Run Claude to fix
        log("DEBUG", "  Spawning Claude agent to fix issues")
        self._update_status(slot, issue, "Fix", "running Claude")
        success, output = self._spawn_claude_agent(worktree, fix_prompt, slot, issue)
        log("DEBUG", f"  Claude returned: success={success}, output_len={len(output)}")

        if not success:
            log("DEBUG", f"  Claude failed: {output[-200:]}")
            self._post_issue_update(
                issue,
                f"⚠️ Failed to fix CI failures automatically.\n\nError: {output[-200:]}",
            )
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "fix failed", duration)

        # Check for changes
        log("DEBUG", "  Checking for changes")
        cp = run(["git", "status", "--porcelain"], cwd=worktree)
        if not cp.stdout.strip():
            log("DEBUG", "  No changes made by Claude")
            self._post_issue_update(issue, "ℹ️ No changes needed - CI may need manual investigation.")
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "no changes made", duration)

        # Commit and push
        log("DEBUG", "  Committing and pushing changes")
        self._update_status(slot, issue, "Git", "committing fix")

        # Call Claude to analyze issue, PR, and changes to create a proper commit
        commit_prompt = f"""You need to commit the CI fix changes for GitHub issue #{issue}.

## Working Directory
{worktree}

## PR Information
This is a fix for PR #{pr_number} which had failing CI checks:
{chr(10).join(failure_info)}

## Your Task
1. Read the GitHub issue for context: `gh issue view {issue}`
2. Read the PR for context: `gh pr view {pr_number}`
3. Run `git diff` to see what files were changed and how
4. Run `git status` to see all modified/added/deleted files
5. Based on the issue, PR, failures, and actual fixes, create a meaningful commit message
6. Commit using: git add -A && git commit -m "your message"

## Commit Message Requirements
- Start with fix: since this addresses CI failures
- First line: brief summary of what was fixed (50 chars or less)
- Body: explain what was broken and how it was fixed
- Be specific about the actual changes made

Example format:
```
fix(core): correct parameter order in variance function

The variance function was passing axis parameter incorrectly,
causing dimension mismatch errors in CI tests.

- Fixed parameter order in _compute_variance call
- Updated test assertions to match expected output shape
```

DO NOT describe what you're doing - just run the commands to commit.
"""
        self._spawn_claude_agent(worktree, commit_prompt, slot, issue)

        # Verify commit happened
        cp = run(["git", "status", "--porcelain"], cwd=worktree)
        if cp.stdout.strip():
            # Claude still didn't commit - force commit ourselves
            log("WARN", f"  Claude failed to commit fix, forcing commit for #{issue}")
            run(["git", "add", "-A"], cwd=worktree)
            run(
                [
                    "git",
                    "commit",
                    "-m",
                    "fix: Address CI failures\n\nAutomated fix by implement_issues.py",
                ],
                cwd=worktree,
            )

        self._update_status(slot, issue, "Git", "pushing fix")
        cp = run(["git", "push", "origin", branch], cwd=worktree)

        if cp.returncode != 0:
            log("DEBUG", f"  Push failed: {cp.stderr}")
            self._post_issue_update(issue, f"⚠️ Failed to push fix: {cp.stderr}")
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "push failed", duration)

        log("DEBUG", "  Push successful, moving to next issue")
        self._post_issue_update(issue, f"🔧 Pushed fix to PR #{pr_number}. Moving to next issue.")

        # Cleanup worktree since we're moving on
        self.worktree_manager.remove(issue)
        # Update state with synchronization, then save (save() handles its own locking)
        with self.state._save_lock:
            del self.state.in_progress[issue]
        self.state.save(self.state_file)

        duration = time.time() - start_time
        log("INFO", f"  Fixed PR #{pr_number} - moving to next issue")
        return WorkerResult(issue, "completed", pr_number, None, duration)

    def _post_issue_update(self, issue: int, message: str) -> None:
        """Post an update comment to the GitHub issue."""
        comment = f"{message}\n\n---\n*Automated by implement_issues.py at {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        proc = subprocess.Popen(
            ["gh", "issue", "comment", str(issue), "--body-file", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            proc.communicate(input=comment, timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            log("WARN", f"Timeout posting update to issue #{issue}")

    def _poll_pr_status(self, pr_number: int, slot: int, issue: int) -> str:
        """Poll PR status until merged, failed, or timeout.

        Returns: "merged", "failed", "timeout", "stalled"
        """
        max_wait = PR_MAX_WAIT
        start = time.time()
        passed_but_not_merged_count = 0
        MAX_PASSED_POLLS = 5  # If passed 5 times but not merged, give up

        while time.time() - start < max_wait:
            # Check shutdown during polling
            if is_shutdown_requested():
                log("INFO", f"Shutdown requested, stopping PR poll for #{pr_number}")
                return "timeout"

            self._update_status(slot, issue, "PR", "checking")

            cp = run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    "state,mergeStateStatus,statusCheckRollup,autoMergeRequest",
                ]
            )

            if cp.returncode != 0:
                log("WARN", f"  Failed to check PR #{pr_number}: {cp.stderr[:100]}")
                self._update_status(slot, issue, "PR", "error")
                time.sleep(PR_POLL_INTERVAL)
                continue

            try:
                data = json.loads(cp.stdout)
            except json.JSONDecodeError as e:
                log("ERROR", f"  Failed to parse PR status: {e}")
                time.sleep(PR_POLL_INTERVAL)
                continue

            state = data.get("state", "").upper()
            checks = data.get("statusCheckRollup", []) or []
            auto_merge = data.get("autoMergeRequest")  # Check auto-merge status

            if state == "MERGED":
                return "merged"

            if state == "CLOSED":
                return "failed"

            # Count check statuses
            pending = sum(1 for c in checks if c.get("status") in ("PENDING", "QUEUED"))
            failing = sum(1 for c in checks if c.get("conclusion") == "FAILURE")

            if failing > 0:
                self._update_status(slot, issue, "PR", f"CI failing ({failing})")
                return "failed"
            elif pending > 0:
                self._update_status(slot, issue, "PR", f"CI running ({pending})")
                passed_but_not_merged_count = 0  # Reset counter
            else:
                # All checks passed
                if auto_merge is None:
                    # Auto-merge not enabled - will never merge automatically
                    log("WARN", f"  PR #{pr_number} has passed CI but auto-merge not enabled")
                    return "stalled"

                # Auto-merge enabled but not merged yet
                passed_but_not_merged_count += 1
                if passed_but_not_merged_count >= MAX_PASSED_POLLS:
                    # Passed CI multiple times but still not merged - something wrong
                    log(
                        "WARN",
                        f"  PR #{pr_number} passed CI {MAX_PASSED_POLLS} times but not merged. "
                        f"Possible merge conflicts or approval needed.",
                    )
                    return "stalled"

                self._update_status(
                    slot, issue, "PR", f"waiting for auto-merge ({passed_but_not_merged_count}/{MAX_PASSED_POLLS})"
                )

            time.sleep(PR_POLL_INTERVAL)

        return "timeout"

    def implement_issue(self, issue: int, slot: int) -> WorkerResult:
        """Full workflow for implementing a single issue."""
        # Set worker slot for this thread so logs go to the right file
        set_worker_slot(slot)

        start_time = time.time()
        worktree: pathlib.Path | None = None

        try:
            # Check shutdown flag at start
            if is_shutdown_requested():
                log("INFO", f"Shutdown requested, exiting worker for #{issue}")
                return WorkerResult(issue, "paused", None, "Shutdown requested", time.time() - start_time)

            issue_info = self.state.issues.get(issue)
            if not issue_info:
                return WorkerResult(issue, "error", None, "Issue not found in state", 0)

            # Fetch title if not already fetched (lazy loading)
            if not issue_info.title:
                issue_info.title = self._fetch_issue_title(issue)

            title = issue_info.title or f"Issue #{issue}"
            log("INFO", f"Implementing #{issue}: {title}")

            # NOTE: PR creation removed - no longer checking for existing PRs

            # 1. Check/create plan
            plan = self._check_or_create_plan(issue, slot)
            if not plan:
                log("WARN", f"  Could not get/create plan for #{issue}, skipping")
                return WorkerResult(
                    issue,
                    "skipped",
                    None,
                    "No plan available",
                    time.time() - start_time,
                )

            # 2. Create worktree
            self._update_status(slot, issue, "Worktree", "creating")
            log("DEBUG", f"  Creating worktree for #{issue}")
            worktree = self.worktree_manager.create(issue, title)
            log("DEBUG", f"  Worktree created: {worktree}")
            # Update state with synchronization, then save (save() handles its own locking)
            log("DEBUG", f"  Updating state for #{issue}")
            with self.state._save_lock:
                self.state.in_progress[issue] = str(worktree)
            self.state.save(self.state_file)
            log("DEBUG", f"  State saved for #{issue}")

            # 3. Fetch and rebase
            self._update_status(slot, issue, "Git", "fetch & rebase")
            log("DEBUG", f"  Running safe git fetch for #{issue}")
            success, error_msg = safe_git_fetch(self.repo_root, "main")
            if not success:
                raise RuntimeError(f"git fetch failed: {error_msg}")

            cp = run(["git", "rebase", "origin/main"], cwd=worktree)
            if cp.returncode != 0:
                # Try merge instead
                run(["git", "rebase", "--abort"], cwd=worktree)
                cp = run(
                    ["git", "merge", "origin/main", "-m", "Merge origin/main"],
                    cwd=worktree,
                )
                if cp.returncode != 0:
                    raise RuntimeError(f"git rebase/merge failed: {cp.stderr}")

            # 4. Run implementation with retries
            implementation_prompt = f"""You are a code implementation agent. Your task is to implement GitHub issue #{issue}.

## Issue Title
{title}

## Working Directory
Your current working directory is: {worktree}
All file operations should be relative to this directory.

## Branch
You are on branch: {worktree.name}

## Implementation Plan
{plan}

## Instructions
1. Read and understand the plan above
2. Create or modify the necessary files to implement the plan
3. Use the Write tool to create new files and Edit tool to modify existing files
4. Write clean, well-documented Mojo code following the existing patterns in src/projectodyssey/core/
5. Run tests with `pixi run mojo test` if applicable
6. COMMIT your changes before finishing using: git add -A && git commit -m "feat: Implement #{issue}"
7. DO NOT just output text - you MUST make actual file changes AND commit them

## Critical Rules
- You MUST create or modify files - do not just describe what to do
- You MUST commit your changes before finishing - branches with uncommitted changes are not acceptable
- Use absolute paths starting with {worktree}
- Follow Mojo v0.26.1+ syntax (out self for constructors, mut self for mutating methods)
- Check existing code patterns before implementing
"""

            success = False
            output = ""
            for attempt in range(1, MAX_RETRIES + 1):
                self._update_status(slot, issue, "Implement", f"attempt {attempt}")
                success, output = self._spawn_claude_agent(worktree, implementation_prompt, slot, issue)
                if success:
                    break
                if attempt < MAX_RETRIES:
                    self._update_status(slot, issue, "Retry", f"in {2**attempt}s")
                    time.sleep(2**attempt)

            if not success:
                raise RuntimeError(f"Implementation failed after {MAX_RETRIES} attempts: {output[-200:]}")

            # 5. Check for changes (uncommitted OR already committed by Claude)
            self._update_status(slot, issue, "Git", "checking changes")
            cp = run(["git", "status", "--porcelain"], cwd=worktree)
            has_uncommitted = bool(cp.stdout.strip())

            # Also check if Claude already committed (compare to origin/main)
            cp = run(["git", "log", "--oneline", "origin/main..HEAD"], cwd=worktree)
            has_new_commits = bool(cp.stdout.strip())

            # NOTE: PR creation removed - no longer checking for existing PRs

            if not has_uncommitted and not has_new_commits:
                log("WARN", f"  No changes made for #{issue}")

                # Release slot before cleanup, then mark as cleaning up
                log("DEBUG", f"  Releasing slot {slot} for #{issue} (no changes)")
                if self.status_tracker:
                    worker_id = slot  # Save for cleanup tracking
                    self.status_tracker.release_slot(slot)
                    self.status_tracker.set_cleanup(issue, worker_id)
                    slot = -1

                self.worktree_manager.remove(issue)
                # Update state with synchronization, then save (save() handles its own locking)
                with self.state._save_lock:
                    del self.state.in_progress[issue]
                self.state.save(self.state_file)
                return WorkerResult(issue, "skipped", None, "No changes made", time.time() - start_time)

            # 6. Get summary
            summary = self._get_summary(worktree, slot, issue)

            # 7. Post summary to issue
            self._update_status(slot, issue, "Summary", "posting")
            summary_comment = f"## Implementation Summary\n\n{summary}\n\n---\n*Automated by implement_issues.py*"
            proc = subprocess.Popen(
                ["gh", "issue", "comment", str(issue), "--body-file", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                proc.communicate(input=summary_comment, timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                log("WARN", f"Timeout posting summary to issue #{issue}")

            # 8. Commit (if Claude didn't already)
            if has_uncommitted:
                self._update_status(slot, issue, "Git", "committing")
                # Call Claude to analyze issue, plan, and changes to create a proper commit
                commit_prompt = f"""You need to commit the changes for GitHub issue #{issue}.

## Issue Title
{title}

## Working Directory
{worktree}

## Implementation Plan
{plan}

## Your Task
1. Read the GitHub issue to understand the full context: `gh issue view {issue}`
2. Run `git diff` to see what files were changed and how
3. Run `git status` to see all modified/added/deleted files
4. Based on the issue, plan, and actual changes, create a meaningful commit message
5. Commit using: git add -A && git commit -m "your message"

## Commit Message Requirements
- Start with feat:, fix:, docs:, refactor:, or test: as appropriate for what was done
- First line: brief summary (50 chars or less)
- Body: explain what was implemented and why, referencing the plan
- Footer: include "Closes #{issue}"

Example format:
```
feat(core): implement variance and std operations

Add variance and standard deviation functions with forward and
backward passes following the pure functional pattern.

- variance(): compute variance along specified axis
- std(): compute standard deviation
- Backward passes for gradient computation

Closes #{issue}
```

DO NOT describe what you're doing - just run the commands to commit.
"""
                self._spawn_claude_agent(worktree, commit_prompt, slot, issue)

                # Verify commit happened
                cp = run(["git", "status", "--porcelain"], cwd=worktree)
                if cp.stdout.strip():
                    # Claude still didn't commit - force commit ourselves
                    log(
                        "WARN",
                        f"  Claude failed to commit, forcing commit for #{issue}",
                    )
                    run(["git", "add", "-A"], cwd=worktree)
                    commit_msg = f"feat: Implement #{issue}\n\nCloses #{issue}\n\n{summary}"
                    run(["git", "commit", "-m", commit_msg], cwd=worktree)

            # 9. Push
            self._update_status(slot, issue, "Git", "pushing")
            branch = worktree.name
            cp = run(["git", "push", "-u", "origin", branch], cwd=worktree)
            if cp.returncode != 0:
                raise RuntimeError(f"git push failed: {cp.stderr}")

            # Release slot BEFORE cleanup so new work can start immediately
            # Cleanup can happen in parallel with other work
            log("DEBUG", f"  Releasing slot {slot} for #{issue} before cleanup")
            if self.status_tracker:
                worker_id = slot  # Save for cleanup tracking
                self.status_tracker.release_slot(slot)
                self.status_tracker.set_cleanup(issue, worker_id)
                slot = -1  # Mark as released to prevent double-release in main loop

            # 10. Cleanup worktree and move to next issue
            # NOTE: PR creation removed - commits are pushed to branches only
            log("DEBUG", f"  Cleaning up worktree for #{issue}")
            self.worktree_manager.remove(issue)
            # Update state with synchronization, then save (save() handles its own locking)
            with self.state._save_lock:
                del self.state.in_progress[issue]
            self.state.save(self.state_file)

            duration = time.time() - start_time
            log("INFO", f"  Committed #{issue} to branch {branch} in {duration:.0f}s")
            self._post_issue_update(issue, f"✅ Committed changes to branch: {branch}")
            return WorkerResult(issue, "completed", None, None, duration)

        except (KeyboardInterrupt, SystemExit):
            # Clean shutdown - preserve state and re-raise
            # Release slot before state updates
            if slot != -1 and self.status_tracker:
                log("DEBUG", f"  Releasing slot {slot} for #{issue} (interrupted)")
                self.status_tracker.release_slot(slot)
                slot = -1

            if issue in self.state.in_progress:
                # Update state with synchronization, then save (save() handles its own locking)
                with self.state._save_lock:
                    self.state.paused_issues[issue] = PausedIssue(
                        worktree=self.state.in_progress[issue],
                        pr=None,
                        reason="Interrupted by user",
                    )
                    del self.state.in_progress[issue]
                self.state.save(self.state_file)
            raise
        except RuntimeError as e:
            # Expected failures (git push failed, etc.)
            log("ERROR", f"  Issue #{issue} failed: {e}")
            error_msg = str(e)[:200]
            self._post_issue_update(issue, f"❌ Implementation failed:\n\n```\n{error_msg}\n```")

            # Release slot before state updates
            if slot != -1 and self.status_tracker:
                log("DEBUG", f"  Releasing slot {slot} for #{issue} (error)")
                self.status_tracker.release_slot(slot)
                slot = -1

            if issue in self.state.in_progress:
                # Update state with synchronization, then save (save() handles its own locking)
                with self.state._save_lock:
                    self.state.paused_issues[issue] = PausedIssue(
                        worktree=self.state.in_progress[issue],
                        pr=None,
                        reason=str(e)[:100],
                    )
                    del self.state.in_progress[issue]
                self.state.save(self.state_file)

            duration = time.time() - start_time
            return WorkerResult(issue, "paused", None, str(e), duration)
        except Exception as e:
            # Unexpected failures - log with traceback
            import traceback

            log("ERROR", f"  Issue #{issue} failed unexpectedly: {e}\n{traceback.format_exc()}")

            error_msg = str(e)[:200]
            self._post_issue_update(issue, f"❌ Implementation failed:\n\n```\n{error_msg}\n```")

            # Release slot before state updates
            if slot != -1 and self.status_tracker:
                log("DEBUG", f"  Releasing slot {slot} for #{issue} (unexpected error)")
                self.status_tracker.release_slot(slot)
                slot = -1

            if issue in self.state.in_progress:
                # Update state with synchronization, then save (save() handles its own locking)
                with self.state._save_lock:
                    self.state.paused_issues[issue] = PausedIssue(
                        worktree=self.state.in_progress[issue],
                        pr=None,
                        reason=str(e)[:100],
                    )
                    del self.state.in_progress[issue]
                self.state.save(self.state_file)

            duration = time.time() - start_time
            return WorkerResult(issue, "paused", None, str(e), duration)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def print_analysis(resolver: DependencyResolver, state: ImplementationState) -> None:
    """Print dependency analysis."""
    priority_labels = {"P0": "Foundation", "P1": "Important", "P2": "Nice-to-have"}

    # Group by priority
    by_priority: dict[str, list[IssueInfo]] = {"P0": [], "P1": [], "P2": []}
    for info in state.issues.values():
        by_priority[info.priority].append(info)

    # Sort each group by issue number
    for priority in by_priority:
        by_priority[priority].sort(key=lambda x: x.number)

    print("\nDependency Analysis:")

    blocked_external = resolver.get_blocked_by_external()
    ready = set(resolver.get_ready_issues())

    for priority in ["P0", "P1", "P2"]:
        issues = by_priority[priority]
        if not issues:
            continue

        print(f"\n  {priority} ({priority_labels[priority]}): {len(issues)} issues")

        for info in issues[:10]:  # Show first 10 per priority
            status_str = ""
            if info.number in state.completed_issues:
                status_str = " [DONE]"
            elif info.number in blocked_external:
                ext_deps = blocked_external[info.number]
                status_str = f" - Blocked by: {', '.join(f'#{d}' for d in ext_deps)} (external)"
            elif info.number in ready:
                status_str = " - Ready"
            elif info.depends_on:
                status_str = f" - Depends: {', '.join(f'#{d}' for d in info.depends_on)}"

            print(f"    #{info.number}{status_str}")

        if len(issues) > 10:
            print(f"    ... ({len(issues) - 10} more)")

    completed_count = len(state.completed_issues)
    ready_count = len(ready)
    blocked_count = len(blocked_external)
    paused_count = len(state.paused_issues)

    print("\nSummary:")
    print(f"  Completed: {completed_count}")
    print(f"  Ready to start: {ready_count}")
    print(f"  Blocked by external deps: {blocked_count}")
    print(f"  Paused (CI failed): {paused_count}")


def print_paused_summary(state: ImplementationState) -> None:
    """Print summary of paused issues."""
    if not state.paused_issues:
        return

    print("\n==========================================")
    print("  Paused Issues (require manual intervention)")
    print("==========================================")

    for issue_num, paused in sorted(state.paused_issues.items()):
        info = state.issues.get(issue_num)
        title = info.title if info else f"Issue #{issue_num}"
        print(f"\n  #{issue_num} - {title}")
        if paused.pr:
            print(f"    PR: #{paused.pr}")
        print(f"    Worktree: {paused.worktree}")
        print(f"    Reason: {paused.reason}")

    issue_list = ",".join(str(n) for n in state.paused_issues.keys())
    print("\nTo retry paused issues after fixing:")
    print(f"  python scripts/implement_issues.py --epic {state.epic_number} --resume --issues {issue_list}")


def main() -> int:
    """CLI entry point for the issue implementer."""
    p = argparse.ArgumentParser(
        description="Orchestrate parallel implementation of GitHub issues using worktrees.\n"
        "Requires Mojo v0.26.1+. Language reference: https://docs.modular.com/mojo/manual/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --epic 2784 --analyze              Show dependency graph
  %(prog)s --epic 2784 --dry-run              Preview what would be done
  %(prog)s --epic 2784 --priority P0          Run P0 issues only
  %(prog)s --epic 2784 --parallel             Run with 4 parallel workers
  %(prog)s --epic 2784 --parallel 2           Run with 2 parallel workers
  %(prog)s --epic 2784 --resume               Resume from previous run
  %(prog)s --epic 2784 --issues 2719,2720     Run specific issues only
  %(prog)s --epic 2784 --cleanup              Cleanup stale worktrees
""",
    )
    p.add_argument("--epic", type=int, required=True, help="Epic issue number to parse")
    p.add_argument(
        "--priority",
        choices=["P0", "P1", "P2"],
        help="Only process issues with this priority",
    )
    p.add_argument("--issues", metavar="N,M,...", help="Only process specific issue numbers")
    p.add_argument(
        "--parallel",
        type=int,
        nargs="?",
        const=MAX_PARALLEL_DEFAULT,
        default=None,
        metavar="N",
        help=f"Run in parallel (default: {MAX_PARALLEL_DEFAULT} workers if N not specified)",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")
    p.add_argument("--analyze", action="store_true", help="Just show dependency graph")
    p.add_argument("--export-graph", metavar="OUTPUT.dot", help="Export dependency graph to Graphviz DOT file")
    p.add_argument("--resume", action="store_true", help="Resume from previous run")
    p.add_argument(
        "--timeout",
        type=int,
        default=ISSUE_TIMEOUT,
        metavar="SEC",
        help=f"Timeout per issue (default: {ISSUE_TIMEOUT})",
    )
    p.add_argument("--cleanup", action="store_true", help="Cleanup stale worktrees and exit")
    p.add_argument(
        "--rollback",
        type=int,
        metavar="ISSUE",
        help="Rollback implementation of specific issue (delete branch, remove from state)",
    )
    p.add_argument("--state-dir", type=pathlib.Path, help="Directory to persist state")
    p.add_argument(
        "--api-delay",
        type=float,
        default=API_DELAY_DEFAULT,
        metavar="SEC",
        help=f"Delay between GitHub API calls (default: {API_DELAY_DEFAULT}s)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose INFO output",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG output (requires --verbose)",
    )
    p.add_argument("--health-check", action="store_true", help="Check dependency status and exit")

    args = p.parse_args()

    # Set verbose/debug mode
    set_verbose(args.verbose, args.debug)

    # Health check mode - run and exit
    if args.health_check:
        exit_code = health_check()
        return exit_code

    # Rollback mode - run and exit
    if args.rollback:
        # Get repo root and state dir
        repo_root = pathlib.Path.cwd()
        state_dir = args.state_dir or (repo_root / ".state")
        exit_code = rollback_issue(args.rollback, state_dir, repo_root)
        return exit_code

    # Verify all required dependencies are available
    check_dependencies()

    # Parse issue numbers if provided
    issues: list[int] | None = None
    if args.issues:
        issues = [int(n.strip()) for n in args.issues.split(",")]

    # If --parallel not specified, run sequentially (1 worker)
    # If --parallel specified without value, use default (4)
    # If --parallel N specified, use N
    parallel_workers = args.parallel if args.parallel is not None else 1

    opts = Options(
        epic=args.epic,
        priority=args.priority,
        issues=issues,
        parallel=parallel_workers,
        dry_run=args.dry_run,
        analyze=args.analyze,
        export_graph=args.export_graph,
        resume=args.resume,
        timeout=args.timeout,
        cleanup=args.cleanup,
        state_dir=args.state_dir,
        api_delay=args.api_delay,
        verbose=args.verbose,
    )

    repo_root = get_repo_root()
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix="implement-issues-"))
    tempdir.chmod(0o700)

    # Cleanup handler
    def cleanup_handler() -> None:
        if tempdir.exists() and not opts.state_dir:
            shutil.rmtree(tempdir, ignore_errors=True)

    atexit.register(cleanup_handler)

    # Track signal count for force-exit on repeated Ctrl+C
    signal_count = [0]

    def signal_handler(signum: int, _frame: Optional["FrameType"]) -> None:
        import traceback

        signal_count[0] += 1

        # Log detailed information about the signal
        signal_name = (
            "SIGINT" if signum == signal.SIGINT else "SIGTERM" if signum == signal.SIGTERM else f"Signal {signum}"
        )
        stack_trace = "".join(traceback.format_stack(_frame))

        if signal_count[0] == 1:
            # First signal: request graceful shutdown
            log("WARN", f"Shutdown signal received: {signal_name} (signal {signum})")
            log("DEBUG", f"Signal stack trace:\n{stack_trace}")
            log("INFO", "Press Ctrl+C again to force exit")
            set_shutdown_requested()
        elif signal_count[0] == 2:
            # Second signal: warn about data loss
            log("WARN", "Second shutdown signal received!")
            log("WARN", "Press Ctrl+C one more time to force exit (may lose data)")
        else:
            # Third signal: force exit immediately
            log("ERROR", "Force exit requested, terminating immediately...")
            cleanup_handler()
            sys.exit(128 + signum)

    log("DEBUG", "Registering signal handlers for SIGINT and SIGTERM")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    log("DEBUG", f"Signal handlers registered. Current shutdown_flag={is_shutdown_requested()}")

    # Initialize managers
    worktree_manager = WorktreeManager(repo_root)

    # Handle cleanup mode
    if opts.cleanup:
        log("INFO", "Cleaning up stale worktrees...")
        count = worktree_manager.cleanup_stale()
        log("INFO", f"Removed {count} stale worktrees")
        return 0

    # Load or create state
    state_dir = opts.state_dir or tempdir
    state_file = state_dir / "implementation_state.json"

    if opts.resume and state_file.exists():
        log("INFO", f"Resuming from state file: {state_file}")
        state = ImplementationState.load(state_file)
        if state.epic_number != opts.epic:
            log("WARN", f"State file is for epic #{state.epic_number}, not #{opts.epic}")
            state = ImplementationState(epic_number=opts.epic)

        # Clean up stale in_progress issues from interrupted execution
        # Move them to paused_issues so they can be retried if needed
        if state.in_progress:
            log("INFO", f"Found {len(state.in_progress)} issues in stale in_progress state, moving to paused")
            for issue_num, worktree_path in state.in_progress.items():
                if issue_num not in state.paused_issues:
                    state.paused_issues[issue_num] = PausedIssue(
                        worktree=worktree_path,
                        pr=None,
                        reason="Interrupted during previous run",
                    )
            state.in_progress.clear()
            state.save(state_file)
    else:
        state = ImplementationState(epic_number=opts.epic)
        state.started_at = dt.datetime.now().isoformat()

    # Parse epic and build issues dict
    if not state.issues:
        # Create a temporary implementer just for parsing
        temp_implementer = IssueImplementer(repo_root, tempdir, opts, state, None, worktree_manager, None)
        state.issues = temp_implementer.parse_epic(opts.epic)

        # Mark already-completed issues
        for issue_num, info in state.issues.items():
            if info.status == "completed":
                state.completed_issues.add(issue_num)

        state.save(state_file)

    # Filter by priority if specified
    if opts.priority:
        filtered = {k: v for k, v in state.issues.items() if v.priority == opts.priority}
        log("INFO", f"Filtered to {len(filtered)} {opts.priority} issues")
        state.issues = filtered
        # Also filter completed/paused to match filtered issues
        filtered_issue_nums = set(state.issues.keys())
        state.completed_issues = state.completed_issues & filtered_issue_nums
        state.paused_issues = {k: v for k, v in state.paused_issues.items() if k in filtered_issue_nums}

    # Filter by specific issues if specified
    if opts.issues:
        filtered = {k: v for k, v in state.issues.items() if k in opts.issues}
        log("INFO", f"Filtered to {len(filtered)} specified issues")
        state.issues = filtered
        # Also filter completed/paused to match filtered issues
        filtered_issue_nums = set(state.issues.keys())
        state.completed_issues = state.completed_issues & filtered_issue_nums
        state.paused_issues = {k: v for k, v in state.paused_issues.items() if k in filtered_issue_nums}

    # Create resolver
    resolver = DependencyResolver(state.issues)
    resolver.initialize_from_state(state)

    # Prefetch external issue states to avoid individual API calls
    all_external = set()
    for info in state.issues.values():
        all_external.update(info.depends_on - set(state.issues.keys()))
    if all_external:
        log("INFO", f"Prefetching states for {len(all_external)} external dependencies...")
        prefetch_issue_states(list(all_external))
        log("INFO", "Prefetch complete")

    # Handle analyze mode
    if opts.analyze:
        log("DEBUG", "Entering analyze mode")
        print_analysis(resolver, state)
        return 0

    # Handle export-graph mode
    if opts.export_graph:
        log("DEBUG", "Entering export-graph mode")
        exit_code = export_dependency_graph(state.issues, opts.export_graph)
        return exit_code

    # Calculate actual worker count (don't spawn more threads than issues)
    log("DEBUG", "Calculating worker count...")
    pending_count = len(state.issues) - len(state.completed_issues) - len(state.paused_issues)
    actual_workers = min(opts.parallel, max(1, pending_count))
    log("DEBUG", f"Worker count: {actual_workers} (pending: {pending_count})")

    # Create status tracker with correct worker count
    log("DEBUG", "Creating status tracker...")
    status_tracker = StatusTracker(actual_workers)
    log("DEBUG", "Status tracker created")

    # Create implementer
    log("DEBUG", "Creating implementer...")
    implementer = IssueImplementer(repo_root, tempdir, opts, state, resolver, worktree_manager, status_tracker)
    log("DEBUG", "Implementer created")

    log("DEBUG", "Printing banner...")
    print()
    print("==========================================")
    print("  ML Odyssey Issue Implementer")
    print(f"  Epic: #{opts.epic}")
    print(f"  Total issues: {len(state.issues)}")
    print(f"  Pending: {pending_count}")
    mode = "Sequential" if actual_workers == 1 else f"Parallel ({actual_workers} workers)"
    if opts.dry_run:
        mode += " (DRY RUN)"
    print(f"  Mode: {mode}")
    print("==========================================")
    print()
    log("DEBUG", "Banner printed")

    if opts.dry_run or opts.analyze:
        log("DEBUG", "Dry-run or analyze mode")
        print_analysis(resolver, state)
        ready = resolver.get_ready_issues()
        if opts.dry_run:
            # Fetch titles only for issues we'll display (first 10)
            implementer.fetch_titles_for_issues(ready[:10])
            print(f"\n[DRY RUN] Would start with {len(ready)} ready issues:")
            for issue_num in ready[:10]:
                issue_info = state.issues.get(issue_num)
                print(f"  #{issue_num} - {issue_info.title if issue_info else 'Unknown'}")
            if len(ready) > 10:
                print(f"  ... ({len(ready) - 10} more)")
        return 0

    # Create log files for persistent per-thread logging
    log("DEBUG", "Creating thread log manager...")
    log_dir = repo_root / "logs"
    log_prefix = f"implement-epic-{opts.epic}-{time.strftime('%Y%m%d-%H%M%S')}"
    merged_log_file = log_dir / f"{log_prefix}.log"  # For merged output on success
    log("DEBUG", f"Log prefix: {log_prefix}")

    # Create thread log manager for per-thread logging
    thread_log_manager = ThreadLogManager(log_dir, log_prefix, actual_workers)
    set_thread_log_manager(thread_log_manager)
    log("DEBUG", "Thread log manager initialized")

    # Set main thread's worker slot
    set_worker_slot(ThreadLogManager.MAIN_THREAD_ID)
    log("DEBUG", "Main thread worker slot set")

    # Results container (shared with run_implementation)
    results: list[WorkerResult] = []
    status_tracker.set_total(len(state.issues))

    def run_implementation() -> int:
        """Run the main implementation loop. Called by CursesUI in background thread."""
        nonlocal results

        # Create executor with daemon threads so they don't block process exit
        # This allows clean shutdown even if workers are stuck in subprocess calls
        executor = DaemonThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix="Worker")
        log("DEBUG", f"Created DaemonThreadPoolExecutor with {actual_workers} daemon worker threads")

        try:
            futures: dict = {}
            active_count = 0
            stall_count = 0  # Track iterations without progress

            while True:
                # Check shutdown flag FIRST - exit gracefully if requested
                if is_shutdown_requested():
                    log("INFO", f"Shutdown requested, breaking main loop (active futures: {len(futures)})")
                    # Cancel any pending futures
                    if futures:
                        log("INFO", f"Cancelling {len(futures)} pending futures")
                        for future in list(futures.keys()):
                            if not future.done():
                                cancelled = future.cancel()
                                log("DEBUG", f"  Cancelled future for #{futures.get(future)}: {cancelled}")
                    break

                status_tracker.update_main("Processing", f"{len(results)} done")

                # Get ready issues (dependencies satisfied)
                ready = resolver.get_ready_issues()

                # NOTE: PR creation removed - no longer checking for existing PRs with CI failures

                # Filter out issues already in progress (in futures)
                ready = [n for n in ready if n not in futures.values()]

                # Spawn new tasks up to max_workers only
                # We have exactly max_workers slots, don't try to exceed that
                spawned_this_iteration = 0
                available_slots = actual_workers - active_count

                log(
                    "DEBUG",
                    f"Loop: ready={len(ready)} futures={len(futures)} active={active_count} avail_slots={available_slots} done={len(results)}",
                )

                # Only spawn if we have both ready issues AND available slots
                if available_slots > 0 and ready:
                    for issue_num in ready[:available_slots]:
                        # Try to acquire slot with timeout
                        try:
                            slot = status_tracker.acquire_slot(issue_num, timeout=5.0)
                        except RuntimeError as e:
                            # Slot acquisition failed - this shouldn't happen with our calculation
                            log("ERROR", f"Failed to acquire slot for #{issue_num}: {e}")
                            # Don't mark as in_progress if we couldn't get a slot
                            continue

                        resolver.mark_in_progress(issue_num)
                        future = executor.submit(implementer.implement_issue, issue_num, slot)
                        futures[future] = issue_num
                        active_count += 1
                        spawned_this_iteration += 1
                        status_tracker.update_main("Spawning", f"#{issue_num}")

                # Check termination conditions
                if not futures and not ready:
                    # Nothing running and nothing ready - we're done or blocked
                    log("INFO", "All processable issues completed or blocked")
                    break

                if not futures and ready and spawned_this_iteration == 0:
                    # Ready issues exist but we couldn't spawn any - something's wrong
                    stall_count += 1
                    if stall_count > 5:
                        log(
                            "WARN",
                            f"Stalled with {len(ready)} ready issues but cannot spawn. Exiting.",
                        )
                        break
                    time.sleep(1)
                    continue
                else:
                    stall_count = 0

                # Wait for at least one completion
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    time.sleep(1.0)  # Poll interval
                    continue

                for future in done_futures:
                    issue_num = futures.pop(future)
                    active_count -= 1

                    try:
                        result = future.result()
                    except Exception as e:
                        log(
                            "ERROR",
                            f"Future for #{issue_num} raised exception: {e}",
                        )
                        result = WorkerResult(issue_num, "paused", None, str(e), 0)

                    results.append(result)
                    status_tracker.increment_completed()

                    # Clear cleanup status now that future is complete
                    status_tracker.clear_cleanup(issue_num)

                    if result.status == "completed":
                        resolver.mark_completed(issue_num)
                        log("DEBUG", f"Main loop: marked #{issue_num} as completed")
                    else:
                        resolver.mark_paused(issue_num)
                        log(
                            "DEBUG",
                            f"Main loop: marked #{issue_num} as paused ({result.status})",
                        )

                    # NOTE: Slot already released by worker - no release needed here
                    # Workers release at lines 2856, 2955, 2978, 3001, 3029

        except (KeyboardInterrupt, SystemExit):
            log("WARN", "Caught interrupt exception, shutting down executor")
            log("DEBUG", f"Executor state before shutdown: pending futures={len(futures)}")
            executor.shutdown(wait=False, cancel_futures=True)
            log("DEBUG", "Executor shutdown (from except) completed")
            raise
        except Exception as e:
            log("ERROR", f"Unexpected exception in main loop: {e}")
            log("DEBUG", f"Executor state before shutdown: pending futures={len(futures)}")
            executor.shutdown(wait=False, cancel_futures=True)
            log("DEBUG", "Executor shutdown (from except) completed")
            raise
        finally:
            # Ensure executor is always shutdown
            log("DEBUG", "Entering finally block")
            log("DEBUG", f"Executor state: pending futures={len(futures)}")

            # Count active futures for logging
            active_futures = sum(1 for f in futures.keys() if not f.done())
            if active_futures > 0:
                log("INFO", f"Shutting down executor (abandoning {active_futures} active daemon threads)...")
            else:
                log("INFO", "Shutting down executor...")

            executor.shutdown(wait=False, cancel_futures=False)
            log("INFO", "Executor shutdown completed")

            # Save state even if interrupted
            log("INFO", "Saving state...")
            state.save(state_file)
            log("INFO", "State saved")

        exit_code = 1 if any(r.status == "error" for r in results) else 0
        log("DEBUG", f"run_implementation returning with exit_code={exit_code}")
        return exit_code

    # Run with curses UI if TTY available, otherwise run directly
    stdout_tty = sys.stdout.isatty()
    stdin_tty = sys.stdin.isatty()
    log("DEBUG", f"TTY detection: stdout={stdout_tty}, stdin={stdin_tty}")

    if stdout_tty and stdin_tty:
        log("DEBUG", "Starting curses UI")
        curses_ui = CursesUI(status_tracker, thread_log_manager, actual_workers)
        exit_code = curses_ui.run(run_implementation)
        log("DEBUG", f"Curses UI completed with exit_code={exit_code}")
    else:
        # Non-TTY mode: run directly without curses
        log("INFO", f"Non-TTY mode: running without curses UI (stdout_tty={stdout_tty}, stdin_tty={stdin_tty})")
        exit_code = run_implementation()
        log("DEBUG", f"Non-TTY mode completed with exit_code={exit_code}")

    log("DEBUG", "Preparing to print summary")
    # Print summary
    completed = sum(1 for r in results if r.status == "completed")
    paused = sum(1 for r in results if r.status == "paused")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")

    # Merge logs on success (no errors), otherwise keep individual logs for debugging
    merged_logs = False
    if errors == 0 and len(results) > 0:
        log("INFO", f"All tasks completed successfully, merging logs into {merged_log_file}")
        thread_log_manager.merge_logs_on_success(merged_log_file)
        merged_logs = True

    # Close all thread log buffers after execution
    thread_log_manager.close_all()
    set_thread_log_manager(None)

    print()
    print("==========================================")
    print("  Summary")
    print(f"  Completed: {completed}")
    print(f"  Paused:    {paused}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(results)}")
    print()

    # Print log file locations
    if merged_logs:
        print(f"  Log file: {merged_log_file} (merged from all threads)")
    else:
        print("  Log files:")
        print(f"    Main:     {log_dir / f'{log_prefix}-main.log'}")
        for worker_id in range(actual_workers):
            print(f"    Worker {worker_id}: {log_dir / f'{log_prefix}-worker-{worker_id}.log'}")

    if opts.state_dir:
        print(f"  State file: {state_file}")
    print("==========================================")

    # Print paused issues summary
    print_paused_summary(state)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
