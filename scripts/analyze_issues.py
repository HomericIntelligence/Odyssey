#!/usr/bin/env python3
"""Analyze GitHub issues using Claude Code CLI to determine dependencies.

This is a Python script that uses Claude Code to intelligently analyze
GitHub issues and determine their dependencies based on content understanding.

Design goals (following plan_issues.py patterns):
- No shell injection surface
- Deterministic, debuggable execution
- State tracking for resumability
- Parallel processing with live status display
- Structured logging and error handling
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime as dt
import io
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import timezone
from typing import TYPE_CHECKING, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[import,no-redef]

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_ISSUES_FETCH = 500
MAX_RETRIES = 3
CLAUDE_TIMEOUT = 300  # 5 minutes per batch
MAX_BODY_SIZE = 1_048_576  # 1MB limit for issue body

# Status display refresh interval
STATUS_REFRESH_INTERVAL = 0.1

# Time constants for AM/PM parsing
NOON_HOUR = 12
MIDNIGHT_HOUR = 0

# Allowed timezones for rate limit parsing
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

RATE_LIMIT_RE = re.compile(
    r"Limit reached.*resets\s+(?P<time>[0-9:apm]+)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)

# Dependency comment header (used to detect existing comments)
DEPENDENCY_HEADER = "## Dependencies (AI-Analyzed)"

# Error messages
ERR_GITHUB_API_FAILED = "GitHub API failed after {retries} attempts: {error}"
ERR_CLAUDE_TIMEOUT = "Claude timed out after {timeout}s"
ERR_CLAUDE_RETRIES = "Claude retries exceeded"
ERR_EMPTY_ANALYSIS = "Empty analysis result"

# JSON schema for Claude's structured output
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Issue numbers this issue depends on (must be done first)",
                    },
                    "blocks": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Issue numbers that depend on this issue",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2"],
                        "description": "P0=critical/foundation, P1=important, P2=nice-to-have",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of dependency analysis",
                    },
                },
                "required": [
                    "issue_number",
                    "depends_on",
                    "blocks",
                    "priority",
                    "reasoning",
                ],
            },
        }
    },
    "required": ["analyses"],
}

ANALYSIS_PROMPT_TEMPLATE = """You are analyzing GitHub issues to determine their dependencies.

## ALL ISSUES (for reference - use these numbers when specifying dependencies):
{reference_list}

## YOUR BATCH TO ANALYZE:
{batch_content}

## INSTRUCTIONS:
For EACH issue in your batch, analyze its content and determine:
1. **depends_on**: What other issues (from the reference list) must be completed BEFORE this issue can start?
2. **blocks**: What other issues cannot start until THIS issue is completed?
3. **priority**:
   - P0: Foundation/core infrastructure (ExTensor, basic ops, core utilities)
   - P1: Important features that build on P0 (training, autograd, models)
   - P2: Nice-to-have improvements (docs, examples, optimizations)
4. **reasoning**: Brief explanation (1-2 sentences)

## DEPENDENCY RULES:
- Core/foundation issues (ExTensor, basic tensor ops) should be P0 and have few dependencies
- Training issues depend on core tensor operations
- Test issues depend on their corresponding implementation issues
- Bug fixes depend on the feature they're fixing existing
- Documentation depends on the feature being documented existing
- Higher-level features (models, training loops) depend on lower-level components

## EXAMPLE OUTPUT:
For an issue about "Implement backward pass for conv2d":
- depends_on: [issue for forward conv2d, issue for gradient system]
- blocks: [issues for training loops that use conv2d]
- priority: P1 (builds on P0 foundation)
- reasoning: "Conv2d backward requires forward pass and gradient infrastructure"

Analyze each issue in your batch and output the structured JSON."""


# ---------------------------------------------------------------------
# Secure file helpers
# ---------------------------------------------------------------------


def write_secure(path: pathlib.Path, content: str) -> None:
    """Write content to file with secure permissions (owner-only read/write)."""
    path.write_text(content)
    path.chmod(0o600)


# ---------------------------------------------------------------------
# Parallel status tracking (from plan_issues.py)
# ---------------------------------------------------------------------


class StatusTracker:
    """Thread-safe status tracker for parallel processing with live display."""

    MAIN_THREAD = -1

    def __init__(self, max_workers: int) -> None:
        """Initialize the status tracker."""
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._slots: dict[int, tuple[int, str, str, float]] = {}
        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None
        self._lines_printed = 0
        self._update_event = threading.Event()
        self._completed_count = 0
        self._total_count = 0
        self._is_tty = sys.stdout.isatty()

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

    def acquire_slot(self, item_id: int) -> int:
        """Acquire a display slot for an item. Returns slot number."""
        with self._lock:
            for slot in range(self._max_workers):
                if slot not in self._slots:
                    self._slots[slot] = (item_id, "Starting", "", time.time())
                    self._update_event.set()
                    return slot
            slot = item_id % self._max_workers
            self._slots[slot] = (item_id, "Starting", "", time.time())
            self._update_event.set()
            return slot

    def update(self, slot: int, item_id: int, stage: str, info: str = "") -> None:
        """Update the status for a slot."""
        with self._lock:
            self._slots[slot] = (item_id, stage, info, time.time())
            self._update_event.set()

    def release_slot(self, slot: int) -> None:
        """Release a slot when done."""
        with self._lock:
            if slot in self._slots:
                del self._slots[slot]
            self._update_event.set()

    def _format_elapsed(self, start_time: float) -> str:
        """Format elapsed time as MM:SS."""
        elapsed = max(0, int(time.time() - start_time))
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _render_status(self) -> list[str]:
        """Render current status as list of lines."""
        with self._lock:
            lines = []

            if self.MAIN_THREAD in self._slots:
                _, stage, info, start_time = self._slots[self.MAIN_THREAD]
                elapsed = self._format_elapsed(start_time)
                progress = f" [{self._completed_count}/{self._total_count}]" if self._total_count > 0 else ""
                info_str = f" - {info}" if info else ""
                lines.append(f"  Main:     [{stage:12}]{progress} ({elapsed}){info_str}")
            else:
                lines.append("  Main:     [Idle        ]")

            for slot in range(self._max_workers):
                if slot in self._slots:
                    item_id, stage, info, start_time = self._slots[slot]
                    elapsed = self._format_elapsed(start_time)
                    info_str = f" - {info}" if info else ""
                    lines.append(f"  Thread {slot}: [{stage:12}] batch_{item_id} ({elapsed}){info_str}")
                else:
                    lines.append(f"  Thread {slot}: [Idle        ]")
            return lines

    def _display_loop(self) -> None:
        """Background thread that refreshes the status display."""
        last_main_stage = ""
        first_render = True

        while not self._stop_event.is_set():
            lines = self._render_status()
            num_lines = len(lines)

            if self._is_tty:
                if not first_render:
                    sys.stdout.write(f"\033[{num_lines}A")
                for line in lines:
                    sys.stdout.write(f"\r{line}\033[K\n")
                sys.stdout.flush()
                first_render = False
                self._lines_printed = num_lines
            else:
                with self._lock:
                    if self.MAIN_THREAD in self._slots:
                        _, stage, info, _ = self._slots[self.MAIN_THREAD]
                        current = f"{stage}:{info}"
                        if current != last_main_stage:
                            progress = f"[{self._completed_count}/{self._total_count}]" if self._total_count > 0 else ""
                            info_str = f" - {info}" if info else ""
                            print(f"  Progress: {stage}{progress}{info_str}")
                            sys.stdout.flush()
                            last_main_stage = current

            self._update_event.wait(STATUS_REFRESH_INTERVAL)
            self._update_event.clear()

    def start_display(self) -> None:
        """Start the background display thread."""
        if self._is_tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        time.sleep(0.05)

    def stop_display(self) -> None:
        """Stop the background display thread."""
        self._stop_event.set()
        self._update_event.set()
        if self._display_thread:
            self._display_thread.join(timeout=1.0)

        if self._is_tty:
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._lines_printed = 0


# ---------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------


def log(level: str, msg: str) -> None:
    """Log a message with timestamp and level prefix."""
    if level == "DEBUG" and not os.environ.get("DEBUG"):
        return
    ts = time.strftime("%H:%M:%S")
    out = sys.stderr if level in {"WARN", "ERROR"} else sys.stdout
    print(f"[{level}] {ts} {msg}", file=out, flush=True)


# ---------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------


def run(cmd: list[str], *, timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    """Run a command and capture output."""
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------
# Rate-limit handling (from plan_issues.py)
# ---------------------------------------------------------------------


def parse_reset_epoch(time_str: str, tz: str) -> int:
    """Parse a rate limit reset time string and return epoch seconds."""
    if tz not in ALLOWED_TIMEZONES:
        tz = "America/Los_Angeles"

    now_utc = dt.datetime.now(timezone.utc)
    today = now_utc.astimezone(ZoneInfo(tz)).date()

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", time_str, re.IGNORECASE)
    if not m:
        return int(time.time()) + 3600

    hour, minute, ampm = m.groups()
    hour = int(hour)
    minute = int(minute or 0)

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < NOON_HOUR:
            hour += NOON_HOUR
        if ampm == "am" and hour == NOON_HOUR:
            hour = MIDNIGHT_HOUR

    local = dt.datetime.combine(
        today,
        dt.time(hour, minute),
        tzinfo=ZoneInfo(tz),
    )

    if local < now_utc.astimezone(ZoneInfo(tz)):
        local += dt.timedelta(days=1)

    return int(local.timestamp())


def detect_rate_limit(text: str) -> int | None:
    """Detect rate limit message in text and return reset epoch if found."""
    m = RATE_LIMIT_RE.search(text)
    if not m:
        return None
    return parse_reset_epoch(m.group("time"), m.group("tz"))


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


# ---------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------


def gh_issue_json(issue: int) -> dict:
    """Fetch issue data from GitHub as JSON."""
    for attempt in range(1, MAX_RETRIES + 1):
        cp = run(["gh", "issue", "view", str(issue), "--json", "title,body,comments"])
        if cp.returncode == 0:
            return json.loads(cp.stdout)

        if attempt < MAX_RETRIES:
            delay = 2**attempt
            log(
                "WARN",
                f"GitHub API error fetching #{issue} (attempt {attempt}/{MAX_RETRIES}): {cp.stderr.strip()}",
            )
            log("INFO", f"Retrying in {delay}s...")
            time.sleep(delay)
        else:
            msg = ERR_GITHUB_API_FAILED.format(retries=MAX_RETRIES, error=cp.stderr.strip())
            raise RuntimeError(msg)

    raise RuntimeError("Unexpected error in gh_issue_json")


def delete_dependency_comments(issue: int, comments: list[dict]) -> int:
    """Delete existing dependency comments from an issue."""
    deleted = 0
    for c in comments:
        comment_body = c.get("body") or ""
        if DEPENDENCY_HEADER not in comment_body:
            continue

        url = c.get("url") or ""
        comment_id = url.split("-")[-1] if "-" in url else ""
        if not comment_id.isdigit():
            log("WARN", f"Could not extract comment ID from URL: {url}")
            continue

        log("INFO", f"Deleting existing dependency comment {comment_id} from #{issue}")
        cp = run(
            [
                "gh",
                "api",
                "-X",
                "DELETE",
                f"/repos/{{owner}}/{{repo}}/issues/comments/{comment_id}",
            ]
        )
        if cp.returncode == 0:
            deleted += 1
        else:
            log("WARN", f"Failed to delete comment {comment_id}: {cp.stderr.strip()}")

    return deleted


def has_dependency_comment(comments: list[dict]) -> bool:
    """Check if issue already has a dependency comment."""
    for c in comments:
        comment_body = c.get("body") or ""
        if DEPENDENCY_HEADER in comment_body:
            return True
    return False


# ---------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------


@dataclasses.dataclass
class AnalysisState:
    """State tracking for resumable analysis."""

    analyzed_batches: set[int] = dataclasses.field(default_factory=set)
    commented_issues: set[int] = dataclasses.field(default_factory=set)
    analyses: dict[int, dict] = dataclasses.field(default_factory=dict)

    def save(self, path: pathlib.Path) -> None:
        """Save state to file."""
        data = {
            "analyzed_batches": list(self.analyzed_batches),
            "commented_issues": list(self.commented_issues),
            "analyses": {str(k): v for k, v in self.analyses.items()},
        }
        write_secure(path, json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: pathlib.Path) -> "AnalysisState":
        """Load state from file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                analyzed_batches=set(data.get("analyzed_batches", [])),
                commented_issues=set(data.get("commented_issues", [])),
                analyses={int(k): v for k, v in data.get("analyses", {}).items()},
            )
        except (json.JSONDecodeError, KeyError) as e:
            log("WARN", f"Failed to load state: {e}")
            return cls()


# ---------------------------------------------------------------------
# Options and Results
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Options:
    """Configuration options for the analyzer."""

    analyze: bool
    apply: bool
    create_epic: bool
    combine: bool
    dry_run: bool
    reanalyze: bool
    cleanup: bool
    parallel: bool
    max_parallel: int
    timeout: int
    throttle_seconds: float
    model: str
    batch_dir: pathlib.Path
    state_dir: pathlib.Path | None
    json_output: bool


@dataclasses.dataclass
class Result:
    """Result of processing a single batch or issue."""

    item_id: int
    status: str
    error: str | None = None


# ---------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------


class IssueAnalyzer:
    """Analyzes GitHub issues using Claude Code to determine dependencies."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        tempdir: pathlib.Path,
        opts: Options,
        status_tracker: StatusTracker | None = None,
    ) -> None:
        """Initialize the analyzer."""
        self.repo_root = repo_root
        self.tempdir = tempdir
        self.opts = opts
        self.status_tracker = status_tracker

        # Use state_dir if provided, otherwise use tempdir
        state_dir = opts.state_dir if opts.state_dir else tempdir
        self.state_file = state_dir / "analyzer_state.json"
        self.state = AnalysisState.load(self.state_file)

        self._throttle_lock = threading.Lock()
        self._last_call = 0.0

    def create_batch_files(self, batch_num: int) -> dict[str, pathlib.Path]:
        """Create diagnostic files for a batch."""
        files = {
            "analysis": self.tempdir / f"batch-{batch_num}-analysis.json",
            "log": self.tempdir / f"batch-{batch_num}-claude.log",
            "prompt": self.tempdir / f"batch-{batch_num}-prompt.txt",
        }
        for path in files.values():
            path.touch(mode=0o600)
        return files

    def throttle(self) -> None:
        """Apply rate limiting between API calls."""
        if self.opts.throttle_seconds <= 0:
            return
        with self._throttle_lock:
            now = time.time()
            delta = now - self._last_call
            if delta < self.opts.throttle_seconds:
                time.sleep(self.opts.throttle_seconds - delta)
            self._last_call = time.time()

    def analyze_batch(
        self,
        batch_num: int,
        batch_file: pathlib.Path,
        reference_file: pathlib.Path,
        slot: int = -1,
    ) -> Result:
        """Analyze a single batch of issues with Claude."""

        def update_status(stage: str, info: str = "") -> None:
            if self.status_tracker and slot >= 0:
                self.status_tracker.update(slot, batch_num, stage, info)

        try:
            log("INFO", f"Analyzing batch {batch_num}")
            update_status("Init", "diag files")

            # Check if already analyzed
            if not self.opts.reanalyze and batch_num in self.state.analyzed_batches:
                log("INFO", f"  Batch {batch_num} already analyzed, skipping")
                update_status("Skipped", "already done")
                return Result(batch_num, "skipped")

            # Create diagnostic files
            diag_files = self.create_batch_files(batch_num)
            log("INFO", f"  Analysis file: {diag_files['analysis']}")
            log("INFO", f"  Log file: {diag_files['log']}")

            # Read input files
            update_status("Reading", "batch file")
            batch_content = batch_file.read_text()
            reference_list = reference_file.read_text()

            # Build prompt
            update_status("Building", "prompt")
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                reference_list=reference_list,
                batch_content=batch_content,
            )
            write_secure(diag_files["prompt"], prompt)

            # Dry-run: show what would be done
            if self.opts.dry_run:
                update_status("DryRun", "preview")
                log("INFO", f"[DRY RUN] Would analyze batch {batch_num}")
                log("INFO", f"  Batch file: {batch_file}")
                log("INFO", f"  Prompt size: {len(prompt)} bytes")
                return Result(batch_num, "dry-run")

            # Call Claude
            for attempt in range(1, MAX_RETRIES + 1):
                if self.opts.throttle_seconds > 0:
                    update_status("Throttling", f"{self.opts.throttle_seconds}s")
                self.throttle()

                start_time = time.time()
                attempt_info = f"attempt {attempt}/{MAX_RETRIES}" if attempt > 1 else ""
                update_status("Calling", f"Claude {attempt_info}".strip())
                log(
                    "INFO",
                    f"  Calling Claude {self.opts.model} (timeout: {self.opts.timeout}s)...",
                )

                # Build command
                cmd = [
                    "claude",
                    "-p",
                    "--output-format",
                    "json",
                    "--json-schema",
                    json.dumps(ANALYSIS_SCHEMA),
                    "--model",
                    self.opts.model,
                    "--tools",
                    "",
                    "--dangerously-skip-permissions",
                    "--no-session-persistence",
                    prompt,
                ]

                try:
                    cp = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.opts.timeout,
                        cwd=self.repo_root,
                    )
                except subprocess.TimeoutExpired:
                    update_status("Timeout", "retrying...")
                    log(
                        "WARN",
                        f"  Claude timed out after {self.opts.timeout}s, retrying...",
                    )
                    continue

                # Save log
                log_content = f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
                write_secure(diag_files["log"], log_content)

                # Check for rate limit
                combined = cp.stdout + cp.stderr
                reset = detect_rate_limit(combined)
                if reset:
                    remaining = reset - int(time.time())
                    update_status("RateLimit", f"~{remaining}s wait")
                    wait_until(reset)
                    continue

                if cp.returncode != 0:
                    update_status("Retrying", f"exit {cp.returncode}")
                    log(
                        "WARN",
                        f"  Claude returned exit code {cp.returncode}, retrying...",
                    )
                    time.sleep(2**attempt)
                    continue

                # Parse output
                update_status("Parsing", "JSON output")
                try:
                    output = json.loads(cp.stdout)
                    # Claude returns structured_output when using --json-schema
                    if "structured_output" in output:
                        analysis = output["structured_output"]
                    elif "result" in output:
                        # Fallback for older format
                        try:
                            analysis = json.loads(output["result"])
                        except (json.JSONDecodeError, TypeError):
                            analysis = {"analyses": []}
                    else:
                        analysis = output
                except json.JSONDecodeError as e:
                    update_status("ParseError", "retrying...")
                    log("WARN", f"  Failed to parse Claude output: {e}")
                    log("DEBUG", f"  stdout: {cp.stdout[:500]}")
                    time.sleep(2**attempt)
                    continue

                # Validate analysis
                if "analyses" not in analysis or not analysis["analyses"]:
                    update_status("EmptyResult", "retrying...")
                    log("WARN", "  Empty analysis result")
                    time.sleep(2**attempt)
                    continue

                # Save analysis
                write_secure(diag_files["analysis"], json.dumps(analysis, indent=2))

                # Update state
                for item in analysis.get("analyses", []):
                    issue_num = item.get("issue_number")
                    if issue_num:
                        self.state.analyses[issue_num] = item

                self.state.analyzed_batches.add(batch_num)
                self.state.save(self.state_file)

                elapsed = time.time() - start_time
                update_status("Done", f"{elapsed:.0f}s")
                log(
                    "INFO",
                    f"  Analyzed {len(analysis.get('analyses', []))} issues in {elapsed:.0f}s",
                )
                return Result(batch_num, "analyzed")

            raise RuntimeError(ERR_CLAUDE_RETRIES)

        except Exception as e:
            update_status("Error", str(e)[:30])
            return Result(batch_num, "error", str(e))

    def apply_comment(self, issue_num: int, analysis: dict, slot: int = -1) -> Result:
        """Apply a dependency comment to a single issue."""

        def update_status(stage: str, info: str = "") -> None:
            if self.status_tracker and slot >= 0:
                self.status_tracker.update(slot, issue_num, stage, info)

        try:
            log("INFO", f"Applying comment to #{issue_num}")
            update_status("Init")

            # Check if already commented
            if not self.opts.reanalyze and issue_num in self.state.commented_issues:
                log("INFO", f"  Issue #{issue_num} already has comment, skipping")
                update_status("Skipped", "already done")
                return Result(issue_num, "skipped")

            # Fetch issue to check for existing comment
            update_status("Fetching", "issue data")
            data = gh_issue_json(issue_num)
            comments = data.get("comments", [])

            # Delete existing dependency comments if reanalyzing
            if self.opts.reanalyze and not self.opts.dry_run:
                update_status("Deleting", "old comments")
                deleted = delete_dependency_comments(issue_num, comments)
                if deleted > 0:
                    log("INFO", f"  Deleted {deleted} existing dependency comment(s)")
            elif has_dependency_comment(comments):
                log(
                    "INFO",
                    f"  Issue #{issue_num} already has dependency comment, skipping",
                )
                update_status("Skipped", "has comment")
                return Result(issue_num, "skipped")

            # Build comment body
            update_status("Building", "comment")
            depends_on = analysis.get("depends_on", [])
            blocks = analysis.get("blocks", [])
            priority = analysis.get("priority", "P2")
            reasoning = analysis.get("reasoning", "")

            comment_parts = [DEPENDENCY_HEADER, ""]
            comment_parts.append(f"**Priority**: {priority}")
            comment_parts.append("")

            if depends_on:
                comment_parts.append("### Depends On")
                for dep in depends_on:
                    comment_parts.append(f"- #{dep}")
                comment_parts.append("")

            if blocks:
                comment_parts.append("### Blocks")
                for blk in blocks:
                    comment_parts.append(f"- #{blk}")
                comment_parts.append("")

            if reasoning:
                comment_parts.append("### Analysis")
                comment_parts.append(reasoning)
                comment_parts.append("")

            comment_parts.append("---")
            comment_parts.append("*Analyzed by Claude Code dependency analyzer*")

            comment_body = "\n".join(comment_parts)

            # Dry-run: show what would be done
            if self.opts.dry_run:
                update_status("DryRun", "preview")
                log("INFO", f"[DRY RUN] Would post comment to #{issue_num}")
                log("INFO", f"  Priority: {priority}")
                log("INFO", f"  Depends on: {depends_on}")
                log("INFO", f"  Blocks: {blocks}")
                return Result(issue_num, "dry-run")

            # Post comment
            update_status("Posting")
            proc = subprocess.Popen(
                ["gh", "issue", "comment", str(issue_num), "--body-file", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(input=comment_body)

            if proc.returncode != 0:
                log("WARN", f"  Failed to post comment: {stderr}")
                return Result(issue_num, "error", stderr)

            # Update state
            self.state.commented_issues.add(issue_num)
            self.state.save(self.state_file)

            update_status("Done", "posted")
            log("INFO", f"  Posted comment to #{issue_num}")
            return Result(issue_num, "posted")

        except Exception as e:
            update_status("Error", str(e)[:30])
            return Result(issue_num, "error", str(e))

    def combine_analyses(self) -> dict:
        """Combine all analyses into a single dependency graph."""
        log("INFO", f"Combining {len(self.state.analyses)} analyses")

        priority_counts = {"P0": 0, "P1": 0, "P2": 0}
        for analysis in self.state.analyses.values():
            priority_counts[analysis.get("priority", "P2")] += 1

        combined = {
            "total_issues": len(self.state.analyses),
            "analyses": self.state.analyses,
            "priority_counts": priority_counts,
        }

        output_file = self.tempdir / "combined_analysis.json"
        write_secure(output_file, json.dumps(combined, indent=2))
        log("INFO", f"Saved combined analysis to {output_file}")

        return combined

    def create_epic(self) -> int | None:
        """Create GitHub epic tracking issue."""
        log("INFO", "Creating epic tracking issue")

        if not self.state.analyses:
            log("ERROR", "No analyses found. Run --analyze first.")
            return None

        priority_counts = {"P0": 0, "P1": 0, "P2": 0}
        for analysis in self.state.analyses.values():
            priority_counts[analysis.get("priority", "P2")] += 1

        # Sort issues by priority then number
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        sorted_issues = sorted(
            self.state.analyses.items(),
            key=lambda x: (
                priority_order.get(x[1].get("priority", "P2"), 2),
                int(x[0]),
            ),
        )

        # Build epic body
        body_parts = [
            "# ML Odyssey Implementation Roadmap",
            "",
            "This epic tracks the prioritized implementation order for all open issues.",
            "",
            "## Summary",
            f"- **Total Issues**: {len(self.state.analyses)}",
            f"- **P0 (Foundation)**: {priority_counts.get('P0', 0)}",
            f"- **P1 (Important)**: {priority_counts.get('P1', 0)}",
            f"- **P2 (Nice-to-have)**: {priority_counts.get('P2', 0)}",
            "",
            "## Prioritized Roadmap",
            "",
        ]

        current_priority = None
        for issue_num, analysis in sorted_issues:
            priority = analysis.get("priority", "P2")
            if priority != current_priority:
                current_priority = priority
                priority_labels = {
                    "P0": "Foundation",
                    "P1": "Important",
                    "P2": "Nice-to-have",
                }
                body_parts.append(f"### {priority}: {priority_labels.get(priority, priority)}")
                body_parts.append("")

            depends = analysis.get("depends_on", [])
            deps_str = f" (depends on: {', '.join(f'#{d}' for d in depends)})" if depends else ""
            body_parts.append(f"- [ ] #{issue_num}{deps_str}")

        body_parts.extend(["", "---", "*Generated by Claude Code dependency analyzer*"])

        epic_body = "\n".join(body_parts)
        epic_title = "[Epic] ML Odyssey Implementation Roadmap"

        if self.opts.dry_run:
            log("INFO", "[DRY RUN] Would create epic:")
            log("INFO", f"  Title: {epic_title}")
            log("INFO", f"  Body size: {len(epic_body)} bytes")
            log("INFO", f"  Issues: {len(sorted_issues)}")
            return None

        # Ensure labels exist
        for label, color, desc in [
            ("epic", "5319e7", "Epic tracking issue"),
            ("tracking", "0e8a16", "Progress tracking"),
        ]:
            run(
                [
                    "gh",
                    "label",
                    "create",
                    label,
                    "--color",
                    color,
                    "--description",
                    desc,
                    "--force",
                ]
            )

        # Create epic
        proc = subprocess.Popen(
            [
                "gh",
                "issue",
                "create",
                "--title",
                epic_title,
                "--label",
                "epic,tracking",
                "--body-file",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(input=epic_body)

        if proc.returncode != 0:
            log("ERROR", f"Failed to create epic: {stderr}")
            return None

        # Extract issue number from URL
        url = stdout.strip()
        issue_num = int(url.split("/")[-1])
        log("INFO", f"Created epic: {url}")
        return issue_num


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    """CLI entry point for the issue analyzer."""
    p = argparse.ArgumentParser(
        description="Analyze GitHub issues using Claude Code CLI to determine dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --analyze                     Analyze all batches
  %(prog)s --analyze --model sonnet      Use Sonnet for higher quality
  %(prog)s --combine                     Combine analyses into dependency graph
  %(prog)s --apply --dry-run             Preview dependency comments
  %(prog)s --apply                       Post dependency comments
  %(prog)s --create-epic                 Create tracking epic
  %(prog)s --analyze --apply --create-epic   Full workflow
  %(prog)s --analyze --parallel          Parallel batch analysis
""",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze batch files to determine dependencies",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply dependency comments to GitHub issues",
    )
    p.add_argument("--create-epic", action="store_true", help="Create tracking epic issue")
    p.add_argument(
        "--combine",
        action="store_true",
        help="Combine analysis files into dependency graph",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")
    p.add_argument(
        "--reanalyze",
        action="store_true",
        help="Re-analyze batches/issues with existing results",
    )
    p.add_argument("--cleanup", action="store_true", help="Delete temp directory on completion")
    p.add_argument("--parallel", action="store_true", help="Process batches in parallel")
    p.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        metavar="N",
        help="Max concurrent jobs (default: 4)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=CLAUDE_TIMEOUT,
        metavar="SEC",
        help=f"Timeout per batch (default: {CLAUDE_TIMEOUT})",
    )
    p.add_argument(
        "--throttle",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Seconds between API calls (default: 1.0)",
    )
    p.add_argument(
        "--model",
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
        help="Claude model (default: haiku)",
    )
    p.add_argument(
        "--batch-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp"),  # nosec B108 - CLI tool default, user-overridable
        help="Directory with batch files",
    )
    p.add_argument(
        "--state-dir",
        type=pathlib.Path,
        help="Directory to persist state (default: tempdir)",
    )
    p.add_argument("--json", action="store_true", help="Output results as JSON")

    args = p.parse_args()

    if not any([args.analyze, args.apply, args.create_epic, args.combine]):
        p.print_help()
        print("\nWorkflow:")
        print("  1. python scripts/analyze_issues_claude.py --analyze")
        print("  2. python scripts/analyze_issues_claude.py --combine")
        print("  3. python scripts/analyze_issues_claude.py --apply --dry-run")
        print("  4. python scripts/analyze_issues_claude.py --apply")
        print("  5. python scripts/analyze_issues_claude.py --create-epic")
        return 1

    opts = Options(
        analyze=args.analyze,
        apply=args.apply,
        create_epic=args.create_epic,
        combine=args.combine,
        dry_run=args.dry_run,
        reanalyze=args.reanalyze,
        cleanup=args.cleanup,
        parallel=args.parallel,
        max_parallel=args.max_parallel,
        timeout=args.timeout,
        throttle_seconds=args.throttle,
        model=args.model,
        batch_dir=args.batch_dir,
        state_dir=args.state_dir,
        json_output=args.json,
    )

    repo_root = get_repo_root()
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix="analyze-issues-"))
    tempdir.chmod(0o700)

    # Register cleanup handler
    def cleanup_handler() -> None:
        if tempdir.exists():
            if opts.cleanup:
                shutil.rmtree(tempdir, ignore_errors=True)
            else:
                log("INFO", f"Temp directory preserved: {tempdir}")

    atexit.register(cleanup_handler)

    def signal_handler(signum: int, _frame: object) -> None:
        cleanup_handler()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Startup status messages
    log("INFO", f"Temp directory: {tempdir}")
    log("INFO", f"Batch directory: {opts.batch_dir}")
    log("INFO", f"Model: {opts.model}")
    if opts.parallel:
        log("INFO", f"Parallel: ENABLED ({opts.max_parallel} jobs)")
    if opts.dry_run:
        log("INFO", "Dry-run: ENABLED (no changes will be made)")
    if opts.reanalyze:
        log("INFO", "Reanalyze: ENABLED (will re-process existing items)")

    # Create status tracker
    status_tracker: StatusTracker | None = None
    if opts.parallel:
        status_tracker = StatusTracker(opts.max_parallel)

    analyzer = IssueAnalyzer(repo_root, tempdir, opts, status_tracker)

    results: list[Result] = []

    # Phase 1: Analyze batches
    if opts.analyze:
        reference_file = opts.batch_dir / "issue_reference.txt"
        if not reference_file.exists():
            log("ERROR", f"Reference file not found: {reference_file}")
            return 1

        batch_files = sorted(opts.batch_dir.glob("batch_*.txt"))
        if not batch_files:
            log("ERROR", f"No batch files found in {opts.batch_dir}")
            return 1

        log("INFO", f"Found {len(batch_files)} batch files to analyze")

        print()
        print("==========================================")
        print("  Issue Dependency Analyzer")
        print(f"  Total batches: {len(batch_files)}")
        if opts.parallel:
            print(f"  Parallel jobs: {opts.max_parallel}")
        print("==========================================")
        print()

        if opts.parallel and status_tracker:
            status_tracker.set_total(len(batch_files))
            status_tracker.update_main("Initializing")
            status_tracker.start_display()

            def process_with_status(batch_file: pathlib.Path, idx: int) -> tuple[Result, str, int]:
                assert status_tracker is not None  # Already checked above
                batch_num = int(batch_file.stem.split("_")[1])
                slot = status_tracker.acquire_slot(batch_num)
                buffer = io.StringIO()
                try:
                    with redirect_stdout(buffer), redirect_stderr(buffer):
                        result = analyzer.analyze_batch(batch_num, batch_file, reference_file, slot)
                    return (result, buffer.getvalue(), slot)
                finally:
                    assert status_tracker is not None  # Already checked above
                    status_tracker.release_slot(slot)
                    status_tracker.increment_completed()

            try:
                with ThreadPoolExecutor(max_workers=opts.max_parallel) as ex:
                    status_tracker.update_main("Spawning", f"{len(batch_files)} tasks")
                    futures = {}
                    for i, batch_file in enumerate(batch_files):
                        futures[ex.submit(process_with_status, batch_file, i)] = i
                        batch_num = int(batch_file.stem.split("_")[1])
                        status_tracker.update_main("Spawning", f"batch_{batch_num}")

                    status_tracker.update_main("Processing")
                    indexed_results: dict[int, tuple[Result, str]] = {}
                    for f in as_completed(futures):
                        idx = futures[f]
                        result, output, _slot = f.result()
                        indexed_results[idx] = (result, output)

                    status_tracker.update_main("Collecting", "results")
                    status_tracker.stop_display()

                    for i in range(len(batch_files)):
                        result, output = indexed_results[i]
                        print(output, end="")
                        results.append(result)
            except Exception:
                status_tracker.stop_display()
                raise
        else:
            for batch_file in batch_files:
                batch_num = int(batch_file.stem.split("_")[1])
                result = analyzer.analyze_batch(batch_num, batch_file, reference_file)
                results.append(result)

    # Phase 2: Combine analyses
    if opts.combine or opts.apply or opts.create_epic:
        analyzer.combine_analyses()

    # Phase 3: Apply comments
    if opts.apply:
        if not analyzer.state.analyses:
            log("ERROR", "No analyses found. Run --analyze first.")
            return 1

        issues = list(analyzer.state.analyses.keys())
        log("INFO", f"Applying comments to {len(issues)} issues")

        print()
        print("==========================================")
        print("  Applying Dependency Comments")
        print(f"  Total issues: {len(issues)}")
        print("==========================================")
        print()

        # Reset status tracker for apply phase
        if opts.parallel and status_tracker:
            status_tracker = StatusTracker(opts.max_parallel)
            status_tracker.set_total(len(issues))
            status_tracker.update_main("Initializing")
            status_tracker.start_display()

            def apply_with_status(issue_num: int, idx: int) -> tuple[Result, str, int]:
                assert status_tracker is not None  # Already checked above
                analysis = analyzer.state.analyses[issue_num]
                slot = status_tracker.acquire_slot(issue_num)
                buffer = io.StringIO()
                try:
                    with redirect_stdout(buffer), redirect_stderr(buffer):
                        result = analyzer.apply_comment(issue_num, analysis, slot)
                    return (result, buffer.getvalue(), slot)
                finally:
                    assert status_tracker is not None  # Already checked above
                    status_tracker.release_slot(slot)
                    status_tracker.increment_completed()

            try:
                with ThreadPoolExecutor(max_workers=opts.max_parallel) as ex:
                    status_tracker.update_main("Spawning", f"{len(issues)} tasks")
                    futures = {}
                    for i, issue_num in enumerate(issues):
                        futures[ex.submit(apply_with_status, issue_num, i)] = i
                        status_tracker.update_main("Spawning", f"#{issue_num}")

                    status_tracker.update_main("Processing")
                    indexed_apply_results: dict[int, tuple[Result, str]] = {}
                    for f in as_completed(futures):
                        idx = futures[f]
                        result, output, _slot = f.result()
                        indexed_apply_results[idx] = (result, output)

                    status_tracker.update_main("Collecting", "results")
                    status_tracker.stop_display()

                    for i in range(len(issues)):
                        result, output = indexed_apply_results[i]
                        print(output, end="")
                        results.append(result)
            except Exception:
                status_tracker.stop_display()
                raise
        else:
            for issue_num in issues:
                analysis = analyzer.state.analyses[issue_num]
                result = analyzer.apply_comment(issue_num, analysis)
                results.append(result)

    # Phase 4: Create epic
    if opts.create_epic:
        epic_num = analyzer.create_epic()
        if epic_num is not None:
            results.append(Result(epic_num, "created"))
        elif not opts.dry_run:
            results.append(Result(0, "error", "Failed to create epic"))

    # Summary
    if opts.json_output:
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        analyzed = sum(1 for r in results if r.status == "analyzed")
        posted = sum(1 for r in results if r.status == "posted")
        created = sum(1 for r in results if r.status == "created")
        errored = sum(1 for r in results if r.status == "error")
        skipped = sum(1 for r in results if r.status in ("skipped", "dry-run"))

        print()
        print("==========================================")
        print("  Summary")
        if analyzed > 0:
            print(f"  Analyzed: {analyzed}")
        if posted > 0:
            print(f"  Posted:   {posted}")
        if created > 0:
            print(f"  Created:  {created}")
        print(f"  Skipped:  {skipped}")
        print(f"  Errors:   {errored}")
        print(f"  Total:    {len(results)}")
        if not opts.cleanup:
            print()
            print(f"  Temp directory: {tempdir}")
        print("==========================================")

    error_count = sum(1 for r in results if r.status == "error")
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
