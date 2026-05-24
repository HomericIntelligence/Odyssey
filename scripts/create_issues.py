#!/usr/bin/env python3
"""
DEPRECATED: This script is no longer used.

The notes/plan/ directory has been removed. Planning is now done directly
through GitHub issues. See .claude/shared/github-issue-workflow.md for the
new workflow.

To create issues, use the GitHub CLI directly:
    gh issue create --title "..." --body "..." --label "..."

---

GitHub Issues Creator (DEPRECATED)

This script automatically creates GitHub issues from github_issue.md files
in the notes/plan directory structure. It supports dry-run mode, resuming
from interruptions, and detailed progress tracking.

Usage:
    python create_issues.py --dry-run              # Show what would be done
    python create_issues.py --section 01-foundation # Only process one section
    python create_issues.py --file notes/plan/.../github_issue.md # Test single file
    python create_issues.py --resume               # Resume from saved state
    python create_issues.py                        # Actually create all issues
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from hephaestus.utils import get_repo_root

# Try to import tqdm for progress bars, fall back to simple counter if not available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for better progress bars: pip install tqdm")


# ANSI color codes for terminal output
class Colors:
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
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = ""
        Colors.OKBLUE = ""
        Colors.OKCYAN = ""
        Colors.OKGREEN = ""
        Colors.WARNING = ""
        Colors.FAIL = ""
        Colors.ENDC = ""
        Colors.BOLD = ""
        Colors.UNDERLINE = ""


def check_github_rate_limit() -> Tuple[int, float]:
    """
    Check GitHub API rate limit status.

    Returns:
        Tuple of (remaining calls, reset time as unix timestamp)

    Raises:
        RuntimeError: If unable to check rate limit
    """
    try:
        result = subprocess.run(["gh", "api", "rate_limit"], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        remaining = data["resources"]["core"]["remaining"]
        reset_time = data["resources"]["core"]["reset"]
        return remaining, reset_time
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to check rate limit: {e.stderr}")
        # Return conservative values if check fails
        return 100, time.time() + 60
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Failed to parse rate limit response: {e}")
        return 100, time.time() + 60


def smart_rate_limit_sleep() -> None:
    """
    Sleep only if necessary based on GitHub API rate limit.

    This function checks the current rate limit and sleeps only when needed:
    - No sleep if remaining > 100 (healthy)
    - Exponential backoff if 10 < remaining <= 100 (low)
    - Wait until reset if remaining <= 10 (critical).
    """
    remaining, reset_time = check_github_rate_limit()

    if remaining <= 10:
        # Critical - wait until reset (max 60 seconds)
        wait_time = max(0, reset_time - time.time())
        if wait_time > 0:
            actual_wait = min(wait_time, 60)
            logging.warning(f"Rate limit critical ({remaining} remaining), waiting {actual_wait:.0f}s until reset")
            time.sleep(actual_wait)
    elif remaining <= 100:
        # Low - exponential backoff based on remaining calls
        # remaining=100 -> 0s, remaining=50 -> 2.5s, remaining=11 -> 4.45s
        backoff = (100 - remaining) / 20
        logging.info(f"Rate limit low ({remaining} remaining), sleeping {backoff:.2f}s")
        time.sleep(backoff)
    else:
        # Healthy - no sleep needed
        logging.debug(f"Rate limit healthy ({remaining} remaining), no sleep")


@dataclass
class Issue:
    """Represents a GitHub issue to be created"""

    title: str
    labels: List[str]
    body: str
    file_path: str
    section_name: str
    issue_type: str
    issue_url: Optional[str] = None
    created: bool = False
    error: Optional[str] = None


@dataclass
class Statistics:
    """Statistics about issues to be created"""

    total_files: int = 0
    total_issues: int = 0
    by_section: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)


class IssueParser:
    """Parses github_issue.md files to extract issue information"""

    ISSUE_TYPES = ["Plan", "Test", "Implementation", "Packaging", "Cleanup"]

    # Pattern to match issue sections - multiple formats
    # Format 1: ## Plan Issue
    # Format 2: **Plan Issue**:
    ISSUE_SECTION_PATTERN = re.compile(
        r"^(?:##\s+|\*\*)(Plan|Test|Implementation|Packaging|Cleanup) Issue(?:\s*$|\*\*:)",
        re.MULTILINE,
    )

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = file_path.read_text(encoding="utf-8")

    def parse(self) -> List[Issue]:
        """Parse all issues from the file"""
        issues = []
        section_name = self._extract_section_name()

        # Split content by issue sections
        sections = self.ISSUE_SECTION_PATTERN.split(self.content)

        # First element is the header before any issue
        sections = sections[1:]  # Skip header

        # Now we have pairs: (issue_type, content)
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                issue_type = sections[i].strip()
                issue_content = sections[i + 1]

                issue = self._parse_issue_section(issue_type, issue_content, section_name)
                if issue:
                    issues.append(issue)

        return issues

    def _extract_section_name(self) -> str:
        """Extract the section name from the file path"""
        # Get relative path from notes/plan/
        parts = self.file_path.parts
        try:
            plan_idx = parts.index("plan")
            section = parts[plan_idx + 1]
            return section
        except (ValueError, IndexError):
            return "unknown"

    def _parse_issue_section(self, issue_type: str, content: str, section_name: str) -> Optional[Issue]:
        """Parse a single issue section"""

        # Extract title - try multiple formats
        # Format 1: **Title**: `[Plan] ...`
        title_match = re.search(r"\*\*Title\*\*:\s*`([^`]+)`", content)
        if not title_match:
            # Format 2: **Title**: [Plan] ...
            title_match = re.search(r"\*\*Title\*\*:\s*(.+?)(?:\n|$)", content)
        if not title_match:
            # Format 3: - Title: [Plan] ...
            title_match = re.search(r"^-\s*Title:\s*(.+?)(?:\n|$)", content, re.MULTILINE)

        if not title_match:
            return None
        title = title_match.group(1).strip()

        # Extract labels - try different formats
        # Format 1: **Labels**: `label1`, `label2`
        labels_match = re.search(r"\*\*Labels\*\*:\s*`([^`]+)`(?:,\s*`([^`]+)`)*", content)
        if not labels_match:
            # Format 2: **Labels**: label1, label2
            labels_match = re.search(r"\*\*Labels\*\*:\s*(.+?)(?:\n\n|\*\*|-)", content)
            if labels_match:
                labels_text = labels_match.group(1)
                labels = [label.strip("` \t") for label in re.findall(r"`([^`]+)`", labels_text)]
                # If no labels with backticks found, try comma-separated
                if not labels:
                    labels = [label.strip() for label in labels_text.split(",")]
            else:
                # Format 3: - Labels: label1, label2
                labels_match = re.search(r"^-\s*Labels:\s*(.+?)(?:\n|$)", content, re.MULTILINE)
                if labels_match:
                    labels_text = labels_match.group(1)
                    labels = [label.strip() for label in labels_text.split(",")]
                else:
                    labels = []
        else:
            labels = [label.strip("` \t") for label in labels_match.groups() if label]

        # Extract body - try different formats
        # Format 1: **Body**: with code block
        body_match = re.search(r"\*\*Body\*\*:\s*\n```\n(.+?)\n```", content, re.DOTALL)

        if not body_match:
            # Format 2: - Body: with code block
            body_match = re.search(r"^-\s*Body:\s*\n```\n(.+?)\n```", content, re.MULTILINE | re.DOTALL)

        if not body_match:
            # Format 3: **Body**: without code block
            body_match = re.search(
                r"\*\*Body\*\*:\s*\n\n(.+?)\n\n\*\*(?:GitHub Issue URL|URL)\*\*:",
                content,
                re.DOTALL,
            )

        if not body_match:
            return None
        body = body_match.group(1).strip()

        # Check if issue already created - try both URL formats
        url_match = re.search(r"(?:\*\*)?(?:GitHub Issue URL|URL)(?:\*\*)?:\s*(.+?)(?:\n|$)", content)
        issue_url = None
        created = False
        if url_match:
            url_text = url_match.group(1).strip()
            if url_text and url_text not in ["[To be created]", "[to be filled]"]:
                issue_url = url_text
                created = True

        return Issue(
            title=title,
            labels=labels,
            body=body,
            file_path=str(self.file_path),
            section_name=section_name,
            issue_type=issue_type,
            issue_url=issue_url,
            created=created,
        )


class IssueCreator:
    """Creates GitHub issues and updates markdown files"""

    def __init__(
        self,
        repo: str,
        dry_run: bool = False,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.repo = repo
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)

    def create_issue(self, issue: Issue) -> bool:
        """Create a GitHub issue"""

        if self.dry_run:
            return True

        # Create temporary file for body
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(issue.body)
            body_file = f.name

        try:
            # Build gh command
            cmd = [
                "gh",
                "issue",
                "create",
                "--title",
                issue.title,
                "--body-file",
                body_file,
                "--repo",
                self.repo,
            ]

            # Add labels
            for label in issue.labels:
                cmd.extend(["--label", label])

            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)

                    # Extract issue URL from output
                    issue_url = result.stdout.strip()
                    issue.issue_url = issue_url
                    issue.created = True

                    self.logger.info(f"Created issue: {issue_url}")
                    return True

                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                    if attempt == self.max_retries - 1:
                        issue.error = "Timeout creating issue"
                        return False

                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error creating issue (attempt {attempt + 1}/{self.max_retries}): {e.stderr}")
                    if attempt == self.max_retries - 1:
                        issue.error = e.stderr
                        return False

        finally:
            # Clean up temp file
            try:
                os.unlink(body_file)
            except (OSError, PermissionError):
                # File may have already been deleted or is not accessible
                pass

        return False

    def update_markdown_file(self, issue: Issue) -> bool:
        """Update the markdown file with the issue URL"""

        if self.dry_run or not issue.issue_url:
            return True

        try:
            file_path = Path(issue.file_path)
            content = file_path.read_text(encoding="utf-8")

            # Find the issue section - try both formats
            issue_section_start = content.find(f"## {issue.issue_type} Issue")
            if issue_section_start == -1:
                # Try bold format
                issue_section_start = content.find(f"**{issue.issue_type} Issue**:")
                if issue_section_start == -1:
                    self.logger.error(f"Could not find issue section for {issue.issue_type}")
                    return False

            # Find the URL line within this section - handle multiple formats
            url_line_pattern = re.compile(
                r"((?:\*\*)?(?:GitHub Issue URL|URL)(?:\*\*)?:\s*)\[(?:To be created|to be filled)\]",
                re.MULTILINE,
            )

            # Also handle bullet format
            url_bullet_pattern = re.compile(
                r"(^-\s*(?:URL|GitHub Issue URL):\s*)\[(?:To be created|to be filled)\]",
                re.MULTILINE,
            )

            # Search from the issue section onwards
            section_content = content[issue_section_start:]
            # Find next section - try both ## and **
            next_section_hash = section_content.find("\n## ", 1)
            next_section_bold = section_content.find("\n**", 1)

            # Use whichever comes first (or -1 if neither found)
            if next_section_hash == -1:
                next_section = next_section_bold
            elif next_section_bold == -1:
                next_section = next_section_hash
            else:
                next_section = min(next_section_hash, next_section_bold)

            if next_section != -1:
                section_content = section_content[:next_section]

            # Try to replace with both patterns
            new_section = url_line_pattern.sub(f"\\1{issue.issue_url}", section_content, count=1)

            # If that didn't work, try bullet pattern
            if new_section == section_content:
                new_section = url_bullet_pattern.sub(f"\\1{issue.issue_url}", section_content, count=1)

            # Reconstruct the full content
            if next_section != -1:
                new_content = (
                    content[:issue_section_start] + new_section + content[issue_section_start + next_section :]
                )
            else:
                new_content = content[:issue_section_start] + new_section

            # Write back
            file_path.write_text(new_content, encoding="utf-8")
            self.logger.info(f"Updated {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating file {issue.file_path}: {e}")
            issue.error = str(e)
            return False


class StateManager:
    """Manages state for resuming interrupted runs"""

    def __init__(self, state_file: Path):
        self.state_file = state_file

    def save_state(self, issues: List[Issue]):
        """Save current state"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "issues": [
                {
                    "file_path": issue.file_path,
                    "issue_type": issue.issue_type,
                    "title": issue.title,
                    "created": issue.created,
                    "issue_url": issue.issue_url,
                    "error": issue.error,
                }
                for issue in issues
            ],
        }

        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_state(self) -> Optional[Dict]:
        """Load saved state"""
        if not self.state_file.exists():
            return None

        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error loading state: {e}")
            return None

    def clear_state(self):
        """Clear saved state"""
        if self.state_file.exists():
            self.state_file.unlink()


def find_all_issue_files(plan_dir: Path, section_filter: Optional[str] = None) -> List[Path]:
    """Find all github_issue.md files"""

    if section_filter:
        search_dir = plan_dir / section_filter
        if not search_dir.exists():
            print(f"{Colors.FAIL}Error: Section '{section_filter}' not found{Colors.ENDC}")
            sys.exit(1)
    else:
        search_dir = plan_dir

    files = sorted(search_dir.rglob("github_issue.md"))
    return files


def parse_all_issues(files: List[Path], resume_state: Optional[Dict] = None) -> Tuple[List[Issue], Statistics]:
    """Parse all issues from files"""

    all_issues = []
    stats = Statistics(total_files=len(files))
    stats.by_section = defaultdict(int)
    stats.by_type = defaultdict(int)

    # Build resume lookup
    resume_lookup = {}
    if resume_state:
        for issue_data in resume_state.get("issues", []):
            key = (issue_data["file_path"], issue_data["issue_type"])
            resume_lookup[key] = issue_data

    print(f"\n{Colors.OKCYAN}Parsing issue files...{Colors.ENDC}")

    iterator = tqdm(files, desc="Parsing files") if HAS_TQDM else files

    for file_path in iterator:
        if not HAS_TQDM:
            print(f"  Parsing: {file_path.relative_to(file_path.parents[3])}")

        parser = IssueParser(file_path)
        issues = parser.parse()

        # Apply resume state if available
        for issue in issues:
            key = (issue.file_path, issue.issue_type)
            if key in resume_lookup:
                resume_data = resume_lookup[key]
                issue.created = resume_data.get("created", False)
                issue.issue_url = resume_data.get("issue_url")
                issue.error = resume_data.get("error")

        all_issues.extend(issues)

        # Update statistics
        for issue in issues:
            stats.total_issues += 1
            stats.by_section[issue.section_name] += 1
            stats.by_type[issue.issue_type] += 1

    return all_issues, stats


def print_dry_run_summary(issues: List[Issue], stats: Statistics, show_limit: int = 5):
    """Print detailed dry-run summary"""

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.WARNING}{Colors.BOLD}DRY RUN MODE - No issues will be created{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

    print(f"Found {Colors.BOLD}{stats.total_files}{Colors.ENDC} github_issue.md files")
    print(f"Total issues to create: {Colors.BOLD}{stats.total_issues:,}{Colors.ENDC}\n")

    # Breakdown by section
    print(f"{Colors.OKBLUE}Breakdown by section:{Colors.ENDC}")
    for section in sorted(stats.by_section.keys()):
        count = stats.by_section[section]
        # Count unique files in this section
        files_in_section = len(set(issue.file_path for issue in issues if issue.section_name == section))
        print(f"  - {section}: {count} issues ({files_in_section} components)")

    # Breakdown by type
    print(f"\n{Colors.OKBLUE}Breakdown by type:{Colors.ENDC}")
    for issue_type in sorted(stats.by_type.keys()):
        count = stats.by_type[issue_type]
        print(f"  - {issue_type}: {count} issues")

    # Show example issues
    print(f"\n{Colors.OKBLUE}Example issues (first {show_limit}):{Colors.ENDC}\n")

    for idx, issue in enumerate(issues[:show_limit], 1):
        print(
            f"{Colors.OKCYAN}[{idx}/{stats.total_issues}]{Colors.ENDC} "
            f"Creating issue in {Path(issue.file_path).relative_to(Path.cwd())}"
        )
        print(f"  Title: {Colors.BOLD}{issue.title}{Colors.ENDC}")
        print(f"  Labels: {', '.join(issue.labels)}")

        body_lines = issue.body.count("\n") + 1
        print(f"  Body: ({body_lines} lines)")

        # Show command that would be executed
        labels_str = ",".join(issue.labels)
        print(f"  {Colors.OKBLUE}Command:{Colors.ENDC} gh issue create \\")
        print(f'           --title "{issue.title}" \\')
        print("           --body-file /tmp/issue_body_xxx.md \\")
        print(f"           --label {labels_str} \\")
        print("           --repo <repo>")
        print(f"  {Colors.OKGREEN}Would update:{Colors.ENDC} {issue.file_path}")
        print()

    if stats.total_issues > show_limit:
        print(f"... ({stats.total_issues - show_limit} more issues)\n")


def create_all_issues(
    issues: List[Issue],
    creator: IssueCreator,
    state_manager: StateManager,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Create all issues and track progress"""

    # Filter out already created issues
    to_create = [issue for issue in issues if not issue.created]

    if not to_create:
        print(f"\n{Colors.OKGREEN}All issues already created!{Colors.ENDC}")
        return len(issues), 0

    print(f"\n{Colors.OKCYAN}Creating {len(to_create)} issues...{Colors.ENDC}\n")

    success_count = 0
    error_count = 0

    iterator = tqdm(to_create, desc="Creating issues", unit="issue") if HAS_TQDM else to_create

    for idx, issue in enumerate(iterator, 1):
        if not HAS_TQDM:
            print(f"[{idx}/{len(to_create)}] Creating: {issue.title}")

        if not dry_run:
            # Create the issue
            if creator.create_issue(issue):
                # Update the markdown file
                if creator.update_markdown_file(issue):
                    success_count += 1
                else:
                    error_count += 1

                # Smart rate limiting based on actual GitHub API status
                smart_rate_limit_sleep()
            else:
                error_count += 1

            # Save state periodically (every 10 issues)
            if idx % 10 == 0:
                state_manager.save_state(issues)
        else:
            success_count += 1

    # Final state save
    if not dry_run:
        state_manager.save_state(issues)

    return success_count, error_count


def create_all_issues_concurrent(
    issues: List[Issue],
    creator: IssueCreator,
    state_manager: StateManager,
    dry_run: bool = False,
    max_workers: int = 5,
) -> Tuple[int, int]:
    """Create all issues concurrently using ThreadPoolExecutor"""

    # Filter out already created issues
    to_create = [issue for issue in issues if not issue.created]

    if not to_create:
        print(f"\n{Colors.OKGREEN}All issues already created!{Colors.ENDC}")
        return len(issues), 0

    print(f"\n{Colors.OKCYAN}Creating {len(to_create)} issues concurrently")
    print(f"(max {max_workers} workers)...{Colors.ENDC}\n")

    success_count = 0
    error_count = 0

    if dry_run:
        # Dry run - just count
        return len(to_create), 0

    def create_single_issue(issue: Issue, idx: int) -> Tuple[bool, int]:
        """Create a single issue and return success status"""
        if not HAS_TQDM:
            print(f"[{idx}/{len(to_create)}] Creating: {issue.title}")

        # Create the issue
        if creator.create_issue(issue):
            # Update the markdown file
            if creator.update_markdown_file(issue):
                # Smart rate limiting
                smart_rate_limit_sleep()
                return True, idx
            else:
                return False, idx
        return False, idx

    # Use ThreadPoolExecutor for concurrent creation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_issue = {
            executor.submit(create_single_issue, issue, idx): issue for idx, issue in enumerate(to_create, 1)
        }

        # Use tqdm if available
        iterator = (
            tqdm(
                as_completed(future_to_issue),
                total=len(to_create),
                desc="Creating issues",
                unit="issue",
            )
            if HAS_TQDM
            else as_completed(future_to_issue)
        )

        completed_count = 0
        for future in iterator:
            issue = future_to_issue[future]
            try:
                success, idx = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logging.error(f"Exception creating issue {issue.title}: {e}")
                error_count += 1

            completed_count += 1

            # Save state periodically (every 10 issues)
            if completed_count % 10 == 0:
                state_manager.save_state(issues)

    # Final state save
    state_manager.save_state(issues)

    return success_count, error_count


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""

    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"create_issues_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger


def get_repo_name() -> str:
    """Get the GitHub repository name from git remote"""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )

        remote_url = result.stdout.strip()

        # Parse repo from URL
        # Handle both HTTPS and SSH formats
        if remote_url.startswith("git@"):
            # SSH: git@github.com:user/repo.git
            match = re.search(r"git@github\.com:(.+?/.+?)(?:\.git)?$", remote_url)
        else:
            # HTTPS: https://github.com/user/repo.git
            match = re.search(r"github\.com/(.+?/.+?)(?:\.git)?$", remote_url)

        if match:
            return match.group(1)

        print(f"{Colors.FAIL}Error: Could not parse repository name from: {remote_url}{Colors.ENDC}")
        sys.exit(1)

    except subprocess.CalledProcessError:
        print(f"{Colors.FAIL}Error: Could not get git remote. Are you in a git repository?{Colors.ENDC}")
        sys.exit(1)


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Create GitHub issues from markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating issues",
    )

    parser.add_argument(
        "--section",
        type=str,
        help="Only process a specific section (e.g., 01-foundation)",
    )

    parser.add_argument("--file", type=str, help="Process a single github_issue.md file (for testing)")

    parser.add_argument("--resume", action="store_true", help="Resume from saved state")

    parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository (default: auto-detect from git remote)",
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers for issue creation (default: 1, use 5 for concurrent)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.file and args.section:
        print(f"{Colors.FAIL}Error: --file and --section are mutually exclusive{Colors.ENDC}")
        sys.exit(1)

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Setup paths
    repo_root = get_repo_root()
    plan_dir = repo_root / "notes" / "plan"
    log_dir = repo_root / "logs"

    # Create timestamped state file in logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = log_dir / f".issue_creation_state_{timestamp}.json"

    if not plan_dir.exists():
        print(f"{Colors.FAIL}Error: Plan directory not found: {plan_dir}{Colors.ENDC}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(log_dir)

    # Get repository name
    repo = args.repo or get_repo_name()
    logger.info(f"Using repository: {repo}")

    print(f"\n{Colors.BOLD}GitHub Issues Creator{Colors.ENDC}")
    print(f"Repository: {Colors.OKCYAN}{repo}{Colors.ENDC}")
    print(
        f"Mode: {Colors.WARNING if args.dry_run else Colors.OKGREEN}"
        f"{'DRY RUN' if args.dry_run else 'LIVE'}{Colors.ENDC}"
    )

    # Load resume state if requested
    resume_state = None
    if args.resume:
        # Find the latest state file in logs directory
        state_files = sorted(log_dir.glob(".issue_creation_state_*.json"))
        if state_files:
            latest_state_file = state_files[-1]
            state_manager = StateManager(latest_state_file)
            resume_state = state_manager.load_state()
            if resume_state:
                print(f"{Colors.OKGREEN}Resuming from saved state: {latest_state_file.name}{Colors.ENDC}")
            else:
                print(
                    f"{Colors.WARNING}Could not load state from {latest_state_file.name}, starting fresh{Colors.ENDC}"
                )
                state_manager = StateManager(state_file)
        else:
            print(f"{Colors.WARNING}No saved state files found in logs/, starting fresh{Colors.ENDC}")
            state_manager = StateManager(state_file)
    else:
        # Create new state manager with timestamped file
        state_manager = StateManager(state_file)

    # Find all issue files or use single file
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"{Colors.FAIL}Error: File not found: {args.file}{Colors.ENDC}")
            sys.exit(1)
        if not file_path.name == "github_issue.md":
            print(f"{Colors.WARNING}Warning: File is not named 'github_issue.md'{Colors.ENDC}")
        issue_files = [file_path]
        print(f"{Colors.OKCYAN}Processing single file: {file_path}{Colors.ENDC}\n")
    else:
        issue_files = find_all_issue_files(plan_dir, args.section)

    if not issue_files:
        print(f"{Colors.FAIL}No github_issue.md files found{Colors.ENDC}")
        sys.exit(1)

    # Parse all issues
    issues, stats = parse_all_issues(issue_files, resume_state)

    # Show dry-run summary
    if args.dry_run:
        print_dry_run_summary(issues, stats)
        print(f"\n{Colors.OKGREEN}Dry run complete. Run without --dry-run to create issues.{Colors.ENDC}")
        return

    # Create issues
    creator = IssueCreator(repo, dry_run=args.dry_run, logger=logger)

    # Choose sequential or concurrent based on workers argument
    if args.workers > 1:
        success_count, error_count = create_all_issues_concurrent(
            issues, creator, state_manager, args.dry_run, max_workers=args.workers
        )
    else:
        success_count, error_count = create_all_issues(issues, creator, state_manager, args.dry_run)

    # Print summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Summary{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

    print(f"Total issues: {stats.total_issues}")
    print(f"{Colors.OKGREEN}Successfully created: {success_count}{Colors.ENDC}")

    if error_count > 0:
        print(f"{Colors.FAIL}Errors: {error_count}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Some issues failed. Check logs and use --resume to retry.{Colors.ENDC}")
    else:
        print(f"\n{Colors.OKGREEN}All issues created successfully!{Colors.ENDC}")
        # Clear state on success
        state_manager.clear_state()

    print(f"\nLogs saved to: {log_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user. Use --resume to continue later.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
