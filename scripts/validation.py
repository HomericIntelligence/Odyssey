#!/usr/bin/env python3
"""
Shared validation utilities for ML Odyssey scripts

This module provides common validation functions used across multiple
validation scripts to avoid duplication and ensure consistency.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


def find_markdown_files(directory: Path, exclude_dirs: Optional[Set[str]] = None) -> List[Path]:
    """
    Find all markdown files in a directory recursively.

    Args:
        directory: Directory to search
        exclude_dirs: Set of directory names to exclude

    Returns:
        List of Path objects for markdown files
    """
    if exclude_dirs is None:
        exclude_dirs = {
            "node_modules",
            ".git",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "dist",
            "build",
            ".tox",
        }

    markdown_files = []
    for md_file in directory.rglob("*.md"):
        # Check if any parent directory is in exclude list
        if any(part in exclude_dirs for part in md_file.parts):
            continue
        markdown_files.append(md_file)

    return sorted(markdown_files)


def validate_file_exists(file_path: Path) -> bool:
    """
    Validate that a file exists and is a regular file.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and is a regular file
    """
    return file_path.exists() and file_path.is_file()


def validate_directory_exists(dir_path: Path) -> bool:
    """
    Validate that a directory exists and is a directory.

    Args:
        dir_path: Path to check

    Returns:
        True if directory exists and is a directory
    """
    return dir_path.exists() and dir_path.is_dir()


def check_required_sections(
    content: str, required_sections: List[str], file_path: Optional[Path] = None
) -> Tuple[bool, List[str]]:
    """
    Check if markdown content has all required sections.

    Args:
        content: Markdown file content
        required_sections: List of required heading names
        file_path: Optional path for logging

    Returns:
        Tuple of (all_found, missing_sections).
    """
    missing: list[str] = []

    for section in required_sections:
        # Match heading at various levels (##, ###, etc.)
        # Note: Double braces {{}} are needed in f-strings to create literal braces for regex
        pattern = rf"^#{{1,6}}\s+{re.escape(section)}\s*$"
        if not re.search(pattern, content, re.MULTILINE):
            missing.append(section)
            if file_path:
                logger.debug(f"{file_path}: Missing section '{section}'")

    return len(missing) == 0, missing


def extract_markdown_links(content: str) -> List[Tuple[str, int]]:
    """
    Extract all markdown links from content.

    Args:
        content: Markdown file content

    Returns:
        List of (link_target, line_number) tuples
    """
    links = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Match [text](link) format
        for match in re.finditer(r"\[([^\]]+)\]\(([^\)]+)\)", line):
            link_target = match.group(2)
            links.append((link_target, line_num))

    return links


def validate_relative_link(link: str, source_file: Path, repo_root: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate a relative markdown link.

    Args:
        link: Link target (can include anchor #section)
        source_file: File containing the link
        repo_root: Repository root directory

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Skip external links
    if link.startswith(("http://", "https://", "mailto:")):
        return True, None

    # Skip anchors within same file
    if link.startswith("#"):
        return True, None

    # Split link and anchor
    if "#" in link:
        file_part, _anchor = link.split("#", 1)
    else:
        file_part, _anchor = link, None

    # Skip empty links
    if not file_part:
        return True, None

    # Resolve relative path
    link_path = (source_file.parent / file_part).resolve()

    # Check if file exists
    if not link_path.exists():
        return False, f"Broken link: {link} (file not found)"

    # If anchor specified, could validate it exists in target
    # (skipped for now to keep validation fast)

    return True, None


def count_markdown_issues(content: str) -> dict:
    """
    Count common markdown issues in content.

    Args:
        content: Markdown file content

    Returns:
        Dictionary of issue counts
    """
    issues = {
        "multiple_blank_lines": 0,
        "missing_language_tags": 0,
        "long_lines": 0,
        "trailing_whitespace": 0,
    }

    lines = content.split("\n")

    # Check for multiple consecutive blank lines
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count > 1:
                issues["multiple_blank_lines"] += 1
        else:
            blank_count = 0

    # Check for code blocks without language tags
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            if not in_code_block:
                # Starting code block
                if line.strip() == "```":
                    issues["missing_language_tags"] += 1
                in_code_block = True
            else:
                # Ending code block
                in_code_block = False

    # Check for long lines (> 120 characters)
    for line in lines:
        if len(line) > 120:
            # Skip lines that are URLs or code
            if not (line.strip().startswith("http") or line.strip().startswith("`")):
                issues["long_lines"] += 1

    # Check for trailing whitespace
    for line in lines:
        if line and line != line.rstrip():
            issues["trailing_whitespace"] += 1

    return issues


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
