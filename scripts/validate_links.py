#!/usr/bin/env python3
"""
Documentation link validation script for ML Odyssey

Validates that all markdown links point to existing files or valid URLs.

Usage:
    python scripts/validate_links.py [directory] [--verbose]

Exit codes:
    0: All links valid
    1: One or more broken links found
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from hephaestus.utils import get_repo_root

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_markdown_files(directory: Path) -> List[Path]:
    """Find all markdown files in directory tree"""
    return list(directory.rglob("*.md"))


def extract_links(content: str, file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Extract markdown links from content

    Returns:
        List of (link_text, line_number, link_target) tuples
    """
    links = []

    # Match markdown links: [text](url)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for line_num, line in enumerate(content.split("\n"), 1):
        for match in re.finditer(link_pattern, line):
            link_text = match.group(1)
            link_target = match.group(2)

            # Skip anchor-only links (#section)
            if link_target.startswith("#"):
                continue

            links.append((link_text, line_num, link_target))

    return links


def is_url(link: str) -> bool:
    """Check if link is a URL (http/https)"""
    try:
        result = urlparse(link)
        return result.scheme in ["http", "https"]
    except (ValueError, TypeError):
        return False


def validate_internal_link(link: str, source_file: Path, repo_root: Path) -> Tuple[bool, str]:
    """
    Validate an internal (file) link

    Returns:
        (is_valid, error_message).
    """
    # Remove anchor if present
    link_path = link.split("#")[0]

    if not link_path:
        # Pure anchor link - skip for now
        return True, ""

    # Resolve relative to source file
    if link_path.startswith("/"):
        # Absolute path from repo root
        target_path = repo_root / link_path.lstrip("/")
    else:
        # Relative path
        target_path = (source_file.parent / link_path).resolve()

    if not target_path.exists():
        return False, f"File not found: {link_path}"

    return True, ""


def validate_links(file_path: Path, repo_root: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate all links in a markdown file

    Returns:
        Dictionary with validation results
    """
    # Get relative path, handling case where file might not be under repo_root
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # File is not under repo_root, use absolute path
        rel_path = str(file_path.resolve())

    result: Dict[str, Any] = {
        "path": rel_path,
        "total_links": 0,
        "valid_links": 0,
        "broken_links": [],
        "skipped_urls": 0,
    }

    try:
        content = file_path.read_text()
        links = extract_links(content, file_path)
        result["total_links"] = len(links)

        for link_text, line_num, link_target in links:
            # Skip external URLs (would need network access to validate)
            if is_url(link_target):
                result["skipped_urls"] += 1
                continue

            # Validate internal link
            is_valid, error = validate_internal_link(link_target, file_path, repo_root)

            if is_valid:
                result["valid_links"] += 1
            else:
                result["broken_links"].append(
                    {
                        "line": line_num,
                        "text": link_text,
                        "target": link_target,
                        "error": error,
                    }
                )

    except Exception as e:
        result["error"] = str(e)

    return result


def validate_all_links(directory: Path, verbose: bool = False) -> Dict[str, Any]:
    """Validate links in all markdown files"""
    results: Dict[str, Any] = {"passed": [], "failed": [], "total_links": 0, "broken_links": 0}

    repo_root = get_repo_root()

    # Resolve directory to absolute path
    directory = directory.resolve()

    markdown_files = find_markdown_files(directory)

    if not markdown_files:
        logger.warning(f"No markdown files found in {directory}")
        return results

    logger.info(f"Found {len(markdown_files)} markdown files\n")

    for md_file in markdown_files:
        result = validate_links(md_file, repo_root, verbose)

        results["total_links"] += result["total_links"]
        results["broken_links"] += len(result["broken_links"])

        if len(result["broken_links"]) == 0:
            results["passed"].append(result["path"])
            if verbose:
                logger.info(f"✓ {result['path']} ({result['total_links']} links)")
        else:
            results["failed"].append(result)
            logger.error(f"✗ {result['path']} ({len(result['broken_links'])} broken)")
            for broken in result["broken_links"]:
                logger.error(f"    Line {broken['line']}: [{broken['text']}]({broken['target']})")
                logger.error(f"      → {broken['error']}")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print validation summary"""
    total_files = len(results["passed"]) + len(results["failed"])
    passed_files = len(results["passed"])
    failed_files = len(results["failed"])
    total_links = results["total_links"]
    broken_links = results["broken_links"]

    logger.info("\n" + "=" * 70)
    logger.info("LINK VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Files with valid links: {passed_files}")
    logger.info(f"Files with broken links: {failed_files}")
    logger.info(f"\nTotal links checked: {total_links}")
    logger.info(f"Broken links: {broken_links}")

    if failed_files > 0:
        logger.info(f"\nFiles with broken links ({failed_files}):")
        for result in results["failed"]:
            logger.info(f"  {result['path']} - {len(result['broken_links'])} broken")

    logger.info("=" * 70)


def main() -> int:
    """Main validation function"""
    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Get directory to check (first positional arg or default to repo root)
    directory = None
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            directory = Path(arg)
            break

    if directory is None:
        directory = get_repo_root()

    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return 1

    logger.info(f"Validating links in: {directory}\n")

    results = validate_all_links(directory, verbose)
    print_summary(results)

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
