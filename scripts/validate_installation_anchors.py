#!/usr/bin/env python3
"""
Validate anchor links from markdown files to docs/getting-started/installation.md

Checks that every link pointing to installation.md that includes an anchor fragment
(e.g., `installation.md#prerequisites`) resolves to an actual heading in that file.

Usage:
    python scripts/validate_installation_anchors.py [--verbose]
    python scripts/validate_installation_anchors.py README.md docs/getting-started/installation.md

Exit codes:
    0: All anchor links are valid (or no anchor links found)
    1: One or more broken anchor links found
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def heading_to_anchor(heading: str) -> str:
    """Convert a markdown heading text to its GitHub-style anchor slug.

    GitHub's algorithm:
    1. Lowercase the text
    2. Replace spaces with hyphens
    3. Remove all characters except [a-z0-9-]

    Args:
        heading: The heading text (without leading # characters and whitespace)

    Returns:
        The GitHub-style anchor slug (without the leading #)
    """
    slug = heading.lower()
    slug = slug.replace(" ", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    # Collapse multiple consecutive hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug


def extract_headings(content: str) -> List[str]:
    """Extract all heading texts from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        List of heading texts (without the leading # characters), in document order
    """
    headings = []
    for line in content.splitlines():
        match = re.match(r"^#{1,6}\s+(.*)", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def extract_installation_links(
    content: str, source_path: str
) -> List[Tuple[str, str, Optional[str]]]:
    """Extract all links pointing to installation.md from markdown content.

    Args:
        content: Full markdown file content
        source_path: Path of the source file (for reporting)

    Returns:
        List of (source_path, link_target, anchor_or_None) tuples.
        anchor_or_None is the fragment string (without leading #) if present,
        else None.
    """
    results = []
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    for line in content.splitlines():
        for match in link_pattern.finditer(line):
            target = match.group(2).strip()

            # Only care about links pointing to installation.md
            # (with or without path prefix, with or without anchor)
            base, _, fragment = target.partition("#")
            if not base.endswith("installation.md"):
                continue

            anchor = fragment if fragment else None
            results.append((source_path, target, anchor))

    return results


def validate(
    source_paths: List[Path],
    installation_path: Path,
) -> List[str]:
    """Validate all installation.md anchor links found in source files.

    Args:
        source_paths: Markdown files to scan for links to installation.md
        installation_path: Path to docs/getting-started/installation.md

    Returns:
        List of human-readable error messages (empty list = all valid)
    """
    errors: List[str] = []

    if not installation_path.exists():
        return [f"installation.md not found: {installation_path}"]

    installation_content = installation_path.read_text(encoding="utf-8")
    headings = extract_headings(installation_content)
    valid_anchors = {heading_to_anchor(h) for h in headings}

    logger.debug("Valid anchors in %s: %s", installation_path, sorted(valid_anchors))

    for source_path in source_paths:
        if not source_path.exists():
            errors.append(f"Source file not found: {source_path}")
            continue

        content = source_path.read_text(encoding="utf-8")
        links = extract_installation_links(content, str(source_path))

        for src, target, anchor in links:
            if anchor is None:
                logger.debug("%s: plain link %s (no anchor — skip)", src, target)
                continue

            if anchor not in valid_anchors:
                errors.append(
                    f"{src}: broken anchor '#{anchor}' in link '{target}' "
                    f"(valid anchors: {sorted(valid_anchors)})"
                )
            else:
                logger.debug("%s: anchor #%s OK", src, anchor)

    return errors


def _find_markdown_files(repo_root: Path) -> List[Path]:
    """Find all markdown files in the repository (excluding vendor dirs).

    Args:
        repo_root: Repository root directory

    Returns:
        List of markdown file paths
    """
    exclude = {".pixi", "build", "dist", ".git", "worktrees"}
    results = []
    for md_file in repo_root.rglob("*.md"):
        if not any(part in exclude for part in md_file.parts):
            results.append(md_file)
    return results


def main(argv: List[str]) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (sys.argv[1:])

    Returns:
        Exit code: 0 for success, 1 for failures
    """
    verbose = "--verbose" in argv or "-v" in argv
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Filter out flag arguments
    positional = [a for a in argv if not a.startswith("-")]

    # Determine repository root (script lives in scripts/, so parent is repo root)
    repo_root = Path(__file__).resolve().parent.parent

    if len(positional) >= 2:
        # Explicit: validate_installation_anchors.py <source...> <installation.md>
        installation_path = Path(positional[-1])
        source_paths = [Path(p) for p in positional[:-1]]
    elif len(positional) == 1:
        # Single positional: treat as installation.md, scan whole repo
        installation_path = Path(positional[0])
        source_paths = _find_markdown_files(repo_root)
    else:
        # Default: scan whole repo against default installation.md location
        installation_path = repo_root / "docs" / "getting-started" / "installation.md"
        source_paths = _find_markdown_files(repo_root)

    logger.info("Checking anchor links → %s", installation_path)
    logger.info("Scanning %d source file(s)", len(source_paths))

    errors = validate(source_paths, installation_path)

    if errors:
        for error in errors:
            logger.error(error)
        logger.error("\n%d broken anchor link(s) found.", len(errors))
        return 1

    logger.info("All installation.md anchor links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
