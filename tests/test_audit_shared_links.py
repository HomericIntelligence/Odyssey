#!/usr/bin/env python3
"""
Unit tests for scripts/audit_shared_links.py

Tests use synthetic in-memory content strings rather than real file I/O,
ensuring fast, hermetic tests that don't depend on the repo layout.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from audit_shared_links import (
    audit,
    extract_linked_shared_paths,
    extract_quick_links_section,
    list_shared_files,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_dir(tmp: Path, filenames: List[str]) -> Path:
    """Create a fake .claude/shared/ directory with the given .md files."""
    shared = tmp / ".claude" / "shared"
    shared.mkdir(parents=True)
    for name in filenames:
        (shared / name).write_text(f"# {name}")
    return shared


def _make_claude_md(tmp: Path, content: str) -> Path:
    """Write synthetic CLAUDE.md content to a temp file."""
    path = tmp / "CLAUDE.md"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# list_shared_files
# ---------------------------------------------------------------------------


class TestListSharedFiles:
    def test_returns_md_files_only(self):
        with TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / ".claude" / "shared"
            shared.mkdir(parents=True)
            (shared / "foo.md").write_text("# foo")
            (shared / "bar.md").write_text("# bar")
            (shared / "not-md.txt").write_text("ignored")

            result = list_shared_files(shared)
            assert ".claude/shared/foo.md" in result
            assert ".claude/shared/bar.md" in result
            assert ".claude/shared/not-md.txt" not in result

    def test_returns_sorted_list(self):
        with TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / ".claude" / "shared"
            shared.mkdir(parents=True)
            for name in ["zzz.md", "aaa.md", "mmm.md"]:
                (shared / name).write_text("")

            result = list_shared_files(shared)
            assert result == sorted(result)

    def test_empty_dir_returns_empty_list(self):
        with TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / ".claude" / "shared"
            shared.mkdir(parents=True)
            assert list_shared_files(shared) == []


# ---------------------------------------------------------------------------
# extract_quick_links_section
# ---------------------------------------------------------------------------


class TestExtractQuickLinksSection:
    def test_extracts_section_content(self):
        content = """\
## Other Section

Some content.

## Quick Links

### Core Guidelines

- [Foo](/.claude/shared/foo.md)

## Another Section

More content.
"""
        section = extract_quick_links_section(content)
        assert "foo.md" in section
        assert "Another Section" not in section

    def test_returns_empty_string_when_missing(self):
        content = "## Only Section\n\nNo quick links here.\n"
        assert extract_quick_links_section(content) == ""

    def test_captures_to_end_of_file_when_last_section(self):
        content = "## Quick Links\n\n- [Bar](/.claude/shared/bar.md)\n"
        section = extract_quick_links_section(content)
        assert "bar.md" in section


# ---------------------------------------------------------------------------
# extract_linked_shared_paths
# ---------------------------------------------------------------------------


class TestExtractLinkedSharedPaths:
    def test_absolute_link_extracted(self):
        section = "- [Foo](/.claude/shared/foo.md)\n"
        assert extract_linked_shared_paths(section) == {".claude/shared/foo.md"}

    def test_relative_link_extracted(self):
        section = "- [Bar](.claude/shared/bar.md)\n"
        assert extract_linked_shared_paths(section) == {".claude/shared/bar.md"}

    def test_anchor_stripped_from_path(self):
        section = "- [Baz](/.claude/shared/baz.md#section)\n"
        assert extract_linked_shared_paths(section) == {".claude/shared/baz.md"}

    def test_multiple_links(self):
        section = "- [A](/.claude/shared/a.md)\n- [B](/.claude/shared/b.md)\n- [C](/.claude/shared/c.md)\n"
        result = extract_linked_shared_paths(section)
        assert result == {
            ".claude/shared/a.md",
            ".claude/shared/b.md",
            ".claude/shared/c.md",
        }

    def test_non_shared_links_ignored(self):
        section = "- [Docs](/docs/some-doc.md)\n- [Foo](/.claude/shared/foo.md)\n"
        result = extract_linked_shared_paths(section)
        assert result == {".claude/shared/foo.md"}

    def test_empty_section_returns_empty_set(self):
        assert extract_linked_shared_paths("") == set()


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


FULL_CLAUDE_MD_TEMPLATE = """\
## Quick Links

### Core Guidelines

{links}

## Next Section

Other content.
"""


class TestAudit:
    def test_clean_state_passes(self):
        with TemporaryDirectory() as tmpdir:
            shared = _make_shared_dir(Path(tmpdir), ["foo.md", "bar.md", "baz.md"])
            links = (
                "- [Foo](/.claude/shared/foo.md)\n- [Bar](/.claude/shared/bar.md)\n- [Baz](/.claude/shared/baz.md)\n"
            )
            content = FULL_CLAUDE_MD_TEMPLATE.format(links=links)
            missing, present = audit(content, shared)
            assert missing == []
            assert sorted(present) == [
                ".claude/shared/bar.md",
                ".claude/shared/baz.md",
                ".claude/shared/foo.md",
            ]

    def test_missing_file_detected(self):
        with TemporaryDirectory() as tmpdir:
            shared = _make_shared_dir(Path(tmpdir), ["foo.md", "bar.md", "missing.md"])
            links = (
                "- [Foo](/.claude/shared/foo.md)\n- [Bar](/.claude/shared/bar.md)\n"
                # missing.md intentionally absent
            )
            content = FULL_CLAUDE_MD_TEMPLATE.format(links=links)
            missing, present = audit(content, shared)
            assert missing == [".claude/shared/missing.md"]
            assert ".claude/shared/foo.md" in present
            assert ".claude/shared/bar.md" in present

    def test_extra_link_in_claude_md_no_false_positive(self):
        """Links in CLAUDE.md that don't correspond to shared files are fine."""
        with TemporaryDirectory() as tmpdir:
            shared = _make_shared_dir(Path(tmpdir), ["foo.md"])
            links = "- [Foo](/.claude/shared/foo.md)\n- [Extra](/.claude/shared/extra-not-on-disk.md)\n"
            content = FULL_CLAUDE_MD_TEMPLATE.format(links=links)
            missing, present = audit(content, shared)
            assert missing == []
            assert present == [".claude/shared/foo.md"]

    def test_all_missing_when_quick_links_absent(self):
        with TemporaryDirectory() as tmpdir:
            shared = _make_shared_dir(Path(tmpdir), ["foo.md", "bar.md"])
            content = "## Some Other Section\n\nNo quick links.\n"
            missing, present = audit(content, shared)
            assert sorted(missing) == [
                ".claude/shared/bar.md",
                ".claude/shared/foo.md",
            ]
            assert present == []


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_0_on_clean_state(self):
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared = _make_shared_dir(tmppath, ["foo.md"])
            content = FULL_CLAUDE_MD_TEMPLATE.format(links="- [Foo](/.claude/shared/foo.md)\n")
            claude_md = _make_claude_md(tmppath, content)
            rc = main(["--claude-md", str(claude_md), "--shared-dir", str(shared)])
            assert rc == 0

    def test_returns_1_on_missing_link(self):
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            shared = _make_shared_dir(tmppath, ["foo.md", "bar.md"])
            content = FULL_CLAUDE_MD_TEMPLATE.format(
                links="- [Foo](/.claude/shared/foo.md)\n"
                # bar.md missing
            )
            claude_md = _make_claude_md(tmppath, content)
            rc = main(["--claude-md", str(claude_md), "--shared-dir", str(shared)])
            assert rc == 1

    def test_returns_1_when_claude_md_missing(self, tmp_path: Path):
        shared = tmp_path / ".claude" / "shared"
        shared.mkdir(parents=True)
        rc = main(
            [
                "--claude-md",
                str(tmp_path / "NONEXISTENT.md"),
                "--shared-dir",
                str(shared),
            ]
        )
        assert rc == 1

    def test_returns_1_when_shared_dir_missing(self, tmp_path: Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("## Quick Links\n")
        rc = main(
            [
                "--claude-md",
                str(claude_md),
                "--shared-dir",
                str(tmp_path / "nonexistent"),
            ]
        )
        assert rc == 1
