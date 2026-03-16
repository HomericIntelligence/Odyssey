#!/usr/bin/env python3
"""
Unit tests for scripts/validate_installation_anchors.py

Tests use synthetic in-memory content and TemporaryDirectory for hermeticity,
mirroring the pattern established in tests/test_audit_shared_links.py.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_installation_anchors import (
    extract_headings,
    extract_installation_links,
    heading_to_anchor,
    main,
    validate,
)


# ---------------------------------------------------------------------------
# heading_to_anchor
# ---------------------------------------------------------------------------


class TestHeadingToAnchor:
    def test_simple_lowercase(self) -> None:
        assert heading_to_anchor("Installation") == "installation"

    def test_spaces_become_hyphens(self) -> None:
        assert heading_to_anchor("Installing Pixi") == "installing-pixi"

    def test_already_lowercase(self) -> None:
        assert heading_to_anchor("prerequisites") == "prerequisites"

    def test_special_chars_stripped(self) -> None:
        # Backticks, parens, etc. are stripped
        assert heading_to_anchor("`just` command not found") == "just-command-not-found"

    def test_backtick_wrapped_heading(self) -> None:
        assert heading_to_anchor("`pixi install` fails with channel errors") == (
            "pixi-install-fails-with-channel-errors"
        )

    def test_parens_stripped(self) -> None:
        assert heading_to_anchor("Run Tests Directly (without interactive shell)") == (
            "run-tests-directly-without-interactive-shell"
        )

    def test_multiple_spaces_collapse(self) -> None:
        # Multiple spaces → single hyphen after substitution
        assert heading_to_anchor("a  b") == "a-b"

    def test_numbers_preserved(self) -> None:
        assert heading_to_anchor("Step 1 Setup") == "step-1-setup"

    def test_leading_trailing_hyphens_stripped(self) -> None:
        # Strings that would produce leading/trailing hyphens
        result = heading_to_anchor("  Hello  ")
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# extract_headings
# ---------------------------------------------------------------------------


class TestExtractHeadings:
    def test_single_h1(self) -> None:
        content = "# Installation\n"
        assert extract_headings(content) == ["Installation"]

    def test_multiple_levels(self) -> None:
        content = "# H1\n## H2\n### H3\n"
        assert extract_headings(content) == ["H1", "H2", "H3"]

    def test_ignores_non_headings(self) -> None:
        content = "# Title\n\nSome paragraph text.\n\n## Section\n"
        assert extract_headings(content) == ["Title", "Section"]

    def test_empty_content(self) -> None:
        assert extract_headings("") == []

    def test_heading_with_backtick(self) -> None:
        content = "### `pixi install` fails with channel errors\n"
        assert extract_headings(content) == ["`pixi install` fails with channel errors"]

    def test_heading_with_parens(self) -> None:
        content = "### Run Tests Directly (without interactive shell)\n"
        assert extract_headings(content) == ["Run Tests Directly (without interactive shell)"]


# ---------------------------------------------------------------------------
# extract_installation_links
# ---------------------------------------------------------------------------


class TestExtractInstallationLinks:
    def test_plain_link_no_anchor(self) -> None:
        content = "[Installation Guide](docs/getting-started/installation.md)\n"
        result = extract_installation_links(content, "README.md")
        assert len(result) == 1
        src, target, anchor = result[0]
        assert src == "README.md"
        assert "installation.md" in target
        assert anchor is None

    def test_link_with_anchor(self) -> None:
        content = "[Prerequisites](docs/getting-started/installation.md#prerequisites)\n"
        result = extract_installation_links(content, "README.md")
        assert len(result) == 1
        _, _, anchor = result[0]
        assert anchor == "prerequisites"

    def test_absolute_path_link(self) -> None:
        content = "[Install](/docs/getting-started/installation.md#installing-pixi)\n"
        result = extract_installation_links(content, "README.md")
        assert len(result) == 1
        _, _, anchor = result[0]
        assert anchor == "installing-pixi"

    def test_non_installation_links_ignored(self) -> None:
        content = (
            "[README](README.md)\n"
            "[Setup](docs/setup.md#section)\n"
            "[Install](docs/getting-started/installation.md#prerequisites)\n"
        )
        result = extract_installation_links(content, "README.md")
        assert len(result) == 1
        _, _, anchor = result[0]
        assert anchor == "prerequisites"

    def test_multiple_installation_links(self) -> None:
        content = "[A](installation.md#prerequisites)\n[B](installation.md#installing-pixi)\n"
        result = extract_installation_links(content, "GUIDE.md")
        assert len(result) == 2
        anchors = {anchor for _, _, anchor in result}
        assert anchors == {"prerequisites", "installing-pixi"}

    def test_empty_content(self) -> None:
        assert extract_installation_links("", "README.md") == []


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

_INSTALLATION_CONTENT = """\
# Installation

## Prerequisites

Install required tools.

## Installing Pixi

Use the install script.

## Troubleshooting

### `pixi install` fails with channel errors

Check network access.
"""


class TestValidate:
    def test_no_anchor_links_passes(self) -> None:
        """A plain link without an anchor should always pass."""
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md)\n", encoding="utf-8")

            errors = validate([readme], installation)
            assert errors == []

    def test_valid_anchor_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#prerequisites)\n", encoding="utf-8")

            errors = validate([readme], installation)
            assert errors == []

    def test_invalid_anchor_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#nonexistent-section)\n", encoding="utf-8")

            errors = validate([readme], installation)
            assert len(errors) == 1
            assert "nonexistent-section" in errors[0]

    def test_backtick_heading_anchor_valid(self) -> None:
        """Anchors derived from headings with backticks should resolve correctly."""
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text(
                "[Troubleshooting](installation.md#pixi-install-fails-with-channel-errors)\n",
                encoding="utf-8",
            )

            errors = validate([readme], installation)
            assert errors == []

    def test_installation_file_missing_returns_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing = tmp / "installation.md"
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md)\n", encoding="utf-8")

            errors = validate([readme], missing)
            assert len(errors) == 1
            assert "not found" in errors[0]

    def test_source_file_missing_returns_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            missing_source = tmp / "MISSING.md"

            errors = validate([missing_source], installation)
            assert len(errors) == 1
            assert "not found" in errors[0]

    def test_multiple_sources_all_valid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")

            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#prerequisites)\n", encoding="utf-8")
            guide = tmp / "guide.md"
            guide.write_text("[Install](installation.md#installing-pixi)\n", encoding="utf-8")

            errors = validate([readme, guide], installation)
            assert errors == []

    def test_multiple_sources_one_broken(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")

            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#prerequisites)\n", encoding="utf-8")
            broken = tmp / "broken.md"
            broken.write_text("[Install](installation.md#does-not-exist)\n", encoding="utf-8")

            errors = validate([readme, broken], installation)
            assert len(errors) == 1
            assert "does-not-exist" in errors[0]


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_0_with_valid_anchors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#prerequisites)\n", encoding="utf-8")

            rc = main([str(readme), str(installation)])
            assert rc == 0

    def test_returns_0_with_no_anchor_links(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md)\n", encoding="utf-8")

            rc = main([str(readme), str(installation)])
            assert rc == 0

    def test_returns_1_with_broken_anchor(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            installation = tmp / "installation.md"
            installation.write_text(_INSTALLATION_CONTENT, encoding="utf-8")
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md#broken-anchor)\n", encoding="utf-8")

            rc = main([str(readme), str(installation)])
            assert rc == 1

    def test_returns_1_when_installation_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readme = tmp / "README.md"
            readme.write_text("[Guide](installation.md)\n", encoding="utf-8")
            missing_installation = tmp / "installation.md"

            rc = main([str(readme), str(missing_installation)])
            assert rc == 1
