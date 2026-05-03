#!/usr/bin/env python3
"""Tests for paper implementation validation logic.

Validates the conventions that paper-validation.yml enforces:
  1. Paper directory structure: README.md + metadata.yml are required
  2. metadata.yml content: required fields (title, authors, year, url)
  3. Implementation presence: implementation/ directory with .mojo files
  4. Test coverage: tests/ directory with test_*.mojo or test_*.py files
  5. Template directory (_template) is excluded from validation

These tests use the actual papers/ directory to verify the template complies
with conventions, and use temporary fixtures to test validation logic in isolation.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
TEMPLATE_DIR = PAPERS_DIR / "_template"

REQUIRED_PAPER_FILES: List[str] = ["README.md", "metadata.yml"]
REQUIRED_METADATA_FIELDS: List[str] = ["title", "authors", "year", "url"]
RECOMMENDED_PAPER_DIRS: List[str] = ["implementation", "tests"]


# ---------------------------------------------------------------------------
# Helpers used by test classes
# ---------------------------------------------------------------------------


def find_paper_dirs(base: Path) -> List[Path]:
    """Return all paper implementation directories (those containing metadata.yml).

    Excludes the _template directory.
    """
    return [
        p.parent
        for p in base.glob("*/metadata.yml")
        if p.parent.name != "_template"
    ]


def validate_paper_structure(paper_dir: Path) -> Tuple[bool, List[str]]:
    """Check required files exist in a paper directory.

    Returns (all_present, list_of_missing_files).
    """
    missing = [f for f in REQUIRED_PAPER_FILES if not (paper_dir / f).is_file()]
    return len(missing) == 0, missing


def parse_metadata(metadata_path: Path) -> Tuple[bool, Dict, List[str]]:
    """Parse and validate a metadata.yml file.

    Returns (valid, parsed_dict, missing_fields).
    """
    try:
        with metadata_path.open() as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return False, {}, REQUIRED_METADATA_FIELDS

    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in data]
    return len(missing) == 0, data, missing


# ---------------------------------------------------------------------------
# Tests: papers/ directory root layout
# ---------------------------------------------------------------------------


class TestPapersDirectoryLayout:
    """Verify the papers/ root directory meets baseline requirements."""

    def test_papers_directory_exists(self) -> None:
        """papers/ directory must exist at the repo root."""
        assert PAPERS_DIR.exists() and PAPERS_DIR.is_dir(), (
            f"papers/ directory not found at {PAPERS_DIR}. "
            "Create it before adding paper implementations."
        )

    def test_papers_readme_exists(self) -> None:
        """papers/README.md must exist to document paper conventions."""
        readme = PAPERS_DIR / "README.md"
        assert readme.is_file(), (
            f"papers/README.md not found at {readme}. "
            "A top-level README is required to guide contributors."
        )

    def test_papers_readme_is_non_empty(self) -> None:
        """papers/README.md must contain meaningful content."""
        readme = PAPERS_DIR / "README.md"
        if readme.is_file():
            assert readme.stat().st_size > 0, "papers/README.md must not be empty"

    def test_template_directory_exists(self) -> None:
        """papers/_template/ must exist as a starter template for new papers."""
        assert TEMPLATE_DIR.exists() and TEMPLATE_DIR.is_dir(), (
            f"papers/_template/ not found at {TEMPLATE_DIR}. "
            "The template directory is required so contributors have a starting point."
        )


# ---------------------------------------------------------------------------
# Tests: paper directory structure validation logic
# ---------------------------------------------------------------------------


class TestPaperStructureValidation:
    """Unit tests for the paper structure validation logic (using tmp fixtures)."""

    def test_valid_paper_passes_structure_check(self, tmp_path: Path) -> None:
        """A directory with README.md and metadata.yml passes structure checks."""
        (tmp_path / "README.md").write_text("# Test Paper\n")
        (tmp_path / "metadata.yml").write_text(
            "title: Test\nauthors: [Author]\nyear: 2020\nurl: https://example.com\n"
        )
        ok, missing = validate_paper_structure(tmp_path)
        assert ok
        assert missing == []

    def test_missing_readme_fails_structure_check(self, tmp_path: Path) -> None:
        """A directory missing README.md fails structure checks."""
        (tmp_path / "metadata.yml").write_text("title: Test\n")
        ok, missing = validate_paper_structure(tmp_path)
        assert not ok
        assert "README.md" in missing

    def test_missing_metadata_fails_structure_check(self, tmp_path: Path) -> None:
        """A directory missing metadata.yml fails structure checks."""
        (tmp_path / "README.md").write_text("# Test\n")
        ok, missing = validate_paper_structure(tmp_path)
        assert not ok
        assert "metadata.yml" in missing

    def test_empty_directory_fails_structure_check(self, tmp_path: Path) -> None:
        """An empty directory fails structure checks for all required files."""
        ok, missing = validate_paper_structure(tmp_path)
        assert not ok
        assert set(missing) == set(REQUIRED_PAPER_FILES)


# ---------------------------------------------------------------------------
# Tests: metadata.yml validation logic
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    """Unit tests for metadata.yml validation logic (using tmp fixtures)."""

    def test_complete_metadata_is_valid(self, tmp_path: Path) -> None:
        """metadata.yml with all required fields passes validation."""
        metadata = {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762",
        }
        metadata_path = tmp_path / "metadata.yml"
        with metadata_path.open("w") as fh:
            yaml.dump(metadata, fh)

        ok, data, missing = parse_metadata(metadata_path)
        assert ok
        assert missing == []
        assert data["title"] == "Attention Is All You Need"

    def test_missing_title_fails_metadata_validation(self, tmp_path: Path) -> None:
        """metadata.yml missing 'title' fails validation."""
        metadata = {"authors": ["Author"], "year": 2020, "url": "https://example.com"}
        metadata_path = tmp_path / "metadata.yml"
        with metadata_path.open("w") as fh:
            yaml.dump(metadata, fh)

        ok, _, missing = parse_metadata(metadata_path)
        assert not ok
        assert "title" in missing

    def test_missing_url_fails_metadata_validation(self, tmp_path: Path) -> None:
        """metadata.yml missing 'url' fails validation."""
        metadata = {"title": "A Paper", "authors": ["Author"], "year": 2020}
        metadata_path = tmp_path / "metadata.yml"
        with metadata_path.open("w") as fh:
            yaml.dump(metadata, fh)

        ok, _, missing = parse_metadata(metadata_path)
        assert not ok
        assert "url" in missing

    def test_invalid_yaml_fails_metadata_validation(self, tmp_path: Path) -> None:
        """A malformed metadata.yml reports all fields as missing."""
        metadata_path = tmp_path / "metadata.yml"
        metadata_path.write_text("{ invalid: yaml: content: [\n")

        ok, data, missing = parse_metadata(metadata_path)
        assert not ok
        assert set(missing) == set(REQUIRED_METADATA_FIELDS)


# ---------------------------------------------------------------------------
# Tests: implementation and test coverage detection
# ---------------------------------------------------------------------------


class TestImplementationPresence:
    """Tests for detecting Mojo implementation files in paper directories."""

    def test_paper_with_mojo_files_detected(self, tmp_path: Path) -> None:
        """A paper with .mojo files in implementation/ is considered implemented."""
        impl_dir = tmp_path / "implementation"
        impl_dir.mkdir()
        (impl_dir / "model.mojo").write_text("fn main(): pass\n")

        mojo_files = list(impl_dir.rglob("*.mojo"))
        assert len(mojo_files) == 1

    def test_paper_without_implementation_dir_is_unimplemented(self, tmp_path: Path) -> None:
        """A paper missing implementation/ has no Mojo files."""
        mojo_files = list(tmp_path.rglob("*.mojo"))
        assert len(mojo_files) == 0

    def test_find_paper_dirs_excludes_template(self) -> None:
        """find_paper_dirs() must not return the _template directory."""
        paper_dirs = find_paper_dirs(PAPERS_DIR)
        names = [d.name for d in paper_dirs]
        assert "_template" not in names, (
            "_template must be excluded from paper validation. "
            "The template directory is a scaffold, not an implementation."
        )

    def test_paper_test_files_detected(self, tmp_path: Path) -> None:
        """A paper with test_*.mojo files in tests/ is considered tested."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_model.mojo").write_text("fn test_forward(): pass\n")
        (tests_dir / "test_layers.mojo").write_text("fn test_conv(): pass\n")

        test_files = list(tests_dir.glob("test_*.mojo")) + list(tests_dir.glob("test_*.py"))
        assert len(test_files) == 2


# ---------------------------------------------------------------------------
# Tests: template directory structure compliance
# ---------------------------------------------------------------------------


class TestTemplateDirectoryCompliance:
    """Verify the _template directory itself meets expected structure conventions."""

    def test_template_has_src_directory(self) -> None:
        """_template must contain a src/ directory."""
        src_dir = TEMPLATE_DIR / "src"
        assert src_dir.is_dir(), (
            f"_template/src/ not found at {src_dir}. "
            "The template must include a src/ directory for Mojo implementation code."
        )

    def test_template_has_tests_directory(self) -> None:
        """_template must contain a tests/ directory."""
        tests_dir = TEMPLATE_DIR / "tests"
        assert tests_dir.is_dir(), (
            f"_template/tests/ not found at {tests_dir}. "
            "The template must include a tests/ directory per TDD conventions."
        )

    def test_template_has_readme(self) -> None:
        """_template must contain a README.md."""
        readme = TEMPLATE_DIR / "README.md"
        assert readme.is_file(), (
            f"_template/README.md not found at {readme}. "
            "The template README guides contributors in implementing new papers."
        )

    def test_template_readme_documents_structure(self) -> None:
        """_template/README.md must mention expected subdirectories."""
        readme = TEMPLATE_DIR / "README.md"
        if not readme.is_file():
            pytest.skip("_template/README.md not found; skipping content check")
        content = readme.read_text(encoding="utf-8")
        for expected_section in ("src/", "tests/"):
            assert expected_section in content, (
                f"_template/README.md must document the '{expected_section}' directory. "
                f"Contributors rely on this to understand the expected paper structure."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
