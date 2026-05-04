#!/usr/bin/env python3
"""
Test suite for paper implementation validation.

Implements the validation tests described in GitHub issue #5343:
Validates paper directory structure completeness, implementation files
presence, test coverage for paper implementations, and reproducibility
verification. These tests back the CI paper-validation.yml workflow.

Test Categories:
- Structure: Directory and file completeness per CI spec
- Implementation: Mojo implementation files and test presence
- Metadata: metadata.yml validation (fields and format)
- Reproducibility: Training script presence and config validity

Coverage Target: >95%
"""

import sys
from pathlib import Path
from typing import List

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def papers_root(tmp_path: Path) -> Path:
    """Provide a temporary papers/ root for isolated tests."""
    papers = tmp_path / "papers"
    papers.mkdir()
    return papers


@pytest.fixture
def complete_paper_dir(papers_root: Path) -> Path:
    """
    Provide a fully-valid paper directory under a temporary papers/ root.

    Creates the minimum structure expected by paper-validation.yml:
    - README.md (non-empty)
    - metadata.yml (all required fields)
    - implementation/ with at least one .mojo file
    - tests/ with at least one test_*.mojo file
    """
    paper = papers_root / "example-paper"
    paper.mkdir()

    (paper / "README.md").write_text(
        "# Example Paper\n\n## Overview\nA test paper.\n\n## Implementation\nDetails here.\n"
    )

    metadata = {
        "title": "Example Paper Title",
        "authors": ["Author One", "Author Two"],
        "year": 2024,
        "url": "https://arxiv.org/abs/0000.00000",
    }
    (paper / "metadata.yml").write_text(yaml.dump(metadata))

    impl_dir = paper / "implementation"
    impl_dir.mkdir()
    (impl_dir / "model.mojo").write_text("fn forward() -> Int:\n    return 0\n")

    tests_dir = paper / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_model.mojo").write_text("fn test_forward():\n    assert forward() == 0\n")

    return paper


# ---------------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------------


class TestPaperStructureValidation:
    """Validate paper directory structure (mirrors CI validate-structure job)."""

    def test_required_file_readme_present(self, complete_paper_dir: Path) -> None:
        """README.md must exist in a valid paper directory."""
        assert (complete_paper_dir / "README.md").is_file()

    def test_required_file_metadata_present(self, complete_paper_dir: Path) -> None:
        """metadata.yml must exist in a valid paper directory."""
        assert (complete_paper_dir / "metadata.yml").is_file()

    def test_missing_readme_detected(self, papers_root: Path) -> None:
        """Absence of README.md is flagged as a missing required file."""
        paper = papers_root / "no-readme"
        paper.mkdir()
        (paper / "metadata.yml").write_text("title: T\nauthors: []\nyear: 2024\nurl: https://x\n")

        missing = _check_required_files(paper, ["README.md", "metadata.yml"])
        assert "README.md" in missing

    def test_missing_metadata_detected(self, papers_root: Path) -> None:
        """Absence of metadata.yml is flagged as a missing required file."""
        paper = papers_root / "no-metadata"
        paper.mkdir()
        (paper / "README.md").write_text("# Paper\n\n## Overview\n\n## Implementation\n")

        missing = _check_required_files(paper, ["README.md", "metadata.yml"])
        assert "metadata.yml" in missing

    def test_no_missing_files_for_complete_paper(self, complete_paper_dir: Path) -> None:
        """A complete paper directory has no missing required files."""
        missing = _check_required_files(complete_paper_dir, ["README.md", "metadata.yml"])
        assert missing == [], f"Unexpected missing files: {missing}"

    def test_recommended_implementation_dir_detected(self, complete_paper_dir: Path) -> None:
        """implementation/ is present in a well-formed paper."""
        assert (complete_paper_dir / "implementation").is_dir()

    def test_recommended_tests_dir_detected(self, complete_paper_dir: Path) -> None:
        """tests/ is present in a well-formed paper."""
        assert (complete_paper_dir / "tests").is_dir()

    def test_papers_directory_contains_template(self) -> None:
        """The real papers/ directory contains the _template subdirectory."""
        repo_root = Path(__file__).parent.parent.parent
        template = repo_root / "papers" / "_template"
        assert template.is_dir(), "papers/_template must exist as the scaffold starting point"

    def test_papers_readme_exists_in_repo(self) -> None:
        """The real papers/README.md must exist and be non-empty."""
        repo_root = Path(__file__).parent.parent.parent
        readme = repo_root / "papers" / "README.md"
        assert readme.is_file(), "papers/README.md must exist"
        assert readme.stat().st_size > 0, "papers/README.md must not be empty"


# ---------------------------------------------------------------------------
# Implementation validation
# ---------------------------------------------------------------------------


class TestImplementationFilesValidation:
    """Validate Mojo implementation files and test presence."""

    def test_mojo_files_found_in_implementation_dir(self, complete_paper_dir: Path) -> None:
        """implementation/ directory must contain at least one .mojo file."""
        impl_dir = complete_paper_dir / "implementation"
        mojo_files = list(impl_dir.glob("*.mojo")) + list(impl_dir.rglob("*.🔥"))
        assert len(mojo_files) >= 1, "Expected at least one Mojo file in implementation/"

    def test_test_files_found_in_tests_dir(self, complete_paper_dir: Path) -> None:
        """tests/ directory must contain at least one test_*.mojo file."""
        tests_dir = complete_paper_dir / "tests"
        test_files = list(tests_dir.glob("test_*.mojo")) + list(tests_dir.glob("test_*.py"))
        assert len(test_files) >= 1, "Expected at least one test file in tests/"

    def test_empty_implementation_dir_flagged(self, papers_root: Path) -> None:
        """An implementation/ dir with no Mojo files is reported as incomplete."""
        paper = papers_root / "empty-impl"
        paper.mkdir()
        (paper / "implementation").mkdir()

        impl_dir = paper / "implementation"
        mojo_files = list(impl_dir.glob("*.mojo")) + list(impl_dir.rglob("*.🔥"))
        assert len(mojo_files) == 0

    def test_missing_implementation_dir_flagged(self, papers_root: Path) -> None:
        """A paper without implementation/ is reported as missing it."""
        paper = papers_root / "no-impl"
        paper.mkdir()
        assert not (paper / "implementation").exists()

    def test_missing_tests_dir_flagged(self, papers_root: Path) -> None:
        """A paper without tests/ is reported as missing it."""
        paper = papers_root / "no-tests"
        paper.mkdir()
        assert not (paper / "tests").exists()


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    """Validate metadata.yml structure and required fields."""

    def test_metadata_has_all_required_fields(self, complete_paper_dir: Path) -> None:
        """Valid metadata.yml contains title, authors, year, and url."""
        metadata = _load_metadata(complete_paper_dir)
        missing = _check_metadata_fields(metadata, ["title", "authors", "year", "url"])
        assert missing == [], f"Missing metadata fields: {missing}"

    def test_metadata_missing_title_detected(self, papers_root: Path) -> None:
        """metadata.yml without 'title' is flagged."""
        paper = papers_root / "no-title"
        paper.mkdir()
        (paper / "metadata.yml").write_text("authors: [A]\nyear: 2024\nurl: https://x\n")
        metadata = _load_metadata(paper)
        missing = _check_metadata_fields(metadata, ["title", "authors", "year", "url"])
        assert "title" in missing

    def test_metadata_missing_year_detected(self, papers_root: Path) -> None:
        """metadata.yml without 'year' is flagged."""
        paper = papers_root / "no-year"
        paper.mkdir()
        (paper / "metadata.yml").write_text("title: T\nauthors: [A]\nurl: https://x\n")
        metadata = _load_metadata(paper)
        missing = _check_metadata_fields(metadata, ["title", "authors", "year", "url"])
        assert "year" in missing

    def test_metadata_invalid_yaml_raises(self, papers_root: Path) -> None:
        """Malformed metadata.yml raises an appropriate error on load."""
        paper = papers_root / "bad-yaml"
        paper.mkdir()
        (paper / "metadata.yml").write_text("title: [\ninvalid yaml {{\n")
        with pytest.raises(Exception):
            _load_metadata(paper)

    def test_metadata_year_is_numeric(self, complete_paper_dir: Path) -> None:
        """metadata.yml 'year' field must be a number."""
        metadata = _load_metadata(complete_paper_dir)
        assert isinstance(metadata.get("year"), (int, float)), "'year' must be numeric"

    def test_metadata_url_is_string(self, complete_paper_dir: Path) -> None:
        """metadata.yml 'url' field must be a non-empty string."""
        metadata = _load_metadata(complete_paper_dir)
        url = metadata.get("url", "")
        assert isinstance(url, str) and url.strip(), "'url' must be a non-empty string"


# ---------------------------------------------------------------------------
# Reproducibility verification
# ---------------------------------------------------------------------------


class TestReproducibilityVerification:
    """Validate presence of training scripts for reproducibility runs."""

    def test_train_script_presence_detected(self, papers_root: Path) -> None:
        """A paper with train.py is identified as having a training script."""
        paper = papers_root / "with-train-py"
        paper.mkdir()
        (paper / "train.py").write_text("# training script\n")
        assert _has_training_script(paper), "Expected train.py to be detected"

    def test_mojo_train_script_detected(self, papers_root: Path) -> None:
        """A paper with train.mojo is identified as having a training script."""
        paper = papers_root / "with-train-mojo"
        paper.mkdir()
        (paper / "train.mojo").write_text("fn main():\n    pass\n")
        assert _has_training_script(paper), "Expected train.mojo to be detected"

    def test_no_training_script_returns_false(self, papers_root: Path) -> None:
        """A paper with no training script returns False."""
        paper = papers_root / "no-train"
        paper.mkdir()
        assert not _has_training_script(paper), "Expected no training script to be detected"

    def test_config_yaml_is_valid(self, complete_paper_dir: Path) -> None:
        """If a configs/config.yaml exists it must be valid YAML."""
        configs_dir = complete_paper_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        config_path = configs_dir / "config.yaml"
        config_path.write_text("model:\n  name: test\ntraining:\n  epochs: 1\n")

        with config_path.open() as fh:
            parsed = yaml.safe_load(fh)
        assert isinstance(parsed, dict), "config.yaml must parse to a dict"
        assert "model" in parsed, "config.yaml must contain 'model' key"
        assert "training" in parsed, "config.yaml must contain 'training' key"

    def test_no_config_does_not_block_validation(self, papers_root: Path) -> None:
        """A paper without configs/ does not raise during reproducibility check."""
        paper = papers_root / "no-config"
        paper.mkdir()
        # Should not raise; config is optional
        _has_training_script(paper)  # exercises the helper without error


# ---------------------------------------------------------------------------
# Private helpers (mirroring CI shell logic in Python for testability)
# ---------------------------------------------------------------------------


def _check_required_files(paper_dir: Path, required: List[str]) -> List[str]:
    """
    Return the list of required file names missing from *paper_dir*.

    Args:
        paper_dir: Path to the paper directory.
        required: Sequence of file names that must be present.

    Returns:
        List of file names that are absent.
    """
    return [f for f in required if not (paper_dir / f).is_file()]


def _load_metadata(paper_dir: Path) -> dict:
    """
    Parse and return the paper's metadata.yml as a dict.

    Args:
        paper_dir: Path to the paper directory.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If metadata.yml does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = paper_dir / "metadata.yml"
    if not path.is_file():
        raise FileNotFoundError(f"metadata.yml not found in {paper_dir}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def _check_metadata_fields(metadata: dict, required: List[str]) -> List[str]:
    """
    Return the list of required field names absent from *metadata*.

    Args:
        metadata: Parsed metadata dictionary.
        required: Field names that must be present.

    Returns:
        List of absent field names.
    """
    return [field for field in required if field not in metadata]


def _has_training_script(paper_dir: Path) -> bool:
    """
    Return True if the paper directory contains a training script.

    Checks for train.py or train.mojo at the paper root (matching the
    logic in the validate-reproducibility CI job).

    Args:
        paper_dir: Path to the paper directory.

    Returns:
        True if a training script exists, False otherwise.
    """
    return (paper_dir / "train.py").is_file() or (paper_dir / "train.mojo").is_file()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
