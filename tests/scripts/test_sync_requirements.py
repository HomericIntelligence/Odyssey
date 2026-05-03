#!/usr/bin/env python3
"""Tests for scripts/sync_requirements.py — requirements file generation."""

import importlib.util
from pathlib import Path
from typing import Dict

import pytest

# Load module from file path (scripts/ has no __init__.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("sync_requirements", _PROJECT_ROOT / "scripts" / "sync_requirements.py")
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# The module re-exports from hephaestus.config.dep_sync; the old top-level
# constants and functions were renamed in the hephaestus refactor.
# Import from the underlying hephaestus module directly.
from hephaestus.config.dep_sync import (  # noqa: E402
    _GENERATED_HEADER as GENERATED_HEADER,
    generate_requirements_content as generate_requirements,
    sync_requirements as _sync_requirements,
    check_requirements_up_to_date as _check_requirements_up_to_date,
)

# Minimal package lists that cover SAMPLE_RESOLVED entries.
CORE_PACKAGES: list[str] = [
    "pytest",
    "pytest-cov",
    "pytest-timeout",
    "pytest-xdist",
    "pyyaml",
    "click",
    "jinja2",
]
DEV_PACKAGES: list[str] = [
    "ruff",
    "mypy",
    "pre-commit",
    "safety",
    "bandit",
    "mkdocs",
    "mkdocs-material",
    "pytest-benchmark",
]


def write_requirements(tmp_path: Path, resolved: Dict[str, str]) -> list:
    """Adapter: call sync_requirements with the test package lists."""
    return _sync_requirements(tmp_path, resolved, CORE_PACKAGES, DEV_PACKAGES)


def check_requirements(tmp_path: Path, resolved: Dict[str, str]) -> bool:
    """Adapter: call check_requirements_up_to_date with the test package lists."""
    return _check_requirements_up_to_date(tmp_path, resolved, CORE_PACKAGES, DEV_PACKAGES)


SAMPLE_RESOLVED: Dict[str, str] = {
    "pytest": "8.4.2",
    "pytest-cov": "6.3.0",
    "pytest-timeout": "2.4.0",
    "pytest-xdist": "3.8.0",
    "ruff": "0.14.14",
    "mypy": "1.19.1",
    "jinja2": "3.1.6",
    "pyyaml": "6.0.3",
    "click": "8.2.1",
    "pre-commit": "4.5.1",
    "safety": "3.7.0",
    "bandit": "1.9.4",
    "mkdocs": "1.6.1",
    "mkdocs-material": "9.7.6",
    "pytest-benchmark": "5.2.3",
}


class TestGenerateRequirements:
    def test_has_header(self) -> None:
        content = generate_requirements(["pytest"], SAMPLE_RESOLVED)
        assert content.startswith(GENERATED_HEADER)

    def test_pins_exact_versions(self) -> None:
        content = generate_requirements(["pytest", "ruff"], SAMPLE_RESOLVED)
        assert "pytest==8.4.2" in content
        assert "ruff==0.14.14" in content

    def test_include_base(self) -> None:
        content = generate_requirements(["pre-commit"], SAMPLE_RESOLVED, include_base="-r requirements.txt")
        assert "-r requirements.txt" in content

    def test_section_comments(self) -> None:
        content = generate_requirements(
            ["click"],
            SAMPLE_RESOLVED,
            section_comments={"click": "# CLI"},
        )
        assert "click==8.2.1  # CLI" in content

    def test_missing_package_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        content = generate_requirements(["nonexistent"], SAMPLE_RESOLVED)
        assert "nonexistent" not in content
        captured = capsys.readouterr()
        assert "nonexistent" in captured.err

    def test_trailing_newline(self) -> None:
        content = generate_requirements(["pytest"], SAMPLE_RESOLVED)
        assert content.endswith("\n")


class TestWriteRequirements:
    def test_creates_both_files(self, tmp_path: Path) -> None:
        paths = write_requirements(tmp_path, SAMPLE_RESOLVED)
        assert len(paths) == 2
        assert (tmp_path / "requirements.txt").exists()
        assert (tmp_path / "requirements-dev.txt").exists()

    def test_core_packages_in_requirements(self, tmp_path: Path) -> None:
        write_requirements(tmp_path, SAMPLE_RESOLVED)
        content = (tmp_path / "requirements.txt").read_text()
        for pkg in CORE_PACKAGES:
            if pkg in SAMPLE_RESOLVED:
                assert f"{pkg}==" in content

    def test_dev_packages_in_requirements_dev(self, tmp_path: Path) -> None:
        write_requirements(tmp_path, SAMPLE_RESOLVED)
        content = (tmp_path / "requirements-dev.txt").read_text()
        for pkg in DEV_PACKAGES:
            if pkg in SAMPLE_RESOLVED:
                assert f"{pkg}==" in content

    def test_dev_includes_base(self, tmp_path: Path) -> None:
        write_requirements(tmp_path, SAMPLE_RESOLVED)
        content = (tmp_path / "requirements-dev.txt").read_text()
        assert "-r requirements.txt" in content


class TestCheckRequirements:
    def test_up_to_date(self, tmp_path: Path) -> None:
        write_requirements(tmp_path, SAMPLE_RESOLVED)
        assert check_requirements(tmp_path, SAMPLE_RESOLVED) is True

    def test_out_of_date(self, tmp_path: Path) -> None:
        write_requirements(tmp_path, SAMPLE_RESOLVED)
        # Tamper with file
        req = tmp_path / "requirements.txt"
        req.write_text("pytest==0.0.0\n")
        assert check_requirements(tmp_path, SAMPLE_RESOLVED) is False

    def test_missing_file(self, tmp_path: Path) -> None:
        assert check_requirements(tmp_path, SAMPLE_RESOLVED) is False
