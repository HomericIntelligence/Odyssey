"""Regression tests for Ruff's explicitly pinned lint surface."""

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_PYTHON_INCLUDES = [
    "*.py",
    "*.pyi",
    "*.ipynb",
    "**/pyproject.toml",
    "**/ruff.toml",
    "**/.ruff.toml",
]


def test_ruff_does_not_inherit_version_dependent_default_rules() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)["tool"]["ruff"]

    assert configuration["lint"]["select"] == ["E4", "E7", "E9", "F"]


def test_ruff_keeps_markdown_out_of_the_python_formatter() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)["tool"]["ruff"]

    assert configuration["include"] == LEGACY_PYTHON_INCLUDES
    assert "*.md" not in configuration["include"]
