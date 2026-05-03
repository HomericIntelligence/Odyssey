#!/usr/bin/env python3
"""Tests for scripts/check_dep_sync.py — dependency consistency validation."""

import importlib.util
import textwrap
from pathlib import Path


# Load module from file path (scripts/ has no __init__.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("check_dep_sync", _PROJECT_ROOT / "scripts" / "check_dep_sync.py")
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

check_dep_sync = _mod.check_dep_sync
check_pyproject_no_deps = _mod.check_pyproject_no_deps
parse_pixi_toml = _mod.parse_pixi_toml
parse_requirements = _mod.parse_requirements

# Private helpers were renamed and are not re-exported via *; import directly.
from hephaestus.config.dep_sync import (  # noqa: E402
    _parse_constraints as parse_pixi_constraints,
    _parse_version as parse_version,
    _version_satisfies as version_satisfies,
)


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_simple(self) -> None:
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_two_parts(self) -> None:
        assert parse_version("3.0") == (3, 0)

    def test_single(self) -> None:
        assert parse_version("5") == (5,)


# ---------------------------------------------------------------------------
# Constraint parsing
# ---------------------------------------------------------------------------


class TestParsePixiConstraints:
    def test_range(self) -> None:
        result = parse_pixi_constraints(">=1.2.0,<2")
        assert len(result) == 2
        assert result[0].op == ">="
        assert result[0].version == (1, 2, 0)
        assert result[1].op == "<"
        assert result[1].version == (2,)

    def test_exact(self) -> None:
        result = parse_pixi_constraints("==0.26.1")
        assert len(result) == 1
        assert result[0].op == "=="

    def test_with_quotes(self) -> None:
        result = parse_pixi_constraints('">=3.0.0,<4"')
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Version satisfies
# ---------------------------------------------------------------------------


class TestVersionSatisfies:
    def test_in_range(self) -> None:
        constraints = parse_pixi_constraints(">=1.0.0,<2")
        assert version_satisfies((1, 5, 0), constraints)

    def test_below_range(self) -> None:
        constraints = parse_pixi_constraints(">=1.0.0,<2")
        assert not version_satisfies((0, 9, 0), constraints)

    def test_above_range(self) -> None:
        constraints = parse_pixi_constraints(">=1.0.0,<2")
        assert not version_satisfies((2, 0, 0), constraints)

    def test_exact_lower(self) -> None:
        constraints = parse_pixi_constraints(">=1.0.0,<2")
        assert version_satisfies((1, 0, 0), constraints)

    def test_exact_upper_excluded(self) -> None:
        constraints = parse_pixi_constraints(">=1.0.0,<2")
        assert not version_satisfies((2, 0, 0), constraints)

    def test_floor_only(self) -> None:
        constraints = parse_pixi_constraints(">=1.7.5")
        assert version_satisfies((1, 9, 4), constraints)
        assert not version_satisfies((1, 7, 4), constraints)


# ---------------------------------------------------------------------------
# pixi.toml parsing
# ---------------------------------------------------------------------------


class TestParsePixiToml:
    def test_basic(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text(
            textwrap.dedent("""\
            [workspace]
            name = "test"

            [dependencies]
            pytest = ">=8.0,<9"
            ruff = ">=0.14.7,<0.16"

            [feature.notebook.dependencies]
            jupyterlab = ">=4.3.0,<5"
        """)
        )
        deps = parse_pixi_toml(pixi)
        assert "pytest" in deps
        assert deps["pytest"] == ">=8.0,<9"
        assert "ruff" in deps
        # Feature dependencies should NOT be included
        assert "jupyterlab" not in deps

    def test_inline_comments(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text(
            textwrap.dedent("""\
            [dependencies]
            nbstripout = ">=0.7.1"  # tracked for sync
        """)
        )
        deps = parse_pixi_toml(pixi)
        assert deps["nbstripout"] == ">=0.7.1"


# ---------------------------------------------------------------------------
# requirements.txt parsing
# ---------------------------------------------------------------------------


class TestParseRequirements:
    def test_basic(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text(
            textwrap.dedent("""\
            # comment
            pytest==8.4.2
            ruff==0.14.14
            -r other.txt
        """)
        )
        pins = parse_requirements(req)
        assert pins == {"pytest": "8.4.2", "ruff": "0.14.14"}

    def test_with_inline_comment(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("click==8.2.1  # CLI framework\n")
        pins = parse_requirements(req)
        assert pins == {"click": "8.2.1"}


# ---------------------------------------------------------------------------
# pyproject.toml checks
# ---------------------------------------------------------------------------


class TestCheckPyprojectNoDeps:
    def test_clean(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [build-system]
            requires = ["setuptools"]

            [project]
            name = "test"
            version = "1.0"

            [tool.pytest.ini_options]
            testpaths = ["tests"]
        """)
        )
        assert check_pyproject_no_deps(pyproject) == []

    def test_has_dependencies(self, tmp_path: Path) -> None:
        # hephaestus checks for [project.dependencies] as a TOML section header,
        # not for inline `dependencies = [...]` under [project].
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project.dependencies]
            pytest = ">=7.0"
        """)
        )
        errors = check_pyproject_no_deps(pyproject)
        assert len(errors) > 0
        assert "dependencies" in errors[0]

    def test_has_optional_dependencies(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"

            [project.optional-dependencies]
            dev = ["ruff>=0.1.0"]
        """)
        )
        errors = check_pyproject_no_deps(pyproject)
        assert len(errors) > 0
        assert "optional-dependencies" in errors[0]


# ---------------------------------------------------------------------------
# Integration: check_dep_sync
# ---------------------------------------------------------------------------


class TestCheckDepSync:
    def _create_repo(self, tmp_path: Path) -> None:
        """Create a minimal consistent repo structure."""
        pixi = tmp_path / "pixi.toml"
        pixi.write_text(
            textwrap.dedent("""\
            [dependencies]
            pytest = ">=8.0,<9"
            ruff = ">=0.14.0,<0.16"
        """)
        )
        req = tmp_path / "requirements.txt"
        req.write_text("pytest==8.4.2\nruff==0.14.14\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"
            version = "1.0"

            [tool.pytest.ini_options]
            testpaths = ["tests"]
        """)
        )

    def test_all_consistent(self, tmp_path: Path) -> None:
        self._create_repo(tmp_path)
        errors = check_dep_sync(tmp_path)
        assert errors == []

    def test_missing_package_in_pixi(self, tmp_path: Path) -> None:
        self._create_repo(tmp_path)
        req = tmp_path / "requirements.txt"
        req.write_text("pytest==8.4.2\nruff==0.14.14\nmissing-pkg==1.0.0\n")
        errors = check_dep_sync(tmp_path)
        assert any("missing-pkg" in e for e in errors)

    def test_version_outside_range(self, tmp_path: Path) -> None:
        self._create_repo(tmp_path)
        req = tmp_path / "requirements.txt"
        req.write_text("pytest==7.0.0\nruff==0.14.14\n")
        errors = check_dep_sync(tmp_path)
        assert any("pytest" in e and "outside" in e for e in errors)

    def test_pyproject_with_deps(self, tmp_path: Path) -> None:
        # hephaestus checks for [project.dependencies] as a TOML section header.
        self._create_repo(tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project.dependencies]
            pytest = ">=7.0"
        """)
        )
        errors = check_dep_sync(tmp_path)
        assert any("dependencies" in e for e in errors)

    def test_no_pixi_toml(self, tmp_path: Path) -> None:
        errors = check_dep_sync(tmp_path)
        assert any("pixi.toml not found" in e for e in errors)
