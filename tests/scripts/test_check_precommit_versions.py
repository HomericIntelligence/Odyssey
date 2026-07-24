#!/usr/bin/env python3
"""Tests for the uv-era pre-commit version consistency check."""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import check_precommit_versions as checker  # noqa: E402


class TestParseRequirement:
    def test_lower_bound(self) -> None:
        assert checker._parse_requirement("mypy>=1.19.1,<2") == ("mypy", "1.19.1")

    def test_exact_version(self) -> None:
        assert checker._parse_requirement("nbstripout==0.7.1") == ("nbstripout", "0.7.1")

    def test_compatible_version(self) -> None:
        assert checker._parse_requirement("PyYAML~=6.0") == ("pyyaml", "6.0")

    def test_extras(self) -> None:
        assert checker._parse_requirement("package[extra]>=2.3") == ("package", "2.3")

    @pytest.mark.parametrize("spec", ["pytest", "pytest<9", "", "git+https://example.invalid/repo"])
    def test_without_supported_lower_bound(self, spec: str) -> None:
        assert checker._parse_requirement(spec) is None


class TestLoadAllPyprojectVersions:
    def test_loads_project_and_dependency_groups(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test"
version = "1.0"
dependencies = ["PyYAML>=6.0", "unconstrained"]

[dependency-groups]
dev = ["mypy>=1.19.1,<2"]
notebook = ["jupyterlab~=4.3"]
ignored-table = { package = "pytest>=9" }
"""
        )

        assert checker.load_all_pyproject_versions(pyproject) == {
            "pyyaml": "6.0",
            "mypy": "1.19.1",
            "jupyterlab": "4.3",
        }

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            checker.load_all_pyproject_versions(tmp_path / "missing.toml")


class TestExternalHooks:
    def test_extracts_only_versioned_external_repositories(self) -> None:
        repositories = [
            {"repo": "https://github.com/pre-commit/mirrors-mypy", "rev": "v1.19.1"},
            {"repo": "local", "hooks": [{"id": "local-hook"}]},
            {"repo": "https://github.com/example/unpinned"},
        ]

        assert checker.extract_external_hooks(repositories) == {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}

    def test_loads_precommit_yaml(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(
            "repos:\n"
            "  - repo: https://github.com/pre-commit/mirrors-mypy\n"
            "    rev: v1.19.1\n"
            "    hooks:\n"
            "      - id: mypy\n"
        )

        repositories = checker.load_precommit_config(config)
        assert repositories[0]["rev"] == "v1.19.1"

    def test_missing_repositories_key_raises(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text("version: 1\n")

        with pytest.raises(ValueError, match="No 'repos'"):
            checker.load_precommit_config(config)


class TestVersionDrift:
    def test_matching_versions_pass(self) -> None:
        external = {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
        declared = {"mypy": "1.19.1"}
        assert checker.check_version_drift(external, declared) == []

    def test_drift_is_reported(self) -> None:
        external = {"https://github.com/pre-commit/mirrors-mypy": "v1.18.0"}
        declared = {"mypy": "1.19.1"}
        assert "DRIFT" in checker.check_version_drift(external, declared)[0]

    def test_missing_dependency_is_reported(self) -> None:
        external = {"https://github.com/kynan/nbstripout": "0.7.1"}
        assert "MISSING" in checker.check_version_drift(external, {})[0]

    def test_untracked_repository_is_ignored(self) -> None:
        external = {"https://github.com/example/untracked": "v1.0.0"}
        assert checker.check_version_drift(external, {}) == []


class TestMain:
    @staticmethod
    def _write_project(root: Path, hook_version: str, dependency_version: str) -> None:
        (root / ".pre-commit-config.yaml").write_text(
            "repos:\n"
            "  - repo: https://github.com/pre-commit/mirrors-mypy\n"
            f"    rev: v{hook_version}\n"
            "    hooks:\n"
            "      - id: mypy\n"
        )
        (root / "pyproject.toml").write_text(
            "[project]\n"
            'name = "test"\n'
            'version = "1.0"\n'
            "dependencies = []\n\n"
            "[dependency-groups]\n"
            f'dev = ["mypy>={dependency_version},<2"]\n'
        )

    def test_consistent_project_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_project(tmp_path, "1.19.1", "1.19.1")
        monkeypatch.setattr(checker, "get_repo_root", lambda: tmp_path)

        assert checker.main() == 0
        assert "consistent with pyproject.toml" in capsys.readouterr().out

    def test_drift_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_project(tmp_path, "1.18.0", "1.19.1")
        monkeypatch.setattr(checker, "get_repo_root", lambda: tmp_path)

        assert checker.main() == 1
        assert "drift" in capsys.readouterr().out.lower()


class TestHookMapping:
    @pytest.mark.parametrize(
        "repository",
        [
            "https://github.com/pre-commit/mirrors-mypy",
            "https://github.com/kynan/nbstripout",
            "https://github.com/adrienverge/yamllint",
        ],
    )
    def test_tracked_repository(self, repository: str) -> None:
        assert repository in checker.DEFAULT_HOOK_TO_PIXI_MAP

    def test_markdownlint_is_not_tracked(self) -> None:
        assert "https://github.com/DavidAnson/markdownlint-cli2" not in checker.DEFAULT_HOOK_TO_PIXI_MAP
