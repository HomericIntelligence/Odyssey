#!/usr/bin/env python3
"""
Tests for scripts/check_precommit_versions.py

Covers:
- normalize_version(): strips leading 'v' from git tags
- parse_pixi_constraint(): extracts lower-bound from constraint strings
- version_tuple(): converts version strings to comparable tuples
- extract_external_hooks(): filters local repos, returns URL→rev mapping
- load_precommit_config(): YAML loading and validation
- load_pixi_versions(): pixi.toml parsing via fallback parser
- check_version_drift(): drift detection and missing-package reporting
- check_version_consistency(): integration with real temp files
- main(): CLI exit codes
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_precommit_versions import (
    DEFAULT_HOOK_TO_PIXI_MAP as HOOK_TO_PIXI_MAP,
    check_version_drift,
    extract_external_hooks,
    load_precommit_config,
    parse_pixi_constraint,
)

# These are not re-exported by the script; import directly from hephaestus.
from hephaestus.ci.precommit import (
    _parse_pixi_dependencies_fallback,
    check_version_consistency,
    check_precommit_versions_main as main,
    normalize_version,
)


def version_tuple(v: str) -> tuple:
    """Convert a dotted version string to a comparable tuple of ints."""
    parts = []
    for part in v.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


# ---------------------------------------------------------------------------
# normalize_version
# ---------------------------------------------------------------------------


class TestNormalizeVersion:
    """Tests for normalize_version()."""

    def test_strips_leading_v(self) -> None:
        assert normalize_version("v1.19.1") == "1.19.1"

    def test_no_leading_v_unchanged(self) -> None:
        assert normalize_version("0.7.1") == "0.7.1"

    def test_bare_major_minor(self) -> None:
        assert normalize_version("v4.5") == "4.5"

    def test_uppercase_v_stripped(self) -> None:
        # lstrip strips all leading 'v' characters
        assert normalize_version("vv1.0.0") == "1.0.0"


# ---------------------------------------------------------------------------
# parse_pixi_constraint
# ---------------------------------------------------------------------------


class TestParsePixiConstraint:
    """Tests for parse_pixi_constraint()."""

    def test_gte_with_upper_bound(self) -> None:
        assert parse_pixi_constraint(">=1.19.1,<2") == "1.19.1"

    def test_equals_constraint(self) -> None:
        assert parse_pixi_constraint("==0.26.1") == "0.26.1"

    def test_gte_only(self) -> None:
        assert parse_pixi_constraint(">=0.7.1") == "0.7.1"

    def test_bare_version(self) -> None:
        assert parse_pixi_constraint("4.5.0") == "4.5.0"

    def test_bare_version_with_spaces(self) -> None:
        assert parse_pixi_constraint("  4.5.0  ") == "4.5.0"

    def test_gte_with_spaces(self) -> None:
        assert parse_pixi_constraint(">=0.12.1,<0.13") == "0.12.1"

    def test_unparseable_returns_none(self) -> None:
        assert parse_pixi_constraint("*") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_pixi_constraint("") is None


# ---------------------------------------------------------------------------
# version_tuple
# ---------------------------------------------------------------------------


class TestVersionTuple:
    """Tests for version_tuple()."""

    def test_three_part(self) -> None:
        assert version_tuple("1.19.1") == (1, 19, 1)

    def test_two_part(self) -> None:
        assert version_tuple("4.5") == (4, 5)

    def test_single_part(self) -> None:
        assert version_tuple("2") == (2,)

    def test_comparison_ordering(self) -> None:
        assert version_tuple("0.7.1") < version_tuple("0.12.1")

    def test_non_numeric_part_becomes_zero(self) -> None:
        assert version_tuple("1.0.alpha") == (1, 0, 0)


# ---------------------------------------------------------------------------
# extract_external_hooks
# ---------------------------------------------------------------------------


class TestExtractExternalHooks:
    """Tests for extract_external_hooks()."""

    def test_returns_external_repos(self) -> None:
        repos = [
            {"repo": "https://github.com/pre-commit/mirrors-mypy", "rev": "v1.19.1"},
        ]
        result = extract_external_hooks(repos)
        assert result == {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}

    def test_skips_local_repos(self) -> None:
        repos = [{"repo": "local", "hooks": [{"id": "my-hook"}]}]
        result = extract_external_hooks(repos)
        assert result == {}

    def test_skips_repos_without_rev(self) -> None:
        repos = [{"repo": "https://github.com/example/repo"}]
        result = extract_external_hooks(repos)
        assert result == {}

    def test_multiple_external_repos(self) -> None:
        repos = [
            {"repo": "https://github.com/pre-commit/mirrors-mypy", "rev": "v1.19.1"},
            {"repo": "https://github.com/DavidAnson/markdownlint-cli2", "rev": "v0.12.1"},
            {"repo": "local", "hooks": []},
        ]
        result = extract_external_hooks(repos)
        assert len(result) == 2
        assert result["https://github.com/DavidAnson/markdownlint-cli2"] == "v0.12.1"


# ---------------------------------------------------------------------------
# load_precommit_config
# ---------------------------------------------------------------------------


class TestLoadPrecommitConfig:
    """Tests for load_precommit_config()."""

    def test_loads_repos_from_valid_yaml(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(
            "repos:\n  - repo: https://github.com/example/repo\n    rev: v1.0.0\n    hooks:\n      - id: some-hook\n"
        )
        repos = load_precommit_config(config)
        assert len(repos) == 1
        assert repos[0]["repo"] == "https://github.com/example/repo"

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_precommit_config(tmp_path / "nonexistent.yaml")

    def test_raises_value_error_missing_repos_key(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text("version: 1\n")
        with pytest.raises(ValueError, match="No 'repos'"):
            load_precommit_config(config)


# ---------------------------------------------------------------------------
# _parse_pixi_dependencies_fallback
# ---------------------------------------------------------------------------


class TestParsePixiDependenciesFallback:
    """Tests for the manual TOML fallback parser."""

    def test_parses_gte_constraint(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text('[dependencies]\nmypy = ">=1.19.1,<2"\n')
        result = _parse_pixi_dependencies_fallback(pixi)
        assert result.get("mypy") == "1.19.1"

    def test_parses_multiple_packages(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text('[dependencies]\nmypy = ">=1.19.1,<2"\nnbstripout = ">=0.7.1,<0.8"\n')
        result = _parse_pixi_dependencies_fallback(pixi)
        assert result.get("nbstripout") == "0.7.1"

    def test_stops_at_next_section(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text('[dependencies]\nmypy = ">=1.19.1,<2"\n[tasks]\nfake = ">=9.9.9"\n')
        result = _parse_pixi_dependencies_fallback(pixi)
        assert "fake" not in result

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text('[dependencies]\n# This is a comment\nmypy = ">=1.19.1,<2"\n')
        result = _parse_pixi_dependencies_fallback(pixi)
        assert result.get("mypy") == "1.19.1"

    def test_package_names_lowercased(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text('[dependencies]\nMyPkg = ">=1.0.0"\n')
        result = _parse_pixi_dependencies_fallback(pixi)
        assert "mypkg" in result


# ---------------------------------------------------------------------------
# check_version_drift
# ---------------------------------------------------------------------------


class TestCheckVersionDrift:
    """Tests for check_version_drift()."""

    def test_no_drift_returns_empty(self) -> None:
        external = {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
        pixi = {"mypy": "1.19.1"}
        assert check_version_drift(external, pixi) == []

    def test_drift_detected(self) -> None:
        external = {"https://github.com/pre-commit/mirrors-mypy": "v1.18.0"}
        pixi = {"mypy": "1.19.1"}
        issues = check_version_drift(external, pixi)
        assert len(issues) == 1
        assert "DRIFT" in issues[0]
        assert "mypy" in issues[0]

    def test_missing_pixi_entry_reported(self) -> None:
        external = {"https://github.com/kynan/nbstripout": "0.7.1"}
        pixi: dict = {}
        issues = check_version_drift(external, pixi)
        assert len(issues) == 1
        assert "MISSING" in issues[0]
        assert "nbstripout" in issues[0]

    def test_unmapped_repo_ignored(self) -> None:
        external = {"https://github.com/unknown/repo": "v1.0.0"}
        pixi: dict = {}
        assert check_version_drift(external, pixi) == []

    def test_all_tracked_repos_consistent(self) -> None:
        # Three tracked repos: mypy, nbstripout, pre-commit-hooks
        # markdownlint-cli2 is excluded (JS tool, versioning mismatch with conda-forge)
        external = {
            "https://github.com/pre-commit/mirrors-mypy": "v1.19.1",
            "https://github.com/DavidAnson/markdownlint-cli2": "v0.12.1",  # not tracked
            "https://github.com/kynan/nbstripout": "0.7.1",
            "https://github.com/pre-commit/pre-commit-hooks": "v4.5.0",
        }
        pixi = {
            "mypy": "1.19.1",
            "nbstripout": "0.7.1",
            "pre-commit-hooks": "4.5.0",
        }
        assert check_version_drift(external, pixi) == []

    def test_multiple_drifts_all_reported(self) -> None:
        external = {
            "https://github.com/pre-commit/mirrors-mypy": "v1.18.0",
            "https://github.com/kynan/nbstripout": "0.6.0",
        }
        pixi = {"mypy": "1.19.1", "nbstripout": "0.7.1"}
        issues = check_version_drift(external, pixi)
        assert len(issues) == 2

    def test_missing_and_drift_both_reported(self) -> None:
        external = {
            "https://github.com/pre-commit/mirrors-mypy": "v1.18.0",
            "https://github.com/kynan/nbstripout": "0.7.1",
        }
        pixi = {"mypy": "1.19.1"}  # nbstripout missing
        issues = check_version_drift(external, pixi)
        assert len(issues) == 2
        texts = " ".join(issues)
        assert "DRIFT" in texts
        assert "MISSING" in texts


# ---------------------------------------------------------------------------
# check_version_consistency (integration)
# ---------------------------------------------------------------------------


class TestCheckVersionConsistency:
    """Integration tests for check_version_consistency()."""

    def _write_precommit(self, path: Path, repos_yaml: str) -> None:
        path.write_text(f"repos:\n{textwrap.indent(repos_yaml, '  ')}")

    def _write_pixi(self, path: Path, deps: str) -> None:
        path.write_text(f"[dependencies]\n{deps}\n")

    def test_consistent_returns_empty(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/pre-commit/mirrors-mypy\n  rev: v1.19.1\n  hooks:\n    - id: mypy\n",
        )
        self._write_pixi(pixi, 'mypy = ">=1.19.1,<2"')
        issues = check_version_consistency(config, pixi)
        assert issues == []

    def test_drift_detected_integration(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/pre-commit/mirrors-mypy\n  rev: v1.18.0\n  hooks:\n    - id: mypy\n",
        )
        self._write_pixi(pixi, 'mypy = ">=1.19.1,<2"')
        issues = check_version_consistency(config, pixi)
        assert len(issues) == 1
        assert "DRIFT" in issues[0]

    def test_missing_pixi_entry_integration(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/kynan/nbstripout\n  rev: 0.7.1\n  hooks:\n    - id: nbstripout\n",
        )
        self._write_pixi(pixi, "")  # no nbstripout entry
        issues = check_version_consistency(config, pixi)
        assert len(issues) == 1
        assert "MISSING" in issues[0]

    def test_local_repos_ignored(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        config.write_text("repos:\n  - repo: local\n    hooks:\n      - id: my-hook\n")
        self._write_pixi(pixi, "")
        issues = check_version_consistency(config, pixi)
        assert issues == []

    def test_raises_on_missing_precommit_config(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text("[dependencies]\n")
        with pytest.raises(FileNotFoundError):
            check_version_consistency(tmp_path / "missing.yaml", pixi)

    def test_raises_on_missing_pixi_toml(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text("repos: []\n")
        with pytest.raises(FileNotFoundError):
            check_version_consistency(config, tmp_path / "missing.toml")


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() exit codes."""

    def _write_precommit(self, path: Path, repos_yaml: str) -> None:
        path.write_text(f"repos:\n{textwrap.indent(repos_yaml, '  ')}")

    def _write_pixi(self, path: Path, deps: str) -> None:
        path.write_text(f"[dependencies]\n{deps}\n")

    def test_exit_0_when_consistent(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/pre-commit/mirrors-mypy\n  rev: v1.19.1\n  hooks:\n    - id: mypy\n",
        )
        self._write_pixi(pixi, 'mypy = ">=1.19.1,<2"')
        rc = main(["--config", str(config), "--pixi", str(pixi)])
        assert rc == 0

    def test_exit_1_when_drift(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/pre-commit/mirrors-mypy\n  rev: v1.18.0\n  hooks:\n    - id: mypy\n",
        )
        self._write_pixi(pixi, 'mypy = ">=1.19.1,<2"')
        rc = main(["--config", str(config), "--pixi", str(pixi)])
        assert rc == 1

    def test_exit_1_when_config_missing(self, tmp_path: Path) -> None:
        pixi = tmp_path / "pixi.toml"
        pixi.write_text("[dependencies]\n")
        rc = main(["--config", str(tmp_path / "missing.yaml"), "--pixi", str(pixi)])
        assert rc == 1

    def test_output_ok_message_when_consistent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        config.write_text("repos: []\n")
        pixi.write_text("[dependencies]\n")
        main(["--config", str(config), "--pixi", str(pixi)])
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_output_drift_message_when_drift(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        pixi = tmp_path / "pixi.toml"
        self._write_precommit(
            config,
            "- repo: https://github.com/pre-commit/mirrors-mypy\n  rev: v1.18.0\n  hooks:\n    - id: mypy\n",
        )
        self._write_pixi(pixi, 'mypy = ">=1.19.1,<2"')
        main(["--config", str(config), "--pixi", str(pixi)])
        captured = capsys.readouterr()
        assert "drift" in captured.out.lower()


# ---------------------------------------------------------------------------
# HOOK_TO_PIXI_MAP coverage
# ---------------------------------------------------------------------------


class TestHookToPixiMap:
    """Verify all four tracked repos appear in HOOK_TO_PIXI_MAP."""

    def test_mirrors_mypy_present(self) -> None:
        assert "https://github.com/pre-commit/mirrors-mypy" in HOOK_TO_PIXI_MAP

    def test_markdownlint_cli2_not_tracked(self) -> None:
        # markdownlint-cli2 is a JS tool; its conda-forge version differs from the
        # npm rev used in .pre-commit-config.yaml, so it is intentionally excluded.
        assert "https://github.com/DavidAnson/markdownlint-cli2" not in HOOK_TO_PIXI_MAP

    def test_nbstripout_present(self) -> None:
        assert "https://github.com/kynan/nbstripout" in HOOK_TO_PIXI_MAP

    def test_pre_commit_hooks_present(self) -> None:
        assert "https://github.com/pre-commit/pre-commit-hooks" in HOOK_TO_PIXI_MAP

    def test_map_values_are_strings(self) -> None:
        for pkg in HOOK_TO_PIXI_MAP.values():
            assert isinstance(pkg, str)
            assert len(pkg) > 0
